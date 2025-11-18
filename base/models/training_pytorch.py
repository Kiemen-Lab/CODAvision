"""
PyTorch training module for semantic segmentation models.

This module implements training functions, loss functions, and trainers for
semantic segmentation models using PyTorch. It mirrors the TensorFlow training
implementation in base.models.training for numerical equivalence.

Key Features:
    - WeightedCrossEntropyLoss: Per-class weighted loss (MATLAB-aligned)
    - PyTorchSegmentationTrainer: Base trainer with full feature set
    - PyTorchDeepLabV3PlusTrainer: Specialized trainer for DeepLabV3+
    - Iteration-based validation (not epoch-based)
    - Early stopping and learning rate scheduling
    - Model checkpointing (save best model)
    - Advanced optimizations: AMP, torch.compile(), gradient accumulation

Author: CODAvision Team
Date: 2025-11-12
"""

import os
import time
import pickle
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from base.models.backbones import model_call
from base.utils.logger import Logger
from base.data.loaders_pytorch import PyTorchSegmentationDataset
from base.data.loaders import load_model_metadata
from base.config import DataConfig, ModelDefaults

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup module logger
module_logger = logging.getLogger(__name__)


class WeightedCrossEntropyLoss(nn.Module):
    """
    Custom PyTorch loss function implementing weighted cross entropy.

    This class matches the behavior of TensorFlow's WeightedSparseCategoricalCrossentropy
    for numerical equivalence. It computes per-class mean loss and applies normalized
    class weights, matching MATLAB's behavior.

    Key Features:
        - Normalizes class weights to sum to 1 (MATLAB behavior)
        - Computes mean loss for each class separately (not per-pixel mean)
        - Handles missing classes gracefully (contribute 0 to loss)
        - Returns sum of weighted class losses (not mean)

    Attributes:
        class_weights (torch.Tensor): Normalized weights for each class (sum to 1)
        num_classes (int): Number of classes
    """

    def __init__(self, class_weights: Union[List[float], np.ndarray]):
        """
        Initialize the weighted cross entropy loss.

        Args:
            class_weights: Weights for each class. Will be normalized to sum to 1.

        Raises:
            ValueError: If class weights contain NaN or infinite values
        """
        super(WeightedCrossEntropyLoss, self).__init__()

        # Convert to tensor
        if isinstance(class_weights, np.ndarray):
            weights = torch.from_numpy(class_weights).float()
        else:
            weights = torch.tensor(class_weights, dtype=torch.float32)

        # Validate weights before normalization
        if torch.any(torch.isnan(weights)):
            raise ValueError(
                f"Class weights contain NaN values: {weights.numpy()}. "
                "This typically indicates an issue with class weight calculation. "
                "Check that training mask files exist and contain valid class labels."
            )

        if torch.any(torch.isinf(weights)):
            raise ValueError(
                f"Class weights contain infinite values: {weights.numpy()}. "
                "This typically indicates division by zero in class weight calculation."
            )

        # Normalize weights to sum to 1 (MATLAB behavior)
        weights_sum = torch.sum(weights)

        if weights_sum == 0 or torch.isnan(weights_sum):
            raise ValueError(
                f"Class weights sum to zero or NaN: sum={weights_sum}. "
                "Cannot normalize weights. Check class weight calculation."
            )

        self.register_buffer('class_weights', weights / weights_sum)
        self.num_classes = len(class_weights)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss using per-class approach (MATLAB-aligned).

        This implementation computes the mean loss for each class separately,
        then weights and sums them. This ensures true class balancing even
        with highly imbalanced datasets, matching MATLAB and TensorFlow behavior.

        Args:
            y_pred: Model predictions [batch, num_classes, height, width]
            y_true: Ground truth labels [batch, height, width] or [batch, 1, height, width]

        Returns:
            Computed loss value (scalar)
        """
        # Ensure y_true is [batch, height, width]
        if y_true.dim() == 4 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)  # Remove channel dimension

        y_true = y_true.long()

        # Get dimensions
        batch_size, num_classes, height, width = y_pred.shape

        # Flatten tensors for easier processing
        y_pred_flat = y_pred.permute(0, 2, 3, 1).reshape(-1, num_classes)  # [N, num_classes]
        y_true_flat = y_true.reshape(-1)  # [N]

        # Compute per-pixel cross entropy loss (with logits)
        per_pixel_loss = F.cross_entropy(
            y_pred_flat,
            y_true_flat,
            reduction='none'  # Don't reduce yet, we need per-pixel losses
        )

        # Per-class mean loss computation (MATLAB-style)
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=y_pred.device)

        for class_idx in range(num_classes):
            # Create binary mask for pixels belonging to this class
            class_mask = (y_true_flat == class_idx).float()
            num_pixels_in_class = torch.sum(class_mask)

            # Compute mean loss for this class (with numerical stability)
            # If class is not present in batch, contribute 0 to total loss
            if num_pixels_in_class > 0:
                masked_loss = per_pixel_loss * class_mask
                class_mean_loss = torch.sum(masked_loss) / num_pixels_in_class

                # Apply normalized class weight
                weighted_class_loss = self.class_weights[class_idx] * class_mean_loss
                total_loss = total_loss + weighted_class_loss

        # Return sum of all class contributions (MATLAB-style: sum not mean)
        return total_loss


class PyTorchSegmentationTrainer:
    """
    Base trainer class for PyTorch segmentation models.

    This class provides a complete training pipeline for semantic segmentation,
    including data loading, model training, validation, early stopping, learning
    rate scheduling, and model checkpointing. It mirrors the TensorFlow
    SegmentationModelTrainer for feature parity.

    Features:
        - Iteration-based validation (configurable frequency)
        - Early stopping with patience
        - Learning rate reduction on plateau
        - Model checkpointing (save best model)
        - Training history tracking
        - Advanced optimizations: AMP, torch.compile(), gradient accumulation

    Attributes:
        model_path (str): Path to model directory
        metadata (dict): Model metadata loaded from disk
        device (torch.device): Device for training (cuda/mps/cpu)
        model (nn.Module): The segmentation model
        optimizer (torch.optim.Optimizer): Optimizer for training
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        criterion (nn.Module): Loss function
        logger (Logger): Training logger
    """

    def __init__(self, model_path: str, logger: Optional[Logger] = None):
        """
        Initialize the PyTorch segmentation trainer.

        Args:
            model_path: Path to the model directory containing metadata
            logger: Optional custom logger instance. If None, creates a Logger with model_path as log_dir
        """
        self.model_path = model_path

        # Setup logger
        if logger is not None:
            self.logger = logger
        else:
            # Create default logger using model_path as log_dir
            log_dir = os.path.join(model_path, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            model_name = os.path.basename(model_path) or 'pytorch_model'
            self.logger = Logger(log_dir, model_name)

        # Load model metadata
        self.metadata = load_model_metadata(model_path)

        # Extract key metadata (handle both snake_case and camelCase keys)
        self.class_names = self.metadata.get('class_names') or self.metadata.get('classNames')
        self.num_classes = len(self.class_names)
        self.image_size = self.metadata.get('image_size') or self.metadata.get('sxy')

        # Get device from config or environment
        device_str = os.getenv('CODAVISION_PYTORCH_DEVICE', ModelDefaults.PYTORCH_DEVICE)
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)

        self.logger.logger.info(f"Using device: {self.device}")

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_iou': [],
            'learning_rates': [],
            'val_iterations': []  # Track which iterations had validation
        }

        # Training settings (can be overridden)
        self.validation_frequency = int(
            os.getenv('CODAVISION_VALIDATION_FREQUENCY', ModelDefaults.VALIDATION_FREQUENCY)
        )
        self.early_stopping_patience = 6  # validations, not epochs
        self.lr_reduction_patience = 1  # validations
        self.lr_reduction_factor = 0.5
        self.min_lr = 1e-7

        # Advanced optimization settings
        self.use_amp = os.getenv('CODAVISION_PYTORCH_AMP', 'false').lower() == 'true' or \
                       ModelDefaults.PYTORCH_AMP
        self.use_compile = os.getenv('CODAVISION_PYTORCH_COMPILE', 'false').lower() == 'true' or \
                          ModelDefaults.PYTORCH_COMPILE
        self.gradient_accumulation_steps = int(
            os.getenv('CODAVISION_GRADIENT_ACCUMULATION_STEPS',
                     getattr(ModelDefaults, 'GRADIENT_ACCUMULATION_STEPS', 1))
        )

        # AMP scaler (only if using AMP and CUDA)
        self.scaler = None
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.logger.info("Automatic Mixed Precision (AMP) enabled")

        self.logger.logger.info(f"Validation frequency: {self.validation_frequency} iterations")
        self.logger.logger.info(f"Early stopping patience: {self.early_stopping_patience} validations")
        self.logger.logger.info(f"LR reduction patience: {self.lr_reduction_patience} validations")

    def _load_model_data(self) -> Tuple[Dict, List[str]]:
        """
        Load annotations and image lists from model directory.

        Returns:
            Tuple of (annotations dict, image list)
        """
        # Load annotations
        annotation_path = os.path.join(self.model_path, 'annotations.pkl')
        with open(annotation_path, 'rb') as f:
            annotations = pickle.load(f)

        # Load image lists
        train_list_path = os.path.join(self.model_path, 'train_list.pkl')
        with open(train_list_path, 'rb') as f:
            image_list = pickle.load(f)

        return annotations, image_list

    def _detect_tile_format(self) -> str:
        """
        Detect tile file format from model metadata or existing files.

        Returns:
            File extension with leading dot (e.g., '.png', '.tif')
        """
        # Try to load tile format from model metadata
        metadata_path = os.path.join(self.model_path, 'net.pkl')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                if 'tile_format' in metadata:
                    fmt = metadata['tile_format']
                    return f'.{fmt}' if not fmt.startswith('.') else fmt
            except Exception as e:
                self.logger.logger.warning(f"Could not load tile_format from metadata: {e}")

        # Fallback: detect from existing files in training/im directory
        training_im_dir = os.path.join(self.model_path, 'training', 'im')
        if os.path.exists(training_im_dir):
            files = os.listdir(training_im_dir)
            for file in files:
                if file.endswith('.png'):
                    return '.png'
                elif file.endswith('.tif') or file.endswith('.tiff'):
                    return '.tif'

        # Default to PNG
        self.logger.logger.warning("Could not detect tile format, defaulting to .png")
        return '.png'

    def _validate_tile_files(self, image_paths: List[str], mask_paths: List[str],
                            dataset_name: str = "dataset") -> None:
        """
        Validate that tile files exist and provide helpful error messages.

        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            dataset_name: Name of dataset for error message (e.g., "training", "validation")

        Raises:
            FileNotFoundError: If any files are missing
        """
        missing_images = [p for p in image_paths if not os.path.exists(p)]
        missing_masks = [p for p in mask_paths if not os.path.exists(p)]

        if missing_images or missing_masks:
            error_msg = f"Missing {dataset_name} tile files:\n"
            if missing_images:
                error_msg += f"  Images: {len(missing_images)}/{len(image_paths)} missing\n"
                error_msg += f"    First 5 missing: {[os.path.basename(p) for p in missing_images[:5]]}\n"
            if missing_masks:
                error_msg += f"  Masks: {len(missing_masks)}/{len(mask_paths)} missing\n"
                error_msg += f"    First 5 missing: {[os.path.basename(p) for p in missing_masks[:5]]}\n"
            error_msg += f"\nExpected directory structure:\n"
            error_msg += f"  {self.model_path}/training/im/     (for images)\n"
            error_msg += f"  {self.model_path}/training/label/  (for masks)\n"
            raise FileNotFoundError(error_msg)

    def build_model(self, architecture: str = 'DeepLabV3_plus', **kwargs) -> nn.Module:
        """
        Build the segmentation model using the factory pattern.

        Args:
            architecture: Model architecture name
            **kwargs: Additional arguments for model creation

        Returns:
            PyTorch segmentation model
        """
        # Use factory pattern to create model (note: model_call uses uppercase parameter names)
        model = model_call(
            architecture,
            IMAGE_SIZE=self.image_size,
            NUM_CLASSES=self.num_classes,
            framework='pytorch',
            wrap_with_adapter=False,  # Get raw nn.Module for PyTorch-native training
            **kwargs
        )

        # Move model to device
        model = model.to(self.device)

        # Optional: Use torch.compile() for optimization (PyTorch 2.0+)
        if self.use_compile:
            try:
                model = torch.compile(model)
                self.logger.logger.info("torch.compile() enabled for model optimization")
            except Exception as e:
                self.logger.logger.info(f"Warning: torch.compile() failed, continuing without it: {e}")

        self.model = model
        return model

    def setup_optimizer(self, learning_rate: float = 0.001, weight_decay: float = 0.0001):
        """
        Setup optimizer (Adam with weight decay for L2 regularization).

        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength (matches TensorFlow kernel_regularizer)
        """
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.logger.logger.info(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")

    def setup_loss(self, class_weights: Union[List[float], np.ndarray]):
        """
        Setup loss function with class weights.

        Args:
            class_weights: Weights for each class
        """
        self.criterion = WeightedCrossEntropyLoss(class_weights)
        self.criterion = self.criterion.to(self.device)
        self.logger.logger.info("Loss function: WeightedCrossEntropyLoss")

    def setup_scheduler(self, mode: str = 'min', factor: float = 0.5,
                       patience: int = 1):
        """
        Setup learning rate scheduler (ReduceLROnPlateau).

        Args:
            mode: 'min' for minimizing metric (loss), 'max' for maximizing (accuracy)
            factor: Factor by which to reduce LR
            patience: Number of validations with no improvement before reducing LR
        """
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold_mode='rel',
            threshold=0.0001,
            min_lr=self.min_lr
        )
        self.logger.logger.info(f"LR Scheduler: ReduceLROnPlateau (patience={patience}, factor={factor})")

    def create_dataloaders(
        self,
        annotations: Dict,
        image_list: List[str],
        batch_size: int = 4,
        num_workers: int = 4,
        validation_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training and validation.

        Args:
            annotations: Dictionary of annotations
            image_list: List of image identifiers
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            validation_split: Fraction of data for validation

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Split into train and validation
        np.random.seed(42)  # For reproducibility
        shuffled_indices = np.random.permutation(len(image_list))
        val_size = int(len(image_list) * validation_split)

        val_indices = shuffled_indices[:val_size]
        train_indices = shuffled_indices[val_size:]

        train_image_list = [image_list[i] for i in train_indices]
        val_image_list = [image_list[i] for i in val_indices]

        # Detect file format from model metadata or existing files
        file_format = self._detect_tile_format()

        # Build full paths to images and masks
        def build_paths(image_ids):
            image_paths = [
                os.path.join(self.model_path, 'training', 'im', f'{img_id}{file_format}')
                for img_id in image_ids
            ]
            mask_paths = [
                os.path.join(self.model_path, 'training', 'label', f'{img_id}{file_format}')
                for img_id in image_ids
            ]
            return image_paths, mask_paths

        train_image_paths, train_mask_paths = build_paths(train_image_list)
        val_image_paths, val_mask_paths = build_paths(val_image_list)

        # Validate that files exist
        self._validate_tile_files(train_image_paths, train_mask_paths, "training")
        self._validate_tile_files(val_image_paths, val_mask_paths, "validation")

        # Create datasets
        train_dataset = PyTorchSegmentationDataset(
            image_paths=train_image_paths,
            mask_paths=train_mask_paths,
            image_size=self.image_size,
            augment=True
        )

        val_dataset = PyTorchSegmentationDataset(
            image_paths=val_image_paths,
            mask_paths=val_mask_paths,
            image_size=self.image_size,
            augment=False  # No augmentation for validation
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(num_workers > 0),
            drop_last=True  # Drop incomplete batches to avoid batch norm issues with batch_size=1
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(num_workers > 0)
        )

        self.logger.logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        self.logger.logger.info(f"Batch size: {batch_size}, Num workers: {num_workers}")

        return train_dataloader, val_dataloader

    def calculate_class_weights(self, annotations: Dict, image_list: List[str]) -> np.ndarray:
        """
        Calculate class weights from label mask files for balanced training.

        Uses median frequency balancing, consistent with TensorFlow implementation.
        Loads actual mask files from disk to count pixel frequencies per class.

        Args:
            annotations: Dictionary of annotations (not used, kept for API compatibility)
            image_list: List of image identifiers

        Returns:
            Array of class weights using median frequency balancing

        Raises:
            ValueError: If no valid mask files are found or all counts are zero
        """
        from PIL import Image
        import glob

        # Construct path to label masks
        label_dir = os.path.join(self.model_path, 'training', 'label')

        # Check if directory exists
        if not os.path.isdir(label_dir):
            raise ValueError(f"Label directory not found: {label_dir}")

        # Track pixel counts per class
        class_pixels = np.zeros(self.num_classes, dtype=np.int64)
        image_pixels = np.zeros(self.num_classes, dtype=np.int64)
        epsilon = 1e-7  # Prevent division by zero

        # Try both png and tif formats (modern uses png, legacy uses tif)
        valid_extensions = ['png', 'tif', 'tiff']
        masks_loaded = 0

        for image_id in image_list:
            mask_path = None

            # Try to find the mask file with supported extensions
            for ext in valid_extensions:
                candidate_path = os.path.join(label_dir, f"{image_id}.{ext}")
                if os.path.exists(candidate_path):
                    mask_path = candidate_path
                    break

            if mask_path is None:
                self.logger.logger.warning(f"Mask file not found for image: {image_id}")
                continue

            try:
                # Load mask as grayscale
                mask = Image.open(mask_path).convert('L')
                mask = np.array(mask, dtype=np.int32)

                # Count pixels per class
                unique, counts = np.unique(mask, return_counts=True)
                total_pixels = mask.size

                for val, count in zip(unique, counts):
                    if 0 <= val < self.num_classes:
                        class_pixels[val] += count
                        image_pixels[val] += total_pixels

                masks_loaded += 1

            except Exception as e:
                self.logger.logger.warning(f"Failed to load mask {mask_path}: {e}")
                continue

        if masks_loaded == 0:
            raise ValueError(f"No valid mask files loaded from {label_dir}")

        self.logger.logger.info(f"Loaded {masks_loaded} mask files for class weight calculation")

        # Calculate frequencies (avoid division by zero)
        freq = class_pixels / (image_pixels + epsilon)

        # Handle invalid values
        freq[np.isinf(freq) | np.isnan(freq)] = epsilon

        # Calculate weights using median frequency balancing
        median_freq = np.median(freq[freq > 0]) if np.any(freq > 0) else epsilon
        class_weights = median_freq / (freq + epsilon)

        # Clamp weights to reasonable range to prevent extreme values
        class_weights = np.clip(class_weights, 0.1, 10.0)

        # Log class distribution information
        self.logger.logger.info("\nClass pixel frequencies:")
        for i, (f, pixels) in enumerate(zip(freq, class_pixels)):
            self.logger.logger.info(f"  Class {i}: {f:.6f} ({pixels:,} pixels)")

        self.logger.logger.info(f"\nMedian frequency: {median_freq:.6f}")
        self.logger.logger.info("\nClass weights (median frequency balancing):")
        for i, w in enumerate(class_weights):
            self.logger.logger.info(f"  Class {i}: {w:.4f}")

        return class_weights.astype(np.float32)

    def run_validation(self, val_dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Run validation and compute metrics.

        Args:
            val_dataloader: Validation data loader

        Returns:
            Tuple of (val_loss, val_accuracy, val_iou)
        """
        self.model.eval()

        val_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        iou_per_class = np.zeros(self.num_classes)
        class_present = np.zeros(self.num_classes, dtype=bool)

        with torch.no_grad():
            for images, masks in val_dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Compute loss
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()

                # Compute accuracy
                predictions = torch.argmax(outputs, dim=1)
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)

                correct_pixels += (predictions == masks).sum().item()
                total_pixels += masks.numel()

                # Compute IoU per class
                for class_idx in range(self.num_classes):
                    pred_mask = (predictions == class_idx)
                    true_mask = (masks == class_idx)

                    intersection = (pred_mask & true_mask).sum().item()
                    union = (pred_mask | true_mask).sum().item()

                    if union > 0:
                        iou_per_class[class_idx] += intersection / union
                        class_present[class_idx] = True

        # Compute average metrics
        val_loss /= len(val_dataloader)
        val_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0

        # Mean IoU across present classes
        if np.any(class_present):
            val_iou = np.mean(iou_per_class[class_present])
        else:
            val_iou = 0.0

        self.model.train()  # Back to training mode

        return val_loss, val_accuracy, val_iou

    def save_checkpoint(
        self,
        iteration: int,
        val_loss: float,
        val_accuracy: float,
        is_best: bool = False
    ):
        """
        Save model checkpoint with model-type-specific naming.

        Saves checkpoints using the same naming convention as TensorFlow models
        to ensure compatibility with inference code.

        Args:
            iteration: Current training iteration
            val_loss: Validation loss at this checkpoint
            val_accuracy: Validation accuracy at this checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'metadata': self.metadata,
            'history': self.history
        }

        # Get model type from metadata (e.g., 'DeepLabV3_plus', 'UNet')
        model_type = self.metadata.get('model_type', 'DeepLabV3_plus')

        # Save latest checkpoint (final model)
        final_model_path = os.path.join(self.model_path, f'{model_type}.pth')
        torch.save(checkpoint, final_model_path)

        # Also save generic checkpoint for backward compatibility
        checkpoint_path = os.path.join(self.model_path, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_model_path = os.path.join(self.model_path, f'best_model_{model_type}.pth')
            torch.save(checkpoint, best_model_path)

            # Also save generic best checkpoint for backward compatibility
            best_checkpoint_path = os.path.join(self.model_path, 'checkpoint_best.pth')
            torch.save(checkpoint, best_checkpoint_path)

            self.logger.logger.info(f"Best model saved (val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f})")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Iteration number to resume from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        iteration = checkpoint.get('iteration', 0)
        self.logger.logger.info(f"Checkpoint loaded from iteration {iteration}")

        return iteration

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        num_workers: int = 4,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict:
        """
        Main training method with full feature set.

        Args:
            epochs: Number of epochs to train
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            validation_split: Fraction of data for validation
            num_workers: Number of data loading workers
            resume_from_checkpoint: Optional path to checkpoint to resume from

        Returns:
            Training history dictionary
        """
        self.logger.logger.info("=" * 50)
        self.logger.logger.info("Starting PyTorch Training")
        self.logger.logger.info("=" * 50)

        # Load data
        annotations, image_list = self._load_model_data()

        # Calculate class weights
        class_weights = self.calculate_class_weights(annotations, image_list)

        # Validate class weights before proceeding with training
        if np.any(np.isnan(class_weights)):
            raise ValueError(
                f"Class weights contain NaN values: {class_weights}. "
                "Training cannot proceed. Check that training mask files exist and contain valid class labels."
            )

        if np.any(np.isinf(class_weights)):
            raise ValueError(
                f"Class weights contain infinite values: {class_weights}. "
                "Training cannot proceed. This typically indicates division by zero."
            )

        if np.all(class_weights == 0):
            raise ValueError(
                "All class weights are zero. Training cannot proceed. "
                "Check that mask files contain valid class labels."
            )

        self.logger.logger.info("Class weights validation passed - all weights are finite and non-zero")

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Setup training components
        self.setup_optimizer(learning_rate=learning_rate)
        self.setup_loss(class_weights=class_weights)
        self.setup_scheduler(mode='min', factor=self.lr_reduction_factor,
                           patience=self.lr_reduction_patience)

        # Create dataloaders
        train_dataloader, val_dataloader = self.create_dataloaders(
            annotations, image_list, batch_size, num_workers, validation_split
        )

        # Resume from checkpoint if specified
        start_iteration = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            start_iteration = self.load_checkpoint(resume_from_checkpoint)

        # Training loop
        self.model.train()
        global_iteration = start_iteration
        best_val_loss = float('inf')
        validations_without_improvement = 0
        validations_since_lr_reduction = 0

        total_iterations = len(train_dataloader) * epochs
        self.logger.logger.info(f"Total iterations: {total_iterations}")

        # Ensure at least one validation
        total_validations = max(1, total_iterations // self.validation_frequency)
        self.logger.logger.info(f"Expected validations: {total_validations}")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (images, masks) in enumerate(train_dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass with optional AMP
                if self.use_amp and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                    # Backward pass with scaled gradients
                    self.scaler.scale(loss).backward()

                    # Optimizer step (with gradient accumulation)
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    # Standard training without AMP
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    # Optimizer step (with gradient accumulation)
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                batch_count += 1
                global_iteration += 1

                # Validation at specified intervals
                if global_iteration % self.validation_frequency == 0 or \
                   global_iteration == total_iterations:

                    # Run validation
                    val_loss, val_accuracy, val_iou = self.run_validation(val_dataloader)

                    # Track metrics
                    self.history['val_loss'].append(val_loss)
                    self.history['val_accuracy'].append(val_accuracy)
                    self.history['val_iou'].append(val_iou)
                    self.history['val_iterations'].append(global_iteration)
                    self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

                    # Log progress
                    elapsed_time = time.time() - start_time
                    self.logger.logger.info(
                        f"Iter {global_iteration}/{total_iterations} | "
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {epoch_loss/batch_count:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_accuracy:.4f} | "
                        f"Val IoU: {val_iou:.4f} | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                        f"Time: {elapsed_time:.1f}s"
                    )

                    # Check for best model
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        validations_without_improvement = 0
                        validations_since_lr_reduction = 0
                    else:
                        validations_without_improvement += 1
                        validations_since_lr_reduction += 1

                    # Save checkpoint
                    self.save_checkpoint(global_iteration, val_loss, val_accuracy, is_best)

                    # Learning rate reduction
                    if self.scheduler:
                        self.scheduler.step(val_loss)

                    # Early stopping
                    if validations_without_improvement >= self.early_stopping_patience:
                        self.logger.logger.info(
                            f"Early stopping triggered after {validations_without_improvement} "
                            f"validations without improvement"
                        )
                        break

            # Record average epoch loss
            self.history['train_loss'].append(epoch_loss / batch_count)

            # Early stopping break from epoch loop
            if validations_without_improvement >= self.early_stopping_patience:
                break

        total_time = time.time() - start_time
        self.logger.logger.info("=" * 50)
        self.logger.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        self.logger.logger.info("=" * 50)

        # Save final training history
        history_path = os.path.join(self.model_path, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)

        return self.history


class PyTorchDeepLabV3PlusTrainer(PyTorchSegmentationTrainer):
    """
    Specialized trainer for DeepLabV3+ architecture.

    This class extends PyTorchSegmentationTrainer with DeepLabV3+ specific
    behavior, matching the TensorFlow DeepLabV3PlusTrainer implementation.
    """

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        num_workers: int = 4,
        resume_from_checkpoint: Optional[str] = None,
        freeze_encoder: bool = True
    ) -> Dict:
        """
        Train DeepLabV3+ model with specific configuration.

        Args:
            epochs: Number of epochs to train
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            validation_split: Fraction of data for validation
            num_workers: Number of data loading workers
            resume_from_checkpoint: Optional path to checkpoint to resume from
            freeze_encoder: Whether to freeze encoder weights (default: True)

        Returns:
            Training history dictionary
        """
        # Build DeepLabV3+ model if not already built
        if self.model is None:
            self.build_model(architecture='DeepLabV3_plus')

        # Optionally freeze encoder
        if freeze_encoder and hasattr(self.model, 'freeze_encoder'):
            self.model.freeze_encoder()
            self.logger.logger.info("Encoder weights frozen (transfer learning)")

        # Call parent train method
        return super().train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
            num_workers=num_workers,
            resume_from_checkpoint=resume_from_checkpoint
        )


# Utility functions

def compute_iou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) for each class.

    Args:
        predictions: Predicted class indices [batch, height, width]
        targets: Ground truth class indices [batch, height, width]
        num_classes: Number of classes

    Returns:
        Array of IoU values for each class
    """
    iou_per_class = np.zeros(num_classes)

    for class_idx in range(num_classes):
        pred_mask = (predictions == class_idx)
        true_mask = (targets == class_idx)

        intersection = (pred_mask & true_mask).sum().item()
        union = (pred_mask | true_mask).sum().item()

        if union > 0:
            iou_per_class[class_idx] = intersection / union
        else:
            iou_per_class[class_idx] = float('nan')  # Class not present

    return iou_per_class


def compute_pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute pixel-wise accuracy.

    Args:
        predictions: Predicted class indices [batch, height, width]
        targets: Ground truth class indices [batch, height, width]

    Returns:
        Pixel accuracy as a float
    """
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0
