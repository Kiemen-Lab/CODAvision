"""
This module implements training functions for semantic segmentation models
using TensorFlow and Keras. It includes custom loss functions, data generators,
and training utilities for various architectures including DeepLabV3+ and UNet.
The original implementation of DeepLabv3+ was based on:
https://keras.io/examples/vision/deeplabv3_plus/
"""

import time
import pickle
import os
import warnings
import platform
from glob import glob
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import tensorflow as tf
import keras
import logging
import traceback

from base.models.backbones import model_call, unfreeze_model
from base.utils.logger import Logger
from base.models.utils import save_model_metadata, setup_gpu, calculate_class_weights, get_model_paths, create_distribution_strategy
from base.data.loaders import create_dataset, create_training_dataset, create_validation_dataset, load_model_metadata
from base.config import DataConfig, ModelDefaults

# Suppress warnings and TensorFlow logs
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"

# Setup a logger for the module level
module_logger = logging.getLogger(__name__)


def is_distributed_dataset(dataset):
    """
    Check if a dataset is a distributed dataset.
    
    Args:
        dataset: TensorFlow dataset to check
        
    Returns:
        Boolean indicating if the dataset is distributed
    """
    return hasattr(dataset, '_input_dataset') or 'DistributedDataset' in type(dataset).__name__


class WeightedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Custom loss function implementing weighted sparse categorical crossentropy.

    This class extends the base Keras Loss class to provide a weighted version
    of sparse categorical crossentropy, useful for handling class imbalance
    in segmentation tasks.

    Attributes:
        class_weights (tf.Tensor): Weights for each class
        from_logits (bool): Whether the predictions are logits
    """

    def __init__(
        self,
        class_weights: Union[List[float], np.ndarray],
        from_logits: bool = True,
        reduction: str = 'sum_over_batch_size',
        name: Optional[str] = None
    ):
        """
        Initialize the weighted sparse categorical crossentropy loss.

        Args:
            class_weights: Weights for each class
            from_logits: Whether the predictions are logits
            reduction: Type of reduction to apply to the loss
            name: Name of the loss function
        """
        super().__init__(reduction=reduction, name=name)

        # Convert and normalize weights to sum to 1 (MATLAB behavior)
        weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        weights_sum = tf.reduce_sum(weights)
        self.class_weights = weights / weights_sum

        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute weighted loss using per-class approach (MATLAB-aligned).

        This implementation computes the mean loss for each class separately,
        then weights and sums them. This ensures true class balancing even
        with highly imbalanced datasets, matching MATLAB's behavior.

        Args:
            y_true: Ground truth labels [batch, height, width, 1]
            y_pred: Model predictions [batch, height, width, num_classes]

        Returns:
            Computed loss value (scalar)
        """
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]

        # Flatten tensors for easier processing
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1, num_classes])

        # Compute per-pixel crossentropy loss
        per_pixel_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_flat, y_pred_flat, from_logits=self.from_logits
        )

        # Per-class mean loss computation (MATLAB-style)
        # Use tf.map_fn for graph mode compatibility
        def compute_class_loss(class_idx):
            """Compute weighted mean loss for a specific class."""
            # Create binary mask for pixels belonging to this class
            class_mask = tf.cast(tf.equal(y_true_flat, class_idx), tf.float32)
            num_pixels_in_class = tf.reduce_sum(class_mask)

            # Compute mean loss for this class (with numerical stability)
            # If class is not present in batch, contribute 0 to total loss
            masked_loss = per_pixel_loss * class_mask
            class_mean_loss = tf.cond(
                num_pixels_in_class > 0,
                lambda: tf.reduce_sum(masked_loss) / num_pixels_in_class,
                lambda: 0.0  # Empty class contributes nothing
            )

            # Apply normalized class weight
            return self.class_weights[class_idx] * class_mean_loss

        # Compute weighted loss for each class using vectorized map
        class_losses = tf.map_fn(
            compute_class_loss,
            tf.range(num_classes),
            dtype=tf.float32
        )

        # Sum all class contributions (MATLAB-style: sum not mean)
        return tf.reduce_sum(class_losses)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights.numpy().tolist(),
            "from_logits": self.from_logits,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WeightedSparseCategoricalCrossentropy':
        """Create an instance from configuration dictionary."""
        class_weights = tf.convert_to_tensor(config.pop("class_weights"), dtype=tf.float32)
        return cls(class_weights=class_weights, **config)


class RegularizationLossCallback(keras.callbacks.Callback):
    """
    Callback for monitoring regularization loss separately from data loss.

    This callback tracks the L2 regularization loss during training and
    logs the ratio of regularization loss to data loss for debugging.
    """

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the regularization loss callback.

        Args:
            logger: Optional logger for outputting regularization loss info
        """
        super().__init__()
        self.logger = logger
        self.regularization_losses = []
        self.data_losses = []
        self.total_losses = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate and log regularization loss at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of metrics from the epoch
        """
        if self.model and logs:
            # Get total loss from logs
            total_loss = logs.get('loss', 0)

            # Calculate regularization loss (sum of all regularization losses in the model)
            regularization_loss = 0
            if hasattr(self.model, 'losses') and self.model.losses:
                regularization_loss = tf.reduce_sum(self.model.losses).numpy()

            # Calculate data loss (total loss minus regularization loss)
            data_loss = total_loss - regularization_loss

            # Store losses
            self.regularization_losses.append(float(regularization_loss))
            self.data_losses.append(float(data_loss))
            self.total_losses.append(float(total_loss))

            # Calculate ratio
            ratio = regularization_loss / data_loss if data_loss > 0 else 0

            # Log if logger is available
            if self.logger:
                self.logger.logger.info(
                    f"Epoch {epoch + 1} - Regularization Loss: {regularization_loss:.6f}, "
                    f"Data Loss: {data_loss:.6f}, Ratio: {ratio:.4f}"
                )


class BatchAccuracyCallback(keras.callbacks.Callback):
    """
    Custom callback for tracking batch-level metrics and performing validation.

    This callback allows for more fine-grained monitoring of training progress,
    including running validation at specified intervals and implementing
    early stopping and learning rate reduction.

    Warning:
        This callback is TensorFlow/Keras-specific and is not compatible with
        PyTorch models. For PyTorch training, use the PyTorchSegmentationTrainer
        classes in base.models.training_pytorch, which implement their own
        training loop with built-in validation, early stopping, and LR scheduling.
    """

    def __init__(
        self,
        model: keras.Model,
        val_data: tf.data.Dataset,
        loss_function: tf.keras.losses.Loss,
        logger: Optional[Logger] = None,
        validation_frequency: int = 128,
        early_stopping: bool = True,
        reduce_lr_on_plateau: bool = True,
        monitor: str = 'val_accuracy',
        es_patience: int = 6,
        lr_patience: int = 1,
        lr_factor: float = 0.75,
        verbose: int = 0,
        save_best_model: bool = True,
        filepath: str = 'best_model.h5',
        **kwargs  # For backward compatibility
    ):
        """
        Initialize the callback.

        Args:
            model: The model being trained
            val_data: Validation dataset
            loss_function: Loss function used for validation
            logger: Logger for recording training information
            validation_frequency: Number of iterations between validations (default: 128)
            early_stopping: Whether to use early stopping
            reduce_lr_on_plateau: Whether to reduce learning rate on plateau
            monitor: Metric to monitor ('val_loss' or 'val_accuracy')
            es_patience: Patience for early stopping
            lr_patience: Patience for learning rate reduction
            lr_factor: Factor to reduce learning rate by
            verbose: Verbosity level
            save_best_model: Whether to save the best model
            filepath: Path to save the best model to
        """
        # Handle deprecated num_validations parameter for backward compatibility
        if 'num_validations' in kwargs:
            import warnings
            num_val = kwargs.pop('num_validations')

            # Convert to iteration-based frequency
            # Estimate steps per epoch (default assumption)
            estimated_steps_per_epoch = 150  # Default estimate
            new_freq = max(1, estimated_steps_per_epoch // num_val)

            warnings.warn(
                f"num_validations={num_val} is deprecated. Use validation_frequency instead. "
                f"Converting to validation_frequency={new_freq} based on estimated "
                f"{estimated_steps_per_epoch} steps/epoch.",
                DeprecationWarning,
                stacklevel=2
            )
            validation_frequency = new_freq

        # Check for environment variable override
        env_freq = os.environ.get('CODAVISION_VALIDATION_FREQUENCY')
        if env_freq:
            try:
                validation_frequency = int(env_freq)
                if logger:
                    logger.info(
                        f"Using validation frequency {env_freq} from environment variable"
                    )
            except ValueError:
                if logger:
                    logger.warning(
                        f"Invalid CODAVISION_VALIDATION_FREQUENCY value: {env_freq}"
                    )

        super(BatchAccuracyCallback, self).__init__()
        self.logger = logger
        self._model = model
        self.loss_function = loss_function

        # Metrics tracking
        self.batch_accuracies = []
        self.batch_numbers = []
        self.batch_losses = []
        self.epoch_indices = []
        self.epoch_numbers = []
        self.current_epoch = 0
        self.val_data = val_data
        self.validation_frequency = validation_frequency
        self.global_step = 0  # Track iterations across all epochs
        self.validation_counter = 0  # Track total number of validations
        self.validation_losses = []
        self.validation_accuracies = []
        self.val_indices = []

        # Early stopping configuration
        self.early_stopping = early_stopping
        self.reduce_lr = reduce_lr_on_plateau
        self.monitor = monitor
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.original_lr_patience = lr_patience
        self.verbose = verbose

        if monitor not in ['val_accuracy', 'val_loss']:
            raise ValueError("Monitor can only be 'val_loss' or 'val_accuracy'")

        self.mode = 'min' if monitor == 'val_loss' else 'max'
        self.save_best_model = save_best_model
        self.save_path = filepath
        self.monitor_op = np.less if self.mode == 'min' else np.greater
        self.best = np.Inf if self.mode == 'min' else -np.Inf
        self.wait = 0
        self.epoch_wait = 0
        self.stopped_epoch = 0
        self.lr_factor = lr_factor

        # Log validation dataset information
        if self.logger:
            self.logger.logger.debug("\nInitial Validation Dataset Information:")
            # Only log dataset info if it's not distributed (to avoid .take() errors)
            if not is_distributed_dataset(val_data):
                self.logger.log_dataset_info(val_data, "Initial-Validation")
            else:
                self.logger.logger.debug("Validation dataset is distributed - skipping detailed analysis")

    @property
    def model(self):
        """Get the model being trained."""
        return self._model

    @model.setter
    def model(self, value):
        """Set the model being trained."""
        self._model = value

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the start of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs (unused)
        """
        self.current_epoch = epoch
        self.epoch_indices.append(self.current_epoch)

        # No longer need to calculate validation steps per epoch
        # Validation is now based on global iteration count
        self.current_step = 0
        # Note: validation_counter is NOT reset per epoch - it tracks total validations

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch.

        Args:
            batch: Current batch number within the epoch
            logs: Dictionary of logs from training
        """
        logs = logs or {}
        self.current_step += 1
        self.global_step += 1  # Track global iteration count

        # Run validation at specified iteration frequency
        if self.global_step % self.validation_frequency == 0:
            self.run_validation()
            self.validation_counter += 1
            # Record the global step
            self.val_indices.append(self.global_step)

        # Record batch metrics
        accuracy = logs.get('accuracy')
        if accuracy is not None:
            self.batch_accuracies.append(accuracy)
            # Record the global batch number (use global_step)
            self.batch_numbers.append(self.global_step)

        loss = logs.get('loss')
        if loss is not None:
            self.batch_losses.append(loss)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.
        Reduces learning rate every epoch to match MATLAB piecewise schedule.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs (unused)
        """
        # MATLAB-style: Drop LR every epoch unconditionally (piecewise schedule)
        old_lr = float(self._model.optimizer.learning_rate.numpy())
        new_lr = old_lr * self.lr_factor
        self._model.optimizer.learning_rate.assign(new_lr)

        if self.verbose > 0 and self.logger:
            self.logger.logger.debug(
                f"\nEpoch {epoch + 1}: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}"
            )

    def on_train_end(self, logs=None):
        """
        Called at the end of training.
        Ensures at least one validation occurs even for very short training runs.

        Args:
            logs: Dictionary of logs (unused)
        """
        if self.validation_counter == 0 and self.val_data is not None:
            if self.logger:
                self.logger.logger.warning(
                    f"No validation occurred during training "
                    f"(training ended before iteration {self.validation_frequency}). "
                    f"Running final validation..."
                )
            self.run_validation()
            self.val_indices.append(self.global_step)

    def run_validation(self):
        """
        Run validation on the validation dataset.

        This method computes validation loss and accuracy, and implements
        early stopping and model saving logic.
        """
        val_loss_total = 0
        val_accuracy_total = 0
        num_batches = 0

        try:
            # Evaluate on validation data
            for x_val, y_val in self.val_data:
                y_val = tf.cast(y_val, dtype=tf.int32)
                val_logits = self._model(x_val, training=False)
                num_classes = val_logits.shape[-1]

                val_logits_flat = tf.reshape(val_logits, [-1, num_classes])
                y_val_flat = tf.reshape(y_val, [-1])

                predictions = tf.argmax(val_logits_flat, axis=1)
                predictions = tf.cast(predictions, dtype=tf.int32)

                val_loss = self.loss_function(y_val_flat, val_logits_flat)
                val_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_val_flat), tf.float32))

                val_loss_total += tf.reduce_mean(val_loss).numpy()
                val_accuracy_total += val_accuracy.numpy()
                num_batches += 1

            # Compute averages
            val_loss_avg = val_loss_total / num_batches
            val_accuracy_avg = val_accuracy_total / num_batches

            # Check for NaN/Inf and terminate training if detected
            if np.isnan(val_loss_avg) or np.isinf(val_loss_avg):
                if self.logger:
                    self.logger.logger.error(
                        f"Validation loss is NaN/Inf at validation #{self.validation_counter}. "
                        f"Terminating training. Check: learning rate, class weights, batch size."
                    )
                self._model.stop_training = True
                return

            # Record validation metrics
            self.validation_losses.append(val_loss_avg)
            self.validation_accuracies.append(val_accuracy_avg)

            # Log validation metrics
            if self.logger:
                self.logger.log_validation_metrics(
                    val_logits, y_val,
                    loss=val_loss_avg,
                    accuracy=val_accuracy_avg
                )

            # Early stopping logic
            if self.early_stopping:
                if self.monitor == 'val_loss':
                    current = val_loss_avg
                elif self.monitor == 'val_accuracy':
                    current = val_accuracy_avg

                if current is None:
                    return

                # Check if this is the best model so far
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0

                    # Save the best model
                    if self.save_best_model:
                        tf.keras.models.save_model(
                            self._model,
                            self.save_path,
                            save_format='tf'
                        )
                        if self.verbose > 0 and self.logger:
                            self.logger.logger.debug(
                                f'\nEpoch {self.current_epoch + 1}: '
                                f'Model saved to {self.save_path}'
                            )
                else:
                    # Increment wait counter for early stopping
                    self.wait += 1
                    if self.wait >= self.es_patience and self.early_stopping:
                        self.stopped_epoch = self.current_epoch
                        self._model.stop_training = True
                        if self.verbose > 0 and self.logger:
                            self.logger.logger.debug(
                                f'\nEpoch {self.current_epoch + 1}: early stopping'
                            )

        except Exception as e:
            # Log errors and re-raise
            if self.logger:
                self.logger.log_error(f"Validation failed: {str(e)}")
            raise


class SegmentationModelTrainer:
    """
    Base class for training semantic segmentation models.

    This class handles common functionality for training semantic segmentation
    models, including data loading, preprocessing, and model training.
    """

    def __init__(self, model_path: str):
        """
        Initialize the trainer.

        Args:
            model_path: Path to the directory containing model data
        """
        self.model_path = model_path
        self.logger = None
        self.model = None
        self.model_data = None
        self.model_type = None
        self.image_size = None
        self.num_classes = None
        self.class_names = None
        self.batch_size = None
        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None
        self.loss_function = None
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.train_steps_per_epoch = 0
        self.val_steps_per_epoch = 0
        self.l2_regularization_weight = None
        self.use_adamw_optimizer = False

        # GPU information for logging
        self.gpu_info = self._setup_gpu()

        # Initialize distribution strategy for multi-GPU training
        self.strategy, self.num_gpus = create_distribution_strategy()

        # Initialize model data and parameters
        self._load_model_data()

    def _setup_gpu(self):
        """
        Configure TensorFlow to use GPU and return GPU information.
        """
        return setup_gpu()

    def _load_model_data(self):
        """
        Load model data from the pickle file.
        """
        try:
            self.model_data = load_model_metadata(self.model_path)

            # Extract needed parameters
            self.class_names = self.model_data.get('classNames')
            self.image_size = self.model_data.get('sxy')
            self.model_type = self.model_data.get('model_type', "DeepLabV3_plus")
            self.batch_size = self.model_data.get('batch_size', 3)
            self.l2_regularization_weight = self.model_data.get('l2_regularization_weight', 0)  # Default 0 to match MATLAB (no regularization)
            self.use_adamw_optimizer = self.model_data.get('use_adamw_optimizer', False)

            # Validate L2 regularization weight
            if self.l2_regularization_weight > 1e-3:
                module_logger.warning(f"L2 regularization weight {self.l2_regularization_weight} is very high and may cause training instability")
                module_logger.warning("Consider using values between 1e-6 and 1e-4 for better stability")
            elif self.l2_regularization_weight > 1e-4:
                module_logger.info(f"Using strong L2 regularization (weight={self.l2_regularization_weight})")
            elif self.l2_regularization_weight > 0:
                module_logger.info(f"Using L2 regularization (weight={self.l2_regularization_weight})")

            # Get standard paths
            self.model_paths = get_model_paths(self.model_path, self.model_type)

            # Validate essential parameters
            if None in [self.class_names, self.image_size, self.model_type]:
                raise ValueError("Missing required parameters in model data file")

            # Set number of classes
            self.num_classes = len(self.class_names) if self.class_names else 0

        except Exception as e:
            raise ValueError(f"Failed to load model data: {e}")

    def _prepare_data(self, seed: Optional[int] = None):
        """
        Prepare training and validation datasets with proper shuffling and optimization.

        This method finds training and validation images and masks,
        and creates optimized TensorFlow datasets for them.

        Args:
            seed: Random seed for shuffling (for reproducibility)

        Raises:
            ValueError: If no training or validation images are found
        """
        # Get paths
        train_path = self.model_paths['train_data']
        val_path = self.model_paths['val_data']

        # Get file format from model metadata (default to 'tif' for backward compatibility)
        tile_format = self.model_data.get('tile_format', 'tif')
        file_pattern = f"*.{tile_format}"

        # Find training images and masks
        train_images = sorted(glob(os.path.join(train_path, 'im', file_pattern)))
        train_masks = sorted(glob(os.path.join(train_path, 'label', file_pattern)))

        # Find validation images and masks
        val_images = sorted(glob(os.path.join(val_path, 'im', file_pattern)))
        val_masks = sorted(glob(os.path.join(val_path, 'label', file_pattern)))

        # Check if we found any data
        if not train_images or not train_masks:
            raise ValueError("No training images or masks found")

        if not val_images or not val_masks:
            raise ValueError("No validation images or masks found")

        # Calculate effective batch size for multi-GPU
        # The global batch size is split across GPUs
        effective_batch_size = self.batch_size
        if self.strategy and self.num_gpus > 1:
            # The batch_size from config is the global batch size
            # Each GPU gets batch_size // num_gpus for training
            module_logger.info(f"Multi-GPU training: Global batch size {self.batch_size}, "
                             f"per-GPU batch size {self.batch_size // self.num_gpus}")
            module_logger.info(f"Validation uses global batch size {self.batch_size} (single GPU)")
            effective_batch_size = self.batch_size // self.num_gpus

        # Create DataConfig for optimized data pipeline
        data_config = DataConfig()

        # Determine if we should cache based on dataset size
        # Cache small datasets (< 100 images) for better performance
        cache_train = len(train_images) < 100
        cache_val = len(val_images) < 100

        # Create optimized training dataset with shuffling
        self.train_dataset = create_training_dataset(
            train_images,
            train_masks,
            self.image_size,
            effective_batch_size,
            data_config=data_config,
            seed=seed,
            cache=cache_train,
            repeat=False  # Don't repeat, let fit() handle epochs
        )

        # Create optimized validation dataset without shuffling
        # Use global batch_size for validation since it runs on single GPU (not distributed)
        self.val_dataset = create_validation_dataset(
            val_images,
            val_masks,
            self.image_size,
            self.batch_size,  # Global batch size for non-distributed validation
            data_config=data_config,
            cache=cache_val
        )

        # Store the number of samples for steps_per_epoch calculation
        self.num_train_samples = len(train_images)
        self.num_val_samples = len(val_images)
        
        # Calculate steps per epoch (important for distributed training)
        self.train_steps_per_epoch = self.num_train_samples // self.batch_size
        self.val_steps_per_epoch = self.num_val_samples // self.batch_size
        
        # Log dataset information
        module_logger.info(f"Training samples: {self.num_train_samples}, "
                         f"steps per epoch: {self.train_steps_per_epoch}")
        module_logger.info(f"Validation samples: {self.num_val_samples}, "
                         f"steps per epoch: {self.val_steps_per_epoch}")

        # Distribute ONLY training dataset across GPUs if using MirroredStrategy
        # Validation dataset remains non-distributed to avoid PerReplica issues
        # in BatchAccuracyCallback.run_validation()
        if self.strategy and self.num_gpus > 1:
            self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            # Note: val_dataset intentionally NOT distributed for simpler callback iteration
            module_logger.info("Validation will run on single GPU (non-distributed) for callback compatibility")

        # Log the datasets if logger is available
        if self.logger:
            # For distributed datasets, the logger will skip detailed analysis
            self.logger.log_dataset_info(self.train_dataset, "Training")
            self.logger.log_dataset_info(self.val_dataset, "Validation")
            # Log that shuffling is enabled
            module_logger.info(f"Training dataset shuffling: ENABLED (buffer_size={data_config.shuffle_buffer_size})")
            module_logger.info(f"Validation dataset shuffling: DISABLED (standard practice)")
        else:
            module_logger.info(f"Training dataset shuffling: ENABLED")
            module_logger.info(f"Validation dataset shuffling: DISABLED")

    def _calculate_class_weights(self, mask_list):
        """
        Calculate class weights based on pixel frequency in labels.
        """
        return calculate_class_weights(mask_list, self.num_classes)

    def _create_loss_function(self):
        """
        Create a weighted loss function based on class frequencies.

        This method calculates class weights and creates a loss function
        that accounts for class imbalance.
        """
        # Get file format from model metadata (default to 'tif' for backward compatibility)
        tile_format = self.model_data.get('tile_format', 'tif')
        file_pattern = f"*.{tile_format}"

        # Find all training masks
        train_path = os.path.join(self.model_path, 'training')
        train_masks = sorted(glob(os.path.join(train_path, 'label', file_pattern)))

        # Calculate class weights
        self.class_weights = self._calculate_class_weights(train_masks)

        # Create the loss function
        self.loss_function = WeightedSparseCategoricalCrossentropy(
            class_weights=self.class_weights,
            from_logits=True,
            reduction='sum_over_batch_size'
        )

    def _create_model(self) -> keras.Model:
        """
        Create a segmentation model.

        This method should be implemented by subclasses to create
        the appropriate model architecture.

        Returns:
            Keras model for semantic segmentation
        """
        raise NotImplementedError(
            "Subclasses must implement _create_model method"
        )

    def _create_callbacks(self, model: keras.Model) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks.

        Args:
            model: The model being trained

        Returns:
            List of Keras callbacks
        """
        # Create logger if not already created
        if not self.logger:
            log_dir = os.path.join(self.model_path, 'logs')
            model_name = self.model_data.get('nm', 'unknown_model')
            self.logger = Logger(log_dir=log_dir, model_name=model_name)
            self.logger.log_system_info()

        # Create custom batch accuracy callback
        best_model_path = os.path.join(
            self.model_path, f"best_model_{self.model_type}.keras"
        )

        batch_callback = BatchAccuracyCallback(
            model=model,
            val_data=self.val_dataset,
            loss_function=self.loss_function,
            logger=self.logger,
            validation_frequency=ModelDefaults.VALIDATION_FREQUENCY,
            early_stopping=True,
            reduce_lr_on_plateau=True,
            monitor='val_accuracy',
            es_patience=6,       # Stop after 6 epochs without improvement
            lr_patience=1,       # Reduce LR after 1 epoch without improvement
            lr_factor=0.75,      # Reduce LR to 75% of current value
            verbose=1,
            save_best_model=True,
            filepath=best_model_path
        )

        callbacks = [batch_callback]

        # Add NaN termination callback to stop training immediately on NaN loss
        callbacks.append(keras.callbacks.TerminateOnNaN())

        # Add regularization loss monitoring if L2 regularization is enabled
        if self.l2_regularization_weight and self.l2_regularization_weight > 0:
            reg_callback = RegularizationLossCallback(logger=self.logger)
            callbacks.append(reg_callback)

        return callbacks

    def _compile_model(self, model: keras.Model):
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            model: The model to compile
        """
        raise NotImplementedError(
            "Subclasses must implement _compile_model method"
        )

    def _train_model(self, model: keras.Model, callbacks: List[keras.callbacks.Callback]):
        """
        Train the model with the prepared datasets.

        Args:
            model: The model to train
            callbacks: List of callbacks to use during training

        Returns:
            Training history
        """
        raise NotImplementedError(
            "Subclasses must implement _train_model method"
        )

    def train(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Train a segmentation model and return training results.

        This is the main method that orchestrates the entire training process.

        Args:
            seed: Random seed for shuffling (for reproducibility)

        Returns:
            Dictionary with training results and model information
        """
        # Start timing
        start_time = time.time()

        # Set seed if provided for reproducibility
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
            module_logger.info(f"Random seed set to {seed} for reproducibility")

        # Prepare data with shuffling for training
        self._prepare_data(seed=seed)

        # Create loss function
        self._create_loss_function()

        # Create model and compile within strategy scope for multi-GPU
        if self.strategy and self.num_gpus > 1:
            with self.strategy.scope():
                # Create model
                model = self._create_model()
                
                # Create callbacks
                callbacks = self._create_callbacks(model)
                
                # Compile model
                self._compile_model(model)
                
                # Log multi-GPU setup
                if self.logger:
                    self.logger.logger.info(f"Multi-GPU training with {self.num_gpus} GPUs using MirroredStrategy")
                else:
                    module_logger.info(f"Multi-GPU training with {self.num_gpus} GPUs using MirroredStrategy")
        else:
            # Single GPU or CPU training
            # Create model
            model = self._create_model()
            
            # Create callbacks
            callbacks = self._create_callbacks(model)
            
            # Compile model
            self._compile_model(model)
            
            # Log single GPU/CPU setup
            if self.logger:
                self.logger.logger.info(f"Single GPU/CPU training")
            else:
                module_logger.info(f"Single GPU/CPU training")

        # Train model
        history = self._train_model(model, callbacks)

        # Save model and training history
        model_name = f"{self.model_type}.keras"
        model.save(os.path.join(self.model_path, model_name))

        # Calculate training time
        training_time = time.time() - start_time
        hours, rem = divmod(training_time, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.logger:
            self.logger.logger.info(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        else:
            module_logger.info(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        # Save metadata
        metadata_update = {
            'history': history.history,
            'training_completed': True,
            'training_time': training_time,
            'num_gpus_used': self.num_gpus,
            'l2_regularization_weight': self.l2_regularization_weight,
            'use_adamw_optimizer': self.use_adamw_optimizer
        }
        save_model_metadata(self.model_path, metadata_update)

        # Save summary if logger is available
        if self.logger:
            self.logger.save_debug_summary()

        return {
            'model': model,
            'history': history.history,
            'training_time': training_time,
            'num_gpus_used': self.num_gpus
        }


class DeepLabV3PlusTrainer(SegmentationModelTrainer):
    """
    Trainer for DeepLabV3+ segmentation model.

    This class implements DeepLabV3+ specific training logic.
    """

    def _create_model(self) -> keras.Model:
        """
        Create a DeepLabV3+ model.

        Returns:
            DeepLabV3+ model instance
        """
        model = model_call(
            name="DeepLabV3_plus",
            IMAGE_SIZE=self.image_size,
            NUM_CLASSES=self.num_classes,
            l2_regularization_weight=self.l2_regularization_weight
        )

        if self.logger and self.l2_regularization_weight > 0:
            # Count regularized layers
            reg_layers = sum(1 for l in model.layers if hasattr(l, 'kernel_regularizer') and l.kernel_regularizer)
            self.logger.logger.info(f"Created DeepLabV3+ model with {reg_layers} L2-regularized layers")

        return model

    def _compile_model(self, model: keras.Model):
        """
        Compile the DeepLabV3+ model.

        Args:
            model: The model to compile
        """
        # Use AdamW for weight decay or regular Adam with kernel regularizers
        if self.use_adamw_optimizer:
            # Check for Metal device (AdamW has issues on Apple Silicon)
            # Use platform detection as Metal devices don't contain "metal" in their string representation
            is_metal = platform.machine() == 'arm64' and platform.system() == 'Darwin'

            if is_metal:
                module_logger.warning("AdamW optimizer not fully supported on Metal devices, falling back to Adam")
                optimizer = keras.optimizers.Adam(
                    learning_rate=0.0005,
                    epsilon=ModelDefaults.OPTIMIZER_EPSILON
                )
            else:
                # AdamW provides weight decay (different from L2 regularization)
                # weight_decay is typically similar to L2 regularization weight
                optimizer = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=0.0005,
                    weight_decay=self.l2_regularization_weight,
                    epsilon=ModelDefaults.OPTIMIZER_EPSILON
                )
        else:
            # Regular Adam optimizer (L2 regularization handled by kernel_regularizer)
            optimizer = keras.optimizers.Adam(
                learning_rate=0.0005,
                epsilon=ModelDefaults.OPTIMIZER_EPSILON
            )

        model.compile(
            optimizer=optimizer,
            loss=self.loss_function,
            metrics=["accuracy"],
        )

    def _train_model(self, model: keras.Model, callbacks: List[keras.callbacks.Callback]):
        """
        Train the DeepLabV3+ model.

        Args:
            model: The model to train
            callbacks: List of callbacks to use during training

        Returns:
            Training history
        """
        if self.logger:
            self.logger.logger.info('Starting DeepLabV3+ model training...')
        else:
            module_logger.info('Starting DeepLabV3+ model training...')

        history = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_steps=self.val_steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
            epochs=8  # Train for 8 epochs
        )

        return history


class UNetTrainer(SegmentationModelTrainer):
    """
    Trainer for UNet segmentation model.

    This class implements UNet specific training logic, including
    transfer learning with frozen and unfrozen layers.
    """

    def _create_model(self) -> keras.Model:
        """
        Create a UNet model.

        Returns:
            UNet model instance
        """
        model = model_call(
            name="UNet",
            IMAGE_SIZE=self.image_size,
            NUM_CLASSES=self.num_classes,
            l2_regularization_weight=self.l2_regularization_weight
        )

        if self.logger and self.l2_regularization_weight > 0:
            # Count regularized layers
            reg_layers = sum(1 for l in model.layers if hasattr(l, 'kernel_regularizer') and l.kernel_regularizer)
            self.logger.logger.info(f"Created UNet model with {reg_layers} L2-regularized layers")

        return model

    def _compile_model(self, model: keras.Model, learning_rate: float = 0.001):
        """
        Compile the UNet model.

        Args:
            model: The model to compile
            learning_rate: Learning rate for the optimizer
        """
        # Use AdamW for weight decay or regular Adam with kernel regularizers
        if self.use_adamw_optimizer:
            # Check for Metal device (AdamW has issues on Apple Silicon)
            # Use platform detection as Metal devices don't contain "metal" in their string representation
            is_metal = platform.machine() == 'arm64' and platform.system() == 'Darwin'

            if is_metal:
                module_logger.warning("AdamW optimizer not fully supported on Metal devices, falling back to Adam")
                optimizer = keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    epsilon=ModelDefaults.OPTIMIZER_EPSILON
                )
            else:
                # AdamW provides weight decay (different from L2 regularization)
                optimizer = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=self.l2_regularization_weight,
                    epsilon=ModelDefaults.OPTIMIZER_EPSILON
                )
        else:
            # Regular Adam optimizer (L2 regularization handled by kernel_regularizer)
            optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate,
                epsilon=ModelDefaults.OPTIMIZER_EPSILON
            )

        model.compile(
            optimizer=optimizer,
            loss=self.loss_function,
            metrics=["accuracy"]
        )

    def _train_model(self, model: keras.Model, callbacks: List[keras.callbacks.Callback]):
        """
        Train the UNet model using transfer learning.

        This method first trains with frozen encoder layers, then unfreezes
        them for fine-tuning.

        Args:
            model: The model to train
            callbacks: List of callbacks to use during training

        Returns:
            Training history from the final phase
        """
        if self.logger:
            self.logger.logger.info('Starting UNet model training...')
            self.logger.logger.info('Phase 1: Training with frozen encoder layers...')
        else:
            module_logger.info('Starting UNet model training...')
            module_logger.info('Phase 1: Training with frozen encoder layers...')

        # Initial training phase with frozen encoder
        initial_epochs = 5

        history_initial = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=initial_epochs,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_steps=self.val_steps_per_epoch,
            callbacks=callbacks,
            verbose=1
        )

        # Fine-tuning phase with unfrozen encoder
        if self.logger:
            self.logger.logger.info('Phase 2: Fine-tuning with unfrozen encoder layers...')
        else:
            module_logger.info('Phase 2: Fine-tuning with unfrozen encoder layers...')
        model = unfreeze_model(model)

        # Recompile with lower learning rate
        # Note: No need to wrap in strategy scope as model was already created within scope
        self._compile_model(model, learning_rate=0.0001)

        # Continue training from where we left off
        history_fine_tuning = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=10,  # Train for additional epochs
            initial_epoch=initial_epochs,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_steps=self.val_steps_per_epoch,
            callbacks=callbacks,
            verbose=1
        )

        # Save initial phase history for logging
        self.model_data['initial_history'] = history_initial.history

        return history_fine_tuning


def train_segmentation_model_cnns(pthDL: str, retrain_model: bool = False) -> None:
    """
    Train a segmentation model with the specified configuration.

    This function is the main entry point for model training. It loads model
    configuration, creates the appropriate trainer, and trains the model.

    The function automatically detects the framework (TensorFlow or PyTorch) from
    the CODAVISION_FRAMEWORK environment variable or base/config.py defaults.

    Args:
        pthDL: Path to the directory containing model data
        retrain_model: Whether to retrain an existing model

    Returns:
        None

    Note:
        - For PyTorch training, use PyTorchDeepLabV3PlusTrainer (in training_pytorch.py)
        - For TensorFlow training, use DeepLabV3PlusTrainer or UNetTrainer
        - PyTorch models use PyTorchKerasAdapter for Keras-compatible API
        - TensorFlow callbacks (BatchAccuracyCallback) are not compatible with PyTorch
    """
    # Load model type from configuration
    try:
        with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
            data = pickle.load(f)
            model_type = data.get('model_type', 'DeepLabV3_plus')
            model_name = data.get('nm', 'unknown_model')

        # Handle '+' in model_type
        if '+' in model_type:
            model_type = model_type.replace('+', '_plus')

        if model_type not in ["DeepLabV3_plus", "UNet"]:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types are 'DeepLabV3_plus' and 'UNet'"
            )
    except Exception as e:
        raise ValueError(f"Failed to load model configuration: {e}")

    # Determine framework
    from base.config import get_framework_config
    framework_config = get_framework_config()
    framework = framework_config['framework']

    # Check if model already exists and should not be retrained
    from base.models.utils import get_model_paths
    model_paths = get_model_paths(pthDL, model_type, framework=framework)
    if os.path.isfile(model_paths['best_model']) and not retrain_model:
        module_logger.info(f'Model already trained with name {model_name}. Use retrain_model=True to retrain.')
        return

    # Create and use the appropriate trainer based on framework
    try:
        if framework == 'pytorch':
            # Use PyTorch trainer
            if model_type == "DeepLabV3_plus":
                from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer
                trainer = PyTorchDeepLabV3PlusTrainer(pthDL)
            else:
                raise ValueError(
                    f"PyTorch trainer for {model_type} not yet implemented. "
                    f"Available PyTorch models: DeepLabV3_plus"
                )
        else:
            # Use TensorFlow trainer
            if model_type == "DeepLabV3_plus":
                trainer = DeepLabV3PlusTrainer(pthDL)
            else:  # model_type == "UNet"
                trainer = UNetTrainer(pthDL)

        # Train the model
        trainer.train()

        module_logger.info(f"Successfully trained {model_type} model with {framework} framework.")

    except Exception as e:
        module_logger.error(f"Error during model training: {e}")
        module_logger.error(f"Traceback:\n{traceback.format_exc()}")