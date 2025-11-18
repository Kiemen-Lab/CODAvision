"""
PyTorch Data Loaders for CODAvision

This module provides PyTorch Dataset and DataLoader utilities for loading
training and validation data for segmentation models. It matches the behavior
of the TensorFlow data pipeline while using PyTorch's native data loading.
"""

from typing import List, Optional, Tuple, Callable, Dict, Any
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as TF

# Set up logging
import logging
logger = logging.getLogger(__name__)


class PyTorchSegmentationDataset(Dataset):
    """
    PyTorch Dataset for segmentation tasks.

    This dataset loads image and mask pairs from disk, applies optional augmentation,
    and returns tensors in PyTorch format (NCHW for images, NHW for masks).

    The dataset is designed to match the behavior of the TensorFlow data pipeline
    in base/data/loaders.py, ensuring compatibility and reproducibility.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_size: int,
        augment: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize the PyTorch segmentation dataset.

        Args:
            image_paths: List of paths to image files (PNG or TIFF)
            mask_paths: List of paths to corresponding mask files
            image_size: Size to resize images to (assumes square images)
            augment: Whether to apply data augmentation (rotation, flip, color jitter)
            seed: Random seed for reproducible augmentation (if None, uses random augmentation)

        Raises:
            ValueError: If image_paths and mask_paths have different lengths
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Number of images ({len(image_paths)}) must match "
                f"number of masks ({len(mask_paths)})"
            )

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment
        self.seed = seed

        # Set random seed if provided for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        logger.debug(
            f"Initialized PyTorchSegmentationDataset with {len(image_paths)} samples, "
            f"image_size={image_size}, augment={augment}, seed={seed}"
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a single image-mask pair.

        Args:
            idx: Index of the sample to load

        Returns:
            Tuple of (image, mask) where:
                - image: Float32 tensor of shape (3, H, W) in range [0, 255]
                - mask: Int64 tensor of shape (H, W) with class labels
        """
        # Set deterministic seed for this sample if seed is provided
        # This ensures reproducibility: same idx with same base seed = same augmentation
        if self.seed is not None and self.augment:
            sample_seed = self.seed + idx
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            random.seed(sample_seed)

        # Load image and mask using PIL (matches TensorFlow loader)
        image = self._load_image(self.image_paths[idx], is_mask=False)
        mask = self._load_image(self.mask_paths[idx], is_mask=True)

        # Apply augmentation if enabled
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # Convert to tensors
        # Image: (H, W, C) -> (C, H, W), float32, range [0, 255]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Mask: (H, W, 1) -> (H, W), int64
        mask = torch.from_numpy(mask).squeeze(-1).long()

        return image, mask

    def _load_image(self, path: str, is_mask: bool) -> np.ndarray:
        """
        Load an image or mask from disk using PIL.

        This function matches the behavior of _load_image_pil_internal in
        base/data/loaders.py for compatibility with the TensorFlow pipeline.

        Args:
            path: Path to the image file
            is_mask: Whether this is a segmentation mask (single channel)

        Returns:
            Numpy array of shape (H, W, C) for images or (H, W, 1) for masks
        """
        try:
            # Load image with PIL (supports PNG, TIFF, JPEG, etc.)
            pil_image = Image.open(path)
            image_array = np.array(pil_image)

            if is_mask:
                # Ensure single channel for masks
                if len(image_array.shape) == 3:
                    # Multi-channel image, take first channel
                    image_array = image_array[:, :, 0]

                # Add channel dimension if needed
                if len(image_array.shape) == 2:
                    image_array = np.expand_dims(image_array, -1)

                # Resize mask using nearest neighbor (preserve integer labels)
                if image_array.shape[0] != self.image_size or image_array.shape[1] != self.image_size:
                    image_pil = Image.fromarray(image_array.squeeze())
                    image_pil = image_pil.resize(
                        (self.image_size, self.image_size),
                        Image.NEAREST  # Use nearest neighbor for masks
                    )
                    image_array = np.array(image_pil)
                    if len(image_array.shape) == 2:
                        image_array = np.expand_dims(image_array, -1)

                return image_array.astype(np.float32)

            else:
                # Ensure 3 channels for RGB images
                if len(image_array.shape) == 2:
                    # Grayscale, convert to RGB
                    image_array = np.stack([image_array] * 3, axis=-1)
                elif len(image_array.shape) == 3 and image_array.shape[-1] == 1:
                    # Single channel, convert to RGB
                    image_array = np.repeat(image_array, 3, axis=-1)
                elif len(image_array.shape) == 3 and image_array.shape[-1] > 3:
                    # More than 3 channels, take first 3
                    image_array = image_array[:, :, :3]

                # Resize image using bilinear interpolation
                if image_array.shape[0] != self.image_size or image_array.shape[1] != self.image_size:
                    image_pil = Image.fromarray(image_array.astype(np.uint8))
                    image_pil = image_pil.resize(
                        (self.image_size, self.image_size),
                        Image.BILINEAR
                    )
                    image_array = np.array(image_pil)

                return image_array.astype(np.float32)

        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            raise

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to image and mask.

        Augmentations include:
        - Random rotation (0°, 90°, 180°, 270°)
        - Random horizontal flip
        - Random vertical flip
        - Color jitter (brightness, contrast, saturation) - image only

        Both image and mask receive the same geometric transformations to maintain
        spatial correspondence.

        Args:
            image: Image array of shape (H, W, C)
            mask: Mask array of shape (H, W, 1)

        Returns:
            Tuple of augmented (image, mask)
        """
        # Convert to PIL for augmentation
        image_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(mask.squeeze(-1).astype(np.uint8))

        # Random rotation (0°, 90°, 180°, 270°)
        if random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image_pil = TF.rotate(image_pil, angle)
                mask_pil = TF.rotate(mask_pil, angle)

        # Random horizontal flip
        if random.random() > 0.5:
            image_pil = TF.hflip(image_pil)
            mask_pil = TF.hflip(mask_pil)

        # Random vertical flip
        if random.random() > 0.5:
            image_pil = TF.vflip(image_pil)
            mask_pil = TF.vflip(mask_pil)

        # Color jitter (apply to image only, not mask)
        if random.random() > 0.5:
            color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )
            image_pil = color_jitter(image_pil)

        # Convert back to numpy
        image_aug = np.array(image_pil).astype(np.float32)
        mask_aug = np.array(mask_pil).astype(np.float32)

        # Restore channel dimensions
        if len(mask_aug.shape) == 2:
            mask_aug = np.expand_dims(mask_aug, -1)

        return image_aug, mask_aug


def _worker_init_fn_with_seed(worker_id: int, base_seed: int):
    """
    Initialize worker with unique seed for reproducibility.

    This function is defined at module level so it can be pickled for
    multi-worker data loading.

    Args:
        worker_id: Worker ID assigned by DataLoader
        base_seed: Base seed to derive worker-specific seed from
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_pytorch_dataloader(
    image_paths: List[str],
    mask_paths: List[str],
    image_size: int,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: Optional[int] = None,
    drop_last: bool = True
) -> DataLoader:
    """
    Create a PyTorch DataLoader for segmentation tasks.

    This function creates a DataLoader that matches the behavior of the TensorFlow
    create_dataset function in base/data/loaders.py.

    Args:
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files
        image_size: Size to resize images to (assumes square images)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation
        num_workers: Number of worker processes for data loading (0 for main process only)
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducibility (shuffling and augmentation)
        drop_last: Whether to drop the last incomplete batch

    Returns:
        PyTorch DataLoader yielding batches of (images, masks) where:
            - images: Float32 tensor of shape (B, 3, H, W)
            - masks: Int64 tensor of shape (B, H, W)

    Example:
        >>> dataloader = create_pytorch_dataloader(
        ...     image_paths=['path/to/img1.png', 'path/to/img2.png'],
        ...     mask_paths=['path/to/mask1.png', 'path/to/mask2.png'],
        ...     image_size=512,
        ...     batch_size=4,
        ...     shuffle=True,
        ...     augment=True,
        ...     num_workers=4
        ... )
        >>> for images, masks in dataloader:
        ...     # images: (4, 3, 512, 512)
        ...     # masks: (4, 512, 512)
        ...     pass
    """
    # Create dataset
    dataset = PyTorchSegmentationDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        image_size=image_size,
        augment=augment,
        seed=seed
    )

    # Create worker init function with seed if provided
    worker_init_fn = None
    if seed is not None:
        from functools import partial
        worker_init_fn = partial(_worker_init_fn_with_seed, base_seed=seed)

    # Create generator for shuffling if seed is provided
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    logger.info(
        f"Created PyTorch DataLoader with {len(dataset)} samples, "
        f"batch_size={batch_size}, shuffle={shuffle}, augment={augment}, "
        f"num_workers={num_workers}"
    )

    return dataloader


def create_training_dataloader(
    image_paths: List[str],
    mask_paths: List[str],
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
    seed: Optional[int] = None
) -> DataLoader:
    """
    Create an optimized PyTorch DataLoader for training.

    This is a convenience function that creates a DataLoader with settings
    optimized for training:
    - Shuffling enabled
    - Data augmentation enabled
    - Multi-worker data loading
    - Memory pinning for GPU transfer

    Args:
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files
        image_size: Size to resize images to
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility

    Returns:
        PyTorch DataLoader configured for training
    """
    return create_pytorch_dataloader(
        image_paths=image_paths,
        mask_paths=mask_paths,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        num_workers=num_workers,
        pin_memory=True,
        seed=seed,
        drop_last=True
    )


def create_validation_dataloader(
    image_paths: List[str],
    mask_paths: List[str],
    image_size: int,
    batch_size: int,
    num_workers: int = 4
) -> DataLoader:
    """
    Create an optimized PyTorch DataLoader for validation.

    This is a convenience function that creates a DataLoader with settings
    optimized for validation:
    - No shuffling (consistent validation)
    - No data augmentation
    - Multi-worker data loading
    - Memory pinning for GPU transfer

    Args:
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files
        image_size: Size to resize images to
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading

    Returns:
        PyTorch DataLoader configured for validation
    """
    return create_pytorch_dataloader(
        image_paths=image_paths,
        mask_paths=mask_paths,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        num_workers=num_workers,
        pin_memory=True,
        seed=None,
        drop_last=True
    )


def get_tile_paths_from_directory(
    model_path: str,
    tile_folder: str = 'training'
) -> Tuple[List[str], List[str]]:
    """
    Get lists of image and mask paths from a tile directory.

    This utility function scans the tile directory structure created by
    create_training_tiles and returns matched lists of image and mask paths.

    Expected directory structure:
        model_path/
            tile_folder/
                im/          # Images
                    img_0001.png
                    img_0002.png
                    ...
                label/       # Masks
                    img_0001.png
                    img_0002.png
                    ...

    Args:
        model_path: Path to the model directory
        tile_folder: Name of the tile folder (e.g., 'training', 'validation')

    Returns:
        Tuple of (image_paths, mask_paths) sorted by filename

    Raises:
        FileNotFoundError: If the tile directories don't exist
        ValueError: If no matching image-mask pairs are found
    """
    import glob

    # Construct paths
    images_dir = os.path.join(model_path, tile_folder, 'im')
    masks_dir = os.path.join(model_path, tile_folder, 'label')

    # Check if directories exist
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")

    # Get all image files (support both PNG and TIFF)
    image_extensions = ['*.png', '*.PNG', '*.tif', '*.tiff', '*.TIF', '*.TIFF']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    # Get corresponding mask paths
    mask_paths = []
    for img_path in image_paths:
        # Get base filename
        img_filename = os.path.basename(img_path)

        # Try to find matching mask (try all extensions)
        mask_found = False
        for ext in image_extensions:
            # Replace extension
            base_name = os.path.splitext(img_filename)[0]
            mask_pattern = ext.replace('*', base_name)
            mask_path = os.path.join(masks_dir, mask_pattern)

            if os.path.exists(mask_path):
                mask_paths.append(mask_path)
                mask_found = True
                break

        if not mask_found:
            raise ValueError(f"No matching mask found for image: {img_path}")

    logger.info(
        f"Found {len(image_paths)} image-mask pairs in "
        f"{os.path.join(model_path, tile_folder)}"
    )

    return image_paths, mask_paths
