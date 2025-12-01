"""
Training Tile Creation Utilities for CODAvision

This module provides functions for creating and managing training tiles from annotations.
It handles the creation of training and validation tiles for deep learning model training.
"""

import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2,40))  # Disable pixel limit safeguard
import glob
import shutil
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2

from base.image.augmentation import edit_annotation_tiles
from base.image.utils import load_image_with_fallback
import gc

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Constants for loop safeguards
MAX_ITERATIONS = 10000
MAX_CONSECUTIVE_FAILURES = 10


def validate_image_list_structure(
    image_list: Dict[str, List[str]],
    model_path: str
) -> None:
    """
    Validate that image_list has the correct directory structure.

    Checks that:
    1. Image files exist at the specified paths
    2. Corresponding label directory exists alongside im/ directory
    3. Corresponding label files exist for each image

    Args:
        image_list: Dictionary with 'tile_name' and 'tile_pth' lists
        model_path: Path to the model directory (for error messages)

    Raises:
        FileNotFoundError: If required files or directories are missing
        ValueError: If image_list structure is invalid

    The expected directory structure is:
        parent_dir/
            im/          <- Images directory (pointed to by tile_pth)
                image.png
            label/       <- Annotation masks directory
                image.png
    """
    if 'tile_name' not in image_list or 'tile_pth' not in image_list:
        raise ValueError(
            "image_list must contain 'tile_name' and 'tile_pth' keys"
        )

    if len(image_list['tile_name']) != len(image_list['tile_pth']):
        raise ValueError(
            f"image_list 'tile_name' and 'tile_pth' must have same length "
            f"(got {len(image_list['tile_name'])} names and {len(image_list['tile_pth'])} paths)"
        )

    if len(image_list['tile_name']) == 0:
        raise ValueError("image_list must contain at least one tile")

    for i, (tile_name, tile_pth) in enumerate(zip(image_list['tile_name'], image_list['tile_pth'])):
        # Check image exists
        image_path = os.path.join(tile_pth, tile_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found: {image_path}\n"
                f"Check that image_list['tile_pth'][{i}] points to the correct directory."
            )

        # Check label directory exists
        # Expected structure: parent_dir/im/ and parent_dir/label/
        parent_dir = os.path.dirname(tile_pth)
        label_dir = os.path.join(parent_dir, 'label')

        if not os.path.exists(label_dir):
            raise FileNotFoundError(
                f"Label directory not found: {label_dir}\n"
                f"Expected directory structure:\n"
                f"  {parent_dir}/\n"
                f"    im/          <- Images directory\n"
                f"      {tile_name}\n"
                f"    label/       <- Annotation masks directory\n"
                f"      {tile_name}\n"
                f"\n"
                f"Current structure has 'im/' directory but missing 'label/' directory.\n"
                f"Please create the label directory and add annotation masks."
            )

        # Check label file exists
        label_path = os.path.join(label_dir, tile_name)
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Label file not found: {label_path}\n"
                f"Image exists at: {image_path}\n"
                f"But corresponding annotation mask is missing.\n"
                f"Please create annotation mask for this image in the label/ directory."
            )

    logger.info(f"[OK] Validated image_list structure: {len(image_list['tile_name'])} tiles with corresponding labels")


def validate_tile_config_compatibility(
    image_shape: Tuple[int, int],
    config: 'TileGenerationConfig',
    model_path: str
) -> None:
    """
    Validate that tile generation config is compatible with image size.

    This function checks whether the combination of image size, reduction factor,
    padding, and disk filter size will result in a valid distance transform for
    tile placement. If the configuration is incompatible, it raises a ValueError
    with clear explanation and actionable solutions.

    Args:
        image_shape: Tuple of (height, width) for the composite image
        config: Tile generation configuration to validate
        model_path: Path to the model directory (for error messages)

    Raises:
        ValueError: If configuration is incompatible with image size, with
                   detailed explanation and suggested solutions

    Example:
        >>> validate_tile_config_compatibility(
        ...     (512, 512),
        ...     LEGACY_CONFIG,
        ...     "path/to/model"
        ... )
        ValueError: Image size (512x512) is too small for tile generation config...
    """
    height, width = image_shape
    min_dim = min(height, width)

    # Calculate effective size after downsampling and padding
    downsampled_size = min_dim / config.reduction_factor
    padding = int(100 / config.reduction_factor)
    effective_size = downsampled_size - 2 * padding

    # Account for disk filter if enabled
    disk_filter_size = 51 if config.use_disk_filter else 0
    if config.use_disk_filter:
        effective_size -= disk_filter_size

    # Minimum effective size needed for valid distance transforms
    MIN_EFFECTIVE_SIZE = 50

    if effective_size < MIN_EFFECTIVE_SIZE:
        # Calculate required minimum size
        required_effective = MIN_EFFECTIVE_SIZE + disk_filter_size + 2 * padding
        required_size = int(required_effective * config.reduction_factor)

        # Calculate what reduction factor would work for this image
        max_working_reduction = max(1, min_dim // (MIN_EFFECTIVE_SIZE + disk_filter_size + 2 * padding))

        error_msg = (
            f"Image size ({min_dim}x{min_dim}) is incompatible with tile generation config.\n"
            f"\n"
            f"Details:\n"
            f"  Configuration mode: {config.mode}\n"
            f"  Reduction factor: {config.reduction_factor}\n"
            f"  Use disk filter: {config.use_disk_filter}\n"
            f"  Disk filter size: {disk_filter_size if config.use_disk_filter else 'N/A'}\n"
            f"  Padding: {padding}\n"
            f"  Effective size after processing: {effective_size:.1f}x{effective_size:.1f}\n"
            f"  Minimum required effective size: {MIN_EFFECTIVE_SIZE}x{MIN_EFFECTIVE_SIZE}\n"
            f"\n"
            f"Solutions (choose one):\n"
            f"  1. Use larger images (minimum {required_size}x{required_size} pixels)\n"
            f"  2. Use MODERN_CONFIG (reduction_factor=10, no disk filter)\n"
            f"  3. Reduce reduction_factor to {max_working_reduction} or lower\n"
            f"  4. Disable disk filter (set use_disk_filter=False)\n"
            f"\n"
            f"For testing with small images, use a scaled configuration.\n"
            f"Example: TileGenerationConfig(reduction_factor=2, use_disk_filter=False, big_tile_size=1024)"
        )

        raise ValueError(error_msg)

    # Log successful validation
    logger.info(
        f"[OK] Validated tile config compatibility: {config.mode} mode with "
        f"{min_dim}x{min_dim} images (effective size: {effective_size:.1f}x{effective_size:.1f})"
    )


def combine_annotations_into_tiles(
    initial_annotations: np.ndarray,
    current_annotations: np.ndarray,
    annotation_percentages: np.ndarray,
    image_list: Dict[str, List[str]],
    num_classes: int,
    model_path: str,
    output_folder: str,
    tile_size: int,
    config: 'TileGenerationConfig',
    background_class: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine annotations into large tiles for training deep neural networks.

    This function creates composite training tiles by selecting and combining
    annotation regions from the provided data. It handles distribution of
    classes to ensure balanced representation in the training data.

    Args:
        initial_annotations: Initial array containing the number of pixels per class per bounding box.
                           Each row is a bounding box and each column is a class.
        current_annotations: Current state of annotation counts (can be modified during processing)
        annotation_percentages: Percentages of annotations used (tracking usage for balanced sampling)
        image_list: Dictionary with lists of image tile paths and names
        num_classes: Number of annotation classes including background
        model_path: Path to the model directory
        output_folder: Output folder name where tiles will be saved (relative to model_path)
        tile_size: Size for the output tiles
        config: Tile generation configuration controlling algorithm behavior
        background_class: Background class index (default: 0)

    Returns:
        Tuple containing:
        - Updated annotation counts array
        - Updated annotation percentage tracking array
    """

    # Validate input structure (fail-fast if directory structure is wrong)
    validate_image_list_structure(image_list, model_path)

    logger.debug(f"Starting combine_annotations_into_tiles with {len(image_list['tile_name'])} tiles")
    logger.debug(f"initial_annotations shape: {initial_annotations.shape}")
    logger.debug(f"current_annotations shape: {current_annotations.shape}")
    logger.debug(f"num_classes: {num_classes}")

    # Use configured big tile size
    big_tile_size = config.big_tile_size
    big_tile_size_with_margin = big_tile_size + 200
    keep_all_classes = 1

    # Create output directories
    output_path_images = os.path.join(model_path, output_folder, 'im')
    output_path_labels = os.path.join(model_path, output_folder, 'label')
    output_path_big_tiles = os.path.join(model_path, output_folder, 'big_tiles')

    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_labels, exist_ok=True)
    os.makedirs(output_path_big_tiles, exist_ok=True)

    # Check if we've already done this work using configured file format
    file_ext = f".{config.file_format}"
    existing_images = [f for f in os.listdir(output_path_images) if f.endswith(file_ext)]
    next_image_number = len(existing_images) + 1

    # Initialize composite canvas
    composite_image = np.full((big_tile_size_with_margin, big_tile_size_with_margin, 3),
                             background_class, dtype=np.float64)
    composite_mask = np.zeros((big_tile_size_with_margin, big_tile_size_with_margin), dtype=np.uint8)
    total_pixels = composite_mask.size

    # Validate that config is compatible with image size (fail-fast before processing)
    validate_tile_config_compatibility(composite_mask.shape, config, model_path)

    # Track class balancing
    class_counts = np.zeros(current_annotations.shape[1])
    logger.debug(f"Initial class_counts shape: {class_counts.shape}")
    fill_ratio = np.sum(class_counts) / total_pixels

    # Iteration control variables
    count = 1
    tile_count = 1
    cutoff_threshold = 0.55
    reduction_factor = config.reduction_factor  # Use configured reduction factor
    last_class_type = 0
    num_tiles_used = np.zeros(len(image_list['tile_name']))
    class_type_counts = np.zeros(num_classes)

    # Pre-compute constant values used in distance transform calculation
    # These values never change across iterations, so computing once saves significant time
    padding = int(100/reduction_factor)

    # Create disk filter if enabled (used for convolution in placement optimization)
    # This 51x51 disk filter with radius 25 is constant throughout all iterations
    h = None
    if config.use_disk_filter:
        disk_radius = 25
        disk_size = 51
        h = np.zeros((disk_size, disk_size))
        center = disk_size // 2
        y_disk, x_disk = np.ogrid[:disk_size, :disk_size]
        disk_mask = (x_disk - center)**2 + (y_disk - center)**2 <= center**2
        h[disk_mask] = 1.0

    # Track whether we've warned about empty distance transforms (warn once only)
    empty_distance_transform_warned = False

    # Main loop with safeguards
    iteration = 1
    consecutive_failures = 0

    while fill_ratio < cutoff_threshold and iteration < MAX_ITERATIONS:
        iteration_start_time = time.time()

        # Select which class to sample using configured rotation frequency
        if count % config.class_rotation_frequency == 1:
            class_type = tile_count - 1
            tile_count = (tile_count % num_classes) + 1
        else:
            tmp = class_counts.copy()
            # Check that last_class_type is within bounds before using as index
            if last_class_type < tmp.shape[0]:
                tmp[last_class_type] = np.max(tmp)
            class_type = np.argmin(tmp)

        # Validate class_type is in bounds
        class_type_counts[class_type if class_type < len(class_type_counts) else len(class_type_counts) - 1] += 1

        # Find candidate tiles
        if class_type < current_annotations.shape[1]:
            candidate_tiles = np.where(current_annotations[:, class_type] > 0)[0]
        else:
            logger.debug(f"Warning: class_type {class_type} is out of bounds for current_annotations with shape {current_annotations.shape}")
            candidate_tiles = np.array([], dtype=int)

        # Try to reset candidate tiles if empty
        if len(candidate_tiles) == 0:
            if class_type < current_annotations.shape[1]:
                current_annotations[:, class_type] = initial_annotations[:, class_type]
                candidate_tiles = np.where(current_annotations[:, class_type] > 0)[0]
            else:
                candidate_tiles = np.array([], dtype=int)

        logger.debug(f"Processing tile {count}, selecting class_type {class_type}")
        logger.debug(f"Number of candidate tiles: {len(candidate_tiles)}")

        # Skip iteration if still no candidate tiles
        if len(candidate_tiles) == 0:
            logger.debug(f"  Skipping class_type {class_type} as no candidate tiles were found")
            count += 1
            last_class_type = min(class_type, current_annotations.shape[1] - 1)  # Keep last_class_type in bounds
            iteration += 1
            continue

        # Select a random tile containing our target class
        selected_tile_idx = np.random.choice(candidate_tiles, size=1, replace=False)
        num_tiles_used[selected_tile_idx[0]] += 1

        # Load the tile image and mask
        tile_name = image_list['tile_name'][selected_tile_idx[0]]
        tile_path = os.path.join(image_list['tile_pth'][selected_tile_idx[0]], tile_name)

        # Find corresponding label path
        path_separator_pos = tile_path.rfind(os.path.sep, 0, tile_path.rfind(os.path.sep))
        label_path = os.path.join(tile_path[0:path_separator_pos], 'label')

        # Load image and annotation mask
        try:
            image = load_image_with_fallback(tile_path, 'RGB')
            annotation_mask = load_image_with_fallback(os.path.join(label_path, tile_name), 'L')
            # Reset consecutive failures on successful load
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            logger.error(
                f"Failed to load tile {tile_path}: {e}\n"
                f"  Consecutive failures: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}"
            )

            # Fail fast if too many consecutive failures
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"Failed to load tiles {consecutive_failures} times consecutively.\n"
                    f"Last attempted path: {tile_path}\n"
                    f"Label path: {os.path.join(label_path, tile_name)}\n"
                    f"\n"
                    f"This likely indicates a problem with:\n"
                    f"  1. File paths in image_list are incorrect\n"
                    f"  2. Image or label files are missing or corrupted\n"
                    f"  3. File permissions prevent reading\n"
                    f"\n"
                    f"Check that all files exist and are readable."
                ) from e

            count += 1
            iteration += 1
            continue

        # Apply optional augmentation using configured frequency
        apply_augmentation = 1 if count % config.class_rotation_frequency == 1 else 0

        # Get augmented tiles with configured rotation cropping behavior
        image, annotation_mask, kept_classes = edit_annotation_tiles(
            image, annotation_mask, apply_augmentation, class_type, class_counts,
            composite_mask.shape[0], keep_all_classes, config.crop_rotations
        )

        # Update tracking of which annotations we've used
        if isinstance(kept_classes, np.ndarray):
            # Handle array case
            for kp_val in kept_classes:
                current_annotations[selected_tile_idx[0], kp_val - 1] = 0
        else:
            # Handle scalar case
            current_annotations[selected_tile_idx[0], kept_classes - 1] = 0
        annotation_percentages[selected_tile_idx[0], kept_classes - 1, 0] += 1
        annotation_percentages[selected_tile_idx[0], kept_classes - 1, 1] = 2

        # Check if we have enough annotation pixels to use
        valid_pixels = (annotation_mask != 0)
        logger.debug(f"After edit_annotation_tiles - kept_classes: {kept_classes}")
        logger.debug(f"Valid pixels in mask: {np.sum(valid_pixels)}")
        if np.sum(valid_pixels) < 30:
            logger.info('  Skipped tile with too few valid pixels')
            continue

        # Find optimal location to place this tile in the composite
        # Downsample for faster distance calculation
        downsampled_mask = composite_mask[::reduction_factor, ::reduction_factor] > 0

        # Apply disk filter if enabled (MATLAB's imfilter)
        # Note: disk filter 'h' and 'padding' are pre-computed outside the loop for performance
        if config.use_disk_filter:
            # Using OpenCV's filter2D for optimal performance (2-4x faster than scipy.ndimage.convolve)
            filtered_mask = cv2.filter2D(downsampled_mask.astype(np.float32), -1, h, borderType=cv2.BORDER_CONSTANT)
        else:
            # Skip disk filter - use downsampled mask directly (modern/CODAvision approach)
            filtered_mask = downsampled_mask.astype(np.float32)

        # Calculate distance transform to find largest empty area (using filtered/downsampled mask)
        dist = cv2.distanceTransform((filtered_mask <= 0).astype(np.uint8), cv2.DIST_L2, 3)
        # Add border padding
        dist[:padding, :] = 0
        dist[:, :padding] = 0
        dist[-padding:, :] = 0
        dist[:, -padding:] = 0

        # Validation: Check for pathological edge case where distance transform is empty
        # This can happen when image is too small for the disk filter size and padding
        max_dist = np.max(dist)
        if max_dist == 0 or dist.size == 0:
            # Warn only once to avoid flooding logs with hundreds of warnings
            if not empty_distance_transform_warned:
                disk_size = 51 if config.use_disk_filter else 0  # Hardcoded disk filter size
                logger.warning(
                    f"Distance transform is empty (max_dist={max_dist}, size={dist.shape}). "
                    f"This may occur when image size ({composite_mask.shape}) is too small for "
                    f"reduction_factor={reduction_factor}, padding={padding}, "
                    f"and disk_filter_size={disk_size if config.use_disk_filter else 'N/A'}. "
                    "Using random placement as fallback for this and subsequent placements."
                )
                empty_distance_transform_warned = True

            # Fallback: use random placement to avoid stacking all tiles at the same location
            # This provides better spatial distribution than always using the center
            x = np.random.randint(padding, composite_mask.shape[0] - padding)
            y = np.random.randint(padding, composite_mask.shape[1] - padding)
        else:
            # Find point with maximum distance from any occupied area
            max_dist_points = np.where(dist == max_dist)
            index = np.random.choice(len(max_dist_points[0]), size=1, replace=False)
            x = int(max_dist_points[0][index[0]] * reduction_factor)
            y = int(max_dist_points[1][index[0]] * reduction_factor)

        # Calculate placement position ensuring we stay in bounds
        annotation_size = np.array(annotation_mask.shape) - 1
        half_size_a = annotation_size // 2
        half_size_b = annotation_size - half_size_a

        # Convert numpy arrays to tuples for slicing
        half_size_a = tuple(map(int, half_size_a))
        half_size_b = tuple(map(int, half_size_b))

        # Adjust position if needed to stay in bounds
        if x + half_size_a[0] + 1 > composite_mask.shape[0]:
            x -= half_size_a[0]
        if y + half_size_a[1] + 1 > composite_mask.shape[1]:
            y -= half_size_a[1]
        if x - half_size_b[0] < 0:
            x += half_size_b[0]
        if y - half_size_b[1] < 0:
            y += half_size_b[1]

        # # Create slices for the region where we'll place this tile
        # region_slice_x = slice(x - half_size_b[0], x + half_size_a[0] + 1)
        # region_slice_y = slice(y - half_size_b[1], y + half_size_a[1] + 1)
        #
        # # Place the tile in the composite
        # temp_mask = composite_mask[region_slice_x, region_slice_y].copy()
        # temp_mask[valid_pixels] = annotation_mask[valid_pixels]
        # composite_mask[region_slice_x, region_slice_y] = temp_mask
        #
        # temp_image = composite_image[region_slice_x, region_slice_y, :].copy()
        # valid_pixels_3d = np.dstack((valid_pixels, valid_pixels, valid_pixels))
        # temp_image[valid_pixels_3d] = image[valid_pixels_3d]
        # composite_image[region_slice_x, region_slice_y, :] = temp_image

        # Use direct slicing instead of slice objects
        # Copy the current mask values in the target region
        temp_mask = composite_mask[x - half_size_b[0]:x + half_size_a[0] + 1,
                                   y - half_size_b[1]:y + half_size_a[1] + 1].copy()
        # Apply the new annotation mask to valid pixels
        temp_mask[valid_pixels] = annotation_mask[valid_pixels]
        # Update the composite mask
        composite_mask[x - half_size_b[0]:x + half_size_a[0] + 1,
                       y - half_size_b[1]:y + half_size_a[1] + 1] = temp_mask

        # Do the same for the image
        temp_image = composite_image[x - half_size_b[0]:x + half_size_a[0] + 1,
                                     y - half_size_b[1]:y + half_size_a[1] + 1, :].copy()
        # Apply the new image to valid pixels
        valid_pixels_3d = np.dstack((valid_pixels, valid_pixels, valid_pixels))
        temp_image[valid_pixels_3d] = image[valid_pixels_3d]
        # Update the composite image
        composite_image[x - half_size_b[0]:x + half_size_a[0] + 1,
                        y - half_size_b[1]:y + half_size_a[1] + 1, :] = temp_image

        # Update fill ratio periodically
        if count % 2 == 0:
            fill_ratio = cv2.countNonZero(composite_mask) / total_pixels
            logger.debug(f"Current fill_ratio: {fill_ratio:.4f}, target: {cutoff_threshold:.4f}")

        # Update class counts from newly added content
        for class_idx in range(current_annotations.shape[1]):
            count_before = class_counts[class_idx]
            additional_count = np.sum(temp_mask == class_idx + 1)
            class_counts[class_idx] += additional_count
            logger.debug(f"Class {class_idx + 1}: Added {additional_count} pixels, now {class_counts[class_idx]}")

        # Periodically recompute the actual class distribution
        if count % 150 == 0 or fill_ratio > cutoff_threshold:
            class_histogram = np.histogram(composite_mask, bins=np.arange(current_annotations.shape[1] + 2))[0]
            class_counts = class_histogram[1:]
            # Avoid division by zero
            class_counts[class_counts == 0] = 1

        # Increment iteration counter
        count += 1
        last_class_type = class_type
        iteration += 1

        # Garbage collection every 50 tiles to prevent memory buildup
        if count % 50 == 0:
            gc.collect()
            logger.debug(f"Performed garbage collection after {count} tiles")

        elapsed_time = time.time() - iteration_start_time

    # Check if we exited due to MAX_ITERATIONS
    if iteration >= MAX_ITERATIONS:
        raise RuntimeError(
            f"Exceeded maximum iterations ({MAX_ITERATIONS}) without reaching fill threshold.\n"
            f"Final fill_ratio: {fill_ratio:.4f}, target: {cutoff_threshold:.4f}\n"
            f"\n"
            f"This may indicate:\n"
            f"  1. Insufficient training data (not enough annotated regions)\n"
            f"  2. Annotation files are empty or contain mostly background\n"
            f"  3. Threshold is set too high for available data\n"
            f"\n"
            f"Consider:\n"
            f"  - Checking annotation quality and coverage\n"
            f"  - Reducing cutoff_threshold (current: {cutoff_threshold})\n"
            f"  - Adding more annotated training data"
        )

    # Crop margins from the final composite
    composite_image = composite_image[100:-100, 100:-100, :].astype(np.float64)
    composite_mask = composite_mask[100:-100, 100:-100].astype(np.uint8)

    # Record final class counts
    for class_idx in range(num_classes - 1):
        class_counts[class_idx] = np.sum(composite_mask == class_idx + 1)

    # Adjust labels to 0-based
    composite_mask[composite_mask == 0] = num_classes
    composite_mask = composite_mask - 1

    # Split the big tile into smaller training tiles
    for row in range(0, composite_image.shape[0], tile_size):
        for col in range(0, composite_image.shape[1], tile_size):
            try:
                # Extract tile
                image_tile = composite_image[row:row + tile_size, col:col + tile_size, :]
                mask_tile = composite_mask[row:row + tile_size, col:col + tile_size]

                # Save tiles using configured file format
                Image.fromarray(image_tile.astype(np.uint8)).save(
                    os.path.join(output_path_images, f"{next_image_number}{file_ext}"))
                Image.fromarray(mask_tile.astype(np.uint8)).save(
                    os.path.join(output_path_labels, f"{next_image_number}{file_ext}"))


                next_image_number += 1
            except ValueError:
                # Handle edge case where tile would exceed bounds
                continue

    # Save the big tile for reference
    big_tile_number = len([f for f in os.listdir(output_path_big_tiles) if f.startswith('HE')]) + 1
    logger.info('  Saving big tile')
    # Save big tiles using configured file format
    Image.fromarray(composite_image.astype(np.uint8)).save(
        os.path.join(output_path_big_tiles, f"HE_tile_{big_tile_number}{file_ext}"))
    Image.fromarray(composite_mask).save(
        os.path.join(output_path_big_tiles, f"label_tile_{big_tile_number}{file_ext}"))

    return current_annotations, annotation_percentages


def create_training_tiles(
    model_path: str,
    annotations: np.ndarray,
    image_list: Dict[str, List[str]],
    create_new_tiles: bool,
    config: Optional['TileGenerationConfig'] = None
) -> None:
    """
    Build training and validation tiles using annotation bounding boxes.

    This function creates training and validation tiles by combining annotation regions,
    handling class distributions to ensure balanced representation. The tiles are saved
    in the designated model directory.

    Args:
        model_path: Path to the directory containing model data
        annotations: Array containing annotation pixel counts by class
        image_list: Dictionary with lists of image tile paths and names
        create_new_tiles: Flag indicating whether to create new tiles or use existing ones
        config: Optional tile generation configuration. If None, uses get_default_tile_config()

    Raises:
        ValueError: If no valid annotations are found
    """
    # Import here to avoid circular import
    from base.config import get_default_tile_config, TileGenerationConfig

    # Get configuration if not provided
    if config is None:
        config = get_default_tile_config()

    # Set deterministic seed if specified in config
    if config.deterministic_seed is not None:
        np.random.seed(config.deterministic_seed)

    # Create file pattern for glob operations using configured format
    file_pattern = f"HE*.{config.file_format}"

    # Load model metadata
    with open(os.path.join(model_path, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        tile_size = data['sxy']
        num_classes = data['nblack']
        class_names = data['classNames']
        num_train_tiles = data['ntrain']
        num_validation_tiles = data['nvalidate']

    # Adjust class names if needed
    if class_names[-1] == "black":
        class_names = class_names[:-1]
    logger.info('')

    # Verify annotations exist
    if annotations is None or len(annotations) == 0:
        raise ValueError(
            'No annotation data found. Please ensure that annotation files exist and contain valid annotations.'
        )

    # Calculate total number of pixels in training dataset
    logger.info('Calculating total number of pixels in the training dataset...')
    count_annotations = sum(annotations)

    # Validate annotations
    if isinstance(count_annotations, (int, float)) or (
            hasattr(count_annotations, 'size') and count_annotations.size == 0):
        raise ValueError(
            'No valid annotations were found. This usually happens when no annotation files are present or they contain no valid annotations. '
            'Please check your annotation directory and ensure annotations are correctly formatted.'
        )

    # Check for empty classes
    if max(count_annotations) == 0:
        raise ValueError(
            'All annotation classes have zero pixels. Please check your annotation files and ensure they contain valid annotations.'
        )

    # Report annotation distribution
    annotation_composition = count_annotations / max(count_annotations) * 100
    for b, count in enumerate(annotation_composition):
        if annotation_composition[b] == 100:
            logger.info(f' There are {count_annotations[b]} pixels of {class_names[b]}. This is the most common class.')
        else:
            logger.info(
                f' There are {count_annotations[b]} pixels of {class_names[b]}, {int(annotation_composition[b])}% of the most common class.')

    # Ensure all classes have annotations
    if 0 in count_annotations:
        raise ValueError(
            'There are no annotations for one or more classes. Please add annotations, check nesting, or remove empty classes.'
        )

    # Create training tiles
    logger.info('')
    logger.info('Building training tiles...')
    annotations_array = np.array(annotations)
    current_annotations = annotations_array.copy()
    annotation_percentages = np.double(annotations_array > 0)
    annotation_percentages = np.dstack((annotation_percentages, annotation_percentages))
    annotation_percentages_original = annotation_percentages.copy()

    output_type = 'training'
    if create_new_tiles and os.path.isdir(os.path.join(model_path, output_type)):
        shutil.rmtree(os.path.join(model_path, output_type))

    big_tiles_path = os.path.join(model_path, output_type, 'big_tiles')

    train_start = time.time()
    if len(glob.glob(os.path.join(big_tiles_path, file_pattern))) >= num_train_tiles:
        logger.info('  Already done.')
    else:
        while len(glob.glob(os.path.join(big_tiles_path, file_pattern))) < num_train_tiles:
            current_annotations, annotation_percentages = combine_annotations_into_tiles(
                annotations_array,
                current_annotations,
                annotation_percentages,
                image_list,
                num_classes,
                model_path,
                output_type,
                tile_size,
                config
            )

            logger.debug(f"After combine_annotations_into_tiles - current_annotations shape: {current_annotations.shape}")
            logger.debug(f"After combine_annotations_into_tiles - unique values in annotation_percentages: {np.unique(annotation_percentages)}")

            elapsed_time = time.time() - train_start
            logger.info(
                f'  {len(glob.glob(os.path.join(big_tiles_path, file_pattern)))} of {num_train_tiles} training images completed in {int(elapsed_time / 60)} minutes')

            # Report usage statistics
            base_class_count = np.sum(annotation_percentages_original[:, :, 0], axis=0)
            used_class_count = np.sum(annotation_percentages[:, :, 0], axis=0)
            base_unique_annotations = np.sum(annotation_percentages_original[:, :, 1] == 1, axis=0)
            used_unique_annotations = np.sum(annotation_percentages[:, :, 1] == 2, axis=0)

            percent_count_used = used_class_count / base_class_count * 100
            percent_unique_used = used_unique_annotations / base_unique_annotations * 100

            for b, class_name in enumerate(class_names):
                logger.info(f'  Used {percent_count_used[b]:.1f}% counts and {percent_unique_used[b]:.1f}% unique annotations of {class_name}')

    # Report training tile creation time
    total_time_train_bigtiles = time.time() - train_start
    hours, rem = divmod(total_time_train_bigtiles, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'  Elapsed time to create training big tiles: {int(hours)}h {int(minutes)}m {int(seconds)}s')

    # Create validation tiles
    output_type = 'validation'
    if create_new_tiles and os.path.isdir(os.path.join(model_path, output_type)):
        shutil.rmtree(os.path.join(model_path, output_type))

    big_tiles_path = os.path.join(model_path, output_type, 'big_tiles')
    current_annotations = annotations_array.copy()
    annotation_percentages = (annotations_array > 0).astype(float)
    annotation_percentages = np.dstack((annotation_percentages, annotation_percentages))
    annotation_percentages_original = annotation_percentages.copy()

    validation_start_time = time.time()
    logger.info('Building validation tiles...')

    if len(glob.glob(os.path.join(big_tiles_path, file_pattern))) >= num_validation_tiles:
        logger.info('  Already done.')
    else:
        while len(glob.glob(os.path.join(big_tiles_path, file_pattern))) < num_validation_tiles:
            current_annotations, annotation_percentages = combine_annotations_into_tiles(
                annotations_array,
                current_annotations,
                annotation_percentages,
                image_list,
                num_classes,
                model_path,
                output_type,
                tile_size,
                config
            )

            elapsed_time = time.time() - validation_start_time
            logger.info(
                f'  {len(glob.glob(os.path.join(big_tiles_path, file_pattern)))} of {num_validation_tiles} validation images completed in {int(elapsed_time / 60)} minutes')

            # Report usage statistics
            base_class_count = np.sum(annotation_percentages_original[:, :, 0], axis=0)
            used_class_count = np.sum(annotation_percentages[:, :, 0], axis=0)
            base_unique_annotations = np.sum(annotation_percentages_original[:, :, 1] == 1, axis=0)
            used_unique_annotations = np.sum(annotation_percentages[:, :, 1] == 2, axis=0)

            percent_count_used = used_class_count / base_class_count * 100
            percent_unique_used = used_unique_annotations / base_unique_annotations * 100

            for b, class_name in enumerate(class_names):
                logger.info(f'  Used {percent_count_used[b]:.1f}% counts and {percent_unique_used[b]:.1f}% unique annotations of {class_name}')

    # Report validation tile creation time
    total_time_validation_bigtiles = time.time() - validation_start_time
    hours, rem = divmod(total_time_validation_bigtiles, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f'  Elapsed time to create validation big tiles: {int(hours)}h {int(minutes)}m {int(seconds)}s')
    logger.info('')

    # Update model metadata with tile format used
    with open(os.path.join(model_path, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)

    data['tile_format'] = config.file_format

    with open(os.path.join(model_path, 'net.pkl'), 'wb') as f:
        pickle.dump(data, f)

    logger.debug(f'Updated model metadata with tile_format: {config.file_format}')

    # ========== Generate Metadata for PyTorch Training ==========
    # PyTorch training requires annotations.pkl and train_list.pkl
    # TensorFlow uses glob-based discovery and doesn't need these files
    logger.info('Generating training metadata for PyTorch compatibility...')

    # 1. Build training image list from generated tiles
    training_image_dir = os.path.join(model_path, 'training', 'im')
    image_files = sorted(glob.glob(os.path.join(training_image_dir, f'*.{config.file_format}')))
    train_image_list = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

    # 2. Create annotations dictionary (lightweight: just map ID -> ID)
    # PyTorch create_dataloaders() only uses dict keys, not values
    # So we use a lightweight mapping instead of loading full masks
    annotations_dict = {image_id: image_id for image_id in train_image_list}

    # 3. Save annotations.pkl
    annotations_pkl_path = os.path.join(model_path, 'annotations.pkl')
    with open(annotations_pkl_path, 'wb') as f:
        pickle.dump(annotations_dict, f)
    logger.info(f'Saved annotations metadata: {annotations_pkl_path} ({len(annotations_dict)} entries)')

    # 4. Save train_list.pkl
    train_list_pkl_path = os.path.join(model_path, 'train_list.pkl')
    with open(train_list_pkl_path, 'wb') as f:
        pickle.dump(train_image_list, f)
    logger.info(f'Saved training image list: {train_list_pkl_path} ({len(train_image_list)} images)')


def create_training_tiles_modern(
    model_path: str,
    annotations: np.ndarray,
    image_list: Dict[str, List[str]],
    create_new_tiles: bool
) -> None:
    """
    Create training tiles using modern algorithm (CODAvision-style).

    This mode has been empirically shown to produce better results for some datasets:
    - Reduction factor: 10 (coarser placement optimization)
    - Disk filter: Disabled
    - Rotation cropping: Disabled (keeps expanded dimensions, reduces black pixels)
    - Class rotation: Every 5th iteration
    - Deterministic seed: 3 (reproducible results)
    - Big tile size: 10240
    - File format: PNG

    Args:
        model_path: Path to the directory containing model data
        annotations: Array containing annotation pixel counts by class
        image_list: Dictionary with lists of image tile paths and names
        create_new_tiles: Flag indicating whether to create new tiles or use existing ones
    """
    from base.config import MODERN_CONFIG
    return create_training_tiles(
        model_path, annotations, image_list, create_new_tiles,
        config=MODERN_CONFIG
    )


def create_training_tiles_legacy(
    model_path: str,
    annotations: np.ndarray,
    image_list: Dict[str, List[str]],
    create_new_tiles: bool
) -> None:
    """
    Create training tiles using legacy algorithm (MATLAB-aligned).

    This mode uses the sophisticated MATLAB-aligned implementation:
    - Reduction factor: 5 (fine placement optimization)
    - Disk filter: Enabled (disk convolution for placement)
    - Rotation cropping: Enabled (MATLAB imrotate behavior)
    - Class rotation: Every 3rd iteration
    - Diverse random runs (no seed for increased variability)
    - Big tile size: 10000
    - File format: TIFF (lossless)

    Args:
        model_path: Path to the directory containing model data
        annotations: Array containing annotation pixel counts by class
        image_list: Dictionary with lists of image tile paths and names
        create_new_tiles: Flag indicating whether to create new tiles or use existing ones
    """
    from base.config import LEGACY_CONFIG
    return create_training_tiles(
        model_path, annotations, image_list, create_new_tiles,
        config=LEGACY_CONFIG
    )