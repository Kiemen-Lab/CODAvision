"""
Training Tile Creation Utilities for CODAvision

This module provides functions for creating and managing training tiles from annotations.
It handles the creation of training and validation tiles for deep learning model training.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: April 2025
"""

import os
import glob
import shutil
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
import cv2

from base.image import edit_annotation_tiles


def combine_annotations_into_tiles(
    initial_annotations: np.ndarray,
    current_annotations: np.ndarray,
    annotation_percentages: np.ndarray,
    image_list: Dict[str, List[str]],
    num_classes: int,
    model_path: str,
    output_folder: str,
    tile_size: int,
    big_tile_size: int = 10240,
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
        big_tile_size: Size of the big tiles before cutting into smaller ones (default: 10240)
        background_class: Background class index (default: 0)

    Returns:
        Tuple containing:
        - Updated annotation counts array
        - Updated annotation percentage tracking array
    """
    big_tile_size_with_margin = big_tile_size + 200
    keep_all_classes = 1

    # Create output directories
    output_path_images = os.path.join(model_path, output_folder, 'im')
    output_path_labels = os.path.join(model_path, output_folder, 'label')
    output_path_big_tiles = os.path.join(model_path, output_folder, 'big_tiles')

    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_labels, exist_ok=True)
    os.makedirs(output_path_big_tiles, exist_ok=True)

    # Check if we've already done this work
    existing_images = [f for f in os.listdir(output_path_images) if f.endswith('.png')]
    next_image_number = len(existing_images) + 1

    # Initialize composite canvas
    composite_image = np.full((big_tile_size_with_margin, big_tile_size_with_margin, 3),
                             background_class, dtype=np.float64)
    composite_mask = np.zeros((big_tile_size_with_margin, big_tile_size_with_margin), dtype=np.uint8)
    total_pixels = composite_mask.size

    # Track class balancing
    class_counts = np.zeros(current_annotations.shape[1])
    fill_ratio = np.sum(class_counts) / total_pixels

    # Iteration control variables
    count = 1
    tile_count = 1
    cutoff_threshold = 0.55
    reduction_factor = 10
    last_class_type = 0
    num_tiles_used = np.zeros(len(image_list['tile_name']))
    class_type_counts = np.zeros(num_classes)

    # Main loop
    iteration = 1

    while fill_ratio < cutoff_threshold:
        iteration_start_time = time.time()

        # Select which class to focus on for this iteration
        if count % 10 == 1:
            class_type = tile_count - 1
            tile_count = (tile_count % num_classes) + 1
        else:
            # Focus on least represented class
            tmp = class_counts.copy()
            tmp[last_class_type] = np.max(tmp)
            class_type = np.argmin(tmp)

        # Track class usage
        if class_type < current_annotations.shape[1]:
            candidate_tiles = np.where(current_annotations[:, class_type] > 0)[0]
        else:
            # Handle the out-of-bounds case - create an empty array
            print(
                f"Warning: class_type {class_type} is out of bounds for current_annotations with shape {current_annotations.shape}")
            candidate_tiles = np.array([], dtype=int)

        # If we've used all tiles for this class, reset from initial annotations
        if len(candidate_tiles) == 0:
            current_annotations[:, class_type] = initial_annotations[:, class_type]
            candidate_tiles = np.where(current_annotations[:, class_type] > 0)[0]

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
        image = cv2.imread(tile_path)
        annotation_mask = cv2.imread(os.path.join(label_path, tile_name), cv2.IMREAD_GRAYSCALE)

        # Apply optional augmentation
        apply_augmentation = 1 if count % 3 == 1 else 0

        # Get augmented tiles
        image, annotation_mask, kept_classes = edit_annotation_tiles(
            image, annotation_mask, apply_augmentation, class_type, class_counts,
            composite_mask.shape[0], keep_all_classes
        )

        # Update tracking of which annotations we've used
        current_annotations[selected_tile_idx[0], kept_classes - 1] = 0
        annotation_percentages[selected_tile_idx[0], kept_classes - 1, 0] += 1
        annotation_percentages[selected_tile_idx[0], kept_classes - 1, 1] = 2

        # Check if we have enough annotation pixels to use
        valid_pixels = (annotation_mask != 0)
        if np.sum(valid_pixels) < 30:
            print('  Skipped tile with too few valid pixels')
            continue

        # Find optimal location to place this tile in the composite
        # Downsample for faster distance calculation
        downsampled_mask = composite_mask[::reduction_factor, ::reduction_factor] > 0
        padding = int(100/reduction_factor)

        # Calculate distance transform to find largest empty area
        dist = cv2.distanceTransform((downsampled_mask <= 0).astype(np.uint8), cv2.DIST_L2, 3)
        # Add border padding
        dist[:padding, :] = 0
        dist[:, :padding] = 0
        dist[-padding:, :] = 0
        dist[:, -padding:] = 0

        # Find point with maximum distance from any occupied area
        max_dist_points = np.where(dist == np.max(dist))
        index = np.random.choice(len(max_dist_points[0]), size=1, replace=False)
        x = int(max_dist_points[0][index[0]] * reduction_factor)
        y = int(max_dist_points[1][index[0]] * reduction_factor)

        # Calculate placement position ensuring we stay in bounds
        annotation_size = np.array(annotation_mask.shape) - 1
        half_size_a = annotation_size // 2
        half_size_b = annotation_size - half_size_a

        # Adjust position if needed to stay in bounds
        if x + half_size_a[0] + 1 > composite_mask.shape[0]:
            x -= half_size_a[0]
        if y + half_size_a[1] + 1 > composite_mask.shape[1]:
            y -= half_size_a[1]
        if x - half_size_b[0] < 0:
            x += half_size_b[0]
        if y - half_size_b[1] < 0:
            y += half_size_b[1]

        # Create slices for the region where we'll place this tile
        region_slice_x = slice(x - half_size_b[0], x + half_size_a[0] + 1)
        region_slice_y = slice(y - half_size_b[1], y + half_size_a[1] + 1)

        # Place the tile in the composite
        temp_mask = composite_mask[region_slice_x, region_slice_y].copy()
        temp_mask[valid_pixels] = annotation_mask[valid_pixels]
        composite_mask[region_slice_x, region_slice_y] = temp_mask

        temp_image = composite_image[region_slice_x, region_slice_y, :].copy()
        valid_pixels_3d = np.dstack((valid_pixels, valid_pixels, valid_pixels))
        temp_image[valid_pixels_3d] = image[valid_pixels_3d]
        composite_image[region_slice_x, region_slice_y, :] = temp_image

        # Update fill ratio periodically
        if count % 2 == 0:
            fill_ratio = cv2.countNonZero(composite_mask) / total_pixels

        # Update class counts from newly added content
        for class_idx in range(current_annotations.shape[1]):
            class_counts[class_idx] += np.sum(temp_mask == class_idx + 1)

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

        elapsed_time = time.time() - iteration_start_time

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

                # Save tiles
                cv2.imwrite(os.path.join(output_path_images, f"{next_image_number}.png"), image_tile)
                Image.fromarray(mask_tile).save(os.path.join(output_path_labels, f"{next_image_number}.png"))

                next_image_number += 1
            except ValueError:
                # Handle edge case where tile would exceed bounds
                continue

    # Save the big tile for reference
    big_tile_number = len([f for f in os.listdir(output_path_big_tiles) if f.startswith('HE')]) + 1
    print('  Saving big tile')
    cv2.imwrite(os.path.join(output_path_big_tiles, f"HE_tile_{big_tile_number}.jpg"), composite_image)
    Image.fromarray(composite_mask).save(os.path.join(output_path_big_tiles, f"label_tile_{big_tile_number}.jpg"))

    return current_annotations, annotation_percentages


def create_training_tiles(
    model_path: str,
    annotations: np.ndarray,
    image_list: Dict[str, List[str]],
    create_new_tiles: bool
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

    Raises:
        ValueError: If no valid annotations are found
    """
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
    print('')

    # Verify annotations exist
    if not annotations or len(annotations) == 0:
        raise ValueError(
            'No annotation data found. Please ensure that annotation files exist and contain valid annotations.'
        )

    # Calculate total number of pixels in training dataset
    print('Calculating total number of pixels in the training dataset...')
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
            print(f' There are {np.round(count*max(count_annotations))} pixels of {class_names[b]}. This is the most common class.')
        else:
            print(
                f' There are {np.round(count*max(count_annotations))} pixels of {class_names[b]}, {int(annotation_composition[b])}% of the most common class.')

    # Ensure all classes have annotations
    if 0 in count_annotations:
        raise ValueError(
            'There are no annotations for one or more classes. Please add annotations, check nesting, or remove empty classes.'
        )

    # Create training tiles
    print('')
    print('Building training tiles...')
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
    if len(glob.glob(os.path.join(big_tiles_path, 'HE*.jpg'))) >= num_train_tiles:
        print('  Already done.')
    else:
        while len(glob.glob(os.path.join(big_tiles_path, 'HE*.jpg'))) < num_train_tiles:
            current_annotations, annotation_percentages = combine_annotations_into_tiles(
                annotations_array,
                current_annotations,
                annotation_percentages,
                image_list,
                num_classes,
                model_path,
                output_type,
                tile_size
            )

            elapsed_time = time.time() - train_start
            print(
                f'  {len(glob.glob(os.path.join(big_tiles_path, "HE*.jpg")))} of {num_train_tiles} training images completed in {int(elapsed_time / 60)} minutes')

            # Report usage statistics
            base_class_count = np.sum(annotation_percentages_original[:, :, 0], axis=0)
            used_class_count = np.sum(annotation_percentages[:, :, 0], axis=0)
            base_unique_annotations = np.sum(annotation_percentages_original[:, :, 1] == 1, axis=0)
            used_unique_annotations = np.sum(annotation_percentages[:, :, 1] == 2, axis=0)

            percent_count_used = used_class_count / base_class_count * 100
            percent_unique_used = used_unique_annotations / base_unique_annotations * 100

            for b, class_name in enumerate(class_names):
                print(f'  Used {percent_count_used[b]:.1f}% counts and {percent_unique_used[b]:.1f}% unique annotations of {class_name}')

    # Report training tile creation time
    total_time_train_bigtiles = time.time() - train_start
    hours, rem = divmod(total_time_train_bigtiles, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'  Elapsed time to create training big tiles: {int(hours)}h {int(minutes)}m {int(seconds)}s')

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
    print('Building validation tiles...')

    if len(glob.glob(os.path.join(big_tiles_path, 'HE*.jpg'))) >= num_validation_tiles:
        print('  Already done.')
    else:
        while len(glob.glob(os.path.join(big_tiles_path, 'HE*.jpg'))) < num_validation_tiles:
            current_annotations, annotation_percentages = combine_annotations_into_tiles(
                annotations_array,
                current_annotations,
                annotation_percentages,
                image_list,
                num_classes,
                model_path,
                output_type,
                tile_size
            )

            elapsed_time = time.time() - validation_start_time
            print(
                f'  {len(glob.glob(os.path.join(big_tiles_path, "HE*.jpg")))} of {num_validation_tiles} validation images completed in {int(elapsed_time / 60)} minutes')

            # Report usage statistics
            base_class_count = np.sum(annotation_percentages_original[:, :, 0], axis=0)
            used_class_count = np.sum(annotation_percentages[:, :, 0], axis=0)
            base_unique_annotations = np.sum(annotation_percentages_original[:, :, 1] == 1, axis=0)
            used_unique_annotations = np.sum(annotation_percentages[:, :, 1] == 2, axis=0)

            percent_count_used = used_class_count / base_class_count * 100
            percent_unique_used = used_unique_annotations / base_unique_annotations * 100

            for b, class_name in enumerate(class_names):
                print(f'  Used {percent_count_used[b]:.1f}% counts and {percent_unique_used[b]:.1f}% unique annotations of {class_name}')

    # Report validation tile creation time
    total_time_validation_bigtiles = time.time() - validation_start_time
    hours, rem = divmod(total_time_validation_bigtiles, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'  Elapsed time to create validation big tiles: {int(hours)}h {int(minutes)}m {int(seconds)}s')
    print('')