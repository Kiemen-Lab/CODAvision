"""
Training Tile Creation Utilities for CODAvision

This module provides functions for creating and managing training tiles from annotations.
It handles the combination of annotations into larger tiles for deep learning model training.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: April 2025
"""

import os
import time
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import cv2
from PIL import Image

from ..image.augmentation import edit_annotation_tiles


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
    # Add margin to the big tile to allow for cropping later
    big_tile_size_with_margin = big_tile_size + 200
    keep_all_classes = 1  # Flag to determine if all classes should be kept
    
    # Create output directories
    output_path_images = os.path.join(model_path, output_folder, 'im')
    output_path_labels = os.path.join(model_path, output_folder, 'label')
    output_path_big_tiles = os.path.join(model_path, output_folder, 'big_tiles')
    
    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_labels, exist_ok=True)
    os.makedirs(output_path_big_tiles, exist_ok=True)
    
    # Check how many images are already processed
    existing_images = [f for f in os.listdir(output_path_images) if f.endswith('.png')]
    next_image_number = len(existing_images) + 1
    
    # Initialize the composite image and mask
    composite_image = np.full((big_tile_size_with_margin, big_tile_size_with_margin, 3), 
                             background_class, dtype=np.float64)
    composite_mask = np.zeros((big_tile_size_with_margin, big_tile_size_with_margin), dtype=np.uint8)
    total_pixels = composite_mask.size
    
    # Initialize class pixel counters
    class_counts = np.zeros(current_annotations.shape[1])
    fill_ratio = np.sum(class_counts) / total_pixels
    
    # Counters and thresholds for the tile filling process
    count = 1
    tile_count = 1
    cutoff_threshold = 0.55
    reduction_factor = 10
    last_class_type = 0
    num_tiles_used = np.zeros(len(image_list['tile_name']))
    class_type_counts = np.zeros(len(class_counts))
    
    # Continue filling until we reach desired coverage
    iteration = 1
    
    while fill_ratio < cutoff_threshold:
        iteration_start_time = time.time()
        
        # Select class to add next - alternating between sequential and least common
        if count % 10 == 1:
            class_type = tile_count - 1
            tile_count = (tile_count % num_classes) + 1
        else:
            # Choose the least represented class
            tmp = class_counts.copy()
            tmp[last_class_type] = np.max(tmp)  # Avoid selecting the same class twice in a row
            class_type = np.argmin(tmp)
        
        # Track the usage of this class type
        class_type_counts[class_type] += 1
        
        # Find tiles containing this class
        candidate_tiles = np.where(current_annotations[:, class_type] > 0)[0]
        
        # If no tiles with this class, reset from initial data
        if len(candidate_tiles) == 0:
            current_annotations[:, class_type] = initial_annotations[:, class_type]
            candidate_tiles = np.where(current_annotations[:, class_type] > 0)[0]
            
        # Randomly select a tile that contains this class
        selected_tile_idx = np.random.choice(candidate_tiles, size=1, replace=False)
        num_tiles_used[selected_tile_idx[0]] += 1
        
        # Get the selected tile information
        tile_name = image_list['tile_name'][selected_tile_idx[0]]
        tile_path = os.path.join(image_list['tile_pth'][selected_tile_idx[0]], tile_name)
        
        # Extract the path to the label folder
        path_separator_pos = tile_path.rfind(os.path.sep, 0, tile_path.rfind(os.path.sep))
        label_path = os.path.join(tile_path[0:path_separator_pos], 'label')
        
        # Load image and its annotation mask
        image = cv2.imread(tile_path)
        annotation_mask = cv2.imread(os.path.join(label_path, tile_name), cv2.IMREAD_GRAYSCALE)
        
        # Apply augmentation randomly
        apply_augmentation = 1 if count % 3 == 1 else 0
        
        # Edit and augment the tile
        image, annotation_mask, kept_classes = edit_annotation_tiles(
            image, annotation_mask, apply_augmentation, class_type, class_counts, 
            composite_mask.shape[0], keep_all_classes
        )
        
        # Mark this annotation as used
        current_annotations[selected_tile_idx[0], kept_classes - 1] = 0
        annotation_percentages[selected_tile_idx[0], kept_classes - 1, 0] += 1
        annotation_percentages[selected_tile_idx[0], kept_classes - 1, 1] = 2
        
        # Find valid pixels in the annotation mask
        valid_pixels = (annotation_mask != 0)
        if np.sum(valid_pixels) < 30:
            print('  Skipped tile with too few valid pixels')
            continue
        
        # Find optimal placement position (area with most free space)
        # Downsample the mask for faster distance calculation
        downsampled_mask = composite_mask[::reduction_factor, ::reduction_factor] > 0
        padding = int(100/reduction_factor)
        
        # Calculate distance transform to find largest empty area
        dist = cv2.distanceTransform((downsampled_mask <= 0).astype(np.uint8), cv2.DIST_L2, 3)
        # Zero out border regions to avoid placing near edges
        dist[:padding, :] = 0
        dist[:, :padding] = 0
        dist[-padding:, :] = 0
        dist[:, -padding:] = 0
        
        # Find point with maximum distance (center of largest empty area)
        max_dist_points = np.where(dist == np.max(dist))
        index = np.random.choice(len(max_dist_points[0]), size=1, replace=False)
        x = int(max_dist_points[0][index[0]] * reduction_factor)
        y = int(max_dist_points[1][index[0]] * reduction_factor)
        
        # Calculate placement boundaries
        annotation_size = np.array(annotation_mask.shape) - 1
        half_size_a = annotation_size // 2
        half_size_b = annotation_size - half_size_a
        
        # Adjust placement if it would go outside the boundaries
        if x + half_size_a[0] + 1 > composite_mask.shape[0]:
            x -= half_size_a[0]
        if y + half_size_a[1] + 1 > composite_mask.shape[1]:
            y -= half_size_a[1]
        if x - half_size_b[0] < 0:
            x += half_size_b[0]
        if y - half_size_b[1] < 0:
            y += half_size_b[1]
        
        # Create region slices for placement
        region_slice_x = slice(x - half_size_b[0], x + half_size_a[0] + 1)
        region_slice_y = slice(y - half_size_b[1], y + half_size_a[1] + 1)
        
        # Update the composite mask and image
        temp_mask = composite_mask[region_slice_x, region_slice_y].copy()
        temp_mask[valid_pixels] = annotation_mask[valid_pixels]
        composite_mask[region_slice_x, region_slice_y] = temp_mask
        
        temp_image = composite_image[region_slice_x, region_slice_y, :].copy()
        valid_pixels_3d = np.dstack((valid_pixels, valid_pixels, valid_pixels))
        temp_image[valid_pixels_3d] = image[valid_pixels_3d]
        composite_image[region_slice_x, region_slice_y, :] = temp_image
        
        # Update fill ratio every few iterations
        if count % 2 == 0:
            fill_ratio = cv2.countNonZero(composite_mask) / total_pixels
        
        # Update class counts 
        for class_idx in range(current_annotations.shape[1]):
            class_counts[class_idx] += np.sum(temp_mask == class_idx + 1)
        
        # Recalculate class distributions periodically
        if count % 150 == 0 or fill_ratio > cutoff_threshold:
            class_histogram = np.histogram(composite_mask, bins=np.arange(current_annotations.shape[1] + 2))[0]
            class_counts = class_histogram[1:]
            # Avoid division by zero
            class_counts[class_counts == 0] = 1
        
        # Prepare for next iteration
        count += 1
        last_class_type = class_type
        iteration += 1
        
        elapsed_time = time.time() - iteration_start_time
    
    # Remove margins from the final tile
    composite_image = composite_image[100:-100, 100:-100, :].astype(np.float64)
    composite_mask = composite_mask[100:-100, 100:-100].astype(np.uint8)
    
    # Update class counts
    for class_idx in range(num_classes - 1):
        class_counts[class_idx] = np.sum(composite_mask == class_idx + 1)
    
    # Remap classes for final output (subtract 1 to make background 0)
    composite_mask[composite_mask == 0] = num_classes
    composite_mask = composite_mask - 1
    
    # Slice the big tile into smaller tiles
    for row in range(0, composite_image.shape[0], tile_size):
        for col in range(0, composite_image.shape[1], tile_size):
            try:
                # Extract tile regions
                image_tile = composite_image[row:row + tile_size, col:col + tile_size, :]
                mask_tile = composite_mask[row:row + tile_size, col:col + tile_size]
                
                # Save tiles
                cv2.imwrite(os.path.join(output_path_images, f"{next_image_number}.png"), image_tile)
                Image.fromarray(mask_tile).save(os.path.join(output_path_labels, f"{next_image_number}.png"))
                
                next_image_number += 1
            except ValueError:
                # Skip if the slice is out of bounds
                continue
    
    # Save the complete big tile
    big_tile_number = len([f for f in os.listdir(output_path_big_tiles) if f.startswith('HE')]) + 1
    print('  Saving big tile')
    cv2.imwrite(os.path.join(output_path_big_tiles, f"HE_tile_{big_tile_number}.jpg"), composite_image)
    Image.fromarray(composite_mask).save(os.path.join(output_path_big_tiles, f"label_tile_{big_tile_number}.jpg"))
    
    return current_annotations, annotation_percentages