"""
Utility Functions for Tissue Area Threshold Detection

This module provides helper functions for image processing and threshold
operations.
"""

import os
import numpy as np
from glob import glob
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_images_list(training_path: str, testing_path: str) -> List[str]:
    """
    Load list of images from training and testing directories.
    
    Args:
        training_path: Path to training images directory
        testing_path: Path to testing images directory
        
    Returns:
        List of image file paths (relative paths for training, full for testing)
    """
    # Look for TIFF files first
    imlist = sorted(glob(os.path.join(training_path, '*.tif')))
    imtestlist = sorted(glob(os.path.join(testing_path, '*.tif')))
    
    # If no TIFF files, look for JPG and PNG
    if not imlist:
        for ext in ['*.jpg', '*.png']:
            imlist.extend(glob(os.path.join(training_path, ext)))
        for ext in ['*.jpg', '*.png']:
            imtestlist.extend(glob(os.path.join(testing_path, ext)))
    
    if not imlist:
        logger.info(f"No TIFF, PNG or JPG image files found in either {training_path} or {testing_path}")
        return []
    
    # Convert training paths to relative paths
    relative_imlist = [os.path.basename(img) for img in imlist]
    
    # Combine lists
    relative_imlist.extend(imtestlist)
    
    return relative_imlist


def select_random_images(image_list: List[str], num_images: int) -> List[str]:
    """
    Select random images from the list.
    
    Args:
        image_list: List of image paths
        num_images: Number of images to select (0 means all)
        
    Returns:
        Selected image paths
    """
    if num_images <= 0 or num_images >= len(image_list):
        return image_list
    
    return list(np.random.choice(image_list, size=num_images, replace=False))


def calculate_resize_factor(image_shape: Tuple[int, ...], 
                          max_width: int = 1500, 
                          max_height: int = 780) -> float:
    """
    Calculate resize factor to fit image in display area.
    
    Args:
        image_shape: Shape of the image
        max_width: Maximum display width
        max_height: Maximum display height
        
    Returns:
        Resize factor
    """
    height, width = image_shape[:2]
    return min(max_width / width, max_height / height)


def extract_cropped_region(image: np.ndarray, 
                         region: 'RegionSelection',
                         resize_factor: float) -> np.ndarray:
    """
    Extract a cropped region from an image.
    
    Args:
        image: Input image
        region: Region selection with normalized coordinates
        resize_factor: Factor used for display
        
    Returns:
        Cropped image region
    """
    # Convert display coordinates to image coordinates
    x_norm = int(np.round(region.x / resize_factor))
    y_norm = int(np.round(region.y / resize_factor))
    
    # Update region with image coordinates
    region.x = x_norm
    region.y = y_norm
    
    # Get crop bounds
    y_start, y_end, x_start, x_end = region.get_bounds(image.shape)
    
    # Handle edge cases with padding if needed
    if 2 * region.size > image.shape[0] or 2 * region.size > image.shape[1]:
        return _pad_and_crop(image, region)
    
    # Simple crop
    return image[y_start:y_end, x_start:x_end, :]


def _pad_and_crop(image: np.ndarray, region: 'RegionSelection') -> np.ndarray:
    """
    Pad image if necessary and then crop.
    
    Args:
        image: Input image
        region: Region selection
        
    Returns:
        Cropped region with padding if needed
    """
    size = region.size
    height, width = image.shape[:2]
    
    # Determine padding needed
    if 2 * size > height and 2 * size > width:
        # Pad both dimensions
        max_dim = max(height, width, 2 * size)
        pad_y = (max_dim - height) // 2
        pad_x = (max_dim - width) // 2
        pad_y1, pad_y2 = pad_y, pad_y + (max_dim - height) % 2
        pad_x1, pad_x2 = pad_x, pad_x + (max_dim - width) % 2
        padded = np.pad(image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), mode='constant')
        return padded
    
    elif 2 * size > height:
        # Pad height only
        pad_y = (2 * size - height) // 2
        pad_y1, pad_y2 = pad_y, pad_y + (2 * size - height) % 2
        padded = np.pad(image, ((pad_y1, pad_y2), (0, 0), (0, 0)), mode='constant')
        y_start, y_end, x_start, x_end = region.get_bounds(padded.shape)
        return padded[y_start:y_end, x_start:x_end, :]
    
    elif 2 * size > width:
        # Pad width only
        pad_x = (2 * size - width) // 2
        pad_x1, pad_x2 = pad_x, pad_x + (2 * size - width) % 2
        padded = np.pad(image, ((0, 0), (pad_x1, pad_x2), (0, 0)), mode='constant')
        y_start, y_end, x_start, x_end = region.get_bounds(padded.shape)
        return padded[y_start:y_end, x_start:x_end, :]
    
    # No padding needed
    y_start, y_end, x_start, x_end = region.get_bounds(image.shape)
    return image[y_start:y_end, x_start:x_end, :]


def create_tissue_mask(image: np.ndarray, 
                      threshold: int, 
                      mode: 'ThresholdMode') -> np.ndarray:
    """
    Create a tissue mask from an image using threshold.
    
    Args:
        image: Input image (RGB)
        threshold: Threshold value
        mode: Threshold mode (H&E or Grayscale)
        
    Returns:
        Binary mask (255 for tissue/whitespace based on mode)
    """
    from .models import ThresholdMode
    
    # Use green channel for thresholding
    green_channel = image[:, :, 1]
    
    if mode == ThresholdMode.HE:
        # In H&E mode, tissue is where green > threshold
        mask = (green_channel > threshold) * 255
    else:
        # In grayscale mode, tissue is where green < threshold
        mask = (green_channel < threshold) * 255
    
    return mask.astype(np.uint8)


def calculate_tissue_mask(path: str, image_name: str, test: bool = False) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Reads an image and returns it along with a binary mask of tissue areas.

    Args:
        path: Directory path where the image is located
        image_name: Name of the image file (without extension)
        test: Whether this is for testing (affects averaging behavior)

    Returns:
        Tuple of:
        - image: The image as a numpy array
        - tissue_mask: Binary mask where tissue areas are True
        - output_path: Path where the tissue mask is saved
    """
    import cv2
    import pickle
    from skimage.morphology import remove_small_objects, disk
    from ..image.utils import load_image_with_fallback
    
    # Create output directory
    output_path = os.path.join(path.rstrip(os.path.sep), 'TA')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Try to load image with different extensions
    image_extensions = ['.tif', '.jpg', '.jp2', '.png']
    image = None
    
    for ext in image_extensions:
        try:
            image = load_image_with_fallback(os.path.join(path, f'{image_name}{ext}'))
            break
        except Exception:
            continue
    
    if image is None:
        raise FileNotFoundError(f"Could not load image {image_name} with any supported extension")

    # Check if mask already exists
    mask_path = os.path.join(output_path, f'{image_name}.tif')
    if os.path.isfile(mask_path):
        tissue_mask = load_image_with_fallback(mask_path, "L")
        logger.info('  Existing TA loaded')
        return image, tissue_mask, output_path

    # Calculate tissue mask
    logger.info('  Calculating TA image')
    mode = 'H&E'
    cutoff = 205  # Default cutoff value
    
    # Try to load saved threshold
    cutoff_file = os.path.join(output_path, 'TA_cutoff.pkl')
    if os.path.isfile(cutoff_file):
        with open(cutoff_file, 'rb') as f:
            data = pickle.load(f)
            cutoffs_list = data['cts']
            mode = data['mode']
            average_TA = data.get('average_TA', False)
            
            # Always use averaged cutoffs in test mode
            if test:
                average_TA = True
                
            if average_TA:
                # Calculate average cutoff
                cutoff = sum(cutoffs_list.values()) / len(cutoffs_list)
            else:
                # Try to get image-specific cutoff
                img_key = f'{os.path.basename(image_name)}.tif'
                if img_key in cutoffs_list:
                    cutoff = cutoffs_list[img_key]

    # Apply threshold based on mode
    if mode == 'H&E':
        tissue_mask = image[:, :, 1] < cutoff  # Green channel for H&E stains
    else:
        tissue_mask = image[:, :, 1] > cutoff  # Inverted for other stains

    # Apply morphological operations to clean up the mask
    kernel_size = 3  # Using larger kernel size from loaders.py for better cleaning
    tissue_mask = tissue_mask.astype(np.uint8)
    kernel = disk(kernel_size)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel.astype(np.uint8))
    
    # Remove small objects
    tissue_mask = remove_small_objects(tissue_mask.astype(bool), min_size=10)
    
    # Additional cleaning step from annotation.py implementation
    inverted_mask = ~tissue_mask
    inverted_mask = remove_small_objects(inverted_mask, min_size=10)
    tissue_mask = ~inverted_mask

    # Save tissue mask
    cv2.imwrite(mask_path, tissue_mask.astype(np.uint8) * 255)

    return image, tissue_mask.astype(np.uint8) * 255, output_path