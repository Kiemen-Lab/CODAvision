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
        # In H&E mode, tissue is where green < threshold
        # (H&E stained tissue appears darker/less green than background)
        mask = (green_channel < threshold) * 255
    else:
        # In grayscale mode, tissue is where green > threshold
        # (tissue appears brighter than background)
        mask = (green_channel > threshold) * 255
    
    return mask.astype(np.uint8)


def calculate_tissue_mask(path: str, image_name: str, test: bool = False, 
                         training_path: str = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Reads an image and returns it along with a binary mask of tissue areas.

    Args:
        path: Directory path where the image is located
        image_name: Name of the image file (without extension)
        test: Whether this is for testing (affects averaging behavior)
        training_path: Optional path to training images directory where tissue masks may exist

    Returns:
        Tuple of:
        - image: The image as a numpy array
        - tissue_mask: Binary mask where tissue areas are True
        - output_path: Path where the tissue mask is saved
    """
    import cv2
    import pickle
    import time
    from skimage.morphology import remove_small_objects, disk
    from ..image.utils import load_image_with_fallback
    
    # Normalize paths to avoid comparison issues
    path = os.path.normpath(os.path.abspath(path))
    if training_path:
        training_path = os.path.normpath(os.path.abspath(training_path))
    
    # Log the paths being used for debugging
    logger.debug(f"calculate_tissue_mask called with:")
    logger.debug(f"  path (normalized): {path}")
    logger.debug(f"  image_name: {image_name}")
    logger.debug(f"  training_path (normalized): {training_path}")
    logger.debug(f"  paths are equal: {path == training_path if training_path else 'N/A'}")
    
    # Create output directory with better handling
    output_path = os.path.join(path.rstrip(os.path.sep), 'TA')
    directory_just_created = False
    if not os.path.isdir(output_path):
        logger.debug(f"Creating TA directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        directory_just_created = True
        # Small delay to ensure filesystem sync
        time.sleep(0.05)

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

    # Get the resolution from the path first (e.g., '10x' from '/path/to/10x/')
    path_parts = path.rstrip(os.path.sep).split(os.path.sep)
    resolution = None
    for part in reversed(path_parts):
        if part.endswith('x') and part[:-1].isdigit():
            resolution = part
            break
    
    # Check if mask already exists in the current directory with enhanced retry logic
    mask_path = os.path.join(output_path, f'{image_name}.tif')
    mask_path = os.path.normpath(os.path.abspath(mask_path))  # Ensure absolute normalized path
    logger.debug(f"Checking for existing mask at: {mask_path}")
    logger.debug(f"  Working directory: {os.getcwd()}")
    logger.debug(f"  Directory just created: {directory_just_created}")
    logger.debug(f"  Resolution detected: {resolution}")
    
    # Create a list of all paths to check for existing masks
    paths_to_check = []
    
    # 1. Primary mask path
    paths_to_check.append(("primary", mask_path))
    
    # 2. Parent directory's TA folder
    parent_path = os.path.dirname(path.rstrip(os.path.sep))
    if parent_path:
        parent_ta_path = os.path.join(parent_path, 'TA', f'{image_name}.tif')
        parent_ta_path = os.path.normpath(os.path.abspath(parent_ta_path))
        if parent_ta_path != mask_path:
            paths_to_check.append(("parent", parent_ta_path))
    
    # 3. Alternative path with resolution
    if resolution and resolution in path:
        path_without_resolution = path.replace(os.path.sep + resolution, '')
        alt_ta_path = os.path.join(path_without_resolution, resolution, 'TA', f'{image_name}.tif')
        alt_ta_path = os.path.normpath(os.path.abspath(alt_ta_path))
        if alt_ta_path not in [p[1] for p in paths_to_check]:
            paths_to_check.append(("alternative", alt_ta_path))
    
    # 4. Training path if provided and different
    if training_path:
        training_path_parts = training_path.rstrip(os.path.sep).split(os.path.sep)
        last_part = training_path_parts[-1] if training_path_parts else ''
        
        if last_part.endswith('x') and last_part[:-1].isdigit():
            training_ta_path = os.path.join(training_path, 'TA', f'{image_name}.tif')
        elif resolution:
            training_ta_path = os.path.join(training_path, resolution, 'TA', f'{image_name}.tif')
        else:
            training_ta_path = os.path.join(training_path, 'TA', f'{image_name}.tif')
        
        training_ta_path = os.path.normpath(os.path.abspath(training_ta_path))
        if training_ta_path not in [p[1] for p in paths_to_check]:
            paths_to_check.append(("training", training_ta_path))
    
    logger.debug(f"Will check {len(paths_to_check)} potential mask locations")
    
    # Check each potential location for existing masks
    for location_name, check_path in paths_to_check:
        logger.debug(f"Checking {location_name} location: {check_path}")
        
        
        # Log directory contents for this location
        ta_dir = os.path.dirname(check_path)
        if os.path.exists(ta_dir):
            ta_contents = os.listdir(ta_dir)
            mask_filename = os.path.basename(check_path)
            logger.debug(f"  Directory exists with {len(ta_contents)} files")
            logger.debug(f"  Looking for: {mask_filename}")
            logger.debug(f"  File in directory: {mask_filename in ta_contents}")
            
        else:
            logger.debug(f"  Directory does not exist: {ta_dir}")
            continue
        
        # Try to load mask from this location with retry logic
        for attempt in range(2):  # Reduced retries per location
            exists = os.path.exists(check_path)
            is_file = os.path.isfile(check_path) if exists else False
            
            # Handle filesystem synchronization issues
            if mask_filename in ta_contents and not exists:
                # Sometimes os.path.exists returns False due to filesystem sync issues
                # Try to open the file directly as a workaround
                try:
                    with open(check_path, 'rb') as f:
                        f.read(1)
                    exists = True
                    is_file = True
                    logger.debug(f"  File {mask_filename} found via direct open despite os.path.exists returning False")
                except Exception:
                    # File truly doesn't exist
                    pass
            
            if exists and is_file:
                try:
                    # Verify file is readable
                    with open(check_path, 'rb') as f:
                        f.read(1)
                    
                    # Load the tissue mask
                    tissue_mask = load_image_with_fallback(check_path, "L")
                    logger.info(f'  Existing TA loaded from {location_name} location: {check_path}')
                    return image, tissue_mask, output_path
                    
                except Exception as e:
                    logger.debug(f"  Failed to read from {location_name} on attempt {attempt + 1}: {type(e).__name__}: {e}")
                    if attempt == 0:
                        time.sleep(0.5)
                    continue
            else:
                logger.debug(f"  File not found at {location_name} location")
                break  # No point retrying if file doesn't exist
    
    # If no existing mask was found in any location, we need to calculate it
    logger.debug(f"No existing tissue mask found after checking all {len(paths_to_check)} locations")

    # Calculate tissue mask
    logger.info('  Calculating TA image')
    mode = 'H&E'
    cutoff = 205  # Default cutoff value
    
    # Try to load saved threshold from current directory first
    cutoff_file = os.path.join(output_path, 'TA_cutoff.pkl')
    cutoff_loaded = False
    logger.debug(f"Looking for TA_cutoff.pkl at: {cutoff_file}")
    
    # If not found in current directory, check parent directory
    if not os.path.isfile(cutoff_file) and parent_path:
        parent_cutoff_file = os.path.join(parent_path, 'TA', 'TA_cutoff.pkl')
        logger.debug(f"Checking parent directory for TA_cutoff.pkl at: {parent_cutoff_file}")
        if os.path.isfile(parent_cutoff_file):
            cutoff_file = parent_cutoff_file
    
    # If still not found, check training directory
    if not os.path.isfile(cutoff_file) and training_path and training_path != path:
        # Use same logic as for tissue mask path
        training_path_parts = training_path.rstrip(os.path.sep).split(os.path.sep)
        last_part = training_path_parts[-1] if training_path_parts else ''
        
        if last_part.endswith('x') and last_part[:-1].isdigit():
            training_cutoff_file = os.path.join(training_path, 'TA', 'TA_cutoff.pkl')
        elif resolution:
            training_cutoff_file = os.path.join(training_path, resolution, 'TA', 'TA_cutoff.pkl')
        else:
            training_cutoff_file = os.path.join(training_path, 'TA', 'TA_cutoff.pkl')
            
        logger.debug(f"Checking training directory for TA_cutoff.pkl at: {training_cutoff_file}")
        if os.path.isfile(training_cutoff_file):
            cutoff_file = training_cutoff_file
    
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