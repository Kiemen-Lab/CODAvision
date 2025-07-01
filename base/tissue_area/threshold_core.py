"""
Tissue Area Threshold Selection Core Logic (GUI-Independent)

This module provides the core functionality for determining optimal tissue area
thresholds in histological images without any GUI dependencies.
"""

import os
import pickle
import numpy as np
from typing import Optional, Dict, List, Any

from base.image.utils import load_image_with_fallback
from .models import (
    ThresholdConfig, ThresholdMode, ImageThresholds, RegionSelection
)
from .utils import (
    select_random_images, calculate_resize_factor,
    extract_cropped_region, create_tissue_mask
)

import logging
logger = logging.getLogger(__name__)


class TissueAreaThresholdSelector:
    """
    Core class for tissue area threshold selection (GUI-independent).
    
    This class provides all the core logic for threshold selection
    without any GUI dependencies.
    """
    
    def __init__(self, config: ThresholdConfig):
        """
        Initialize the threshold selector.
        
        Args:
            config: Configuration for the threshold selection process
        """
        self.config = config
        self.downsampled_path = config.training_path  # Use training path directly
        self.thresholds = self._load_existing_thresholds()
        
    def _get_local_images(self) -> List[str]:
        """Get list of images from both training and testing paths."""
        from glob import glob
        
        # Get images from training path
        imlist = sorted(glob(os.path.join(self.config.training_path, '*.tif')))
        imtestlist = sorted(glob(os.path.join(self.config.testing_path, '*.tif')))
        
        # If no TIFF files, look for JPG and PNG
        if not imlist:
            for ext in ['*.jpg', '*.png']:
                imlist.extend(glob(os.path.join(self.config.training_path, ext)))
                imtestlist.extend(glob(os.path.join(self.config.testing_path, ext)))
        
        # Convert training paths to basenames only (matching original behavior)
        imlist = [os.path.basename(img) for img in imlist]
        # Keep full paths for test images
        imlist.extend(imtestlist)
        
        return imlist
    
    def _load_existing_thresholds(self) -> ImageThresholds:
        """Load existing threshold data if available."""
        if os.path.isfile(self.config.threshold_file_path):
            try:
                with open(self.config.threshold_file_path, 'rb') as f:
                    data = pickle.load(f)
                return ImageThresholds.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load existing thresholds: {e}")
        
        return ImageThresholds()
    
    def save_thresholds(self):
        """Save threshold data to file."""
        os.makedirs(self.config.output_path, exist_ok=True)
        
        with open(self.config.threshold_file_path, 'wb') as f:
            pickle.dump(self.thresholds.to_dict(), f)
    
    def needs_processing(self) -> bool:
        """
        Check if processing is needed.
        
        Returns:
            True if processing is needed, False otherwise
        """
        # If no existing thresholds, we need to process
        if not self.thresholds.thresholds:
            return True
        
        # Check based on mode
        if self.config.redo:
            return True
        
        # Check if all images have thresholds
        # For threshold selection, we only need training images
        all_images = self._get_local_images()
        processed_images = set(self.thresholds.thresholds.keys())
        
        return len(all_images) > len(processed_images)
    
    def has_existing_evaluation(self) -> bool:
        """
        Check if a tissue mask evaluation already exists.
        
        Returns:
            True if evaluation exists, False otherwise
        """
        # First check if threshold file exists
        if not os.path.isfile(self.config.threshold_file_path):
            return False
            
        # Check if we have threshold data
        if not self.thresholds.thresholds:
            return False
            
        # Check if at least some tissue mask files exist
        ta_dir = self.config.output_path
        if not os.path.exists(ta_dir):
            return False
            
        # Look for actual tissue mask files
        mask_files = [f for f in os.listdir(ta_dir) if f.endswith('.tif')]
        
        # If we have thresholds but no mask files, evaluation is incomplete
        if not mask_files:
            logger.debug("Threshold file exists but no tissue mask files found")
            return False
            
        return True
    
    def is_evaluation_compatible(self, expected_mode: Optional[ThresholdMode] = None) -> bool:
        """
        Check if existing evaluation is compatible with current settings.
        
        Args:
            expected_mode: Expected threshold mode (H&E or Grayscale)
            
        Returns:
            True if compatible, False otherwise
        """
        if not self.has_existing_evaluation():
            return False
            
        # If no expected mode specified, assume compatible
        if expected_mode is None:
            return True
            
        # Check if mode matches
        return self.thresholds.mode == expected_mode
    
    def get_images_to_process(self) -> List[str]:
        """
        Get list of images that need processing.
        
        Returns:
            List of image paths to process
        """
        all_images = self._get_local_images()
        
        if self.config.redo:
            # In redo mode, process all images
            return all_images
        
        # Filter out already processed images
        processed_images = set(self.thresholds.thresholds.keys())
        unprocessed = [
            img for img in all_images 
            if os.path.basename(img) not in processed_images
        ]
        
        # Sample if needed
        if len(unprocessed) > self.config.num_images and self.config.num_images > 0:
            return select_random_images(unprocessed, self.config.num_images)
        
        return unprocessed
    
    def prepare_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Load and prepare an image for processing.
        
        Args:
            image_path: Path to the image (can be basename or full path)
            
        Returns:
            Dictionary with image data or None if loading failed
        """
        try:
            # Determine full path based on whether it's a basename or full path
            if os.path.isabs(image_path) and os.path.exists(image_path):
                full_path = image_path
                image_name = os.path.basename(image_path)
            else:
                # Try to find in training path first
                image_name = os.path.basename(image_path)
                full_path = os.path.join(self.config.training_path, image_name)
                if not os.path.exists(full_path):
                    # Try testing path if not in training
                    full_path = os.path.join(self.config.testing_path, image_name)
            
            # Load image
            image = load_image_with_fallback(full_path)
            if image is None:
                logger.error(f"Failed to load image: {full_path}")
                return None
            
            # Calculate resize factor for display
            resize_factor = calculate_resize_factor(
                image.shape, 
                max_width=1500,
                max_height=780
            )
            
            return {
                'image': image,
                'image_path': full_path,
                'image_name': image_name,
                'resize_factor': resize_factor
            }
            
        except Exception as e:
            logger.error(f"Error preparing image {image_path}: {e}")
            return None
    
    def get_initial_threshold(self, image_name: str) -> int:
        """
        Get initial threshold value for an image.
        
        Args:
            image_name: Name of the image
            
        Returns:
            Initial threshold value
        """
        # Check if we have an existing threshold
        if image_name in self.thresholds.thresholds:
            return self.thresholds.thresholds[image_name]
        
        # Return default based on mode
        return 205  # Default for H&E mode
    
    def extract_region(
        self,
        image: np.ndarray,
        region: RegionSelection,
        resize_factor: float
    ) -> np.ndarray:
        """
        Extract a region from the image.
        
        Args:
            image: The full image
            region: Region selection
            resize_factor: Display resize factor
            
        Returns:
            Cropped region
        """
        return extract_cropped_region(image, region, resize_factor)
    
    def save_image_threshold(self, image_name: str, threshold: int, mode: ThresholdMode = None):
        """
        Save threshold value for an image.
        
        Args:
            image_name: Name of the image
            threshold: Threshold value
            mode: Threshold mode (optional, will use current mode if not specified)
        """
        self.thresholds.thresholds[image_name] = threshold
        if mode is not None:
            self.thresholds.mode = mode
        self.save_thresholds()
    
    def create_tissue_masks(self, mode: ThresholdMode = ThresholdMode.HE):
        """
        Create tissue masks for all images with thresholds.
        
        Args:
            mode: Threshold mode to use
        """
        # Create tissue masks for all images with saved thresholds
        import cv2
        from PIL import Image as PILImage
        import gc
        import platform
        
        logger.info("Creating tissue masks...")
        
        # Ensure output directory exists
        os.makedirs(self.config.output_path, exist_ok=True)
        
        # Process images in batches to prevent file handle exhaustion
        batch_size = 10  # Process 10 images at a time
        threshold_items = list(self.thresholds.thresholds.items())
        total_images = len(threshold_items)
        
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch = threshold_items[batch_start:batch_end]
            
            logger.debug(f"Processing batch {batch_start//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
            
            for image_name, threshold in batch:
                try:
                    # Find the full path for this image
                    full_path = None
                    base_name = os.path.splitext(image_name)[0]
                    
                    # Try different extensions and paths
                    for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
                        # Try training path
                        test_path = os.path.join(self.config.training_path, base_name + ext)
                        if os.path.exists(test_path):
                            full_path = test_path
                            break
                        # Try testing path
                        test_path = os.path.join(self.config.testing_path, base_name + ext)
                        if os.path.exists(test_path):
                            full_path = test_path
                            break
                    
                    if not full_path:
                        logger.warning(f"Could not find image file for {image_name}")
                        continue
                    
                    # Check if mask already exists
                    mask_path = os.path.join(self.config.output_path, f'{base_name}.tif')
                    if os.path.exists(mask_path) and not self.config.redo:
                        logger.info(f"Existing TA loaded: {mask_path}")
                        continue
                    
                    # Load image using fallback method
                    image = load_image_with_fallback(full_path)
                    if image is None:
                        logger.error(f"Failed to load image: {full_path}")
                        continue
                    
                    # Create tissue mask based on mode
                    if self.thresholds.mode == ThresholdMode.HE:
                        # In H&E mode, tissue is where green < threshold
                        tissue_mask = (image[:, :, 1] < threshold) * 255
                    else:
                        # In grayscale mode, tissue is where green > threshold
                        tissue_mask = (image[:, :, 1] > threshold) * 255
                    
                    # Save tissue mask
                    tissue_mask = tissue_mask.astype(np.uint8)
                    
                    # Handle Windows UNC paths for saving
                    if platform.system() == 'Windows' and mask_path.startswith('\\\\'):
                        try:
                            # Try direct file writing for UNC paths
                            import io
                            mask_image = PILImage.fromarray(tissue_mask, mode='L')
                            buffer = io.BytesIO()
                            mask_image.save(buffer, format='TIFF')
                            buffer.seek(0)
                            
                            with open(mask_path, 'wb') as f:
                                f.write(buffer.getvalue())
                            
                            del buffer
                            del mask_image
                        except Exception as save_error:
                            logger.error(f"Failed to save mask {mask_path}: {save_error}")
                            continue
                    else:
                        # Standard saving for non-UNC paths
                        try:
                            cv2.imwrite(mask_path, tissue_mask)
                        except Exception as cv_error:
                            logger.debug(f"OpenCV failed to save {mask_path}: {cv_error}. Trying PIL.")
                            # Fallback to PIL
                            mask_image = PILImage.fromarray(tissue_mask, mode='L')
                            mask_image.save(mask_path)
                            del mask_image
                    
                    logger.info(f"Saved tissue mask: {mask_path}")
                    
                    # Clean up memory
                    del image
                    del tissue_mask
                    
                except Exception as e:
                    logger.error(f"Failed to create tissue mask for {image_name}: {e}")
                    continue
            
            # Force garbage collection after each batch
            gc.collect()
            
            # Small delay to allow OS to reclaim file handles
            import time
            if batch_end < total_images:
                time.sleep(0.1)  # 100ms delay between batches
        
        logger.info("Tissue mask creation completed.")


def determine_optimal_TA_core(
    training_path: str,
    testing_path: str,
    num_images: int = 0,
    redo: bool = False,
    display_size: int = 600,
    sample_size: int = 20
) -> ImageThresholds:
    """
    Core function for determining tissue area thresholds (no GUI).
    
    This function can be used for automated processing or testing
    without any GUI dependencies.
    
    Args:
        training_path: Path to training images
        testing_path: Path to testing images
        num_images: Number of images to process
        redo: Whether to redo threshold selection
        display_size: Size of display region (for compatibility)
        sample_size: Number of images to sample
        
    Returns:
        ImageThresholds object with determined thresholds
    """
    # Create a compatible config object
    config = ThresholdConfig(
        training_path=training_path,
        testing_path=testing_path,
        num_images=num_images,
        redo=redo,
        region_size=display_size
    )
    
    selector = TissueAreaThresholdSelector(config)
    
    # In non-GUI mode, we can only work with existing thresholds
    # or apply default thresholds
    if not selector.thresholds.thresholds:
        # Apply default thresholds to all images
        # Get images from training path
        from glob import glob
        all_images = sorted(glob(os.path.join(training_path, '*.tif')))
        if not all_images:
            for ext in ['*.jpg', '*.png']:
                all_images.extend(glob(os.path.join(training_path, ext)))
        default_threshold = 205  # Default H&E threshold
        
        for image_path in all_images:
            image_name = os.path.basename(image_path)
            selector.save_image_threshold(image_name, default_threshold)
    
    # Create tissue masks
    selector.create_tissue_masks()
    
    return selector.thresholds