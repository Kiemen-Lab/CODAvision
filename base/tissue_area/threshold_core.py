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
    
    def __init__(self, config: ThresholdConfig, downsampled_path: str = None):
        """
        Initialize the threshold selector.
        
        Args:
            config: Configuration for the threshold selection process
            downsampled_path: Path to downsampled images (optional)
        """
        self.config = config
        self.downsampled_path = downsampled_path or os.path.join(config.training_path, 'downsampled_tiles')
        self.thresholds = self._load_existing_thresholds()
        
    def _get_local_images(self) -> List[str]:
        """Get list of images from the downsampled path."""
        from glob import glob
        
        # Look for TIFF files first
        imlist = sorted(glob(os.path.join(self.downsampled_path, '*.tif')))
        
        # If no TIFF files, look for JPG and PNG
        if not imlist:
            for ext in ['*.jpg', '*.png']:
                imlist.extend(glob(os.path.join(self.downsampled_path, ext)))
        
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
            image_path: Path to the image
            
        Returns:
            Dictionary with image data or None if loading failed
        """
        try:
            # Load image
            image = load_image_with_fallback(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Calculate resize factor for display
            resize_factor = calculate_resize_factor(
                image.shape, 
                max_width=1500,
                max_height=780
            )
            
            return {
                'image': image,
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
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
    
    def save_image_threshold(self, image_name: str, threshold: int):
        """
        Save threshold value for an image.
        
        Args:
            image_name: Name of the image
            threshold: Threshold value
        """
        self.thresholds.thresholds[image_name] = threshold
        self.save_thresholds()
    
    def create_tissue_masks(self, mode: ThresholdMode = ThresholdMode.HE):
        """
        Create tissue masks for all images with thresholds.
        
        Args:
            mode: Threshold mode to use
        """
        for image_name, threshold in self.thresholds.thresholds.items():
            image_path = os.path.join(self.downsampled_path, image_name)
            
            # Load image
            image = load_image_with_fallback(image_path)
            if image is None:
                logger.warning(f"Failed to load image for mask creation: {image_path}")
                continue
            
            # Create mask
            mask = create_tissue_mask(image, threshold, mode)
            
            # Save mask
            mask_name = image_name.replace('.tif', '_tissue_mask.tif')
            mask_path = os.path.join(self.config.output_path, mask_name)
            
            try:
                from PIL import Image
                Image.fromarray(mask).save(mask_path)
                logger.info(f"Saved tissue mask: {mask_path}")
            except Exception as e:
                logger.error(f"Failed to save mask {mask_path}: {e}")


def determine_optimal_TA_core(
    downsampled_path: str,
    output_path: str,
    test_ta_mode: str = '',
    display_size: int = 600,
    sample_size: int = 20
) -> ImageThresholds:
    """
    Core function for determining tissue area thresholds (no GUI).
    
    This function can be used for automated processing or testing
    without any GUI dependencies.
    
    Args:
        downsampled_path: Path to downsampled images
        output_path: Path for output files
        test_ta_mode: Mode for testing
        display_size: Size of display region (for compatibility)
        sample_size: Number of images to sample
        
    Returns:
        ImageThresholds object with determined thresholds
    """
    # Create a compatible config object
    # Note: ThresholdConfig expects training_path, testing_path, num_images, redo
    # We need to adapt the parameters
    config = ThresholdConfig(
        training_path=os.path.dirname(downsampled_path),  # Parent directory
        testing_path=output_path,  # Using output_path as testing_path
        num_images=sample_size,
        redo=(test_ta_mode == 'redo'),
        region_size=display_size
    )
    
    selector = TissueAreaThresholdSelector(config, downsampled_path)
    
    # In non-GUI mode, we can only work with existing thresholds
    # or apply default thresholds
    if not selector.thresholds.thresholds:
        # Apply default thresholds to all images
        # Get images from downsampled path only
        from glob import glob
        all_images = sorted(glob(os.path.join(downsampled_path, '*.tif')))
        if not all_images:
            for ext in ['*.jpg', '*.png']:
                all_images.extend(glob(os.path.join(downsampled_path, ext)))
        default_threshold = 205  # Default H&E threshold
        
        for image_path in all_images:
            image_name = os.path.basename(image_path)
            selector.save_image_threshold(image_name, default_threshold)
    
    # Create tissue masks
    selector.create_tissue_masks()
    
    return selector.thresholds