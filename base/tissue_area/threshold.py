"""
Tissue Area Threshold Selection Core Logic

This module provides the main functionality for determining optimal tissue area
thresholds in histological images.
"""

import os
import sys
import pickle
import shutil
import numpy as np
from typing import Optional, Tuple, List
from PySide6 import QtWidgets

from base.image import load_image_with_fallback
from .models import (
    ThresholdConfig, ThresholdMode, ImageThresholds, RegionSelection
)
from .utils import (
    load_images_list, select_random_images, calculate_resize_factor,
    extract_cropped_region, create_tissue_mask
)
from .gui.dialogs import (
    ImageDisplayDialog, RegionCheckDialog, ThresholdSelectionDialog,
    ImageSelectionDialog
)

import logging
logger = logging.getLogger(__name__)


class TissueAreaThresholdSelector:
    """
    Main class for tissue area threshold selection.
    
    This class orchestrates the process of determining optimal thresholds
    for distinguishing tissue from whitespace in histological images.
    """
    
    def __init__(self, config: ThresholdConfig):
        """
        Initialize the threshold selector.
        
        Args:
            config: Configuration for the threshold selection process
        """
        self.config = config
        self.thresholds = self._load_existing_thresholds()
        self.app = None
        
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
    
    def _save_thresholds(self):
        """Save threshold data to file."""
        os.makedirs(self.config.output_path, exist_ok=True)
        
        with open(self.config.threshold_file_path, 'wb') as f:
            pickle.dump(self.thresholds.to_dict(), f)
    
    def _ensure_qt_app(self):
        """Ensure Qt application is available."""
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
    
    def run(self) -> bool:
        """
        Run the threshold selection process.
        
        Returns:
            True if successful, False if cancelled
        """
        print('Answer prompt pop-up window regarding tissue masks to proceed')
        
        # Check if we need to process
        if not self._needs_processing():
            return True
        
        # Get list of images to process
        images_to_process = self._get_images_to_process()
        if not images_to_process:
            return True
        
        # Process each image
        return self._process_images(images_to_process)
    
    def _needs_processing(self) -> bool:
        """Check if processing is needed."""
        # If no existing thresholds, we need to process
        if not self.thresholds.thresholds:
            return True
        
        # Load all images
        all_images = load_images_list(
            self.config.training_images_path,
            self.config.testing_images_path
        )
        
        # Check if we have thresholds for all images
        if self.config.num_images == 0:
            # Processing all images
            unprocessed = set(all_images) - set(self.thresholds.thresholds.keys())
            if not unprocessed and not self.config.redo:
                logger.info('   Optimal cutoff already chosen for all images, skip this step')
                return False
        else:
            # Processing subset
            if self.thresholds.thresholds and not self.config.redo:
                logger.info('   Optimal cutoff already chosen, skip this step')
                return False
        
        return True
    
    def _get_images_to_process(self) -> List[str]:
        """Get list of images to process."""
        all_images = load_images_list(
            self.config.training_images_path,
            self.config.testing_images_path
        )
        
        if not all_images:
            return []
        
        # Handle re-evaluation
        if self.config.redo and self.thresholds.thresholds:
            if self.config.num_images == 0:
                # Ask which images to re-evaluate
                self._ensure_qt_app()
                dialog = ImageSelectionDialog(self.config.training_images_path)
                dialog.show()
                self.app.exec()
                
                if dialog.apply_all:
                    return all_images
                else:
                    return dialog.images if dialog.images else []
        
        # Handle normal processing
        if self.config.num_images > 0:
            # Select random subset
            num_to_select = min(self.config.num_images, len(all_images))
            selected = select_random_images(all_images, num_to_select)
            logger.info(f'Evaluating {num_to_select} randomly selected images to choose a good whitespace detection...')
            self.thresholds.average_threshold = True
            return selected
        else:
            # Process all images
            unprocessed = set(all_images) - set(self.thresholds.thresholds.keys())
            if unprocessed:
                logger.info(f'Evaluating all training images to choose a good whitespace detection...')
                self.thresholds.average_threshold = False
                return list(unprocessed)
            else:
                return all_images
    
    def _process_images(self, images: List[str]) -> bool:
        """
        Process a list of images to determine thresholds.
        
        Args:
            images: List of image paths to process
            
        Returns:
            True if successful, False if cancelled
        """
        self._ensure_qt_app()
        
        count = 0
        total = len(images)
        
        for image_path in images:
            count += 1
            
            # Load image
            logger.info(f'    Loading image {count} of {total}: {os.path.basename(image_path)}')
            image = self._load_image(image_path)
            if image is None:
                continue
            
            logger.info('     Image loaded')
            
            # Process image to get threshold
            threshold = self._process_single_image(image)
            if threshold is None:
                logger.info('Whitespace detection process stopped by the user')
                return False
            
            # Save threshold
            image_name = os.path.basename(image_path)
            self.thresholds.set_threshold(image_name, threshold)
        
        # Save results
        self._save_thresholds()
        return True
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load an image from path."""
        # Handle test images with full path
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return load_image_with_fallback(image_path)
        
        # Handle training images with relative path
        full_path = os.path.join(self.config.training_images_path, image_path)
        if os.path.exists(full_path):
            return load_image_with_fallback(full_path)
        
        logger.error(f"Could not find image: {image_path}")
        return None
    
    def _process_single_image(self, image: np.ndarray) -> Optional[int]:
        """
        Process a single image to determine its threshold.
        
        Args:
            image: Image array
            
        Returns:
            Threshold value or None if cancelled
        """
        resize_factor = calculate_resize_factor(image.shape)
        current_threshold = self.thresholds.mode.default_threshold
        
        while True:
            # Get region selection from user
            region = self._get_region_selection(image, resize_factor)
            if region is None:
                return None
            
            # Extract cropped region
            cropped = extract_cropped_region(image, region, resize_factor)
            
            # Check if region is acceptable
            if self._check_region(cropped):
                break
        
        # Select threshold for the region
        threshold, mode = self._select_threshold(cropped, current_threshold)
        if threshold is None:
            return None
        
        # Update mode if changed
        if mode != self.thresholds.mode:
            self.thresholds.mode = mode
        
        return threshold
    
    def _get_region_selection(self, image: np.ndarray, 
                            resize_factor: float) -> Optional[RegionSelection]:
        """Get region selection from user."""
        dialog = ImageDisplayDialog(image.shape, resize_factor)
        dialog.show()
        dialog.update_image(image)
        self.app.exec()
        
        return dialog.get_clicked_region(self.config.region_size)
    
    def _check_region(self, cropped: np.ndarray) -> bool:
        """Check if the selected region is acceptable."""
        dialog = RegionCheckDialog(self.config.region_size)
        dialog.show()
        dialog.update_image(cropped)
        self.app.exec()
        
        return not dialog.do_again
    
    def _select_threshold(self, cropped: np.ndarray, 
                         initial_threshold: int) -> Tuple[Optional[int], Optional[ThresholdMode]]:
        """
        Select threshold for the cropped region.
        
        Returns:
            Tuple of (threshold, mode) or (None, None) if cancelled
        """
        dialog = ThresholdSelectionDialog(
            initial_threshold, 
            self.thresholds.mode,
            self.config.region_size
        )
        dialog.show()
        dialog.update_images(cropped)
        self.app.exec()
        
        if dialog.stop:
            return None, None
        
        return dialog.threshold, dialog.mode


def determine_optimal_TA(pthim: str, pthtestim: str, numims: int, redo: bool):
    """
    Determine optimal tissue area threshold.
    
    This is a backward-compatible wrapper for the refactored functionality.
    
    Args:
        pthim: Path to training images
        pthtestim: Path to testing images
        numims: Number of images to process (0 for all)
        redo: Whether to redo existing thresholds
    """
    config = ThresholdConfig(
        training_path=pthim,
        testing_path=pthtestim,
        num_images=numims,
        redo=redo
    )
    
    selector = TissueAreaThresholdSelector(config)
    selector.run()
    
    # Copy threshold file to test directory
    try:
        test_ta_path = os.path.join(pthtestim, 'TA')
        os.makedirs(test_ta_path, exist_ok=True)
        shutil.copy(
            config.threshold_file_path,
            os.path.join(test_ta_path, 'TA_cutoff.pkl')
        )
    except Exception as e:
        logger.error(f'Failed to copy TA cutoff file: {e}')