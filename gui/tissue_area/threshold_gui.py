"""
GUI Wrapper for Tissue Area Threshold Selection

This module provides the GUI interface for tissue area threshold selection,
separating the UI concerns from the core logic.
"""

import os
import sys
from typing import Optional, List, Tuple
from PySide6 import QtWidgets, QtCore
import numpy as np

from base.tissue_area.models import ThresholdConfig, ThresholdMode, RegionSelection
from base.tissue_area.threshold_core import TissueAreaThresholdSelector
from .dialogs import (
    ImageDisplayDialog, RegionCheckDialog, ThresholdSelectionDialog,
    ImageSelectionDialog
)

import logging
logger = logging.getLogger(__name__)


class TissueAreaThresholdGUI:
    """
    GUI wrapper for tissue area threshold selection.
    
    This class handles all GUI interactions while delegating core logic
    to the TissueAreaThresholdSelector.
    """
    
    def __init__(self, config: ThresholdConfig, downsampled_path: str = None):
        """
        Initialize the GUI wrapper.
        
        Args:
            config: Configuration for the threshold selection process
            downsampled_path: Path to downsampled images (optional)
        """
        self.config = config
        self.downsampled_path = downsampled_path
        self.selector = TissueAreaThresholdSelector(config, downsampled_path)
        self.app = None
        
        # Store additional parameters that GUI needs
        self.test_ta_mode = 'redo' if config.redo else ''
        self.display_size = config.region_size
        
    def _ensure_qt_app(self):
        """Ensure Qt application is available."""
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
    
    def _show_dialog_modal(self, dialog):
        """Show a dialog window and wait for it to close."""
        dialog.show()
        # Process events until the dialog is closed
        while dialog.isVisible():
            self.app.processEvents()
            QtCore.QThread.msleep(50)  # Small delay to prevent CPU spinning
        
        # Ensure the dialog is properly deleted
        dialog.deleteLater()
        self.app.processEvents()  # Process any pending deletion events
    
    def run(self) -> bool:
        """
        Run the threshold selection process with GUI.
        
        Returns:
            True if successful, False if cancelled
        """
        self._ensure_qt_app()
        print('Answer prompt pop-up window regarding tissue masks to proceed')
        
        # Check if we need to process
        if not self.selector.needs_processing():
            return True
        
        # Get list of images to process
        images_to_process = self._get_images_to_process_gui()
        if not images_to_process:
            return True
        
        # Process each image with GUI
        return self._process_images_with_gui(images_to_process)
    
    def _get_images_to_process_gui(self) -> List[str]:
        """Get list of images to process using GUI dialog."""
        # Get initial image list from selector
        initial_images = self.selector.get_images_to_process()
        
        if self.test_ta_mode == 'redo':
            # Show image selection dialog
            dialog = ImageSelectionDialog(self.downsampled_path)
            self._show_dialog_modal(dialog)
            
            # Get results before closing
            apply_all = dialog.apply_all
            images = dialog.images
            dialog.close()  # Ensure it's closed
            
            if apply_all:
                return initial_images
            elif images:
                return images
            else:
                return []
        
        return initial_images
    
    def _process_images_with_gui(self, images_to_process: List[str]) -> bool:
        """Process images with GUI interactions."""
        if not images_to_process:
            print("No images to process!")
            return True
        
        processed_count = 0
        for idx, image_path in enumerate(images_to_process):
            print(f"\nProcessing image {idx + 1}/{len(images_to_process)}: {image_path}")
            
            # Load and prepare image
            image_data = self.selector.prepare_image(image_path)
            if image_data is None:
                print(f"Failed to prepare image: {image_path}")
                # Save default threshold for failed images
                image_name = os.path.basename(image_path)
                self.selector.save_image_threshold(image_name, 205)  # Default H&E threshold
                continue
            
            # Get threshold via GUI
            threshold = self._get_threshold_with_gui(
                image_data['image'],
                image_data['resize_factor'],
                image_data['image_name']
            )
            
            if threshold is None:
                return False  # User cancelled
            
            # Save the threshold
            self.selector.save_image_threshold(image_data['image_name'], threshold)
            processed_count += 1
        
        # Create tissue masks even if some images failed
        print(f"\nProcessed {processed_count} images successfully")
        if processed_count > 0 or len(images_to_process) > 0:
            print("Creating tissue masks...")
            self.selector.create_tissue_masks()
        
        return True
    
    def _get_threshold_with_gui(
        self,
        image: np.ndarray,
        resize_factor: float,
        image_name: str
    ) -> Optional[int]:
        """
        Get threshold value using GUI dialogs.
        
        Args:
            image: The image to process
            resize_factor: Factor for display resizing
            image_name: Name of the image
            
        Returns:
            Selected threshold value or None if cancelled
        """
        # Get initial threshold
        threshold = self.selector.get_initial_threshold(image_name)
        mode = ThresholdMode.HE
        
        # Interactive region selection loop
        while True:
            # Display full image and get clicked region
            display_dialog = ImageDisplayDialog(image.shape, resize_factor)
            display_dialog.update_image(image)
            self._show_dialog_modal(display_dialog)
            
            region = display_dialog.get_clicked_region(self.display_size)
            display_dialog.close()  # Ensure it's closed
            
            if region is None:
                return None  # User cancelled
            
            # Extract and display cropped region
            cropped = self.selector.extract_region(image, region, resize_factor)
            
            # Check if region is good
            check_dialog = RegionCheckDialog(self.display_size)
            check_dialog.update_image(cropped)
            self._show_dialog_modal(check_dialog)
            
            do_again = check_dialog.do_again
            check_dialog.close()  # Ensure it's closed
            
            if not do_again:
                break
        
        # Select threshold value
        threshold_dialog = ThresholdSelectionDialog(
            threshold, mode, self.display_size
        )
        threshold_dialog.update_images(cropped)
        self._show_dialog_modal(threshold_dialog)
        
        # Get the result before the dialog is deleted
        if threshold_dialog.stop:
            result = None  # User cancelled
        else:
            result = threshold_dialog.threshold
        
        # Explicitly close and delete the dialog
        threshold_dialog.close()
        
        return result


def determine_optimal_TA_gui(
    downsampled_path: str,
    output_path: str,
    test_ta_mode: str = '',
    display_size: int = 600,
    sample_size: int = 20
) -> int:
    """
    GUI entry point for determining optimal tissue area threshold.
    
    This function maintains the original interface while using the new
    decoupled architecture.
    
    Args:
        downsampled_path: Path to downsampled images
        output_path: Path for output files
        test_ta_mode: Mode for testing ('redo' or '')
        display_size: Size of display region
        sample_size: Number of images to sample
        
    Returns:
        Number of thresholds determined
    """
    # Create a compatible config object
    # Note: ThresholdConfig expects training_path, testing_path, num_images, redo
    import os
    config = ThresholdConfig(
        training_path=os.path.dirname(downsampled_path),  # Parent directory
        testing_path=output_path,  # Using output_path as testing_path
        num_images=sample_size,
        redo=(test_ta_mode == 'redo'),
        region_size=display_size
    )
    
    gui = TissueAreaThresholdGUI(config, downsampled_path)
    success = gui.run()
    
    if success:
        return len(gui.selector.thresholds.thresholds)
    else:
        return 0