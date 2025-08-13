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
    ImageSelectionDialog, TissueMaskExistsDialog
)

import logging
logger = logging.getLogger(__name__)


class TissueAreaThresholdGUI:
    """
    GUI wrapper for tissue area threshold selection.
    
    This class handles all GUI interactions while delegating core logic
    to the TissueAreaThresholdSelector.
    """
    
    def __init__(self, config: ThresholdConfig):
        """
        Initialize the GUI wrapper.
        
        Args:
            config: Configuration for the threshold selection process
        """
        self.config = config
        self.downsampled_path = config.training_path  # Use training path for images
        self.selector = TissueAreaThresholdSelector(config)
        self.app = None
        
        # Store additional parameters that GUI needs
        self.test_ta_mode = 'redo' if config.redo else ''
        self.display_size = config.region_size
        
        # Initialize mode from existing thresholds if available
        self._last_mode = self.selector.thresholds.mode if self.selector.thresholds else ThresholdMode.HE
        
    def _ensure_qt_app(self):
        """Ensure Qt application is available."""
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
    
    def _show_dialog_modal(self, dialog):
        """Show a dialog window and wait for it to close."""
        # Check if dialog has finished signal (QDialog)
        if hasattr(dialog, 'finished'):
            # Create an event loop for proper modal dialog handling
            event_loop = QtCore.QEventLoop()
            
            # Connect dialog close event to event loop quit
            dialog.finished.connect(event_loop.quit)
            
            # Show dialog as modal
            dialog.setWindowModality(QtCore.Qt.ApplicationModal)
            dialog.show()
            
            # Execute event loop - blocks until dialog is closed
            event_loop.exec_()
        else:
            # Fall back to original implementation for custom dialogs
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
        logger.info('Opening tissue mask selection dialog - user interaction required')
        
        # Check if evaluation already exists and not in explicit redo mode
        if self.selector.has_existing_evaluation() and not self.config.redo:
            # Check if existing evaluation is compatible with current mode
            if not self.selector.is_evaluation_compatible(self._last_mode):
                logger.warning(f"Existing evaluation mode ({self.selector.thresholds.mode.value}) "
                             f"doesn't match expected mode ({self._last_mode.value})")
                # Force redo if modes don't match
                self.config.redo = True
                self.test_ta_mode = 'redo'
            else:
                # Show dialog asking if user wants to keep current evaluation
                dialog = TissueMaskExistsDialog()
                self._show_dialog_modal(dialog)
                
                keep_current = dialog.keep_current
                dialog.close()
                
                if keep_current:
                    logger.info('User chose to keep current tissue mask evaluation')
                    return True
                else:
                    # User wants to redo - update the config
                    self.config.redo = True
                    self.test_ta_mode = 'redo'
        
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
            logger.info("No images to process!")
            return True
        
        processed_count = 0
        for idx, image_path in enumerate(images_to_process):
            logger.info(f"Processing image {idx + 1}/{len(images_to_process)}: {image_path}")
            
            # Load and prepare image
            image_data = self.selector.prepare_image(image_path)
            if image_data is None:
                logger.error(f"Failed to prepare image: {image_path}")
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
            
            # Get the mode from the last threshold dialog
            mode = getattr(self, '_last_mode', ThresholdMode.HE)
            
            # Save the threshold with mode
            self.selector.save_image_threshold(image_data['image_name'], threshold, mode)
            processed_count += 1
        
        # Create tissue masks even if some images failed
        logger.info(f"Processed {processed_count} images successfully")
        if processed_count > 0 or len(images_to_process) > 0:
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
        # Ensure Qt application is available
        self._ensure_qt_app()
        
        # Get initial threshold
        threshold = self.selector.get_initial_threshold(image_name)
        mode = self._last_mode  # Use the current mode instead of always defaulting to H&E
        
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
        
        # Get the result and mode before the dialog is deleted
        if threshold_dialog.stop:
            result = None  # User cancelled
        else:
            result = threshold_dialog.threshold
            # Store the mode for later use
            self._last_mode = threshold_dialog.mode
        
        # Explicitly close and delete the dialog
        threshold_dialog.close()
        
        return result


def determine_optimal_TA_gui(
    training_path: str,
    testing_path: str,
    num_images: int = 0,
    redo: bool = False,
    display_size: int = 600,
    sample_size: int = 20
) -> int:
    """
    GUI entry point for determining optimal tissue area threshold.
    
    This function maintains the original interface while using the new
    decoupled architecture.
    
    Args:
        training_path: Path to training images
        testing_path: Path to testing images
        num_images: Number of images to process for threshold selection
        redo: Whether to redo threshold selection
        display_size: Size of display region
        sample_size: Number of images to sample
        
    Returns:
        Number of thresholds determined
    """
    # Create a compatible config object
    config = ThresholdConfig(
        training_path=training_path,
        testing_path=testing_path,
        num_images=num_images,
        redo=redo,
        region_size=display_size
    )
    
    gui = TissueAreaThresholdGUI(config)
    success = gui.run()
    
    if success:
        return len(gui.selector.thresholds.thresholds)
    else:
        return 0