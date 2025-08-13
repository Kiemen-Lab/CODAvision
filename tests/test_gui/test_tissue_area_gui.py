"""
Test Suite for Tissue Area GUI Components

This module tests the GUI components for tissue area threshold selection,
ensuring proper functionality and interaction.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from PySide6 import QtWidgets, QtCore
from PySide6.QtTest import QTest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from base.tissue_area.models import ThresholdConfig, ThresholdMode, RegionSelection
from gui.tissue_area.threshold_gui import TissueAreaThresholdGUI, determine_optimal_TA_gui
from gui.tissue_area.dialogs import (
    ImageDisplayDialog, RegionCheckDialog, ThresholdSelectionDialog,
    ImageSelectionDialog
)


@pytest.fixture
def qt_app():
    """Create a Qt application for testing."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app
    app.quit()


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_dir = tempfile.mkdtemp()
    training_path = os.path.join(temp_dir, 'training')
    testing_path = os.path.join(temp_dir, 'testing')
    os.makedirs(training_path)
    os.makedirs(testing_path)
    
    yield training_path, testing_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestImageDisplayDialog:
    """Test the image display dialog."""
    
    def test_dialog_initialization(self, qt_app):
        """Test dialog initializes correctly."""
        dialog = ImageDisplayDialog((800, 600), 0.5)
        
        assert dialog.windowTitle() == "Click on a location at the edge of tissue and whitespace"
        assert dialog.clicked_position is None
        assert dialog.resize_factor == 0.5
        
        dialog.close()
    
    def test_image_update(self, qt_app):
        """Test image can be updated in dialog."""
        dialog = ImageDisplayDialog((100, 100), 1.0)
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        dialog.update_image(test_image)
        
        # Check that pixmap was set
        assert dialog.ui.whole_im.pixmap() is not None
        
        dialog.close()
    
    def test_mouse_click_capture(self, qt_app):
        """Test mouse click position is captured."""
        dialog = ImageDisplayDialog((100, 100), 1.0)
        
        # Simulate mouse click
        test_pos = QtCore.QPointF(50.0, 50.0)
        event = Mock()
        event.position.return_value = test_pos
        
        # Mock the geometry check
        dialog.ui.whole_im.geometry = Mock(return_value=QtCore.QRect(0, 0, 100, 100))
        dialog.ui.whole_im.geometry().contains = Mock(return_value=True)
        
        dialog.mousePressEvent(event)
        
        assert dialog.clicked_position is not None
        dialog.close()


class TestRegionCheckDialog:
    """Test the region check dialog."""
    
    def test_dialog_buttons(self, qt_app):
        """Test dialog button functionality."""
        dialog = RegionCheckDialog(600)
        
        # Test "looks good" button
        dialog.on_good()
        assert dialog.do_again == False
        
        # Reset and test "new location" button
        dialog.do_again = True
        dialog.on_new()
        assert dialog.do_again == True  # Should remain True
        
        dialog.close()
    
    def test_image_display(self, qt_app):
        """Test cropped image display."""
        dialog = RegionCheckDialog(100)
        
        # Create test cropped image
        cropped = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        dialog.update_image(cropped)
        
        # Check that pixmap was set
        assert dialog.ui.cropped.pixmap() is not None
        
        dialog.close()


class TestThresholdSelectionDialog:
    """Test the threshold selection dialog."""
    
    def test_initialization(self, qt_app):
        """Test dialog initialization with different modes."""
        # Test H&E mode
        dialog = ThresholdSelectionDialog(205, ThresholdMode.HE, 600)
        assert dialog.threshold == 205
        assert dialog.mode == ThresholdMode.HE
        assert dialog.ui.TA_selection.value() == 205
        dialog.close()
        
        # Test Grayscale mode
        dialog = ThresholdSelectionDialog(50, ThresholdMode.GRAYSCALE, 600)
        assert dialog.threshold == 50
        assert dialog.mode == ThresholdMode.GRAYSCALE
        assert dialog.ui.TA_selection.value() == 50
        dialog.close()
    
    def test_threshold_adjustment(self, qt_app):
        """Test threshold value adjustment."""
        dialog = ThresholdSelectionDialog(205, ThresholdMode.HE, 600)
        
        # Test raise button
        initial_value = dialog.threshold
        dialog.on_raise()
        assert dialog.threshold == initial_value + 1
        
        # Test decrease button
        dialog.on_decrease()
        assert dialog.threshold == initial_value
        
        # Test slider
        dialog.ui.TA_selection.setValue(220)
        dialog.update_slider()
        assert dialog.threshold == 220
        
        dialog.close()
    
    def test_mode_switching(self, qt_app):
        """Test switching between H&E and Grayscale modes."""
        dialog = ThresholdSelectionDialog(205, ThresholdMode.HE, 600)
        
        # Switch to Grayscale
        dialog.on_mode()
        assert dialog.mode == ThresholdMode.GRAYSCALE
        assert dialog.threshold == 50  # Default for grayscale
        
        # Switch back to H&E
        dialog.on_mode()
        assert dialog.mode == ThresholdMode.HE
        assert dialog.threshold == 205  # Default for H&E
        
        dialog.close()
    
    def test_apply_button(self, qt_app):
        """Test apply button functionality."""
        dialog = ThresholdSelectionDialog(205, ThresholdMode.HE, 600)
        
        assert dialog.stop == True
        dialog.on_apply()
        assert dialog.stop == False
        
        dialog.close()


class TestImageSelectionDialog:
    """Test the image selection dialog for redo mode."""
    
    def test_initialization(self, qt_app):
        """Test dialog initialization."""
        dialog = ImageSelectionDialog('/test/path')
        
        assert dialog.windowTitle() == "Confirm tissue mask evaluation"
        assert dialog.base_path == '/test/path'
        assert dialog.images == []
        assert dialog.apply_all == False
        
        dialog.close()
    
    def test_apply_all_button(self, qt_app):
        """Test apply all button."""
        dialog = ImageSelectionDialog('/test/path')
        
        dialog.on_apply_all()
        assert dialog.apply_all == True
        
        dialog.close()
    
    def test_image_list_management(self, qt_app):
        """Test adding and removing images from list."""
        dialog = ImageSelectionDialog('/test/path')
        
        # Add some images directly
        dialog.images = ['image1.tif', 'image2.tif']
        dialog.ui.image_LW.addItem('image1.tif')
        dialog.ui.image_LW.addItem('image2.tif')
        
        # Test delete functionality
        dialog.ui.image_LW.setCurrentRow(0)
        dialog.on_delete_image()
        
        assert len(dialog.images) == 1
        assert dialog.images[0] == 'image2.tif'
        
        dialog.close()


class TestTissueAreaThresholdGUI:
    """Test the main GUI wrapper class."""
    
    @pytest.fixture
    def config(self, temp_dirs):
        """Create test configuration."""
        training_path, testing_path = temp_dirs
        return ThresholdConfig(
            training_path=training_path,
            testing_path=testing_path,
            num_images=2,
            redo=False
        )
    
    def test_gui_initialization(self, config):
        """Test GUI wrapper initialization."""
        gui = TissueAreaThresholdGUI(config)
        
        assert gui.config == config
        assert gui.downsampled_path == config.training_path
        assert gui.selector is not None
        assert gui.test_ta_mode == ''
        assert gui.display_size == 600
    
    def test_gui_with_redo_mode(self, temp_dirs):
        """Test GUI in redo mode."""
        training_path, testing_path = temp_dirs
        config = ThresholdConfig(
            training_path=training_path,
            testing_path=testing_path,
            num_images=2,
            redo=True
        )
        
        gui = TissueAreaThresholdGUI(config)
        assert gui.test_ta_mode == 'redo'
    
    @patch('gui.tissue_area.threshold_gui.ImageDisplayDialog')
    @patch('gui.tissue_area.threshold_gui.RegionCheckDialog')
    @patch('gui.tissue_area.threshold_gui.ThresholdSelectionDialog')
    def test_threshold_selection_flow(self, mock_threshold, mock_check, mock_display, 
                                     config, qt_app):
        """Test the threshold selection flow with mocked dialogs."""
        gui = TissueAreaThresholdGUI(config)
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock dialog returns
        mock_display_instance = Mock()
        mock_display_instance.get_clicked_region.return_value = RegionSelection(50, 50, 600)
        mock_display_instance.isVisible.return_value = False  # Dialog is not visible (closes immediately)
        mock_display_instance.show = Mock()
        mock_display_instance.close = Mock()
        mock_display_instance.deleteLater = Mock()
        mock_display.return_value = mock_display_instance
        
        mock_check_instance = Mock()
        mock_check_instance.do_again = False
        mock_check_instance.isVisible.return_value = False
        mock_check_instance.show = Mock()
        mock_check_instance.close = Mock()
        mock_check_instance.deleteLater = Mock()
        mock_check.return_value = mock_check_instance
        
        mock_threshold_instance = Mock()
        mock_threshold_instance.stop = False
        mock_threshold_instance.threshold = 210
        mock_threshold_instance.mode = ThresholdMode.HE
        mock_threshold_instance.isVisible.return_value = False
        mock_threshold_instance.show = Mock()
        mock_threshold_instance.close = Mock()
        mock_threshold_instance.deleteLater = Mock()
        mock_threshold.return_value = mock_threshold_instance
        
        # Test threshold selection
        threshold = gui._get_threshold_with_gui(test_image, 1.0, 'test.tif')
        
        assert threshold == 210
        assert mock_display.called
        assert mock_check.called
        assert mock_threshold.called


class TestDetermineOptimalTAGui:
    """Test the main GUI entry point function."""
    
    @pytest.fixture
    def sample_setup(self, temp_dirs):
        """Create sample setup for testing."""
        training_path, testing_path = temp_dirs
        
        # Create dummy images
        for i in range(2):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = os.path.join(training_path, f'img_{i}.tif')
            import cv2
            cv2.imwrite(img_path, img)
        
        return training_path, testing_path
    
    @patch('gui.tissue_area.threshold_gui.TissueAreaThresholdGUI')
    def test_gui_entry_point(self, mock_gui_class, sample_setup):
        """Test the GUI entry point function."""
        training_path, testing_path = sample_setup
        
        # Mock the GUI instance
        mock_gui = Mock()
        mock_gui.run.return_value = True
        mock_gui.selector = Mock()
        mock_gui.selector.thresholds = Mock()
        mock_gui.selector.thresholds.thresholds = {'img1.tif': 205, 'img2.tif': 210}
        mock_gui_class.return_value = mock_gui
        
        # Call the function
        result = determine_optimal_TA_gui(
            training_path=training_path,
            testing_path=testing_path,
            num_images=2,
            redo=False
        )
        
        assert result == 2  # Two thresholds
        assert mock_gui_class.called
        assert mock_gui.run.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])