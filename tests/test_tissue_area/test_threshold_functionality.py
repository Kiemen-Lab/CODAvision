"""
Test Suite for Tissue Area Threshold Functionality

This module tests the core tissue area threshold detection functionality,
ensuring that the refactored code maintains backward compatibility.
"""

import os
import sys
import pytest
import numpy as np
import pickle
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from base.tissue_area import determine_optimal_TA
from base.tissue_area.models import ThresholdConfig, ThresholdMode, ImageThresholds
from base.tissue_area.threshold_core import TissueAreaThresholdSelector
from base.tissue_area.utils import create_tissue_mask, calculate_tissue_mask


class TestThresholdBackwardCompatibility:
    """Test backward compatibility with original determine_optimal_TA function."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        training_path = os.path.join(temp_dir, 'training')
        testing_path = os.path.join(temp_dir, 'testing')
        os.makedirs(training_path)
        os.makedirs(testing_path)
        
        yield training_path, testing_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_images(self, temp_dirs):
        """Create sample test images."""
        training_path, testing_path = temp_dirs
        
        # Create dummy images
        for i in range(3):
            # Training images
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = os.path.join(training_path, f'train_{i}.tif')
            import cv2
            cv2.imwrite(img_path, img)
            
            # Testing images
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = os.path.join(testing_path, f'test_{i}.tif')
            cv2.imwrite(img_path, img)
        
        return training_path, testing_path
    
    @patch('base.tissue_area.threshold._check_gui_available')
    def test_old_calling_convention(self, mock_gui_check, sample_images):
        """Test that old 4-argument calling convention still works."""
        # Force non-GUI mode for testing
        mock_gui_check.return_value = False
        
        training_path, testing_path = sample_images
        
        # Test with old-style arguments (pthim, pthtestim, nTA, redo)
        result = determine_optimal_TA(training_path, testing_path, 2, False)
        
        # Should return an integer (number of thresholds)
        assert isinstance(result, int)
        assert result >= 0
        
        # Check that TA directory was created
        ta_dir = os.path.join(training_path, 'TA')
        assert os.path.exists(ta_dir)
        
        # Check that TA_cutoff.pkl was created
        cutoff_file = os.path.join(ta_dir, 'TA_cutoff.pkl')
        assert os.path.exists(cutoff_file)
    
    @patch('base.tissue_area.threshold._check_gui_available')
    def test_threshold_file_structure(self, mock_gui_check, sample_images):
        """Test that saved threshold file has correct structure."""
        # Force non-GUI mode for testing
        mock_gui_check.return_value = False
        
        training_path, testing_path = sample_images
        
        # Run threshold selection
        determine_optimal_TA(training_path, testing_path, 0, False)
        
        # Load and check the saved file
        cutoff_file = os.path.join(training_path, 'TA', 'TA_cutoff.pkl')
        with open(cutoff_file, 'rb') as f:
            data = pickle.load(f)
        
        # Check required fields
        assert 'cts' in data  # thresholds dictionary
        assert 'mode' in data  # H&E or Grayscale
        assert 'imlist' in data  # list of processed images
        assert isinstance(data['cts'], dict)
        assert data['mode'] in ['H&E', 'Grayscale']
    
    def test_tissue_mask_creation(self, sample_images):
        """Test that tissue masks can be created correctly."""
        training_path, testing_path = sample_images
        
        # Create a test image
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        test_img[:50, :, 1] = 250  # Make top half bright in green channel
        
        # Test H&E mode (tissue is where green > threshold)
        mask_he = create_tissue_mask(test_img, 200, ThresholdMode.HE)
        assert mask_he[0, 0] == 255  # Top half is bright (> 200), is tissue/whitespace
        assert mask_he[75, 50] == 0  # Bottom half is dark (< 200), not tissue/whitespace
        
        # Test Grayscale mode (tissue is where green < threshold)  
        mask_gray = create_tissue_mask(test_img, 200, ThresholdMode.GRAYSCALE)
        assert mask_gray[0, 0] == 0  # Top half is bright (250 > 200), not tissue
        assert mask_gray[75, 50] == 255  # Bottom half is darker (128 < 200), is tissue


class TestThresholdCore:
    """Test core threshold selection functionality."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        training_path = os.path.join(temp_dir, 'training')
        testing_path = os.path.join(temp_dir, 'testing')
        os.makedirs(training_path)
        os.makedirs(testing_path)
        
        yield training_path, testing_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_dirs):
        """Create a test configuration."""
        training_path, testing_path = temp_dirs
        return ThresholdConfig(
            training_path=training_path,
            testing_path=testing_path,
            num_images=2,
            redo=False
        )
    
    def test_threshold_selector_initialization(self, config):
        """Test TissueAreaThresholdSelector initialization."""
        selector = TissueAreaThresholdSelector(config)
        
        assert selector.config == config
        assert selector.downsampled_path == config.training_path
        assert isinstance(selector.thresholds, ImageThresholds)
    
    def test_save_and_load_thresholds(self, config):
        """Test saving and loading threshold data."""
        selector = TissueAreaThresholdSelector(config)
        
        # Save some thresholds
        selector.save_image_threshold('test1.tif', 205)
        selector.save_image_threshold('test2.tif', 210)
        
        # Create new selector and check it loads saved data
        selector2 = TissueAreaThresholdSelector(config)
        assert 'test1.tif' in selector2.thresholds.thresholds
        assert selector2.thresholds.thresholds['test1.tif'] == 205
        assert 'test2.tif' in selector2.thresholds.thresholds
        assert selector2.thresholds.thresholds['test2.tif'] == 210
    
    def test_image_list_includes_both_paths(self, config, temp_dirs):
        """Test that image list includes images from both training and testing paths."""
        training_path, testing_path = temp_dirs
        
        # Create test images first
        for i in range(3):
            # Training images
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = os.path.join(training_path, f'train_{i}.tif')
            import cv2
            cv2.imwrite(img_path, img)
            
            # Testing images
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = os.path.join(testing_path, f'test_{i}.tif')
            cv2.imwrite(img_path, img)
        
        selector = TissueAreaThresholdSelector(config)
        images = selector._get_local_images()
        
        # Should have images from both directories
        assert len(images) == 6  # 3 training + 3 testing
        
        # Training images should be basenames only
        assert 'train_0.tif' in images
        assert 'train_1.tif' in images
        assert 'train_2.tif' in images
        
        # Testing images should also be basenames (consistent with training)
        assert 'test_0.tif' in images
        assert 'test_1.tif' in images
        assert 'test_2.tif' in images


class TestCalculateTissueMask:
    """Test the calculate_tissue_mask utility function."""
    
    @pytest.fixture
    def test_image_dir(self):
        """Create a test directory with an image."""
        temp_dir = tempfile.mkdtemp()
        
        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img[:50, :, 1] = 250  # Make top half bright in green channel
        
        img_path = os.path.join(temp_dir, 'test_image.tif')
        import cv2
        cv2.imwrite(img_path, img)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_calculate_tissue_mask_creates_ta_directory(self, test_image_dir):
        """Test that calculate_tissue_mask creates TA directory."""
        image, mask, output_path = calculate_tissue_mask(
            test_image_dir, 'test_image', test=False
        )
        
        # Check TA directory was created
        assert output_path == os.path.join(test_image_dir, 'TA')
        assert os.path.exists(output_path)
        
        # Check mask was saved
        mask_path = os.path.join(output_path, 'test_image.tif')
        assert os.path.exists(mask_path)
    
    def test_calculate_tissue_mask_uses_saved_threshold(self, test_image_dir):
        """Test that calculate_tissue_mask uses saved threshold values."""
        # Create TA directory and threshold file
        ta_dir = os.path.join(test_image_dir, 'TA')
        os.makedirs(ta_dir)
        
        # Save threshold data
        threshold_data = {
            'cts': {'test_image.tif': 180},
            'mode': 'H&E',
            'average_TA': False
        }
        with open(os.path.join(ta_dir, 'TA_cutoff.pkl'), 'wb') as f:
            pickle.dump(threshold_data, f)
        
        # Calculate mask
        image, mask, output_path = calculate_tissue_mask(
            test_image_dir, 'test_image', test=False
        )
        
        # Mask should use the saved threshold of 180
        # In H&E mode, tissue is where green < threshold
        # Top half has green=250 (> 180), so not tissue (black)
        # Bottom half has green=128 (< 180), so is tissue (white)
        assert mask[25, 50] == 0  # Top half
        assert mask[75, 50] == 255  # Bottom half


class TestLiverDataIntegration:
    """Test with liver tissue dataset settings."""
    
    def test_liver_data_configuration(self):
        """Test that the liver data configuration can be created."""
        colormap = np.array([
            [64, 128, 128],   # PDAC
            [255, 255, 0],    # bile_duct
            [255, 0, 0],      # vasculature
            [255, 0, 255],    # hepatocyte
            [0, 0, 0],        # immune
            [255, 128, 64],   # stroma
            [0, 0, 255]       # whitespace
        ])
        
        classnames = ['PDAC', 'bile_duct', 'vasculature', 'hepatocyte', 
                      'immune', 'stroma', 'whitespace']
        
        ws = [[0, 0, 0, 0, 2, 0, 2], [7, 6], [1, 2, 3, 4, 5, 6, 7], 
              [6, 4, 2, 3, 5, 1, 7], []]
        
        # Verify configuration is valid
        assert len(classnames) == 7
        assert colormap.shape == (7, 3)
        assert len(ws) == 5
        
        # Model configuration for liver dataset
        model_config = {
            'pthim': '/Users/tnewton3/Desktop/liver_tissue_data',
            'pthtest': '/Users/tnewton3/Desktop/liver_tissue_data/testing_image',
            'nm': '06_12_2025',
            'umpix': 1,  # 10x resolution
            'classNames': classnames,
            'cmap': colormap,
            'WS': ws
        }
        
        # This configuration should be loadable
        assert model_config['umpix'] == 1  # 10x resolution
        assert model_config['nm'] == '06_12_2025'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])