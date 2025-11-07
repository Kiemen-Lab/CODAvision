"""
Integration Test Suite for Complete Tissue Area Workflow

This module tests the complete tissue area threshold workflow including
GUI interaction, file I/O, and training pipeline integration.
"""

import os
import sys
import pytest
import numpy as np
import pickle
import tempfile
import shutil
from pathlib import Path
import cv2
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from base import determine_optimal_TA
from base.tissue_area.utils import calculate_tissue_mask
from base.tissue_area.models import ThresholdConfig, ImageThresholds


class TestCompleteWorkflow:
    """Test the complete tissue area workflow from end to end."""
    
    @pytest.fixture
    def liver_data_setup(self):
        """Create a mock liver tissue data setup."""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        training_path = os.path.join(temp_dir, 'liver_tissue_data')
        testing_path = os.path.join(temp_dir, 'liver_tissue_data', 'testing_image')
        os.makedirs(training_path)
        os.makedirs(testing_path)
        
        # Create sample images with realistic properties
        # Training images - varying green channel values to test thresholding
        for i in range(3):
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            # Create regions with different intensities
            img[:250, :, 1] = 230  # Bright green (whitespace in H&E)
            img[250:, :, 1] = 180   # Darker green (tissue in H&E)
            # Add some variation
            noise = np.random.randint(-10, 10, (500, 500))
            img[:, :, 1] = np.clip(img[:, :, 1] + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(training_path, f'liver_train_{i}.tif'), img)
        
        # Testing image
        test_img = np.zeros((500, 500, 3), dtype=np.uint8)
        test_img[:300, :, 1] = 220  # Mostly bright
        test_img[300:, :, 1] = 190   # Some tissue
        cv2.imwrite(os.path.join(testing_path, 'liver_test.tif'), test_img)
        
        yield training_path, testing_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('base.tissue_area.threshold._check_gui_available')
    def test_full_workflow_no_gui(self, mock_gui_check, liver_data_setup):
        """Test the full workflow without GUI (using defaults)."""
        # Force non-GUI mode
        mock_gui_check.return_value = False
        
        training_path, testing_path = liver_data_setup
        
        # Step 1: Run tissue area threshold determination
        result = determine_optimal_TA(training_path, testing_path, 0, False)
        
        # Verify outputs
        assert isinstance(result, int)
        assert result > 0
        
        # Check TA directory structure
        ta_dir = os.path.join(training_path, 'TA')
        assert os.path.exists(ta_dir)
        
        # Check threshold file
        threshold_file = os.path.join(ta_dir, 'TA_cutoff.pkl')
        assert os.path.exists(threshold_file)
        
        # Load and verify threshold data
        with open(threshold_file, 'rb') as f:
            data = pickle.load(f)
        
        assert 'cts' in data
        assert 'mode' in data
        assert 'imlist' in data
        assert len(data['cts']) > 0
        
        # Step 2: Test tissue mask creation
        image, mask, output_path = calculate_tissue_mask(
            training_path, 'liver_train_0', test=False
        )
        
        # Verify mask was created
        mask_file = os.path.join(ta_dir, 'liver_train_0.tif')
        assert os.path.exists(mask_file)
        
        # Verify mask properties
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.uint8
        assert np.unique(mask).tolist() in [[0, 255], [0], [255]]
    
    @patch('base.tissue_area.threshold._check_gui_available')
    def test_workflow_with_redo(self, mock_gui_check, liver_data_setup):
        """Test workflow with redo functionality."""
        # Force non-GUI mode
        mock_gui_check.return_value = False
        
        training_path, testing_path = liver_data_setup
        
        # First run
        determine_optimal_TA(training_path, testing_path, 2, False)
        
        # Get initial threshold values
        with open(os.path.join(training_path, 'TA', 'TA_cutoff.pkl'), 'rb') as f:
            initial_data = pickle.load(f)
        
        # Redo with different parameters (this would use GUI in real scenario)
        # For testing, we'll just run again with redo=True
        determine_optimal_TA(training_path, testing_path, 0, True)
        
        # Verify the file still exists
        assert os.path.exists(os.path.join(training_path, 'TA', 'TA_cutoff.pkl'))
    
    @patch('base.tissue_area.threshold._check_gui_available')
    def test_mixed_image_formats(self, mock_gui_check):
        """Test with mixed image formats (TIFF, JPG, PNG)."""
        # Force non-GUI mode
        mock_gui_check.return_value = False
        
        temp_dir = tempfile.mkdtemp()
        try:
            training_path = os.path.join(temp_dir, 'mixed_formats')
            testing_path = os.path.join(temp_dir, 'mixed_formats_test')
            os.makedirs(training_path)
            os.makedirs(testing_path)
            
            # Create images in different formats
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            cv2.imwrite(os.path.join(training_path, 'img1.tif'), test_img)
            cv2.imwrite(os.path.join(training_path, 'img2.jpg'), test_img)
            cv2.imwrite(os.path.join(training_path, 'img3.png'), test_img)
            cv2.imwrite(os.path.join(testing_path, 'test.tif'), test_img)
            
            # Run threshold determination
            result = determine_optimal_TA(training_path, testing_path, 0, False)
            
            assert result > 0
            
            # Check that all images were processed
            with open(os.path.join(training_path, 'TA', 'TA_cutoff.pkl'), 'rb') as f:
                data = pickle.load(f)
            
            # Should have thresholds for all images
            assert len(data['cts']) >= 3
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_model_metadata_compatibility(self, liver_data_setup):
        """Test compatibility with net.pkl loading."""
        training_path, testing_path = liver_data_setup
        
        # Create a mock net.pkl file
        model_data = {
            'pthim': os.path.join(training_path, '10x'),
            'pthtest': testing_path,
            'nm': '06_12_2025',
            'umpix': 1,  # 10x
            'WS': [[0, 0, 0, 0, 2, 0, 2], [7, 6], [1, 2, 3, 4, 5, 6, 7], 
                   [6, 4, 2, 3, 5, 1, 7], []],
            'cmap': np.array([
                [64, 128, 128],
                [255, 255, 0],
                [255, 0, 0],
                [255, 0, 255],
                [0, 0, 0],
                [255, 128, 64],
                [0, 0, 255]
            ]),
            'classNames': ['PDAC', 'bile_duct', 'vasculature', 'hepatocyte', 
                          'immune', 'stroma', 'whitespace'],
            'nwhite': 7,
            'scale': 1.0,
            'sxy': 512,
            'ntrain': 1000,
            'nvalidate': 200,
            'nTA': 5,
            'batch_size': 32,
            'model_type': 'DeepLabV3+',
            'final_df': None,  # Would contain pandas DataFrame
            'combined_df': None
        }
        
        # Save net.pkl
        model_dir = os.path.join(training_path, '06_12_2025')
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'net.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        
        # Verify it can be loaded
        with open(os.path.join(model_dir, 'net.pkl'), 'rb') as f:
            loaded_data = pickle.load(f)
        
        assert loaded_data['nm'] == '06_12_2025'
        assert loaded_data['umpix'] == 1
        assert len(loaded_data['classNames']) == 7
    
    @patch('base.tissue_area.threshold._check_gui_available')
    def test_ta_file_copy_behavior(self, mock_gui_check, liver_data_setup):
        """Test that TA files are copied correctly between directories."""
        # Force non-GUI mode
        mock_gui_check.return_value = False
        
        training_path, testing_path = liver_data_setup
        
        # Run threshold determination
        determine_optimal_TA(training_path, testing_path, 0, False)
        
        # Check original TA file exists
        original_ta = os.path.join(training_path, 'TA', 'TA_cutoff.pkl')
        assert os.path.exists(original_ta)
        
        # Simulate copying TA file to testing directory (as done in gui/application.py)
        testing_ta_dir = os.path.join(testing_path, 'TA')
        os.makedirs(testing_ta_dir, exist_ok=True)
        
        testing_ta = os.path.join(testing_ta_dir, 'TA_cutoff.pkl')
        shutil.copy(original_ta, testing_ta)
        
        # Verify copy succeeded and files match
        assert os.path.exists(testing_ta)
        
        with open(original_ta, 'rb') as f:
            original_data = pickle.load(f)
        
        with open(testing_ta, 'rb') as f:
            copied_data = pickle.load(f)
        
        assert original_data['mode'] == copied_data['mode']
        assert original_data['cts'] == copied_data['cts']


class TestErrorHandling:
    """Test error handling in tissue area workflow."""
    
    @patch('base.tissue_area.threshold._check_gui_available')
    def test_empty_directory(self, mock_gui_check):
        """Test handling of empty directories."""
        # Force non-GUI mode
        mock_gui_check.return_value = False
        
        temp_dir = tempfile.mkdtemp()
        try:
            training_path = os.path.join(temp_dir, 'empty_train')
            testing_path = os.path.join(temp_dir, 'empty_test')
            os.makedirs(training_path)
            os.makedirs(testing_path)
            
            # Should handle empty directories gracefully
            result = determine_optimal_TA(training_path, testing_path, 0, False)
            
            # With no images, should still create TA directory
            assert os.path.exists(os.path.join(training_path, 'TA'))
            
        finally:
            shutil.rmtree(temp_dir)
    
    @patch('base.tissue_area.threshold._check_gui_available')
    def test_corrupted_threshold_file(self, mock_gui_check):
        """Test handling of corrupted threshold file."""
        # Force non-GUI mode
        mock_gui_check.return_value = False
        
        temp_dir = tempfile.mkdtemp()
        try:
            training_path = os.path.join(temp_dir, 'corrupted')
            testing_path = os.path.join(temp_dir, 'corrupted_test')
            os.makedirs(training_path)
            os.makedirs(testing_path)
            
            # Create TA directory with corrupted file
            ta_dir = os.path.join(training_path, 'TA')
            os.makedirs(ta_dir)
            with open(os.path.join(ta_dir, 'TA_cutoff.pkl'), 'w') as f:
                f.write("corrupted data")
            
            # Create a test image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(training_path, 'test.tif'), img)
            
            # Should handle corrupted file and create new one
            result = determine_optimal_TA(training_path, testing_path, 0, False)
            
            assert result >= 0
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])