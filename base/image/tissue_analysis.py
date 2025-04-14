"""
Tissue Analysis Utilities for CODAvision

This module provides functionality for analyzing tissue areas in histological images,
including automatic and interactive threshold determination for tissue detection.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: April 2025
"""

import os
import pickle
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from skimage import morphology
from skimage.morphology import remove_small_objects


class TissueAnalyzer:
    """
    Class for analyzing and optimizing tissue detection parameters.
    
    This class provides methods for determining optimal thresholds 
    for distinguishing tissue from background in histological images.
    """
    
    def __init__(self, image_path: str, verbose: bool = True):
        """
        Initialize the TissueAnalyzer with an image directory.
        
        Args:
            image_path: Path to directory containing images to analyze
            verbose: Whether to print status messages during analysis
        """
        self.image_path = image_path
        self.verbose = verbose
        self.output_path = os.path.join(image_path, 'TA')
        os.makedirs(self.output_path, exist_ok=True)
        
    def load_images(self) -> List[str]:
        """
        Load image file paths from the specified directory.
        
        Returns:
            List of image filenames in the directory
        """
        image_list = [f for f in os.listdir(self.image_path) if f.endswith('.tif')]
        
        if not image_list:
            jpg_files = [f for f in os.listdir(self.image_path) if f.endswith('.jpg')]
            png_files = [f for f in os.listdir(self.image_path) if f.endswith('.png')]
            
            if jpg_files:
                image_list.extend(jpg_files)
            if png_files:
                image_list.extend(png_files)
                
        if not image_list and self.verbose:
            print(f"No TIFF, PNG or JPG image files found in {self.image_path}")
            
        return image_list
    
    def load_thresholds(self) -> Tuple[Dict[str, float], str, bool]:
        """
        Load existing tissue threshold values if available.
        
        Returns:
            Tuple containing:
            - Dictionary of thresholds for each image
            - Analysis mode ('H&E' or 'Grayscale')
            - Whether to use average threshold across images
        """
        threshold_file = os.path.join(self.output_path, 'TA_cutoff.pkl')
        
        if os.path.isfile(threshold_file):
            with open(threshold_file, 'rb') as f:
                data = pickle.load(f)
                thresholds = data.get('cts', {})
                mode = data.get('mode', 'H&E')
                average_ta = data.get('average_TA', False)
                return thresholds, mode, average_ta
        
        return {}, 'H&E', False
        
    def save_thresholds(self, thresholds: Dict[str, float], image_list: List[str], 
                       mode: str, average_ta: bool) -> None:
        """
        Save tissue analysis thresholds to a pickle file.
        
        Args:
            thresholds: Dictionary of threshold values for each image
            image_list: List of analyzed images
            mode: Analysis mode ('H&E' or 'Grayscale')
            average_ta: Whether to use average threshold across images
        """
        with open(os.path.join(self.output_path, 'TA_cutoff.pkl'), 'wb') as f:
            pickle.dump({
                'cts': thresholds,
                'imlist': image_list, 
                'mode': mode, 
                'average_TA': average_ta
            }, f)
    
    def generate_tissue_mask(self, image: np.ndarray, threshold: float, mode: str = 'H&E') -> np.ndarray:
        """
        Generate a binary tissue mask using the specified threshold.
        
        Args:
            image: Input image array (RGB)
            threshold: Intensity threshold value
            mode: Analysis mode ('H&E' or 'Grayscale')
            
        Returns:
            Binary mask with tissue areas marked
        """
        if mode == 'H&E':
            # For H&E stained images, use green channel with values below threshold
            tissue_mask = image[:, :, 1] < threshold
        else:
            # For grayscale, use green channel with values above threshold
            tissue_mask = image[:, :, 1] > threshold
            
        # Clean up the mask with morphological operations
        kernel_size = 3
        tissue_mask = tissue_mask.astype(np.uint8)
        kernel = morphology.disk(kernel_size)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel.astype(np.uint8))
        tissue_mask = remove_small_objects(tissue_mask.astype(bool), min_size=10)
        
        return tissue_mask
    
    def calculate_average_threshold(self, thresholds: Dict[str, float], default: float = 205) -> float:
        """
        Calculate the average threshold from existing values.
        
        Args:
            thresholds: Dictionary of threshold values
            default: Default threshold value if no thresholds exist
            
        Returns:
            Average threshold value
        """
        if not thresholds:
            return default
            
        return sum(thresholds.values()) / len(thresholds)


def determine_optimal_TA(image_path: str, num_images: int = 0) -> None:
    """
    Determine the optimal tissue analysis threshold for images.
    
    This function provides a GUI-based workflow for selecting appropriate
    thresholds to distinguish tissue from background in histological images.
    
    Args:
        image_path: Path to the directory containing images
        num_images: Number of images to analyze (0 for all)
    """
    # Import here to avoid circular imports
    from gui.components.tissue_analysis_dialog import TissueAnalysisController
    
    # Run the interactive tissue analysis process
    controller = TissueAnalysisController(image_path, num_images)
    controller.run()