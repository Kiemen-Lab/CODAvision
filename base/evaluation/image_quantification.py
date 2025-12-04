"""
Image Quantification Module for CODAvision

This module provides functionality for quantifying the tissue composition
of classified images, computing pixel counts and percentage compositions
for each tissue class.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from base.image.utils import load_image_with_fallback

# Set up logging
import logging
logger = logging.getLogger(__name__)


class ImageQuantifier:
    """
    A class for quantifying tissue composition in classified images.
    
    This class analyzes segmented images to count pixels of each tissue class
    and calculate tissue composition percentages. Results are saved to a CSV file.
    
    Attributes:
        model_path (str): Path to the directory containing model data
        image_path (str): Path to the directory containing images to quantify
        model_data (Dict): Dictionary containing model metadata
        class_names (List[str]): List of class names from the model
        nwhite (int): Index of the whitespace class
        output_path (str): Path where quantification results will be saved
    """
    
    def __init__(self, model_path: str, image_path: str):
        """
        Initialize the ImageQuantifier with model and image paths.
        
        Args:
            model_path: Path to directory containing model data
            image_path: Path to directory containing images to quantify
        
        Raises:
            FileNotFoundError: If model data file is not found
            ValueError: If required model data is missing
        """
        self.model_path = model_path
        self.image_path = image_path
        self.model_data = None
        self.class_names = None
        self.nwhite = None
        self.output_path = None
        
        self._load_model_data()
        self._setup_output_path()
    
    def _load_model_data(self) -> None:
        """
        Load model metadata from the pickle file.
        
        Raises:
            FileNotFoundError: If model data file doesn't exist
            ValueError: If required parameters are missing
        """
        model_data_file = os.path.join(self.model_path, 'net.pkl')
        
        if not os.path.exists(model_data_file):
            raise FileNotFoundError(f"Model data file not found: {model_data_file}")
        
        try:
            with open(model_data_file, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.class_names = self.model_data.get('classNames')
            self.nwhite = self.model_data.get('nwhite')
            self.model_name = self.model_data.get('nm')
            self.model_type = self.model_data.get('model_type')
            
            if None in (self.class_names, self.nwhite, self.model_name, self.model_type):
                raise ValueError("Missing required parameters in model data file")
                
        except Exception as e:
            raise ValueError(f"Failed to load model data: {e}")
    
    def _setup_output_path(self) -> None:
        """
        Set up the output path for quantification results.
        """
        self.output_path = os.path.join(
            self.image_path, 
            f"classification_{self.model_name}_{self.model_type}"
        )
        
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(
                f"Classification directory not found: {self.output_path}. "
                "Please run image classification before quantification."
            )
    
    def _create_column_headers(self) -> List[str]:
        """
        Create column headers for the CSV file based on class names.
        
        Returns:
            List of column names for the CSV file
        """
        # Create column headers for the CSV file
        headers = ['Image name']
        
        # Add pixel count columns for each class
        for class_name in self.class_names[:-1]:  # Excluding 'black' class
            headers.append(f'{class_name} PC')

        # Add tissue composition columns (excluding whitespace class)
        for i, class_name in enumerate(self.class_names[:-1]):
            if i + 1 != self.nwhite:  # Skip whitespace class
                headers.append(f"{class_name} TC(%)")
        
        return headers
    
    def _process_image(self, image_path: str) -> Tuple[str, List[Union[str, int, float]]]:
        """
        Process a single image and compute quantification metrics.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (image_name, image_data) where image_data is a list of metrics
            
        Raises:
            RuntimeError: If there's an error processing the image
        """
        try:
            image_name = os.path.basename(image_path)
            
            # Read the image in grayscale mode
            image = load_image_with_fallback(image_path, mode="L")
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            
            # Create tissue mask (everything except whitespace)
            tissue = image != self.nwhite
            tissue_pixels = np.sum(tissue)
            
            if tissue_pixels == 0:
                raise RuntimeError(f"No tissue pixels found in image: {image_path}")
            
            # Count pixels for each class
            class_counts = [np.sum(image == i + 1) for i in range(len(self.class_names) - 1)]
            
            # Calculate tissue composition percentages
            tissue_compositions = [
                (count / tissue_pixels * 100) if i + 1 != self.nwhite else 0 
                for i, count in enumerate(class_counts)
            ]
            
            # Combine data for CSV
            image_data = [image_name] + class_counts + [
                comp for i, comp in enumerate(tissue_compositions) if i + 1 != self.nwhite
            ]
            
            return image_name, image_data
            
        except Exception as e:
            raise RuntimeError(f"Error processing image {image_path}: {e}")
    
    def quantify(self) -> str:
        """
        Quantify all classified images and save results to CSV.
        
        Returns:
            Path to the CSV file containing quantification results
            
        Raises:
            FileNotFoundError: If no classified images are found
        """
        logger.info('Quantifying images...')
        
        # Get column headers
        headers = self._create_column_headers()
        
        # Set up output CSV file
        csv_file = os.path.join(self.output_path, 'image_quantifications.csv')
        df = pd.DataFrame(columns=headers)
        df.to_csv(csv_file, index=False)
        
        # Find all classified images
        files = [f for f in os.listdir(self.output_path) if f.endswith('.tif')]
        num_files = len(files)
        
        if num_files == 0:
            raise FileNotFoundError(f"No classified images (.tif) found in {self.output_path}")
        
        # Process each image
        for j, image_name in enumerate(files):
            logger.info(f"Image {j + 1} / {num_files}: {image_name}")
            
            try:
                _, image_data = self._process_image(os.path.join(self.output_path, image_name))
                
                # Append to CSV
                image_df = pd.DataFrame([image_data], columns=headers)
                image_df.to_csv(csv_file, mode='a', header=False, index=False)
                
            except RuntimeError as e:
                logger.info(f"  Warning: {str(e)}")
                continue
        
        # Add additional information
        additional_info = pd.DataFrame([
            ['Model name:', self.model_path], 
            ['File location:', self.output_path]
        ])
        additional_info.to_csv(csv_file, mode='a', header=False, index=False)
        
        logger.info(f"Quantification complete. Results saved to: {csv_file}")
        return csv_file


def quantify_images(model_path: str, image_path: str) -> str:
    """
    Quantify tissue composition in classified images.
    
    This function serves as a compatibility wrapper around the ImageQuantifier class,
    preserving the original function signature for backward compatibility.
    
    Args:
        model_path: Path to the directory containing model data
        image_path: Path to the directory containing images to quantify
        
    Returns:
        Path to the CSV file containing quantification results
    """
    quantifier = ImageQuantifier(model_path, image_path)
    return quantifier.quantify()


if __name__ == '__main__':
    model_path = r'\\path\Kiemendata\Valentina Matos\tissues for methods paper\human liver\CODA_python_08_30_2024'
    image_path = r'\\path\Kiemendata\Valentina Matos\tissues for methods paper\human liver\10x'
    quantify_images(model_path, image_path)