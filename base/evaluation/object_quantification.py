"""
Object Quantification Module for CODAvision

This module provides functionality for quantifying and analyzing objects in classified
images based on connected components analysis. It identifies, measures, and catalogs
objects belonging to specific annotation classes.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: April 2025
"""

import os
import warnings
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Union, Optional, Tuple
from skimage.measure import label
from skimage.morphology import remove_small_objects
from base.image.utils import load_image_with_fallback

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Suppress warnings to avoid cluttering output
warnings.filterwarnings("ignore")


class ObjectQuantifier:
    """
    A class for quantifying and analyzing objects in classified images.
    
    This class identifies connected components in classified images for specific 
    annotation classes and analyzes their properties (size, count, etc.). Results
    are saved to CSV files for further analysis.
    
    Attributes:
        model_path (str): Path to the directory containing model data
        output_path (str): Path to the directory containing classified images to analyze
        class_names (List[str]): Names of annotation classes from the model
    """
    
    def __init__(self, model_path: str, output_path: str):
        """
        Initialize the ObjectQuantifier with model and output paths.
        
        Args:
            model_path: Path to the directory containing model data file 'net.pkl'
            output_path: Path to the directory containing classified images to analyze
            
        Raises:
            FileNotFoundError: If model data file isn't found
            ValueError: If required model data is missing
        """
        self.model_path = model_path
        self.output_path = output_path
        self.class_names = None
        
        self._load_model_data()
    
    def _load_model_data(self) -> None:
        """
        Load model metadata from the pickle file.
        
        Raises:
            FileNotFoundError: If model data file isn't found
            ValueError: If required model data is missing
        """
        model_data_file = os.path.join(self.model_path, 'net.pkl')
        
        if not os.path.exists(model_data_file):
            raise FileNotFoundError(f"Model data file not found: {model_data_file}")
        
        try:
            with open(model_data_file, 'rb') as f:
                data = pickle.load(f)
            
            self.class_names = data.get('classNames')
            
            if self.class_names is None:
                raise ValueError("Missing class names in model data")
                
        except Exception as e:
            raise ValueError(f"Failed to load model data: {e}")
    
    def quantify_class(self, class_id: int, min_size: int = 500) -> str:
        """
        Quantify objects for a specific annotation class.
        
        This method analyzes all classified images in the output path for the
        specified annotation class, identifies connected components (objects),
        and records their sizes. Results are saved to a CSV file.
        
        Args:
            class_id: Integer ID of the annotation class to analyze
            min_size: Minimum object size in pixels to include in analysis
            
        Returns:
            Path to the CSV file containing the results
            
        Raises:
            ValueError: If the class ID is invalid or no images are found
            IOError: If there's an error writing to the output file
        """
        if class_id <= 0 or class_id > len(self.class_names):
            raise ValueError(f"Invalid class ID: {class_id}. Must be between 1 and {len(self.class_names)}")
        
        # Get the class name for the specified ID
        class_name = self.class_names[class_id-1]
        logger.info(f'_______Starting object analysis for {class_name}________')
        
        # Define output CSV file path
        csv_file = os.path.join(self.output_path, f'{class_name}_count_analysis.csv')
        
        # Find all TIFF files in the output directory
        image_files = [os.path.join(self.output_path, f) for f in os.listdir(self.output_path) 
                      if f.endswith('.tif')]
        
        if not image_files:
            raise ValueError(f"No TIFF files found in {self.output_path}")
        
        # List to store properties of all objects
        all_props = []
        
        # Process each image
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            logger.info(f'Processing image {image_name}')
            
            # Read the classified image
            img = load_image_with_fallback(image_path, mode="L")
            if img is None:
                logger.info(f"  Warning: Failed to read image {image_path}")
                continue
            
            logger.info(f'  Analyzing annotation class: {class_name}')
            
            # Create mask for the target class
            label_mask = (img == class_id)
            
            # Label connected components
            labeled = label(label_mask, connectivity=1)
            labeled = remove_small_objects(labeled, min_size=min_size, connectivity=1)
            
            # Calculate object sizes
            object_sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)
            
            # Filter by minimum size
            filtered_labels = [object_ID for object_ID, size in enumerate(object_sizes, start=1) 
                              if size >= min_size]
            
            # Sort labels by size (descending)
            sorted_labels = sorted(filtered_labels, key=lambda x: object_sizes[x-1], reverse=True)
            
            # Create new labeled image with sorted labels
            new_labeled = np.zeros_like(labeled)
            for new_label, label_id in enumerate(sorted_labels, start=1):
                new_labeled[labeled == label_id] = new_label
            
            # Get final object sizes
            object_sizes = np.bincount(new_labeled.ravel())[1:]
            
            # Record object properties
            for object_ID, size in enumerate(object_sizes, start=1):
                if size >= min_size:
                    all_props.append([image_name, f"{class_name} {object_ID}", size])
        
        if not all_props:
            logger.info(f"  No objects found for class '{class_name}' with minimum size {min_size}")
            return ""
        
        # Create DataFrame and save to CSV
        props_df = pd.DataFrame(all_props, columns=["Image", "Object ID", "Object Size (pixels)"])
        logger.info(f'DataFrame to be written for {class_name}:\n{props_df}')
        
        try:
            props_df.to_csv(csv_file, mode='w', header=True, index=False)
        except PermissionError as e:
            logger.error(f"PermissionError: {e}")
            return ""
        except ValueError as e:
            logger.error(f"ValueError: {e}")
            return ""
        
        logger.info('_______Object analysis completed________')
        return csv_file
    
    def quantify_multiple_classes(self, class_ids: List[int], min_size: int = 500) -> List[str]:
        """
        Quantify objects for multiple annotation classes.
        
        Args:
            class_ids: List of class IDs to analyze
            min_size: Minimum object size in pixels to include in analysis
            
        Returns:
            List of paths to CSV files containing the results
        """
        csv_files = []
        
        for class_id in class_ids:
            csv_file = self.quantify_class(class_id, min_size)
            if csv_file:
                csv_files.append(csv_file)
        
        return csv_files


def quantify_objects(pthDL: str, quantpath: str, tissue: int, min_size: int = 500) -> str:
    """
    Quantifies the connected components in each classified image (for the specified annotation class) located in the
    specified quantpath, and writes the results to separate CSV files for each class in the tissue list.

    Args:
        pthDL: Path to the directory containing the model data file 'net.pkl'.
        quantpath: Path to the directory containing the images to be analyzed.
        tissue: Integer ID of annotation class label to be analyzed.
        min_size: Minimum object size in pixels to include in analysis.

    Returns:
        Path to the CSV file containing the results (empty string if failed)
    """
    try:
        quantifier = ObjectQuantifier(pthDL, quantpath)
        return quantifier.quantify_class(tissue, min_size)
    except Exception as e:
        logger.error(f"Error in object quantification: {e}")
        return ""


if __name__ == '__main__':
    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\mouse lung\annotations\01_30_2025_metsquantification'
    quantpath = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\mouse lung\annotations\5x\classification_01_30_2025_metsquantification_DeepLabV3_plus'
    tissue = 4
    quantify_objects(pthDL, quantpath, tissue)