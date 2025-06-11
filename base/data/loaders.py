"""
Data Loaders for CODAvision

This module provides functions and classes for loading various types of data
including images, annotations, and creating TensorFlow datasets for training.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import numpy as np
import tensorflow as tf
import pickle

# Set up logging
import logging
logger = logging.getLogger(__name__)


def read_image(
    image_input: Union[str, np.ndarray],
    image_size: int,
    mask: bool = False
) -> Optional[tf.Tensor]:
    """
    Read and preprocess an image for segmentation.

    Args:
        image_input: Either a file path to an image or a numpy array containing image data
        image_size: Size to which the image should be resized (assumes square images)
        mask: Whether the image is a segmentation mask (single channel) or not (RGB)

    Returns:
        Preprocessed image tensor, or None if there was an error reading the image
    """
    try:
        if isinstance(image_input, np.ndarray):
            # Convert numpy array to tensor
            image = tf.convert_to_tensor(image_input)
            image = tf.image.resize(image, [image_size, image_size])
        else:
            # Load from file
            image = tf.io.read_file(image_input)
            if mask:
                image = tf.image.decode_png(image, channels=1)
                image.set_shape([None, None, 1])
                image = tf.image.resize(images=image, size=[image_size, image_size])
            else:
                image = tf.image.decode_png(image, channels=3)
                image.set_shape([None, None, 3])
                image = tf.image.resize(images=image, size=[image_size, image_size])
        return image
    except Exception as e:
        logger.info(f"Error reading image {image_input}: {e}")
        return None


def create_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    image_size: int,
    batch_size: int
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from lists of image and mask paths.

    Args:
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files
        image_size: Size to resize images and masks to (assumes square images)
        batch_size: Number of samples per batch

    Returns:
        TensorFlow dataset containing batches of (image, mask) pairs
    """
    def load_data(image_path, mask_path):
        """Inner function to load an image and its corresponding mask."""
        image = read_image(image_path, image_size)
        mask = read_image(mask_path, image_size, mask=True)
        return image, mask
    
    # Create a dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Map the load_data function to preprocess the images and masks
    dataset = dataset.map(
        lambda img, mask: load_data(img, mask),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset


def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Load model metadata from a pickle file.
    
    Args:
        model_path: Path to the directory containing the model metadata
        
    Returns:
        Dictionary containing model metadata
        
    Raises:
        FileNotFoundError: If the model data file doesn't exist
        ValueError: If essential parameters are missing
    """
    data_file = os.path.join(model_path, 'net.pkl')
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Model data file not found: {data_file}")
    
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # Handle model type format for consistency
        if 'model_type' in data and '+' in data['model_type']:
            data['model_type'] = data['model_type'].replace('+', '_plus')
        
        # Check for essential parameters
        essential_params = ['classNames', 'sxy', 'nblack', 'nwhite']
        missing_params = [param for param in essential_params if param not in data]
        
        if missing_params:
            raise ValueError(f"Missing required parameters in model data: {missing_params}")
        
        return data
    
    except Exception as e:
        raise ValueError(f"Failed to load model data: {e}")


class DataGenerator:
    """
    Class for generating TensorFlow datasets from image and mask paths.

    This class handles loading and preprocessing of images and masks for
    training and validation of segmentation models.
    """

    def __init__(self, image_size: int, batch_size: int):
        """
        Initialize the data generator.

        Args:
            image_size: Size to resize images and masks to (assumes square images)
            batch_size: Number of samples per batch
        """
        self.image_size = image_size
        self.batch_size = batch_size

    def read_image(self, image_path: str, mask: bool = False) -> tf.Tensor:
        """
        Read and preprocess an image or mask.

        Args:
            image_path: Path to the image file
            mask: Whether the image is a mask (single channel) or not (RGB)

        Returns:
            Preprocessed image tensor
        """
        return read_image(image_path, self.image_size, mask)

    def load_data(self, image_path: str, mask_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load an image and its corresponding mask.

        Args:
            image_path: Path to the image file
            mask_path: Path to the mask file

        Returns:
            Tuple of (image, mask) tensors
        """
        image = self.read_image(image_path)
        mask = self.read_image(mask_path, mask=True)
        return image, mask

    def create_dataset(
        self,
        image_paths: List[str],
        mask_paths: List[str]
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from lists of image and mask paths.

        Args:
            image_paths: List of paths to image files
            mask_paths: List of paths to mask files

        Returns:
            TensorFlow dataset containing batches of (image, mask) pairs
        """
        return create_dataset(image_paths, mask_paths, self.image_size, self.batch_size)