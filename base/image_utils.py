"""
Image Utilities for Semantic Segmentation

This module provides common image processing utilities used across the semantic
segmentation pipeline, including loading, preprocessing, and dataset creation.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: March 13, 2025
"""

from typing import Optional, Tuple, Union, List

import numpy as np
import tensorflow as tf
import keras
import cv2
from PIL import Image

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
        print(f"Error reading image {image_input}: {e}")
        return None

def convert_to_array(image_path: str, prediction_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert image and prediction mask to numpy arrays with consistent dimensions.
    
    Args:
        image_path: Path to the image file
        prediction_mask: Prediction mask as numpy array
        
    Returns:
        Tuple of (image array, prediction mask array)
    """
    # Read image
    image = cv2.imread(image_path)
    image = image[:, :, ::-1]  # BGR to RGB

    # Resize if needed
    if image.shape[0] > 20000 or image.shape[1] > 20000:
        # Convert to PIL image for better large image handling
        image_pil = Image.fromarray(image)
        # Downsample by a factor of 10
        image_pil = image_pil.resize((image_pil.width // 10, image_pil.height // 10))
        image = np.array(image_pil)
        
        # Also downsample the prediction mask
        prediction_mask_pil = Image.fromarray(prediction_mask)
        prediction_mask_pil = prediction_mask_pil.resize((prediction_mask_pil.width // 10, prediction_mask_pil.height // 10))
        prediction_mask = np.array(prediction_mask_pil)

    return image, prediction_mask

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
    
    # Map the loading function to each element
    dataset = dataset.map(
        lambda img, mask: load_data(img, mask),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset

def decode_segmentation_masks(mask: np.ndarray, colormap: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Decode class indices into RGB colors for visualization.
    
    Args:
        mask: Segmentation mask with class indices
        colormap: Array of RGB color values for each class
        n_classes: Number of classes
        
    Returns:
        RGB image representing the segmentation mask
    """
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_overlay(image: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """
    Create an overlay of a colored mask on an image.
    
    Args:
        image: Background image
        colored_mask: Colored segmentation mask
        alpha: Weight of the original image in the blend (0-1)
        
    Returns:
        Blended overlay image
    """
    if isinstance(image, tf.Tensor):
        image = keras.utils.array_to_img(image)
        image = np.array(image).astype(np.uint8)
    
    overlay = cv2.addWeighted(image, alpha, colored_mask, 1 - alpha, 0)
    return overlay