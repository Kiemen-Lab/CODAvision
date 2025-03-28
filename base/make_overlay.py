"""
Overlay Generation for Semantic Segmentation

This module provides functions for creating visual overlays of segmentation masks
on original images.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated March 13, 2025
"""

import os
import cv2
import tensorflow as tf
import numpy as np

from base.image.utils import decode_segmentation_masks, get_overlay
from base.data.loaders import read_image_overlay, convert_to_array


def read_image_overlay(image_input):
    """
    Read an image for overlay creation.

    Args:
        image_input: Path to image file or numpy array

    Returns:
        TensorFlow tensor containing the image
    """
    try:
        if isinstance(image_input, np.ndarray):
            # Already a numpy array, convert to tensor
            image = tf.convert_to_tensor(image_input)
        else:
            # Load from file
            image = tf.io.read_file(image_input)
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
        return image
    except Exception as e:
        print(f"Error reading image {image_input}: {e}")
        return None


def make_overlay(image_path, prediction_mask, colormap, save_path):
    """
    Create and save an overlay of a segmentation mask on an image.

    Args:
        image_path: Path to the original image
        prediction_mask: Segmentation mask with class indices
        colormap: Color map for visualization
        save_path: Directory to save the overlay image

    Returns:
        Overlay image as numpy array
    """
    os.makedirs(save_path, exist_ok=True)

    # Convert image and prediction mask to arrays with consistent dimensions
    image_array, prediction_mask = convert_to_array(image_path, prediction_mask)

    # Convert to tensor for processing
    image_tensor = read_image_overlay(image_array)

    # Convert prediction mask to colored visualization
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, n_classes=len(colormap))

    # Create the overlay
    overlay = get_overlay(image_tensor, prediction_colormap, alpha=0.65)

    # Save the overlay image
    overlay_image = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    save_file_path = os.path.join(save_path, os.path.basename(image_path)[:-3]+'jpg')
    cv2.imwrite(save_file_path, overlay_image)

    return overlay