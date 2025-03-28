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