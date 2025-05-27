"""
Image Utilities for Semantic Segmentation

This module provides common image processing utilities used across the semantic
segmentation pipeline, including loading, preprocessing, visualization, and overlay creation.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: April 2025
"""

from typing import Optional, Tuple, Union, List, Any

import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = "0"  # Set max image size for OpenCV

import numpy as np
import tensorflow as tf
import keras
import cv2

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Set up logging
import logging
logger = logging.getLogger(__name__)


def load_image_with_fallback(image_path: str, mode: str = "RGB") -> np.ndarray:
    """
    Attempts to load an image using OpenCV. If it fails, falls back to Pillow.

    Args:
        image_path: Path to the image file.
        mode: Mode to convert the image to when using Pillow (default: "RGB").

    Returns:
        The loaded image as a NumPy array.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if mode == "L" else cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to load image with OpenCV")
        if mode == "RGB" and len(image.shape) == 3:  # Convert BGR to RGB
            image = image[:, :, ::-1]
        return image
    except:
        with Image.open(image_path) as img:
            return np.array(img.convert(mode))


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


def get_overlay(
    image: Union[np.ndarray, tf.Tensor],
    colored_mask: np.ndarray,
    alpha: float = 0.65
) -> np.ndarray:
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


def read_image_overlay(image_input: Union[str, np.ndarray]) -> Optional[tf.Tensor]:
    """
    Read an image for overlay creation.

    Args:
        image_input: Path to image file or numpy array

    Returns:
        TensorFlow tensor containing the image, or None if reading fails
    """
    try:
        if isinstance(image_input, np.ndarray):
            # If it's already a numpy array, just convert to tensor
            image = tf.convert_to_tensor(image_input)
        else:
            # Otherwise, read from file
            image = tf.io.read_file(image_input)
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
        return image
    except Exception as e:
        logger.error(f"Error reading image {image_input}: {e}")
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
    # Read the image
    # image = cv2.imread(image_path)
    image = load_image_with_fallback(image_path)

    image = image[:, :, ::-1]  # Convert BGR to RGB

    # Handle large images by resizing
    if image.shape[0] > 20000 or image.shape[1] > 20000:
        # Convert to PIL image for resizing
        image_pil = Image.fromarray(image)
        # Resize while maintaining aspect ratio
        image_pil = image_pil.resize((image_pil.width // 4, image_pil.height // 4), Image.LANCZOS)
        image = np.array(image_pil)

        # Resize prediction mask to match
        prediction_mask_pil = Image.fromarray(prediction_mask)
        prediction_mask_pil = prediction_mask_pil.resize((prediction_mask_pil.width // 4, prediction_mask_pil.height // 4), Image.LANCZOS)
        prediction_mask = np.array(prediction_mask_pil)

    return image, prediction_mask


def create_overlay(
    image_path: str,
    prediction_mask: np.ndarray,
    colormap: np.ndarray,
    save_path: Optional[str] = None,
    alpha: float = 0.65
) -> np.ndarray:
    """
    Create and optionally save an overlay of a segmentation mask on an image.

    Args:
        image_path: Path to the original image
        prediction_mask: Segmentation mask with class indices
        colormap: Color map for visualization with RGB values for each class
        save_path: Directory to save the overlay image (None to skip saving)
        alpha: Weight of the original image in the blend (0-1)

    Returns:
        Overlay image as numpy array
    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # Convert image and mask to properly sized arrays
    image_array, prediction_mask = convert_to_array(image_path, prediction_mask)

    # Get tensor representation of the image
    image_tensor = read_image_overlay(image_array)
    if image_tensor is None:
        raise ValueError(f"Failed to read image at {image_path}")

    # Create colormap from prediction mask
    prediction_colormap = decode_segmentation_masks(
        prediction_mask,
        colormap,
        n_classes=len(colormap)
    )

    # Create overlay by blending original image with colormap
    overlay = get_overlay(image_tensor, prediction_colormap, alpha=alpha)

    # Save the overlay if a save path is provided
    if save_path is not None:
        overlay_image = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        if image_path.lower().endswith(('.tiff', '.tif')):
            output_filename = os.path.basename(image_path)[:-4] + '.jpg'
        else:
            output_filename = os.path.basename(image_path)[:-3] + 'jpg'
        save_file_path = os.path.join(save_path, output_filename)
        cv2.imwrite(save_file_path, overlay_image)

    return overlay