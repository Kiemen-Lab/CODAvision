"""
Semantic Segmentation Module

This module provides the SemanticSegmenter class for performing semantic segmentation
on images using trained deep learning models.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated March 13, 2025
"""

from typing import Optional, Union, Tuple
import numpy as np
import tensorflow as tf

from base.image.utils import read_image


class SemanticSegmenter:
    """
    A class for performing semantic segmentation on images using a trained model.

    This class provides methods for loading images, running inference with a trained model,
    and generating segmentation masks.
    """

    @staticmethod
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
        return read_image(image_input, image_size, mask)

    @staticmethod
    def infer(model: tf.keras.Model, image_tensor: tf.Tensor) -> np.ndarray:
        """
        Run inference on an image using a trained segmentation model.

        Args:
            model: Trained TensorFlow/Keras segmentation model
            image_tensor: Preprocessed image tensor to segment

        Returns:
            Segmentation mask as a 2D numpy array, where each pixel value corresponds to a class label
        """
        predictions = model.predict(np.expand_dims(image_tensor, axis=0), verbose=0)
        predictions = np.squeeze(predictions)
        predictions = np.argmax(predictions, axis=2)
        return predictions

    @classmethod
    def segment_image(
            cls,
            image_input: Union[str, np.ndarray],
            image_size: int,
            model: tf.keras.Model
    ) -> np.ndarray:
        """
        Perform semantic segmentation on an input image using a pre-trained model.

        Args:
            image_input: Either a file path to an image or a numpy array containing image data
            image_size: Size to which the input image should be resized (assumes square images)
            model: Pre-trained TensorFlow/Keras model for semantic segmentation

        Returns:
            2D array representing the segmentation mask, where each pixel value corresponds to a class label

        Raises:
            AssertionError: If there is an error reading the image
        """
        image_tensor = cls.read_image(image_input, image_size)
        assert image_tensor is not None, f"Error: Could not read the image from input {image_input}"
        prediction_mask = cls.infer(model, image_tensor)
        return prediction_mask


def semantic_seg(image_path: Union[str, np.ndarray], image_size: int, model: tf.keras.Model) -> np.ndarray:
    """
    Perform semantic segmentation on an input image using a pre-trained model.

    Args:
        image_path: Path to the input image file or numpy array containing image data
        image_size: Size to which the input image should be resized
        model: Pre-trained TensorFlow/Keras model for semantic segmentation

    Returns:
        2D array representing the segmentation mask, where each pixel value corresponds to a class label

    Raises:
        AssertionError: If there is an error reading the image
    """
    return SemanticSegmenter.segment_image(image_path, image_size, model)