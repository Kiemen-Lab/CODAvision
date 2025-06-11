"""
Image Classification with Semantic Segmentation Models

This module provides functionality for classifying images using trained semantic segmentation models.
It processes images by dividing them into overlapping tiles, classifying each tile with the model,
and then reconstructing the full classified image.

The module includes:
- ImageClassifier class: An object-oriented implementation with methods for loading model data,
  processing images, and creating visualizations of the classification results.
- classify_images function: A backward-compatible wrapper around the ImageClassifier class
  that maintains the original function signature.

Example usage:
    # Using the function interface
    output_path = classify_images("path/to/images", "path/to/model", "DeepLabV3_plus")

    # Using the class interface
    classifier = ImageClassifier("path/to/images", "path/to/model", "DeepLabV3_plus")
    output_path = classifier.classify(color_overlay=True, color_mask=True, display=True)
"""

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from glob import glob
from PIL import Image
from typing import Tuple, List
from scipy.ndimage import binary_fill_holes

from base.models.utils import get_model_paths
from base.image.segmentation import semantic_seg
from base.image.utils import decode_segmentation_masks, create_overlay, load_image_with_fallback
from base.data.loaders import load_model_metadata

# Set up logging
import logging
logger = logging.getLogger(__name__)


class ImageClassifier:
    """
    A class for classifying images using a trained semantic segmentation model.

    This class handles loading model data, processing images, and saving classification results.
    """


    def __init__(self, image_path: str, model_path: str, model_type: str):
        """
        Initialize the ImageClassifier.

        Args:
            image_path: Path to the directory containing images to classify
            model_path: Path to the directory containing the model data
            model_type: Type of model to use for classification
        """
        self.image_path = image_path
        self.model_path = model_path
        self.model_type = model_type

        # Will be initialized later
        self.model = None
        self.class_names = None
        self.color_map = None
        self.nblack = None
        self.nwhite = None
        self.image_size = None
        self.model_name = None
        self.output_path = None
        self.model_paths = None

        # Display options
        self.should_create_color_overlay = False
        self.should_create_color_mask = False

        # Load model data
        self._load_model_data()

    def _load_model_data(self) -> None:
        """
        Load model metadata from pickle file.

        Raises:
            FileNotFoundError: If the model data file doesn't exist
            ValueError: If essential parameters are missing
        """
        try:
            # Use the model utils to load metadata
            data = load_model_metadata(self.model_path)

            # Extract needed parameters
            self.class_names = data.get('classNames')
            self.color_map = data.get('cmap')
            self.nblack = data.get('nblack')
            self.nwhite = data.get('nwhite')
            self.image_size = data.get('sxy')
            self.model_name = data.get('nm')

            # Get standard paths
            self.model_paths = get_model_paths(self.model_path, self.model_type)

            # Set output path
            self.output_path = os.path.join(self.image_path, f'classification_{self.model_name}_{self.model_type}')

        except Exception as e:
            raise ValueError(f"Failed to load model data: {e}")

    def _load_model(self) -> 'keras.Model':
        """
        Load the trained model.

        Returns:
            Loaded model

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        try:
            # Try best model first
            model_path = self.model_paths['best_model']

            if not os.path.exists(model_path):
                # Fall back to final model
                model_path = self.model_paths['final_model']

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"No model file found at {model_path}")

            from base.models.backbones import model_call
            model = model_call(self.model_type, IMAGE_SIZE=self.image_size, NUM_CLASSES=len(self.class_names))
            model.load_weights(model_path)

            return model

        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def _create_output_directories(self) -> str:
        """
        Create output directories for classified images.

        Returns:
            Path to the output directory
        """
        os.makedirs(self.output_path, exist_ok=True)

        if self.should_create_color_overlay:
            overlay_path = os.path.join(self.output_path, 'check_classification')
            os.makedirs(overlay_path, exist_ok=True)

        if self.should_create_color_mask:
            mask_path = os.path.join(self.output_path, 'color')
            os.makedirs(mask_path, exist_ok=True)

        return self.output_path

    def _get_tissue_mask(self, image_path: str) -> np.ndarray:
        """
        Get or create a tissue mask for the image.

        Args:
            image_path: Path to the image file

        Returns:
            Tissue mask as numpy array
        """
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]

        try:
            # Try to load existing mask
            ta_path = os.path.join(self.image_path, 'TA', f"{base_name}.png")
            if not os.path.exists(ta_path):
                ta_path = os.path.join(self.image_path, 'TA', f"{base_name}.tif")

            tissue_mask = Image.open(ta_path)
            tissue_mask = binary_fill_holes(np.array(tissue_mask))
        except:
            # Create a mask if none exists
            image = load_image_with_fallback(image_path)
            tissue_mask = np.array(image[:,:,1]) < 220
            tissue_mask = binary_fill_holes(tissue_mask.astype(bool))

        return tissue_mask

    def _process_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single image with the model.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (original image, classified image)
        """
        # Load the image
        image = load_image_with_fallback(image_path)

        # Get tissue mask
        tissue_mask = self._get_tissue_mask(image_path)

        # Pad the image for tile processing
        padding = self.image_size + 100
        padded_image = np.pad(
            image,
            pad_width=((padding, padding), (padding, padding), (0, 0)),
            mode='constant',
            constant_values=0
        )

        padded_mask = np.pad(
            tissue_mask,
            pad_width=((padding, padding), (padding, padding)),
            mode='constant',
            constant_values=True
        )

        # Initialize the classified image
        classified_image = np.zeros(padded_mask.shape, dtype=np.uint8)

        # Process in tiles with overlap
        step_size = self.image_size - 2 * 100  # 200 pixels overlap
        for row in range(self.image_size, padded_image.shape[0] - self.image_size, step_size):
            for col in range(self.image_size, padded_image.shape[1] - self.image_size, step_size):
                # Extract tile
                tile = padded_image[row:row + self.image_size, col:col + self.image_size, :]

                # Classify tile
                tile_classified = semantic_seg(tile, image_size=self.image_size, model=self.model)

                # Remove border (100 pixels on each side) to avoid edge effects
                tile_classified = tile_classified[100:-100, 100:-100]

                # Add to the result image
                classified_image[row + 100:row + self.image_size - 100, col + 100:col + self.image_size - 100] = tile_classified

        # Crop back to original size
        original_image = padded_image[padding:-padding, padding:-padding, :]
        classified_image = classified_image[padding:-padding, padding:-padding]

        # Post-process the classification
        classified_image = classified_image + 1
        classified_image[np.logical_or(classified_image == self.nblack, classified_image == 0)] = self.nwhite

        return original_image, classified_image

    def _create_color_overlay(self, image_path: str, classified_image: np.ndarray) -> np.ndarray:
        """
        Create a color overlay of the classification on the original image.

        Args:
            image_path: Path to the original image
            classified_image: Classified image as numpy array

        Returns:
            Overlay image as numpy array
        """
        save_path = os.path.join(self.output_path, 'check_classification')
        return create_overlay(image_path, classified_image - 1, colormap=self.color_map, save_path=save_path)

    def _create_color_mask(self, classified_image: np.ndarray, image_name: str) -> np.ndarray:
        """
        Create a color mask from the classified image.

        Args:
            classified_image: Classified image as numpy array
            image_name: Name of the image file

        Returns:
            Color mask as numpy array
        """
        # Adjust class indices (subtract 1)
        classified_image = classified_image - 1

        # Create colored visualization
        color_image = decode_segmentation_masks(
            classified_image,
            self.color_map,
            n_classes=len(self.class_names) - 1
        )

        # Save the color mask
        save_path = os.path.join(self.output_path, 'color', image_name)
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, color_image_bgr)

        return color_image

    def _find_images(self) -> List[str]:
        """
        Find all images in the specified directory.

        Returns:
            List of image file paths
        """
        image_list = []

        # First try TIFF files
        tiff_files = glob(os.path.join(self.image_path, "*.tif"))
        if tiff_files:
            image_list.extend(tiff_files)

        # Fall back to JPG and PNG if no TIFFs
        if not image_list:
            jpg_files = glob(os.path.join(self.image_path, "*.jpg"))
            png_files = glob(os.path.join(self.image_path, "*.png"))
            image_list.extend(jpg_files)
            image_list.extend(png_files)

        if not image_list:
            logger.error(f"No TIFF, PNG or JPG image files found in {self.image_path}")

        return sorted(image_list)

    def _display_results(self, original_image: np.ndarray, classified_image: np.ndarray) -> None:
        """
        Display the original image and its classification.

        Args:
            original_image: Original image as numpy array
            classified_image: Classified image as numpy array
        """
        prediction_colormap = decode_segmentation_masks(
            classified_image,
            self.color_map,
            n_classes=len(self.class_names) - 1
        )

        # Display the images
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(original_image)
        axs[1].imshow(keras.utils.array_to_img(prediction_colormap))
        for ax in axs:
            ax.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        plt.pause(0.2)

    def classify(self, color_overlay: bool = True, color_mask: bool = False, display: bool = True) -> str:
        """
        Classify all images in the specified directory.

        Args:
            color_overlay: Whether to create a color overlay on the original image
            color_mask: Whether to create a color mask
            display: Whether to display a sample of the results

        Returns:
            Path to the directory containing classified images
        """
        start_time = time.time()

        # Set display options
        self.should_create_color_overlay = color_overlay
        self.should_create_color_mask = color_mask

        # Create output directories
        self._create_output_directories()

        # Load the model
        self.model = self._load_model()

        # Find images to classify
        image_list = self._find_images()

        if not image_list:
            return self.output_path

        print('   ')

        # For displaying results
        first_image = None
        first_img_prediction = None

        # Process each image
        for i, img_path in enumerate(image_list):
            classification_start = time.time()
            img_name = os.path.basename(img_path)
            logger.info(f'Starting classification of image {i + 1} of {len(image_list)}: {img_name}')

            # Define output path
            output_path = os.path.join(self.output_path, f"{os.path.splitext(img_name)[0]}.tif")

            # Skip if already classified
            if os.path.exists(output_path):
                logger.info(f'Image already classified, skipping: {img_name}')
                continue

            # Process the image
            original_image, classified_image = self._process_image(img_path)

            # Save the classified image
            classified_image_pil = Image.fromarray(classified_image)
            classified_image_pil.save(output_path)

            # Create color overlay if requested
            if self.should_create_color_overlay:
                self._create_color_overlay(img_path, classified_image)

            # Create color mask if requested
            if self.should_create_color_mask:
                self._create_color_mask(classified_image, img_name)

            # Store the first image for display
            if i == 0:
                first_image = original_image
                first_img_prediction = classified_image - 1

            elapsed_time = round(time.time() - classification_start)
            logger.info(f'Image {i + 1} of {len(image_list)} took {elapsed_time} s')

        # Display results if requested
        if display and first_image is not None and first_img_prediction is not None:
            self._display_results(first_image, first_img_prediction)

        # Report total time
        end_time = time.time() - start_time
        hours, rem = divmod(end_time, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(f'  Total time for classification: {int(hours)}h {int(minutes)}m {int(seconds)}s')

        return self.output_path


def classify_images(pthim: str, pthDL: str, name: str, color_overlay_HE: bool = True, color_mask: bool = False, disp: bool = True) -> str:
    """
    Classify images using a trained semantic segmentation model.

    Args:
        pthim: Path to the directory containing images to classify
        pthDL: Path to the directory containing model data
        name: Type of model to use for classification (e.g., "DeepLabV3_plus")
        color_overlay_HE: Whether to create a color overlay on the original image. Defaults to True.
        color_mask: Whether to create a color mask. Defaults to False.
        disp: Whether to display a sample of the results. Defaults to True.

    Returns:
        Path to the directory containing classified images
    """
    classifier = ImageClassifier(pthim, pthDL, name)
    return classifier.classify(color_overlay=color_overlay_HE, color_mask=color_mask, display=disp)