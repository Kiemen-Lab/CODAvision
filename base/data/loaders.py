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
from PIL import Image

# Import configuration
from base.config import DataConfig

# Set up logging
import logging
logger = logging.getLogger(__name__)


def _load_image_pil_internal(image_path, image_size: int, mask: bool) -> np.ndarray:
    """
    Internal helper function to load images using PIL (for TIFF support).
    This function is designed to be called via tf.py_function.

    Args:
        image_path: File path (can be string, bytes, or numpy array of bytes)
        image_size: Size to which the image should be resized
        mask: Whether the image is a segmentation mask (single channel) or not (RGB)

    Returns:
        Numpy array of the loaded and preprocessed image
    """
    # Convert to string path - handle various input types from tf.py_function
    if isinstance(image_path, bytes):
        path_str = image_path.decode('utf-8')
    elif isinstance(image_path, np.ndarray):
        # numpy array of bytes (from tensor)
        if image_path.dtype == np.object_ or image_path.dtype.kind == 'O':
            path_str = image_path.item().decode('utf-8')
        else:
            path_str = str(image_path.item())
    elif isinstance(image_path, str):
        path_str = image_path
    else:
        # Try converting to string
        try:
            # If it's a tensor object, try to extract the value
            path_val = image_path.numpy() if hasattr(image_path, 'numpy') else image_path
            if isinstance(path_val, bytes):
                path_str = path_val.decode('utf-8')
            else:
                path_str = str(path_val)
        except:
            path_str = str(image_path)
            # Remove tensor wrapper if present
            if "tf.Tensor" in path_str:
                # Extract the path from the tensor representation
                import re
                match = re.search(r"b'([^']+)'", path_str)
                if match:
                    path_str = match.group(1)

    # Load image with PIL
    pil_image = Image.open(path_str)
    image_array = np.array(pil_image)

    if mask:
        # Ensure single channel for masks
        if len(image_array.shape) == 3:
            # Multi-channel image, take first channel
            image_array = image_array[:, :, 0]
        # Add channel dimension
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, -1)
        # Resize
        if image_array.shape[0] != image_size or image_array.shape[1] != image_size:
            image_pil = Image.fromarray(image_array.squeeze())
            image_pil = image_pil.resize((image_size, image_size), Image.BILINEAR)
            image_array = np.array(image_pil)
            if len(image_array.shape) == 2:
                image_array = np.expand_dims(image_array, -1)
        return image_array.astype(np.float32)
    else:
        # Ensure 3 channels for RGB images
        if len(image_array.shape) == 2:
            # Grayscale, convert to RGB
            image_array = np.stack([image_array] * 3, axis=-1)
        elif len(image_array.shape) == 3 and image_array.shape[-1] == 1:
            # Single channel, convert to RGB
            image_array = np.repeat(image_array, 3, axis=-1)
        elif len(image_array.shape) == 3 and image_array.shape[-1] > 3:
            # More than 3 channels, take first 3
            image_array = image_array[:, :, :3]
        # Resize
        if image_array.shape[0] != image_size or image_array.shape[1] != image_size:
            image_pil = Image.fromarray(image_array.astype(np.uint8))
            image_pil = image_pil.resize((image_size, image_size), Image.BILINEAR)
            image_array = np.array(image_pil)
        return image_array.astype(np.float32)

# Don't let AutoGraph convert this function
_load_image_pil_internal = tf.autograph.experimental.do_not_convert(_load_image_pil_internal)


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
            # Detect file format based on extension
            is_tiff = isinstance(image_input, str) and image_input.lower().endswith(('.tif', '.tiff'))

            if is_tiff:
                # Use PIL to read TIFF files (TensorFlow 2.10 doesn't support decode_tiff)
                pil_image = Image.open(image_input)
                # Convert to numpy array
                image_array = np.array(pil_image)

                if mask:
                    # Ensure single channel for masks
                    if len(image_array.shape) == 3:
                        # Multi-channel image, take first channel
                        image_array = image_array[:, :, 0]
                    # Add channel dimension
                    if len(image_array.shape) == 2:
                        image_array = np.expand_dims(image_array, -1)
                    # Convert to tensor and resize
                    image = tf.convert_to_tensor(image_array, dtype=tf.float32)
                    image = tf.image.resize(images=image, size=[image_size, image_size])
                    image.set_shape([image_size, image_size, 1])
                else:
                    # Ensure 3 channels for RGB images
                    if len(image_array.shape) == 2:
                        # Grayscale, convert to RGB
                        image_array = np.stack([image_array] * 3, axis=-1)
                    elif len(image_array.shape) == 3 and image_array.shape[-1] == 1:
                        # Single channel, convert to RGB
                        image_array = np.repeat(image_array, 3, axis=-1)
                    elif len(image_array.shape) == 3 and image_array.shape[-1] > 3:
                        # More than 3 channels, take first 3
                        image_array = image_array[:, :, :3]
                    # Convert to tensor and resize
                    image = tf.convert_to_tensor(image_array, dtype=tf.float32)
                    image = tf.image.resize(images=image, size=[image_size, image_size])
                    image.set_shape([image_size, image_size, 3])
            else:
                # Use TensorFlow's decode_image for other formats (PNG, JPEG, GIF, BMP)
                image = tf.io.read_file(image_input)
                if mask:
                    # Decode mask (single channel)
                    image = tf.io.decode_image(image, channels=1, expand_animations=False)
                    image.set_shape([None, None, 1])
                    image = tf.image.resize(images=image, size=[image_size, image_size])
                else:
                    # Decode RGB image
                    image = tf.io.decode_image(image, channels=3, expand_animations=False)
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
    batch_size: int,
    shuffle: bool = False,
    shuffle_buffer_size: Optional[int] = None,
    seed: Optional[int] = None,
    cache: bool = False
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from lists of image and mask paths.

    Pipeline order: load → cache (optional) → shuffle (optional) → batch → prefetch

    Args:
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files
        image_size: Size to resize images and masks to (assumes square images)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Size of the shuffle buffer (if None, uses full dataset size)
        seed: Random seed for shuffling (for reproducibility)
        cache: Whether to cache the dataset in memory after loading

    Returns:
        TensorFlow dataset containing batches of (image, mask) pairs
    """
    # Define wrapper functions outside of load_data to avoid AutoGraph issues
    def load_image_wrapper(path):
        return _load_image_pil_internal(path, image_size, False)

    def load_mask_wrapper(path):
        return _load_image_pil_internal(path, image_size, True)

    def load_data(image_path, mask_path):
        """Inner function to load an image and its corresponding mask using tf.py_function."""
        # Use tf.py_function to load TIFF files with PIL
        image = tf.py_function(
            func=load_image_wrapper,
            inp=[image_path],
            Tout=tf.float32
        )
        mask = tf.py_function(
            func=load_mask_wrapper,
            inp=[mask_path],
            Tout=tf.float32
        )
        # Set shapes explicitly (required after tf.py_function)
        image.set_shape([image_size, image_size, 3])
        mask.set_shape([image_size, image_size, 1])
        return image, mask

    # Create a dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Map the load_data function to preprocess the images and masks
    dataset = dataset.map(
        lambda img, mask: load_data(img, mask),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Cache after loading images if requested
    if cache:
        dataset = dataset.cache()

    # Shuffle if requested (with reshuffle each iteration for training)
    if shuffle:
        buffer_size = shuffle_buffer_size or len(image_paths)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)

    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_training_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    image_size: int,
    batch_size: int,
    data_config: Optional[DataConfig] = None,
    seed: Optional[int] = None,
    cache: bool = False,
    repeat: bool = False
) -> tf.data.Dataset:
    """
    Create an optimized TensorFlow training dataset with shuffling and performance optimizations.

    This function creates a dataset specifically optimized for training with:
    - Data loading and preprocessing
    - Optional caching of loaded images for small datasets
    - Shuffling for randomization (with reshuffle each epoch)
    - Optional repeating for continuous training
    - Prefetching for performance

    Pipeline order: load → cache (optional) → shuffle → repeat (optional) → batch → prefetch

    Args:
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files
        image_size: Size to resize images and masks to (assumes square images)
        batch_size: Number of samples per batch
        data_config: DataConfig object with dataset configuration (if None, uses defaults)
        seed: Random seed for shuffling (for reproducibility)
        cache: Whether to cache the dataset in memory (recommended for small datasets)
        repeat: Whether to repeat the dataset indefinitely (usually False, let fit() handle epochs)

    Returns:
        Optimized TensorFlow dataset for training
    """
    if data_config is None:
        data_config = DataConfig()

    # Define wrapper functions outside of load_data to avoid AutoGraph issues
    def load_image_wrapper(path):
        return _load_image_pil_internal(path, image_size, False)

    def load_mask_wrapper(path):
        return _load_image_pil_internal(path, image_size, True)

    def load_data(image_path, mask_path):
        """Inner function to load an image and its corresponding mask using tf.py_function."""
        # Use tf.py_function to load TIFF files with PIL
        image = tf.py_function(
            func=load_image_wrapper,
            inp=[image_path],
            Tout=tf.float32
        )
        mask = tf.py_function(
            func=load_mask_wrapper,
            inp=[mask_path],
            Tout=tf.float32
        )
        # Set shapes explicitly (required after tf.py_function)
        image.set_shape([image_size, image_size, 3])
        mask.set_shape([image_size, image_size, 1])
        return image, mask

    # Create a dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Map the load_data function to preprocess the images and masks
    num_parallel_calls = tf.data.AUTOTUNE if data_config.num_parallel_calls == -1 else data_config.num_parallel_calls
    dataset = dataset.map(
        lambda img, mask: load_data(img, mask),
        num_parallel_calls=num_parallel_calls
    )

    # Cache after loading images if requested (for small datasets)
    if cache:
        dataset = dataset.cache()

    # Always shuffle training data
    # Use full dataset shuffle (MATLAB behavior), but cap at 50000 for memory safety
    shuffle_buffer_size = min(len(image_paths), 50000)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)

    # Repeat the dataset for continuous training
    if repeat:
        dataset = dataset.repeat()

    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch for performance
    dataset = dataset.prefetch(data_config.prefetch_buffer_size)

    return dataset


def create_validation_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    image_size: int,
    batch_size: int,
    data_config: Optional[DataConfig] = None,
    cache: bool = False
) -> tf.data.Dataset:
    """
    Create an optimized TensorFlow validation dataset without shuffling.

    This function creates a dataset specifically optimized for validation with:
    - Data loading and preprocessing
    - Optional caching of loaded images for performance
    - No shuffling (for consistent validation)
    - Prefetching for performance

    Pipeline order: load → cache (optional) → batch → prefetch

    Args:
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files
        image_size: Size to resize images and masks to (assumes square images)
        batch_size: Number of samples per batch
        data_config: DataConfig object with dataset configuration (if None, uses defaults)
        cache: Whether to cache the dataset in memory (recommended for validation sets)

    Returns:
        Optimized TensorFlow dataset for validation
    """
    if data_config is None:
        data_config = DataConfig()

    # Define wrapper functions outside of load_data to avoid AutoGraph issues
    def load_image_wrapper(path):
        return _load_image_pil_internal(path, image_size, False)

    def load_mask_wrapper(path):
        return _load_image_pil_internal(path, image_size, True)

    def load_data(image_path, mask_path):
        """Inner function to load an image and its corresponding mask using tf.py_function."""
        # Use tf.py_function to load TIFF files with PIL
        image = tf.py_function(
            func=load_image_wrapper,
            inp=[image_path],
            Tout=tf.float32
        )
        mask = tf.py_function(
            func=load_mask_wrapper,
            inp=[mask_path],
            Tout=tf.float32
        )
        # Set shapes explicitly (required after tf.py_function)
        image.set_shape([image_size, image_size, 3])
        mask.set_shape([image_size, image_size, 1])
        return image, mask

    # Create a dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Map the load_data function to preprocess the images and masks
    num_parallel_calls = tf.data.AUTOTUNE if data_config.num_parallel_calls == -1 else data_config.num_parallel_calls
    dataset = dataset.map(
        lambda img, mask: load_data(img, mask),
        num_parallel_calls=num_parallel_calls
    )

    # Cache after loading images if requested (useful for validation sets)
    if cache:
        dataset = dataset.cache()

    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch for performance
    dataset = dataset.prefetch(data_config.prefetch_buffer_size)

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
        mask_paths: List[str],
        shuffle: bool = False,
        shuffle_buffer_size: Optional[int] = None,
        seed: Optional[int] = None,
        cache: bool = False
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from lists of image and mask paths.

        Args:
            image_paths: List of paths to image files
            mask_paths: List of paths to mask files
            shuffle: Whether to shuffle the dataset
            shuffle_buffer_size: Size of the shuffle buffer
            seed: Random seed for shuffling
            cache: Whether to cache the dataset in memory

        Returns:
            TensorFlow dataset containing batches of (image, mask) pairs
        """
        return create_dataset(
            image_paths, mask_paths, self.image_size, self.batch_size,
            shuffle=shuffle, shuffle_buffer_size=shuffle_buffer_size,
            seed=seed, cache=cache
        )