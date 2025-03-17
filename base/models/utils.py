"""
Model Utilities for Semantic Segmentation

This module provides common utilities for loading, saving, and managing 
semantic segmentation models and their metadata.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: March 13, 2025
"""

import os
import pickle
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import tensorflow as tf


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
        
        # Standardize model type format (replace '+' with '_plus')
        if 'model_type' in data and '+' in data['model_type']:
            data['model_type'] = data['model_type'].replace('+', '_plus')
        
        # Verify essential parameters
        essential_params = ['classNames', 'sxy', 'nblack', 'nwhite']
        missing_params = [param for param in essential_params if param not in data]
        
        if missing_params:
            raise ValueError(f"Missing required parameters in model data: {missing_params}")
        
        return data
    
    except Exception as e:
        raise ValueError(f"Failed to load model data: {e}")


def save_model_metadata(model_path: str, metadata: Dict[str, Any]) -> None:
    """
    Save model metadata to a pickle file.
    
    Args:
        model_path: Path to the directory where metadata will be saved
        metadata: Dictionary containing model metadata
    """
    os.makedirs(model_path, exist_ok=True)
    data_file = os.path.join(model_path, 'net.pkl')
    
    # Ensure model_type is standardized
    if 'model_type' in metadata and '+' in metadata['model_type']:
        metadata['model_type'] = metadata['model_type'].replace('+', '_plus')
    
    # Update existing file if it exists
    if os.path.exists(data_file):
        try:
            with open(data_file, 'rb') as f:
                existing_data = pickle.load(f)
            
            existing_data.update(metadata)
            with open(data_file, 'wb') as f:
                pickle.dump(existing_data, f)
        except Exception as e:
            print(f"Warning: Failed to update existing metadata file: {e}")
            # Fall back to creating a new file
            with open(data_file, 'wb') as f:
                pickle.dump(metadata, f)
    else:
        # Create a new file
        with open(data_file, 'wb') as f:
            pickle.dump(metadata, f)


def setup_gpu() -> Dict[str, Any]:
    """
    Configure TensorFlow to use GPU and return GPU information.
    
    Returns:
        Dictionary with GPU information or empty dict if no GPU is available
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    gpu_info = {}
    
    if physical_devices:
        try:
            # Set memory growth to avoid allocating all memory at once
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # Use only the first GPU
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            print(f"TensorFlow is using the following GPU: {logical_devices[0]}")
            
            # Get GPU memory info if possible
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # First GPU
                    gpu_info = {
                        'device': gpu.id,
                        'name': gpu.name,
                        'total_memory': f"{gpu.memoryTotal:.1f} MB",
                        'free_memory': f"{gpu.memoryFree:.1f} MB",
                        'used_memory': f"{gpu.memoryUsed:.1f} MB",
                        'utilization': f"{gpu.load * 100:.1f}%"
                    }
            except ImportError:
                print("GPUtil not available - limited GPU information will be displayed")
                gpu_info = {'device': 'GPU available but detailed info unavailable'}
                
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU available. Operations will proceed on the CPU.")
        print("Ensure that the NVIDIA GPU and CUDA are correctly installed if you intended to use a GPU.")
    
    return gpu_info


def calculate_class_weights(mask_list: List[str], num_classes: int) -> np.ndarray:
    """
    Calculate class weights based on pixel frequency in label masks.
    
    Args:
        mask_list: List of paths to mask images
        num_classes: Number of classes
        
    Returns:
        Array of class weights
    """
    class_pixels = np.zeros(num_classes)
    image_pixels = np.zeros(num_classes)
    epsilon = 1e-5  # Prevent division by zero
    
    for mask_path in mask_list:
        # Load mask
        mask = tf.keras.preprocessing.image.load_img(
            mask_path, color_mode='grayscale')
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        mask = mask.astype(int)
        
        # Count pixels per class
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        for val, count in zip(unique, counts):
            if val < num_classes:
                class_pixels[val] += count
                image_pixels[val] += total_pixels
    
    # Calculate frequencies
    freq = class_pixels / (image_pixels + epsilon)
    
    # Handle invalid values
    freq[np.isinf(freq) | np.isnan(freq)] = epsilon
    
    # Calculate weights using median frequency balancing
    median_freq = np.median(freq)
    class_weights = median_freq / (freq + epsilon)
    
    # Print class distribution information
    print("\nClass frequencies:")
    for i, f in enumerate(freq):
        print(f"Class {i}: {f:.4f}")
    
    print("\nClass weights:")
    for i, w in enumerate(class_weights):
        print(f"Class {i}: {w:.4f}")
    
    return class_weights.astype(np.float32)


def get_model_paths(model_path: str, model_type: str) -> Dict[str, str]:
    """
    Get standard paths for model files.
    
    Args:
        model_path: Base directory for the model
        model_type: Type of model (e.g., 'DeepLabV3_plus', 'UNet')
        
    Returns:
        Dictionary containing paths for model files
    """
    # Standardize model type format
    if '+' in model_type:
        model_type = model_type.replace('+', '_plus')
    
    return {
        'best_model': os.path.join(model_path, f'best_model_{model_type}.keras'),
        'final_model': os.path.join(model_path, f'{model_type}.keras'),
        'logs': os.path.join(model_path, 'logs'),
        'metadata': os.path.join(model_path, 'net.pkl'),
        'train_data': os.path.join(model_path, 'training'),
        'val_data': os.path.join(model_path, 'validation'),
        'confusion_matrix': os.path.join(model_path, f'confusion_matrix_{model_type}.png')
    }