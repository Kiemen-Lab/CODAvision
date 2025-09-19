"""
Centralized Configuration Module for CODAvision

This module provides centralized configuration management for the CODAvision
package, eliminating magic numbers and hardcoded values throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum


class ThresholdDefaults:
    """Default values for tissue area threshold selection."""
    HE_DEFAULT = 205  # Default threshold for H&E stained images
    GRAYSCALE_DEFAULT = 128  # Default threshold for grayscale images
    BATCH_SIZE = 10  # Number of images to process in each batch
    BATCH_DELAY_MS = 100  # Delay between batches in milliseconds


class DisplayDefaults:
    """Default values for display and visualization."""
    MAX_WIDTH = 1500  # Maximum display width in pixels
    MAX_HEIGHT = 780  # Maximum display height in pixels
    REGION_SIZE = 600  # Default region size for threshold selection
    SAMPLE_SIZE = 20  # Default number of images to sample


class FileExtensions:
    """Supported file extensions for different image types."""
    TIFF_EXTENSIONS = ['.tif', '.tiff']
    STANDARD_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    ALL_IMAGE_EXTENSIONS = TIFF_EXTENSIONS + STANDARD_EXTENSIONS
    
    @classmethod
    def get_search_patterns(cls) -> List[str]:
        """Get glob patterns for all supported image types."""
        return ['*' + ext for ext in cls.ALL_IMAGE_EXTENSIONS]


class ModelDefaults:
    """Default values for model training and architecture."""
    INPUT_SIZE = 512  # Default input size for segmentation models
    NUM_FILTERS = 256  # Default number of filters in convolution blocks
    KERNEL_SIZE = 3  # Default kernel size for convolutions
    
    # Training defaults
    BATCH_SIZE = 8  # Default batch size for training
    LEARNING_RATE = 1e-4  # Default initial learning rate
    EPOCHS = 100  # Default number of training epochs
    
    # Early stopping and learning rate reduction
    ES_PATIENCE = 6  # Patience for early stopping
    LR_PATIENCE = 1  # Patience for learning rate reduction
    LR_FACTOR = 0.75  # Factor for learning rate reduction
    MIN_LR = 1e-7  # Minimum learning rate
    
    # Validation
    NUM_VALIDATIONS = 3  # Number of validation runs during training


class LoggingConfig:
    """Configuration for logging system."""
    LOG_ROTATION_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_ROTATION_BACKUP_COUNT = 5  # Keep 5 backup files
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEBUG_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


@dataclass
class GPUConfig:
    """Configuration for GPU usage and memory management."""
    memory_growth: bool = True  # Allow GPU memory growth
    memory_limit: float = 0.9  # Use up to 90% of GPU memory
    multi_gpu_strategy: str = 'mirrored'  # Strategy for multi-GPU training
    mixed_precision: bool = False  # Use mixed precision training


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    prefetch_buffer_size: int = 2  # Prefetch buffer size for tf.data
    shuffle_buffer_size: int = 1000  # Shuffle buffer size
    num_parallel_calls: int = -1  # Use tf.data.AUTOTUNE
    cache: bool = True  # Cache dataset in memory if possible


@dataclass
class RuntimeConfig:
    """Runtime configuration that can be modified at runtime."""
    debug_mode: bool = False
    verbose: bool = True
    use_gpu: bool = True
    num_workers: int = 4  # Number of parallel workers
    seed: int = 42  # Random seed for reproducibility
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


# Global runtime configuration instance
runtime_config = RuntimeConfig()


def get_threshold_config(mode: str = 'HE') -> int:
    """
    Get the default threshold value for a given mode.
    
    Args:
        mode: The threshold mode ('HE' or 'grayscale')
        
    Returns:
        Default threshold value for the specified mode
    """
    if mode.upper() == 'HE':
        return ThresholdDefaults.HE_DEFAULT
    else:
        return ThresholdDefaults.GRAYSCALE_DEFAULT


def get_display_config() -> dict:
    """
    Get display configuration as a dictionary.
    
    Returns:
        Dictionary containing display configuration values
    """
    return {
        'max_width': DisplayDefaults.MAX_WIDTH,
        'max_height': DisplayDefaults.MAX_HEIGHT,
        'region_size': DisplayDefaults.REGION_SIZE,
        'sample_size': DisplayDefaults.SAMPLE_SIZE
    }


def get_model_config() -> dict:
    """
    Get model configuration as a dictionary.
    
    Returns:
        Dictionary containing model configuration values
    """
    return {
        'input_size': ModelDefaults.INPUT_SIZE,
        'num_filters': ModelDefaults.NUM_FILTERS,
        'kernel_size': ModelDefaults.KERNEL_SIZE,
        'batch_size': ModelDefaults.BATCH_SIZE,
        'learning_rate': ModelDefaults.LEARNING_RATE,
        'epochs': ModelDefaults.EPOCHS,
        'es_patience': ModelDefaults.ES_PATIENCE,
        'lr_patience': ModelDefaults.LR_PATIENCE,
        'lr_factor': ModelDefaults.LR_FACTOR,
        'min_lr': ModelDefaults.MIN_LR,
        'num_validations': ModelDefaults.NUM_VALIDATIONS
    }