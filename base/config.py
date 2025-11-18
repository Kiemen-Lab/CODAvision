"""
Centralized Configuration Module for CODAvision

This module provides centralized configuration management for the CODAvision
package, eliminating magic numbers and hardcoded values throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import os


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

    # Optimizer defaults
    OPTIMIZER_EPSILON = 1e-8  # Epsilon value for Adam and AdamW optimizers (for numerical stability)

    # Early stopping and learning rate reduction
    ES_PATIENCE = 6  # Patience for early stopping
    LR_PATIENCE = 1  # Patience for learning rate reduction
    LR_FACTOR = 0.75  # Factor for learning rate reduction
    MIN_LR = 1e-7  # Minimum learning rate

    # Validation
    VALIDATION_FREQUENCY = 128  # Number of iterations between validations

    # Tile Generation
    TILE_GENERATION_MODE = "modern"  # Default tile generation mode ("modern" or "legacy")

    # ===== FRAMEWORK CONFIGURATION =====
    DEFAULT_FRAMEWORK = "pytorch"  # or "tensorflow"

    # PyTorch-specific settings
    PYTORCH_DEVICE = "auto"  # auto, cuda, mps, cpu
    PYTORCH_COMPILE = False  # torch.compile() optimization (PyTorch 2.0+)
    PYTORCH_AMP = False  # Automatic Mixed Precision
    GRADIENT_ACCUMULATION_STEPS = 1  # Gradient accumulation for larger effective batch size

    # Model availability matrix
    TENSORFLOW_MODELS = ["DeepLabV3_plus", "UNet"]
    PYTORCH_MODELS = ["DeepLabV3_plus"]  # Start with DeepLabV3+


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


@dataclass
class TileGenerationConfig:
    """Configuration for tile generation behavior."""
    mode: str = "modern"  # "modern" or "legacy" or custom name
    reduction_factor: int = 10
    use_disk_filter: bool = False
    crop_rotations: bool = False
    class_rotation_frequency: int = 5
    deterministic_seed: Optional[int] = 3
    big_tile_size: int = 10240
    file_format: str = "png"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate mode (allow custom modes for flexibility)
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError(f"mode must be a non-empty string, got '{self.mode}'")

        # Validate reduction_factor
        if not 1 <= self.reduction_factor <= 20:
            raise ValueError(f"reduction_factor must be 1-20, got {self.reduction_factor}")

        # Validate file_format and normalize to 3-char extension
        if not isinstance(self.file_format, str):
            raise ValueError(f"file_format must be a string, got {type(self.file_format)}")

        # Normalize file format to 3-char extension
        format_lower = self.file_format.lower()
        if format_lower in ("tif", "tiff"):
            self.file_format = "tif"
        elif format_lower == "png":
            self.file_format = "png"
        elif format_lower in ("jpg", "jpeg"):
            self.file_format = "jpg"
        else:
            raise ValueError(f"file_format must be 'tif'/'tiff', 'png', or 'jpg'/'jpeg', got '{self.file_format}'")


# Preset tile generation configurations
MODERN_CONFIG = TileGenerationConfig(
    mode="modern",
    reduction_factor=10,
    use_disk_filter=False,
    crop_rotations=False,
    class_rotation_frequency=5,
    deterministic_seed=3,
    big_tile_size=10240,
    file_format="png"
)

LEGACY_CONFIG = TileGenerationConfig(
    mode="legacy",
    reduction_factor=5,
    use_disk_filter=True,
    crop_rotations=True,
    class_rotation_frequency=3,
    deterministic_seed=None,
    big_tile_size=10000,
    file_format="tif"
)


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
        'validation_frequency': ModelDefaults.VALIDATION_FREQUENCY
    }


def get_default_tile_config() -> TileGenerationConfig:
    """
    Get default tile configuration from environment or ModelDefaults.

    Checks the CODAVISION_TILE_GENERATION_MODE environment variable first,
    falling back to ModelDefaults.TILE_GENERATION_MODE if not set.

    Returns:
        TileGenerationConfig: Either MODERN_CONFIG or LEGACY_CONFIG preset

    Raises:
        ValueError: If the mode is not 'modern' or 'legacy'
    """
    mode = os.environ.get('CODAVISION_TILE_GENERATION_MODE',
                         ModelDefaults.TILE_GENERATION_MODE).lower()

    if mode == "legacy":
        return LEGACY_CONFIG
    elif mode == "modern":
        return MODERN_CONFIG
    else:
        raise ValueError(f"Invalid TILE_GENERATION_MODE: '{mode}'. Must be 'modern' or 'legacy'.")


def get_framework_config() -> dict:
    """
    Get framework configuration from environment or defaults.

    Environment variables:
        CODAVISION_FRAMEWORK: 'tensorflow' or 'pytorch'
        CODAVISION_PYTORCH_DEVICE: 'auto', 'cuda', 'mps', 'cpu'
        CODAVISION_PYTORCH_COMPILE: '1' or '0'
        CODAVISION_PYTORCH_AMP: '1' or '0'
        CODAVISION_GRADIENT_ACCUMULATION_STEPS: int (default: 1)

    Returns:
        Dictionary with framework configuration

    Raises:
        ValueError: If framework is not 'tensorflow' or 'pytorch'
    """
    framework = os.environ.get('CODAVISION_FRAMEWORK', ModelDefaults.DEFAULT_FRAMEWORK).lower()

    if framework not in ['tensorflow', 'pytorch']:
        raise ValueError(f"Invalid framework: {framework}. Must be 'tensorflow' or 'pytorch'")

    return {
        'framework': framework,
        'pytorch_device': os.environ.get('CODAVISION_PYTORCH_DEVICE', ModelDefaults.PYTORCH_DEVICE),
        'pytorch_compile': os.environ.get('CODAVISION_PYTORCH_COMPILE', '0') == '1',
        'pytorch_amp': os.environ.get('CODAVISION_PYTORCH_AMP', '0') == '1',
        'gradient_accumulation_steps': int(os.environ.get('CODAVISION_GRADIENT_ACCUMULATION_STEPS',
                                                          str(ModelDefaults.GRADIENT_ACCUMULATION_STEPS))),
    }

class FrameworkConfig:
    """
    Framework configuration manager for PyTorch/TensorFlow switching.

    Provides a unified API for getting and setting the deep learning framework.
    """

    @staticmethod
    def get_framework() -> str:
        """
        Get the current framework ('pytorch' or 'tensorflow').

        Returns:
            Current framework name

        Example:
            >>> framework = FrameworkConfig.get_framework()
            >>> print(framework)
            'pytorch'
        """
        config = get_framework_config()
        return config['framework']

    @staticmethod
    def set_framework(framework: str):
        """
        Set the framework by updating the environment variable.

        Args:
            framework: Framework name ('pytorch' or 'tensorflow')

        Raises:
            ValueError: If framework is not valid

        Example:
            >>> FrameworkConfig.set_framework('pytorch')
            >>> assert FrameworkConfig.get_framework() == 'pytorch'
        """
        framework = framework.lower()
        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError(f"Invalid framework: {framework}. Must be 'tensorflow' or 'pytorch'")

        os.environ['CODAVISION_FRAMEWORK'] = framework

    @staticmethod
    def get_device() -> str:
        """Get the PyTorch device setting."""
        config = get_framework_config()
        return config['pytorch_device']

    @staticmethod
    def is_pytorch_compile_enabled() -> bool:
        """Check if PyTorch compilation is enabled."""
        config = get_framework_config()
        return config['pytorch_compile']

    @staticmethod
    def is_amp_enabled() -> bool:
        """Check if Automatic Mixed Precision is enabled."""
        config = get_framework_config()
        return config['pytorch_amp']

    @staticmethod
    def get_gradient_accumulation_steps() -> int:
        """Get gradient accumulation steps."""
        config = get_framework_config()
        return config['gradient_accumulation_steps']

    @staticmethod
    def get_all_config() -> dict:
        """Get complete framework configuration."""
        return get_framework_config()
