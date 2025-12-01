"""
Framework-agnostic base classes for semantic segmentation models.

This module provides abstract base classes and enums for both TensorFlow
and PyTorch implementations, ensuring a consistent interface.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
import numpy as np


class Framework(Enum):
    """Supported deep learning frameworks."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"


class BaseSegmentationModelInterface(ABC):
    """
    Abstract interface for semantic segmentation models across frameworks.

    This interface defines methods that both TensorFlow and PyTorch
    implementations must provide for seamless framework interchangeability.
    """

    def __init__(self, input_size: int, num_classes: int, l2_regularization_weight: float = 0):
        """
        Initialize the segmentation model.

        Args:
            input_size: Size of input images (assumes square images)
            num_classes: Number of segmentation classes
            l2_regularization_weight: L2 regularization weight (default: 0)

        Raises:
            ValueError: If l2_regularization_weight is negative
        """
        if l2_regularization_weight < 0:
            raise ValueError(f"L2 regularization weight must be >= 0, got {l2_regularization_weight}")

        self.input_size = input_size
        self.num_classes = num_classes
        self.l2_regularization_weight = l2_regularization_weight
        self.framework = self._get_framework()

    @abstractmethod
    def _get_framework(self) -> Framework:
        """Return the framework this model uses."""
        pass

    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and return the segmentation model.

        Returns:
            Model object (tf.keras.Model or torch.nn.Module)
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.

        Args:
            x: Input images [B, H, W, C] in numpy format

        Returns:
            Predictions [B, H, W, num_classes] as numpy array
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to file."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from file."""
        pass


def detect_framework_availability() -> dict:
    """
    Detect which frameworks are available in the environment.

    Returns:
        Dictionary with framework availability status
    """
    availability = {
        'tensorflow': False,
        'pytorch': False,
        'tensorflow_version': None,
        'pytorch_version': None,
        'tensorflow_gpu': False,
        'pytorch_gpu': False,
    }

    # Check TensorFlow
    try:
        import tensorflow as tf
        availability['tensorflow'] = True
        availability['tensorflow_version'] = tf.__version__
        availability['tensorflow_gpu'] = len(tf.config.list_physical_devices('GPU')) > 0
    except (ImportError, OSError, RuntimeError):
        pass

    # Check PyTorch
    try:
        import torch
        availability['pytorch'] = True
        availability['pytorch_version'] = torch.__version__
        availability['pytorch_gpu'] = torch.cuda.is_available() or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
    except (ImportError, OSError, RuntimeError):
        pass

    return availability
