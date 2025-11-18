"""
Model factory and framework router for semantic segmentation.

This module provides a unified interface for creating models across
different frameworks (TensorFlow, PyTorch).

Original DeepLabV3+ implementation based on: https://keras.io/examples/vision/deeplabv3_plus/
"""

from typing import Any

from base.config import get_framework_config, ModelDefaults
from base.models.base import Framework, detect_framework_availability

# Backward compatibility: re-export TensorFlow implementations
# This allows existing code to import directly from backbones module
from base.models.backbones_tf import (
    DeepLabV3Plus,
    UNet,
    BaseSegmentationModel
)


def model_call(
    name: str,
    IMAGE_SIZE: int,
    NUM_CLASSES: int,
    l2_regularization_weight: float = 0,
    framework: str = None,
    wrap_with_adapter: bool = True
) -> Any:
    """
    Factory function to create a segmentation model.

    Args:
        name: Model type ('UNet' or 'DeepLabV3_plus')
        IMAGE_SIZE: Size of input images
        NUM_CLASSES: Number of segmentation classes
        l2_regularization_weight: L2 regularization weight (default: 0)
        framework: Framework to use ('tensorflow' or 'pytorch').
                  If None, uses configuration default.
        wrap_with_adapter: For PyTorch models, whether to wrap with PyTorchKerasAdapter.
                          True (default): Returns adapter with Keras-compatible API for inference.
                          False: Returns raw nn.Module for PyTorch-native training.

    Returns:
        Model instance (TensorFlow, PyTorch, or PyTorchKerasAdapter)

    Raises:
        ValueError: If model name or framework is invalid
        ImportError: If requested framework is not available
    """
    # Determine framework
    if framework is None:
        config = get_framework_config()
        framework = config['framework']

    framework = framework.lower()

    # Check availability
    avail = detect_framework_availability()

    if framework == 'tensorflow':
        if not avail['tensorflow']:
            raise ImportError(
                "TensorFlow framework requested but not installed. "
                "Install with: pip install -e . "
                "Or switch to PyTorch with: export CODAVISION_FRAMEWORK=pytorch"
            )

        from base.models.backbones_tf import DeepLabV3Plus as TFDeepLabV3Plus
        from base.models.backbones_tf import UNet as TFUNet

        if name == "UNet":
            return TFUNet(IMAGE_SIZE, NUM_CLASSES, l2_regularization_weight).build_model()
        elif name == "DeepLabV3_plus":
            return TFDeepLabV3Plus(IMAGE_SIZE, NUM_CLASSES, l2_regularization_weight).build_model()
        else:
            raise ValueError(f'Invalid model name: {name}')

    elif framework == 'pytorch':
        if not avail['pytorch']:
            raise ImportError(
                "PyTorch framework requested but not installed. "
                "Install with: pip install -e '.[pytorch]' "
                "Or switch to TensorFlow with: export CODAVISION_FRAMEWORK=tensorflow"
            )

        # Check model availability in PyTorch
        if name not in ModelDefaults.PYTORCH_MODELS:
            raise ValueError(
                f"Model '{name}' not yet implemented in PyTorch. "
                f"Available: {ModelDefaults.PYTORCH_MODELS}"
            )

        from base.models.backbones_pytorch import PyTorchDeepLabV3Plus

        if name == "DeepLabV3_plus":
            pytorch_model = PyTorchDeepLabV3Plus(IMAGE_SIZE, NUM_CLASSES, l2_regularization_weight).build_model()

            # Conditionally wrap based on use case
            if wrap_with_adapter:
                # For TensorFlow-style inference (Keras API compatibility)
                from base.models.wrappers import PyTorchKerasAdapter
                return PyTorchKerasAdapter(pytorch_model)
            else:
                # For PyTorch-native training (raw nn.Module with .to(), .train(), etc.)
                return pytorch_model
        else:
            raise ValueError(f'PyTorch implementation for {name} not available')

    else:
        raise ValueError(f"Invalid framework: {framework}. Must be 'tensorflow' or 'pytorch'")


def unfreeze_model(model):
    """
    Unfreeze encoder layers for fine-tuning.
    Framework-agnostic wrapper.

    Args:
        model: The model to unfreeze (TensorFlow, PyTorch, or PyTorchKerasAdapter)

    Returns:
        Model with unfrozen encoder layers

    Raises:
        ValueError: If model framework cannot be determined
    """
    # Check for PyTorchKerasAdapter first (must come before nn.Module check)
    try:
        from base.models.wrappers import PyTorchKerasAdapter
        if isinstance(model, PyTorchKerasAdapter):
            from base.models.backbones_pytorch import unfreeze_model as torch_unfreeze
            # Unfreeze the wrapped PyTorch model
            torch_unfreeze(model.model)
            # Return the adapter (not the unwrapped model)
            return model
    except ImportError:
        pass

    # Check for TensorFlow model
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            from base.models.backbones_tf import unfreeze_model as tf_unfreeze
            return tf_unfreeze(model)
    except ImportError:
        pass

    # Check for raw PyTorch model (without adapter)
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            from base.models.backbones_pytorch import unfreeze_model as torch_unfreeze
            return torch_unfreeze(model)
    except ImportError:
        pass

    raise ValueError("Could not determine model framework")
