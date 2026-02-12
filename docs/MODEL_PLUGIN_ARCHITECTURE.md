# CODAvision Model Plugin Architecture

## Overview

CODAvision uses a flexible plugin architecture that enables seamless integration of segmentation models from both TensorFlow and PyTorch frameworks. The architecture provides:

- **Framework-agnostic interface**: Abstract base classes define a consistent API across frameworks
- **Keras-compatible adapter**: PyTorch models can be wrapped to provide a Keras-like API
- **Factory pattern**: Unified model creation through `model_call()`
- **Automatic format conversion**: NHWC (TensorFlow) ↔ NCHW (PyTorch) handled transparently

## 1. Framework Enum and Abstract Interface

The framework-agnostic interface is defined in `base/models/base.py`:

```python
# base/models/base.py
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
        """Build and return the segmentation model."""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference on input data. Input: [B, H, W, C], Output: [B, H, W, num_classes]"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to file."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from file."""
        pass
```

The module also provides a utility function for detecting available frameworks:

```python
def detect_framework_availability() -> dict:
    """
    Detect which frameworks are available in the environment.

    Returns:
        Dictionary with keys: 'tensorflow', 'pytorch', 'tensorflow_version',
        'pytorch_version', 'tensorflow_gpu', 'pytorch_gpu'
    """
```

## 2. TensorFlow Implementation

TensorFlow models are implemented in `base/models/backbones_tf.py`:

```python
# base/models/backbones_tf.py
from base.models.base import BaseSegmentationModelInterface, Framework


class BaseSegmentationModel(BaseSegmentationModelInterface):
    """
    TensorFlow implementation of base segmentation model.

    Extends the framework-agnostic interface with TensorFlow-specific
    functionality while maintaining compatibility with the existing codebase.
    """

    def _get_framework(self) -> Framework:
        return Framework.TENSORFLOW

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not built. Call build_model() first.")
        return self.model.predict(x)

    def save(self, path: str) -> None:
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not built. Call build_model() first.")
        self.model.save(path)

    def load(self, path: str) -> None:
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not built. Call build_model() first.")
        self.model.load_weights(path)


class DeepLabV3Plus(BaseSegmentationModel):
    """
    TensorFlow implementation of DeepLabV3+ with ResNet50 backbone.

    Features ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context
    and a decoder with skip connections for detailed segmentation.
    """

    def build_model(self) -> Model:
        # Creates ResNet50 encoder with ASPP and decoder
        # Returns tf.keras.Model
        ...


class UNet(BaseSegmentationModel):
    """
    TensorFlow implementation of UNet with ResNet50 encoder.

    Classic encoder-decoder architecture with skip connections
    at corresponding resolutions.
    """

    def build_model(self) -> Model:
        # Creates ResNet50-based UNet
        # Returns tf.keras.Model
        ...
```

Available TensorFlow models: `DeepLabV3_plus`, `UNet`

## 3. PyTorch Implementation

PyTorch models are implemented in `base/models/backbones_pytorch.py`:

```python
# base/models/backbones_pytorch.py
from base.models.base import BaseSegmentationModelInterface, Framework


class PyTorchBaseSegmentationModel(BaseSegmentationModelInterface):
    """
    Abstract base class for PyTorch-based segmentation models.

    Provides common interface and utilities including device management
    and tensor format conversion (NCHW <-> NHWC).
    """

    def __init__(self, input_size: int, num_classes: int, l2_regularization_weight: float = 0.0):
        super().__init__(input_size, num_classes, l2_regularization_weight)
        self.device = get_pytorch_device()  # Auto-detects CUDA → MPS → CPU
        self.model: Optional[nn.Module] = None

    def _get_framework(self) -> Framework:
        return Framework.PYTORCH

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run prediction with automatic NHWC → NCHW → NHWC conversion.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        self.model.eval()
        with torch.no_grad():
            # Convert NumPy (NHWC) to PyTorch tensor (NCHW)
            x_tensor = torch.from_numpy(x).float()
            x_tensor = x_tensor.permute(0, 3, 1, 2)  # NHWC → NCHW
            x_tensor = x_tensor.to(self.device)

            output = self.model(x_tensor)

            # Convert back to NHWC format
            output = output.permute(0, 2, 3, 1)  # NCHW → NHWC
            output = output.cpu().numpy()

        return output

    def save(self, filepath: str) -> None:
        """Save model with configuration metadata."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'l2_regularization_weight': self.l2_regularization_weight,
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load model and restore configuration."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.input_size = checkpoint['input_size']
        self.num_classes = checkpoint['num_classes']
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])


class PyTorchDeepLabV3Plus(PyTorchBaseSegmentationModel):
    """
    PyTorch implementation of DeepLabV3+ with ResNet50 backbone.

    Matches the TensorFlow version including:
    - Preprocessing (RGB→BGR, ImageNet mean subtraction)
    - ASPP module with dilation rates [1, 6, 12, 18]
    - Decoder with skip connections
    """

    def build_model(self) -> nn.Module:
        model = DeepLabV3PlusModel(
            num_classes=self.num_classes,
            input_size=self.input_size
        )
        model = model.to(self.device)
        self.model = model
        return model
```

Available PyTorch models: `DeepLabV3_plus`

## 4. Keras-Compatible Adapter for PyTorch

The `PyTorchKerasAdapter` in `base/models/wrappers.py` wraps PyTorch models to provide a Keras-like API:

```python
# base/models/wrappers.py
class PyTorchKerasAdapter:
    """
    Adapter that wraps PyTorch models to provide a Keras-compatible API.

    Enables PyTorch models to be used with existing TensorFlow-style inference
    and training code. Handles automatic NHWC ↔ NCHW conversion.
    """

    def __init__(self, pytorch_model: nn.Module, device: Optional[str] = None):
        self.model = pytorch_model
        self.device = device or self._auto_detect_device()
        self._optimizer = None
        self._loss_fn = None
        self._compiled = False

    def predict(self, x: np.ndarray, batch_size: int = 32, verbose: int = 0) -> np.ndarray:
        """
        Generate predictions. Handles NHWC→NCHW→NHWC conversion automatically.

        Args:
            x: Input array in NHWC format (batch, height, width, channels)
            batch_size: Batch size for inference
            verbose: Verbosity mode

        Returns:
            Predictions in NHWC format (batch, height, width, num_classes)
        """

    def compile(self, optimizer='adam', loss=None, metrics=None, **kwargs):
        """Configure the model for training (Keras-compatible)."""

    def fit(self, x=None, y=None, batch_size=32, epochs=1, validation_data=None, **kwargs):
        """Train the model (Keras-compatible)."""

    def test_on_batch(self, x, y, **kwargs):
        """Test the model on a single batch."""

    def save(self, filepath, **kwargs):
        """Save model to .pth file."""

    def load_weights(self, filepath, **kwargs):
        """Load model weights from .pth file."""

    def summary(self):
        """Print model summary."""

    @property
    def trainable(self) -> bool:
        """Whether the model is trainable."""

    @trainable.setter
    def trainable(self, value: bool):
        """Set trainability for all parameters."""
```

Key features:
- **Automatic format conversion**: NHWC (TensorFlow) ↔ NCHW (PyTorch)
- **Full Keras API**: `predict()`, `compile()`, `fit()`, `test_on_batch()`, `save()`, `load_weights()`
- **Multi-device support**: CPU, CUDA, MPS (Apple Silicon)

## 5. Factory Function

The `model_call()` function in `base/models/backbones.py` provides unified model creation:

```python
# base/models/backbones.py
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
                   If None, uses configuration default from ModelDefaults.DEFAULT_FRAMEWORK.
        wrap_with_adapter: For PyTorch models, whether to wrap with PyTorchKerasAdapter.
                           True (default): Returns adapter with Keras-compatible API.
                           False: Returns raw nn.Module for PyTorch-native training.

    Returns:
        Model instance (TensorFlow Model, PyTorch nn.Module, or PyTorchKerasAdapter)

    Raises:
        ValueError: If model name or framework is invalid
        ImportError: If requested framework is not available
    """
    if framework is None:
        config = get_framework_config()
        framework = config['framework']

    framework = framework.lower()
    avail = detect_framework_availability()

    if framework == 'tensorflow':
        if not avail['tensorflow']:
            raise ImportError("TensorFlow not installed")

        if name == "UNet":
            return TFUNet(IMAGE_SIZE, NUM_CLASSES, l2_regularization_weight).build_model()
        elif name == "DeepLabV3_plus":
            return TFDeepLabV3Plus(IMAGE_SIZE, NUM_CLASSES, l2_regularization_weight).build_model()

    elif framework == 'pytorch':
        if not avail['pytorch']:
            raise ImportError("PyTorch not installed")

        if name not in ModelDefaults.PYTORCH_MODELS:
            raise ValueError(f"Model '{name}' not implemented in PyTorch. Available: {ModelDefaults.PYTORCH_MODELS}")

        if name == "DeepLabV3_plus":
            pytorch_model = PyTorchDeepLabV3Plus(IMAGE_SIZE, NUM_CLASSES, l2_regularization_weight).build_model()

            if wrap_with_adapter:
                return PyTorchKerasAdapter(pytorch_model)
            else:
                return pytorch_model
```

## 6. Model Availability

Models are configured in `base/config.py`:

```python
# base/config.py
class ModelDefaults:
    # Framework defaults
    DEFAULT_FRAMEWORK = "tensorflow"  # "pytorch" or "tensorflow"

    # Model availability matrix
    TENSORFLOW_MODELS = ["DeepLabV3_plus", "UNet"]
    PYTORCH_MODELS = ["DeepLabV3_plus"]
```

The framework is configured via `ModelDefaults.DEFAULT_FRAMEWORK` in `base/config.py`:
```python
class ModelDefaults:
    DEFAULT_FRAMEWORK = "pytorch"  # or "tensorflow"
```

## 7. Usage Examples

### Using the Factory (Recommended)

```python
from base.models.backbones import model_call

# Create model using default framework (from config/environment)
model = model_call("DeepLabV3_plus", 512, 5)

# Explicitly specify framework
tf_model = model_call("DeepLabV3_plus", 512, 5, framework="tensorflow")
pytorch_model = model_call("DeepLabV3_plus", 512, 5, framework="pytorch")

# Get raw PyTorch nn.Module (for custom training loops)
raw_model = model_call("DeepLabV3_plus", 512, 5, framework="pytorch", wrap_with_adapter=False)
```

### Manual Adapter Creation

```python
from base.models.backbones_pytorch import PyTorchDeepLabV3Plus
from base.models.wrappers import PyTorchKerasAdapter

# Build PyTorch model
builder = PyTorchDeepLabV3Plus(input_size=512, num_classes=5, l2_regularization_weight=0)
pytorch_model = builder.build_model()

# Wrap with Keras-compatible adapter
adapter = PyTorchKerasAdapter(pytorch_model)

# Use with Keras-style API
predictions = adapter.predict(images_nhwc)
adapter.save('model.pth')
```

### Framework Detection

```python
from base.models.base import detect_framework_availability

avail = detect_framework_availability()
print(f"TensorFlow: {avail['tensorflow']} (GPU: {avail['tensorflow_gpu']})")
print(f"PyTorch: {avail['pytorch']} (GPU: {avail['pytorch_gpu']})")
```

## 8. Design Principles

1. **Framework Agnostic**: `BaseSegmentationModelInterface` defines a consistent API without framework assumptions
2. **Plugin Pattern**: New models are added by implementing the interface and registering in `ModelDefaults`
3. **Adapter Pattern**: `PyTorchKerasAdapter` provides Keras API compatibility for PyTorch models
4. **Factory Pattern**: `model_call()` handles instantiation and framework routing
5. **Single Responsibility**: Each class has one clear purpose (interface, implementation, adapter, factory)
6. **Open/Closed Principle**: New models can be added without modifying existing code

## 9. Adding a New Model

To add a new model architecture:

1. **Implement the interface** in the appropriate framework module:
   ```python
   # base/models/backbones_pytorch.py
   class PyTorchMyModel(PyTorchBaseSegmentationModel):
       def build_model(self) -> nn.Module:
           return MyModelModule(self.num_classes, self.input_size)
   ```

2. **Register in config**:
   ```python
   # base/config.py
   PYTORCH_MODELS = ["DeepLabV3_plus", "MyModel"]
   ```

3. **Add to factory**:
   ```python
   # base/models/backbones.py
   elif name == "MyModel":
       pytorch_model = PyTorchMyModel(...).build_model()
   ```

The model is then automatically available through `model_call()` and the GUI.
