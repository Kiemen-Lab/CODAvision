# CODAvision Model Plugin Architecture

## Overview
CODAvision uses a flexible plugin architecture that allows seamless integration of models from different deep learning frameworks. This document demonstrates how to add new segmentation architectures, including examples for both TensorFlow (current) and PyTorch (future).

## 1. Abstract Base Class Pattern

```python
# base/models/backbones.py
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseSegmentationModel(ABC):
    """
    Framework-agnostic base class for all segmentation models.
    """

    def __init__(self, input_size: int, num_classes: int,
                 l2_regularization_weight: float = 1e-4):
        """
        Initialize the segmentation model.

        Args:
            input_size: Size of input images (assumes square)
            num_classes: Number of segmentation classes
            l2_regularization_weight: Regularization strength
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.l2_regularization_weight = l2_regularization_weight
        self.framework = self._get_framework()

    @abstractmethod
    def _get_framework(self) -> str:
        """Return the framework name ('tensorflow' or 'pytorch')."""
        pass

    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and return the segmentation model.

        Returns:
            Model object (tf.keras.Model or torch.nn.Module)
        """
        pass
```

## 2. TensorFlow Implementation (Current)

```python
# base/models/backbones.py (continued)
import tensorflow as tf
from tensorflow.keras import layers, Model

class DeepLabV3Plus(BaseSegmentationModel):
    """TensorFlow implementation of DeepLabV3+."""

    def _get_framework(self) -> str:
        return "tensorflow"

    def build_model(self) -> Model:
        """Build DeepLabV3+ using TensorFlow/Keras."""
        # Input layer
        inputs = tf.keras.Input(shape=(self.input_size, self.input_size, 3))

        # Encoder (ResNet50 backbone)
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=inputs
        )

        # ASPP module
        x = base_model.output
        x = self._aspp_module(x)

        # Decoder
        x = self._decoder(x, base_model)

        # Output layer
        outputs = layers.Conv2D(
            self.num_classes,
            kernel_size=1,
            activation='softmax'
        )(x)

        return Model(inputs, outputs, name='DeepLabV3Plus')

    def _aspp_module(self, x):
        """Atrous Spatial Pyramid Pooling."""
        # Implementation details...
        return x

    def _decoder(self, x, encoder):
        """Decoder with skip connections."""
        # Implementation details...
        return x
```

## 3. PyTorch Implementation (Future)

```python
# base/models/backbones_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchSegmentationModel(BaseSegmentationModel):
    """Base class for PyTorch segmentation models."""

    def _get_framework(self) -> str:
        return "pytorch"

    def to_keras_compatible(self) -> Any:
        """
        Wrap PyTorch model for compatibility with existing pipeline.
        Returns a wrapper that mimics Keras API.
        """
        return PyTorchKerasWrapper(self.build_model())


class EfficientSegNet(PyTorchSegmentationModel):
    """Example PyTorch implementation of a custom architecture."""

    def build_model(self) -> nn.Module:
        """Build model using PyTorch."""
        return EfficientSegNetModule(
            in_channels=3,
            num_classes=self.num_classes,
            input_size=self.input_size
        )


class EfficientSegNetModule(nn.Module):
    """PyTorch module for EfficientSegNet."""

    def __init__(self, in_channels: int, num_classes: int, input_size: int):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Classifier
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc_features = self.encoder(x)

        # Decoder
        dec_features = self.decoder(enc_features)

        # Classification
        logits = self.classifier(dec_features)

        return F.softmax(logits, dim=1)
```

## 4. Framework Wrapper for Compatibility

```python
# base/models/wrappers.py
class PyTorchKerasWrapper:
    """
    Wrapper to make PyTorch models compatible with Keras-like API.
    Enables seamless integration with existing pipeline.
    """

    def __init__(self, pytorch_model: nn.Module):
        self.model = pytorch_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compile(self, optimizer, loss, metrics=None):
        """Mimic Keras compile method."""
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = self._get_loss_function(loss)
        self.metrics = metrics or []

    def fit(self, x=None, y=None, batch_size=32, epochs=10,
            validation_data=None, callbacks=None):
        """Mimic Keras fit method."""
        # Training loop implementation
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training step
            train_loss = self._train_epoch(x, y, batch_size)
            history['loss'].append(train_loss)

            # Validation step
            if validation_data:
                val_loss = self._validate(validation_data)
                history['val_loss'].append(val_loss)

        return history

    def predict(self, x, batch_size=32):
        """Mimic Keras predict method."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            # Convert numpy to tensor and predict
            x_tensor = torch.from_numpy(x).to(self.device)
            output = self.model(x_tensor)
            predictions = output.cpu().numpy()

        return predictions

    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.__class__.__name__
        }, filepath)

    def load_weights(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
```

## 5. Factory Pattern with Multi-Framework Support

```python
# base/models/backbones.py
def model_call(name: str, IMAGE_SIZE: int, NUM_CLASSES: int,
               l2_regularization_weight: float = 1e-4,
               framework: str = "auto") -> Any:
    """
    Factory function to create models from any supported framework.

    Args:
        name: Model architecture name
        IMAGE_SIZE: Input image size
        NUM_CLASSES: Number of classes
        l2_regularization_weight: Regularization weight
        framework: Framework to use ('tensorflow', 'pytorch', 'auto')

    Returns:
        Model instance (framework-specific or wrapped)
    """
    # Auto-detect framework if not specified
    if framework == "auto":
        framework = _detect_preferred_framework()

    # TensorFlow models
    tensorflow_models = {
        "DeepLabV3_plus": DeepLabV3Plus,
        "UNet": UNet,
        "PSPNet": PSPNet,  # Additional TF model
    }

    # PyTorch models
    pytorch_models = {
        "EfficientSegNet": EfficientSegNet,
        "TransUNet": TransUNet,  # Additional PyTorch model
        "SegFormer": SegFormer,  # Transformer-based model
    }

    # Check both registries
    if name in tensorflow_models:
        model_class = tensorflow_models[name]
        model = model_class(IMAGE_SIZE, NUM_CLASSES, l2_regularization_weight)
        return model.build_model()

    elif name in pytorch_models:
        model_class = pytorch_models[name]
        model = model_class(IMAGE_SIZE, NUM_CLASSES, l2_regularization_weight)
        # Return wrapped PyTorch model for compatibility
        return model.to_keras_compatible()

    else:
        available = list(tensorflow_models.keys()) + list(pytorch_models.keys())
        raise ValueError(
            f"Model '{name}' not found. Available models: {available}"
        )


def _detect_preferred_framework() -> str:
    """Detect which framework to prefer based on availability."""
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            return "tensorflow"
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return "pytorch"
    except ImportError:
        pass

    # Default to TensorFlow if both available
    return "tensorflow"
```

## 6. Unified Trainer with Framework Detection

```python
# base/models/training.py
class SegmentationModelTrainer:
    """
    Unified trainer that handles both TensorFlow and PyTorch models.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.framework = None
        self._load_model_data()

    def build_model(self):
        """Build model using appropriate framework."""
        model = model_call(
            name=self.model_type,
            IMAGE_SIZE=self.image_size,
            NUM_CLASSES=self.num_classes,
            l2_regularization_weight=self.l2_weight
        )

        # Detect framework from model
        if hasattr(model, '__module__'):
            if 'tensorflow' in model.__module__ or 'keras' in model.__module__:
                self.framework = 'tensorflow'
            elif 'torch' in model.__module__:
                self.framework = 'pytorch'
        elif isinstance(model, PyTorchKerasWrapper):
            self.framework = 'pytorch'
        else:
            self.framework = 'tensorflow'  # Default

        return model

    def train(self):
        """Train model using framework-appropriate methods."""
        self.model = self.build_model()

        if self.framework == 'tensorflow':
            self._train_tensorflow()
        elif self.framework == 'pytorch':
            self._train_pytorch()
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def _train_tensorflow(self):
        """TensorFlow-specific training logic."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
            callbacks=self._get_callbacks()
        )

        return history

    def _train_pytorch(self):
        """PyTorch-specific training logic (using wrapper)."""
        # The wrapper provides Keras-like API
        self.model.compile(
            optimizer='adam',
            loss='cross_entropy',
            metrics=['accuracy']
        )

        history = self.model.fit(
            x=self.train_data,
            y=self.train_labels,
            validation_data=(self.val_data, self.val_labels),
            epochs=self.epochs,
            batch_size=self.batch_size
        )

        return history
```

## 7. GUI Integration

```python
# gui/components/ui_definitions.py
def setup_model_dropdown(self):
    """Setup model selection dropdown with all available architectures."""

    # Get all available models from both frameworks
    from base.models.backbones import get_available_models

    self.model_type_CB = QComboBox(self.tab_4)

    # Add models grouped by framework
    available_models = get_available_models()

    for framework, models in available_models.items():
        for model_name in models:
            display_name = f"{model_name} ({framework[:2].upper()})"
            self.model_type_CB.addItem(display_name, model_name)

    self.model_type_CB.setObjectName(u"model_type_CB")


# base/models/backbones.py
def get_available_models() -> dict:
    """Return all available models grouped by framework."""
    return {
        "tensorflow": ["DeepLabV3_plus", "UNet", "PSPNet"],
        "pytorch": ["EfficientSegNet", "TransUNet", "SegFormer"]
    }
```

## 8. Usage Example

```python
# Example: Adding a new PyTorch model
class MyCustomModel(PyTorchSegmentationModel):
    """Custom PyTorch architecture."""

    def build_model(self) -> nn.Module:
        return MyCustomModule(
            num_classes=self.num_classes,
            input_size=self.input_size
        )

# Register in factory (base/models/backbones.py)
pytorch_models["MyCustomModel"] = MyCustomModel

# The model is now automatically available in:
# - GUI dropdown
# - Training pipeline
# - Inference pipeline
# No other changes needed!
```

## Key Design Principles

1. **Framework Agnostic**: Abstract base class doesn't assume any specific framework
2. **Plugin Pattern**: New models are simply added to the registry
3. **Adapter Pattern**: Wrappers provide consistent API across frameworks
4. **Factory Pattern**: Central function handles model instantiation
5. **Single Responsibility**: Each class has one clear purpose
6. **Open/Closed Principle**: Open for extension, closed for modification

This architecture ensures that adding new models (regardless of framework) requires minimal changes to existing code, maintaining backward compatibility while enabling future flexibility.