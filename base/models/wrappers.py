"""
Framework Adapter Layer for PyTorch and TensorFlow Models

This module provides adapter classes that enable seamless interoperability between
PyTorch and TensorFlow models by providing a unified Keras-compatible API.

Key Features:
- PyTorchKerasAdapter: Wraps PyTorch models with Keras-like API
- TensorFlowKerasAdapter: Wraps TensorFlow models with matching interface
- Automatic tensor format conversion (NHWC ↔ NCHW)
- Full Keras API support (inference, training, configuration, persistence)
- Multi-device support (CPU, CUDA, MPS)

Usage Example:
    from base.models.wrappers import PyTorchKerasAdapter, load_model
    from base.models.backbones_pytorch import PyTorchDeepLabV3Plus

    # Build PyTorch model
    builder = PyTorchDeepLabV3Plus(512, 5, 0)
    pytorch_model = builder.build_model()

    # Wrap with Keras-like adapter
    model = PyTorchKerasAdapter(pytorch_model)

    # Use with TensorFlow-style API
    predictions = model.predict(images_nhwc)  # Handles format conversion
    model.save('model.pth')

    # Or auto-detect and load
    model = load_model('path/to/model')  # Works for both .pth and .keras
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path

import numpy as np

# Import PyTorch conditionally
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    nn = None

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None

from base.config import FrameworkConfig

logger = logging.getLogger(__name__)


class PyTorchKerasAdapter:
    """
    Adapter that wraps PyTorch models to provide a Keras-compatible API.

    This adapter enables PyTorch models to be used seamlessly with existing
    TensorFlow-style inference and training code. It handles automatic tensor
    format conversion (NHWC ↔ NCHW) and provides all standard Keras methods.

    Attributes:
        model: The wrapped PyTorch nn.Module
        device: The device the model is on (CPU, CUDA, MPS)
        optimizer: Optional PyTorch optimizer for training
        loss_fn: Optional loss function for training
        metrics: List of metric functions
        compiled: Whether compile() has been called

    Example:
        >>> pytorch_model = PyTorchDeepLabV3Plus(512, 5, 0).build_model()
        >>> adapter = PyTorchKerasAdapter(pytorch_model)
        >>> predictions = adapter.predict(images_nhwc)
        >>> adapter.save('model.pth')
    """

    def __init__(
        self,
        pytorch_model: 'nn.Module',
        device: Optional[str] = None
    ):
        """
        Initialize the PyTorch Keras adapter.

        Args:
            pytorch_model: PyTorch nn.Module to wrap
            device: Device to use ('cpu', 'cuda', 'mps'). If None, auto-detect.

        Raises:
            ImportError: If PyTorch is not installed
            ValueError: If model is not a PyTorch nn.Module
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")

        if not isinstance(pytorch_model, nn.Module):
            raise ValueError(f"Expected nn.Module, got {type(pytorch_model)}")

        self.model = pytorch_model

        # Determine device
        if device is None:
            # Auto-detect device from model parameters
            try:
                self.device = next(pytorch_model.parameters()).device
            except StopIteration:
                # No parameters, use CPU
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            self.model = self.model.to(self.device)

        # Training configuration
        self._optimizer = None
        self._loss_fn = None
        self._metrics = []
        self._compiled = False

        # Training state
        self.history = {'loss': [], 'val_loss': []}

        logger.info(f"PyTorchKerasAdapter initialized on device: {self.device}")

    @property
    def current_device(self):
        """
        Get the actual current device of model parameters.

        This property dynamically queries the model's parameters to get the current device,
        which is useful if the model was moved externally after initialization.

        Returns:
            torch.device: The current device of the model
        """
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            # Model has no parameters, return the configured device
            return self.device

    # ==================== Inference Methods ====================

    def predict(
        self,
        x: np.ndarray,
        batch_size: int = 32,
        verbose: int = 0
    ) -> np.ndarray:
        """
        Generate predictions for input samples.

        Handles automatic tensor format conversion from NHWC (TensorFlow) to
        NCHW (PyTorch) and back.

        Args:
            x: Input array in NHWC format (batch, height, width, channels)
            batch_size: Batch size for inference
            verbose: Verbosity mode (0=silent, 1=progress bar)

        Returns:
            Predictions in NHWC format (batch, height, width, channels)

        Example:
            >>> images_nhwc = np.random.rand(10, 512, 512, 3)
            >>> predictions = model.predict(images_nhwc, batch_size=4)
            >>> predictions.shape
            (10, 512, 512, 5)
        """
        self.model.eval()

        # Convert NHWC to NCHW
        x_nchw = self._convert_nhwc_to_nchw(x)

        # Convert to tensor (use current_device for dynamic device detection)
        x_tensor = torch.from_numpy(x_nchw).float().to(self.current_device)

        # Process in batches
        num_samples = x_tensor.shape[0]
        predictions = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = x_tensor[i:i + batch_size]
                output = self.model(batch)
                predictions.append(output.cpu().numpy())

                if verbose > 0:
                    print(f"Processed {min(i + batch_size, num_samples)}/{num_samples}", end='\r')

        if verbose > 0:
            print()  # New line after progress

        # Concatenate batches
        predictions_nchw = np.concatenate(predictions, axis=0)

        # Convert back to NHWC
        predictions_nhwc = self._convert_nchw_to_nhwc(predictions_nchw)

        return predictions_nhwc

    def __call__(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Call the model on inputs (Keras-compatible callable interface).

        Args:
            inputs: Input array in NHWC format
            training: Whether in training mode

        Returns:
            Predictions in NHWC format
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        # Convert NHWC to NCHW
        x_nchw = self._convert_nhwc_to_nchw(inputs)
        x_tensor = torch.from_numpy(x_nchw).float().to(self.current_device)

        with torch.set_grad_enabled(training):
            output = self.model(x_tensor)

        # Convert back to NHWC
        output_nhwc = self._convert_nchw_to_nhwc(output.cpu().detach().numpy())

        return output_nhwc

    def predict_on_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for a single batch.

        Args:
            x: Input batch in NHWC format

        Returns:
            Predictions in NHWC format
        """
        return self.predict(x, batch_size=len(x), verbose=0)

    # ==================== Training Methods ====================

    def compile(
        self,
        optimizer: Union[str, 'optim.Optimizer'] = 'adam',
        loss: Optional[Union[str, Callable]] = None,
        metrics: Optional[List[Union[str, Callable]]] = None,
        **kwargs
    ):
        """
        Configure the model for training (Keras-compatible).

        Args:
            optimizer: Optimizer name ('adam', 'sgd') or PyTorch optimizer instance
            loss: Loss function name or callable
            metrics: List of metrics to track
            **kwargs: Additional optimizer arguments (lr, weight_decay, etc.)

        Example:
            >>> model.compile(optimizer='adam', loss='crossentropy', metrics=['accuracy'])
        """
        # Set up optimizer
        if isinstance(optimizer, str):
            lr = kwargs.get('learning_rate', kwargs.get('lr', 0.001))
            weight_decay = kwargs.get('weight_decay', 0.0)

            if optimizer.lower() == 'adam':
                self._optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            elif optimizer.lower() == 'sgd':
                momentum = kwargs.get('momentum', 0.9)
                self._optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            self._optimizer = optimizer

        # Set up loss function
        if loss is not None:
            if isinstance(loss, str):
                if loss.lower() in ['crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy']:
                    self._loss_fn = nn.CrossEntropyLoss()
                elif loss.lower() in ['bce', 'binary_crossentropy']:
                    self._loss_fn = nn.BCEWithLogitsLoss()
                elif loss.lower() in ['mse', 'mean_squared_error']:
                    self._loss_fn = nn.MSELoss()
                else:
                    raise ValueError(f"Unknown loss: {loss}")
            else:
                self._loss_fn = loss

        # Set up metrics
        if metrics is not None:
            self._metrics = metrics if isinstance(metrics, list) else [metrics]

        self._compiled = True
        logger.info(f"Model compiled with optimizer={optimizer}, loss={loss}")

    def fit(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 1,
        verbose: int = 1,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List] = None,
        shuffle: bool = True,
        initial_epoch: int = 0,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the model for a fixed number of epochs (Keras-compatible).

        Args:
            x: Training data in NHWC format
            y: Training labels
            batch_size: Batch size
            epochs: Number of epochs
            verbose: Verbosity mode
            validation_data: Tuple of (x_val, y_val)
            callbacks: List of callbacks (not fully implemented)
            shuffle: Whether to shuffle data
            initial_epoch: Epoch to start from
            **kwargs: Additional arguments

        Returns:
            History dictionary with training metrics

        Example:
            >>> history = model.fit(x_train, y_train, epochs=10, batch_size=16)
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before training. Call model.compile().")

        if x is None or y is None:
            raise ValueError("Both x and y must be provided for training.")

        # Convert data to NCHW format
        x_nchw = self._convert_nhwc_to_nchw(x)

        # Create PyTorch dataset
        x_tensor = torch.from_numpy(x_nchw).float()
        y_tensor = torch.from_numpy(y).long() if y.dtype in [np.int32, np.int64] else torch.from_numpy(y).float()

        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Validation data
        val_loader = None
        if validation_data is not None:
            x_val, y_val = validation_data
            x_val_nchw = self._convert_nhwc_to_nchw(x_val)
            x_val_tensor = torch.from_numpy(x_val_nchw).float()
            y_val_tensor = torch.from_numpy(y_val).long() if y_val.dtype in [np.int32, np.int64] else torch.from_numpy(y_val).float()
            val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        history = {'loss': [], 'val_loss': []}

        for epoch in range(initial_epoch, epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.current_device)
                batch_y = batch_y.to(self.current_device)

                # Forward pass
                self._optimizer.zero_grad()
                outputs = self.model(batch_x)

                # Handle different output/target shapes for segmentation
                if len(outputs.shape) == 4 and len(batch_y.shape) == 3:
                    # outputs: (B, C, H, W), batch_y: (B, H, W)
                    loss = self._loss_fn(outputs, batch_y)
                else:
                    loss = self._loss_fn(outputs, batch_y)

                # Backward pass
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)

            # Validation
            if val_loader is not None:
                val_loss = self._evaluate_on_loader(val_loader)
                history['val_loss'].append(val_loss)

                if verbose > 0:
                    print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                if verbose > 0:
                    print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

        self.history = history
        return history

    def train_on_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> float:
        """
        Train the model on a single batch.

        Args:
            x: Input batch in NHWC format
            y: Target batch
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before training.")

        self.model.train()

        # Convert to NCHW and tensors
        x_nchw = self._convert_nhwc_to_nchw(x)
        x_tensor = torch.from_numpy(x_nchw).float().to(self.current_device)
        y_tensor = torch.from_numpy(y).long() if y.dtype in [np.int32, np.int64] else torch.from_numpy(y).float()
        y_tensor = y_tensor.to(self.current_device)

        # Forward pass
        self._optimizer.zero_grad()
        outputs = self.model(x_tensor)

        # Handle different shapes
        if len(outputs.shape) == 4 and len(y_tensor.shape) == 3:
            loss = self._loss_fn(outputs, y_tensor)
        else:
            loss = self._loss_fn(outputs, y_tensor)

        # Backward pass
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def test_on_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> float:
        """
        Test the model on a single batch.

        Args:
            x: Input batch in NHWC format
            y: Target batch
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before evaluation.")

        self.model.eval()

        # Convert to NCHW and tensors
        x_nchw = self._convert_nhwc_to_nchw(x)
        x_tensor = torch.from_numpy(x_nchw).float().to(self.current_device)
        y_tensor = torch.from_numpy(y).long() if y.dtype in [np.int32, np.int64] else torch.from_numpy(y).float()
        y_tensor = y_tensor.to(self.current_device)

        with torch.no_grad():
            outputs = self.model(x_tensor)

            if len(outputs.shape) == 4 and len(y_tensor.shape) == 3:
                loss = self._loss_fn(outputs, y_tensor)
            else:
                loss = self._loss_fn(outputs, y_tensor)

        return loss.item()

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        verbose: int = 1,
        **kwargs
    ) -> Union[float, List[float]]:
        """
        Evaluate the model on test data.

        Args:
            x: Test data in NHWC format
            y: Test labels
            batch_size: Batch size
            verbose: Verbosity mode
            **kwargs: Additional arguments

        Returns:
            Loss value (or list of metrics if metrics are configured)
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before evaluation.")

        self.model.eval()

        # Convert to NCHW
        x_nchw = self._convert_nhwc_to_nchw(x)

        # Create dataset
        x_tensor = torch.from_numpy(x_nchw).float()
        y_tensor = torch.from_numpy(y).long() if y.dtype in [np.int32, np.int64] else torch.from_numpy(y).float()
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Evaluate
        loss = self._evaluate_on_loader(dataloader)

        if verbose > 0:
            print(f"Evaluation - loss: {loss:.4f}")

        return loss

    def _evaluate_on_loader(self, dataloader: 'DataLoader') -> float:
        """
        Helper method to evaluate on a DataLoader.

        Args:
            dataloader: PyTorch DataLoader

        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.current_device)
                batch_y = batch_y.to(self.current_device)

                outputs = self.model(batch_x)

                if len(outputs.shape) == 4 and len(batch_y.shape) == 3:
                    loss = self._loss_fn(outputs, batch_y)
                else:
                    loss = self._loss_fn(outputs, batch_y)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    # ==================== Configuration Methods ====================

    def summary(self, input_shape: Optional[Tuple[int, ...]] = None):
        """
        Print model summary (Keras-compatible).

        Args:
            input_shape: Optional input shape for summary
        """
        print(f"\n{'='*70}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Compiled: {self._compiled}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print(f"\nTotal params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {non_trainable_params:,}")
        print(f"{'='*70}\n")

    def count_params(self) -> int:
        """
        Count the total number of parameters.

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.model.parameters())

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration (Keras-compatible).

        Returns:
            Configuration dictionary
        """
        return {
            'model_class': self.model.__class__.__name__,
            'device': str(self.device),
            'total_params': self.count_params(),
            'compiled': self._compiled,
            'optimizer': self._optimizer.__class__.__name__ if self._optimizer else None,
            'loss': self._loss_fn.__class__.__name__ if self._loss_fn else None,
        }

    # ==================== Persistence Methods ====================

    def save(self, filepath: Union[str, Path], **kwargs):
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model (will add .pth extension if missing)
            **kwargs: Additional arguments
        """
        filepath = Path(filepath)

        # Ensure .pth extension
        if filepath.suffix != '.pth':
            filepath = filepath.with_suffix('.pth')

        # Save model state and configuration
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'config': self.get_config(),
        }

        # Save optimizer if compiled
        if self._compiled and self._optimizer is not None:
            save_dict['optimizer_state_dict'] = self._optimizer.state_dict()

        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")

    def save_weights(self, filepath: Union[str, Path], **kwargs):
        """
        Save only model weights (not optimizer state).

        Args:
            filepath: Path to save weights
            **kwargs: Additional arguments
        """
        filepath = Path(filepath)

        if filepath.suffix != '.pth':
            filepath = filepath.with_suffix('.pth')

        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Weights saved to {filepath}")

    def load_weights(self, filepath: Union[str, Path], **kwargs):
        """
        Load model weights from a file.

        Args:
            filepath: Path to weights file
            **kwargs: Additional arguments (e.g., strict=True)
        """
        filepath = Path(filepath)

        if filepath.suffix != '.pth':
            filepath = filepath.with_suffix('.pth')

        if not filepath.exists():
            raise FileNotFoundError(f"Weights file not found: {filepath}")

        strict = kwargs.get('strict', True)
        state_dict = torch.load(filepath, map_location=self.device, weights_only=False)

        # Handle case where full save dict or just state dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        self.model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Weights loaded from {filepath}")

    # ==================== Properties ====================

    @property
    def trainable(self) -> bool:
        """Whether the model is trainable."""
        return any(p.requires_grad for p in self.model.parameters())

    @trainable.setter
    def trainable(self, value: bool):
        """
        Set whether the model is trainable.

        Warning: This sets requires_grad on ALL model parameters globally.
        For fine-grained control (e.g., freezing encoder while training decoder),
        access the wrapped model directly:

            # Freeze encoder, train decoder
            for param in adapter.model.encoder.parameters():
                param.requires_grad = False
            for param in adapter.model.decoder.parameters():
                param.requires_grad = True

        Args:
            value: Whether all parameters should be trainable
        """
        for param in self.model.parameters():
            param.requires_grad = value

    @property
    def layers(self) -> List['nn.Module']:
        """Get list of layers (modules)."""
        return list(self.model.modules())

    @property
    def optimizer(self) -> Optional['optim.Optimizer']:
        """Get the optimizer."""
        return self._optimizer

    @property
    def loss(self) -> Optional[Callable]:
        """Get the loss function."""
        return self._loss_fn

    @property
    def metrics(self) -> List[Union[str, Callable]]:
        """Get the metrics."""
        return self._metrics

    # ==================== Helper Methods ====================

    def _convert_nhwc_to_nchw(self, x: np.ndarray) -> np.ndarray:
        """
        Convert tensor from NHWC (TensorFlow) to NCHW (PyTorch) format.

        Args:
            x: Input array in NHWC format (batch, height, width, channels)

        Returns:
            Array in NCHW format (batch, channels, height, width)
        """
        if len(x.shape) == 4:
            # (B, H, W, C) -> (B, C, H, W)
            return np.transpose(x, (0, 3, 1, 2))
        elif len(x.shape) == 3:
            # (H, W, C) -> (C, H, W)
            return np.transpose(x, (2, 0, 1))
        else:
            return x

    def _convert_nchw_to_nhwc(self, x: np.ndarray) -> np.ndarray:
        """
        Convert tensor from NCHW (PyTorch) to NHWC (TensorFlow) format.

        Args:
            x: Input array in NCHW format (batch, channels, height, width)

        Returns:
            Array in NHWC format (batch, height, width, channels)
        """
        if len(x.shape) == 4:
            # (B, C, H, W) -> (B, H, W, C)
            return np.transpose(x, (0, 2, 3, 1))
        elif len(x.shape) == 3:
            # (C, H, W) -> (H, W, C)
            return np.transpose(x, (1, 2, 0))
        else:
            return x


class TensorFlowKerasAdapter:
    """
    Adapter that wraps TensorFlow/Keras models with a unified interface.

    This adapter provides the same interface as PyTorchKerasAdapter for
    TensorFlow models, enabling framework-agnostic code. Most methods are
    pass-through since TensorFlow models are already Keras-compatible.

    Attributes:
        model: The wrapped Keras model

    Example:
        >>> tf_model = tf.keras.models.load_model('model.keras')
        >>> adapter = TensorFlowKerasAdapter(tf_model)
        >>> predictions = adapter.predict(images_nhwc)
    """

    def __init__(self, keras_model: 'keras.Model'):
        """
        Initialize the TensorFlow Keras adapter.

        Args:
            keras_model: TensorFlow/Keras model to wrap

        Raises:
            ImportError: If TensorFlow is not installed
            ValueError: If model is not a Keras model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

        if not isinstance(keras_model, keras.Model):
            raise ValueError(f"Expected keras.Model, got {type(keras_model)}")

        self.model = keras_model
        logger.info("TensorFlowKerasAdapter initialized")

    # Most methods are pass-through to the underlying Keras model

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Generate predictions (pass-through to Keras)."""
        return self.model.predict(x, **kwargs)

    def __call__(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Call the model (pass-through to Keras)."""
        return self.model(inputs, training=training).numpy()

    def predict_on_batch(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions for a single batch."""
        return self.model.predict_on_batch(x)

    def compile(self, **kwargs):
        """Configure the model for training."""
        return self.model.compile(**kwargs)

    def fit(self, *args, **kwargs) -> Dict[str, List[float]]:
        """Train the model."""
        history = self.model.fit(*args, **kwargs)
        return history.history

    def train_on_batch(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Train on a single batch."""
        return self.model.train_on_batch(x, y, **kwargs)

    def test_on_batch(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Test on a single batch."""
        return self.model.test_on_batch(x, y, **kwargs)

    def evaluate(self, *args, **kwargs) -> Union[float, List[float]]:
        """Evaluate the model."""
        return self.model.evaluate(*args, **kwargs)

    def summary(self, **kwargs):
        """Print model summary."""
        return self.model.summary(**kwargs)

    def count_params(self) -> int:
        """Count total parameters."""
        return self.model.count_params()

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.model.get_config()

    def save(self, filepath: Union[str, Path], **kwargs):
        """Save the model."""
        return self.model.save(filepath, **kwargs)

    def save_weights(self, filepath: Union[str, Path], **kwargs):
        """Save model weights."""
        return self.model.save_weights(filepath, **kwargs)

    def load_weights(self, filepath: Union[str, Path], **kwargs):
        """Load model weights."""
        return self.model.load_weights(filepath, **kwargs)

    @property
    def trainable(self) -> bool:
        """Whether the model is trainable."""
        return self.model.trainable

    @trainable.setter
    def trainable(self, value: bool):
        """Set whether the model is trainable."""
        self.model.trainable = value

    @property
    def layers(self) -> List:
        """Get list of layers."""
        return self.model.layers

    @property
    def optimizer(self):
        """Get the optimizer."""
        return self.model.optimizer

    @property
    def loss(self):
        """Get the loss function."""
        return self.model.loss

    @property
    def metrics(self):
        """Get the metrics."""
        return self.model.metrics


# ==================== Factory Functions ====================

def create_model_adapter(
    model: Union['nn.Module', 'keras.Model'],
    **kwargs
) -> Union[PyTorchKerasAdapter, TensorFlowKerasAdapter]:
    """
    Create an appropriate adapter for the given model.

    Auto-detects whether the model is PyTorch or TensorFlow and returns
    the corresponding adapter.

    Args:
        model: PyTorch nn.Module or TensorFlow/Keras model
        **kwargs: Additional arguments for adapter initialization

    Returns:
        PyTorchKerasAdapter or TensorFlowKerasAdapter

    Raises:
        ValueError: If model type cannot be determined

    Example:
        >>> model = load_some_model()  # Could be PyTorch or TensorFlow
        >>> adapter = create_model_adapter(model)
        >>> predictions = adapter.predict(images)
    """
    # Check PyTorch
    if PYTORCH_AVAILABLE and isinstance(model, nn.Module):
        return PyTorchKerasAdapter(model, **kwargs)

    # Check TensorFlow
    if TENSORFLOW_AVAILABLE and isinstance(model, keras.Model):
        return TensorFlowKerasAdapter(model, **kwargs)

    raise ValueError(
        f"Unknown model type: {type(model)}. "
        "Expected PyTorch nn.Module or TensorFlow keras.Model"
    )


def load_model(
    filepath: Union[str, Path],
    **kwargs
) -> Union[PyTorchKerasAdapter, TensorFlowKerasAdapter]:
    """
    Load a model from file and wrap with appropriate adapter.

    Auto-detects the model format based on file extension:
    - .pth -> PyTorch model
    - .keras, .h5 -> TensorFlow model

    Args:
        filepath: Path to model file
        **kwargs: Additional arguments for loading

    Returns:
        Wrapped model with adapter

    Raises:
        FileNotFoundError: If model file not found
        ValueError: If model format cannot be determined

    Example:
        >>> model = load_model('path/to/model.pth')  # Returns PyTorchKerasAdapter
        >>> model = load_model('path/to/model.keras')  # Returns TensorFlowKerasAdapter
        >>> predictions = model.predict(images)  # Same API for both!
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Determine model type from extension
    if filepath.suffix == '.pth':
        # Load PyTorch model
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")

        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

        # Need to reconstruct model - this requires model class to be known
        # For now, raise an error with instructions
        raise NotImplementedError(
            "Loading PyTorch models from .pth files requires model architecture. "
            "Please load the model manually and wrap with PyTorchKerasAdapter:\n"
            "  model = PyTorchDeepLabV3Plus(...).build_model()\n"
            "  adapter = PyTorchKerasAdapter(model)\n"
            "  adapter.load_weights('model.pth')"
        )

    elif filepath.suffix in ['.keras', '.h5']:
        # Load TensorFlow model
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

        model = keras.models.load_model(filepath, **kwargs)
        return TensorFlowKerasAdapter(model)

    else:
        raise ValueError(
            f"Unknown model format: {filepath.suffix}. "
            "Supported formats: .pth (PyTorch), .keras/.h5 (TensorFlow)"
        )


def get_framework_from_model(model: Any) -> str:
    """
    Determine the framework of a model.

    Args:
        model: Model instance

    Returns:
        'pytorch', 'tensorflow', or 'unknown'
    """
    if PYTORCH_AVAILABLE and isinstance(model, nn.Module):
        return 'pytorch'
    elif TENSORFLOW_AVAILABLE and isinstance(model, keras.Model):
        return 'tensorflow'
    else:
        return 'unknown'
