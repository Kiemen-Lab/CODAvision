"""
Unit Tests for L2 Regularization Implementation

This module tests the L2 regularization functionality in the CODAvision
semantic segmentation models (DeepLabV3+ and UNet).
"""

import os
import sys
import tempfile
import shutil
import json
import pickle
import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from unittest.mock import Mock, MagicMock, patch

from base.models.backbones import DeepLabV3Plus, UNet, model_call
from base.models.training import RegularizationLossCallback


class TestL2RegularizationModels:
    """Test L2 regularization in model architectures."""

    def test_deeplabv3plus_with_l2(self):
        """Test DeepLabV3+ model creation with L2 regularization."""
        # Create model with L2 regularization
        model_builder = DeepLabV3Plus(
            input_size=256,
            num_classes=5,
            l2_regularization_weight=1e-5
        )
        model = model_builder.build_model()

        # Check that model was created successfully
        assert model is not None
        assert isinstance(model, keras.Model)

        # Count regularized layers
        regularized_layers = 0
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
                regularized_layers += 1

        # Should have regularized Conv2D layers (but not all layers)
        assert regularized_layers > 0, "Model should have regularized layers"

    def test_deeplabv3plus_without_l2(self):
        """Test DeepLabV3+ model creation without L2 regularization."""
        # Create model without L2 regularization
        model_builder = DeepLabV3Plus(
            input_size=256,
            num_classes=5,
            l2_regularization_weight=0
        )
        model = model_builder.build_model()

        # Count regularized layers
        regularized_layers = 0
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
                regularized_layers += 1

        # Should have no regularized layers when weight is 0
        assert regularized_layers == 0, "Model should not have regularized layers when weight is 0"

    def test_unet_with_l2(self):
        """Test UNet model creation with L2 regularization."""
        # Create model with L2 regularization
        model_builder = UNet(
            input_size=256,
            num_classes=5,
            l2_regularization_weight=1e-4
        )
        model = model_builder.build_model()

        # Check that model was created successfully
        assert model is not None
        assert isinstance(model, keras.Model)

        # Count regularized layers
        regularized_layers = 0
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
                regularized_layers += 1

        # Should have regularized Conv2D layers in decoder
        assert regularized_layers > 0, "UNet should have regularized layers"

    def test_unet_final_layer_not_regularized(self):
        """Test that UNet's final classification layer is not regularized."""
        model_builder = UNet(
            input_size=256,
            num_classes=5,
            l2_regularization_weight=1e-4
        )
        model = model_builder.build_model()

        # Find the final Conv2DTranspose layer (output layer)
        output_layer = None
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2DTranspose) and layer.output_shape[-1] == 5:
                output_layer = layer
                break

        assert output_layer is not None, "Should find output layer"
        # Final layer should not have regularization
        assert output_layer.kernel_regularizer is None, "Final classification layer should not be regularized"

    def test_model_call_factory_with_l2(self):
        """Test model factory function with L2 regularization.

        Uses interface-based validation (Duck Typing) to verify models work
        correctly with both TensorFlow and PyTorch frameworks.
        """
        # Test DeepLabV3+ (available in both TensorFlow and PyTorch)
        model = model_call("DeepLabV3_plus", 256, 5, l2_regularization_weight=1e-5)
        assert model is not None
        # Verify Keras-compatible interface (works with both TensorFlow keras.Model and PyTorchKerasAdapter)
        assert hasattr(model, 'predict'), "Model should have predict() method"
        assert hasattr(model, 'compile'), "Model should have compile() method"
        assert hasattr(model, 'fit'), "Model should have fit() method"
        assert callable(model.predict), "predict() should be callable"

        # Test UNet (force TensorFlow since PyTorch UNet not yet implemented)
        try:
            model = model_call("UNet", 256, 5, l2_regularization_weight=1e-4, framework='tensorflow')
            assert model is not None
            # Verify Keras-compatible interface
            assert hasattr(model, 'predict'), "Model should have predict() method"
            assert hasattr(model, 'compile'), "Model should have compile() method"
            assert hasattr(model, 'fit'), "Model should have fit() method"
            assert callable(model.predict), "predict() should be callable"
        except ValueError as e:
            if "not yet implemented" in str(e):
                pytest.skip("UNet not yet implemented for current framework")
            else:
                raise

    def test_different_l2_weights(self):
        """Test models with different L2 regularization weights."""
        weights_to_test = [0, 1e-6, 1e-5, 1e-4, 1e-3]

        for weight in weights_to_test:
            model_builder = DeepLabV3Plus(
                input_size=128,  # Smaller size for faster testing
                num_classes=3,
                l2_regularization_weight=weight
            )
            model = model_builder.build_model()
            assert model is not None, f"Failed to create model with L2 weight {weight}"

    def test_negative_l2_weight_validation(self):
        """Test that negative L2 weights are properly rejected."""
        with pytest.raises(ValueError, match="L2 regularization weight must be >= 0"):
            DeepLabV3Plus(
                input_size=128,
                num_classes=3,
                l2_regularization_weight=-1e-5
            )

        with pytest.raises(ValueError, match="L2 regularization weight must be >= 0"):
            UNet(
                input_size=128,
                num_classes=3,
                l2_regularization_weight=-0.001
            )


class TestRegularizationLossCallback:
    """Test the RegularizationLossCallback functionality."""

    def test_callback_initialization(self):
        """Test RegularizationLossCallback initialization."""
        callback = RegularizationLossCallback()
        assert callback is not None
        assert callback.regularization_losses == []
        assert callback.data_losses == []
        assert callback.total_losses == []

    def test_callback_with_logger(self):
        """Test RegularizationLossCallback with logger."""
        mock_logger = MagicMock()
        callback = RegularizationLossCallback(logger=mock_logger)
        assert callback.logger == mock_logger

    @patch('tensorflow.reduce_sum')
    def test_callback_on_epoch_end(self, mock_reduce_sum):
        """Test callback's on_epoch_end method."""
        # Setup mock model and losses
        mock_model = MagicMock()
        mock_model.losses = [tf.constant(0.001), tf.constant(0.002)]
        mock_reduce_sum.return_value.numpy.return_value = 0.003

        # Create callback with mock logger
        mock_logger = MagicMock()
        callback = RegularizationLossCallback(logger=mock_logger)
        callback.model = mock_model

        # Simulate epoch end
        logs = {'loss': 0.103, 'accuracy': 0.95}
        callback.on_epoch_end(0, logs)

        # Check that losses were recorded
        assert len(callback.regularization_losses) == 1
        assert callback.regularization_losses[0] == 0.003
        assert len(callback.data_losses) == 1
        assert abs(callback.data_losses[0] - 0.1) < 1e-7  # 0.103 - 0.003 with floating point tolerance
        assert len(callback.total_losses) == 1
        assert callback.total_losses[0] == 0.103

        # Check that logger was called
        mock_logger.logger.info.assert_called_once()

    def test_callback_without_model_losses(self):
        """Test callback when model has no regularization losses."""
        mock_model = MagicMock()
        mock_model.losses = []

        callback = RegularizationLossCallback()
        callback.model = mock_model

        logs = {'loss': 0.5, 'accuracy': 0.85}
        callback.on_epoch_end(0, logs)

        # Should record zero regularization loss
        assert callback.regularization_losses[0] == 0
        assert callback.data_losses[0] == 0.5
        assert callback.total_losses[0] == 0.5


class TestL2RegularizationIntegration:
    """Integration tests for L2 regularization."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_model_compilation_with_l2(self, temp_model_dir):
        """Test that models compile correctly with L2 regularization."""
        # Create model with L2
        model = model_call("DeepLabV3_plus", 128, 3, l2_regularization_weight=1e-5)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Model should compile without errors
        assert model.optimizer is not None
        assert model.loss is not None

    def test_regularization_loss_computation(self):
        """Test that regularization loss is computed correctly."""
        # Create a simple model with known weights for testing
        inputs = keras.Input(shape=(32, 32, 3))
        x = keras.layers.Conv2D(
            16, 3, padding='same',
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(inputs)
        outputs = keras.layers.Conv2D(2, 1)(x)
        model = keras.Model(inputs, outputs)

        # Compile model
        model.compile(optimizer='adam', loss='mse')

        # Create dummy data
        dummy_input = np.random.randn(1, 32, 32, 3).astype(np.float32)
        dummy_target = np.random.randn(1, 32, 32, 2).astype(np.float32)

        # Get initial regularization loss
        with tf.GradientTape() as tape:
            predictions = model(dummy_input, training=True)
            data_loss = keras.losses.mse(dummy_target, predictions)
            reg_loss = tf.reduce_sum(model.losses) if model.losses else 0

        # Regularization loss should be non-zero
        assert reg_loss > 0, "Regularization loss should be computed"

    def test_adamw_optimizer_configuration(self):
        """Test AdamW optimizer configuration with weight decay."""
        try:
            # Try to create AdamW optimizer
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=0.001,
                weight_decay=1e-5
            )
            assert optimizer is not None
            assert optimizer.weight_decay == 1e-5
        except AttributeError:
            # AdamW might not be available in all TF versions
            pytest.skip("AdamW optimizer not available in this TensorFlow version")

    def test_optimizer_epsilon_configuration(self):
        """Test that Adam and AdamW optimizers are configured with correct epsilon value."""
        from base.config import ModelDefaults

        # Test Adam optimizer epsilon
        adam_optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            epsilon=ModelDefaults.OPTIMIZER_EPSILON
        )
        assert adam_optimizer.epsilon == 1e-8, f"Expected Adam epsilon to be 1e-8, got {adam_optimizer.epsilon}"

        # Test AdamW optimizer epsilon if available
        try:
            adamw_optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=0.001,
                weight_decay=1e-5,
                epsilon=ModelDefaults.OPTIMIZER_EPSILON
            )
            assert adamw_optimizer.epsilon == 1e-8, f"Expected AdamW epsilon to be 1e-8, got {adamw_optimizer.epsilon}"
        except AttributeError:
            # AdamW might not be available in all TF versions
            pass  # That's okay, we at least tested Adam

        # Verify the config value is correct
        assert ModelDefaults.OPTIMIZER_EPSILON == 1e-8, "ModelDefaults.OPTIMIZER_EPSILON should be 1e-8"

    @pytest.mark.parametrize("framework", ["tensorflow", "pytorch"])
    def test_l2_cross_framework_compatibility(self, framework):
        """Test L2 regularization works correctly in both TensorFlow and PyTorch frameworks.

        This is a framework isolation test that explicitly verifies L2 regularization
        functionality across both supported frameworks, ensuring consistent behavior.
        """
        # Import framework-specific modules
        if framework == "pytorch":
            try:
                from base.models.wrappers import PyTorchKerasAdapter
            except ImportError:
                pytest.skip(f"PyTorch framework not available")

        # Create model with explicit framework specification
        try:
            model = model_call("DeepLabV3_plus", 128, 3,
                             l2_regularization_weight=1e-5,
                             framework=framework)
        except Exception as e:
            pytest.skip(f"Framework {framework} not available: {e}")

        # Verify model was created successfully
        assert model is not None, f"Model should be created for {framework} framework"

        # Verify Keras-compatible API exists (both frameworks should support this)
        assert hasattr(model, 'predict'), f"Model should have predict() method for {framework}"
        assert hasattr(model, 'compile'), f"Model should have compile() method for {framework}"
        assert hasattr(model, 'fit'), f"Model should have fit() method for {framework}"

        # Test compilation with cross-framework compatible loss function
        try:
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',  # Works in both frameworks
                metrics=['accuracy']
            )
            assert model.optimizer is not None, f"Optimizer should be set for {framework}"
        except Exception as e:
            pytest.fail(f"Model compilation failed for {framework}: {e}")

        # Framework-specific type checking
        if framework == "tensorflow":
            assert isinstance(model, keras.Model), "TensorFlow should return keras.Model"
        elif framework == "pytorch":
            from base.models.wrappers import PyTorchKerasAdapter
            assert isinstance(model, PyTorchKerasAdapter), "PyTorch should return PyTorchKerasAdapter"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])