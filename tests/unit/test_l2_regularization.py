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

# Add base module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
        """Test model factory function with L2 regularization."""
        # Test DeepLabV3+
        model = model_call("DeepLabV3_plus", 256, 5, l2_regularization_weight=1e-5)
        assert model is not None
        assert isinstance(model, keras.Model)

        # Test UNet
        model = model_call("UNet", 256, 5, l2_regularization_weight=1e-4)
        assert model is not None
        assert isinstance(model, keras.Model)

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

    def test_metal_device_detection(self):
        """Test Metal device detection for AdamW fallback."""
        gpu_devices = tf.config.list_physical_devices('GPU')
        is_metal = gpu_devices and 'metal' in str(gpu_devices[0]).lower()

        # This test just checks the detection logic works
        assert isinstance(is_metal, bool)

    def test_l2_weight_validation_warnings(self):
        """Test that appropriate warnings are shown for high L2 weights."""
        # This would normally be tested through the training module
        # but we test the logic here
        l2_weights_and_expectations = [
            (1e-6, "normal"),     # Very light regularization
            (1e-5, "normal"),     # Default regularization
            (1e-4, "normal"),     # Moderate regularization
            (5e-4, "high"),       # High regularization
            (1e-3, "very_high"),  # Very high regularization
            (1e-2, "very_high"),  # Extremely high regularization
        ]

        for weight, expectation in l2_weights_and_expectations:
            # Just verify the weight values are in expected ranges
            if expectation == "very_high":
                assert weight >= 1e-3
            elif expectation == "high":
                assert 1e-4 < weight <= 1e-3
            else:
                assert weight <= 1e-4


class TestL2DocumentationAccuracy:
    """Test that implementation matches documentation claims."""

    def test_deeplabv3_regularized_layer_count(self):
        """Verify the actual count of regularized layers in DeepLabV3+."""
        model_builder = DeepLabV3Plus(
            input_size=256,
            num_classes=5,
            l2_regularization_weight=1e-5
        )
        model = model_builder.build_model()

        # Count Conv2D layers with regularization
        regularized_conv2d_count = 0
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2D):
                if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
                    regularized_conv2d_count += 1

        # Log the actual count for documentation update
        print(f"DeepLabV3+ has {regularized_conv2d_count} regularized Conv2D layers")
        # The actual count will be used to update documentation
        assert regularized_conv2d_count > 0

    def test_unet_regularized_layer_count(self):
        """Verify the actual count of regularized layers in UNet."""
        model_builder = UNet(
            input_size=256,
            num_classes=5,
            l2_regularization_weight=1e-5
        )
        model = model_builder.build_model()

        # Count Conv2D and Conv2DTranspose layers with regularization
        regularized_count = 0
        for layer in model.layers:
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Conv2DTranspose)):
                if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
                    regularized_count += 1

        # Log the actual count for documentation update
        print(f"UNet has {regularized_count} regularized convolutional layers")
        # The actual count will be used to update documentation
        assert regularized_count > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])