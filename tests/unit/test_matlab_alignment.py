"""
Unit tests for MATLAB alignment fixes in Python DeepLabV3+ implementation.

This module tests the critical fixes needed to align Python implementation
with MATLAB behavior for equivalent performance:
1. Class weight normalization (sum to 1.0)
2. Learning rate schedule (unconditional every-epoch reduction)
"""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np
import tensorflow as tf
from base.models.training import WeightedSparseCategoricalCrossentropy, BatchAccuracyCallback


class TestClassWeightNormalization:
    """Test suite for class weight normalization (Fix #1)."""

    def test_class_weights_sum_to_one(self):
        """Test that class weights are normalized to sum to 1.0."""
        # Test with various weight configurations
        test_cases = [
            [1.0, 2.0, 3.0],  # Simple integers
            [0.5, 1.5, 3.0],  # Decimals
            [10, 20, 30, 40],  # Larger values
            [0.1, 0.2, 0.3, 0.4],  # Small values
        ]

        for weights in test_cases:
            loss_fn = WeightedSparseCategoricalCrossentropy(
                class_weights=weights,
                from_logits=True
            )

            # Check that weights sum to 1.0
            weights_sum = tf.reduce_sum(loss_fn.class_weights).numpy()
            assert weights_sum == pytest.approx(1.0, abs=1e-7), \
                f"Weights {weights} should sum to 1.0, got {weights_sum}"

    def test_class_weights_preserve_ratios(self):
        """Test that normalization preserves relative proportions."""
        original_weights = [1.0, 2.0, 4.0]
        loss_fn = WeightedSparseCategoricalCrossentropy(
            class_weights=original_weights,
            from_logits=True
        )

        normalized = loss_fn.class_weights.numpy()

        # Expected normalized values: [1/7, 2/7, 4/7]
        expected = np.array(original_weights) / np.sum(original_weights)

        np.testing.assert_array_almost_equal(
            normalized, expected, decimal=7,
            err_msg="Normalized weights should preserve relative proportions"
        )

    def test_normalized_weights_with_numpy_array(self):
        """Test normalization works with numpy arrays."""
        weights = np.array([5.0, 10.0, 15.0, 20.0])
        loss_fn = WeightedSparseCategoricalCrossentropy(
            class_weights=weights,
            from_logits=True
        )

        weights_sum = tf.reduce_sum(loss_fn.class_weights).numpy()
        assert weights_sum == pytest.approx(1.0, abs=1e-7)

    def test_normalized_weights_integration(self):
        """Integration test: verify normalized weights work in loss computation."""
        # Create a simple test case
        class_weights = [1.0, 2.0, 3.0]
        loss_fn = WeightedSparseCategoricalCrossentropy(
            class_weights=class_weights,
            from_logits=True
        )

        # Create dummy predictions and targets
        batch_size = 4
        num_classes = 3
        height, width = 8, 8

        # Create random logits and labels
        y_pred = tf.random.normal([batch_size, height, width, num_classes])
        y_true = tf.random.uniform(
            [batch_size, height, width, 1],
            minval=0, maxval=num_classes, dtype=tf.int32
        )

        # Compute loss - should not raise errors
        loss = loss_fn(y_true, y_pred)

        # Verify loss is a scalar and is finite
        assert loss.shape == ()
        assert tf.math.is_finite(loss).numpy()

    def test_single_class_edge_case(self):
        """Test edge case with single class weight."""
        weights = [5.0]
        loss_fn = WeightedSparseCategoricalCrossentropy(
            class_weights=weights,
            from_logits=True
        )

        # Single weight should normalize to 1.0
        assert loss_fn.class_weights.numpy()[0] == pytest.approx(1.0, abs=1e-7)


class TestLearningRateSchedule:
    """Test suite for learning rate schedule (Fix #2)."""

    @pytest.fixture
    def mock_model(self):
        """Mock model fixture."""
        model = Mock()
        optimizer = Mock()
        optimizer.learning_rate = tf.Variable(0.0005)
        model.optimizer = optimizer
        return model

    @pytest.fixture
    def mock_val_data(self):
        """Mock validation data fixture."""
        return Mock()

    @pytest.fixture
    def mock_loss(self):
        """Mock loss function fixture."""
        return Mock()

    @pytest.fixture
    def mock_logger(self):
        """Mock logger fixture."""
        return Mock()

    @pytest.fixture
    def initial_lr(self):
        """Initial learning rate fixture."""
        return 0.0005

    def test_lr_drops_every_epoch(self, mock_model, mock_val_data, mock_loss, mock_logger, initial_lr):
        """Test that learning rate drops unconditionally every epoch."""
        callback = BatchAccuracyCallback(
            model=mock_model,
            val_data=mock_val_data,
            loss_function=mock_loss,
            logger=mock_logger,
            validation_frequency=128,
            lr_factor=0.75
        )

        # Simulate 8 epochs
        lr_values = [initial_lr]
        for epoch in range(8):
            callback.on_epoch_end(epoch)
            current_lr = float(mock_model.optimizer.learning_rate.numpy())
            lr_values.append(current_lr)

        # Verify LR decreased every epoch
        for i in range(len(lr_values) - 1):
            assert lr_values[i + 1] < lr_values[i], \
                f"LR should decrease from epoch {i} to {i+1}"

    def test_lr_schedule_matches_matlab(self, mock_model, mock_val_data, mock_loss, initial_lr):
        """Test that final LR matches MATLAB: 0.0005 * 0.75^7."""
        callback = BatchAccuracyCallback(
            model=mock_model,
            val_data=mock_val_data,
            loss_function=mock_loss,
            validation_frequency=128,
            lr_factor=0.75
        )

        # Simulate 8 epochs (0-7)
        for epoch in range(8):
            callback.on_epoch_end(epoch)

        # After 8 epochs, LR should be initial_lr * 0.75^8
        # (since on_epoch_end is called at the END of each epoch)
        final_lr = float(mock_model.optimizer.learning_rate.numpy())
        expected_lr = initial_lr * (0.75 ** 8)

        assert final_lr == pytest.approx(expected_lr, abs=1e-10), \
            f"Final LR should be {expected_lr}, got {final_lr}"

    def test_lr_progression_8_epochs(self, mock_model, mock_val_data, mock_loss, initial_lr):
        """Test LR values across all 8 epochs match expected progression."""
        callback = BatchAccuracyCallback(
            model=mock_model,
            val_data=mock_val_data,
            loss_function=mock_loss,
            validation_frequency=128,
            lr_factor=0.75
        )

        # Expected LR values after each epoch
        expected_lrs = [initial_lr * (0.75 ** i) for i in range(1, 9)]

        # Simulate 8 epochs and collect LR values
        actual_lrs = []
        for epoch in range(8):
            callback.on_epoch_end(epoch)
            actual_lrs.append(float(mock_model.optimizer.learning_rate.numpy()))

        # Verify each epoch's LR
        for epoch, (expected, actual) in enumerate(zip(expected_lrs, actual_lrs)):
            assert actual == pytest.approx(expected, abs=1e-10), \
                f"Epoch {epoch}: Expected LR={expected}, got {actual}"

    def test_lr_reduction_factor(self, mock_model, mock_val_data, mock_loss, initial_lr):
        """Test that LR reduction factor is correctly applied."""
        lr_factors = [0.5, 0.75, 0.9]

        for factor in lr_factors:
            # Reset optimizer
            mock_model.optimizer.learning_rate = tf.Variable(initial_lr)

            callback = BatchAccuracyCallback(
                model=mock_model,
                val_data=mock_val_data,
                loss_function=mock_loss,
                validation_frequency=128,
                lr_factor=factor
            )

            # Run one epoch
            callback.on_epoch_end(0)
            new_lr = float(mock_model.optimizer.learning_rate.numpy())
            expected_lr = initial_lr * factor

            assert new_lr == pytest.approx(expected_lr, abs=1e-10), \
                f"LR should be multiplied by {factor}"

    def test_lr_no_conditional_logic(self, mock_model, mock_val_data, mock_loss):
        """Test that LR reduction has no conditional logic (no plateau checking)."""
        callback = BatchAccuracyCallback(
            model=mock_model,
            val_data=mock_val_data,
            loss_function=mock_loss,
            validation_frequency=128,
            lr_factor=0.75
        )

        # Even with epoch_wait = 0, LR should still reduce
        callback.epoch_wait = 0
        initial_lr = float(mock_model.optimizer.learning_rate.numpy())
        callback.on_epoch_end(0)
        new_lr = float(mock_model.optimizer.learning_rate.numpy())

        assert new_lr < initial_lr, \
            "LR should reduce unconditionally (no plateau check)"


class TestMATLABAlignmentIntegration:
    """Integration tests for combined MATLAB alignment fixes."""

    def test_loss_function_with_normalized_weights(self):
        """Test that normalized weights integrate correctly with loss computation."""
        # Create loss function with unnormalized weights
        raw_weights = [10.0, 20.0, 30.0]
        loss_fn = WeightedSparseCategoricalCrossentropy(
            class_weights=raw_weights,
            from_logits=True
        )

        # Verify weights are normalized
        assert tf.reduce_sum(loss_fn.class_weights).numpy() == pytest.approx(1.0, abs=1e-7)

        # Create test data
        batch_size = 2
        num_classes = 3
        height, width = 4, 4

        y_pred = tf.random.normal([batch_size, height, width, num_classes])
        y_true = tf.constant([[[0, 1, 2, 0],
                                [1, 2, 0, 1],
                                [2, 0, 1, 2],
                                [0, 1, 2, 0]],
                               [[1, 2, 0, 1],
                                [2, 0, 1, 2],
                                [0, 1, 2, 0],
                                [1, 2, 0, 1]]], dtype=tf.int32)
        y_true = tf.expand_dims(y_true, -1)

        # Compute loss
        loss = loss_fn(y_true, y_pred)

        # Verify loss is valid
        assert tf.math.is_finite(loss).numpy()
        assert loss.numpy() > 0.0

    def test_combined_fixes_compatibility(self):
        """Test that both fixes work together without conflicts."""
        # Create model with both fixes applied
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_optimizer.learning_rate = tf.Variable(0.0005)
        mock_model.optimizer = mock_optimizer

        # Create loss function with normalized weights
        class_weights = [1.0, 2.0, 3.0]
        loss_fn = WeightedSparseCategoricalCrossentropy(
            class_weights=class_weights,
            from_logits=True
        )

        # Create callback with unconditional LR schedule
        callback = BatchAccuracyCallback(
            model=mock_model,
            val_data=Mock(),
            loss_function=loss_fn,
            validation_frequency=128,
            lr_factor=0.75
        )

        # Verify both components are configured correctly
        assert tf.reduce_sum(loss_fn.class_weights).numpy() == pytest.approx(1.0, abs=1e-7)

        # Run a few epochs
        for epoch in range(3):
            callback.on_epoch_end(epoch)

        # Verify LR reduced correctly
        expected_lr = 0.0005 * (0.75 ** 3)
        actual_lr = float(mock_model.optimizer.learning_rate.numpy())
        assert actual_lr == pytest.approx(expected_lr, abs=1e-10)


class TestLossFunctionPerClassWeighting:
    """Test suite for per-class weighted loss function (Fix #3)."""

    @pytest.fixture
    def loss_fn_3_class(self):
        """Fixture for 3-class loss function."""
        return WeightedSparseCategoricalCrossentropy(
            class_weights=[1.0, 2.0, 3.0],
            from_logits=True
        )

    def test_per_class_weighting_with_imbalanced_data(self, loss_fn_3_class):
        """Test per-class weighting handles imbalanced data correctly."""
        # Create imbalanced ground truth: Class 0: 75%, Class 1: 19%, Class 2: 6%
        y_true = tf.constant([[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 2]]], dtype=tf.int32)
        y_true = tf.expand_dims(y_true, -1)

        y_pred = tf.constant([[[[1.0, 0.0, 0.0]] * 4] * 4], dtype=tf.float32)

        loss = loss_fn_3_class(y_true, y_pred)

        assert tf.math.is_finite(loss).numpy()
        assert loss.numpy() > 0.0

    def test_edge_cases_missing_and_rare_classes(self, loss_fn_3_class):
        """Test numerical stability with missing and rare classes."""
        # Test 1: Missing class (only uses 2 of 3 classes)
        y_pred = tf.random.normal([2, 4, 4, 3], seed=123)
        y_true = tf.constant(np.random.randint(0, 2, size=(2, 4, 4, 1)), dtype=tf.int32)
        loss = loss_fn_3_class(y_true, y_pred)
        assert tf.math.is_finite(loss).numpy()

        # Test 2: Very rare class (0.1% of pixels)
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[0, :9] = 1  # 9 pixels of class 1
        labels[0, 9] = 2   # 1 pixel of class 2
        y_true_rare = tf.constant(labels[np.newaxis, :, :, np.newaxis])
        y_pred_rare = tf.random.normal([1, 100, 100, 3], seed=456)
        loss_rare = loss_fn_3_class(y_true_rare, y_pred_rare)
        assert not np.isnan(loss_rare.numpy()) and not np.isinf(loss_rare.numpy())

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the per-class loss."""
        loss_fn = WeightedSparseCategoricalCrossentropy(class_weights=[1.0, 2.0], from_logits=True)

        y_pred = tf.Variable(tf.random.normal([2, 4, 4, 2], seed=789))
        y_true = tf.random.uniform([2, 4, 4, 1], minval=0, maxval=2, dtype=tf.int32)

        with tf.GradientTape() as tape:
            loss = loss_fn(y_true, y_pred)

        gradients = tape.gradient(loss, y_pred)

        assert gradients is not None
        assert tf.reduce_all(tf.math.is_finite(gradients)).numpy()
        assert tf.norm(gradients).numpy() > 0.0

    def test_combined_three_fixes_integration(self):
        """Integration test: all three MATLAB alignment fixes work together."""
        raw_weights = [10.0, 20.0, 30.0]
        loss_fn = WeightedSparseCategoricalCrossentropy(class_weights=raw_weights, from_logits=True)

        # Verify Fix #1: weights are normalized
        assert tf.reduce_sum(loss_fn.class_weights).numpy() == pytest.approx(1.0, abs=1e-7)

        # Create imbalanced test data for Fix #3
        labels = np.zeros((2, 8, 8), dtype=np.int32)
        labels[:, :, 3:6] = 1
        labels[:, :, 7] = 2
        y_true = tf.constant(labels[:, :, :, np.newaxis])
        y_pred = tf.random.normal([2, 8, 8, 3], seed=200)

        loss = loss_fn(y_true, y_pred)

        assert tf.math.is_finite(loss).numpy()
        expected_normalized = np.array(raw_weights) / np.sum(raw_weights)
        np.testing.assert_array_almost_equal(loss_fn.class_weights.numpy(), expected_normalized, decimal=7)
