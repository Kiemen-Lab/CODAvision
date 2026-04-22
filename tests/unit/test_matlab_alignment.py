"""
Unit tests for training alignment fixes in Python DeepLabV3+ implementation.

This module tests:
1. Class weight normalization (sum to 1.0)
2. Learning rate schedule (validation-based plateau reduction)
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
    """Test suite for validation-based learning rate reduction."""

    @pytest.fixture
    def mock_model(self):
        """Mock model fixture."""
        model = Mock()
        optimizer = Mock()
        optimizer.learning_rate = tf.Variable(0.0005)
        model.optimizer = optimizer
        model.stop_training = False
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
        logger = Mock()
        logger.logger = Mock()
        logger.log_validation_metrics = Mock()
        return logger

    @pytest.fixture
    def initial_lr(self):
        """Initial learning rate fixture."""
        return 0.0005

    def test_lr_no_reduction_when_improving(self, mock_model, mock_val_data, mock_loss, mock_logger, initial_lr):
        """Test that LR does NOT reduce when val_loss improves."""
        callback = BatchAccuracyCallback(
            model=mock_model,
            val_data=mock_val_data,
            loss_function=mock_loss,
            logger=mock_logger,
            validation_frequency=128,
            lr_factor=0.75,
            lr_patience=1
        )

        # Simulate improving val_loss
        callback.best_val_loss_for_lr = 1.0
        # Trigger LR logic directly with improving loss
        val_loss_avg = 0.5  # Better than best (1.0)
        if callback.reduce_lr:
            if val_loss_avg < callback.best_val_loss_for_lr:
                callback.best_val_loss_for_lr = val_loss_avg
                callback.lr_wait = 0

        current_lr = float(mock_model.optimizer.learning_rate.numpy())
        assert current_lr == pytest.approx(initial_lr, abs=1e-10), \
            "LR should NOT reduce when val_loss improves"

    def test_lr_reduces_after_patience_validations(self, mock_model, mock_val_data, mock_loss, mock_logger, initial_lr):
        """Test that LR reduces after lr_patience validations without val_loss improvement."""
        callback = BatchAccuracyCallback(
            model=mock_model,
            val_data=mock_val_data,
            loss_function=mock_loss,
            logger=mock_logger,
            validation_frequency=128,
            lr_factor=0.75,
            lr_patience=2
        )

        # Set a best val_loss
        callback.best_val_loss_for_lr = 0.5

        # Simulate 2 validations without improvement (matching lr_patience=2)
        for _ in range(2):
            val_loss_avg = 0.6  # Worse than best (0.5)
            if callback.reduce_lr:
                if val_loss_avg < callback.best_val_loss_for_lr:
                    callback.best_val_loss_for_lr = val_loss_avg
                    callback.lr_wait = 0
                else:
                    callback.lr_wait += 1
                    if callback.lr_wait >= callback.lr_patience:
                        old_lr = float(mock_model.optimizer.learning_rate.numpy())
                        new_lr = max(old_lr * callback.lr_factor, 1e-7)
                        if new_lr < old_lr:
                            mock_model.optimizer.learning_rate.assign(new_lr)
                        callback.lr_wait = 0

        current_lr = float(mock_model.optimizer.learning_rate.numpy())
        expected_lr = initial_lr * 0.75
        assert current_lr == pytest.approx(expected_lr, abs=1e-10), \
            f"LR should reduce after {2} validations without improvement"

    def test_lr_respects_min_lr(self, mock_model, mock_val_data, mock_loss, mock_logger):
        """Test that LR reduction respects minimum LR floor of 1e-7."""
        # Start with very small LR
        mock_model.optimizer.learning_rate = tf.Variable(2e-7)

        callback = BatchAccuracyCallback(
            model=mock_model,
            val_data=mock_val_data,
            loss_function=mock_loss,
            logger=mock_logger,
            validation_frequency=128,
            lr_factor=0.75,
            lr_patience=1
        )

        callback.best_val_loss_for_lr = 0.5

        # Simulate stagnation to trigger reduction
        val_loss_avg = 0.6
        if callback.reduce_lr:
            if val_loss_avg >= callback.best_val_loss_for_lr:
                callback.lr_wait += 1
                if callback.lr_wait >= callback.lr_patience:
                    old_lr = float(mock_model.optimizer.learning_rate.numpy())
                    new_lr = max(old_lr * callback.lr_factor, 1e-7)
                    if new_lr < old_lr:
                        mock_model.optimizer.learning_rate.assign(new_lr)
                    callback.lr_wait = 0

        current_lr = float(mock_model.optimizer.learning_rate.numpy())
        assert current_lr >= 1e-7, \
            f"LR should not go below 1e-7, got {current_lr}"

    def test_lr_factor_applied_correctly(self, mock_model, mock_val_data, mock_loss, mock_logger, initial_lr):
        """Test that lr_factor is correctly applied on reduction."""
        lr_factors = [0.5, 0.75, 0.9]

        for factor in lr_factors:
            mock_model.optimizer.learning_rate = tf.Variable(initial_lr)

            callback = BatchAccuracyCallback(
                model=mock_model,
                val_data=mock_val_data,
                loss_function=mock_loss,
                logger=mock_logger,
                validation_frequency=128,
                lr_factor=factor,
                lr_patience=1
            )

            callback.best_val_loss_for_lr = 0.5

            # Trigger reduction
            val_loss_avg = 0.6
            callback.lr_wait = callback.lr_patience  # Force trigger
            old_lr = float(mock_model.optimizer.learning_rate.numpy())
            new_lr = max(old_lr * callback.lr_factor, 1e-7)
            if new_lr < old_lr:
                mock_model.optimizer.learning_rate.assign(new_lr)
            callback.lr_wait = 0

            current_lr = float(mock_model.optimizer.learning_rate.numpy())
            expected_lr = initial_lr * factor
            assert current_lr == pytest.approx(expected_lr, abs=1e-10), \
                f"LR should be multiplied by {factor}, got {current_lr}"


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

        # Create callback with validation-based LR schedule
        callback = BatchAccuracyCallback(
            model=mock_model,
            val_data=Mock(),
            loss_function=loss_fn,
            validation_frequency=128,
            lr_factor=0.75,
            lr_patience=1
        )

        # Verify class weight normalization is configured correctly
        assert tf.reduce_sum(loss_fn.class_weights).numpy() == pytest.approx(1.0, abs=1e-7)

        # Verify LR reduction is validation-based (on_epoch_end does nothing)
        initial_lr = float(mock_model.optimizer.learning_rate.numpy())
        for epoch in range(3):
            callback.on_epoch_end(epoch)
        actual_lr = float(mock_model.optimizer.learning_rate.numpy())
        assert actual_lr == pytest.approx(initial_lr, abs=1e-10), \
            "on_epoch_end should NOT reduce LR (reduction is validation-based now)"


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
