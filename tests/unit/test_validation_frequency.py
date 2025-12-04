"""
Unit tests for iteration-based validation frequency in BatchAccuracyCallback.

This module tests the new iteration-based validation functionality to ensure
it matches MATLAB's ValidationFrequency behavior.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import warnings
import os
import numpy as np
import tensorflow as tf
from base.models.training import BatchAccuracyCallback


class TestValidationFrequency:
    """Test suite for iteration-based validation frequency."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_val_data = Mock()
        self.mock_loss = Mock()
        self.mock_logger = Mock()

        # Clear any environment variables that might affect tests
        if 'CODAVISION_VALIDATION_FREQUENCY' in os.environ:
            del os.environ['CODAVISION_VALIDATION_FREQUENCY']

        yield

        # Cleanup after test
        if 'CODAVISION_VALIDATION_FREQUENCY' in os.environ:
            del os.environ['CODAVISION_VALIDATION_FREQUENCY']

    def test_validation_frequency_initialization(self):
        """Test that validation_frequency is properly initialized."""
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            logger=self.mock_logger,
            validation_frequency=128
        )

        assert callback.validation_frequency == 128
        assert callback.global_step == 0

    def test_validation_at_correct_iterations(self):
        """Test that validation occurs at correct iteration intervals."""
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            logger=self.mock_logger,
            validation_frequency=128
        )

        # Mock the run_validation method to track calls
        callback.run_validation = Mock()

        # Set params that would normally be set by Keras
        callback.params = {'steps': 50}

        # Simulate training for multiple epochs
        for epoch in range(3):
            callback.on_epoch_begin(epoch)
            for batch in range(50):  # 50 batches per epoch
                callback.on_batch_end(batch)

        # Total iterations: 3 epochs × 50 batches = 150
        # Validations should occur at iterations 128 only
        assert callback.run_validation.call_count == 1
        assert callback.val_indices == [128]

    def test_multiple_validations(self):
        """Test multiple validations over longer training."""
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            validation_frequency=100  # Lower frequency for testing
        )

        callback.run_validation = Mock()
        callback.params = {'steps': 100}

        # Simulate 300 iterations
        for epoch in range(3):
            callback.on_epoch_begin(epoch)
            for batch in range(100):
                callback.on_batch_end(batch)

        # Should validate at iterations 100, 200, 300
        assert callback.run_validation.call_count == 3
        assert callback.val_indices == [100, 200, 300]

    def test_no_validation_edge_case(self):
        """Test edge case where training ends before first validation."""
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            logger=self.mock_logger,
            validation_frequency=128
        )

        callback.run_validation = Mock()
        callback.params = {'steps': 100}

        # Simulate only 100 iterations (less than 128)
        callback.on_epoch_begin(0)
        for batch in range(100):
            callback.on_batch_end(batch)

        # No validation should occur during training
        assert callback.run_validation.call_count == 0

        # But validation should occur at train_end
        callback.on_train_end()
        assert callback.run_validation.call_count == 1

    def test_backward_compatibility_warning(self):
        """Test deprecated num_validations parameter handling."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            callback = BatchAccuracyCallback(
                model=self.mock_model,
                val_data=self.mock_val_data,
                loss_function=self.mock_loss,
                num_validations=3  # Deprecated parameter
            )

            # Check that a deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "num_validations" in str(w[0].message)

            # Should be converted to approximately validation_frequency=50
            # (150 estimated steps per epoch / 3)
            assert callback.validation_frequency == 50

    def test_environment_variable_override(self):
        """Test that environment variable can override validation frequency."""
        os.environ['CODAVISION_VALIDATION_FREQUENCY'] = '256'

        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            logger=self.mock_logger,
            validation_frequency=128  # Should be overridden
        )

        assert callback.validation_frequency == 256

        # Clean up
        del os.environ['CODAVISION_VALIDATION_FREQUENCY']

    def test_global_step_tracking(self):
        """Test that global_step tracks iterations across epochs."""
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            validation_frequency=200
        )

        callback.run_validation = Mock()
        callback.params = {'steps': 50}

        # Track global_step values
        global_steps = []

        # Override on_batch_end to record global_step
        original_on_batch_end = callback.on_batch_end
        def track_batch_end(batch, logs=None):
            original_on_batch_end(batch, logs)
            global_steps.append(callback.global_step)

        callback.on_batch_end = track_batch_end

        # Simulate 2 epochs of 50 batches each
        for epoch in range(2):
            callback.on_epoch_begin(epoch)
            for batch in range(50):
                callback.on_batch_end(batch)

        # Verify global_step increments continuously across epochs
        assert len(global_steps) == 100
        assert global_steps[-1] == 100

        # Verify sequential increment
        for i in range(1, len(global_steps)):
            assert global_steps[i] == global_steps[i-1] + 1

    def test_batch_numbers_use_global_step(self):
        """Test that batch_numbers uses global_step instead of epoch-relative numbers."""
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            validation_frequency=1000  # High value to avoid validation
        )

        callback.params = {'steps': 10}

        # Simulate 2 epochs with accuracy logs
        for epoch in range(2):
            callback.on_epoch_begin(epoch)
            for batch in range(10):
                logs = {'accuracy': 0.5 + batch * 0.01}
                callback.on_batch_end(batch, logs)

        # Verify batch_numbers are global, not reset per epoch
        assert len(callback.batch_numbers) == 20
        # Should be 1, 2, 3, ..., 20
        expected_batch_numbers = list(range(1, 21))
        assert callback.batch_numbers == expected_batch_numbers

    def test_validation_counter_increments(self):
        """Test that validation_counter increments with each validation."""
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            validation_frequency=50
        )

        callback.run_validation = Mock()
        callback.params = {'steps': 60}

        # Simulate 2 epochs
        for epoch in range(2):
            callback.on_epoch_begin(epoch)
            for batch in range(60):
                callback.on_batch_end(batch)

        # Total iterations: 120
        # Validations at: 50, 100
        assert callback.validation_counter == 2

    def test_on_train_end_with_validation_data(self):
        """Test on_train_end only runs validation if no validations occurred."""
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            logger=self.mock_logger,
            validation_frequency=50
        )

        callback.run_validation = Mock()
        callback.params = {'steps': 30}

        # Simulate short training (30 iterations, less than 50)
        callback.on_epoch_begin(0)
        for batch in range(30):
            callback.on_batch_end(batch)

        assert callback.validation_counter == 0

        # on_train_end should trigger validation
        callback.on_train_end()
        assert callback.run_validation.call_count == 1

    def test_on_train_end_no_validation_if_already_validated(self):
        """Test on_train_end doesn't run validation if validations already occurred."""
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            validation_frequency=50
        )

        callback.run_validation = Mock()
        callback.params = {'steps': 60}

        # Simulate training with 60 iterations
        callback.on_epoch_begin(0)
        for batch in range(60):
            callback.on_batch_end(batch)

        # Should have validated at iteration 50
        assert callback.run_validation.call_count == 1
        assert callback.validation_counter == 1

        # on_train_end should NOT trigger additional validation
        callback.on_train_end()
        assert callback.run_validation.call_count == 1  # Still 1

    def test_invalid_environment_variable(self):
        """Test handling of invalid environment variable value."""
        os.environ['CODAVISION_VALIDATION_FREQUENCY'] = 'invalid'

        # Should use default value and log warning
        callback = BatchAccuracyCallback(
            model=self.mock_model,
            val_data=self.mock_val_data,
            loss_function=self.mock_loss,
            logger=self.mock_logger,
            validation_frequency=128
        )

        # Should fall back to provided value
        assert callback.validation_frequency == 128

        # Check that warning was logged
        self.mock_logger.warning.assert_called_once()

        # Clean up
        del os.environ['CODAVISION_VALIDATION_FREQUENCY']