"""
Unit tests for PyTorch training system.

Tests the PyTorch training implementation including loss functions, training loops,
checkpointing, and scheduler functionality.

Author: CODAvision Team
Date: 2025-11-12
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import pickle

# Skip entire module if PyTorch not available
pytest_pytorch = pytest.importorskip("torch")
torch = pytest_pytorch


class TestWeightedCrossEntropyLoss:
    """Test suite for weighted cross-entropy loss function."""

    def test_loss_function_numerical_equivalence(self):
        """Test that PyTorch loss matches TensorFlow loss numerically."""
        try:
            import tensorflow as tf
            from base.models.training import WeightedSparseCategoricalCrossentropy
        except ImportError:
            pytest.skip("TensorFlow not available for comparison")

        from base.models.training_pytorch import WeightedCrossEntropyLoss

        # Test with balanced classes
        class_weights = [1.0, 1.0, 1.0]
        batch_size, height, width = 2, 32, 32
        num_classes = len(class_weights)

        # Generate random predictions (logits) and labels
        np.random.seed(42)
        predictions_np = np.random.randn(batch_size, height, width, num_classes).astype(np.float32)
        labels_np = np.random.randint(0, num_classes, size=(batch_size, height, width)).astype(np.int32)

        # TensorFlow loss
        tf_loss_fn = WeightedSparseCategoricalCrossentropy(class_weights, from_logits=True)
        tf_predictions = tf.constant(predictions_np)
        tf_labels = tf.constant(labels_np[..., np.newaxis])
        tf_loss = tf_loss_fn(tf_labels, tf_predictions).numpy()

        # PyTorch loss
        pytorch_loss_fn = WeightedCrossEntropyLoss(class_weights)
        torch_predictions = torch.from_numpy(predictions_np.transpose(0, 3, 1, 2))
        torch_labels = torch.from_numpy(labels_np)
        pytorch_loss = pytorch_loss_fn(torch_predictions, torch_labels).item()

        # Compare losses
        tolerance = 1e-5
        difference = abs(tf_loss - pytorch_loss)

        assert difference < tolerance, \
            f"Loss difference {difference:.2e} exceeds tolerance {tolerance:.2e} " \
            f"(TF: {tf_loss:.8f}, PT: {pytorch_loss:.8f})"

    def test_loss_with_imbalanced_classes(self):
        """Test loss with imbalanced class weights."""
        from base.models.training_pytorch import WeightedCrossEntropyLoss

        class_weights = [1.0, 2.0, 5.0]
        loss_fn = WeightedCrossEntropyLoss(class_weights)

        # Create predictions and labels
        predictions = torch.randn(2, 3, 32, 32)  # NCHW
        labels = torch.randint(0, 3, (2, 32, 32))

        loss = loss_fn(predictions, labels)

        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_loss_with_missing_class(self):
        """Test loss when some classes are not present in the batch."""
        try:
            import tensorflow as tf
            from base.models.training import WeightedSparseCategoricalCrossentropy
        except ImportError:
            pytest.skip("TensorFlow not available for comparison")

        from base.models.training_pytorch import WeightedCrossEntropyLoss

        class_weights = [1.0, 1.0, 1.0]
        batch_size, height, width = 2, 32, 32
        num_classes = 3

        # Generate predictions and labels (only use classes 0 and 1)
        predictions_np = np.random.randn(batch_size, height, width, num_classes).astype(np.float32)
        labels_np = np.random.randint(0, 2, size=(batch_size, height, width)).astype(np.int32)

        # TensorFlow
        tf_loss_fn = WeightedSparseCategoricalCrossentropy(class_weights, from_logits=True)
        tf_predictions = tf.constant(predictions_np)
        tf_labels = tf.constant(labels_np[..., np.newaxis])
        tf_loss = tf_loss_fn(tf_labels, tf_predictions).numpy()

        # PyTorch
        pytorch_loss_fn = WeightedCrossEntropyLoss(class_weights)
        torch_predictions = torch.from_numpy(predictions_np.transpose(0, 3, 1, 2))
        torch_labels = torch.from_numpy(labels_np)
        pytorch_loss = pytorch_loss_fn(torch_predictions, torch_labels).item()

        # Compare
        tolerance = 1e-5
        difference = abs(tf_loss - pytorch_loss)

        assert difference < tolerance, \
            f"Loss difference with missing class {difference:.2e} exceeds tolerance {tolerance:.2e}"


class TestTrainingLoop:
    """Test suite for training loop execution."""

    def test_training_loop_executes(self, temp_dir, synthetic_training_data):
        """Test that training loop executes without errors."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        trainer = PyTorchDeepLabV3PlusTrainer(model_path)

        # Train for 2 epochs
        history = trainer.train(
            epochs=2,
            batch_size=2,
            learning_rate=0.001,
            validation_split=0.3,
            num_workers=0
        )

        # Check that training completed
        assert len(history['train_loss']) >= 2, \
            f"Expected at least 2 epochs, got {len(history['train_loss'])}"

        # Check that loss decreased significantly (at least 10%)
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        loss_decrease = (initial_loss - final_loss) / initial_loss

        assert loss_decrease >= 0.10, \
            f"Loss should decrease by at least 10%: initial={initial_loss:.4f}, " \
            f"final={final_loss:.4f}, decrease={loss_decrease*100:.1f}%"

        # Check that validation ran
        assert len(history['val_loss']) > 0, "Validation should have run"

    def test_training_with_validation_split(self, temp_dir, synthetic_training_data):
        """Test training with validation split."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        trainer = PyTorchDeepLabV3PlusTrainer(model_path)

        history = trainer.train(
            epochs=1,
            batch_size=2,
            validation_split=0.3,  # 30% validation
            num_workers=0
        )

        assert 'val_loss' in history, "History should contain validation loss"
        assert len(history['val_loss']) > 0, "Validation should have occurred"
        assert 'val_iterations' in history, "History should track validation iterations"


class TestValidationFrequency:
    """Test suite for validation frequency configuration."""

    def test_validation_frequency_configuration(self, temp_dir, synthetic_training_data):
        """Test that validation runs at correct iteration intervals."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        # Set custom validation frequency
        validation_frequency = 5
        prev_freq = os.environ.get('CODAVISION_VALIDATION_FREQUENCY')
        os.environ['CODAVISION_VALIDATION_FREQUENCY'] = str(validation_frequency)

        try:
            trainer = PyTorchDeepLabV3PlusTrainer(model_path)

            history = trainer.train(
                epochs=1,
                batch_size=2,
                learning_rate=0.001,
                validation_split=0.2,
                num_workers=0
            )

            # Check validation iterations
            val_iterations = history['val_iterations']

            assert len(val_iterations) >= 1, "At least one validation should have occurred"

            # Check that validations occurred at multiples of validation_frequency
            for it in val_iterations[:-1]:  # Exclude last iteration (end of epoch)
                assert it % validation_frequency == 0, \
                    f"Validation at iteration {it} not at multiple of {validation_frequency}"

        finally:
            # Restore previous setting
            if prev_freq is None:
                os.environ.pop('CODAVISION_VALIDATION_FREQUENCY', None)
            else:
                os.environ['CODAVISION_VALIDATION_FREQUENCY'] = prev_freq


class TestModelCheckpointing:
    """Test suite for model checkpointing."""

    def test_checkpoint_files_created(self, temp_dir, synthetic_training_data):
        """Test that checkpoint files are saved during training."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        trainer = PyTorchDeepLabV3PlusTrainer(model_path)

        history = trainer.train(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            validation_split=0.3,
            num_workers=0
        )

        # Check that checkpoint files exist
        latest_checkpoint = os.path.join(model_path, 'checkpoint_latest.pth')
        best_checkpoint = os.path.join(model_path, 'checkpoint_best.pth')

        assert os.path.exists(latest_checkpoint), "Latest checkpoint should be saved"
        assert os.path.exists(best_checkpoint), "Best checkpoint should be saved"

    def test_checkpoint_loading(self, temp_dir, synthetic_training_data):
        """Test that checkpoints can be loaded correctly."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        # Train and save checkpoint
        trainer = PyTorchDeepLabV3PlusTrainer(model_path)
        history = trainer.train(epochs=1, batch_size=2, num_workers=0)

        # Load checkpoint in new trainer
        best_checkpoint = os.path.join(model_path, 'checkpoint_best.pth')

        trainer2 = PyTorchDeepLabV3PlusTrainer(model_path)
        trainer2.build_model()
        trainer2.setup_optimizer()
        trainer2.setup_loss(class_weights=[1.0, 1.0, 1.0])
        trainer2.setup_scheduler()

        iteration = trainer2.load_checkpoint(best_checkpoint)

        assert iteration > 0, f"Checkpoint should have iteration > 0, got {iteration}"
        assert trainer2.model is not None, "Model should be loaded"


class TestLearningRateScheduling:
    """Test suite for learning rate scheduling."""

    def test_learning_rate_tracked(self, temp_dir, synthetic_training_data):
        """Test that learning rates are tracked during training."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        trainer = PyTorchDeepLabV3PlusTrainer(model_path)
        trainer.lr_reduction_patience = 1
        trainer.lr_reduction_factor = 0.5

        history = trainer.train(
            epochs=3,
            batch_size=2,
            learning_rate=0.001,
            validation_split=0.3,
            num_workers=0
        )

        # Check that learning rates were tracked
        assert 'learning_rates' in history, "History should contain learning rates"
        assert len(history['learning_rates']) > 0, "Learning rates should be tracked"

        # Verify all learning rates are positive
        for lr in history['learning_rates']:
            assert lr > 0, f"Learning rate should be positive, got {lr}"


class TestOptimizerConfiguration:
    """Test suite for optimizer configuration."""

    def test_optimizer_initialization(self, temp_dir, synthetic_training_data):
        """Test that optimizer is configured correctly."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        trainer = PyTorchDeepLabV3PlusTrainer(model_path)
        trainer.build_model()

        learning_rate = 0.001
        trainer.setup_optimizer(learning_rate=learning_rate)

        assert trainer.optimizer is not None, "Optimizer should be initialized"
        assert isinstance(trainer.optimizer, torch.optim.Adam), "Should use Adam optimizer"

        # Check learning rate
        param_groups = trainer.optimizer.param_groups
        assert len(param_groups) > 0, "Optimizer should have parameter groups"

        actual_lr = param_groups[0]['lr']
        assert actual_lr == learning_rate, \
            f"Learning rate should be {learning_rate}, got {actual_lr}"
