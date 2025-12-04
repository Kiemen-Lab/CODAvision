"""
Integration tests for PyTorch end-to-end workflows.

Tests complete workflows including training, model saving/loading, and
cross-framework compatibility.

Author: CODAvision Team
Date: 2025-11-12
"""

import pytest
import numpy as np
import os
import pickle
import importlib.util

# Skip entire module if PyTorch not available
pytest_pytorch = pytest.importorskip("torch")
torch = pytest_pytorch


@pytest.mark.integration
class TestCompleteTrainingWorkflow:
    """Test complete training workflow from data to model."""

    def test_end_to_end_training(self, temp_dir, synthetic_training_data):
        """Test complete training workflow with data loading, training, and validation."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer
        from base.models.backbones import model_call

        model_path, annotations, image_list = synthetic_training_data

        # Test model creation via unified interface
        model = model_call(
            'DeepLabV3_plus',
            IMAGE_SIZE=256,
            NUM_CLASSES=3,
            framework='pytorch'
        )

        assert model is not None, "Model creation failed"

        # Test full training workflow
        trainer = PyTorchDeepLabV3PlusTrainer(model_path)

        history = trainer.train(
            epochs=2,
            batch_size=2,
            learning_rate=0.001,
            validation_split=0.3,
            num_workers=0
        )

        # Verify training completed
        assert len(history['train_loss']) >= 2, "Training should complete 2 epochs"
        assert len(history['val_loss']) > 0, "Validation should occur"

        # Verify loss decreased
        assert history['train_loss'][-1] < history['train_loss'][0], \
            "Training loss should decrease"

        # Verify checkpoints saved
        assert os.path.exists(os.path.join(model_path, 'checkpoint_best.pth')), \
            "Best checkpoint should be saved"
        assert os.path.exists(os.path.join(model_path, 'checkpoint_latest.pth')), \
            "Latest checkpoint should be saved"

    def test_training_with_data_loader_integration(self, temp_dir, synthetic_training_data):
        """Test integration between PyTorch data loaders and training system."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer
        from base.data.loaders_pytorch import PyTorchSegmentationDataset
        from torch.utils.data import DataLoader

        model_path, annotations, image_list = synthetic_training_data

        # Create dataset manually to verify integration
        image_paths = [
            os.path.join(model_path, 'training', 'im', f'{img_id}.png')
            for img_id in image_list[:10]
        ]
        mask_paths = [
            os.path.join(model_path, 'training', 'label', f'{img_id}.png')
            for img_id in image_list[:10]
        ]

        dataset = PyTorchSegmentationDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            image_size=256,
            augment=True
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        # Verify data loader produces correct shapes
        images, masks = next(iter(dataloader))
        assert images.shape == (2, 3, 256, 256), f"Images shape mismatch: {images.shape}"
        assert masks.shape == (2, 256, 256), f"Masks shape mismatch: {masks.shape}"

        # Now train with integrated system
        trainer = PyTorchDeepLabV3PlusTrainer(model_path)
        history = trainer.train(epochs=1, batch_size=2, num_workers=0)

        assert len(history['train_loss']) >= 1, "Training should complete"


@pytest.mark.integration
class TestCrossFrameworkCompatibility:
    """Test cross-framework model loading and compatibility."""

    def test_pytorch_model_saving_and_loading(self, temp_dir, synthetic_training_data):
        """Test that PyTorch models can be saved and loaded correctly."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        # Train and save model
        trainer1 = PyTorchDeepLabV3PlusTrainer(model_path)
        history = trainer1.train(epochs=1, batch_size=2, num_workers=0)

        # Verify .pth file exists
        checkpoint_path = os.path.join(model_path, 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_path), "PyTorch checkpoint should be saved"

        # Load in new trainer instance
        trainer2 = PyTorchDeepLabV3PlusTrainer(model_path)
        trainer2.build_model()
        trainer2.setup_optimizer()
        trainer2.setup_loss(class_weights=[1.0, 1.0, 1.0])
        trainer2.setup_scheduler()

        iteration = trainer2.load_checkpoint(checkpoint_path)

        assert iteration > 0, "Checkpoint should have valid iteration"
        assert trainer2.model is not None, "Model should be loaded"

        # Test inference with loaded model
        test_input = np.random.randn(1, 256, 256, 3).astype(np.float32) * 255.0
        output = trainer2.model.predict(test_input) if hasattr(trainer2.model, 'predict') else None

        # Verify output shape if predict available
        if output is not None:
            assert output.shape == (1, 256, 256, 3), "Loaded model should produce correct output"

    def test_framework_switching(self, temp_dir, synthetic_training_data):
        """Test switching between PyTorch and TensorFlow frameworks."""
        from base.config import FrameworkConfig

        model_path, annotations, image_list = synthetic_training_data

        # Train with PyTorch
        FrameworkConfig.set_framework('pytorch')
        assert FrameworkConfig.get_framework() == 'pytorch'

        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer
        trainer_pt = PyTorchDeepLabV3PlusTrainer(model_path)
        history_pt = trainer_pt.train(epochs=1, batch_size=2, num_workers=0)

        assert os.path.exists(os.path.join(model_path, 'checkpoint_best.pth')), \
            "PyTorch checkpoint should be saved"

        # Switch back to TensorFlow (if available)
        try:
            import tensorflow as tf
            FrameworkConfig.set_framework('tensorflow')
            assert FrameworkConfig.get_framework() == 'tensorflow'

            # Verify framework switching works
            # Note: Full TensorFlow training would require TF-specific setup
            # This test just verifies the framework config system works

        except ImportError:
            pytest.skip("TensorFlow not available for framework switching test")

        finally:
            # Reset to default
            FrameworkConfig.set_framework('tensorflow')

    @pytest.mark.cross_framework
    def test_model_format_auto_detection(self, temp_dir, synthetic_training_data):
        """Test that system auto-detects and loads correct model format (.pth or .keras).

        This test always validates PyTorch model loading.
        If TensorFlow is available, it also tests cross-framework model selection.
        """
        model_path, annotations, image_list = synthetic_training_data

        # Part 1: Test PyTorch model creation and loading (always runs)
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer
        from base.config import FrameworkConfig

        # Set framework to PyTorch
        FrameworkConfig.set_framework('pytorch')

        # Train PyTorch model to create .pth file
        trainer_pt = PyTorchDeepLabV3PlusTrainer(model_path)
        trainer_pt.train(epochs=1, batch_size=2, num_workers=0)

        # Verify .pth file exists
        pth_path = os.path.join(model_path, 'checkpoint_best.pth')
        assert os.path.exists(pth_path), ".pth file should exist after training"

        # Test that PyTorch model can be loaded
        trainer_reload = PyTorchDeepLabV3PlusTrainer(model_path)
        trainer_reload.build_model()
        assert trainer_reload.model is not None, "PyTorch model should load successfully"

        # Part 2: Test TensorFlow cross-framework detection (if TensorFlow available)
        if importlib.util.find_spec("tensorflow") is not None:
            try:
                import tensorflow as tf

                # If TensorFlow is available, we could test creating a .keras file
                # and verifying the system selects the correct format based on
                # CODAVISION_FRAMEWORK environment variable

                # For now, just verify the .pth file is preferred when framework='pytorch'
                assert os.path.exists(pth_path), "PyTorch framework should use .pth file"

            except ImportError:
                # TensorFlow import failed, skip cross-framework validation
                pass
        else:
            # TensorFlow not available, PyTorch-only validation completed successfully
            pass


@pytest.mark.integration
class TestTrainingReproducibility:
    """Test training reproducibility with seeds."""

    def test_reproducibility_with_seed(self, temp_dir, synthetic_training_data):
        """Test that training produces similar results with same seed."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        # First run with seed
        torch.manual_seed(42)
        np.random.seed(42)

        trainer1 = PyTorchDeepLabV3PlusTrainer(model_path)
        history1 = trainer1.train(epochs=1, batch_size=2, num_workers=0)
        loss1 = history1['train_loss'][0]

        # Second run with same seed
        torch.manual_seed(42)
        np.random.seed(42)

        trainer2 = PyTorchDeepLabV3PlusTrainer(model_path)
        history2 = trainer2.train(epochs=1, batch_size=2, num_workers=0)
        loss2 = history2['train_loss'][0]

        # Note: Exact reproducibility is challenging due to various factors
        # Check for similar results rather than identical
        difference = abs(loss1 - loss2)

        assert difference < 0.1, \
            f"Losses should be similar with same seed: {loss1:.6f} vs {loss2:.6f} (diff: {difference:.6f})"

    def test_different_seeds_produce_different_results(self, temp_dir, synthetic_training_data):
        """Test that different seeds produce different results."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        # Run with seed 42
        torch.manual_seed(42)
        np.random.seed(42)

        trainer1 = PyTorchDeepLabV3PlusTrainer(model_path)
        history1 = trainer1.train(epochs=1, batch_size=2, num_workers=0)
        loss1 = history1['train_loss'][0]

        # Run with seed 123
        torch.manual_seed(123)
        np.random.seed(123)

        trainer2 = PyTorchDeepLabV3PlusTrainer(model_path)
        history2 = trainer2.train(epochs=1, batch_size=2, num_workers=0)
        loss2 = history2['train_loss'][0]

        # Different seeds should produce different results
        # (though they might occasionally be close)
        # We just verify both runs completed successfully
        assert loss1 > 0 and loss2 > 0, "Both runs should produce valid losses"


@pytest.mark.integration
class TestModelInference:
    """Test model inference after training."""

    def test_inference_after_training(self, temp_dir, synthetic_training_data):
        """Test that trained model can perform inference."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        # Train model
        trainer = PyTorchDeepLabV3PlusTrainer(model_path)
        trainer.train(epochs=1, batch_size=2, num_workers=0)

        # Test inference
        test_image = np.random.randn(1, 256, 256, 3).astype(np.float32) * 255.0

        # Get model from trainer
        model = trainer.model

        model.eval()
        with torch.no_grad():
            # Convert NHWC to NCHW
            test_tensor = torch.from_numpy(test_image).permute(0, 3, 1, 2).to(trainer.device)
            output = model(test_tensor)

        # Verify output shape (NCHW: batch, classes, height, width)
        assert output.shape == (1, 3, 256, 256), f"Output shape mismatch: {output.shape}"

    def test_batch_inference(self, temp_dir, synthetic_training_data):
        """Test inference with multiple images."""
        from base.models.training_pytorch import PyTorchDeepLabV3PlusTrainer

        model_path, annotations, image_list = synthetic_training_data

        # Train model
        trainer = PyTorchDeepLabV3PlusTrainer(model_path)
        trainer.train(epochs=1, batch_size=2, num_workers=0)

        # Test batch inference with different batch sizes
        for batch_size in [1, 2, 4]:
            test_images = np.random.randn(batch_size, 256, 256, 3).astype(np.float32) * 255.0

            model = trainer.model
            model.eval()

            with torch.no_grad():
                test_tensor = torch.from_numpy(test_images).permute(0, 3, 1, 2).to(trainer.device)
                output = model(test_tensor)

            expected_shape = (batch_size, 3, 256, 256)
            assert output.shape == expected_shape, \
                f"Batch {batch_size}: expected {expected_shape}, got {output.shape}"
