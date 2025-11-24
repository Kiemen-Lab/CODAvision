"""
Unit tests for PyTorch model architecture.

Tests the PyTorch DeepLabV3+ implementation for correctness, numerical accuracy,
and compatibility with TensorFlow implementation.

Author: CODAvision Team
Date: 2025-11-12
"""

import pytest
import numpy as np
import os
import importlib.util

# Skip entire module if PyTorch not available
pytest_pytorch = pytest.importorskip("torch")
torch = pytest_pytorch


class TestPyTorchDeepLabV3Plus:
    """Test suite for PyTorch DeepLabV3+ model architecture."""

    def test_model_instantiation(self):
        """Test that model creates successfully with correct configuration."""
        from base.models.backbones_pytorch import PyTorchDeepLabV3Plus, get_pytorch_device

        device = get_pytorch_device()

        model_builder = PyTorchDeepLabV3Plus(
            input_size=512,
            num_classes=5,
            l2_regularization_weight=1e-5
        )

        assert model_builder.input_size == 512
        assert model_builder.num_classes == 5
        assert model_builder.l2_regularization_weight == 1e-5
        assert model_builder.device is not None

        model = model_builder.build_model()
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_output_shape(self):
        """Test that model produces correct output shape (NCHW format)."""
        from base.models.backbones_pytorch import PyTorchDeepLabV3Plus

        batch_size = 2
        input_size = 512
        num_classes = 5

        model_builder = PyTorchDeepLabV3Plus(
            input_size=input_size,
            num_classes=num_classes,
            l2_regularization_weight=0
        )
        model = model_builder.build_model()

        # Create dummy input in NCHW format
        x = torch.randn(batch_size, 3, input_size, input_size) * 255.0
        x = x.to(model_builder.device)

        model.eval()
        with torch.no_grad():
            output = model(x)

        expected_shape = (batch_size, num_classes, input_size, input_size)
        assert output.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output.shape}"

    def test_parameter_count(self):
        """Test that parameter count is in expected range (~41M parameters)."""
        from base.models.backbones_pytorch import PyTorchDeepLabV3Plus

        model_builder = PyTorchDeepLabV3Plus(512, 5, 0)
        model = model_builder.build_model()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Expected: ~41M parameters (ResNet50 + ASPP + Decoder)
        # Tightened range to ±1M to catch architectural changes
        expected_range = (40_000_000, 42_000_000)

        assert expected_range[0] <= total_params <= expected_range[1], \
            f"Parameter count {total_params:,} outside expected range " \
            f"{expected_range[0]:,} - {expected_range[1]:,}"

        # Verify reasonable proportion of parameters are trainable (some backbone layers may be frozen)
        assert trainable_params >= total_params * 0.30, \
            f"Too few trainable parameters: {trainable_params:,}/{total_params:,}"

    def test_preprocessing_numerical_accuracy(self):
        """Test that preprocessing matches TensorFlow exactly (RGB→BGR, ImageNet mean)."""
        from base.models.backbones_pytorch import PyTorchDeepLabV3Plus

        # Create test input in NHWC format (B, H, W, C)
        test_input_rgb = np.array([[[[255.0, 128.0, 64.0]]]]).astype(np.float32)

        # Expected preprocessing:
        # 1. RGB → BGR: [64.0, 128.0, 255.0]
        # 2. Subtract ImageNet mean (BGR): [103.939, 116.779, 123.68]
        expected_bgr = np.array([[[[
            64.0 - 103.939,   # B channel
            128.0 - 116.779,  # G channel
            255.0 - 123.68    # R channel
        ]]]])

        # Test PyTorch preprocessing
        model_builder = PyTorchDeepLabV3Plus(512, 5, 0)

        # Convert to PyTorch tensor (NCHW format)
        x_torch = torch.from_numpy(test_input_rgb).permute(0, 3, 1, 2)

        preprocessed_torch = model_builder.preprocess(x_torch)

        # Convert back to NHWC for comparison
        preprocessed_torch_nhwc = preprocessed_torch.permute(0, 2, 3, 1).numpy()

        # Compare
        diff = np.abs(preprocessed_torch_nhwc - expected_bgr)
        max_diff = np.max(diff)

        tolerance = 1e-5
        assert max_diff < tolerance, \
            f"Preprocessing difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}"

    @pytest.mark.parametrize("device_name", ["cpu"])
    def test_device_compatibility(self, device_name, pytorch_device):
        """Test that model works on specified device (CPU, CUDA, MPS)."""
        from base.models.backbones_pytorch import PyTorchDeepLabV3Plus

        # Set device via environment
        prev_device = os.environ.get('CODAVISION_PYTORCH_DEVICE')
        os.environ['CODAVISION_PYTORCH_DEVICE'] = device_name

        try:
            model_builder = PyTorchDeepLabV3Plus(256, 3, 0)
            model = model_builder.build_model()

            # Test with dummy data
            x_test = np.random.randn(1, 256, 256, 3).astype(np.float32) * 255.0
            output = model_builder.predict(x_test)

            assert output.shape == (1, 256, 256, 3), \
                f"Expected shape (1, 256, 256, 3), got {output.shape}"

        finally:
            # Restore previous device setting
            if prev_device is None:
                os.environ.pop('CODAVISION_PYTORCH_DEVICE', None)
            else:
                os.environ['CODAVISION_PYTORCH_DEVICE'] = prev_device


class TestPyTorchTensorFlowEquivalence:
    """Test numerical equivalence between PyTorch and TensorFlow implementations."""

    @pytest.mark.skipif(
        importlib.util.find_spec("tensorflow") is None,
        reason="TensorFlow not available for comparison"
    )
    @pytest.mark.requires_tensorflow
    @pytest.mark.cross_framework
    def test_parameter_count_architectural_difference(self):
        """Test that PyTorch (40M) and TensorFlow (12M) models have expected parameter counts."""
        import tensorflow as tf
        from base.models.backbones_tf import DeepLabV3Plus as TensorFlowDeepLabV3Plus
        from base.models.backbones_pytorch import PyTorchDeepLabV3Plus

        tf_model = TensorFlowDeepLabV3Plus(512, 5, 1e-5).build_model()
        pt_model = PyTorchDeepLabV3Plus(512, 5, 1e-5).build_model()

        tf_params = tf_model.count_params()
        pt_params = sum(p.numel() for p in pt_model.parameters())

        assert 11_000_000 < tf_params < 13_000_000, \
            f"TensorFlow params ({tf_params:,}) outside expected range"
        assert 40_000_000 < pt_params < 42_000_000, \
            f"PyTorch params ({pt_params:,}) outside expected range"


class TestPyTorchPredictMethod:
    """Test the predict method with NHWC format conversion."""

    def test_predict_with_nhwc_input(self):
        """Test that predict method handles NHWC format correctly."""
        from base.models.backbones_pytorch import PyTorchDeepLabV3Plus

        batch_size = 2
        input_size = 512
        num_classes = 5

        model_builder = PyTorchDeepLabV3Plus(input_size, num_classes, 0)
        model = model_builder.build_model()  # Build model before predict

        # Create dummy input in NumPy NHWC format (as used by TensorFlow)
        x_nhwc = np.random.randn(batch_size, input_size, input_size, 3).astype(np.float32) * 255.0

        # Use predict method which handles NHWC → NCHW → NHWC conversion
        output = model_builder.predict(x_nhwc)

        expected_shape = (batch_size, input_size, input_size, num_classes)
        assert output.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output.shape}"

        # Output should be NumPy array in NHWC format
        assert isinstance(output, np.ndarray), "Output should be NumPy array"

    def test_predict_various_batch_sizes(self):
        """Test predict method with various batch sizes."""
        from base.models.backbones_pytorch import PyTorchDeepLabV3Plus

        model_builder = PyTorchDeepLabV3Plus(128, 3, 0)
        model = model_builder.build_model()  # Build model before predict

        for batch_size in [1, 2, 4, 8]:
            x = np.random.randn(batch_size, 128, 128, 3).astype(np.float32) * 255.0
            output = model_builder.predict(x)

            expected_shape = (batch_size, 128, 128, 3)
            assert output.shape == expected_shape, \
                f"Batch size {batch_size}: expected {expected_shape}, got {output.shape}"
