"""
PyTorch implementations of segmentation model backbones.

This module provides PyTorch-based implementations of segmentation models,
specifically DeepLabV3+ with ResNet50 backbone. The implementations are designed
to match the behavior of the TensorFlow versions in backbones_tf.py.

Key features:
- Device management (CPU/CUDA/MPS)
- Preprocessing matching TensorFlow's ImageNet normalization
- Tensor format conversion (NCHW <-> NHWC)
- Compatible with the BaseSegmentationModelInterface
"""

from abc import abstractmethod
from typing import Optional, Tuple
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from base.models.base import BaseSegmentationModelInterface, Framework


def get_pytorch_device() -> str:
    """
    Auto-detect the best available PyTorch device.

    Checks in order: CUDA → MPS (Apple Silicon) → CPU
    Can be overridden with CODAVISION_PYTORCH_DEVICE environment variable.

    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed. Install with: pip install torch torchvision")

    # Check for environment variable override
    env_device = os.getenv('CODAVISION_PYTORCH_DEVICE', '').lower()
    if env_device in ['cuda', 'mps', 'cpu']:
        if env_device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available, falling back to CPU")
            return 'cpu'
        if env_device == 'mps' and not torch.backends.mps.is_available():
            print(f"Warning: MPS requested but not available, falling back to CPU")
            return 'cpu'
        return env_device

    # Auto-detect
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def unfreeze_model(model: nn.Module, encoder_module_name: str = 'encoder') -> None:
    """
    Unfreeze all parameters in the model for fine-tuning.

    Args:
        model: PyTorch model to unfreeze
        encoder_module_name: Name of the encoder module to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


class PyTorchBaseSegmentationModel(BaseSegmentationModelInterface):
    """
    Abstract base class for PyTorch-based segmentation models.

    This class provides the common interface and utilities for all PyTorch
    segmentation models, including device management and tensor format conversion.

    Attributes:
        input_size: Input image size (height and width)
        num_classes: Number of segmentation classes
        l2_regularization_weight: L2 regularization weight (used by optimizer)
        device: PyTorch device ('cuda', 'mps', or 'cpu')
        model: The PyTorch nn.Module model instance
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        l2_regularization_weight: float = 0.0
    ):
        """
        Initialize PyTorch segmentation model.

        Args:
            input_size: Input image size (assumed square)
            num_classes: Number of output classes
            l2_regularization_weight: L2 regularization weight for weight decay
        """
        super().__init__(input_size, num_classes, l2_regularization_weight)
        self.device = get_pytorch_device()
        self.model: Optional[nn.Module] = None

    def _get_framework(self) -> Framework:
        """Return the framework type."""
        return Framework.PYTORCH

    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Build and return the PyTorch model.

        Returns:
            nn.Module: The constructed PyTorch model
        """
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run prediction on input images.

        Handles tensor format conversion from NumPy (NHWC) to PyTorch (NCHW)
        and back to NumPy (NHWC) for compatibility with existing code.

        Args:
            x: Input images as NumPy array with shape (B, H, W, C)

        Returns:
            np.ndarray: Predictions with shape (B, H, W, num_classes)
        """
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        self.model.eval()
        with torch.no_grad():
            # Convert NumPy (NHWC) to PyTorch tensor (NCHW)
            x_tensor = torch.from_numpy(x).float()
            x_tensor = x_tensor.permute(0, 3, 1, 2)  # NHWC → NCHW
            x_tensor = x_tensor.to(self.device)

            # Forward pass
            output = self.model(x_tensor)

            # Convert back to NHWC format
            output = output.permute(0, 2, 3, 1)  # NCHW → NHWC
            output = output.cpu().numpy()

        return output

    def save(self, filepath: str) -> None:
        """
        Save the PyTorch model to disk.

        Args:
            filepath: Path to save the model (will add .pth extension if not present)
        """
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        if not filepath.endswith('.pth'):
            filepath += '.pth'

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'l2_regularization_weight': self.l2_regularization_weight,
        }, filepath)

    def load(self, filepath: str) -> None:
        """
        Load a PyTorch model from disk.

        Args:
            filepath: Path to the saved model
        """
        if not filepath.endswith('.pth'):
            filepath += '.pth'

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Update instance attributes
        self.input_size = checkpoint['input_size']
        self.num_classes = checkpoint['num_classes']
        self.l2_regularization_weight = checkpoint['l2_regularization_weight']

        # Build model and load weights
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.

    ASPP captures multi-scale contextual information by applying parallel
    atrous convolutions with different dilation rates. This implementation
    matches the TensorFlow DeepLabV3+ architecture.

    Architecture:
    - Global average pooling branch
    - 1×1 convolution (no dilation)
    - 3×3 atrous convolution (dilation rate 6)
    - 3×3 atrous convolution (dilation rate 12)
    - 3×3 atrous convolution (dilation rate 18)
    - Concatenate all branches and fuse with 1×1 convolution
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        """
        Initialize ASPP module.

        Args:
            in_channels: Number of input channels (typically 2048 from ResNet50)
            out_channels: Number of output channels for each branch (default 256)
        """
        super().__init__()

        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 1×1 convolution (no dilation)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3×3 atrous convolution (dilation rate 6)
        self.atrous_conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3×3 atrous convolution (dilation rate 12)
        self.atrous_conv12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3×3 atrous convolution (dilation rate 18)
        self.atrous_conv18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion layer: concatenate 5 branches (5 × out_channels) and reduce to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ASPP module.

        Args:
            x: Input tensor with shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor with shape (B, 256, H, W)
        """
        size = x.shape[2:]  # (H, W)

        # Global average pooling branch (upsample to original size)
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)

        # Apply all parallel branches
        feat1x1 = self.conv1x1(x)
        feat6 = self.atrous_conv6(x)
        feat12 = self.atrous_conv12(x)
        feat18 = self.atrous_conv18(x)

        # Concatenate all branches
        out = torch.cat([global_feat, feat1x1, feat6, feat12, feat18], dim=1)

        # Fuse with 1×1 convolution
        out = self.project(out)

        return out


class DecoderModule(nn.Module):
    """
    Decoder module for DeepLabV3+.

    The decoder combines:
    - High-level semantic features from ASPP (stride 16)
    - Low-level detail features from encoder (stride 4)

    Architecture:
    1. Upsample ASPP output by 4× (from stride 16 to stride 4)
    2. Process low-level features with 1×1 convolution
    3. Concatenate high-level and low-level features
    4. Apply two 3×3 convolutions
    5. Upsample by 4× to original resolution
    """

    def __init__(self, low_level_channels: int, num_classes: int):
        """
        Initialize decoder module.

        Args:
            low_level_channels: Number of channels in low-level features (from encoder)
            num_classes: Number of output classes
        """
        super().__init__()

        # Process low-level features
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Two 3×3 convolutions after concatenation (256 + 48 = 304 channels input)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Final classification layer (no regularization, as per TensorFlow implementation)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(
        self,
        high_level_feat: torch.Tensor,
        low_level_feat: torch.Tensor,
        input_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            high_level_feat: ASPP output with shape (B, 256, H/16, W/16)
            low_level_feat: Encoder low-level features with shape (B, C, H/4, W/4)
            input_size: Original input size (H, W) for final upsampling

        Returns:
            torch.Tensor: Output logits with shape (B, num_classes, H, W)
        """
        # Process low-level features
        low_level_feat = self.low_level_conv(low_level_feat)

        # Upsample high-level features by 4×
        high_level_feat = F.interpolate(
            high_level_feat,
            size=low_level_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # Concatenate
        x = torch.cat([high_level_feat, low_level_feat], dim=1)

        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)

        # Final upsampling to original resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        # Classification
        x = self.classifier(x)

        return x


class PyTorchDeepLabV3Plus(PyTorchBaseSegmentationModel):
    """
    PyTorch implementation of DeepLabV3+ with ResNet50 backbone.

    This implementation matches the TensorFlow version in backbones_tf.py,
    including preprocessing, architecture, and output format.

    Architecture:
    - Encoder: ResNet50 pretrained on ImageNet
    - ASPP: Multi-scale context aggregation with dilation rates [1, 6, 12, 18]
    - Decoder: Skip connection fusion and upsampling
    - Output: Logits without activation (for use with cross-entropy loss)

    Key features:
    - Preprocessing matches TensorFlow (RGB→BGR, ImageNet mean subtraction)
    - Tensor format conversion (NCHW ↔ NHWC)
    - Device management (CPU/CUDA/MPS)
    - Compatible with existing training pipelines
    """

    # ImageNet mean values in BGR order (matches TensorFlow preprocessing)
    IMAGENET_MEAN_BGR = [103.939, 116.779, 123.68]

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        l2_regularization_weight: float = 0.0
    ):
        """
        Initialize PyTorch DeepLabV3+ model.

        Args:
            input_size: Input image size (assumed square)
            num_classes: Number of segmentation classes
            l2_regularization_weight: L2 regularization weight (used by optimizer's weight_decay)
        """
        super().__init__(input_size, num_classes, l2_regularization_weight)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input images to match TensorFlow preprocessing.

        Applies:
        1. RGB → BGR channel reversal
        2. ImageNet mean subtraction (BGR order: [103.939, 116.779, 123.68])

        This matches the custom preprocessing in TensorFlow's backbones_tf.py
        to ensure numerical equivalence.

        Args:
            x: Input tensor with shape (B, C, H, W) in RGB order, values [0, 255]

        Returns:
            torch.Tensor: Preprocessed tensor with shape (B, C, H, W) in BGR order
        """
        # RGB → BGR: reverse channel order
        x = torch.flip(x, dims=[1])  # Flip along channel dimension

        # Subtract ImageNet mean (BGR order)
        mean = torch.tensor(
            self.IMAGENET_MEAN_BGR,
            dtype=x.dtype,
            device=x.device
        ).view(1, 3, 1, 1)

        x = x - mean

        return x

    def build_model(self) -> nn.Module:
        """
        Build the complete DeepLabV3+ model.

        Returns:
            nn.Module: Complete DeepLabV3+ model
        """
        model = DeepLabV3PlusModel(
            num_classes=self.num_classes,
            input_size=self.input_size
        )

        model = model.to(self.device)
        self.model = model

        return model


class DeepLabV3PlusModel(nn.Module):
    """
    Complete DeepLabV3+ model combining encoder, ASPP, and decoder.

    This is the actual nn.Module that performs the forward pass.
    Separated from PyTorchDeepLabV3Plus to allow for cleaner architecture.
    """

    def __init__(self, num_classes: int, input_size: int):
        """
        Initialize DeepLabV3+ model.

        Args:
            num_classes: Number of output classes
            input_size: Input image size (for final upsampling)
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Encoder: ResNet50 pretrained on ImageNet
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Extract layers for multi-scale feature extraction
        # layer2: stride 4 (low-level features)
        # layer4: stride 16 (high-level features)
        self.encoder_conv1 = resnet50.conv1
        self.encoder_bn1 = resnet50.bn1
        self.encoder_relu = resnet50.relu
        self.encoder_maxpool = resnet50.maxpool
        self.encoder_layer1 = resnet50.layer1
        self.encoder_layer2 = resnet50.layer2  # Output channels: 512
        self.encoder_layer3 = resnet50.layer3
        self.encoder_layer4 = resnet50.layer4  # Output channels: 2048

        # Freeze encoder initially (matching TensorFlow behavior)
        for param in self.parameters():
            param.requires_grad = False

        # ASPP module
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)

        # Decoder module
        self.decoder = DecoderModule(low_level_channels=512, num_classes=num_classes)

        # Preprocessing (ImageNet mean in BGR order)
        self.register_buffer(
            'imagenet_mean_bgr',
            torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input: RGB → BGR and subtract ImageNet mean.

        Args:
            x: Input tensor (B, C, H, W) in RGB order

        Returns:
            torch.Tensor: Preprocessed tensor in BGR order
        """
        # RGB → BGR
        x = torch.flip(x, dims=[1])

        # Subtract ImageNet mean
        x = x - self.imagenet_mean_bgr

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete DeepLabV3+ model.

        Args:
            x: Input tensor with shape (B, C, H, W) in RGB order, values [0, 255]

        Returns:
            torch.Tensor: Output logits with shape (B, num_classes, H, W)
        """
        input_size = (x.shape[2], x.shape[3])

        # Preprocessing
        x = self.preprocess(x)

        # Encoder: extract multi-scale features
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)

        x = self.encoder_layer1(x)
        low_level_feat = self.encoder_layer2(x)  # Stride 4, channels: 512
        x = self.encoder_layer3(low_level_feat)
        high_level_feat = self.encoder_layer4(x)  # Stride 16, channels: 2048

        # ASPP
        aspp_out = self.aspp(high_level_feat)  # Stride 16, channels: 256

        # Decoder
        output = self.decoder(aspp_out, low_level_feat, input_size)

        return output
