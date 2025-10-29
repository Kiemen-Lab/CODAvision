"""
Semantic Segmentation Model Architectures

This module provides implementations of popular semantic segmentation architectures
including DeepLabV3+ and UNet. It includes classes and functions for creating,
configuring, and manipulating these models.

The implementations follow object-oriented design principles and modern Python
coding standards while maintaining backward compatibility with the rest of the
CODAvision codebase.

Original DeepLabV3+ implementation based on: https://keras.io/examples/vision/deeplabv3_plus/

Note: Custom preprocessing functions are used instead of tf.keras.applications.*.preprocess_input
to avoid serialization issues with Ellipsis objects that prevent model saving.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, Model, applications

# Configure TensorFlow to reduce verbosity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class BaseSegmentationModel(ABC):
    """
    Abstract base class for semantic segmentation models.

    This class defines the common interface for all segmentation models
    in the codebase, ensuring consistent usage patterns.

    Attributes:
        input_size (int): Size of the input images (assumes square images)
        num_classes (int): Number of segmentation classes
        l2_regularization_weight (float): L2 regularization weight for Conv2D layers
    """

    def __init__(self, input_size: int, num_classes: int, l2_regularization_weight: float = 0):
        """
        Initialize the segmentation model.

        Args:
            input_size: Size of the input images (assumes square images)
            num_classes: Number of segmentation classes
            l2_regularization_weight: L2 regularization weight for Conv2D layers (default: 0 to match MATLAB)

        Raises:
            ValueError: If l2_regularization_weight is negative
        """
        if l2_regularization_weight < 0:
            raise ValueError(f"L2 regularization weight must be >= 0, got {l2_regularization_weight}")

        self.input_size = input_size
        self.num_classes = num_classes
        self.l2_regularization_weight = l2_regularization_weight  # Set to 0 to match MATLAB (no regularization)

    @abstractmethod
    def build_model(self) -> Model:
        """
        Build and return the segmentation model.

        Returns:
            A compiled Keras model ready for training or inference
        """
        pass


class DeepLabV3Plus(BaseSegmentationModel):
    """
    Implementation of the DeepLabV3+ architecture for semantic segmentation.

    DeepLabV3+ combines a powerful encoder backbone (ResNet50) with
    atrous spatial pyramid pooling and an effective decoder module for
    detailed segmentation results.

    Attributes:
        input_size (int): Size of the input images (assumes square images)
        num_classes (int): Number of segmentation classes
        l2_regularization_weight (float): L2 regularization weight for Conv2D layers
    """

    def resnet50_preprocess(self, x: tf.Tensor) -> tf.Tensor:
        """
        Custom preprocessing for ResNet50 that avoids serialization issues.
        
        This replaces tf.keras.applications.resnet50.preprocess_input which
        uses Ellipsis in its implementation and causes serialization errors.
        
        The preprocessing performs channel-wise centering using ImageNet statistics.
        
        Args:
            x: Input tensor
            
        Returns:
            Preprocessed tensor
        """
        # ImageNet mean values for each channel (BGR order)
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        mean = tf.reshape(mean, [1, 1, 1, 3])
        
        # Convert RGB to BGR by reversing channels
        x = tf.reverse(x, axis=[-1])
        
        # Subtract ImageNet mean
        x = x - mean
        
        return x

    def convolution_block(
        self,
        block_input: tf.Tensor,
        num_filters: int = 256,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        use_bias: bool = False
    ) -> tf.Tensor:
        """
        Create a convolution block with batch normalization and ReLU activation.

        Args:
            block_input: Input tensor
            num_filters: Number of convolution filters
            kernel_size: Size of the convolution kernel
            dilation_rate: Dilation rate for atrous convolution
            use_bias: Whether to use bias in convolution

        Returns:
            Output tensor after convolution, batch normalization, and activation
        """
        # Apply L2 regularization to Conv2D layers (but not to pretrained encoder)
        kernel_regularizer = tf.keras.regularizers.l2(self.l2_regularization_weight) if self.l2_regularization_weight > 0 else None

        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=kernel_regularizer,
        )(block_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def dilated_spatial_pyramid_pooling(self, dspp_input: tf.Tensor) -> tf.Tensor:
        """
        Create a dilated spatial pyramid pooling module.

        This module uses multiple parallel atrous convolutions with different
        dilation rates to capture multi-scale context information.

        Args:
            dspp_input: Input tensor

        Returns:
            Output tensor after DSPP processing
        """
        dims = dspp_input.shape

        # Global average pooling branch
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = self.convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3], dims[-2]),
            interpolation="bilinear",
        )(x)

        # Multiple atrous convolution branches with different dilation rates
        out_1 = self.convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        # Concatenate all branches and apply final convolution
        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = self.convolution_block(x, kernel_size=1)
        return output

    def build_model(self) -> Model:
        """
        Build and return the DeepLabV3+ model.

        The model uses ResNet50 as the encoder backbone, followed by
        dilated spatial pyramid pooling and a decoder that combines
        features from different levels of the encoder.

        Returns:
            Compiled DeepLabV3+ model
        """
        # Input layer
        model_input = tf.keras.Input(shape=(self.input_size, self.input_size, 3))
        # Use custom preprocessing to avoid Ellipsis serialization issue
        preprocessed = self.resnet50_preprocess(model_input)

        # Encoder (ResNet50 backbone)
        resnet50 = applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=preprocessed
        )

        # Get intermediate feature maps for skip connections
        x = resnet50.get_layer("conv4_block6_2_relu").output

        # Apply dilated spatial pyramid pooling
        x = self.dilated_spatial_pyramid_pooling(x)

        # Upsampling and skip connection with earlier feature maps
        input_a = layers.UpSampling2D(
            size=(4, 4),
            interpolation="bilinear",
        )(x)

        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = self.convolution_block(input_b, num_filters=48, kernel_size=1)

        # Combine feature maps and apply convolutions
        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = self.convolution_block(x)
        x = self.convolution_block(x)

        # Final upsampling to original resolution
        x = layers.UpSampling2D(
            size=(4, 4),
            interpolation="bilinear",
        )(x)

        # Output segmentation map (no regularization on final layer)
        outputs = layers.Conv2D(self.num_classes, kernel_size=(1, 1), padding="same")(x)

        # Create and return model
        model = Model(model_input, outputs, name='DeepLabV3_plus')
        return model


class UNet(BaseSegmentationModel):
    """
    Implementation of the UNet architecture for semantic segmentation.

    UNet is a popular encoder-decoder architecture with skip connections
    between encoder and decoder at corresponding resolutions, which helps
    preserve spatial information.

    This implementation uses ResNet50 as the encoder backbone.

    Attributes:
        input_size (int): Size of the input images (assumes square images)
        num_classes (int): Number of segmentation classes
        l2_regularization_weight (float): L2 regularization weight for Conv2D layers
    """

    def conv_block(self, inputs: tf.Tensor, num_filters: int, kernel_size: int = 3) -> tf.Tensor:
        """
        Create a convolutional block with batch normalization and ReLU activation.

        Args:
            inputs: Input tensor
            num_filters: Number of convolution filters
            kernel_size: Size of the convolution kernel

        Returns:
            Output tensor after convolutions, batch normalization, and activation
        """
        # Apply L2 regularization to decoder Conv2D layers
        kernel_regularizer = tf.keras.regularizers.l2(self.l2_regularization_weight) if self.l2_regularization_weight > 0 else None

        x = layers.Conv2D(num_filters, kernel_size, padding="same", kernel_regularizer=kernel_regularizer)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(num_filters, kernel_size, padding="same", kernel_regularizer=kernel_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def decoder_block(
        self,
        inputs: tf.Tensor,
        skip_features: tf.Tensor,
        num_filters: int
    ) -> tf.Tensor:
        """
        Create a decoder block with transposed convolution and skip connections.

        Args:
            inputs: Input tensor from previous layer
            skip_features: Skip connection features from encoder
            num_filters: Number of convolution filters

        Returns:
            Output tensor after processing
        """
        # Apply L2 regularization to decoder transposed convolution
        kernel_regularizer = tf.keras.regularizers.l2(self.l2_regularization_weight) if self.l2_regularization_weight > 0 else None
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", kernel_regularizer=kernel_regularizer)(inputs)
        x = layers.Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def build_unet_resnet50(self, input_shape: Tuple[int, int, int], num_classes: int) -> Model:
        """
        Build a UNet model with ResNet50 encoder.

        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of segmentation classes

        Returns:
            UNet model with ResNet50 encoder
        """
        # Input layer
        inputs = layers.Input(input_shape)

        # Apply custom preprocessing to avoid Ellipsis serialization issue
        # ImageNet mean values for each channel (BGR order)
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        mean = tf.reshape(mean, [1, 1, 1, 3])
        # Convert RGB to BGR and subtract ImageNet mean
        preprocessed = tf.reverse(inputs, axis=[-1]) - mean

        # Encoder (ResNet50 backbone)
        resnet50 = applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=preprocessed,
        )

        # Extract skip connection features from different layers
        s1 = resnet50.get_layer("conv1_relu").output
        s2 = resnet50.get_layer("conv2_block3_out").output
        s3 = resnet50.get_layer("conv3_block4_out").output
        s4 = resnet50.get_layer("conv4_block6_out").output

        # Bridge/bottleneck
        b1 = resnet50.get_layer("conv5_block3_out").output

        # Decoder with skip connections
        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        # Final upsampling and output layer
        # Note: No regularization on final classification layer (best practice to avoid bias in predictions)
        outputs = layers.Conv2DTranspose(
            num_classes,
            (2, 2),
            strides=2,
            padding="same",
            activation="softmax"
        )(d4)

        # Create and return model
        model = Model(inputs, outputs, name="UNetResNet50")
        return model

    def build_model(self) -> Model:
        """
        Build and return the UNet model.

        Returns:
            UNet model with frozen encoder layers
        """
        input_shape = (self.input_size, self.input_size, 3)
        model = self.build_unet_resnet50(input_shape, self.num_classes)

        # Freeze encoder layers (ResNet50 backbone)
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer') and 'resnet' in layer.name:
                layer.trainable = False

        return model


def unfreeze_model(model: Model) -> Model:
    """
    Unfreeze the encoder layers of a model for fine-tuning.

    Args:
        model: The model with frozen encoder layers

    Returns:
        Model with unfrozen encoder layers
    """
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer') and 'resnet' in layer.name:
            layer.trainable = True
    return model


def model_call(name: str, IMAGE_SIZE: int, NUM_CLASSES: int, l2_regularization_weight: float = 0) -> Model:
    """
    Factory function to create a segmentation model of the specified type.

    Args:
        name: Model type ('UNet' or 'DeepLabV3_plus')
        IMAGE_SIZE: Size of input images (assumes square images)
        NUM_CLASSES: Number of segmentation classes
        l2_regularization_weight: L2 regularization weight for Conv2D layers (default: 0 to match MATLAB)

    Returns:
        The requested segmentation model

    Raises:
        ValueError: If an invalid model name is provided
    """
    if name == "UNet":
        model = UNet(IMAGE_SIZE, NUM_CLASSES, l2_regularization_weight).build_model()
    elif name == "DeepLabV3_plus":
        model = DeepLabV3Plus(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES, l2_regularization_weight=l2_regularization_weight).build_model()
    else:
        raise ValueError(f'Incorrect Model Name / Pretrained model of {name} does not exist')

    return model