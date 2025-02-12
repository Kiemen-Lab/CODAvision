"""
DeepLabV3+ ans UNet Implementation for Semantic Segmentation

This script implements the DeepLabV3+ and UNet architecture for semantic image segmentation
using TensorFlow and Keras. It includes the backbones for both models.

Original implementation based on: https://keras.io/examples/vision/deeplabv3_plus/

Authors:
    Valentina Matos (Johns Hopkins - Kiemen/Wirtz Lab)
    Tyler Newton (JHU - DSAI)
    Arrun Sivasubramanian (Johns Hopkins - Kiemen Lab)

Date: February 11, 2025
"""

import os
import keras
from keras import layers, applications
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2DTranspose, Concatenate



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
os.system("nvcc --version")
import warnings


warnings.filterwarnings('ignore')




class DeepLabV3Plus:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes

    def convolution_block(self, block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False):
        x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same",
                          use_bias=use_bias,
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          )(block_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def dilated_spatial_pyramid_pooling(self, dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = self.convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear",
        )(x)

        out_1 = self.convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = self.convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = self.convolution_block(x, kernel_size=1)
        return output

    def build_model(self):
        # inputs = layers.Input(self.input_size) ARRUN
        model_input = keras.Input(shape=(self.input_size, self.input_size, 3))
        preprocessed = tf.keras.applications.resnet50.preprocess_input(model_input)
        # preprocessed = applications.resnet50.preprocess_input(inputs)
        resnet50 = applications.ResNet50(weights="imagenet", include_top=False, input_tensor=preprocessed)

        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = self.dilated_spatial_pyramid_pooling(x)

        input_a = layers.UpSampling2D(
            size=(self.input_size // 4 // x.shape[1], self.input_size // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)

        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = self.convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = self.convolution_block(x)
        x = self.convolution_block(x)

        x = layers.UpSampling2D(
            size=(self.input_size // x.shape[1], self.input_size // x.shape[2]),
            interpolation="bilinear",
        )(x)

        outputs = layers.Conv2D(self.num_classes, kernel_size=(1, 1), padding="same")(x)

        model = Model(preprocessed, outputs, name='DeepLabV3_plus')

        return model

def unfreeze_model(model):
    """Unfreeze the encoder layers for fine-tuning"""
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer') and 'resnet' in layer.name:
            layer.trainable = True
    return model


class UNet:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes

    def conv_block(self, inputs, num_filters, kernel_size=3):
        """Convolutional block with batch normalization"""
        x = Conv2D(num_filters, kernel_size, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, kernel_size, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def decoder_block(self, inputs, skip_features, num_filters):
        """Decoder block with skip connections"""
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def UNetResNet50(self, input_shape, num_classes):
        """UNet with ResNet50 encoder"""
        inputs = Input(input_shape)
        # Preprocess input
        preprocessed = tf.keras.applications.resnet50.preprocess_input(inputs)

        # ResNet50 as encoder (pre-trained on ImageNet)
        resnet50 = ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=preprocessed,
        )

        # Skip connections (getting intermediate layers)
        s1 = resnet50.get_layer("conv1_relu").output  # 256
        s2 = resnet50.get_layer("conv2_block3_out").output  # 128
        s3 = resnet50.get_layer("conv3_block4_out").output  # 64
        s4 = resnet50.get_layer("conv4_block6_out").output  # 32

        # Bridge
        b1 = resnet50.get_layer("conv5_block3_out").output  # 16

        # Decoder
        d1 = self.decoder_block(b1, s4, 512)  # 32
        d2 = self.decoder_block(d1, s3, 256)  # 64
        d3 = self.decoder_block(d2, s2, 128)  # 128
        d4 = self.decoder_block(d3, s1, 64)  # 256

        # Additional upsampling layers to reach 1024x1024
        d5 = Conv2DTranspose(32, (2, 2), strides=2, padding="same")(d4)  # 1024
        d5 = self.conv_block(d5, 32)


        outputs = Conv2D(num_classes, (1, 1), activation="sigmoid")(d5)

        model = Model(inputs, outputs, name="UNetResNet50")
        return model

    def build_unet(self, image_size, num_classes):
        """Create and return the UNet-ResNet50 model"""
        input_shape = (image_size, image_size, 3)
        model = self.UNetResNet50(input_shape, num_classes)

        # Freeze the encoder layers initially
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer') and 'resnet' in layer.name:
                layer.trainable = False

        return model


# Instantiate and build
def model_call(name, IMAGE_SIZE, NUM_CLASSES):
    if name == "UNet":
        model = UNet(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_unet(IMAGE_SIZE, NUM_CLASSES)
    elif name == "DeepLabV3_plus":
        model = DeepLabV3Plus(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()
    else:
        raise ValueError(f'Incorrect Model Name / Pretrained model of {name} does not exist')

    return model