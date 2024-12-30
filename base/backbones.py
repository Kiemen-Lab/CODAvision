"""
Author: Arrun Sivasubramanian (Johns Hopkins - Kiemen Lab)
Date: November 01, 2024
"""

import time
import pickle
import keras
from keras import layers, models, applications
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os

os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
os.system("nvcc --version")
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, Model


class WeightedClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def call(self, inputs):
        # Apply weights to the softmax outputs
        weights = tf.constant([[self.class_weights[i] for i in range(len(self.class_weights))]],
            dtype=tf.float32
        )
        weighted_outputs = inputs * weights
        return weighted_outputs

    def get_config(self):
        # Get the base config first
        config = super(WeightedClassificationLayer, self).get_config()
        # Add class_weights to the config
        config.update({
            "class_weights": self.class_weights,
        })
        return config

class DeepLabV3Plus:
    def __init__(self, input_size, num_classes, class_weights):
        self.input_size = input_size
        self.num_classes = num_classes
        self.class_weights = class_weights

    def convolution_block(self, block_input, num_filters=256, kernel_size=(1, 1), strides=None, padding='valid',
                          dilation_rate=1, use_bias=False):
        if isinstance(padding, int):
            if padding == 0:
                if strides:
                    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                      padding="valid",
                                      use_bias=use_bias,
                                      kernel_initializer=tf.keras.initializers.HeNormal(),
                                      strides=strides)(block_input)
                else:
                    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                      padding="valid",
                                      use_bias=use_bias,
                                      kernel_initializer=tf.keras.initializers.HeNormal(),
                                      )(block_input)

            else:
                x = layers.ZeroPadding2D(padding=(padding, padding))(block_input)
                if strides:
                    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                      padding="valid",
                                      use_bias=use_bias,
                                      kernel_initializer=tf.keras.initializers.HeNormal(),
                                      strides=strides)(x)
                else:
                    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                      padding="valid",
                                      use_bias=use_bias,
                                      kernel_initializer=tf.keras.initializers.HeNormal(),
                                      )(x)
        else:
            if strides:
                x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                                  use_bias=use_bias,
                                  kernel_initializer=tf.keras.initializers.HeNormal(), strides=strides
                                  )(block_input)
            else:
                x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                                  use_bias=use_bias,
                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                  )(block_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def bifurcation_block(self, block_input, populate_right=False, num1=64, ker1=(1, 1), str1=(1, 1), pad1='valid',
                          num2=64,
                          ker2=(3, 3), str2=(1, 1), pad2='same', num3=256, ker3=(1, 1), str3=(1, 1), pad3='valid',
                          numr=256):
        a = self.convolution_block(block_input, num1, ker1, str1, pad1)
        a = self.convolution_block(a, num2, ker2, str2, pad2)
        a = layers.Conv2D(num3, ker3, str3, pad3)(a)
        a = layers.BatchNormalization()(a)

        if populate_right:
            b = layers.Conv2D(numr, ker1, str1, pad1)(block_input)
            b = layers.BatchNormalization()(b)

            x = layers.add([a, b])
            x = layers.Activation('relu')(x)
            return x
        else:
            x = layers.add([a, block_input])
            x = layers.Activation('relu')(x)
            return x

    def dilated_spatial_pyramid_pooling(self, block_input):
        a = self.convolution_block(block_input, 256, (1, 1), (1, 1), 'same')
        b = self.convolution_block(block_input, 256, (3, 3), (1, 1), 'same', dilation_rate=(6, 6))
        c = self.convolution_block(block_input, 256, (3, 3), (1, 1), 'same', dilation_rate=(12, 12))
        d = self.convolution_block(block_input, 256, (3, 3), (1, 1), 'same', dilation_rate=(18, 18))
        output = layers.Concatenate(axis=-1)([a, b, c, d])
        return output

    def build_model(self):
        model_input = keras.Input(shape=(self.input_size, self.input_size, 3))
        x = self.convolution_block(model_input, 64, (7, 7), (2, 2), 3)
        x = layers.MaxPooling2D((3, 3), (2, 2), 'same')(x)

        x = self.bifurcation_block(x, True)
        x = self.bifurcation_block(x, False)

        z = self.convolution_block(x, 48, (1, 1), (1, 1), 'valid')

        y = self.bifurcation_block(x, True, 128, (1, 1), (2, 2), 'valid', 128, (3, 3), (1, 1),
                                   'same', 512, (1, 1), (1, 1), 'valid', 512)
        for i in range(0, 3):
            y = self.bifurcation_block(y, False, 128, (1, 1), (1, 1), 'valid', 128, (3, 3), (1, 1),
                                       'same', 512, (1, 1), (1, 1), 'valid')

        y = self.bifurcation_block(y, True, 256, (1, 1), (2, 2), 'same', 256, (3, 3), (1, 1),
                                   'same', 1024, (1, 1), (1, 1), 'valid', 1024)

        for i in range(0, 5):
            y = self.bifurcation_block(y, False, 256, (1, 1), (1, 1), 'valid', 256, (3, 3), (1, 1),
                                       'same', 1024, (1, 1), (1, 1), 'valid')

        a = self.convolution_block(y, 512, (1, 1), (1, 1), 'valid')
        a = self.convolution_block(a, 512, (3, 3), (1, 1), 'same', dilation_rate=(2, 2))
        a = layers.Conv2D(2048, (1, 1), (1, 1), 'valid')(a)
        a = layers.BatchNormalization()(a)
        b = layers.Conv2D(2048, (1, 1), (1, 1), 'valid')(y)
        b = layers.BatchNormalization()(b)
        y = layers.add([a, b])
        y = layers.Activation('relu')(y)

        for i in range(0, 2):
            a = self.convolution_block(y, 512, (1, 1), (1, 1), 'valid')
            a = self.convolution_block(a, 512, (3, 3), (1, 1), 'same', dilation_rate=(2, 2))
            a = layers.Conv2D(2048, (1, 1), (1, 1), 'valid')(a)
            a = layers.BatchNormalization()(a)
            y = layers.add([a, y])
            y = layers.Activation('relu')(y)

        y = self.dilated_spatial_pyramid_pooling(y)

        y = self.convolution_block(y, 256, (1, 1), (1, 1), 'valid')
        y = layers.Conv2DTranspose(256, (8, 8), (4, 4))(y)
        y = layers.Cropping2D(cropping=((2, 2), (2, 2)))(y)

        x = layers.Concatenate(axis=-1)([y, z])

        x = self.convolution_block(x, 256, (3, 3), (1, 1), 'same')
        x = self.convolution_block(x, 256, (3, 3), (1, 1), 'same')
        x = layers.Conv2D(self.num_classes, (1, 1), (1, 1), 'valid')(x)
        x = layers.Conv2DTranspose(self.num_classes, (8, 8), (4, 4))(x)
        x = layers.Cropping2D(cropping=((2, 2), (2, 2)))(x)
        x = layers.Activation('softmax')(x)
        output = WeightedClassificationLayer(self.class_weights, name='classification')(x)

        model = models.Model(model_input, output, name='resnet50')
        return model


class UNet:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes

    def conv_block(self, inputs, num_filters):
        x = layers.Conv2D(num_filters, (3, 3), padding="same")(inputs)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
        x = layers.ReLU()(x)
        return x

    def encoder_block(self, inputs, num_filters):
        x = self.conv_block(inputs, num_filters)
        p = layers.MaxPooling2D((2, 2))(x)
        return x, p

    def decoder_block(self, inputs, skip_features, num_filters):
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = layers.Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def build_model(self):
        # inputs = layers.Input(self.input_size) #ARRUN
        inputs = keras.Input(shape=(self.input_size, self.input_size, 3))

        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        b1 = self.conv_block(p4, 1024)

        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        outputs = layers.Conv2D(self.num_classes, (1, 1), activation="sigmoid")(d4)

        model = Model(inputs, outputs, name="UNet")
        return model


class UNet3Plus:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes

    def downsampler(self, fmap, count):
        for i in range(count):
            fmap = layers.MaxPool2D(2, dtype='float32')(fmap)
        return fmap

    def upsampler(self, fmap, count):
        for i in range(count):
            fmap = layers.UpSampling2D(2, dtype='float32')(fmap)
        return fmap

    def encoder_block(self, input_features, num_filters, layer):

        conv1_1 = layers.Conv2D(num_filters, 3, padding='same', activation=layers.ReLU(), dtype='float32',
                                name=f'conv-e-{layer}_1_1')(input_features)
        conv1_2 = layers.Conv2D(num_filters, 3, padding='same', activation=layers.ReLU(), dtype='float32',
                                name=f'conv-e-{layer}_1_2')(conv1_1)

        maxpool_fin = layers.MaxPool2D(2, dtype='float32', name=f'maxpool_fin-{layer}')(conv1_2)
        return maxpool_fin

    def decoder_block(self, input_layer, down_skip_connection, up_skip_connection, num_filters, layer):
        convt1_1 = layers.Conv2DTranspose(num_filters, 3, padding='same', activation=layers.ReLU(), dtype='float32',
                                          name=f'convt-d-{layer}_1_1')(input_layer)
        convt1_2 = layers.Conv2DTranspose(num_filters, 3, padding='same', activation=layers.ReLU(), dtype='float32',
                                          name=f'convt-d-{layer}_1_2')(convt1_1)

        if (len(down_skip_connection) == 5):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [self.downsampler(down_skip_connection[0], 4), self.downsampler(down_skip_connection[1], 3),
                 self.downsampler(down_skip_connection[2], 2), self.downsampler(down_skip_connection[3], 1),
                 down_skip_connection[4]])
        elif (len(down_skip_connection) == 4):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [self.downsampler(down_skip_connection[0], 3), self.downsampler(down_skip_connection[1], 2),
                 self.downsampler(down_skip_connection[2], 1), down_skip_connection[3]])
        elif (len(down_skip_connection) == 3):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [self.downsampler(down_skip_connection[0], 2), self.downsampler(down_skip_connection[1], 1),
                 down_skip_connection[2]])
        elif (len(down_skip_connection) == 2):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [self.downsampler(down_skip_connection[0], 1), down_skip_connection[1]])
        elif (len(down_skip_connection) == 1):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [down_skip_connection[0]])
        else:
            print("ERROR INITIALIZING DOWN SKIPS!!")

        if (len(up_skip_connection) == 4):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 4), self.upsampler(up_skip_connection[1], 3),
                 self.upsampler(up_skip_connection[2], 2), self.upsampler(up_skip_connection[3], 1)])
        elif (len(up_skip_connection) == 3):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 3), self.upsampler(up_skip_connection[1], 2),
                 self.upsampler(up_skip_connection[2], 1)])
        elif (len(up_skip_connection) == 2):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 2), self.upsampler(up_skip_connection[1], 1)])
        elif (len(up_skip_connection) == 1):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 1)])

        if (len(up_skip_connection) == 0):
            concat_123 = layers.Concatenate(dtype='float32', name=f'conc-ud-{layer}_123')([convt1_2, skip_connection_d])
        else:
            concat_123 = layers.Concatenate(dtype='float32', name=f'conc-ud-{layer}_123')(
                [convt1_2, skip_connection_d, skip_connection_u])

        conv_fin = layers.Conv2D(num_filters, 3, padding='same', activation=layers.ReLU(), dtype='float32',
                                 name=f'conv_fin-ud-{layer}_123')(concat_123)
        upsampling_fin = layers.UpSampling2D(2, dtype='float32', name=f'upsampling_fin-{layer}')(conv_fin)

        return upsampling_fin

    def bottleneck(self, input_layer, layer, drop=0.2):

        feature_layer = layers.Conv2D(512, 3, padding='same', activation='linear', dtype='float32',
                                      name=f'feature_layer-b-{layer}')(input_layer)
        attention_layer = layers.Conv2D(512, 3, padding='same', activation='sigmoid', dtype='float32',
                                        name=f'attention_layer-b-{layer}')(feature_layer)
        new_input_features = layers.MultiHeadAttention(num_heads=3, key_dim=3, attention_axes=(2, 3), dtype='float32',
                                                       name=f'MHSA_layer-b-{layer}')(input_layer, attention_layer)

        layer_norma = layers.LayerNormalization(dtype='float32', name=f'LN-b-{layer}')(new_input_features)
        if (drop):
            drop = layers.Dropout(drop, dtype='float32', name=f'dropout-b-{layer}')(layer_norma)
            return drop
        return layer_norma

    def build_model(self):
        # input_layer = layers.Input(self.input_size) #ARRUN
        input_layer = keras.Input(shape=(self.input_size, self.input_size, 3))

        e1 = self.encoder_block(input_layer, 32, str(1))
        e2 = self.encoder_block(e1, 64, str(2))
        e3 = self.encoder_block(e2, 128, str(3))
        e4 = self.encoder_block(e3, 256, str(4))
        e5 = self.encoder_block(e4, 512, str(5))
        b = self.bottleneck(e5, str(1))
        d1 = self.decoder_block(input_layer=b, down_skip_connection=[e1, e2, e3, e4, e5], up_skip_connection=[],
                                num_filters=256, layer=str(1))
        d2 = self.decoder_block(d1, [e1, e2, e3, e4], [b], 256, str(2))
        d3 = self.decoder_block(d2, [e1, e2, e3], [b, d1], 128, str(3))
        d4 = self.decoder_block(d3, [e1, e2], [b, d1, d2], 64, str(4))
        d_intout1 = self.decoder_block(d4, [e1], [b, d1, d2, d3], 32, str(5))
        d_intout2 = layers.Conv2D(16, 3, padding='same', activation=layers.LeakyReLU(), name='feature_smoothen_1')(
            d_intout1)
        d_intout3 = layers.Conv2D(16, 3, padding='same', activation=layers.LeakyReLU(), name='feature_smoothen_2')(
            d_intout2)
        d_out = layers.Conv2D(self.num_classes, kernel_size=1, padding='same', activation='softmax', name='segmaps')(
            d_intout3)

        model = Model(inputs=input_layer, outputs=[d_out], name="UNet3_plus")
        return model


class TransUNet:
    def __init__(self, input_size, num_classes, num_filters=64, num_heads=8, transformer_units=512,
                 transformer_depth=4):
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.transformer_depth = transformer_depth

    def convolution_block(self, x, filters, kernel_size=3, strides=1):
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        return tf.nn.relu(x)

    def encoder_block(self, x, filters):
        x = self.convolution_block(x, filters)
        x = self.convolution_block(x, filters)
        p = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x, p

    def transformer_block(self, x):
        seq_len = x.shape[1] * x.shape[2]
        num_channels = x.shape[-1]

        x = layers.Reshape((seq_len, num_channels))(x)

        pos_encoding = layers.Embedding(input_dim=seq_len, output_dim=num_channels)(tf.range(seq_len))
        x += pos_encoding

        for _ in range(self.transformer_depth):
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=num_channels // self.num_heads)(x, x)
            x = x + attn_output
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = x + layers.Dense(self.transformer_units, activation="relu")(x)

        new_h, new_w = self.input_size // 16, self.input_size // 16
        x = layers.Reshape((new_h, new_w, num_channels))(x)
        return x

    def decoder_block(self, x, skip, filters):
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip])
        x = self.convolution_block(x, filters)
        x = self.convolution_block(x, filters)
        return x

    def build_model(self):
        # inputs = layers.Input(shape=self.input_size) #ARRUN
        inputs = keras.Input(shape=(self.input_size, self.input_size, 3))

        skip1, p1 = self.encoder_block(inputs, self.num_filters)
        skip2, p2 = self.encoder_block(p1, self.num_filters * 2)
        skip3, p3 = self.encoder_block(p2, self.num_filters * 4)
        skip4, p4 = self.encoder_block(p3, self.num_filters * 8)

        bottleneck = self.transformer_block(p4)

        d4 = self.decoder_block(bottleneck, skip4, self.num_filters * 8)
        d3 = self.decoder_block(d4, skip3, self.num_filters * 4)
        d2 = self.decoder_block(d3, skip2, self.num_filters * 2)
        d1 = self.decoder_block(d2, skip1, self.num_filters)

        outputs = layers.Conv2D(self.num_classes, kernel_size=(1, 1), activation="softmax")(d1)

        model = Model(inputs=inputs, outputs=outputs, name="TransUNet")
        return model


class CASe_UNet:
    def __init__(self, input_size, num_classes):
        self.input_size = (input_size, input_size, 3)
        self.num_classes = num_classes

    def downsampler(self, fmap, count):
        for i in range(count):
            fmap = layers.MaxPool2D(2, dtype='float32')(fmap)
        return fmap

    def upsampler(self, fmap, count):
        for i in range(count):
            fmap = layers.UpSampling2D(2, dtype='float32')(fmap)
        return fmap

    def encoder_block(self, input_features, num_filters, layer, filter_size=[3, 5, 7]):
        conv1_1 = layers.Conv2D(num_filters, filter_size[0], padding='same', activation=layers.LeakyReLU(alpha=0.1),
                                dtype='float32', name=f'conv-e-{layer}_1_1')(input_features)
        conv1_2 = layers.Conv2D(num_filters, filter_size[1], padding='same', activation=layers.LeakyReLU(alpha=0.1),
                                dtype='float32', name=f'conv-e-{layer}_1_2')(input_features)
        conv1_3 = layers.Conv2D(num_filters, filter_size[2], padding='same', activation=layers.LeakyReLU(alpha=0.1),
                                dtype='float32', name=f'conv-e-{layer}_1_3')(input_features)

        concat_12 = layers.Concatenate(dtype='float32', name=f'conc-e-{layer}_12')([conv1_1, conv1_2])
        concat_13 = layers.Concatenate(dtype='float32', name=f'conc-e-{layer}_13')([conv1_1, conv1_3])
        concat_23 = layers.Concatenate(dtype='float32', name=f'conc-e-{layer}_23')([conv1_2, conv1_3])

        conv2_1 = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32',
                                name=f'conv-e-{layer}_2_1')(concat_12)
        conv2_2 = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32',
                                name=f'conv-e-{layer}_2_2')(concat_23)
        conv2_3 = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32',
                                name=f'conv-e-{layer}_2_3')(concat_13)

        concat_123 = layers.Concatenate(dtype='float32', name=f'conc-e-{layer}_123')([conv2_1, conv2_2, conv2_3])

        conv_fin = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1),
                                 dtype='float32', name=f'conv_fin-e-{layer}_123')(concat_123)
        maxpool_fin = layers.MaxPooling2D(2, dtype='float32', name=f'maxpool_fin-{layer}')(conv_fin)
        return maxpool_fin

    def decoder_block(self, input_layer, down_skip_connection, up_skip_connection, num_filters, layer,
                      filter_size=[3, 5, 7]):
        convt1_1 = layers.Conv2DTranspose(num_filters, filter_size[0], padding='same',
                                          activation=layers.LeakyReLU(alpha=0.1), dtype='float32',
                                          name=f'convt-d-{layer}_1_1')(input_layer)
        convt1_2 = layers.Conv2DTranspose(num_filters, filter_size[1], padding='same',
                                          activation=layers.LeakyReLU(alpha=0.1), dtype='float32',
                                          name=f'convt-d-{layer}_1_2')(input_layer)
        convt1_3 = layers.Conv2DTranspose(num_filters, filter_size[2], padding='same',
                                          activation=layers.LeakyReLU(alpha=0.1), dtype='float32',
                                          name=f'convt-d-{layer}_1_3')(input_layer)

        concat_12 = layers.Concatenate(dtype='float32', name=f'conc-d-{layer}_12')([convt1_1, convt1_2])
        concat_13 = layers.Concatenate(dtype='float32', name=f'conc-d-{layer}_23')([convt1_2, convt1_3])
        concat_23 = layers.Concatenate(dtype='float32', name=f'conc-d-{layer}_13')([convt1_1, convt1_3])

        conv2_1 = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32',
                                name=f'conv-d-{layer}_2_1')(concat_12)
        conv2_2 = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32',
                                name=f'conv-d-{layer}_2_2')(concat_23)
        conv2_3 = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32',
                                name=f'conv-d-{layer}_2_3')(concat_13)

        if (len(down_skip_connection) == 5):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [self.downsampler(down_skip_connection[0], 4), self.downsampler(down_skip_connection[1], 3),
                 self.downsampler(down_skip_connection[2], 2), self.downsampler(down_skip_connection[3], 1),
                 down_skip_connection[4]])
        elif (len(down_skip_connection) == 4):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [self.downsampler(down_skip_connection[0], 3), self.downsampler(down_skip_connection[1], 2),
                 self.downsampler(down_skip_connection[2], 1), down_skip_connection[3]])
        elif (len(down_skip_connection) == 3):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [self.downsampler(down_skip_connection[0], 2), self.downsampler(down_skip_connection[1], 1),
                 down_skip_connection[2]])
        elif (len(down_skip_connection) == 2):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [self.downsampler(down_skip_connection[0], 1), down_skip_connection[1]])
        elif (len(down_skip_connection) == 1):
            skip_connection_d = layers.Concatenate(dtype='float32', name=f'conc-down-d-{layer}')(
                [down_skip_connection[0]])
        else:
            print("ERROR INITIALIZING DOWN SKIPS!!")

        if (len(up_skip_connection) == 4):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 4), self.upsampler(up_skip_connection[1], 3),
                 self.upsampler(up_skip_connection[2], 2), self.upsampler(up_skip_connection[3], 1)])
        elif (len(up_skip_connection) == 3):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 3), self.upsampler(up_skip_connection[1], 2),
                 self.upsampler(up_skip_connection[2], 1)])
        elif (len(up_skip_connection) == 2):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 2), self.upsampler(up_skip_connection[1], 1)])
        elif (len(up_skip_connection) == 1):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 1)])

        if (len(up_skip_connection) == 0):
            concat_123 = layers.Concatenate(dtype='float32', name=f'conc-ud-{layer}_123')(
                [conv2_1, conv2_2, conv2_3, skip_connection_d])
        else:
            concat_123 = layers.Concatenate(dtype='float32', name=f'conc-ud-{layer}_123')(
                [conv2_1, conv2_2, conv2_3, skip_connection_d, skip_connection_u])

        conv_fin = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1),
                                 dtype='float32', name=f'conv_fin-ud-{layer}_123')(concat_123)
        upsampling_fin = layers.UpSampling2D(2, dtype='float32', name=f'upsampling_fin-{layer}')(conv_fin)

        return upsampling_fin

        if (len(up_skip_connection) == 4):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 4), self.upsampler(up_skip_connection[1], 3),
                 self.upsampler(up_skip_connection[2], 2), self.upsampler(up_skip_connection[3], 1)])
        elif (len(up_skip_connection) == 3):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 3), self.upsampler(up_skip_connection[1], 2),
                 self.upsampler(up_skip_connection[2], 1)])
        elif (len(up_skip_connection) == 2):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 2), self.upsampler(up_skip_connection[1], 1)])
        elif (len(up_skip_connection) == 1):
            skip_connection_u = layers.Concatenate(dtype='float32', name=f'conc-up-d-{layer}')(
                [self.upsampler(up_skip_connection[0], 1)])

        if (len(up_skip_connection) == 0):
            concat_123 = layers.Concatenate(dtype='float32', name=f'conc-ud-{layer}_123')(
                [conv2_1, conv2_2, conv2_3, skip_connection_d])
        else:
            concat_123 = layers.Concatenate(dtype='float32', name=f'conc-ud-{layer}_123')(
                [conv2_1, conv2_2, conv2_3, skip_connection_d, skip_connection_u])

        conv_fin = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(), dtype='float32',
                                 name=f'conv_fin-ud-{layer}_123')(concat_123)
        upsampling_fin = layers.UpSampling2D(2, dtype='float32', name=f'upsampling_fin-{layer}')(conv_fin)

        return upsampling_fin

    def bottleneck(self, input_layer, layer, drop=0.2):

        feature_layer = layers.Conv2D(512, 3, padding='same', activation='linear', dtype='float32',
                                      name=f'feature_layer-b-{layer}')(input_layer)
        attention_layer = layers.Conv2D(512, 3, padding='same', activation='sigmoid', dtype='float32',
                                        name=f'attention_layer-b-{layer}')(feature_layer)
        new_input_features = layers.MultiHeadAttention(num_heads=3, key_dim=3, attention_axes=(2, 3), dtype='float32',
                                                       name=f'MHSA_layer-b-{layer}')(input_layer, attention_layer)

        layer_norma = layers.LayerNormalization(dtype='float32', name=f'LN-b-{layer}')(new_input_features)
        if (drop):
            drop = layers.Dropout(drop, dtype='float32', name=f'dropout-b-{layer}')(layer_norma)
            return drop
        return layer_norma

    def build_model(self):

        inputs = layers.Input(self.input_size)
        e1 = self.encoder_block(inputs, 32, str(1))
        e2 = self.encoder_block(e1, 64, str(2))
        e3 = self.encoder_block(e2, 128, str(3))
        e4 = self.encoder_block(e3, 256, str(4))
        e5 = self.encoder_block(e4, 512, str(5))
        b = self.bottleneck(e5, str(1), drop=0.2)
        d1 = self.decoder_block(b, [e1, e2, e3, e4, e5], [], 256, str(1))
        d2 = self.decoder_block(d1, [e1, e2, e3, e4], [b], 256, str(2))
        d3 = self.decoder_block(d2, [e1, e2, e3], [b, d1], 128, str(3))
        d4 = self.decoder_block(d3, [e1, e2], [b, d1, d2], 64, str(4))
        d_intout1 = self.decoder_block(d4, [e1], [b, d1, d2, d3], 32, str(5))
        d_intout2 = layers.Conv2D(16, 3, padding='same', activation=layers.LeakyReLU(), name='feature_smoothen_1')(
            d_intout1)
        d_intout3 = layers.Conv2D(16, 3, padding='same', activation=layers.LeakyReLU(), name='feature_smoothen_2')(
            d_intout2)
        d_out = layers.Conv2D(self.num_classes, kernel_size=1, padding='same', activation='softmax', name='segmaps')(
            d_intout3)

        model = Model(inputs=inputs, outputs=[d_out], name="CASe_UNet")

        return model


# Instantiate and build
def model_call(name, IMAGE_SIZE, NUM_CLASSES, CLASS_WEIGHTS = None):
    if (name == "UNet"):
        model = UNet(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()
    elif (name == "DeepLabV3_plus"):
        model = DeepLabV3Plus(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES, class_weights= CLASS_WEIGHTS).build_model()
    elif (name == "UNet3_plus"):
        model = UNet3Plus(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()
    elif (name == "TransUNet"):
        model = TransUNet(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()
    elif (name == "CASe_UNet"):
        model = CASe_UNet(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()
    else:
        raise ValueError(f'Incorrect Model Name / Pretrained model of {name} does not exist')

    return model