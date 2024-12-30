import keras
from keras import layers, models, applications
import tensorflow as tf
from keras.utils.vis_utils import plot_model

class DeepLabV3Plus:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes

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

    def bifurcation_block(self, block_input, populate_right=False, num1=64, ker1=(1, 1), str1=(1, 1), pad1='valid', num2=64,
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

    def dilated_spatial_pyramid_pooling(self,block_input):
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
        output = layers.Activation('softmax')(x)

        model = models.Model(model_input, output, name='resnet50')

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
class ClaudeDeepLabV3Plus:
    def __init__(self, sxy, num_classes,class_weights):
        self.sxy = sxy
        self.num_classes = num_classes
        self.class_weights = class_weights

    def build_model(self):
        """
        Create DeepLabV3+ model with ResNet50 backbone

        Args:
            input_shape (tuple): Input shape (height, width, channels)
            num_classes (int): Number of classes
            class_weights (dict): Class weights dictionary

        Returns:
            tf.keras.Model: DeepLabV3+ model
        """
        input_shape = (self.sxy, self.sxy, 3)
        # Create base ResNet50 model
        base_model = ResNet50(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
        )

        # Extract features at different levels
        input_layer = base_model.input
        low_level_features = base_model.get_layer('conv2_block3_out').output
        encoder_output = base_model.output

        # ASPP (Atrous Spatial Pyramid Pooling)
        x = Conv2D(256, 1, padding='same', use_bias=False)(encoder_output)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Decoder
        x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        # Low-level features processing
        low_level_features = Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
        low_level_features = BatchNormalization()(low_level_features)
        low_level_features = ReLU()(low_level_features)

        # Concatenate features
        x = Concatenate()([x, low_level_features])

        # Final convolutions
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        # Final classification layer
        outputs = Conv2D(self.num_classes, 1, padding='same', activation='softmax', name='classification')(x)

        # Create model
        model = Model(inputs=input_layer, outputs=outputs)

        # Compile model with weighted categorical crossentropy
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            class_weights=self.class_weights
        )

        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )

        return model