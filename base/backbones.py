"""
Author: Arrun Sivasubramanian (Johns Hopkins - Kiemen Lab)
Date: November 01, 2024
"""

import time
import pickle
import keras
from keras import layers, models, applications, activations
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
os.system("nvcc --version")
import warnings
import GPUtil
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, Model

class DeepLabV3Plus:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes

    def convolution_block(self, block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False):
        x = layers.Conv2D(num_filters,kernel_size=kernel_size,dilation_rate=dilation_rate,padding="same",use_bias=use_bias,
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
        inputs = layers.Input(self.input_size)
        #preprocessed = applications.resnet50.preprocess_input(inputs)
        resnet50 = applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
        
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = self.dilated_spatial_pyramid_pooling(x)

        input_a = layers.UpSampling2D(
            size=(self.input_size[0] // 4 // x.shape[1], self.input_size[1] // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
        
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = self.convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = self.convolution_block(x)
        x = self.convolution_block(x)

        x = layers.UpSampling2D(
            size=(self.input_size[0] // x.shape[1], self.input_size[1] // x.shape[2]),
            interpolation="bilinear",
        )(x)
        
        outputs = layers.Conv2D(self.num_classes, kernel_size=(1, 1), padding="same")(x)
        
        model = Model(inputs, outputs, name='DeepLabV3+')
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
        inputs = layers.Input(self.input_size)

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

    def downsampler(self,fmap,count):
        for i in range(count):
            fmap = layers.MaxPool2D(2, dtype='float32')(fmap)
        return fmap

    def upsampler(self,fmap,count):
        for i in range(count):
            fmap = layers.UpSampling2D(2, dtype='float32')(fmap)
        return fmap
    
    def encoder_block(self,input_features,num_filters,layer):

        conv1_1 = layers.Conv2D(num_filters,3, padding = 'same',activation = layers.ReLU(), dtype='float32',name = f'conv-e-{layer}_1_1')(input_features)
        conv1_2 = layers.Conv2D(num_filters,3, padding = 'same',activation = layers.ReLU(), dtype='float32',name = f'conv-e-{layer}_1_2')(conv1_1)

        maxpool_fin = layers.MaxPool2D(2, dtype='float32',name = f'maxpool_fin-{layer}')(conv1_2)
        return maxpool_fin
    
    def decoder_block(self,input_layer,down_skip_connection, up_skip_connection, num_filters, layer):
        convt1_1 = layers.Conv2DTranspose(num_filters, 3, padding = 'same',activation = layers.ReLU(), dtype='float32',name = f'convt-d-{layer}_1_1')(input_layer)
        convt1_2 = layers.Conv2DTranspose(num_filters, 3, padding = 'same',activation = layers.ReLU(), dtype='float32',name = f'convt-d-{layer}_1_2')(convt1_1)

        if(len(down_skip_connection) == 5):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],4),self.downsampler(down_skip_connection[1],3),self.downsampler(down_skip_connection[2],2),self.downsampler(down_skip_connection[3],1),down_skip_connection[4]])
        elif(len(down_skip_connection) == 4):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],3),self.downsampler(down_skip_connection[1],2),self.downsampler(down_skip_connection[2],1),down_skip_connection[3]])
        elif(len(down_skip_connection) == 3):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],2),self.downsampler(down_skip_connection[1],1),down_skip_connection[2]])
        elif(len(down_skip_connection) == 2):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],1),down_skip_connection[1]])
        elif(len(down_skip_connection) == 1):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([down_skip_connection[0]])
        else:
            print("ERROR INITIALIZING DOWN SKIPS!!")


        if(len(up_skip_connection) == 4):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],4),self.upsampler(up_skip_connection[1],3),self.upsampler(up_skip_connection[2],2),self.upsampler(up_skip_connection[3],1)])
        elif(len(up_skip_connection) == 3):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],3),self.upsampler(up_skip_connection[1],2),self.upsampler(up_skip_connection[2],1)])
        elif(len(up_skip_connection) == 2):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],2),self.upsampler(up_skip_connection[1],1)])
        elif(len(up_skip_connection) == 1):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],1)])

        if(len(up_skip_connection) == 0):
            concat_123 = layers.Concatenate(dtype='float32',name = f'conc-ud-{layer}_123')([convt1_2,skip_connection_d])
        else:
            concat_123 = layers.Concatenate(dtype='float32',name = f'conc-ud-{layer}_123')([convt1_2,skip_connection_d,skip_connection_u])

        conv_fin = layers.Conv2D(num_filters,3, padding = 'same',activation = layers.ReLU(), dtype='float32',name = f'conv_fin-ud-{layer}_123')(concat_123)
        upsampling_fin = layers.UpSampling2D(2, dtype='float32',name = f'upsampling_fin-{layer}')(conv_fin)

        return upsampling_fin

    def bottleneck(self, input_layer,layer,drop = 0.2):
    
        feature_layer = layers.Conv2D(512,3,padding = 'same',activation = 'linear', dtype='float32',name = f'feature_layer-b-{layer}')(input_layer)
        attention_layer = layers.Conv2D(512,3,padding = 'same',activation = 'sigmoid', dtype='float32',name = f'attention_layer-b-{layer}')(feature_layer)
        new_input_features = layers.MultiHeadAttention(num_heads=3, key_dim=3, attention_axes=(2, 3), dtype='float32',name = f'MHSA_layer-b-{layer}')(input_layer,attention_layer)

        layer_norma = layers.LayerNormalization(dtype='float32',name = f'LN-b-{layer}')(new_input_features)
        if(drop):
            drop = layers.Dropout(drop, dtype='float32',name = f'dropout-b-{layer}')(layer_norma)
            return drop
        return batch_norma


    def build_model(self):
        input_layer = layers.Input(self.input_size)
        e1 = self.encoder_block(input_layer,32,str(1))
        e2 = self.encoder_block(e1,64,str(2))
        e3 = self.encoder_block(e2,128,str(3))
        e4 = self.encoder_block(e3,256,str(4))
        e5 = self.encoder_block(e4,512,str(5))
        b = self.bottleneck(e5,str(1))
        d1 = self.decoder_block(input_layer = b,down_skip_connection = [e1,e2,e3,e4,e5],up_skip_connection=[],num_filters = 256,layer = str(1))
        d2 = self.decoder_block(d1,[e1,e2,e3,e4],[b],256,str(2))
        d3 = self.decoder_block(d2,[e1,e2,e3],[b,d1],128,str(3))
        d4 = self.decoder_block(d3,[e1,e2],[b,d1,d2],64,str(4))
        d_intout1 = self.decoder_block(d4,[e1],[b,d1,d2,d3],32,str(5))
        d_intout2 = layers.Conv2D(16,3,padding = 'same',activation = layers.LeakyReLU(),name = 'feature_smoothen_1')(d_intout1)
        d_intout3 = layers.Conv2D(16,3,padding = 'same',activation = layers.LeakyReLU(),name = 'feature_smoothen_2')(d_intout2)
        d_out = layers.Conv2D(self.num_classes, kernel_size = 1, padding = 'same', activation = 'softmax',name = 'segmaps')(d_intout3)

        model = Model(inputs = input_layer, outputs = [d_out],name = "UNet3+")
        return model
        
class TransUNet:
    def __init__(self, input_size, num_classes, num_filters=64, num_heads=8, transformer_units=512, transformer_depth=4):
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

        new_h, new_w = self.input_size[0] // 16, self.input_size[1] // 16
        x = layers.Reshape((new_h, new_w, num_channels))(x)
        return x


    def decoder_block(self, x, skip, filters):
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip])
        x = self.convolution_block(x, filters)
        x = self.convolution_block(x, filters)
        return x

    def build_model(self):
        inputs = layers.Input(shape=self.input_size)

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
        
        model = Model(inputs=inputs, outputs=outputs, name = "TransUNet")
        return model

class CASe_UNet:
    def __init__(self, input_size, num_classes):
        self.input_size = (input_size,input_size,3)
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
        conv1_1 = layers.Conv2D(num_filters, filter_size[0], padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32', name=f'conv-e-{layer}_1_1')(input_features)
        conv1_2 = layers.Conv2D(num_filters, filter_size[1], padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32', name=f'conv-e-{layer}_1_2')(input_features)
        conv1_3 = layers.Conv2D(num_filters, filter_size[2], padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32', name=f'conv-e-{layer}_1_3')(input_features)

        concat_12 = layers.Concatenate(dtype='float32', name=f'conc-e-{layer}_12')([conv1_1, conv1_2])
        concat_13 = layers.Concatenate(dtype='float32', name=f'conc-e-{layer}_13')([conv1_1, conv1_3])
        concat_23 = layers.Concatenate(dtype='float32', name=f'conc-e-{layer}_23')([conv1_2, conv1_3])

        conv2_1 = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32', name=f'conv-e-{layer}_2_1')(concat_12)
        conv2_2 = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32', name=f'conv-e-{layer}_2_2')(concat_23)
        conv2_3 = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32', name=f'conv-e-{layer}_2_3')(concat_13)

        concat_123 = layers.Concatenate(dtype='float32', name=f'conc-e-{layer}_123')([conv2_1, conv2_2, conv2_3])

        conv_fin = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(alpha=0.1), dtype='float32', name=f'conv_fin-e-{layer}_123')(concat_123)
        maxpool_fin = layers.MaxPooling2D(2, dtype='float32', name=f'maxpool_fin-{layer}')(conv_fin)
        return maxpool_fin
    
    def decoder_block(self,input_layer,down_skip_connection, up_skip_connection, num_filters, layer, filter_size=[3,5,7]):
        convt1_1 = layers.Conv2DTranspose(num_filters, filter_size[0], padding = 'same',activation = layers.LeakyReLU(alpha=0.1), dtype='float32',name = f'convt-d-{layer}_1_1')(input_layer)
        convt1_2 = layers.Conv2DTranspose(num_filters, filter_size[1], padding = 'same',activation = layers.LeakyReLU(alpha=0.1), dtype='float32',name = f'convt-d-{layer}_1_2')(input_layer)
        convt1_3 = layers.Conv2DTranspose(num_filters, filter_size[2], padding = 'same',activation = layers.LeakyReLU(alpha=0.1), dtype='float32',name = f'convt-d-{layer}_1_3')(input_layer)

        concat_12 = layers.Concatenate(dtype='float32',name = f'conc-d-{layer}_12')([convt1_1,convt1_2])
        concat_13 = layers.Concatenate(dtype='float32',name = f'conc-d-{layer}_23')([convt1_2,convt1_3])
        concat_23 = layers.Concatenate(dtype='float32',name = f'conc-d-{layer}_13')([convt1_1,convt1_3])

        conv2_1 = layers.Conv2D(num_filters,1, padding = 'same',activation = layers.LeakyReLU(alpha=0.1), dtype='float32',name = f'conv-d-{layer}_2_1')(concat_12)
        conv2_2 = layers.Conv2D(num_filters,1, padding = 'same',activation = layers.LeakyReLU(alpha=0.1), dtype='float32',name = f'conv-d-{layer}_2_2')(concat_23)
        conv2_3 = layers.Conv2D(num_filters,1, padding = 'same',activation = layers.LeakyReLU(alpha=0.1), dtype='float32',name = f'conv-d-{layer}_2_3')(concat_13)

        if(len(down_skip_connection) == 5):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],4),self.downsampler(down_skip_connection[1],3),self.downsampler(down_skip_connection[2],2),self.downsampler(down_skip_connection[3],1),down_skip_connection[4]])
        elif(len(down_skip_connection) == 4):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],3),self.downsampler(down_skip_connection[1],2),self.downsampler(down_skip_connection[2],1),down_skip_connection[3]])
        elif(len(down_skip_connection) == 3):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],2),self.downsampler(down_skip_connection[1],1),down_skip_connection[2]])
        elif(len(down_skip_connection) == 2):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],1),down_skip_connection[1]])
        elif(len(down_skip_connection) == 1):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([down_skip_connection[0]])
        else:
            print("ERROR INITIALIZING DOWN SKIPS!!")


        if(len(up_skip_connection) == 4):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],4),self.upsampler(up_skip_connection[1],3),self.upsampler(up_skip_connection[2],2),self.upsampler(up_skip_connection[3],1)])
        elif(len(up_skip_connection) == 3):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],3),self.upsampler(up_skip_connection[1],2),self.upsampler(up_skip_connection[2],1)])
        elif(len(up_skip_connection) == 2):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],2),self.upsampler(up_skip_connection[1],1)])
        elif(len(up_skip_connection) == 1):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],1)])

        if(len(up_skip_connection) == 0):
            concat_123 = layers.Concatenate(dtype='float32',name = f'conc-ud-{layer}_123')([conv2_1,conv2_2,conv2_3,skip_connection_d])
        else:
            concat_123 = layers.Concatenate(dtype='float32',name = f'conc-ud-{layer}_123')([conv2_1,conv2_2,conv2_3,skip_connection_d,skip_connection_u])

        conv_fin = layers.Conv2D(num_filters,1, padding = 'same',activation = layers.LeakyReLU(alpha=0.1), dtype='float32',name = f'conv_fin-ud-{layer}_123')(concat_123)
        upsampling_fin = layers.UpSampling2D(2, dtype='float32',name = f'upsampling_fin-{layer}')(conv_fin)

        return upsampling_fin


        if(len(up_skip_connection) == 4):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],4),self.upsampler(up_skip_connection[1],3),self.upsampler(up_skip_connection[2],2),self.upsampler(up_skip_connection[3],1)])
        elif(len(up_skip_connection) == 3):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],3),self.upsampler(up_skip_connection[1],2),self.upsampler(up_skip_connection[2],1)])
        elif(len(up_skip_connection) == 2):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],2),self.upsampler(up_skip_connection[1],1)])
        elif(len(up_skip_connection) == 1):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],1)])

        if(len(up_skip_connection) == 0):
            concat_123 = layers.Concatenate(dtype='float32',name = f'conc-ud-{layer}_123')([conv2_1,conv2_2,conv2_3,skip_connection_d])
        else:
            concat_123 = layers.Concatenate(dtype='float32',name = f'conc-ud-{layer}_123')([conv2_1,conv2_2,conv2_3,skip_connection_d,skip_connection_u])

        conv_fin = layers.Conv2D(num_filters,1, padding = 'same',activation = layers.LeakyReLU(), dtype='float32',name = f'conv_fin-ud-{layer}_123')(concat_123)
        upsampling_fin = layers.UpSampling2D(2, dtype='float32',name = f'upsampling_fin-{layer}')(conv_fin)

        return upsampling_fin

    
    def bottleneck(self,input_layer,layer,drop = 0.2):

        feature_layer = layers.Conv2D(512,3,padding = 'same',activation = 'linear', dtype='float32',name = f'feature_layer-b-{layer}')(input_layer)
        attention_layer = layers.Conv2D(512,3,padding = 'same',activation = 'sigmoid', dtype='float32',name = f'attention_layer-b-{layer}')(feature_layer)
        new_input_features = layers.MultiHeadAttention(num_heads=3, key_dim=3, attention_axes=(2, 3), dtype='float32',name = f'MHSA_layer-b-{layer}')(input_layer,attention_layer)

        layer_norma = layers.LayerNormalization(dtype='float32',name = f'LN-b-{layer}')(new_input_features)
        if(drop):
            drop = layers.Dropout(drop, dtype='float32',name = f'dropout-b-{layer}')(layer_norma)
            return drop
        return layer_norma

    def build_model(self):
        
        inputs = layers.Input(self.input_size)
        e1 = self.encoder_block(inputs,32,str(1))
        e2 = self.encoder_block(e1,64,str(2))
        e3 = self.encoder_block(e2,128,str(3))
        e4 = self.encoder_block(e3,256,str(4))
        e5 = self.encoder_block(e4,512,str(5))
        b = self.bottleneck(e5,str(1),drop = 0.2)
        d1 = self.decoder_block(b,[e1,e2,e3,e4,e5],[],256,str(1))
        d2 = self.decoder_block(d1,[e1,e2,e3,e4],[b],256,str(2))
        d3 = self.decoder_block(d2,[e1,e2,e3],[b,d1],128,str(3))
        d4 = self.decoder_block(d3,[e1,e2],[b,d1,d2],64,str(4))
        d_intout1 = self.decoder_block(d4,[e1],[b,d1,d2,d3],32,str(5))
        d_intout2 = layers.Conv2D(16,3,padding = 'same',activation = layers.LeakyReLU(),name = 'feature_smoothen_1')(d_intout1)
        d_intout3 = layers.Conv2D(16,3,padding = 'same',activation = layers.LeakyReLU(),name = 'feature_smoothen_2')(d_intout2)
        d_out = layers.Conv2D(self.num_classes, kernel_size = 1, padding = 'same', activation = 'softmax',name = 'segmaps')(d_intout3)

        model = Model(inputs = inputs, outputs = [d_out], name="CASe_UNet")
        
        return model


class CoREAM2_SegNet:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
    
    def downsampler(self, fmap,count):
        for i in range(count):
            fmap = layers.MaxPool2D(2, dtype='float32')(fmap)
        return fmap

    def upsampler(self, fmap,count):
        for i in range(count):
            fmap = layers.UpSampling2D(2, dtype='float32')(fmap)
        return fmap

    def residual_encoder(self, input_features,num_filters,layer,filter_size=[3,5,7]):
        #input_copy = input_features
        conv_1 = layers.Conv2D(num_filters, 3, padding='same')(input_features)
        ln_1 = layers.LayerNormalization()(conv_1)
        activation_enc = activations.gelu(ln_1)
    
        conv1_1 = layers.Conv2D(num_filters,filter_size[0], padding = 'same',activation = layers.LeakyReLU(), dtype='float32',name = f'conv-e-{layer}_1_1')(input_features)
        conv1_2 = layers.Conv2D(num_filters,filter_size[1], padding = 'same',activation = layers.LeakyReLU(), dtype='float32',name = f'conv-e-{layer}_1_2')(input_features)
        conv1_3 = layers.Conv2D(num_filters,filter_size[2], padding = 'same',activation = layers.LeakyReLU(), dtype='float32',name = f'conv-e-{layer}_1_3')(input_features)
    
        concat_12 = layers.Concatenate(dtype='float32',name = f'conc-e-{layer}_12')([conv1_1,conv1_2])
        concat_13 = layers.Concatenate(dtype='float32',name = f'conc-e-{layer}_23')([conv1_2,conv1_3])
        concat_23 = layers.Concatenate(dtype='float32',name = f'conv-e-{layer}_13')([conv1_1,conv1_3])
    
        conv2_1 = layers.Conv2D(filter_size[0],1, padding = 'same',activation = layers.LeakyReLU(), dtype='float32',name = f'conv-e-{layer}_2_1')(concat_12)
        conv2_2 = layers.Conv2D(filter_size[1],1, padding = 'same',activation = layers.LeakyReLU(), dtype='float32',name = f'conv-e-{layer}_2_2')(concat_23)
        conv2_3 = layers.Conv2D(filter_size[2],1, padding = 'same',activation = layers.LeakyReLU(), dtype='float32',name = f'conv-e-{layer}_2_3')(concat_13)
    
        concat_123 = layers.Concatenate(dtype='float32',name = f'conc-e-{layer}_123')([conv2_1,conv2_2,conv2_3])
        conv_fin = layers.Conv2D(num_filters,1, padding = 'same',activation = layers.LeakyReLU(), dtype='float32',name = f'conv-e-{layer}_fin')(concat_123)
    
        add = layers.Add()([activation_enc,conv_fin])
        ln_2 = layers.LayerNormalization()(add)
        out = activations.gelu(ln_2)
    
        maxpool_fin = layers.MaxPool2D(2, dtype='float32',name = f'maxpool_fin-{layer}')(out)
        return maxpool_fin

    def residual_decoder(self, input_features, num_filters, layer, filter_size=[3, 5, 7]):
        conv_1 = layers.Conv2D(num_filters, 3, padding='same')(input_features)
        ln_1 = layers.LayerNormalization()(conv_1)
        activation_dec = activations.gelu(ln_1)
        
        convt1_1 = layers.Conv2DTranspose(num_filters, filter_size[0], padding='same', activation=layers.LeakyReLU(), dtype='float32', name=f'convt-d-{layer}_1_1')(input_features)
        convt1_2 = layers.Conv2DTranspose(num_filters, filter_size[1], padding='same', activation=layers.LeakyReLU(), dtype='float32', name=f'convt-d-{layer}_1_2')(input_features)
        convt1_3 = layers.Conv2DTranspose(num_filters, filter_size[2], padding='same', activation=layers.LeakyReLU(), dtype='float32', name=f'convt-d-{layer}_1_3')(input_features)
    
        concat_12 = layers.Concatenate(dtype='float32', name=f'conc-d-{layer}_12')([convt1_1, convt1_2])
        concat_13 = layers.Concatenate(dtype='float32', name=f'conc-d-{layer}_23')([convt1_2, convt1_3])
        concat_23 = layers.Concatenate(dtype='float32', name=f'conc-d-{layer}_13')([convt1_1, convt1_3])
    
        conv2_1 = layers.Conv2D(filter_size[0], 1, padding='same', activation=layers.LeakyReLU(), dtype='float32', name=f'conv-d-{layer}_2_1')(concat_12)
        conv2_2 = layers.Conv2D(filter_size[1], 1, padding='same', activation=layers.LeakyReLU(), dtype='float32', name=f'conv-d-{layer}_2_2')(concat_23)
        conv2_3 = layers.Conv2D(filter_size[2], 1, padding='same', activation=layers.LeakyReLU(), dtype='float32', name=f'conv-d-{layer}_2_3')(concat_13)
        
        concat_123 = layers.Concatenate(dtype='float32', name=f'conc-d-{layer}_123')([conv2_1, conv2_2, conv2_3])
        
        conv_fin = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(), dtype='float32', name=f'conv-d-{layer}_fin')(concat_123)
    
        add = layers.Add()([activation_dec, conv_fin])
        ln_2 = layers.LayerNormalization()(add)
        out = activations.gelu(ln_2)
        
        upsampling_fin = layers.UpSampling2D(2, dtype='float32', name=f'upsampling_fin-{layer}')(out)
        
        return upsampling_fin

    def pixel_attention(self,input_features, num_chans):
        pointwise_conv = layers.GlobalAveragePooling2D()(input_features)
        ln_layer = layers.BatchNormalization()(pointwise_conv)
        prob_dist = layers.Softmax(axis=-1)(ln_layer)
    
        low_lvl = layers.Conv2D(num_chans, 1, padding='same')(input_features)
        mult_layer = layers.Multiply()([low_lvl, prob_dist])
    
        add_layer = layers.Add()([mult_layer, low_lvl])
    
        return add_layer

    def skip_concat(self, down_skip_connection, up_skip_connection, num_filters, layer):
        if(len(down_skip_connection) == 4):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],4),self.downsampler(down_skip_connection[1],3),self.downsampler(down_skip_connection[2],2),self.downsampler(down_skip_connection[3],1)])
        elif(len(down_skip_connection) == 3):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],3),self.downsampler(down_skip_connection[1],2),self.downsampler(down_skip_connection[2],1)])
        elif(len(down_skip_connection) == 2):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],2),self.downsampler(down_skip_connection[1],1)])
        elif(len(down_skip_connection) == 1):
            skip_connection_d = layers.Concatenate(dtype='float32',name = f'conc-down-d-{layer}')([self.downsampler(down_skip_connection[0],1)])
        else:
            print("ERROR INITIALIZING DOWN SKIPS!!")
        
        if(len(up_skip_connection) == 4):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],3),self.upsampler(up_skip_connection[1],2),self.upsampler(up_skip_connection[2],1),up_skip_connection[3]])
        elif(len(up_skip_connection) == 3):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],2),self.upsampler(up_skip_connection[1],1),up_skip_connection[2]])
        elif(len(up_skip_connection) == 2):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([self.upsampler(up_skip_connection[0],1),up_skip_connection[1]])
        elif(len(up_skip_connection) == 1):
            skip_connection_u = layers.Concatenate(dtype='float32',name = f'conc-up-d-{layer}')([up_skip_connection[0]])
        else:
            print("ERROR INITIALIZING UP SKIPS!!")
        
        concat_123 = layers.Concatenate(dtype='float32',name = f'conc-ud-{layer}_123')([skip_connection_d,skip_connection_u])
        conv_fin = layers.Conv2D(num_filters, 1, padding='same', activation=layers.LeakyReLU(), dtype='float32', name=f'conv-s-{layer}_fin')(concat_123)
        
        ln = layers.LayerNormalization()(conv_fin)
        out = activations.gelu(ln)
        
        return out

    def build_model(self):
        input_layer = layers.Input((self.input_size,self.input_size,3))
        e1 = self.residual_encoder(input_layer, 32, str(1))
        e2 = self.residual_encoder(e1, 64, str(2))
        e3 = self.residual_encoder(e2, 128, str(3))
        e4 = self.residual_encoder(e3, 256, str(4))
        
        c1 = self.skip_concat([e1,e2,e3],[e4],256,str(1))
        afm1 = self.pixel_attention(c1,256)
        d1 = self.residual_decoder(afm1, 128, str(1))
        
        c2 = self.skip_concat([e1,e2],[e4,d1],128,str(2))
        afm2 = self.pixel_attention(c2,128)
        d2 = self.residual_decoder(afm2, 64, str(2))
        
        c3 = self.skip_concat([e1],[e4,d1,d2],64,str(3))
        afm3 = self.pixel_attention(c3,64)
        d3 = self.residual_decoder(afm3, 32, str(3))
        
        afm4 = self.pixel_attention(d3,32)
        d4 = self.residual_decoder(afm4, 16, str(5))
        
        d_intout1 = layers.Conv2D(16,3,padding = 'same',activation = layers.LeakyReLU(),name = 'feature_smoothen_1')(d4)
        d_intout2 = layers.Conv2D(16,3,padding = 'same',activation = layers.LeakyReLU(),name = 'feature_smoothen_2')(d_intout1)
        d_out = layers.Conv2D(self.num_classes, kernel_size = 1, padding = 'same', activation = 'softmax',name = 'segmaps')(d_intout2)
        
        model = Model(inputs = input_layer, outputs = [d_out],name = "CoREAM2_SegNet")
        return model 
        
# Instantiate and build
def model_call(name, IMAGE_SIZE, NUM_CLASSES):
    if(name == "UNet"):
        model = UNet(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()    
    elif(name == "DeepLabV3+"):
        model = DeepLabV3Plus(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()
    elif(name == "UNet3+"):
        model = UNet3Plus(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()   
    elif(name == "TransUNet"):
        model = TransUNet(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model() 
    elif(name == "CASe_UNet"):
        model = CASe_UNet(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()
    elif(name == "CoREAM2_SegNet"):
        model = CoREAM2_SegNet(input_size=IMAGE_SIZE, num_classes=NUM_CLASSES).build_model()
    else:
        raise ValueError(f'Incorrect Model Name / Pretrained model of {name} does not exist')
                                                   
    return model