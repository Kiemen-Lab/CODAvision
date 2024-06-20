# https://keras.io/examples/vision/deeplabv3_plus/
"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: June 17, 2024
"""

import time
import pickle
import keras
from keras import layers
from keras import ops
import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
os.system("nvcc --version")
from glob import glob
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io


def train_segmentation_model(pthDL):
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        classNames = data['classNames']
        IMAGE_SIZE = data['sxy']
        nm = data['nm']
        if 'model' in f:
            raise ValueError(f'A network has already been trained for model {nm}. Choose a new model name to retrain.')

    # Define paths to training and validation directories
    pthTrain = os.path.join(pthDL, 'training')
    pthValidation = os.path.join(pthDL, 'validation')

    # Get paths to training images and labels
    train_images = sorted(glob(os.path.join(pthTrain, 'im', "*.png")))
    train_masks = sorted(glob(os.path.join(pthTrain, 'label', "*.png")))

    # Get paths to validation images and labels
    val_images = sorted(glob(os.path.join(pthValidation, 'im', "*.png")))
    val_masks = sorted(glob(os.path.join(pthValidation, 'label', "*.png")))

    # Define constants
    BATCH_SIZE = 4
    NUM_CLASSES = len(classNames)  # Number of classes

    # Create TensorFlow dataset
    def read_image(image_path, mask=False):
        image = tf_io.read_file(image_path)
        if mask:
            image = tf_image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        else:
            image = tf_image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        return image

    def load_data(image_list, mask_list):
        image = read_image(image_list)
        mask = read_image(mask_list, mask=True)
        return image, mask

    def data_generator(image_list, mask_list):
        dataset = tf_data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(load_data, num_parallel_calls=tf_data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        return dataset

    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)

    #_____________________Build DeepLabV3+ model_____________________
    def convolution_block(
            block_input,
            num_filters=256,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
    ):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
        )(block_input)
        x = layers.BatchNormalization()(x)
        return ops.nn.relu(x)

    def DilatedSpatialPyramidPooling(dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = convolution_block(x, kernel_size=1, use_bias=True)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear",
        )(x)

        out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
        out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
        out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
        out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = convolution_block(x, kernel_size=1)
        return output

    def DeeplabV3Plus(image_size, num_classes):
        model_input = keras.Input(shape=(image_size, image_size, 3))
        preprocessed = keras.applications.resnet50.preprocess_input(model_input)
        resnet50 = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=preprocessed
        )
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = DilatedSpatialPyramidPooling(x)

        input_a = layers.UpSampling2D(
            size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
            interpolation="bilinear",
        )(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = convolution_block(x)
        x = convolution_block(x)
        x = layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
        model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
        return keras.Model(inputs=model_input, outputs=model_output)

    model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    # model.summary()

    # Ensure TensorFlow is set to use the GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Set memory growth to avoid using all GPU memory
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            print(f"TensorFlow is using the following GPU: {logical_devices[0]}")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU available. Ensure that the NVIDIA GPU and CUDA are correctly installed.")

    # Training
    print('Starting model training...')
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=["accuracy"],
    )
    start = time.time()
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=8)

    training_time = time.time() - start
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # Save model
    print('Saving model...')
    data['model'] = model
    data['history'] = history.history  # Get the model history
    with open(os.path.join(pthDL, 'net.pkl'), 'wb') as f:
        pickle.dump(data, f)

    #_______________________PLotting____________________________

    # Plot Accuracy (Training and Validation)
    sns.set_palette("colorblind")

    # Plot loss and accuracy in a single figure
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    sns.lineplot(data=history.history, x=range(1, len(history.epoch) + 1), y='loss', label='Training Loss')
    sns.lineplot(data=history.history, x=range(1, len(history.epoch) + 1), y='val_loss', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    sns.lineplot(data=history.history, x=range(1, len(history.epoch) + 1), y='accuracy', label='Training Accuracy')
    sns.lineplot(data=history.history, x=range(1, len(history.epoch) + 1), y='val_accuracy',
                 label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return

