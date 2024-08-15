# https://keras.io/examples/vision/deeplabv3_plus/
"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: June 17, 2024
"""

import time
import pickle
import keras
from keras import layers
import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
os.system("nvcc --version")
from glob import glob
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def train_segmentation_model(pthDL):
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
        return tf.nn.relu(x)

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
        preprocessed = tf.keras.applications.resnet50.preprocess_input(model_input)
        resnet50 = tf.keras.applications.ResNet50(
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

    class BatchAccCall(keras.callbacks.Callback):
        def __init__(self, val_data, num_validations=3):
            super(BatchAccCall, self).__init__()
            self.batch_accuracies = []
            self.batch_numbers = []
            self.epoch_indices = []
            self.epoch_numbers = []
            self.current_epoch = 0
            self.val_data = val_data
            self.num_validations = num_validations
            self.batch_accuracies = []
            self.batch_numbers = []
            self.validation_losses = []
            self.validation_accuracies = []
            self.batch_count = 0
            self.val_indices = []

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch
            self.epoch_indices.append(self.current_epoch)
            self.validation_steps = np.linspace(0, self.params['steps'], self.num_validations + 1, dtype=int)[1:]
            self.validation_counter = 0
            self.current_step = 0

        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            self.current_step += 1
            self.batch_count += 1
            if self.current_step in self.validation_steps:
                self.run_validation()
                self.val_indices.append(self.current_step+self.params['steps']*(self.current_epoch+1))
            accuracy = logs.get('accuracy')  # Use the metric name you specified
            if accuracy is not None:
                self.batch_accuracies.append(accuracy)
                self.batch_numbers.append(self.params['steps'] * self.current_epoch + batch + 1)

        def run_validation(self):
            val_loss_total = 0
            val_accuracy_total = 0
            num_batches = 0

            for x_val, y_val in self.val_data:
                y_val = tf.cast(y_val, dtype=tf.int32)
                val_logits = self.model(x_val, training=False)
                num_classes = val_logits.shape[-1]
                val_logits_flat = tf.reshape(val_logits, [-1, num_classes])
                y_val_flat = tf.reshape(y_val, [-1])
                predictions = tf.argmax(val_logits_flat, axis=1)
                predictions = tf.cast(predictions, dtype=tf.int32)
                val_loss = tf.keras.losses.sparse_categorical_crossentropy(y_val_flat, val_logits_flat, from_logits=True)
                val_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_val_flat), tf.float32))

                val_loss_total += tf.reduce_mean(val_loss).numpy()
                val_accuracy_total += val_accuracy.numpy()
                num_batches += 1

            val_loss_avg = val_loss_total / num_batches
            val_accuracy_avg = val_accuracy_total / num_batches

            self.validation_losses.append(val_loss_avg)
            self.validation_accuracies.append(val_accuracy_avg)

            print('')
            print(f"Validation at step {self.current_step}: loss = {val_loss_avg}, accuracy = {val_accuracy_avg}")

        def on_train_end(self, logs=None):
            # Plot after training ends
            plt.figure(figsize=(12, 6))
            plt.plot(self.batch_numbers[50:], self.batch_accuracies[50:],color='blue', label='Training Accuracy', linestyle='-')
            plt.plot(self.val_indices,self.validation_accuracies, color='red', label='Validation Accuracy', linestyle='-')
            for epoch in self.epoch_indices:
                plt.axvline(x=(epoch+1) * self.params['steps'], color='grey', linestyle='--', linewidth=1)
                plt.text(
                    (epoch + 0.5) * self.params['steps'],
                    plt.ylim()[1] * 0.95,
                    f'Epoch {epoch}',
                    color='grey',
                    rotation=90,
                    verticalalignment='top',
                    horizontalalignment='right'
                )
            plt.title('Batch-wise Accuracy Over All Epochs')
            plt.xlabel('Batch Number')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.show()

            if self.validation_losses:
                plt.figure(figsize=(12, 6))
                plt.plot(
                    range(1, len(self.validation_losses) + 1),
                    self.validation_losses,
                    linestyle='-',
                    color='red',
                    label='Validation Loss'
                )
                for epoch in self.epoch_indices:
                    plt.axvline(x=(epoch+1) * self.num_validations, color='grey', linestyle='--', linewidth=1)
                    plt.text(
                        (epoch + 0.5) *  self.num_validations,
                        plt.ylim()[1] * 0.95,
                        f'Epoch {epoch}',
                        color='grey',
                        rotation=90,
                        verticalalignment='top',
                        horizontalalignment='right'
                    )
                plt.title('Validation Loss Over Epochs')
                plt.xlabel('Validation Step')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.show()

    model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    # model.summary()

    # Training
    print('Starting model training...')
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=["accuracy"],
    )
    num_validations = 3

    plotcall = BatchAccCall(val_data=val_dataset, num_validations=num_validations)
    checkpoint = ModelCheckpoint(
        filepath= os.path.join(pthDL, 'best_val_net'),  # Path to save the model
        monitor='val_accuracy',  # Metric to monitor
        save_best_only=True,  # Save only the best model
        mode='max',  # 'max' for validation accuracy
        verbose=1  # Print messages when saving
    )
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                    patience=2,
                                    mode='max',
                                    verbose=1)

    start = time.time()
    history = model.fit(train_dataset, validation_data=val_dataset, callbacks=[plotcall,early_stopping], verbose=1, epochs=8)

    training_time = time.time() - start
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # Save model
    print('Saving model...')
    # model.save(os.path.join(pthDL, 'net'))
    data['model'] = model
    data['history'] = history.history  # Get the model history
    with open(os.path.join(pthDL, 'net.pkl'), 'wb') as f:
        pickle.dump(data, f)

    #_______________________PLotting____________________________

    # Plot Accuracy (Training and Validation)
    #sns.set_palette("colorblind")

    # Plot loss and accuracy in a single figure
    #plt.figure(figsize=(12, 6))

    # Plot loss
    #plt.subplot(1, 2, 1)
    #sns.lineplot(data=history.history, x=range(1, len(history.epoch) + 1), y='loss', label='Training Loss')
    #sns.lineplot(data=history.history, x=range(1, len(history.epoch) + 1), y='val_loss', label='Validation Loss')
    #plt.title('Training and Validation Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.grid(True)

    # Plot accuracy
    #plt.subplot(1, 2, 2)
    #sns.lineplot(data=history.history, x=range(1, len(history.epoch) + 1), y='accuracy', label='Training Accuracy')
    #sns.lineplot(data=history.history, x=range(1, len(history.epoch) + 1), y='val_accuracy',
    #             label='Validation Accuracy')
    #plt.title('Training and Validation Accuracy')
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.grid(True)

    #plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    #plt.title('Model accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.legend(['Train', 'Validation'], loc='upper left')

    #plt.tight_layout()
    #plt.show()


    return

