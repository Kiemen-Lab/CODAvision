"""
DeepLabV3+ Implementation for Semantic Segmentation

This script implements the DeepLabV3+ architecture for semantic image segmentation
using TensorFlow and Keras. It includes custom loss functions, data generators,
and training utilities.

Original implementation based on: https://keras.io/examples/vision/deeplabv3_plus/

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Date: January 10, 2025
"""
from base.backbones import *

import time
import pickle
import os
import warnings
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, models
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
import GPUtil

# Suppress warnings and TF logging
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"

def calculate_class_weights(mask_list, num_classes, image_size):
    """
    Calculate class weights using median frequency balancing.

    This function computes class weights based on the frequency of each class in the dataset,
    using pixel frequencies per image rather than global counts.

    Args:
        mask_list (list): List of paths to mask images
        num_classes (int): Number of classes in the segmentation task
        image_size (int): Size of the input images (assuming square images)

    Returns:
        np.ndarray: Array of class weights as float32 values

    Note:
        The function uses median frequency balancing to handle class imbalance,
        where rare classes get higher weights than common classes.
    """
    class_pixels = np.zeros(num_classes)
    image_pixels = np.zeros(num_classes)
    epsilon = 1e-5

    total_pixels = image_size * image_size

    for mask_path in mask_list:
        mask = tf.keras.preprocessing.image.load_img(mask_path, color_mode='grayscale')
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        mask = mask.astype(int)

        for i in range(num_classes):
            pixels_in_class = np.sum(mask == i)
            class_pixels[i] += pixels_in_class
            if pixels_in_class > 0:
                image_pixels[i] += total_pixels

    # Calculate frequency for each class
    freq = class_pixels / image_pixels

    # Handle division by zero and invalid values
    freq[np.isinf(freq) | np.isnan(freq)] = epsilon

    # Calculate weights using median frequency balancing
    median_freq = np.median(freq)
    class_weights = median_freq / freq

    return class_weights.astype(np.float32)


class WeightedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Custom loss function implementing weighted sparse categorical crossentropy.

    This class extends the base Keras Loss class to provide a weighted version
    of sparse categorical crossentropy, useful for handling class imbalance
    in segmentation tasks.

    Args:
        class_weights (array-like): Weights for each class
        from_logits (bool): Whether the predictions are logits
        reduction (str): Type of reduction to apply to the loss
        name (str, optional): Name of the loss function
    """

    def __init__(self, class_weights, from_logits=True, reduction='sum_over_batch_size', name=None):
        super().__init__(reduction=reduction, name=name)
        self.class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Compute the weighted loss between predictions and targets.

        Args:
            y_true (tf.Tensor): Ground truth labels
            y_pred (tf.Tensor): Model predictions

        Returns:
            tf.Tensor: Computed loss value
        """
        y_true = tf.cast(y_true, tf.int32)
        y_true_flat = tf.reshape(y_true, [-1])

        # Get weights for each sample based on its true class
        sample_weights = tf.gather(self.class_weights, y_true_flat)

        # Calculate regular sparse categorical crossentropy
        losses = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_flat,
            tf.reshape(y_pred, [tf.shape(y_true_flat)[0], -1]),
            from_logits=self.from_logits
        )

        # Apply weights to the losses
        weighted_losses = losses * sample_weights
        return tf.reduce_mean(weighted_losses)

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights.numpy().tolist(),
            "from_logits": self.from_logits,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create an instance from configuration dictionary."""
        class_weights = tf.convert_to_tensor(config.pop("class_weights"), dtype=tf.float32)
        return cls(class_weights=class_weights, **config)

def train_segmentation_model_cnns(pthDL, retrain_model = False): #ADDED NAME
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        model_type = data['model_type']

    if not (os.path.isfile(os.path.join(pthDL, 'best_model_'+model_type+'.keras'))) or retrain_model:
        #Start training time
        start_time = time.time()


        # Ensure TensorFlow is set to use the GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        available_memory = 0
        if physical_devices:
            try:
                # Set memory growth to avoid using all GPU memory
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                tf.config.set_visible_devices(physical_devices[0], 'GPU')
                logical_devices = tf.config.list_logical_devices('GPU')
                print(f"TensorFlow is using the following GPU: {logical_devices[0]}")
                gpus = GPUtil.getGPUs()
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append({
                        'device': gpu.id,
                        'total_memory': gpu.memoryTotal / 1024,  # MB
                        'free_memory': gpu.memoryFree / 1024,  # MB
                        'used_memory': gpu.memoryUsed / 1024  # MB
                    })
                gpu_info = gpu_info[0]
                available_memory = gpu_info['free_memory']
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        else:
            print("No GPU available. Ensure that the NVIDIA GPU and CUDA are correctly installed.")

        with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
            data = pickle.load(f)
            classNames = data['classNames']
            IMAGE_SIZE = data['sxy']
            model_type = data['model_type']
            nm = data['nm']
            BATCH_SIZE = data['batch_size']
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

        # BATCH_SIZE = 3
        #BATCH_SIZE = 3 #11/13/2024
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

        # Define loss function
        # loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) # unweighted loss
        print("Calculating class weights...")
        class_weights = calculate_class_weights(train_masks, NUM_CLASSES, IMAGE_SIZE)
        print("Class weights:", class_weights)
        loss = WeightedSparseCategoricalCrossentropy(
            class_weights=class_weights,
            from_logits=True,
            reduction='sum_over_batch_size'
        )

        class BatchAccCall(keras.callbacks.Callback):
            def __init__(self, model, val_data, num_validations=3, early_stopping=True, reduceLRonPlateau=True,
                         monitor='val_accuracy', ES_patience=6, RLRoP_patience=1, factor=0.75, verbose=0,
                         save_best_model=True, filepath='best_model.h5'):
                super(BatchAccCall, self).__init__()
                self._model = model
                self.batch_accuracies = []
                self.batch_numbers = []
                self.batch_losses = []
                self.epoch_indices = []
                self.epoch_numbers = []
                self.current_epoch = 0
                self.val_data = val_data
                self.num_validations = num_validations
                self.validation_losses = []
                self.validation_accuracies = []
                self.val_indices = []
                self.early_stopping = early_stopping
                self.RLRoP = reduceLRonPlateau
                self.monitor = monitor
                self.ES_patience = ES_patience
                self.RLRoP_patience = RLRoP_patience
                self.original_RLRoP_patience = RLRoP_patience
                self.verbose = verbose
                if monitor not in ['val_accuracy', 'val_loss']:
                    raise ValueError("Monitor can only be 'val_loss' or 'val_accuracy'")
                self.mode = 'min' if monitor == 'val_loss' else 'max'
                self.save_best_model = save_best_model
                self.save_path = filepath
                self.monitor_op = np.less if self.mode == 'min' else np.greater
                self.best = np.Inf if self.mode == 'min' else -np.Inf
                self.wait = 0
                self.epoch_wait = 0
                self.stopped_epoch = 0
                self.factor = factor

            @property

            def model(self):
                return self._model

            @model.setter
            def model(self, value):
                self._model = value

            def on_epoch_begin(self, epoch, logs=None):
                self.current_epoch = epoch
                self.epoch_indices.append(self.current_epoch)
                self.validation_steps = np.linspace(0, self.params['steps'], self.num_validations + 1, dtype=int)[1:]
                self.validation_counter = 0
                self.current_step = 0

            def on_batch_end(self, batch, logs=None):
                batch_end = time.time()
                logs = logs or {}
                self.current_step += 1
                if self.current_step in self.validation_steps:
                    self.run_validation()
                    self.val_indices.append(self.current_step + self.params['steps'] * (self.current_epoch))
                accuracy = logs.get('accuracy')  # Use the metric name you specified
                if accuracy is not None:
                    self.batch_accuracies.append(accuracy)
                    self.batch_numbers.append(self.params['steps'] * self.current_epoch + batch + 1)
                loss = logs.get('loss')
                if loss is not None:
                    self.batch_losses.append(loss)

            def on_epoch_end(self, epoch, logs=None):
                self.epoch_wait += 1
                if self.epoch_wait > self.RLRoP_patience and self.RLRoP:
                    old_lr = float(self._model.optimizer.learning_rate.numpy())
                    new_lr = old_lr * self.factor
                    self._model.optimizer.learning_rate.assign(new_lr)
                    self.epoch_wait = 0

            def run_validation(self):
                val_loss_total = 0
                val_accuracy_total = 0
                num_batches = 0

                for x_val, y_val in self.val_data:
                    y_val = tf.cast(y_val, dtype=tf.int32)
                    val_logits = self._model(x_val, training=False)
                    num_classes = val_logits.shape[-1]
                    val_logits_flat = tf.reshape(val_logits, [-1, num_classes])
                    y_val_flat = tf.reshape(y_val, [-1])
                    predictions = tf.argmax(val_logits_flat, axis=1)
                    predictions = tf.cast(predictions, dtype=tf.int32)
                    val_loss = tf.keras.losses.sparse_categorical_crossentropy(y_val_flat, val_logits_flat,
                                                                               from_logits=True)
                    val_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_val_flat), tf.float32))

                    val_loss_total += tf.reduce_mean(val_loss).numpy()
                    val_accuracy_total += val_accuracy.numpy()
                    num_batches += 1

                val_loss_avg = val_loss_total / num_batches
                val_accuracy_avg = val_accuracy_total / num_batches

                self.validation_losses.append(val_loss_avg)
                self.validation_accuracies.append(val_accuracy_avg)

                if self.early_stopping:
                    if self.monitor == 'val_loss':
                        current = val_loss_avg
                    elif self.monitor == 'val_accuracy':
                        current = val_accuracy_avg

                    if current is None:
                        return

                    if self.monitor_op(current, self.best):
                        self.best = current
                        self.wait = 0
                        # Save the model if the `save_best_model` flag is set
                        if self.save_best_model:
                            self._model.save(self.save_path)
                            if self.verbose > 0:
                                print(f'\nEpoch {self.current_epoch + 1}: Model saved to {self.save_path}')
                    else:
                        self.wait += 1
                        if self.wait >= self.ES_patience and self.early_stopping:
                            self.stopped_epoch = self.current_epoch
                            self._model.stop_training = True
                            if self.verbose > 0:
                                print(f'\nEpoch {self.current_epoch + 1}: early stopping')

            def on_train_end(self, logs=None):
                # Plot after training ends
                plt.figure(figsize=(12, 6))
                plt.plot(self.batch_numbers[50:], self.batch_accuracies[50:], color='blue', label='Training Accuracy',
                         linestyle='-')
                plt.plot(self.val_indices, self.validation_accuracies, color='red', label='Validation Accuracy',
                         linestyle='-')
                for epoch in self.epoch_indices:
                    plt.axvline(x=(epoch + 1) * self.params['steps'], color='grey', linestyle='--', linewidth=1)
                    plt.text(
                        (epoch + 0.5) * self.params['steps'],
                        plt.ylim()[1] * 0.95,
                        f'Epoch {epoch}',
                        color='grey',
                        rotation=90,
                        verticalalignment='top',
                        horizontalalignment='right'
                    )
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Iteration')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                #plt.show()

                if self.validation_losses:
                    plt.figure(figsize=(12, 6))
                    plt.plot(self.batch_numbers[50:], self.batch_losses[50:], color='blue', label='Training Loss',
                             linestyle='-')
                    plt.plot(self.val_indices, self.validation_losses, linestyle='-', color='red', label='Validation Loss')
                    for epoch in self.epoch_indices:
                        plt.axvline(x=(epoch + 1) * self.params['steps'], color='grey', linestyle='--', linewidth=1)
                        plt.text(
                            (epoch + 0.5) * self.params['steps'],
                            plt.ylim()[1] * 0.85,
                            f'Epoch {epoch}',
                            color='grey',
                            rotation=90,
                            verticalalignment='top',
                            horizontalalignment='right'
                        )
                    plt.title('Training and Validation Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True)
                    #plt.show()

        # Train the model
        model = model_call(model_type,IMAGE_SIZE=IMAGE_SIZE, NUM_CLASSES=NUM_CLASSES)
        #model.summary()

        print('Starting model training...')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss=loss,
            metrics=["accuracy"],
        )
        num_validations = 3
        best_mod_name = f"best_model_{model_type}.keras"
        plotcall = BatchAccCall(model=model, val_data=val_dataset, num_validations=num_validations,
                                filepath=os.path.join(pthDL, best_mod_name), RLRoP_patience=1, factor=0.75)

        history = model.fit(train_dataset, validation_data=val_dataset, verbose=1, callbacks=plotcall, epochs=8)

        # Save model
        print('Saving model...')
        mod_name = f"{model_type}.keras"
        model.save(os.path.join(pthDL, mod_name))
        # data['model'] = model
        data['history'] = history.history  # Get the model history
        with open(os.path.join(pthDL, 'net.pkl'), 'wb') as f:
            pickle.dump(data, f)

        # End training time
        training_time = time.time() - start_time
        hours, rem = divmod(training_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    else:
        print('Model already trained with the same name')

    return