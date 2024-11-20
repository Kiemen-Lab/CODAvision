# https://keras.io/examples/vision/deeplabv3_plus/
"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: October 07, 2024
"""

import time
import pickle
import keras
from keras import layers, models
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import numpy as np
import matplotlib.pyplot as plt


os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
# os.system("nvcc --version")
from glob import glob
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
import warnings
import GPUtil
warnings.filterwarnings('ignore')


def train_segmentation_model(pthDL):
    if not (os.path.isfile(os.path.join(pthDL, 'best_model_net.keras'))):
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
        #Dynamic batch size based on available memory (work in progress)
        # if available_memory<5:
        #     BATCH_SIZE = 3
        # elif available_memory<25 and available_memory>5:
        #     BATCH_SIZE = 4
        # elif available_memory<40 and available_memory>25:
        #     BATCH_SIZE = 5
        # elif available_memory<55 and available_memory>40:
        #     BATCH_SIZE = 6
        # print(f'The batch size is {BATCH_SIZE}')

        BATCH_SIZE = 3
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

        # Define loss function
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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
            return layers.ReLU()(x) # tf.nn.relu(x)

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
            def __init__(self, model, val_data, num_validations=3, early_stopping=True, reduceLRonPlateau=True,
                         monitor='val_accuracy', ES_patience=6, RLRoP_patience=1, factor=0.75, verbose=0,
                         save_best_model=True, filepath='best_model.h5'):
                super(BatchAccCall, self).__init__()
                self._model = model # store as a protected attribute to maintain compatibility with Keras' callbacks
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
                accuracy = logs.get('accuracy')
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

        # Train the model
        model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
        # model.summary()

        print('Starting model training...')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss=loss,
            metrics=["accuracy"],
        )
        num_validations = 3

        plotcall = BatchAccCall(model=model, val_data=val_dataset, num_validations=num_validations,
                                filepath=os.path.join(pthDL, 'best_model_net.keras'), RLRoP_patience=1, factor=0.75)

        history = model.fit(train_dataset, validation_data=val_dataset, callbacks=plotcall, verbose=1, epochs=8)

        # Save model
        print('Saving model...')
        model.save(os.path.join(pthDL, 'net.keras'))
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