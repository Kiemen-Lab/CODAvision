"""
DeepLabV3+ Implementation for Semantic Segmentation

This module implements training functions for semantic segmentation models
using TensorFlow and Keras. It includes custom loss functions, data generators,
and training utilities for various architectures including DeepLabV3+ and UNet.

Original implementation based on: https://keras.io/examples/vision/deeplabv3_plus/

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)
    Arrun Sivasubramanian (Johns Hopkins - Kiemen Lab)

Date: March 10, 2025
"""

import time
import pickle
import os
import warnings
from glob import glob
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import tensorflow as tf
import keras
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
import GPUtil

from base.backbones import model_call, unfreeze_model
from .logger import Logger

# Suppress warnings and TensorFlow logs
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"


class WeightedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Custom loss function implementing weighted sparse categorical crossentropy.

    This class extends the base Keras Loss class to provide a weighted version
    of sparse categorical crossentropy, useful for handling class imbalance
    in segmentation tasks.

    Attributes:
        class_weights (tf.Tensor): Weights for each class
        from_logits (bool): Whether the predictions are logits
    """

    def __init__(
        self,
        class_weights: Union[List[float], np.ndarray],
        from_logits: bool = True,
        reduction: str = 'sum_over_batch_size',
        name: Optional[str] = None
    ):
        """
        Initialize the weighted sparse categorical crossentropy loss.

        Args:
            class_weights: Weights for each class
            from_logits: Whether the predictions are logits
            reduction: Type of reduction to apply to the loss
            name: Name of the loss function
        """
        super().__init__(reduction=reduction, name=name)
        self.class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the weighted loss between predictions and targets.

        Args:
            y_true: Ground truth labels
            y_pred: Model predictions

        Returns:
            Computed loss value
        """
        y_true = tf.cast(y_true, tf.int32)
        y_true_flat = tf.reshape(y_true, [-1])

        # Prevent extreme values for numerical stability
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # Apply class weights to each pixel
        sample_weights = tf.gather(self.class_weights, y_true_flat)

        # Compute the sparse categorical crossentropy
        losses = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_flat,
            tf.reshape(y_pred, [tf.shape(y_true_flat)[0], -1]),
            from_logits=self.from_logits
        )

        # Apply weights and return mean
        weighted_losses = losses * sample_weights
        return tf.reduce_mean(weighted_losses)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights.numpy().tolist(),
            "from_logits": self.from_logits,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WeightedSparseCategoricalCrossentropy':
        """Create an instance from configuration dictionary."""
        class_weights = tf.convert_to_tensor(config.pop("class_weights"), dtype=tf.float32)
        return cls(class_weights=class_weights, **config)


class DataGenerator:
    """
    Class for generating TensorFlow datasets from image and mask paths.

    This class handles loading and preprocessing of images and masks for
    training and validation of segmentation models.
    """

    def __init__(self, image_size: int, batch_size: int):
        """
        Initialize the data generator.

        Args:
            image_size: Size to resize images and masks to (assumes square images)
            batch_size: Number of samples per batch
        """
        self.image_size = image_size
        self.batch_size = batch_size

    def read_image(self, image_path: str, mask: bool = False) -> tf.Tensor:
        """
        Read and preprocess an image or mask.

        Args:
            image_path: Path to the image file
            mask: Whether the image is a mask (single channel) or not (RGB)

        Returns:
            Preprocessed image tensor
        """
        image = tf_io.read_file(image_path)
        if mask:
            image = tf_image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
        else:
            image = tf_image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])

        # Resize to the target size
        image = tf_image.resize(images=image, size=[self.image_size, self.image_size])
        return image

    def load_data(self, image_path: str, mask_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load an image and its corresponding mask.

        Args:
            image_path: Path to the image file
            mask_path: Path to the mask file

        Returns:
            Tuple of (image, mask) tensors
        """
        image = self.read_image(image_path)
        mask = self.read_image(mask_path, mask=True)
        return image, mask

    def create_dataset(
        self,
        image_paths: List[str],
        mask_paths: List[str]
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from lists of image and mask paths.

        Args:
            image_paths: List of paths to image files
            mask_paths: List of paths to mask files

        Returns:
            TensorFlow dataset containing batches of (image, mask) pairs
        """
        # Create a dataset from the file paths
        dataset = tf_data.Dataset.from_tensor_slices((image_paths, mask_paths))

        # Map the loading function to each element
        dataset = dataset.map(
            lambda img, mask: self.load_data(img, mask),
            num_parallel_calls=tf_data.AUTOTUNE
        )

        # Batch the dataset
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        return dataset


class BatchAccuracyCallback(keras.callbacks.Callback):
    """
    Custom callback for tracking batch-level metrics and performing validation.

    This callback allows for more fine-grained monitoring of training progress,
    including running validation at specified intervals and implementing
    early stopping and learning rate reduction.
    """

    def __init__(
        self,
        model: keras.Model,
        val_data: tf.data.Dataset,
        loss_function: tf.keras.losses.Loss,
        logger: Optional[Logger] = None,
        num_validations: int = 3,
        early_stopping: bool = True,
        reduce_lr_on_plateau: bool = True,
        monitor: str = 'val_accuracy',
        es_patience: int = 6,
        lr_patience: int = 1,
        lr_factor: float = 0.75,
        verbose: int = 0,
        save_best_model: bool = True,
        filepath: str = 'best_model.h5'
    ):
        """
        Initialize the callback.

        Args:
            model: The model being trained
            val_data: Validation dataset
            loss_function: Loss function used for validation
            logger: Logger for recording training information
            num_validations: Number of validations to perform per epoch
            early_stopping: Whether to use early stopping
            reduce_lr_on_plateau: Whether to reduce learning rate on plateau
            monitor: Metric to monitor ('val_loss' or 'val_accuracy')
            es_patience: Patience for early stopping
            lr_patience: Patience for learning rate reduction
            lr_factor: Factor to reduce learning rate by
            verbose: Verbosity level
            save_best_model: Whether to save the best model
            filepath: Path to save the best model to
        """
        super(BatchAccuracyCallback, self).__init__()
        self.logger = logger
        self._model = model
        self.loss_function = loss_function

        # Metrics tracking
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

        # Early stopping configuration
        self.early_stopping = early_stopping
        self.reduce_lr = reduce_lr_on_plateau
        self.monitor = monitor
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.original_lr_patience = lr_patience
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
        self.lr_factor = lr_factor

        # Log validation dataset information if logger is available
        if self.logger:
            self.logger.logger.info("\nInitial Validation Dataset Information:")
            self.logger.log_dataset_info(val_data, "Initial-Validation")

    @property
    def model(self):
        """Get the model being trained."""
        return self._model

    @model.setter
    def model(self, value):
        """Set the model being trained."""
        self._model = value

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the start of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs (unused)
        """
        self.current_epoch = epoch
        self.epoch_indices.append(self.current_epoch)

        # Set validation steps throughout the epoch
        self.validation_steps = np.linspace(
            0, self.params['steps'],
            self.num_validations + 1,
            dtype=int
        )[1:]

        self.validation_counter = 0
        self.current_step = 0

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch.

        Args:
            batch: Current batch number within the epoch
            logs: Dictionary of logs from training
        """
        logs = logs or {}
        self.current_step += 1

        # Run validation at specified steps
        if self.current_step in self.validation_steps:
            self.run_validation()
            # Record the global step (across all epochs)
            self.val_indices.append(
                self.current_step + self.params['steps'] * self.current_epoch
            )

        # Record batch metrics
        accuracy = logs.get('accuracy')
        if accuracy is not None:
            self.batch_accuracies.append(accuracy)
            # Record the global batch number
            self.batch_numbers.append(
                self.params['steps'] * self.current_epoch + batch + 1
            )

        loss = logs.get('loss')
        if loss is not None:
            self.batch_losses.append(loss)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs (unused)
        """
        # Check if learning rate should be reduced
        self.epoch_wait += 1
        if self.epoch_wait > self.lr_patience and self.reduce_lr:
            old_lr = float(self._model.optimizer.learning_rate.numpy())
            new_lr = old_lr * self.lr_factor
            self._model.optimizer.learning_rate.assign(new_lr)

            if self.verbose > 0 and self.logger:
                self.logger.logger.info(
                    f"\nEpoch {epoch + 1}: Reducing learning rate from {old_lr} to {new_lr}"
                )

            self.epoch_wait = 0

    def run_validation(self):
        """
        Run validation on the validation dataset.

        This method computes validation loss and accuracy, and implements
        early stopping and model saving logic.
        """
        val_loss_total = 0
        val_accuracy_total = 0
        num_batches = 0

        try:
            # Evaluate on validation data
            for x_val, y_val in self.val_data:
                y_val = tf.cast(y_val, dtype=tf.int32)
                val_logits = self._model(x_val, training=False)
                num_classes = val_logits.shape[-1]

                val_logits_flat = tf.reshape(val_logits, [-1, num_classes])
                y_val_flat = tf.reshape(y_val, [-1])

                predictions = tf.argmax(val_logits_flat, axis=1)
                predictions = tf.cast(predictions, dtype=tf.int32)

                val_loss = self.loss_function(y_val_flat, val_logits_flat)
                val_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_val_flat), tf.float32))

                val_loss_total += tf.reduce_mean(val_loss).numpy()
                val_accuracy_total += val_accuracy.numpy()
                num_batches += 1

            # Compute averages
            val_loss_avg = val_loss_total / num_batches
            val_accuracy_avg = val_accuracy_total / num_batches

            # Record validation metrics
            self.validation_losses.append(val_loss_avg)
            self.validation_accuracies.append(val_accuracy_avg)

            # Log validation metrics
            if self.logger:
                self.logger.log_validation_metrics(
                    val_logits, y_val,
                    loss=val_loss_avg,
                    accuracy=val_accuracy_avg
                )

            # Early stopping logic
            if self.early_stopping:
                if self.monitor == 'val_loss':
                    current = val_loss_avg
                elif self.monitor == 'val_accuracy':
                    current = val_accuracy_avg

                if current is None:
                    return

                # Check if this is the best model so far
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0

                    # Save the best model
                    if self.save_best_model:
                        tf.keras.models.save_model(
                            self._model,
                            self.save_path,
                            save_format='tf'
                        )
                        if self.verbose > 0 and self.logger:
                            self.logger.logger.info(
                                f'\nEpoch {self.current_epoch + 1}: '
                                f'Model saved to {self.save_path}'
                            )
                else:
                    # Increment wait counter for early stopping
                    self.wait += 1
                    if self.wait >= self.es_patience and self.early_stopping:
                        self.stopped_epoch = self.current_epoch
                        self._model.stop_training = True
                        if self.verbose > 0 and self.logger:
                            self.logger.logger.info(
                                f'\nEpoch {self.current_epoch + 1}: early stopping'
                            )

        except Exception as e:
            # Log errors and re-raise
            if self.logger:
                self.logger.log_error(f"Validation failed: {str(e)}")
            raise


class SegmentationModelTrainer:
    """
    Base class for training semantic segmentation models.

    This class handles common functionality for training semantic segmentation
    models, including data loading, preprocessing, and model training.
    """

    def __init__(self, model_path: str):
        """
        Initialize the trainer.

        Args:
            model_path: Path to the directory containing model data
        """
        self.model_path = model_path
        self.logger = None
        self.model = None
        self.model_data = None
        self.model_type = None
        self.image_size = None
        self.num_classes = None
        self.class_names = None
        self.batch_size = None
        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None
        self.loss_function = None

        # GPU information for logging
        self.gpu_info = self._setup_gpu()

        # Initialize model data and parameters
        self._load_model_data()

    def _setup_gpu(self) -> Dict[str, Any]:
        """
        Configure TensorFlow to use GPU and return GPU information.

        Returns:
            Dictionary with GPU information or empty dict if no GPU is available
        """
        physical_devices = tf.config.list_physical_devices('GPU')
        gpu_info = {}

        if physical_devices:
            try:
                # Configure GPU memory growth
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)

                # Use only the first GPU
                tf.config.set_visible_devices(physical_devices[0], 'GPU')
                logical_devices = tf.config.list_logical_devices('GPU')
                print(f"TensorFlow is using the following GPU: {logical_devices[0]}")

                # Get GPU memory information
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use the first GPU
                    gpu_info = {
                        'device': gpu.id,
                        'total_memory': gpu.memoryTotal / 1024,  # GB
                        'free_memory': gpu.memoryFree / 1024,    # GB
                        'used_memory': gpu.memoryUsed / 1024     # GB
                    }
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU available. Training will proceed on the CPU.")
            print("Ensure that the NVIDIA GPU and CUDA are correctly installed if you intended to use a GPU.")

        return gpu_info

    def _load_model_data(self):
        """
        Load model data from the pickle file.

        This method loads model configuration and parameters from
        the provided model path.

        Raises:
            FileNotFoundError: If the model data file doesn't exist
            ValueError: If essential parameters are missing
        """
        data_file = os.path.join(self.model_path, 'net.pkl')

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Model data file not found: {data_file}")

        try:
            with open(data_file, 'rb') as f:
                self.model_data = pickle.load(f)

            # Extract key parameters
            self.class_names = self.model_data.get('classNames')
            self.image_size = self.model_data.get('sxy')
            self.model_type = self.model_data.get('model_type')
            self.batch_size = self.model_data.get('batch_size', 3)  # Default to 3 if not specified

            # Handle '+' in model_type
            if self.model_type and '+' in self.model_type:
                self.model_type = self.model_type.replace('+', '_plus')

            # Check for required parameters
            if None in [self.class_names, self.image_size, self.model_type]:
                raise ValueError("Missing required parameters in model data file")

            # Compute number of classes
            self.num_classes = len(self.class_names) if self.class_names else 0

        except Exception as e:
            raise ValueError(f"Failed to load model data: {e}")

    def _prepare_data(self):
        """
        Prepare training and validation datasets.

        This method finds training and validation images and masks,
        and creates TensorFlow datasets for them.

        Raises:
            ValueError: If no training or validation images are found
        """
        # Create paths for training and validation data
        train_path = os.path.join(self.model_path, 'training')
        val_path = os.path.join(self.model_path, 'validation')

        # Find training images and masks
        train_images = sorted(glob(os.path.join(train_path, 'im', "*.png")))
        train_masks = sorted(glob(os.path.join(train_path, 'label', "*.png")))

        # Find validation images and masks
        val_images = sorted(glob(os.path.join(val_path, 'im', "*.png")))
        val_masks = sorted(glob(os.path.join(val_path, 'label', "*.png")))

        # Check if we found any data
        if not train_images or not train_masks:
            raise ValueError("No training images or masks found")

        if not val_images or not val_masks:
            raise ValueError("No validation images or masks found")

        # Create data generator and datasets
        data_generator = DataGenerator(self.image_size, self.batch_size)
        self.train_dataset = data_generator.create_dataset(train_images, train_masks)
        self.val_dataset = data_generator.create_dataset(val_images, val_masks)

        # Log the datasets if logger is available
        if self.logger:
            self.logger.log_dataset_info(self.train_dataset, "Training")
            self.logger.log_dataset_info(self.val_dataset, "Validation")

    def _calculate_class_weights(self, mask_list: List[str]) -> np.ndarray:
        """
        Calculate class weights based on pixel frequency in labels.

        Args:
            mask_list: List of paths to mask images

        Returns:
            Array of class weights
        """
        # Initialize arrays for pixel counts and image counts
        class_pixels = np.zeros(self.num_classes)
        image_pixels = np.zeros(self.num_classes)
        epsilon = 1e-5

        total_pixels = self.image_size * self.image_size

        # Count pixels for each class in all masks
        for mask_path in mask_list:
            mask = tf.keras.preprocessing.image.load_img(
                mask_path, color_mode='grayscale')
            mask = tf.keras.preprocessing.image.img_to_array(mask)
            mask = mask.astype(int)

            # Count pixels for each class
            for i in range(self.num_classes):
                pixels_in_class = np.sum(mask == i)
                class_pixels[i] += pixels_in_class
                if pixels_in_class > 0:
                    image_pixels[i] += total_pixels

        # Calculate frequencies
        freq = class_pixels / (image_pixels + epsilon)

        # Handle infinite or NaN values
        freq[np.isinf(freq) | np.isnan(freq)] = epsilon

        # Calculate weights using median frequency balancing
        median_freq = np.median(freq)
        class_weights = median_freq / (freq + epsilon)

        return class_weights.astype(np.float32)

    def _create_loss_function(self):
        """
        Create a weighted loss function based on class frequencies.

        This method calculates class weights and creates a loss function
        that accounts for class imbalance.
        """
        # Find all training masks
        train_path = os.path.join(self.model_path, 'training')
        train_masks = sorted(glob(os.path.join(train_path, 'label', "*.png")))

        # Calculate class weights
        self.class_weights = self._calculate_class_weights(train_masks)

        # Create the loss function
        self.loss_function = WeightedSparseCategoricalCrossentropy(
            class_weights=self.class_weights,
            from_logits=True,
            reduction='sum_over_batch_size'
        )

    def _create_model(self) -> keras.Model:
        """
        Create a segmentation model.

        This method should be implemented by subclasses to create
        the appropriate model architecture.

        Returns:
            Keras model for semantic segmentation
        """
        raise NotImplementedError(
            "Subclasses must implement _create_model method"
        )

    def _create_callbacks(self, model: keras.Model) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks.

        Args:
            model: The model being trained

        Returns:
            List of Keras callbacks
        """
        # Create logger if not already created
        if not self.logger:
            log_dir = os.path.join(self.model_path, 'logs')
            model_name = self.model_data.get('nm', 'unknown_model')
            self.logger = Logger(log_dir=log_dir, model_name=model_name)
            self.logger.log_system_info()

        # Create custom batch accuracy callback
        best_model_path = os.path.join(
            self.model_path, f"best_model_{self.model_type}.keras"
        )

        batch_callback = BatchAccuracyCallback(
            model=model,
            val_data=self.val_dataset,
            loss_function=self.loss_function,
            logger=self.logger,
            num_validations=3,  # Validate 3 times per epoch
            early_stopping=True,
            reduce_lr_on_plateau=True,
            monitor='val_accuracy',
            es_patience=6,       # Stop after 6 epochs without improvement
            lr_patience=1,       # Reduce LR after 1 epoch without improvement
            lr_factor=0.75,      # Reduce LR to 75% of current value
            verbose=1,
            save_best_model=True,
            filepath=best_model_path
        )

        return [batch_callback]

    def _compile_model(self, model: keras.Model):
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            model: The model to compile
        """
        raise NotImplementedError(
            "Subclasses must implement _compile_model method"
        )

    def _train_model(self, model: keras.Model, callbacks: List[keras.callbacks.Callback]):
        """
        Train the model with the prepared datasets.

        Args:
            model: The model to train
            callbacks: List of callbacks to use during training

        Returns:
            Training history
        """
        raise NotImplementedError(
            "Subclasses must implement _train_model method"
        )

    def train(self) -> Dict[str, Any]:
        """
        Train a segmentation model and return training results.

        This is the main method that orchestrates the entire training process.

        Returns:
            Dictionary with training results and model information
        """
        # Start timing
        start_time = time.time()

        # Prepare data
        self._prepare_data()

        # Create loss function
        self._create_loss_function()

        # Create model
        model = self._create_model()

        # Create callbacks
        callbacks = self._create_callbacks(model)

        # Compile model
        self._compile_model(model)

        # Train model
        history = self._train_model(model, callbacks)

        # Save model and training history
        model_name = f"{self.model_type}.keras"
        model.save(os.path.join(self.model_path, model_name))

        # Update model data with training history
        self.model_data['history'] = history.history
        with open(os.path.join(self.model_path, 'net.pkl'), 'wb') as f:
            pickle.dump(self.model_data, f)

        # Calculate training time
        training_time = time.time() - start_time
        hours, rem = divmod(training_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        # Save summary if logger is available
        if self.logger:
            self.logger.save_debug_summary()

        return {
            'model': model,
            'history': history.history,
            'training_time': training_time
        }


class DeepLabV3PlusTrainer(SegmentationModelTrainer):
    """
    Trainer for DeepLabV3+ segmentation model.

    This class implements DeepLabV3+ specific training logic.
    """

    def _create_model(self) -> keras.Model:
        """
        Create a DeepLabV3+ model.

        Returns:
            DeepLabV3+ model instance
        """
        return model_call(
            name="DeepLabV3_plus",
            IMAGE_SIZE=self.image_size,
            NUM_CLASSES=self.num_classes
        )

    def _compile_model(self, model: keras.Model):
        """
        Compile the DeepLabV3+ model.

        Args:
            model: The model to compile
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss=self.loss_function,
            metrics=["accuracy"],
        )

    def _train_model(self, model: keras.Model, callbacks: List[keras.callbacks.Callback]):
        """
        Train the DeepLabV3+ model.

        Args:
            model: The model to train
            callbacks: List of callbacks to use during training

        Returns:
            Training history
        """
        print('Starting DeepLabV3+ model training...')

        history = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1,
            epochs=8  # Train for 8 epochs
        )

        return history


class UNetTrainer(SegmentationModelTrainer):
    """
    Trainer for UNet segmentation model.

    This class implements UNet specific training logic, including
    transfer learning with frozen and unfrozen layers.
    """

    def _create_model(self) -> keras.Model:
        """
        Create a UNet model.

        Returns:
            UNet model instance
        """
        return model_call(
            name="UNet",
            IMAGE_SIZE=self.image_size,
            NUM_CLASSES=self.num_classes
        )

    def _compile_model(self, model: keras.Model, learning_rate: float = 0.001):
        """
        Compile the UNet model.

        Args:
            model: The model to compile
            learning_rate: Learning rate for the optimizer
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.loss_function,
            metrics=["accuracy"]
        )

    def _train_model(self, model: keras.Model, callbacks: List[keras.callbacks.Callback]):
        """
        Train the UNet model using transfer learning.

        This method first trains with frozen encoder layers, then unfreezes
        them for fine-tuning.

        Args:
            model: The model to train
            callbacks: List of callbacks to use during training

        Returns:
            Training history from the final phase
        """
        print('Starting UNet model training...')

        # Initial training phase with frozen encoder
        initial_epochs = 5

        print('Phase 1: Training with frozen encoder layers...')
        history_initial = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=initial_epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Fine-tuning phase with unfrozen encoder
        print('Phase 2: Fine-tuning with unfrozen encoder layers...')
        model = unfreeze_model(model)

        # Recompile with lower learning rate
        self._compile_model(model, learning_rate=0.0001)

        # Continue training from where we left off
        history_fine_tuning = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=10,  # Train for additional epochs
            initial_epoch=initial_epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Save initial phase history for logging
        self.model_data['initial_history'] = history_initial.history

        return history_fine_tuning


def train_segmentation_model_cnns(pthDL: str, retrain_model: bool = False) -> None:
    """
    Train a segmentation model with the specified configuration.

    This function is the main entry point for model training. It loads model
    configuration, creates the appropriate trainer, and trains the model.

    Args:
        pthDL: Path to the directory containing model data
        retrain_model: Whether to retrain an existing model

    Returns:
        None
    """
    # Load model type from configuration
    try:
        with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
            data = pickle.load(f)
            model_type = data.get('model_type')
            model_name = data.get('nm', 'unknown_model')

        # Handle '+' in model_type
        if '+' in model_type:
            model_type = model_type.replace('+', '_plus')

        if model_type not in ["DeepLabV3_plus", "UNet"]:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types are 'DeepLabV3_plus' and 'UNet'"
            )
    except Exception as e:
        raise ValueError(f"Failed to load model configuration: {e}")

    # Check if model already exists and should not be retrained
    if os.path.isfile(os.path.join(pthDL, f'best_model_{model_type}.keras')) and not retrain_model:
        print(f'Model already trained with name {model_name}. Use retrain_model=True to retrain.')
        return

    # Create and use the appropriate trainer
    try:
        if model_type == "DeepLabV3_plus":
            trainer = DeepLabV3PlusTrainer(pthDL)
        else:  # model_type == "UNet"
            trainer = UNetTrainer(pthDL)

        # Train the model
        trainer.train()

        print(f"Successfully trained {model_type} model.")

    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        print(traceback.format_exc())