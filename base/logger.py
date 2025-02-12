"""
A training logger for machine learning models that provides comprehensive monitoring and debugging capabilities.

This logger offers detailed tracking of training metrics, system resources, and model behavior during training,
with support for both CPU and GPU environments. It implements hierarchical logging levels, memory tracking,
and validation metrics analysis.

Features:
- Multi-level logging (DEBUG, INFO) with separate log files for different logging levels
- Real-time system resource monitoring (CPU, RAM, GPU)
- Batch-level metrics tracking with learning rate monitoring
- Detailed validation metrics analysis with class distribution tracking
- Dataset analysis capabilities including shape and class distribution logging
- Automatic debug summary generation
- GPU monitoring with fallback options using both GPUtil and TensorFlow

Usage:
    logger = Logger(log_dir='path/to/logs', model_name='my_model')

    # Log batch metrics
    logger.log_batch_metrics(batch_number=1, metrics={'loss': 0.5, 'accuracy': 0.95}, model=model)

    # Log validation results
    logger.log_validation_metrics(val_logits, y_val, loss=0.4, accuracy=0.96)

    # Analyze dataset
    logger.log_dataset_info(train_dataset, 'training')

    # Save debug summary
    logger.save_debug_summary()
"""

import time
import os
import numpy as np
import tensorflow as tf
import logging
import psutil
import GPUtil
from datetime import datetime


class Logger:
    """
    Enhanced training logger with debug capabilities for BatchAccCall callback.
    """

    def __init__(self, log_dir, model_name):
        self.log_dir = log_dir
        self.model_name = model_name

        # Create logging directories
        os.makedirs(log_dir, exist_ok=True)
        self.debug_dir = os.path.join(log_dir, 'debug')
        os.makedirs(self.debug_dir, exist_ok=True)

        # Initialize logger
        self.logger = self._setup_logger()

        # Initialize debug counters and trackers
        self.validation_runs = 0
        self.memory_logs = []
        self.validation_history = []
        self.batch_history = []
        self.gradient_history = []

        # Start time tracking
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def _setup_logger(self):
        """Set up logging configuration with separate handlers for different log levels."""
        logger = logging.getLogger(f'training_logger_{self.model_name}')
        logger.setLevel(logging.DEBUG)

        # Create handlers
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Main log file
        main_log_file = os.path.join(self.log_dir, f'training_log_{timestamp}.log')
        main_handler = logging.FileHandler(main_log_file)
        main_handler.setLevel(logging.INFO)

        # Debug log file
        debug_log_file = os.path.join(self.debug_dir, f'debug_log_{timestamp}.log')
        debug_handler = logging.FileHandler(debug_log_file)
        debug_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatters
        main_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        debug_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        main_handler.setFormatter(main_formatter)
        console_handler.setFormatter(main_formatter)
        debug_handler.setFormatter(debug_formatter)

        logger.addHandler(main_handler)
        logger.addHandler(console_handler)
        logger.addHandler(debug_handler)

        return logger

    def get_gpu_info(self):
        """
        Get GPU information using multiple methods.
        Returns a list of GPU information dictionaries or error message.
        """
        try:
            # First try GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = []
                for gpu in gpus:
                    info = {
                        'device': f"GPU {gpu.id}",
                        'name': gpu.name
                    }

                    # Validate memory values
                    if (isinstance(gpu.memoryUsed, (int, float)) and
                            isinstance(gpu.memoryTotal, (int, float)) and
                            gpu.memoryUsed > 0 and gpu.memoryTotal > 0):
                        info.update({
                            'memory_used': f"{gpu.memoryUsed:.1f}",
                            'memory_total': f"{gpu.memoryTotal:.1f}",
                            'utilization': f"{gpu.load * 100:.1f}"
                        })
                    gpu_info.append(info)
                return gpu_info

            # If GPUtil failed, try tensorflow
            tf_gpus = tf.config.list_physical_devices('GPU')
            if tf_gpus:
                gpu_info = []
                for gpu in tf_gpus:
                    try:
                        info = {'device': gpu.name}
                        memory_info = tf.config.experimental.get_memory_info(gpu.name)
                        info.update({
                            'current_memory': f"{memory_info['current'] / (1024 * 1024):.1f}",
                            'peak_memory': f"{memory_info['peak'] / (1024 * 1024):.1f}"
                        })
                        gpu_info.append(info)
                    except Exception as e:
                        gpu_info.append({
                            'device': gpu.name,
                            'error': str(e)
                        })
                return gpu_info

            return "No GPU devices found"

        except Exception as e:
            return f"Error getting GPU info: {str(e)}"

    def log_system_info(self):
        """Log detailed system resource information with robust GPU monitoring."""
        try:
            # System memory and CPU info
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            # Process-specific memory
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Log basic system info
            self.logger.info("\nSystem Resources:")
            self.logger.info(f"CPU Usage: {cpu_percent}%")
            self.logger.info(
                f"System Memory: {memory.used / 1024 / 1024 / 1024:.1f}GB / {memory.total / 1024 / 1024 / 1024:.1f}GB ({memory.percent}%)")
            self.logger.info(f"Process Memory: {process_memory:.1f}MB")

            # Track memory info
            memory_info = {
                'timestamp': time.time(),
                'total_memory': memory.total / (1024 * 1024 * 1024),
                'used_memory': memory.used / (1024 * 1024 * 1024),
                'process_memory': process_memory,
                'cpu_percent': cpu_percent
            }
            self.memory_logs.append(memory_info)

            # GPU Information
            self.logger.info("\nGPU Information:")
            gpu_info = self.get_gpu_info()

            if isinstance(gpu_info, list):
                for gpu in gpu_info:
                    self.logger.info(f"\nDevice: {gpu['device']}")
                    if 'name' in gpu:
                        self.logger.info(f"Name: {gpu['name']}")
                    if 'error' in gpu:
                        self.logger.warning(f"Error: {gpu['error']}")
                    else:
                        if 'memory_used' in gpu and 'memory_total' in gpu:
                            self.logger.info(f"Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB")
                        if 'utilization' in gpu:
                            self.logger.info(f"Utilization: {gpu['utilization']}%")
                        if 'current_memory' in gpu:
                            self.logger.info(f"Current Memory: {gpu['current_memory']}MB")
                        if 'peak_memory' in gpu:
                            self.logger.info(f"Peak Memory: {gpu['peak_memory']}MB")
            else:
                self.logger.info(gpu_info)

        except Exception as e:
            self.logger.error(f"Error in system info logging: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def log_batch_metrics(self, batch_number, metrics, model=None):
        """Log batch-level metrics with additional debugging information."""
        current_time = time.time()
        time_since_last = current_time - self.last_log_time

        batch_info = {
            'batch_number': batch_number,
            'timestamp': current_time,
            'time_since_last': time_since_last,
            'metrics': metrics
        }

        # Add learning rate if model is provided
        if model is not None:
            try:
                lr = float(model.optimizer.learning_rate.numpy())
                batch_info['learning_rate'] = lr
                metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                self.logger.info(f"Batch {batch_number}: {metrics_str} - lr: {lr:.6f}")
            except Exception as e:
                self.logger.warning(f"Could not get learning rate: {str(e)}")
                metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                self.logger.info(f"Batch {batch_number}: {metrics_str}")
        else:
            metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.logger.info(f"Batch {batch_number}: {metrics_str}")

        self.batch_history.append(batch_info)
        self.last_log_time = current_time

        # Log system resources periodically
        if batch_number % 100 == 0:
            self.log_system_info()

    def log_dataset_info(self, dataset, dataset_name):
        """
        Log information about a dataset including shapes and class distribution.

        Args:
            dataset: TensorFlow dataset
            dataset_name: Name of the dataset (e.g., 'training', 'validation')
        """
        try:
            # Get shapes from the first batch
            for batch in dataset.take(1):
                if isinstance(batch, tuple) and len(batch) == 2:
                    images, masks = batch
                else:
                    self.logger.error(f"Unexpected batch format in {dataset_name} dataset")
                    return

                self.logger.info(f"\n{dataset_name} Dataset Information:")
                self.logger.info(f"Image batch shape: {images.shape}")
                self.logger.info(f"Mask batch shape: {masks.shape}")

                # Calculate class distribution in the batch
                try:
                    # Ensure mask is the right shape and type
                    mask_flat = tf.cast(tf.reshape(masks, [-1]), tf.int32)
                    unique_classes, _, counts = tf.unique_with_counts(mask_flat)

                    # Calculate percentages
                    total_pixels = tf.reduce_sum(counts)
                    percentages = (counts / total_pixels) * 100

                    # Create class distribution dictionary
                    class_distribution = {}
                    for cls, count, percentage in zip(unique_classes.numpy(),
                                                      counts.numpy(),
                                                      percentages.numpy()):
                        class_distribution[f"Class {int(cls)}"] = {
                            'count': int(count),
                            'percentage': f"{percentage:.2f}%"
                        }

                    self.logger.info("Class distribution in sample batch:")
                    for cls, stats in class_distribution.items():
                        self.logger.info(f"  {cls}: {stats['count']} pixels ({stats['percentage']})")

                except Exception as e:
                    self.logger.error(f"Error calculating class distribution: {str(e)}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")

            # Count total batches
            try:
                total_batches = sum(1 for _ in dataset)
                self.logger.info(f"Total number of batches: {total_batches}")
            except Exception as e:
                self.logger.error(f"Error counting batches: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error analyzing {dataset_name} dataset: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def log_validation_metrics(self, val_logits, y_val, loss, accuracy):
        """Enhanced validation metrics logging with detailed dataset analysis."""
        self.validation_runs += 1

        validation_info = {
            'run_number': self.validation_runs,
            'timestamp': time.time(),
            'loss': loss,
            'accuracy': accuracy
        }

        # Log shapes of validation data
        self.logger.info(f"\nValidation Run {self.validation_runs} Dataset Shapes:")
        self.logger.info(f"Logits shape: {val_logits.shape}")
        self.logger.info(f"Labels shape: {y_val.shape}")

        try:
            # Analyze class distribution in validation set
            y_val_flat = tf.cast(tf.reshape(y_val, [-1]), tf.int32)
            unique_classes, _, counts = tf.unique_with_counts(y_val_flat)
            total_pixels = tf.reduce_sum(counts)
            class_distribution = {
                f"Class {int(cls)}": f"{(count / total_pixels * 100):.2f}%"
                for cls, count in zip(unique_classes.numpy(), counts.numpy())
            }
            self.logger.info("Validation class distribution:")
            for cls, percentage in class_distribution.items():
                self.logger.info(f"  {cls}: {percentage}")

            # Check for prediction bias
            predictions = tf.argmax(val_logits, axis=-1)
            pred_unique, _, pred_counts = tf.unique_with_counts(tf.reshape(predictions, [-1]))
            pred_distribution = {
                f"Class {int(cls)}": f"{(count / tf.reduce_sum(pred_counts) * 100):.2f}%"
                for cls, count in zip(pred_unique.numpy(), pred_counts.numpy())
            }
            self.logger.info("Prediction distribution:")
            for cls, percentage in pred_distribution.items():
                self.logger.info(f"  {cls}: {percentage}")

        except Exception as e:
            self.logger.error(f"Error analyzing class distributions: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        # Check for NaN values
        if np.isnan(loss):
            self.logger.error("Validation loss is NaN! Detailed analysis:")
            self.logger.error(
                f"Number of NaN values in logits: {tf.reduce_sum(tf.cast(tf.math.is_nan(val_logits), tf.int32))}")
            self.logger.error(
                f"Number of infinite values in logits: {tf.reduce_sum(tf.cast(tf.math.is_inf(val_logits), tf.int32))}")

        # Check for stuck accuracy
        if len(self.validation_history) >= 2:
            prev_acc = self.validation_history[-1]['accuracy']
            if abs(prev_acc - accuracy) < 1e-10:  # Exact match
                self.logger.warning(
                    f"Validation accuracy hasn't changed: {accuracy}. "
                    "This might indicate a problem in the validation pipeline."
                )

        # Save validation metrics
        self.validation_history.append(validation_info)

        self.logger.info(f"Validation Metrics - Run {self.validation_runs}:")
        self.logger.info(f"Loss: {loss:.4f}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")

    def log_error(self, error_msg, include_trace=True):
        """Log error messages with optional stack trace."""
        self.logger.error(f"Error occurred: {error_msg}")
        if include_trace:
            import traceback
            self.logger.error(f"Stack trace:\n{traceback.format_exc()}")

    def log_warning(self, warning_msg):
        """Log warning messages."""
        self.logger.warning(warning_msg)

    def save_debug_summary(self):
        """Save a summary of debugging information to a file."""
        summary_file = os.path.join(self.debug_dir, 'debug_summary.txt')

        with open(summary_file, 'w') as f:
            f.write("Training Debug Summary\n")
            f.write("=====================\n\n")

            # Memory usage summary
            f.write("Memory Usage Summary:\n")
            if self.memory_logs:
                peak_memory = max(log['process_memory'] for log in self.memory_logs)
                avg_memory = np.mean([log['process_memory'] for log in self.memory_logs])
                f.write(f"Peak Memory Usage: {peak_memory:.2f} MB\n")
                f.write(f"Average Memory Usage: {avg_memory:.2f} MB\n\n")

            # Validation statistics
            f.write("Validation Statistics:\n")
            if self.validation_history:
                accuracies = [v['accuracy'] for v in self.validation_history]
                losses = [v['loss'] for v in self.validation_history]
                f.write(f"Number of validation runs: {self.validation_runs}\n")
                f.write(f"Mean validation accuracy: {np.mean(accuracies):.4f}\n")
                f.write(f"Std validation accuracy: {np.std(accuracies):.4f}\n")
                f.write(f"Mean validation loss: {np.mean(losses):.4f}\n")
                f.write(f"Std validation loss: {np.std(losses):.4f}\n\n")

            # Training progress
            f.write("Training Progress:\n")
            if self.batch_history:
                total_batches = len(self.batch_history)
                total_time = self.batch_history[-1]['timestamp'] - self.batch_history[0]['timestamp']
                avg_time_per_batch = total_time / total_batches
                f.write(f"Total batches processed: {total_batches}\n")
                f.write(f"Average time per batch: {avg_time_per_batch:.3f}s\n")