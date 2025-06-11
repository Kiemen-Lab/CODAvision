"""
Training Logger for Machine Learning Models

This module provides a comprehensive logging system for tracking, monitoring, and
debugging machine learning model training. It includes functionality for monitoring
system resources, training metrics, and model behavior.

Features:
- Multi-level logging (DEBUG, INFO) with separate log files
- Real-time system resource monitoring (CPU, RAM, GPU)
- Batch-level metrics tracking with learning rate monitoring
- Detailed validation metrics analysis with class distribution tracking
- Dataset analysis capabilities
- Automatic debug summary generation
- GPU monitoring with fallback options
"""

import time
import os
import numpy as np
import tensorflow as tf
import logging
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any


class Logger:
    """
    Enhanced training logger with comprehensive monitoring and debugging capabilities.

    This logger offers detailed tracking of training metrics, system resources, and
    model behavior during training, with support for both CPU and GPU environments.

    Attributes:
        log_dir (str): Directory where log files will be stored
        model_name (str): Name of the model being trained
        debug_dir (str): Directory for debug-level logs
        logger (logging.Logger): Underlying logger instance
        validation_runs (int): Counter for validation runs
        memory_logs (List[Dict]): History of memory usage
        validation_history (List[Dict]): History of validation metrics
        batch_history (List[Dict]): History of batch metrics
        gradient_history (List[Dict]): History of gradient metrics
    """

    def __init__(self, log_dir: str, model_name: str):
        """
        Initialize the logger with specified log directory and model name.

        Args:
            log_dir: Path to directory where logs will be stored
            model_name: Name of the model being trained
        """
        self.log_dir = log_dir
        self.model_name = model_name

        # Create log directories
        os.makedirs(log_dir, exist_ok=True)
        self.debug_dir = os.path.join(log_dir, 'debug')
        os.makedirs(self.debug_dir, exist_ok=True)

        # Setup logging configuration
        self.logger = self._setup_logger()

        # Initialize tracking variables
        self.validation_runs = 0
        self.memory_logs = []
        self.validation_history = []
        self.batch_history = []
        self.gradient_history = []

        # Record start time
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration with separate handlers for different log levels.

        Returns:
            Configured logger instance
        """
        logger_name = f'training_logger_{self.model_name}'
        logger = logging.getLogger(logger_name)

        # Clear any existing handlers to avoid duplication
        logger.handlers.clear()

        # Set logger level to DEBUG so it processes all messages
        logger.setLevel(logging.DEBUG)

        # Prevent propagation to parent loggers (including root logger)
        logger.propagate = False

        # Generate timestamp for log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Main log file handler (INFO and above)
        main_log_file = os.path.join(self.log_dir, f'training_log_{timestamp}.log')
        main_handler = logging.FileHandler(main_log_file)
        main_handler.setLevel(logging.INFO)

        # Debug log file handler (DEBUG and above)
        debug_log_file = os.path.join(self.debug_dir, f'debug_log_{timestamp}.log')
        debug_handler = logging.FileHandler(debug_log_file)
        debug_handler.setLevel(logging.DEBUG)

        # Console handler (INFO and above only)
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

        # Apply formatters
        main_handler.setFormatter(main_formatter)
        console_handler.setFormatter(main_formatter)
        debug_handler.setFormatter(debug_formatter)

        # Add handlers
        logger.addHandler(main_handler)
        logger.addHandler(console_handler)
        logger.addHandler(debug_handler)

        return logger

    def get_gpu_info(self) -> Union[List[Dict[str, Any]], str]:
        """
        Get GPU information using multiple methods with fallbacks.

        Returns:
            List of GPU information dictionaries or error message string
        """
        try:
            # First try using GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = []
                for gpu in gpus:
                    info = {
                        'device': f"GPU {gpu.id}",
                        'name': gpu.name
                    }

                    # Only add memory metrics if available and valid
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

            # Fallback to TensorFlow GPU information
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

    def log_system_info(self) -> None:
        """
        Log detailed system resource information with robust GPU monitoring.
        """
        try:
            # Log CPU and memory information
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            # Log process-specific memory usage
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Log general system information
            self.logger.debug("\nSystem Resources:")
            self.logger.debug(f"CPU Usage: {cpu_percent}%")
            self.logger.debug(
                f"System Memory: {memory.used / 1024 / 1024 / 1024:.1f}GB / {memory.total / 1024 / 1024 / 1024:.1f}GB ({memory.percent}%)")
            self.logger.debug(f"Process Memory: {process_memory:.1f}MB")

            # Record memory information for trend analysis
            memory_info = {
                'timestamp': time.time(),
                'total_memory': memory.total / (1024 * 1024 * 1024),
                'used_memory': memory.used / (1024 * 1024 * 1024),
                'process_memory': process_memory,
                'cpu_percent': cpu_percent
            }
            self.memory_logs.append(memory_info)

            # Log GPU information if available
            self.logger.debug("\nGPU Information:")
            gpu_info = self.get_gpu_info()

            if isinstance(gpu_info, list):
                for gpu in gpu_info:
                    self.logger.debug(f"\nDevice: {gpu['device']}")
                    if 'name' in gpu:
                        self.logger.debug(f"Name: {gpu['name']}")
                    if 'error' in gpu:
                        self.logger.debug(f"Error: {gpu['error']}")
                    else:
                        if 'memory_used' in gpu and 'memory_total' in gpu:
                            self.logger.debug(f"Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB")
                        if 'utilization' in gpu:
                            self.logger.debug(f"Utilization: {gpu['utilization']}%")
                        if 'current_memory' in gpu:
                            self.logger.debug(f"Current Memory: {gpu['current_memory']}MB")
                        if 'peak_memory' in gpu:
                            self.logger.debug(f"Peak Memory: {gpu['peak_memory']}MB")
            else:
                self.logger.debug(gpu_info)

        except Exception as e:
            self.logger.error(f"Error in system info logging: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def log_batch_metrics(self, batch_number: int, metrics: Dict[str, float], model: Optional[tf.keras.Model] = None) -> None:
        """
        Log batch-level metrics with additional debugging information.

        Args:
            batch_number: Current batch number
            metrics: Dictionary of metric names and values
            model: Optional model for extracting learning rate
        """
        current_time = time.time()
        time_since_last = current_time - self.last_log_time

        batch_info = {
            'batch_number': batch_number,
            'timestamp': current_time,
            'time_since_last': time_since_last,
            'metrics': metrics
        }

        # Try to get learning rate if model is provided
        if model is not None:
            try:
                lr = float(model.optimizer.learning_rate.numpy())
                batch_info['learning_rate'] = lr
                metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                self.logger.debug(f"Batch {batch_number}: {metrics_str} - lr: {lr:.6f}")
            except Exception as e:
                self.logger.warning(f"Could not get learning rate: {str(e)}")
                metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                self.logger.debug(f"Batch {batch_number}: {metrics_str}")
        else:
            metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.logger.debug(f"Batch {batch_number}: {metrics_str}")

        self.batch_history.append(batch_info)
        self.last_log_time = current_time

        # Periodically log system information
        if batch_number % 100 == 0:
            self.log_system_info()

    def log_dataset_info(self, dataset: tf.data.Dataset, dataset_name: str) -> None:
        """
        Log information about a dataset including shapes and class distribution.

        Args:
            dataset: TensorFlow dataset to analyze
            dataset_name: Name of the dataset (e.g., 'training', 'validation')
        """
        try:
            # Examine first batch to get dataset structure
            for batch in dataset.take(1):
                if isinstance(batch, tuple) and len(batch) == 2:
                    images, masks = batch
                else:
                    self.logger.error(f"Unexpected batch format in {dataset_name} dataset")
                    return

                self.logger.debug(f"\n{dataset_name} Dataset Information:")
                self.logger.debug(f"Image batch shape: {images.shape}")
                self.logger.debug(f"Mask batch shape: {masks.shape}")

                # Calculate class distribution
                try:
                    # Flatten masks and count unique values
                    mask_flat = tf.cast(tf.reshape(masks, [-1]), tf.int32)
                    unique_classes, _, counts = tf.unique_with_counts(mask_flat)

                    # Calculate percentages
                    total_pixels = tf.reduce_sum(counts)
                    percentages = (counts / total_pixels) * 100

                    # Format for logging
                    class_distribution = {}
                    for cls, count, percentage in zip(unique_classes.numpy(),
                                                      counts.numpy(),
                                                      percentages.numpy()):
                        class_distribution[f"Class {int(cls)}"] = {
                            'count': int(count),
                            'percentage': f"{percentage:.2f}%"
                        }

                    self.logger.debug("Class distribution in sample batch:")
                    for cls, stats in class_distribution.items():
                        self.logger.debug(f"  {cls}: {stats['count']} pixels ({stats['percentage']})")

                except Exception as e:
                    self.logger.error(f"Error calculating class distribution: {str(e)}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")

            # Count total number of batches
            try:
                total_batches = sum(1 for _ in dataset)
                self.logger.debug(f"Total number of batches: {total_batches}")
            except Exception as e:
                self.logger.error(f"Error counting batches: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error analyzing {dataset_name} dataset: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def log_validation_metrics(self, val_logits: tf.Tensor, y_val: tf.Tensor, loss: float, accuracy: float) -> None:
        """
        Log validation metrics with detailed dataset analysis.

        Args:
            val_logits: Model output logits
            y_val: Ground truth labels
            loss: Validation loss value
            accuracy: Validation accuracy value
        """
        self.validation_runs += 1

        validation_info = {
            'run_number': self.validation_runs,
            'timestamp': time.time(),
            'loss': loss,
            'accuracy': accuracy
        }

        # Log dataset shapes
        self.logger.debug(f"\nValidation Run {self.validation_runs} Dataset Shapes:")
        self.logger.debug(f"Logits shape: {val_logits.shape}")
        self.logger.debug(f"Labels shape: {y_val.shape}")

        try:
            # Analyze ground truth class distribution
            y_val_flat = tf.cast(tf.reshape(y_val, [-1]), tf.int32)
            unique_classes, _, counts = tf.unique_with_counts(y_val_flat)
            total_pixels = tf.reduce_sum(counts)
            class_distribution = {
                f"Class {int(cls)}": f"{(count / total_pixels * 100):.2f}%"
                for cls, count in zip(unique_classes.numpy(), counts.numpy())
            }
            self.logger.debug("Validation class distribution:")
            for cls, percentage in class_distribution.items():
                self.logger.debug(f"  {cls}: {percentage}")

            # Analyze model prediction distribution
            predictions = tf.argmax(val_logits, axis=-1)
            pred_unique, _, pred_counts = tf.unique_with_counts(tf.reshape(predictions, [-1]))
            pred_distribution = {
                f"Class {int(cls)}": f"{(count / tf.reduce_sum(pred_counts) * 100):.2f}%"
                for cls, count in zip(pred_unique.numpy(), pred_counts.numpy())
            }
            self.logger.debug("Prediction distribution:")
            for cls, percentage in pred_distribution.items():
                self.logger.debug(f"  {cls}: {percentage}")

        except Exception as e:
            self.logger.error(f"Error analyzing class distributions: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        # Check for NaN loss
        if np.isnan(loss):
            self.logger.error("Validation loss is NaN! Detailed analysis:")
            self.logger.error(
                f"Number of NaN values in logits: {tf.reduce_sum(tf.cast(tf.math.is_nan(val_logits), tf.int32))}")
            self.logger.error(
                f"Number of infinite values in logits: {tf.reduce_sum(tf.cast(tf.math.is_inf(val_logits), tf.int32))}")

        # Check for unchanged accuracy (possible training issue)
        if len(self.validation_history) >= 2:
            prev_acc = self.validation_history[-1]['accuracy']
            if abs(prev_acc - accuracy) < 1e-10:  # Essentially unchanged
                self.logger.warning(
                    f"Validation accuracy hasn't changed: {accuracy}. "
                    "This might indicate a problem in the validation pipeline."
                )

        # Store validation info
        self.validation_history.append(validation_info)

        self.logger.debug(f"Validation Metrics - Run {self.validation_runs}:")
        self.logger.debug(f"Loss: {loss:.4f}")
        self.logger.debug(f"Accuracy: {accuracy:.4f}")

    def log_error(self, error_msg: str, include_trace: bool = True) -> None:
        """
        Log error messages with optional stack trace.

        Args:
            error_msg: Error message to log
            include_trace: Whether to include stack trace
        """
        self.logger.error(f"Error occurred: {error_msg}")
        if include_trace:
            import traceback
            self.logger.error(f"Stack trace:\n{traceback.format_exc()}")

    def log_warning(self, warning_msg: str) -> None:
        """
        Log warning messages.

        Args:
            warning_msg: Warning message to log
        """
        self.logger.warning(warning_msg)

    def save_debug_summary(self) -> None:
        """
        Save a summary of debugging information to a file.

        This creates a human-readable report of training statistics.
        """
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