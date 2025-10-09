"""
Hyperparameter Grid Search for DeepLabV3+ Model

This script performs a systematic grid search over hyperparameters for training
a DeepLabV3+ segmentation model. It builds upon the non-gui_workflow.py script
but explores multiple hyperparameter combinations to find optimal settings.

Basic Usage:
    # Run full hyperparameter search with default settings
    python hyperparameter_search.py

    # Run small test grid (8 combinations) for quick validation
    python hyperparameter_search.py --small

    # Resume from checkpoint after interruption
    python hyperparameter_search.py --resume

    # Specify custom output directory
    python hyperparameter_search.py --output-dir ./my_search_results

Advanced Usage Examples:

1. Custom Hyperparameter Grid (modify in main()):
    ```python
    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3],  # Learning rates to test
        'batch_size': [2, 4, 8],              # Batch sizes for GPU memory
        'epochs': [10, 20, 30],                # Training epochs
        'es_patience': [5, 10],                # Early stopping patience
        'lr_factor': [0.5, 0.75],              # LR reduction on plateau
        'optimizer': ['adam', 'sgd'],         # Different optimizers
        'augmentation': [True, False]         # Data augmentation on/off
    }
    ```

2. Programmatic Usage:
    ```python
    from hyperparameter_search import HyperparameterSearcher

    # Configure base settings
    base_config = {
        'pth': '/path/to/data',
        'pthim': '/path/to/images',
        'model_type': 'DeepLabV3_plus',
        'classNames': ['class1', 'class2', 'class3'],
        # ... other configuration
    }

    # Define search space
    param_grid = {
        'learning_rate': [0.0001, 0.001],
        'batch_size': [2, 4],
        'epochs': [5, 10]
    }

    # Create and run searcher
    searcher = HyperparameterSearcher(base_config, param_grid, 'results/')
    searcher.run_search(resume=False)
    ```

3. Analyzing Results After Completion:
    ```python
    import pandas as pd
    import json

    # Load summary CSV
    results = pd.read_csv('hyperparameter_search_results/results_summary.csv')

    # Find best configuration
    best_idx = results['dice_coefficient'].idxmax()
    best_config = results.iloc[best_idx]
    print(f"Best dice coefficient: {best_config['dice_coefficient']:.4f}")
    print(f"Best parameters: LR={best_config['learning_rate']}, "
          f"Batch={best_config['batch_size']}")

    # Load detailed metrics for best model
    with open(f'hyperparameter_search_results/experiment_{best_idx:03d}/metrics.json') as f:
        best_metrics = json.load(f)
    ```

4. Running with Different Data Configurations:
    ```python
    # Modify base_config for different resolutions
    base_config['resolution'] = '20x'  # Change from 10x to 20x
    base_config['pthim'] = os.path.join(base_config['pth'], '20x')
    base_config['sxy'] = 512  # Smaller tiles for higher resolution

    # For different tissue types
    base_config['classNames'] = ['tumor', 'stroma', 'necrosis', 'normal']
    base_config['cmap'] = np.array([[255,0,0], [0,255,0], [0,0,255], [128,128,128]])
    ```

5. Parallel Execution (using GNU parallel or similar):
    ```bash
    # Split search into multiple jobs
    python hyperparameter_search.py --subset 0 --total-subsets 4 &  # First quarter
    python hyperparameter_search.py --subset 1 --total-subsets 4 &  # Second quarter
    python hyperparameter_search.py --subset 2 --total-subsets 4 &  # Third quarter
    python hyperparameter_search.py --subset 3 --total-subsets 4 &  # Fourth quarter
    ```

6. Integration with Existing Workflow:
    ```python
    # Use best model from search in production
    best_config_path = 'hyperparameter_search_results/best_config.json'
    with open(best_config_path) as f:
        best_params = json.load(f)

    # Apply best parameters to production training
    trainer = DeepLabV3PlusTrainer(
        model_path='production_model',
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        epochs=best_params['epochs']
    )
    ```

Output Structure:
    hyperparameter_search_results/
    ├── experiment_XXX/          # Individual experiment folders
    │   ├── config.json         # Hyperparameter configuration
    │   └── metrics.json        # Performance metrics
    ├── best_model/             # Best performing model files
    ├── results_summary.csv     # All results in tabular format
    ├── summary_report.txt      # Human-readable summary
    ├── parallel_coordinates_plot.png  # Interactive hyperparameter visualization
    └── traditional_plots.png   # Performance plots

Notes:
    - Default grid: 576 combinations (4×4×4×3×3)
    - Small grid: 8 combinations (2×2×2×1×1)
    - Each experiment saves ~5-10KB (metrics only)
    - Best model saves full ~50-100MB model files
    - Checkpoint allows resuming after interruption
    - Results CSV enables post-hoc analysis
"""

import os
import sys
import numpy as np
import logging
import argparse
import json
import shutil
import time
import platform
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import traceback

# Add parent directory to path to import base modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.models.utils import create_initial_model_metadata, save_model_metadata
from base.data.annotation import load_annotation_data
from base.data.tiles import create_training_tiles
from base.models.training import DeepLabV3PlusTrainer
from base.evaluation.testing import test_segmentation_model
from base.config import ModelDefaults
from hyperparameter_utils import (
    HyperparameterGrid, ExperimentTracker, 
    plot_hyperparameter_results, plot_parallel_coordinates, create_summary_report,
    load_checkpoint, save_checkpoint
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'hyperparameter_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class HyperparameterSearcher:
    """
    Class to orchestrate hyperparameter grid search for segmentation models.
    """

    def __init__(self, base_config: Dict[str, Any], param_grid: Dict[str, list],
                 output_dir: str = 'hyperparameter_search_results'):
        """
        Initialize the hyperparameter searcher.

        Args:
            base_config: Base configuration (paths, data settings, etc.)
            param_grid: Dictionary of hyperparameters to search
            output_dir: Directory to store results
        """
        self.base_config = base_config
        self.param_grid = HyperparameterGrid(param_grid)
        self.output_dir = output_dir
        self.tracker = ExperimentTracker(output_dir)
        self.checkpoint_file = os.path.join(output_dir, 'checkpoint.pkl')
        self.best_metric_value = -float('inf')
        self.best_experiment_id = None

        logger.info(f"Initialized hyperparameter search with {len(self.param_grid)} combinations")
        logger.info(f"Parameters to search: {list(param_grid.keys())}")
        
    def prepare_data_once(self) -> Tuple[Any, Any, Any]:
        """
        Prepare data once for all experiments.
        
        Returns:
            Tuple of (ctlist0, numann0, create_new_tiles)
        """
        logger.info("Preparing data for all experiments...")
        
        # Create a temporary directory for initial data preparation
        temp_pthDL = os.path.join(self.output_dir, 'temp_data_prep')
        os.makedirs(temp_pthDL, exist_ok=True)
        
        # Filter base_config to only include parameters accepted by create_initial_model_metadata
        metadata_params = {
            'pthim': self.base_config['pthim'],
            'WS': self.base_config['WS'],
            'nm': self.base_config['nm'],
            'umpix': self.base_config['umpix'],
            'cmap': self.base_config['cmap'],
            'sxy': self.base_config['sxy'],
            'classNames': self.base_config['classNames'],
            'ntrain': self.base_config['ntrain'],
            'nvalidate': self.base_config['nvalidate'],
            'model_type': self.base_config.get('model_type', 'DeepLabV3_plus'),
            'pthtest': self.base_config.get('pthtest', None)
        }
        
        # Save initial metadata for data preparation
        create_initial_model_metadata(
            pthDL=temp_pthDL,
            **metadata_params
        )
        
        # Load annotation data
        ctlist0, numann0, create_new_tiles = load_annotation_data(
            temp_pthDL, 
            self.base_config['pth'],
            self.base_config['pthim'],
            self.base_config.get('classCheck', [])
        )
        
        # Create training tiles once
        create_training_tiles(temp_pthDL, numann0, ctlist0, create_new_tiles)
        
        # Store the prepared data path
        self.prepared_data_path = temp_pthDL
        
        return ctlist0, numann0, create_new_tiles
    
    def _safe_rmtree(self, path: str, max_retries: int = 5) -> None:
        """
        Safely remove a directory tree with retry logic for Windows file locking.

        Args:
            path: Path to directory to remove
            max_retries: Maximum number of retry attempts
        """
        if not os.path.exists(path):
            return

        for attempt in range(max_retries):
            try:
                shutil.rmtree(path)
                return
            except PermissionError as e:
                if attempt < max_retries - 1:
                    # On Windows, wait a bit for file system to release locks
                    logger.warning(f"Attempt {attempt + 1} failed to remove {path}: {str(e)}")
                    logger.warning("Waiting for file system to release locks...")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    # Last attempt failed, log error but continue
                    logger.error(f"Failed to remove directory {path} after {max_retries} attempts")
                    logger.error(f"Error: {str(e)}")
                    logger.warning("Continuing despite cleanup failure")

    def run_single_experiment(self, experiment_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single hyperparameter experiment.

        Args:
            experiment_id: Unique identifier for the experiment
            params: Hyperparameter configuration

        Returns:
            Dictionary of metrics from the experiment
        """
        logger.info(f"Starting experiment {experiment_id}")
        logger.info(f"Parameters: {params}")

        # Create minimal experiment directory for config and metrics only
        exp_dir = os.path.join(self.output_dir, f'experiment_{experiment_id}')
        os.makedirs(exp_dir, exist_ok=True)

        # Use a temporary directory for model training
        temp_model_dir = os.path.join(self.output_dir, 'temp_training')
        if os.path.exists(temp_model_dir):
            self._safe_rmtree(temp_model_dir)

        # Copy prepared data to temporary training directory
        pthDL = temp_model_dir
        shutil.copytree(self.prepared_data_path, pthDL, dirs_exist_ok=True)
        
        # Update metadata with hyperparameters
        save_model_metadata(pthDL, {
            'hyperparameters': params,
            'experiment_id': experiment_id,
            'batch_size': params.get('batch_size', 3),
            'learning_rate': params.get('learning_rate', 0.0005),
            'epochs': params.get('epochs', 8),
            'es_patience': params.get('es_patience', 6),
            'lr_factor': params.get('lr_factor', 0.75),
            'l2_regularization_weight': params.get('l2_regularization_weight', 1e-5),
            'use_adamw_optimizer': params.get('use_adamw_optimizer', False)
        })
        
        # Save experiment configuration
        self.tracker.save_experiment_config(experiment_id, params, exp_dir)

        trainer = None  # Initialize trainer variable
        try:
            # Train the model with custom hyperparameters
            trainer = CustomDeepLabV3PlusTrainer(pthDL, **params)
            train_result = trainer.train()

            # Extract history from the result dictionary
            history_dict = train_result.get('history', {}) if isinstance(train_result, dict) else {}

            # Create a mock history object for extract_metrics compatibility
            class HistoryWrapper:
                def __init__(self, history_dict):
                    self.history = history_dict

            history = HistoryWrapper(history_dict)

            # Extract metrics from training history
            metrics = self.extract_metrics(history, trainer)

            # Add training time if available
            if isinstance(train_result, dict) and 'training_time' in train_result:
                metrics['training_time_seconds'] = train_result['training_time']

            # Save training history details for later analysis
            if history_dict:
                metrics['training_history'] = {
                    'loss': history_dict.get('loss', []),
                    'accuracy': history_dict.get('accuracy', []),
                    'val_loss': history_dict.get('val_loss', []),
                    'val_accuracy': history_dict.get('val_accuracy', [])
                }
            
            # Test the model if test path is provided
            if 'pthtest' in self.base_config:
                test_metrics = test_segmentation_model(
                    pthDL,
                    self.base_config['pthtest'],
                    self.base_config.get('pthtestim', os.path.join(self.base_config['pthtest'], '10x')),
                    show_fig=False
                )
                if test_metrics:
                    # Extract test accuracy from confusion matrix
                    if 'confusion_with_metrics' in test_metrics:
                        cm_with_metrics = test_metrics['confusion_with_metrics']
                        # The overall accuracy is in the bottom-right corner
                        test_accuracy = cm_with_metrics[-1, -1]
                        metrics['test_accuracy'] = float(test_accuracy) / 100.0  # Convert percentage to fraction

                        # Calculate per-class metrics
                        n_classes = len(cm_with_metrics) - 1
                        if n_classes > 0:
                            # Precision values are in the last row (excluding the accuracy cell)
                            precision_values = cm_with_metrics[-1, :-1]
                            metrics['avg_precision'] = float(np.mean(precision_values)) / 100.0

                            # Recall values are in the last column (excluding the accuracy cell)
                            recall_values = cm_with_metrics[:-1, -1]
                            metrics['avg_recall'] = float(np.mean(recall_values)) / 100.0

                            # Calculate F1 score (harmonic mean of precision and recall)
                            if metrics['avg_precision'] > 0 and metrics['avg_recall'] > 0:
                                metrics['f1_score'] = 2.0 * (metrics['avg_precision'] * metrics['avg_recall']) / (metrics['avg_precision'] + metrics['avg_recall'])
                            else:
                                metrics['f1_score'] = 0.0

                            # Calculate Matthews Correlation Coefficient (MCC) if possible
                            # Using approximation from overall accuracy for multi-class
                            if metrics['test_accuracy'] > 0:
                                # Simplified MCC calculation for multi-class
                                chance_accuracy = 1.0 / n_classes
                                metrics['mcc'] = (metrics['test_accuracy'] - chance_accuracy) / (1.0 - chance_accuracy)

                            # Store per-class metrics for detailed analysis
                            if 'classNames' in self.base_config:
                                class_names = self.base_config['classNames'][:-1]  # Exclude background
                                for i, class_name in enumerate(class_names[:n_classes]):
                                    class_precision = float(precision_values[i]) / 100.0
                                    class_recall = float(recall_values[i]) / 100.0

                                    metrics[f'precision_{class_name}'] = class_precision
                                    metrics[f'recall_{class_name}'] = class_recall

                                    # Calculate per-class F1 scores
                                    if class_precision > 0 and class_recall > 0:
                                        metrics[f'f1_{class_name}'] = 2.0 * (class_precision * class_recall) / (class_precision + class_recall)
                                    else:
                                        metrics[f'f1_{class_name}'] = 0.0

                            # Calculate weighted F1 score based on class distribution
                            if 'classNames' in self.base_config:
                                f1_scores = [metrics.get(f'f1_{class_name}', 0.0) for class_name in class_names[:n_classes]]
                                metrics['weighted_f1_score'] = float(np.mean(f1_scores))

                    # Calculate distribution mismatch metrics
                    if 'test_accuracy' in metrics and 'best_val_accuracy' in metrics:
                        metrics['val_test_gap'] = metrics['best_val_accuracy'] - metrics['test_accuracy']
                        metrics['generalization_score'] = metrics['test_accuracy'] / max(metrics['best_val_accuracy'], 0.001)

                        # More nuanced distribution mismatch analysis
                        if metrics['val_test_gap'] > 0.15:
                            metrics['distribution_mismatch_warning'] = True
                            metrics['mismatch_severity'] = 'critical'
                        elif metrics['val_test_gap'] > 0.1:
                            metrics['distribution_mismatch_warning'] = True
                            metrics['mismatch_severity'] = 'high'
                        elif metrics['val_test_gap'] > 0.05:
                            metrics['distribution_mismatch_warning'] = True
                            metrics['mismatch_severity'] = 'medium'
                        else:
                            metrics['distribution_mismatch_warning'] = False
                            metrics['mismatch_severity'] = 'low'

                        # Add recommendations based on mismatch
                        if metrics['val_test_gap'] < 0:
                            metrics['test_performance'] = 'better_than_validation'
                        elif metrics['val_test_gap'] < 0.02:
                            metrics['test_performance'] = 'excellent_generalization'
                        elif metrics['val_test_gap'] < 0.05:
                            metrics['test_performance'] = 'good_generalization'
                        else:
                            metrics['test_performance'] = 'poor_generalization'

                    # Store the full test metrics for detailed analysis
                    metrics['test_metrics_available'] = True
            
            # Save metrics
            self.tracker.save_experiment_results(experiment_id, metrics, exp_dir)

            # Check if this is the best model so far
            val_acc = metrics.get('val_accuracy', metrics.get('best_val_accuracy', 0))

            if val_acc > self.best_metric_value:
                logger.info(f"New best model found! Val accuracy: {val_acc:.4f}")
                self.best_metric_value = val_acc
                self.best_experiment_id = experiment_id

                # Save this model as the current best
                best_model_dir = os.path.join(self.output_dir, 'current_best_model')
                if os.path.exists(best_model_dir):
                    shutil.rmtree(best_model_dir)
                shutil.copytree(pthDL, best_model_dir)

                # Save best model metadata
                with open(os.path.join(self.output_dir, 'current_best.json'), 'w') as f:
                    json.dump({
                        'experiment_id': experiment_id,
                        'parameters': params,
                        'metrics': metrics,
                        'val_accuracy': val_acc
                    }, f, indent=2)

            # Clean up temporary training directory
            if os.path.exists(temp_model_dir):
                # Close logger if it exists to release file handles
                if trainer and hasattr(trainer, 'logger') and trainer.logger:
                    try:
                        trainer.logger.close()
                        logger.info("Closed trainer logger to release file handles")
                    except Exception as e:
                        logger.warning(f"Failed to close trainer logger: {str(e)}")

                self._safe_rmtree(temp_model_dir)
                logger.info(f"Cleaned up temporary training directory")

            logger.info(f"Experiment {experiment_id} completed successfully")
            logger.info(f"Metrics: {metrics}")

            return metrics
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            logger.error(traceback.format_exc())

            # Save error information
            metrics = {'error': str(e), 'status': 'failed'}
            self.tracker.save_experiment_results(experiment_id, metrics, exp_dir)

            # Clean up temporary training directory on error
            if os.path.exists(temp_model_dir):
                # Close logger if it exists to release file handles
                if trainer and hasattr(trainer, 'logger') and trainer.logger:
                    try:
                        trainer.logger.close()
                        logger.info("Closed trainer logger to release file handles")
                    except Exception as e2:
                        logger.warning(f"Failed to close trainer logger: {str(e2)}")

                self._safe_rmtree(temp_model_dir)

            return metrics
    
    def extract_metrics(self, history, trainer) -> Dict[str, float]:
        """
        Extract relevant metrics from training history with enhanced interpretability metrics.

        Args:
            history: Keras training history object
            trainer: Trainer instance

        Returns:
            Dictionary of metrics including distribution mismatch indicators
        """
        metrics = {}

        if history and hasattr(history, 'history'):
            hist = history.history

            # Get final epoch metrics
            if 'loss' in hist and hist['loss']:
                metrics['final_loss'] = float(hist['loss'][-1])
                metrics['best_loss'] = float(min(hist['loss']))
                metrics['min_loss'] = metrics['best_loss']
                metrics['loss_reduction'] = float(hist['loss'][0] - hist['loss'][-1]) if len(hist['loss']) > 1 else 0.0

            if 'accuracy' in hist and hist['accuracy']:
                metrics['final_accuracy'] = float(hist['accuracy'][-1])
                metrics['best_accuracy'] = float(max(hist['accuracy']))
                metrics['max_train_accuracy'] = metrics['best_accuracy']
                metrics['accuracy_improvement'] = float(hist['accuracy'][-1] - hist['accuracy'][0]) if len(hist['accuracy']) > 1 else 0.0

            if 'val_loss' in hist and hist['val_loss']:
                metrics['final_val_loss'] = float(hist['val_loss'][-1])
                metrics['best_val_loss'] = float(min(hist['val_loss']))
                metrics['min_val_loss'] = metrics['best_val_loss']

            if 'val_accuracy' in hist and hist['val_accuracy']:
                metrics['final_val_accuracy'] = float(hist['val_accuracy'][-1])
                metrics['best_val_accuracy'] = float(max(hist['val_accuracy']))
                metrics['max_val_accuracy'] = metrics['best_val_accuracy']
                # Use best validation accuracy as primary metric
                metrics['val_accuracy'] = metrics['best_val_accuracy']

            # Training duration
            metrics['epochs_trained'] = len(hist.get('loss', []))

            # Calculate overfitting indicators
            if 'best_accuracy' in metrics and 'best_val_accuracy' in metrics:
                metrics['train_val_gap'] = metrics['best_accuracy'] - metrics['best_val_accuracy']
                metrics['overfitting_ratio'] = metrics['best_accuracy'] / max(metrics['best_val_accuracy'], 0.001)
                metrics['overfitting_severity'] = 'none' if metrics['train_val_gap'] < 0.05 else ('low' if metrics['train_val_gap'] < 0.1 else ('medium' if metrics['train_val_gap'] < 0.15 else 'high'))

            # Calculate convergence metrics
            if 'loss' in hist and len(hist['loss']) > 1:
                # Measure how much the loss decreased in the last 25% of epochs
                quarter_point = max(1, len(hist['loss']) // 4)
                recent_loss_change = hist['loss'][-1] - hist['loss'][-quarter_point]
                metrics['loss_convergence_rate'] = abs(recent_loss_change) / quarter_point

                # Check if model is still improving (negative means improving)
                metrics['loss_still_improving'] = recent_loss_change < -0.001

            if 'val_loss' in hist and len(hist['val_loss']) > 3:
                # Check for validation loss stability (lower variance is better)
                last_epochs = min(5, len(hist['val_loss']))
                recent_val_losses = hist['val_loss'][-last_epochs:]
                metrics['val_loss_stability'] = float(np.std(recent_val_losses))

                # Check for validation loss trend
                if len(hist['val_loss']) > 1:
                    val_loss_trend = hist['val_loss'][-1] - hist['val_loss'][-2]
                    metrics['val_loss_increasing'] = val_loss_trend > 0.001

            # Early stopping behavior
            if 'val_accuracy' in hist and hist['val_accuracy']:
                # Find epoch with best validation accuracy
                best_epoch = np.argmax(hist['val_accuracy']) + 1
                total_epochs = len(hist['val_accuracy'])
                metrics['best_epoch'] = best_epoch
                metrics['epochs_after_best'] = total_epochs - best_epoch
                # Early stopping efficiency (1.0 means stopped at best, 0 means trained way past best)
                metrics['early_stopping_efficiency'] = best_epoch / total_epochs if total_epochs > 0 else 0

                # Check if we stopped too early or too late
                if metrics['epochs_after_best'] == 0:
                    metrics['early_stopping_status'] = 'optimal'
                elif metrics['epochs_after_best'] <= 2:
                    metrics['early_stopping_status'] = 'good'
                else:
                    metrics['early_stopping_status'] = 'late'

            # Learning rate decay effectiveness
            if 'best_val_accuracy' in metrics and 'final_val_accuracy' in metrics:
                # If final is worse than best, LR decay might not be helping
                metrics['lr_decay_effectiveness'] = 1.0 - abs(metrics['best_val_accuracy'] - metrics['final_val_accuracy'])

        # Add any additional metrics from trainer if available
        if hasattr(trainer, 'additional_metrics'):
            metrics.update(trainer.additional_metrics)

        # Add composite data generation metrics if available
        if hasattr(self, 'data_generation_stats'):
            metrics.update(self.data_generation_stats)

        return metrics
    
    def run_search(self, resume: bool = False) -> None:
        """
        Run the complete hyperparameter search.
        
        Args:
            resume: Whether to resume from a checkpoint
        """
        logger.info("Starting hyperparameter grid search...")
        
        # Load checkpoint if resuming
        checkpoint = load_checkpoint(self.checkpoint_file) if resume else {'completed_experiments': [], 'last_index': -1}
        completed = set(checkpoint['completed_experiments'])
        
        # Prepare data once for all experiments
        if not hasattr(self, 'prepared_data_path') or not os.path.exists(self.prepared_data_path):
            self.prepare_data_once()
        
        # Run experiments
        for i, params in enumerate(self.param_grid):
            experiment_id = f"{i:03d}"
            
            # Skip if already completed
            if experiment_id in completed:
                logger.info(f"Skipping completed experiment {experiment_id}")
                continue
            
            logger.info(f"Running experiment {experiment_id}/{len(self.param_grid)-1:03d}")
            
            # Run experiment
            metrics = self.run_single_experiment(experiment_id, params)
            
            # Track results
            self.tracker.add_result(experiment_id, params, metrics)
            
            # Update checkpoint
            completed.add(experiment_id)
            checkpoint = {'completed_experiments': list(completed), 'last_index': i}
            save_checkpoint(self.checkpoint_file, checkpoint)
        
        logger.info("Hyperparameter search completed!")

        # Clean up temporary data preparation directory
        if hasattr(self, 'prepared_data_path') and os.path.exists(self.prepared_data_path):
            self._safe_rmtree(self.prepared_data_path)
            logger.info("Cleaned up temporary data preparation directory")

        # Find and save best model
        self.save_best_model()

        # Generate reports and visualizations
        self.generate_reports()
    
    def save_best_model(self) -> None:
        """
        Identify and save the best performing model, and run final testing with confusion matrix.
        """
        try:
            # Load best model info from saved file
            best_info_file = os.path.join(self.output_dir, 'current_best.json')
            if os.path.exists(best_info_file):
                with open(best_info_file, 'r') as f:
                    best_info = json.load(f)
                best_id = best_info['experiment_id']
                best_exp = best_info
                logger.info(f"Best model: Experiment {best_id} with val_accuracy={best_exp['val_accuracy']:.4f}")
            else:
                # Fallback to getting from tracker if current_best.json doesn't exist
                best_id, best_exp = self.tracker.get_best_experiment('val_accuracy', maximize=True)
                logger.info(f"Best model: Experiment {best_id} with val_accuracy={best_exp.get('val_accuracy', 0):.4f}")

            # Rename current_best_model to best_model
            current_best_dir = os.path.join(self.output_dir, 'current_best_model')
            best_model_dir = os.path.join(self.output_dir, 'best_model')

            if os.path.exists(current_best_dir):
                if os.path.exists(best_model_dir):
                    shutil.rmtree(best_model_dir)
                shutil.move(current_best_dir, best_model_dir)
            else:
                # If current_best doesn't exist, we may need to recreate from experiment data
                logger.warning("Current best model directory not found, unable to save best model")
                return
            
            # Save final best configuration
            best_config_file = os.path.join(self.output_dir, 'best_config.json')
            with open(best_config_file, 'w') as f:
                # Include all available information
                if isinstance(best_exp, dict):
                    json.dump(best_exp, f, indent=2, default=str)
                else:
                    json.dump({'experiment_id': best_id, 'metrics': best_exp}, f, indent=2, default=str)
            
            # Run final test with confusion matrix visualization if test data is available
            if 'pthtest' in self.base_config:
                logger.info("\nRunning final test on best model with confusion matrix visualization...")
                logger.info("=" * 60)
                
                test_metrics = test_segmentation_model(
                    best_model_dir,
                    self.base_config['pthtest'],
                    self.base_config.get('pthtestim', os.path.join(self.base_config['pthtest'], '10x')),
                    show_fig=True  # Show confusion matrix for best model
                )
                
                if test_metrics and 'confusion_with_metrics' in test_metrics:
                    cm_with_metrics = test_metrics['confusion_with_metrics']
                    test_accuracy = cm_with_metrics[-1, -1]
                    
                    logger.info(f"\nBest Model Final Test Results:")
                    logger.info(f"  Test Accuracy: {test_accuracy:.2f}%")
                    
                    # Calculate and report per-class metrics
                    n_classes = len(cm_with_metrics) - 1
                    if n_classes > 0:
                        precision_values = cm_with_metrics[-1, :-1]
                        recall_values = cm_with_metrics[:-1, -1]
                        
                        logger.info(f"  Average Precision: {np.mean(precision_values):.2f}%")
                        logger.info(f"  Average Recall: {np.mean(recall_values):.2f}%")
                        
                        # Report per-class metrics if class names are available
                        if 'classNames' in self.base_config:
                            class_names = self.base_config['classNames'][:-1]  # Exclude background
                            logger.info("\n  Per-class Performance:")
                            for i, class_name in enumerate(class_names[:n_classes]):
                                logger.info(f"    {class_name:15s} - Precision: {precision_values[i]:6.2f}%, Recall: {recall_values[i]:6.2f}%")
                
                logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to save best model: {str(e)}")
    
    def analyze_distribution_mismatch(self) -> Dict[str, Any]:
        """
        Analyze distribution mismatch between training, validation, and test data.

        Returns:
            Dictionary containing analysis results and recommendations
        """
        analysis = {
            'high_mismatch_experiments': [],
            'avg_val_test_gap': 0,
            'avg_train_val_gap': 0,
            'recommendations': [],
            'warnings': []
        }

        # Load results if not in memory
        if not self.tracker.experiments and os.path.exists(self.tracker.results_file):
            import pandas as pd
            df = pd.read_csv(self.tracker.results_file)
            experiments = df.to_dict('records')
        else:
            experiments = self.tracker.experiments

        if not experiments:
            return analysis

        # Collect metrics
        val_test_gaps = []
        train_val_gaps = []
        generalization_scores = []

        for exp in experiments:
            if 'val_test_gap' in exp and exp['val_test_gap'] is not None:
                val_test_gaps.append(exp['val_test_gap'])

                # Track high mismatch experiments
                if exp['val_test_gap'] > 0.1:
                    analysis['high_mismatch_experiments'].append({
                        'id': exp['experiment_id'],
                        'gap': exp['val_test_gap'],
                        'val_acc': exp.get('best_val_accuracy', 0),
                        'test_acc': exp.get('test_accuracy', 0)
                    })

            if 'train_val_gap' in exp and exp['train_val_gap'] is not None:
                train_val_gaps.append(exp['train_val_gap'])

            if 'generalization_score' in exp and exp['generalization_score'] is not None:
                generalization_scores.append(exp['generalization_score'])

        # Calculate averages
        if val_test_gaps:
            analysis['avg_val_test_gap'] = np.mean(val_test_gaps)
            analysis['max_val_test_gap'] = np.max(val_test_gaps)
            analysis['min_val_test_gap'] = np.min(val_test_gaps)

        if train_val_gaps:
            analysis['avg_train_val_gap'] = np.mean(train_val_gaps)
            analysis['max_train_val_gap'] = np.max(train_val_gaps)

        if generalization_scores:
            analysis['avg_generalization_score'] = np.mean(generalization_scores)

        # Generate recommendations based on analysis
        if analysis['avg_val_test_gap'] > 0.15:
            analysis['warnings'].append("CRITICAL: Average validation-test gap exceeds 15%")
            analysis['recommendations'].append(
                "The synthetic composite training data significantly differs from real test images. "
                "Consider using real annotated tiles for training or validation."
            )
        elif analysis['avg_val_test_gap'] > 0.1:
            analysis['warnings'].append("WARNING: Moderate validation-test gap detected (>10%)")
            analysis['recommendations'].append(
                "Consider adding more diverse real tiles to training data or using "
                "domain adaptation techniques."
            )

        if analysis['avg_train_val_gap'] > 0.1:
            analysis['warnings'].append("WARNING: Significant overfitting detected")
            analysis['recommendations'].append(
                "Model is overfitting to training data. Consider: "
                "1) Increasing L2 regularization weight (current experiments test up to 1e-3), "
                "2) Using AdamW optimizer for weight decay, "
                "3) Reducing model capacity, "
                "4) Adding more data augmentation"
            )

        # Check consistency across experiments
        if val_test_gaps and np.std(val_test_gaps) > 0.05:
            analysis['recommendations'].append(
                "High variance in validation-test gaps across experiments suggests "
                "some hyperparameters handle distribution shift better. "
                "Focus on configurations with lower val-test gaps."
            )

        return analysis

    def generate_reports(self) -> None:
        """
        Generate enhanced summary reports and visualizations with distribution analysis.
        """
        logger.info("Generating reports and visualizations...")

        # Perform distribution mismatch analysis
        mismatch_analysis = self.analyze_distribution_mismatch()

        # Log warnings and recommendations
        if mismatch_analysis['warnings']:
            logger.warning("\n" + "="*60)
            logger.warning("DISTRIBUTION MISMATCH ANALYSIS")
            logger.warning("="*60)
            for warning in mismatch_analysis['warnings']:
                logger.warning(f"  {warning}")
            logger.warning("")

        if mismatch_analysis['recommendations']:
            logger.info("\nRECOMMENDATIONS FOR IMPROVING MODEL GENERALIZATION:")
            for i, rec in enumerate(mismatch_analysis['recommendations'], 1):
                logger.info(f"{i}. {rec}")
            logger.info("")

        # Calculate and report storage savings
        total_size = 0
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))

        size_mb = total_size / (1024 * 1024)
        logger.info(f"\nStorage used: {size_mb:.2f} MB")
        logger.info(f"Experiments run: {len(self.param_grid)}")
        logger.info(f"Average storage per experiment: {size_mb/len(self.param_grid):.2f} MB")
        logger.info("(Only best model files kept, metrics saved for all experiments)\n")

        # Import additional visualization functions
        from hyperparameter_utils import (
            plot_correlation_heatmap, plot_learning_curves,
            plot_parameter_importance, create_summary_dashboard,
            plot_metric_comparison
        )

        # Create text summary report
        report_file = os.path.join(self.output_dir, 'summary_report.txt')
        report = create_summary_report(self.tracker, report_file)
        print("\n" + report)

        # 1. Create comprehensive dashboard (NEW)
        dashboard_file = os.path.join(self.output_dir, 'summary_dashboard.png')
        try:
            logger.info("Creating comprehensive summary dashboard...")
            create_summary_dashboard(self.tracker, dashboard_file)
            logger.info(f"Summary dashboard saved to {dashboard_file}")
        except Exception as e:
            logger.warning(f"Failed to create summary dashboard: {str(e)}")

        # 2. Create improved parallel coordinates plot
        parallel_plot_file = os.path.join(self.output_dir, 'parallel_coordinates_plot.png')
        try:
            logger.info("Creating enhanced parallel coordinates plot...")

            # Determine best metric for coloring based on available data
            if self.tracker.experiments:
                df = pd.DataFrame(self.tracker.experiments)
                # Priority: generalization_score > f1_score > val_test_gap > val_accuracy
                if 'generalization_score' in df.columns and df['generalization_score'].notna().sum() > 0:
                    color_metric = 'generalization_score'
                    logger.info("Using generalization_score for coloring (best generalization indicator)")
                elif 'f1_score' in df.columns and df['f1_score'].notna().sum() > 0:
                    color_metric = 'f1_score'
                    logger.info("Using f1_score for coloring (balanced performance metric)")
                elif 'val_test_gap' in df.columns and df['val_test_gap'].notna().sum() > 0:
                    color_metric = 'val_test_gap'
                    logger.info("Using val_test_gap for coloring (distribution mismatch indicator)")
                else:
                    color_metric = 'val_accuracy'
                    logger.info("Using val_accuracy for coloring (default metric)")
            else:
                color_metric = 'val_accuracy'

            plot_parallel_coordinates(self.tracker, parallel_plot_file,
                                    metric_to_color=color_metric,
                                    highlight_top_n=5)
            logger.info(f"Parallel coordinates plot saved to {parallel_plot_file}")
        except Exception as e:
            logger.warning(f"Failed to create parallel coordinates plot: {str(e)}")

        # 3. Create enhanced traditional plots with individual points
        traditional_plot_file = os.path.join(self.output_dir, 'hyperparameter_plots.png')
        try:
            logger.info("Creating enhanced hyperparameter plots...")
            plot_hyperparameter_results(self.tracker, traditional_plot_file,
                                       show_individual_points=True)
            logger.info(f"Hyperparameter plots saved to {traditional_plot_file}")
        except Exception as e:
            logger.warning(f"Failed to create hyperparameter plots: {str(e)}")

        # 4. Create distribution mismatch comparison plot (NEW)
        comparison_file = os.path.join(self.output_dir, 'metric_comparison.png')
        try:
            logger.info("Creating metric comparison plot...")
            plot_metric_comparison(self.tracker, comparison_file)
            logger.info(f"Metric comparison plot saved to {comparison_file}")
        except Exception as e:
            logger.warning(f"Failed to create metric comparison plot: {str(e)}")

        # 5. Create correlation heatmap
        correlation_file = os.path.join(self.output_dir, 'correlation_heatmap.png')
        try:
            logger.info("Creating correlation heatmap...")
            plot_correlation_heatmap(self.tracker, correlation_file)
            logger.info(f"Correlation heatmap saved to {correlation_file}")
        except Exception as e:
            logger.warning(f"Failed to create correlation heatmap: {str(e)}")

        # 5. Create parameter importance plot (NEW)
        importance_file = os.path.join(self.output_dir, 'parameter_importance.png')
        try:
            logger.info("Creating parameter importance plot...")
            plot_parameter_importance(self.tracker, importance_file)
            logger.info(f"Parameter importance plot saved to {importance_file}")
        except Exception as e:
            logger.warning(f"Failed to create parameter importance plot: {str(e)}")

        # 6. Create learning curves for top experiments (NEW)
        curves_file = os.path.join(self.output_dir, 'learning_curves.png')
        try:
            logger.info("Creating learning curves for top experiments...")
            plot_learning_curves(self.tracker, curves_file, top_n=5)
            logger.info(f"Learning curves saved to {curves_file}")
        except Exception as e:
            logger.warning(f"Failed to create learning curves: {str(e)}")


class CustomDeepLabV3PlusTrainer(DeepLabV3PlusTrainer):
    """
    Extended DeepLabV3+ trainer that accepts custom hyperparameters.
    """
    
    def __init__(self, model_path: str,
                 learning_rate: float = 0.0005,
                 batch_size: int = 3,
                 epochs: int = 8,
                 es_patience: int = 6,
                 lr_patience: int = 1,
                 lr_factor: float = 0.75,
                 validation_frequency: int = ModelDefaults.VALIDATION_FREQUENCY,
                 l2_regularization_weight: float = 1e-4,
                 use_adamw_optimizer: bool = False,
                 **kwargs):
        """
        Initialize trainer with custom hyperparameters.
        
        Args:
            model_path: Path to model data
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of epochs to train
            es_patience: Early stopping patience
            lr_patience: Learning rate reduction patience
            lr_factor: Learning rate reduction factor
            validation_frequency: Number of iterations between validations
            **kwargs: Additional parameters
        """
        # Store hyperparameters
        self.custom_learning_rate = learning_rate
        self.custom_batch_size = batch_size
        self.custom_epochs = epochs
        self.custom_es_patience = es_patience
        self.custom_lr_patience = lr_patience
        self.custom_lr_factor = lr_factor
        self.custom_validation_frequency = validation_frequency
        self.custom_l2_regularization_weight = l2_regularization_weight
        self.custom_use_adamw_optimizer = use_adamw_optimizer
        
        # Override batch size and regularization in model data
        super().__init__(model_path)
        self.batch_size = batch_size
        self.l2_regularization_weight = l2_regularization_weight
        self.use_adamw_optimizer = use_adamw_optimizer
        
        # Re-prepare data with new batch size
        self._prepare_data()
    
    def _compile_model(self, model):
        """
        Compile model with custom learning rate and optional L2 regularization.
        """
        import keras
        import tensorflow as tf

        # Use AdamW for weight decay or regular Adam with kernel regularizers
        if self.custom_use_adamw_optimizer:
            # Check for Metal device (AdamW has issues on Apple Silicon)
            gpu_devices = tf.config.list_physical_devices('GPU')
            is_metal = gpu_devices and 'metal' in str(gpu_devices[0]).lower()

            if is_metal:
                self.logger.logger.warning("AdamW optimizer not fully supported on Metal devices, falling back to Adam")
                optimizer = keras.optimizers.Adam(
                    learning_rate=self.custom_learning_rate,
                    epsilon=ModelDefaults.OPTIMIZER_EPSILON
                )
            else:
                optimizer = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=self.custom_learning_rate,
                    weight_decay=self.custom_l2_regularization_weight,
                    epsilon=ModelDefaults.OPTIMIZER_EPSILON
                )
        else:
            optimizer = keras.optimizers.Adam(
                learning_rate=self.custom_learning_rate,
                epsilon=ModelDefaults.OPTIMIZER_EPSILON
            )

        model.compile(
            optimizer=optimizer,
            loss=self.loss_function,
            metrics=["accuracy"],
        )
    
    def _train_model(self, model, callbacks):
        """
        Train model with custom epochs.
        """
        if self.logger:
            self.logger.logger.info(f'Starting DeepLabV3+ training with custom hyperparameters...')
            self.logger.logger.info(f'Learning rate: {self.custom_learning_rate}')
            self.logger.logger.info(f'Batch size: {self.custom_batch_size}')
            self.logger.logger.info(f'Epochs: {self.custom_epochs}')
            self.logger.logger.info(f'L2 regularization weight: {self.custom_l2_regularization_weight}')
            self.logger.logger.info(f'Using AdamW optimizer: {self.custom_use_adamw_optimizer}')
        
        # Update callbacks with custom parameters
        for callback in callbacks:
            if hasattr(callback, 'patience'):
                if 'EarlyStopping' in str(type(callback)):
                    callback.patience = self.custom_es_patience
                elif 'ReduceLROnPlateau' in str(type(callback)):
                    callback.patience = self.custom_lr_patience
                    callback.factor = self.custom_lr_factor
            if hasattr(callback, 'validation_frequency'):
                callback.validation_frequency = self.custom_validation_frequency
        
        history = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_steps=self.val_steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
            epochs=self.custom_epochs
        )
        
        # Store additional metrics
        self.additional_metrics = {
            'actual_epochs': len(history.history.get('loss', [])),
            'final_lr': float(model.optimizer.learning_rate.numpy())
        }
        
        return history


def main():
    """
    Main function to run hyperparameter search.
    """
    parser = argparse.ArgumentParser(description='Hyperparameter grid search for DeepLabV3+ model')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--small', action='store_true', help='Run a small test grid')
    parser.add_argument('--michigan', action='store_true',
                       help='Use Michigan pancreas-optimized hyperparameters for class imbalance')
    parser.add_argument('--output-dir', type=str, default='hyperparameter_search_results',
                       help='Directory to store results')
    args = parser.parse_args()

    # Base configuration - switch between datasets based on args
    if args.michigan:
        # Michigan pancreas dataset configuration
        base_config = {
            'pth': '/Volumes/kiemen-lab-data/Valentina Matos/PanIn Lifespan study/Michigan pancreas model',
            'pthim': os.path.join('/Volumes/kiemen-lab-data/Valentina Matos/PanIn Lifespan study/Michigan pancreas model', '5x'),
            'umpix': 2,  # Microns per pixel at 5x magnification
            'pthtest': os.path.join('/Volumes/kiemen-lab-data/Valentina Matos/PanIn Lifespan study/Michigan pancreas model', 'testing annotations'),
            'pthtestim': os.path.join('/Volumes/kiemen-lab-data/Valentina Matos/PanIn Lifespan study/Michigan pancreas model', 'testing annotations', '5x'),
            'nm': 'michigan_hyperparam_search',
            'resolution': '5x',
            'WS': [[2, 0, 0, 1, 0, 0, 2, 0, 0, 2, 2, 0, 0],  # Michigan-specific whitespace handling
                   [7, 6],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 6],
                   [13, 6, 5, 4, 1, 2, 3, 8, 11, 12, 10, 9, 7],
                   []],
            'sxy': 1024,
            'cmap': np.array([
                [0, 255, 255],    # islets - cyan
                [0, 0, 255],      # ducts - blue
                [85, 255, 0],     # vasculature - lime green
                [255, 255, 127],  # fat - light yellow
                [170, 0, 255],    # acini - purple
                [255, 170, 255],  # ecm - light pink
                [255, 255, 255],  # whitespace - white
                [255, 0, 0],      # panin - red
                [63, 63, 63],     # noise - dark gray
                [255, 0, 254],    # nerves - magenta
                [255, 128, 0],    # endo - orange
                [128, 0, 128]     # immuno - dark purple
            ], dtype=np.int32),
            'classNames': ['islets', 'ducts', 'vasculature', 'fat', 'acini', 'ecm',
                          'whitespace', 'panin', 'noise', 'nerves', 'endo', 'immuno'],
            'classCheck': [],
            'ntrain': 24,      # Increased for better class representation (12 classes)
            'nvalidate': 4,    # Increased for stable validation metrics
            'model_type': 'DeepLabV3_plus'
        }
        logger.info("Using Michigan pancreas dataset configuration")
        logger.info(f"Dataset path: {base_config['pth']}")
        logger.info(f"Number of classes: {len(base_config['classNames'])}")
    else:
        # Default liver tissue dataset configuration
        base_config = {
            'pth': '/Users/tnewton3/Desktop/liver_tissue_data',
            'pthim': os.path.join('/Users/tnewton3/Desktop/liver_tissue_data', '10x'),
            'umpix': 1,
            'pthtest': os.path.join('/Users/tnewton3/Desktop/liver_tissue_data', 'testing_image'),
            'pthtestim': os.path.join('/Users/tnewton3/Desktop/liver_tissue_data', 'testing_image', '10x'),
            'nm': 'hyperparam_search_model',
            'resolution': '10x',
            'WS': [[0, 0, 0, 0, 2, 0, 2],
                   [7, 6],
                   [1, 2, 3, 4, 5, 6, 7],
                   [6, 4, 2, 3, 5, 1, 7],
                   []],
            'sxy': 1024,
            'cmap': np.array([[230, 190, 100],
                              [65, 155, 210],
                              [145, 35, 35],
                              [158, 24, 118],
                              [30, 50, 50],
                              [235, 188, 215],
                              [255, 255, 255]]),
            'classNames': ['PDAC', 'bile duct', 'vasculature', 'hepatocyte', 'immune', 'stroma', 'whitespace'],
            'classCheck': [],
            'ntrain': 24,
            'nvalidate': 4,
            'model_type': 'DeepLabV3_plus'
        }
    
    # Define hyperparameter grid
    if args.small:
        # Small test grid for quick testing
        param_grid = {
            'learning_rate': [0.0005, 0.001],
            'batch_size': [2, 3],
            'epochs': [4, 6],
            'es_patience': [4],
            'lr_factor': [0.75]
        }
    elif args.michigan:
        # Michigan Pancreas-optimized param_grid (for addressing class imbalance)
        # Based on confusion matrix analysis showing 74.4% overall accuracy with poor performance
        # on minority classes (immuno: 8.7% recall, noise: 22.9% recall, whitespace: 65.7% recall)
        # This configuration specifically targets:
        # - Severe class imbalance issues
        # - Need for better minority class learning
        # - Prevention of overfitting on majority classes
        param_grid = {
            'learning_rate': [0.000005, 0.00001, 0.00005, 0.0001],  # Lower LR for imbalanced classes
            'batch_size': [2, 3],  # Small batches for 12-class problem with memory constraints
            'epochs': [12, 16, 20, 24],  # Extended training for minority classes
            'es_patience': [8, 10, 12],  # More patience for complex class boundaries
            'lr_factor': [0.5, 0.75],  # Conservative LR reduction for stability
            'l2_regularization_weight': [1e-5, 5e-5, 1e-4, 5e-4],  # Stronger regularization to prevent overfitting
            'use_adamw_optimizer': [False]  # Test AdamW for better weight decay with imbalanced data
        }
        logger.info("Using Michigan pancreas-optimized hyperparameter grid")
        logger.info(f"Total combinations: {len(list(HyperparameterGrid(param_grid)))}")
        logger.info("Optimized for class imbalance with focus on minority classes (immuno, noise, whitespace)")
    else:
        # Full hyperparameter grid - Revised based on analysis of 576 experiments
        # param_grid = {
        #     'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        #     'batch_size': [2, 3, 4, 6],
        #     'epochs': [4, 6, 8, 10],
        #     'es_patience': [4, 6, 8],
        #     'lr_factor': [0.5, 0.75, 0.9]
        # }
        # Changes:
        # - Removed batch_size=6 (caused 34.7% failure rate, likely OOM)
        # - Added finer granularity for learning rates around optimal regions
        # - Extended epochs to explore longer training
        # Performance baseline: F1 scores 0.9478-0.9504 across all valid runs
        param_grid = {
            'learning_rate': [0.00005, 0.0001, 0.0002, 0.0005, 0.001],
            'batch_size': [2, 3, 4],  # Removed 6 due to OOM failures
            'epochs': [6, 8, 10, 12, 14],  # Extended for longer training exploration
            'es_patience': [4, 6, 8],
            'lr_factor': [0.5, 0.75, 0.9],
            'l2_regularization_weight': [0, 1e-6, 1e-5, 1e-4]  # Added L2 regularization (adjusted range)
        }
    
    # Create and run searcher
    searcher = HyperparameterSearcher(base_config, param_grid, args.output_dir)
    searcher.run_search(resume=args.resume)
    
    logger.info("Hyperparameter search complete!")


if __name__ == '__main__':
    main()