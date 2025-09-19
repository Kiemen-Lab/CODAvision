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
            'lr_factor': params.get('lr_factor', 0.75)
        })
        
        # Save experiment configuration
        self.tracker.save_experiment_config(experiment_id, params, exp_dir)

        trainer = None  # Initialize trainer variable
        try:
            # Train the model with custom hyperparameters
            trainer = CustomDeepLabV3PlusTrainer(pthDL, **params)
            history = trainer.train()

            # Extract metrics from training history
            metrics = self.extract_metrics(history, trainer)

            # Save training history details
            if history and hasattr(history, 'history'):
                metrics['training_history'] = {
                    'loss': history.history.get('loss', []),
                    'accuracy': history.history.get('accuracy', []),
                    'val_loss': history.history.get('val_loss', []),
                    'val_accuracy': history.history.get('val_accuracy', [])
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
        Extract relevant metrics from training history.
        
        Args:
            history: Keras training history object
            trainer: Trainer instance
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if history and hasattr(history, 'history'):
            hist = history.history
            
            # Get final epoch metrics
            if 'loss' in hist:
                metrics['final_loss'] = float(hist['loss'][-1])
                metrics['best_loss'] = float(min(hist['loss']))
            
            if 'accuracy' in hist:
                metrics['final_accuracy'] = float(hist['accuracy'][-1])
                metrics['best_accuracy'] = float(max(hist['accuracy']))
            
            if 'val_loss' in hist:
                metrics['final_val_loss'] = float(hist['val_loss'][-1])
                metrics['best_val_loss'] = float(min(hist['val_loss']))
            
            if 'val_accuracy' in hist:
                metrics['final_val_accuracy'] = float(hist['val_accuracy'][-1])
                metrics['best_val_accuracy'] = float(max(hist['val_accuracy']))
                # Use best validation accuracy as primary metric
                metrics['val_accuracy'] = metrics['best_val_accuracy']
            
            # Training duration
            metrics['epochs_trained'] = len(hist.get('loss', []))
        
        # Add any additional metrics from trainer if available
        if hasattr(trainer, 'additional_metrics'):
            metrics.update(trainer.additional_metrics)
        
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
    
    def generate_reports(self) -> None:
        """
        Generate enhanced summary reports and visualizations.
        """
        logger.info("Generating reports and visualizations...")

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
            plot_parameter_importance, create_summary_dashboard
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
            plot_parallel_coordinates(self.tracker, parallel_plot_file,
                                    metric_to_color='val_accuracy',
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

        # 4. Create correlation heatmap (NEW)
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
                 num_validations: int = 3,
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
            num_validations: Number of validations per epoch
            **kwargs: Additional parameters
        """
        # Store hyperparameters
        self.custom_learning_rate = learning_rate
        self.custom_batch_size = batch_size
        self.custom_epochs = epochs
        self.custom_es_patience = es_patience
        self.custom_lr_patience = lr_patience
        self.custom_lr_factor = lr_factor
        self.custom_num_validations = num_validations
        
        # Override batch size in model data
        super().__init__(model_path)
        self.batch_size = batch_size
        
        # Re-prepare data with new batch size
        self._prepare_data()
    
    def _compile_model(self, model):
        """
        Compile model with custom learning rate.
        """
        import keras
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.custom_learning_rate),
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
        
        # Update callbacks with custom parameters
        for callback in callbacks:
            if hasattr(callback, 'patience'):
                if 'EarlyStopping' in str(type(callback)):
                    callback.patience = self.custom_es_patience
                elif 'ReduceLROnPlateau' in str(type(callback)):
                    callback.patience = self.custom_lr_patience
                    callback.factor = self.custom_lr_factor
            if hasattr(callback, 'num_validations'):
                callback.num_validations = self.custom_num_validations
        
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
    parser.add_argument('--output-dir', type=str, default='hyperparameter_search_results',
                       help='Directory to store results')
    args = parser.parse_args()
    
    # Base configuration (same as non-gui_workflow.py)
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
        'ntrain': 15,
        'nvalidate': 3,
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
    else:
        # Full hyperparameter grid
        param_grid = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'batch_size': [2, 3, 4, 6],
            'epochs': [4, 6, 8, 10],
            'es_patience': [4, 6, 8],
            'lr_factor': [0.5, 0.75, 0.9]
        }
    
    # Create and run searcher
    searcher = HyperparameterSearcher(base_config, param_grid, args.output_dir)
    searcher.run_search(resume=args.resume)
    
    logger.info("Hyperparameter search complete!")


if __name__ == '__main__':
    main()