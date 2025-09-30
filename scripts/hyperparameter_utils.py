"""
Hyperparameter Search Utilities

This module provides utility functions for hyperparameter grid search,
including parameter grid generation, experiment tracking, and result analysis.
"""

import os
import json
import csv
import pickle
import shutil
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


class HyperparameterGrid:
    """
    Class to manage hyperparameter grid generation and iteration.
    """
    
    def __init__(self, param_dict: Dict[str, List[Any]]):
        """
        Initialize hyperparameter grid.
        
        Args:
            param_dict: Dictionary where keys are parameter names and 
                       values are lists of parameter values to try
        """
        self.param_dict = param_dict
        self.param_names = list(param_dict.keys())
        self.param_values = list(param_dict.values())
        self.combinations = list(product(*self.param_values))
        
    def __len__(self) -> int:
        """Return the total number of combinations."""
        return len(self.combinations)
    
    def __iter__(self):
        """Iterate over parameter combinations."""
        for combo in self.combinations:
            yield dict(zip(self.param_names, combo))
    
    def get_combination(self, index: int) -> Dict[str, Any]:
        """
        Get a specific parameter combination by index.
        
        Args:
            index: Index of the combination to retrieve
            
        Returns:
            Dictionary of parameter names and values
        """
        if index >= len(self.combinations):
            raise IndexError(f"Index {index} out of range for {len(self.combinations)} combinations")
        return dict(zip(self.param_names, self.combinations[index]))


class ExperimentTracker:
    """
    Class to track hyperparameter search experiments and results.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize experiment tracker.
        
        Args:
            base_path: Base directory for storing experiment results
        """
        self.base_path = base_path
        self.results_file = os.path.join(base_path, 'results_summary.csv')
        self.experiments = []
        os.makedirs(base_path, exist_ok=True)
        
    def create_experiment_dir(self, experiment_id: str) -> str:
        """
        Create a minimal directory for a specific experiment (config and metrics only).

        Args:
            experiment_id: Unique identifier for the experiment

        Returns:
            Path to the experiment directory
        """
        exp_dir = os.path.join(self.base_path, f'experiment_{experiment_id}')
        os.makedirs(exp_dir, exist_ok=True)
        # No longer creating model or logs subdirectories to save space
        return exp_dir
    
    def save_experiment_config(self, experiment_id: str, params: Dict[str, Any], 
                              exp_dir: Optional[str] = None) -> None:
        """
        Save experiment configuration to JSON file.
        
        Args:
            experiment_id: Unique identifier for the experiment
            params: Hyperparameter configuration
            exp_dir: Optional experiment directory path
        """
        if exp_dir is None:
            exp_dir = os.path.join(self.base_path, f'experiment_{experiment_id}')
            
        config = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'parameters': params
        }
        
        config_file = os.path.join(exp_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_experiment_results(self, experiment_id: str, metrics: Dict[str, Any],
                               exp_dir: Optional[str] = None) -> None:
        """
        Save experiment results to JSON file.

        Args:
            experiment_id: Unique identifier for the experiment
            metrics: Dictionary of metric names and values
            exp_dir: Optional experiment directory path
        """
        if exp_dir is None:
            exp_dir = os.path.join(self.base_path, f'experiment_{experiment_id}')

        # Ensure directory exists
        os.makedirs(exp_dir, exist_ok=True)

        # Save metrics with default converter for numpy/special types
        metrics_file = os.path.join(exp_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            # Convert lists to save space - keep only summary stats for training history
            metrics_copy = metrics.copy()
            if 'training_history' in metrics_copy:
                history = metrics_copy['training_history']
                metrics_copy['training_summary'] = {
                    'final_loss': history['loss'][-1] if history.get('loss') else None,
                    'final_accuracy': history['accuracy'][-1] if history.get('accuracy') else None,
                    'final_val_loss': history['val_loss'][-1] if history.get('val_loss') else None,
                    'final_val_accuracy': history['val_accuracy'][-1] if history.get('val_accuracy') else None,
                    'epochs_completed': len(history.get('loss', []))
                }
                # Keep only last 5 epochs of detailed history for debugging
                for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                    if key in history and len(history[key]) > 5:
                        history[key] = history[key][-5:]

            json.dump(metrics_copy, f, indent=2, default=str)
    
    def add_result(self, experiment_id: str, params: Dict[str, Any],
                   metrics: Dict[str, Any]) -> None:
        """
        Add an experiment result to the tracker.

        Args:
            experiment_id: Unique identifier for the experiment
            params: Hyperparameter configuration
            metrics: Dictionary of metric names and values
        """
        # Create a clean copy of metrics for CSV, excluding complex nested structures
        csv_metrics = {}
        for key, value in metrics.items():
            # Skip complex nested structures that don't serialize well to CSV
            if key in ['training_history', 'training_summary', 'test_metrics']:
                continue
            # Convert boolean values to string for better CSV compatibility
            if isinstance(value, bool):
                csv_metrics[key] = str(value)
            # Ensure numeric values are properly formatted
            elif isinstance(value, (int, float, np.integer, np.floating)):
                csv_metrics[key] = float(value) if not np.isnan(float(value)) else None
            # Keep string values as-is
            elif isinstance(value, str):
                csv_metrics[key] = value
            # Skip other complex types
            elif not isinstance(value, (dict, list, tuple)):
                csv_metrics[key] = value

        result = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            **params,
            **csv_metrics
        }
        self.experiments.append(result)

        # Save to CSV
        self._save_to_csv(result)
    
    def _save_to_csv(self, result: Dict[str, Any]) -> None:
        """
        Append a result to the CSV file.

        Args:
            result: Dictionary containing experiment results
        """
        import pandas as pd

        # If file exists, load it to get all columns
        if os.path.exists(self.results_file):
            existing_df = pd.read_csv(self.results_file)
            existing_columns = list(existing_df.columns)

            # Get all unique columns (existing + new)
            all_columns = existing_columns.copy()
            for key in result.keys():
                if key not in all_columns:
                    all_columns.append(key)

            # Ensure result has values for all columns (use None for missing)
            for col in all_columns:
                if col not in result:
                    result[col] = None

            # Append new row
            new_row_df = pd.DataFrame([result])
            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)

            # Save back to CSV
            combined_df.to_csv(self.results_file, index=False)
        else:
            # First row - create new CSV
            df = pd.DataFrame([result])
            df.to_csv(self.results_file, index=False)
    
    def get_best_experiment(self, metric: str = 'val_accuracy',
                           maximize: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing experiment based on a metric.

        Args:
            metric: Metric to use for comparison
            maximize: Whether to maximize (True) or minimize (False) the metric

        Returns:
            Tuple of (experiment_id, experiment_dict)
        """
        if not self.experiments:
            # Load from CSV if not in memory
            if os.path.exists(self.results_file):
                df = pd.read_csv(self.results_file)
                self.experiments = df.to_dict('records')
            else:
                raise ValueError("No experiments found")

        # Try different metric name variants
        metric_variants = [metric]
        if 'val_accuracy' in metric:
            metric_variants.extend(['best_val_accuracy', 'final_val_accuracy'])
        elif 'val_loss' in metric:
            metric_variants.extend(['best_val_loss', 'final_val_loss'])

        # Filter experiments with valid metric values (try all variants)
        valid_experiments = []
        used_metric = None
        for variant in metric_variants:
            valid_experiments = [exp for exp in self.experiments
                               if variant in exp and exp[variant] is not None
                               and not (isinstance(exp[variant], str) and exp[variant].lower() == 'nan')]
            if valid_experiments:
                used_metric = variant
                break

        if not valid_experiments:
            raise ValueError(f"No experiments with metric '{metric}' (or variants) found")

        # Convert metric values to float if needed
        for exp in valid_experiments:
            try:
                exp[used_metric] = float(exp[used_metric])
            except (ValueError, TypeError):
                pass

        if maximize:
            best_exp = max(valid_experiments, key=lambda x: x[used_metric])
        else:
            best_exp = min(valid_experiments, key=lambda x: x[used_metric])

        return best_exp['experiment_id'], best_exp
    
    def save_best_model(self, source_exp_id: str, dest_name: str = 'best_model') -> str:
        """
        Copy the best model to a dedicated directory.
        
        Args:
            source_exp_id: Experiment ID of the best model
            dest_name: Name for the best model directory
            
        Returns:
            Path to the best model directory
        """
        source_dir = os.path.join(self.base_path, f'experiment_{source_exp_id}', 'model')
        dest_dir = os.path.join(self.base_path, dest_name)
        
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)
        
        logger.info(f"Best model saved to {dest_dir}")
        return dest_dir


def plot_hyperparameter_results(tracker: ExperimentTracker,
                               output_file: Optional[str] = None,
                               show_individual_points: bool = True) -> None:
    """
    Create enhanced visualization plots for hyperparameter search results.

    Args:
        tracker: ExperimentTracker instance with results
        output_file: Optional path to save the plot
        show_individual_points: Whether to show individual experiment points
    """
    # Load data
    if not tracker.experiments and os.path.exists(tracker.results_file):
        df = pd.read_csv(tracker.results_file)
    else:
        df = pd.DataFrame(tracker.experiments)

    if df.empty:
        logger.warning("No results to plot")
        # Create empty plot with message
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No experiment results available',
               ha='center', va='center', fontsize=16, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Clean up figure
        return

    # Identify hyperparameters (columns that vary across experiments)
    param_cols = []
    metric_cols = []

    for col in df.columns:
        if col in ['experiment_id', 'timestamp', 'error', 'status']:
            continue
        # Check if it's a metric (typically contains 'loss', 'accuracy', etc.)
        if any(metric in col.lower() for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'test']):
            # Only include numeric columns
            if pd.api.types.is_numeric_dtype(df[col]) or df[col].apply(lambda x: isinstance(x, (int, float))).all():
                metric_cols.append(col)
        else:
            # Check if values vary (likely a hyperparameter)
            try:
                if df[col].nunique() > 1:
                    param_cols.append(col)
            except:
                # Skip columns that can't be processed
                continue

    if not param_cols:
        logger.warning("No varying hyperparameters found for plotting")
        return

    if not metric_cols:
        logger.warning("No valid metrics found for plotting")
        return

    # Auto-adjust subplot size based on content with limits
    n_params = len(param_cols)
    n_metrics = len(metric_cols)

    # Dynamic figure sizing with reasonable limits
    fig_width = min(96, max(40, 14.0 * n_params))  # Limit max width
    fig_height = min(80, max(24, 12 * n_metrics))   # Limit max height

    fig, axes = plt.subplots(n_metrics, n_params,
                            figsize=(fig_width, fig_height),
                            squeeze=False)  # Always return 2D array

    # Set overall style with proper fallback
    available_styles = plt.style.available
    if 'seaborn-v0_8-whitegrid' in available_styles:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'ggplot' in available_styles:
        plt.style.use('ggplot')
    else:
        # Use default style if no good alternatives
        pass

    # Plot each metric vs each parameter
    for i, metric in enumerate(metric_cols):
        for j, param in enumerate(param_cols):
            ax = axes[i, j]

            try:
                # Filter out NaN and infinite values
                valid_data = df[[param, metric]].copy()
                valid_data = valid_data.replace([np.inf, -np.inf], np.nan).dropna()

                if valid_data.empty:
                    ax.text(0.5, 0.5, 'No valid data',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='gray')
                    ax.set_xlabel(param.replace('_', ' ').title(), fontsize=9)
                    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=9)
                    ax.set_facecolor('#f5f5f5')
                    continue

                # Group by parameter value
                grouped = valid_data.groupby(param)[metric].agg(['mean', 'std', 'count'])

                # Sort index if numeric
                if pd.api.types.is_numeric_dtype(grouped.index):
                    grouped = grouped.sort_index()

                # Handle case where std might be NaN (single value groups)
                grouped['std'] = grouped['std'].fillna(0)

                # Skip if no valid data
                if grouped.empty or grouped['mean'].isna().all():
                    ax.text(0.5, 0.5, 'Insufficient data',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='gray')
                    ax.set_xlabel(param.replace('_', ' ').title(), fontsize=9)
                    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=9)
                    ax.set_facecolor('#f5f5f5')
                    continue

                # Create x positions
                x_positions = np.arange(len(grouped))

                # Plot individual points if requested
                if show_individual_points:
                    for idx, param_val in enumerate(grouped.index):
                        param_data = valid_data[valid_data[param] == param_val][metric]
                        # Add jitter for better visibility
                        jitter = np.random.normal(0, 0.05, len(param_data))
                        ax.scatter(np.full(len(param_data), idx) + jitter,
                                 param_data, alpha=0.3, s=20, color='gray', zorder=1)

                # Plot mean with error bars
                ax.errorbar(x_positions, grouped['mean'],
                           yerr=grouped['std'], marker='o', markersize=8,
                           capsize=5, capthick=2, linewidth=2,
                           color='darkblue', ecolor='darkblue',
                           alpha=0.8, zorder=3, label='Mean ± Std')

                # Add confidence intervals if enough data
                if grouped['count'].min() >= 3:
                    # Calculate 95% confidence interval
                    ci = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
                    ax.fill_between(x_positions,
                                  grouped['mean'] - ci,
                                  grouped['mean'] + ci,
                                  alpha=0.2, color='blue', zorder=2)

                # Highlight best value
                if 'accuracy' in metric.lower() or 'precision' in metric.lower() or 'recall' in metric.lower():
                    best_idx = grouped['mean'].idxmax()
                else:  # For loss metrics, lower is better
                    best_idx = grouped['mean'].idxmin()

                best_pos = list(grouped.index).index(best_idx)
                ax.scatter(best_pos, grouped.loc[best_idx, 'mean'],
                         color='green', s=200, marker='*', zorder=5,
                         edgecolors='darkgreen', linewidth=2)

                # Set labels and formatting
                ax.set_xticks(x_positions)
                if param == 'learning_rate':
                    # Handle mixed types in learning_rate formatting
                    try:
                        ax.set_xticklabels([f'{float(val):.1e}' for val in grouped.index],
                                          rotation=45, ha='right', fontsize=8)
                    except (ValueError, TypeError):
                        ax.set_xticklabels(grouped.index, rotation=45, ha='right', fontsize=8)
                else:
                    ax.set_xticklabels(grouped.index, rotation=45, ha='right', fontsize=8)

                ax.set_xlabel(param.replace('_', ' ').title(), fontsize=9, fontweight='bold')
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=9, fontweight='bold')
                ax.set_title(f'{metric.replace("_", " ").title()} vs {param.replace("_", " ").title()}',
                           fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_facecolor('#fafafa')

                # Add sample size annotations
                for idx, (param_val, row) in enumerate(grouped.iterrows()):
                    ax.text(idx, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                           f'n={int(row["count"])}', ha='center', fontsize=7, color='gray')

            except Exception as e:
                logger.warning(f"Failed to plot {metric} vs {param}: {str(e)}")
                ax.text(0.5, 0.5, f'Plot failed:\n{str(e)[:50]}',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=9, color='red')
                ax.set_xlabel(param.replace('_', ' ').title(), fontsize=9)
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=9)
                ax.set_facecolor('#ffeeee')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_file}")
    else:
        plt.show()


def create_summary_report(tracker: ExperimentTracker, 
                         output_file: Optional[str] = None) -> str:
    """
    Create a text summary report of hyperparameter search results.
    
    Args:
        tracker: ExperimentTracker instance with results
        output_file: Optional path to save the report
        
    Returns:
        Summary report as string
    """
    if not tracker.experiments and os.path.exists(tracker.results_file):
        df = pd.read_csv(tracker.results_file)
    else:
        df = pd.DataFrame(tracker.experiments)
    
    if df.empty:
        return "No experiments to summarize"
    
    report = []
    report.append("=" * 80)
    report.append("HYPERPARAMETER SEARCH SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal experiments: {len(df)}")
    report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Best performing models
    report.append("-" * 40)
    report.append("BEST PERFORMING MODELS")
    report.append("-" * 40)
    
    metrics_to_check = ['val_accuracy', 'val_loss', 'final_val_accuracy', 'final_val_loss', 'test_accuracy', 'avg_precision', 'avg_recall']
    
    for metric in metrics_to_check:
        if metric in df.columns:
            maximize = 'accuracy' in metric
            try:
                exp_id, best_exp = tracker.get_best_experiment(metric, maximize)
                report.append(f"\nBest {metric}: {best_exp[metric]:.4f}")
                report.append(f"  Experiment ID: {exp_id}")
                
                # Show hyperparameters for best model
                param_cols = [col for col in df.columns 
                            if col not in ['experiment_id', 'timestamp'] 
                            and not any(m in col.lower() for m in ['loss', 'accuracy'])]
                
                for param in param_cols:
                    if param in best_exp:
                        report.append(f"  {param}: {best_exp[param]}")
            except ValueError as e:
                report.append(f"\n{metric}: No valid experiments found")
    
    # Parameter importance (based on correlation with performance)
    report.append("\n" + "-" * 40)
    report.append("PARAMETER IMPORTANCE")
    report.append("-" * 40)
    
    if 'val_accuracy' in df.columns:
        param_cols = [col for col in df.columns 
                     if col not in ['experiment_id', 'timestamp'] 
                     and not any(m in col.lower() for m in ['loss', 'accuracy'])]
        
        for param in param_cols:
            # Convert to numeric if possible
            try:
                df[param] = pd.to_numeric(df[param])
                correlation = df[param].corr(df['val_accuracy'])
                report.append(f"\n{param} correlation with val_accuracy: {correlation:.3f}")
            except:
                # For non-numeric parameters, show performance by value
                grouped = df.groupby(param)['val_accuracy'].mean()
                report.append(f"\n{param} average val_accuracy:")
                for val, acc in grouped.items():
                    report.append(f"  {val}: {acc:.4f}")
    
    # Distribution Mismatch Analysis
    report.append("\n" + "-" * 40)
    report.append("DISTRIBUTION MISMATCH ANALYSIS")
    report.append("-" * 40)

    # Check for distribution mismatch metrics
    if 'val_test_gap' in df.columns:
        val_test_gaps = df['val_test_gap'].dropna()
        if len(val_test_gaps) > 0:
            report.append(f"\nValidation-Test Gap:")
            report.append(f"  Average: {val_test_gaps.mean():.4f}")
            report.append(f"  Max:     {val_test_gaps.max():.4f}")
            report.append(f"  Min:     {val_test_gaps.min():.4f}")

            # Count problematic experiments
            high_gap_count = (val_test_gaps > 0.1).sum()
            critical_gap_count = (val_test_gaps > 0.2).sum()
            report.append(f"\n  Experiments with >10% gap: {high_gap_count}/{len(val_test_gaps)} ({100*high_gap_count/len(val_test_gaps):.1f}%)")
            report.append(f"  Experiments with >20% gap: {critical_gap_count}/{len(val_test_gaps)} ({100*critical_gap_count/len(val_test_gaps):.1f}%)")

    if 'train_val_gap' in df.columns:
        train_val_gaps = df['train_val_gap'].dropna()
        if len(train_val_gaps) > 0:
            report.append(f"\nTraining-Validation Gap (Overfitting):")
            report.append(f"  Average: {train_val_gaps.mean():.4f}")
            report.append(f"  Max:     {train_val_gaps.max():.4f}")

            overfit_count = (train_val_gaps > 0.1).sum()
            report.append(f"  Experiments with >10% gap: {overfit_count}/{len(train_val_gaps)} ({100*overfit_count/len(train_val_gaps):.1f}%)")

    if 'generalization_score' in df.columns:
        gen_scores = df['generalization_score'].dropna()
        if len(gen_scores) > 0:
            report.append(f"\nGeneralization Score (test_acc/val_acc):")
            report.append(f"  Average: {gen_scores.mean():.4f}")
            report.append(f"  Best:    {gen_scores.max():.4f}")
            report.append(f"  Worst:   {gen_scores.min():.4f}")

    # Interpretability Guide
    report.append("\n" + "-" * 40)
    report.append("INTERPRETABILITY GUIDE")
    report.append("-" * 40)
    report.append("\n• Validation-Test Gap: Measures how well validation metrics predict test performance")
    report.append("  - <5%: Good generalization, validation metrics are reliable")
    report.append("  - 5-10%: Acceptable, minor distribution differences")
    report.append("  - 10-20%: Concerning, synthetic training data may not represent real data well")
    report.append("  - >20%: Critical, consider using real tiles instead of composite images")
    report.append("\n• Training-Validation Gap: Indicates overfitting level")
    report.append("  - <5%: Good fit, model generalizes well to validation data")
    report.append("  - 5-10%: Minor overfitting, acceptable")
    report.append("  - >10%: Significant overfitting, consider regularization")
    report.append("\n• Generalization Score: Ratio of test to validation accuracy")
    report.append("  - ~1.0: Ideal, validation accurately predicts test performance")
    report.append("  - <0.8: Poor generalization, distribution mismatch")
    report.append("  - >1.1: Unusual, validation may be harder than test")

    # Summary statistics
    report.append("\n" + "-" * 40)
    report.append("SUMMARY STATISTICS")
    report.append("-" * 40)

    for col in df.columns:
        if any(m in col.lower() for m in ['loss', 'accuracy', 'precision', 'recall']):
            if pd.api.types.is_numeric_dtype(df[col]):
                report.append(f"\n{col}:")
                report.append(f"  Mean: {df[col].mean():.4f}")
                report.append(f"  Std:  {df[col].std():.4f}")
                report.append(f"  Min:  {df[col].min():.4f}")
                report.append(f"  Max:  {df[col].max():.4f}")

    report_str = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_str)
        logger.info(f"Report saved to {output_file}")
    
    return report_str


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """
    Load a checkpoint file to resume hyperparameter search.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(checkpoint_file):
        return {'completed_experiments': [], 'last_index': -1}
    
    with open(checkpoint_file, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(checkpoint_file: str, data: Dict[str, Any]) -> None:
    """
    Save checkpoint data for resuming hyperparameter search.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        data: Dictionary containing checkpoint data
    """
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)


def plot_metric_comparison(tracker: ExperimentTracker,
                          output_file: Optional[str] = None) -> None:
    """
    Create visualization comparing training, validation, and test metrics.

    Args:
        tracker: ExperimentTracker instance with results
        output_file: Optional path to save the plot
    """
    # Load data
    if not tracker.experiments and os.path.exists(tracker.results_file):
        df = pd.read_csv(tracker.results_file)
    else:
        df = pd.DataFrame(tracker.experiments)

    if df.empty:
        logger.warning("No results for metric comparison plot")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Train vs Validation vs Test Accuracy Comparison
    ax = axes[0, 0]
    if 'best_accuracy' in df.columns and 'best_val_accuracy' in df.columns and 'test_accuracy' in df.columns:
        x = np.arange(len(df))
        width = 0.25

        train_acc = df['best_accuracy'].fillna(0)
        val_acc = df['best_val_accuracy'].fillna(0)
        test_acc = df['test_accuracy'].fillna(0)

        ax.bar(x - width, train_acc, width, label='Train', color='blue', alpha=0.7)
        ax.bar(x, val_acc, width, label='Validation', color='green', alpha=0.7)
        ax.bar(x + width, test_acc, width, label='Test', color='red', alpha=0.7)

        ax.set_xlabel('Experiment ID')
        ax.set_ylabel('Accuracy')
        ax.set_title('Train vs Validation vs Test Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Distribution Gap Analysis
    ax = axes[0, 1]
    if 'val_test_gap' in df.columns and 'train_val_gap' in df.columns:
        gaps_df = df[['val_test_gap', 'train_val_gap']].dropna()
        if not gaps_df.empty:
            gaps_df.plot(kind='box', ax=ax)
            ax.axhline(y=0.1, color='orange', linestyle='--', label='10% threshold')
            ax.axhline(y=0.2, color='red', linestyle='--', label='20% critical')
            ax.set_ylabel('Gap (fraction)')
            ax.set_title('Distribution Gap Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # 3. Generalization Score Distribution
    ax = axes[1, 0]
    if 'generalization_score' in df.columns:
        gen_scores = df['generalization_score'].dropna()
        if len(gen_scores) > 0:
            ax.hist(gen_scores, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(x=1.0, color='green', linestyle='--', label='Ideal (1.0)')
            ax.axvline(x=0.8, color='orange', linestyle='--', label='Poor (<0.8)')
            ax.set_xlabel('Generalization Score (test/val)')
            ax.set_ylabel('Count')
            ax.set_title('Generalization Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # 4. Metric Correlation with Test Performance
    ax = axes[1, 1]
    if 'test_accuracy' in df.columns and 'best_val_accuracy' in df.columns:
        valid_data = df[['best_val_accuracy', 'test_accuracy']].dropna()
        if not valid_data.empty:
            ax.scatter(valid_data['best_val_accuracy'], valid_data['test_accuracy'],
                      alpha=0.6, s=50, c='blue')

            # Add diagonal line for perfect correlation
            min_val = min(valid_data.min())
            max_val = max(valid_data.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')

            # Calculate and display correlation
            correlation = valid_data['best_val_accuracy'].corr(valid_data['test_accuracy'])
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top')

            ax.set_xlabel('Validation Accuracy')
            ax.set_ylabel('Test Accuracy')
            ax.set_title('Validation vs Test Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle('Model Performance and Distribution Mismatch Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Metric comparison plot saved to {output_file}")
    else:
        plt.show()


def plot_correlation_heatmap(tracker: ExperimentTracker,
                            output_file: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Create a correlation heatmap between hyperparameters and metrics.

    Args:
        tracker: ExperimentTracker instance with results
        output_file: Optional path to save the plot
        figsize: Figure size as (width, height) tuple
    """
    # Load data
    if not tracker.experiments and os.path.exists(tracker.results_file):
        df = pd.read_csv(tracker.results_file)
    else:
        df = pd.DataFrame(tracker.experiments)

    if df.empty:
        logger.warning("No results for correlation heatmap")
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available for correlation heatmap',
               ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return

    # Select numeric columns only
    numeric_cols = []
    for col in df.columns:
        if col in ['experiment_id', 'timestamp', 'error', 'status']:
            continue
        try:
            # Try to convert to numeric and filter out infinite values
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].notna().sum() > 0:  # Has at least some numeric values
                numeric_cols.append(col)
        except Exception as e:
            logger.debug(f"Could not convert column {col} to numeric: {e}")
            continue

    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for correlation heatmap")
        # Create plot with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Not enough numeric columns for correlation analysis',
               ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return

    # Calculate correlation matrix with NaN handling
    try:
        corr_matrix = df[numeric_cols].corr(method='pearson')
        # Replace any remaining NaN values with 0
        corr_matrix = corr_matrix.fillna(0)
    except Exception as e:
        logger.error(f"Failed to calculate correlation matrix: {e}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap with custom colormap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Mask upper triangle
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Prepare labels
    labels = [col.replace('_', ' ').title() for col in numeric_cols]

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
               annot=True, fmt='.2f', square=True, linewidths=1,
               cbar_kws={"shrink": 0.8, "label": "Correlation"},
               xticklabels=labels, yticklabels=labels,
               ax=ax)

    # Adjust label formatting
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    ax.set_title('Hyperparameter-Metric Correlation Heatmap',
                fontsize=14, fontweight='bold', pad=20)

    # Add divider lines between parameters and metrics
    param_count = sum(1 for col in numeric_cols if not any(
        m in col.lower() for m in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'test', 'epochs', 'final_lr']))

    if param_count > 0 and param_count < len(numeric_cols):
        ax.axhline(y=param_count, color='red', linewidth=2, alpha=0.5)
        ax.axvline(x=param_count, color='red', linewidth=2, alpha=0.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to {output_file}")
    else:
        plt.show()


def plot_learning_curves(tracker: ExperimentTracker,
                        output_file: Optional[str] = None,
                        top_n: int = 5,
                        figsize: Optional[Tuple[int, int]] = None) -> None:
    """
    Plot learning curves for top N experiments.

    Args:
        tracker: ExperimentTracker instance with results
        output_file: Optional path to save the plot
        top_n: Number of top experiments to show
        figsize: Figure size (auto-sized if None)
    """
    # Load experiment data
    if not tracker.experiments and os.path.exists(tracker.results_file):
        df = pd.read_csv(tracker.results_file)
    else:
        df = pd.DataFrame(tracker.experiments)

    if df.empty:
        logger.warning("No results for learning curves")
        # Create empty plot with message
        if figsize is None:
            figsize = (14, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'No data available',
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        fig.suptitle('Learning Curves - No Data Available', fontsize=14)
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return

    # Find top experiments by validation accuracy
    metric_col = None
    for col in ['val_accuracy', 'best_val_accuracy', 'final_val_accuracy']:
        if col in df.columns:
            metric_col = col
            break

    if not metric_col:
        logger.warning("No validation accuracy metric found")
        # Create empty plot with message
        if figsize is None:
            figsize = (14, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'No validation metric found',
                   ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        fig.suptitle('Learning Curves - No Validation Metrics', fontsize=14)
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return

    # Get top N experiments
    top_experiments = df.nlargest(min(top_n, len(df)), metric_col)

    # Auto-size figure
    if figsize is None:
        figsize = (14, 8)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Colors for different experiments
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    # Plot learning curves for each top experiment
    for idx, (exp_idx, exp_row) in enumerate(top_experiments.iterrows()):
        exp_id = exp_row.get('experiment_id', f'Exp_{exp_idx}')

        # Try to load training history from experiment directory
        exp_dir = os.path.join(tracker.base_path, f'experiment_{exp_id}')
        metrics_file = os.path.join(exp_dir, 'metrics.json')

        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                if 'training_history' in metrics:
                    history = metrics['training_history']

                    # Plot training loss
                    if 'loss' in history and history['loss']:
                        axes[0].plot(history['loss'], label=f'Exp {exp_id}',
                                   color=colors[idx], linewidth=2, alpha=0.8)

                    # Plot validation loss
                    if 'val_loss' in history and history['val_loss']:
                        axes[1].plot(history['val_loss'], label=f'Exp {exp_id}',
                                   color=colors[idx], linewidth=2, alpha=0.8)

                    # Plot training accuracy
                    if 'accuracy' in history and history['accuracy']:
                        axes[2].plot(history['accuracy'], label=f'Exp {exp_id}',
                                   color=colors[idx], linewidth=2, alpha=0.8)

                    # Plot validation accuracy
                    if 'val_accuracy' in history and history['val_accuracy']:
                        axes[3].plot(history['val_accuracy'], label=f'Exp {exp_id}',
                                   color=colors[idx], linewidth=2, alpha=0.8)
            except Exception as e:
                logger.debug(f"Could not load training history for experiment {exp_id}: {e}")
                # Continue to next experiment instead of failing

    # Configure subplots
    titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']
    y_labels = ['Loss', 'Loss', 'Accuracy', 'Accuracy']

    for ax, title, ylabel in zip(axes, titles, y_labels):
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    fig.suptitle(f'Learning Curves - Top {top_n} Experiments',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves saved to {output_file}")
    else:
        plt.show()


def plot_parameter_importance(tracker: ExperimentTracker,
                             output_file: Optional[str] = None,
                             method: str = 'correlation',
                             figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot parameter importance based on correlation or variance analysis.

    Args:
        tracker: ExperimentTracker instance with results
        output_file: Optional path to save the plot
        method: Method to calculate importance ('correlation' or 'variance')
        figsize: Figure size
    """
    # Load data
    if not tracker.experiments and os.path.exists(tracker.results_file):
        df = pd.read_csv(tracker.results_file)
    else:
        df = pd.DataFrame(tracker.experiments)

    if df.empty:
        logger.warning("No results for parameter importance")
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available for parameter importance analysis',
               ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return

    # Find target metric
    target_metric = None
    for col in ['val_accuracy', 'best_val_accuracy', 'test_accuracy']:
        if col in df.columns:
            target_metric = col
            break

    if not target_metric:
        logger.warning("No target metric found for importance analysis")
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No target metric (accuracy) found for importance analysis',
               ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return

    # Identify parameter columns
    param_cols = []
    for col in df.columns:
        if col in ['experiment_id', 'timestamp', 'error', 'status']:
            continue
        if not any(m in col.lower() for m in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'test', 'epochs', 'final']):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0 and df[col].nunique() > 1:
                    param_cols.append(col)
            except:
                continue

    if not param_cols:
        logger.warning("No parameters found for importance analysis")
        return

    importances = {}

    if method == 'correlation':
        # Calculate absolute correlation with target metric
        for param in param_cols:
            valid_data = df[[param, target_metric]].dropna()
            if len(valid_data) > 1:
                corr = valid_data[param].corr(valid_data[target_metric])
                importances[param] = abs(corr)
    else:  # variance method
        # Calculate importance based on variance when grouped by parameter
        for param in param_cols:
            valid_data = df[[param, target_metric]].dropna()
            if len(valid_data) > 1:
                # Group by parameter value and get variance between groups
                grouped = valid_data.groupby(param)[target_metric].mean()
                if len(grouped) > 1:
                    importances[param] = grouped.std()

    if not importances:
        logger.warning("Could not calculate parameter importances")
        return

    # Sort by importance
    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    params = [p[0].replace('_', ' ').title() for p in sorted_params]
    values = [p[1] for p in sorted_params]

    # Create bar plot
    bars = ax.barh(params, values, color='steelblue', edgecolor='navy', alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlabel(f'Importance ({method.title()})', fontsize=11, fontweight='bold')
    ax.set_title(f'Parameter Importance for {target_metric.replace("_", " ").title()}',
                fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_facecolor('#f8f8f8')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Parameter importance plot saved to {output_file}")
    else:
        plt.show()


def create_summary_dashboard(tracker: ExperimentTracker,
                            output_file: Optional[str] = None,
                            figsize: Tuple[int, int] = (20, 16)) -> None:
    """
    Create a comprehensive dashboard with multiple visualizations.

    Args:
        tracker: ExperimentTracker instance with results
        output_file: Optional path to save the plot
        figsize: Figure size for the dashboard
    """
    # Load data
    if not tracker.experiments and os.path.exists(tracker.results_file):
        df = pd.read_csv(tracker.results_file)
    else:
        df = pd.DataFrame(tracker.experiments)

    if df.empty:
        logger.warning("No results for dashboard")
        # Create empty dashboard with message
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No experiment data available for dashboard',
               ha='center', va='center', fontsize=20, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        fig.suptitle('Hyperparameter Search Summary Dashboard - No Data', fontsize=18)
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)

    # 1. Top experiments table (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_top_experiments_table(df, ax1)

    # 2. Parameter importance (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    _plot_mini_importance(df, ax2)

    # 3. Correlation heatmap (middle-left, spans 2 rows)
    ax3 = fig.add_subplot(gs[1:3, 0])
    _plot_mini_correlation(df, ax3)

    # 4. Performance distribution (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_performance_distribution(df, ax4)

    # 5. Convergence analysis (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    _plot_convergence_analysis(df, ax5)

    # 6. Parameter vs Performance scatter (bottom row, spans all columns)
    ax6 = fig.add_subplot(gs[2, 1:])
    _plot_parameter_scatter(df, ax6)

    # 7. Experiment timeline (bottom)
    ax7 = fig.add_subplot(gs[3, :])
    _plot_experiment_timeline(df, ax7)

    # Add overall title
    fig.suptitle('Hyperparameter Search Summary Dashboard',
                fontsize=18, fontweight='bold', y=0.98)

    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8, alpha=0.5)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Summary dashboard saved to {output_file}")
    else:
        plt.show()


def _plot_top_experiments_table(df: pd.DataFrame, ax) -> None:
    """Helper function to plot top experiments table."""
    ax.axis('off')

    # Validate dataframe
    if df.empty:
        ax.text(0.5, 0.5, 'No experiment data', ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Top Experiments', fontsize=12, fontweight='bold')
        return

    # Find best experiments
    metric_col = None
    for col in ['val_accuracy', 'best_val_accuracy', 'final_val_accuracy']:
        if col in df.columns:
            # Check if column has valid numeric data
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                if numeric_data.notna().sum() > 0:
                    metric_col = col
                    df[col] = numeric_data  # Update with cleaned data
                    break
            except:
                continue

    if not metric_col:
        ax.text(0.5, 0.5, 'No valid metrics available', ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Top Experiments', fontsize=12, fontweight='bold')
        return

    # Get top experiments, handling NaN values
    valid_df = df[df[metric_col].notna()]
    if valid_df.empty:
        ax.text(0.5, 0.5, 'No valid experiments', ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Top Experiments', fontsize=12, fontweight='bold')
        return

    top_df = valid_df.nlargest(min(5, len(valid_df)), metric_col)

    # Select columns to display
    display_cols = ['experiment_id']
    param_cols = ['learning_rate', 'batch_size', 'epochs', 'lr_factor']
    metric_cols = [metric_col, 'test_accuracy'] if 'test_accuracy' in df.columns else [metric_col]

    for col in param_cols + metric_cols:
        if col in top_df.columns:
            display_cols.append(col)

    table_data = top_df[display_cols].round(4)

    # Create table
    table = ax.table(cellText=table_data.values,
                    colLabels=[col.replace('_', ' ').title() for col in display_cols],
                    cellLoc='center',
                    loc='center',
                    colColours=['lightblue'] * len(display_cols))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    ax.set_title('Top 5 Experiments', fontsize=12, fontweight='bold', pad=20)


def _plot_mini_importance(df: pd.DataFrame, ax) -> None:
    """Helper function to plot parameter importance."""
    # Find target metric
    target_metric = None
    for col in ['val_accuracy', 'best_val_accuracy']:
        if col in df.columns:
            target_metric = col
            break

    if not target_metric:
        ax.text(0.5, 0.5, 'No metrics for importance', ha='center', va='center', fontsize=10, color='gray')
        ax.set_title('Parameter Importance', fontsize=11, fontweight='bold')
        ax.axis('off')
        return

    # Calculate importances
    param_cols = ['learning_rate', 'batch_size', 'epochs', 'lr_factor', 'es_patience']
    importances = {}

    for param in param_cols:
        if param in df.columns:
            try:
                df[param] = pd.to_numeric(df[param], errors='coerce')
                valid_data = df[[param, target_metric]].dropna()
                if len(valid_data) > 1:
                    corr = valid_data[param].corr(valid_data[target_metric])
                    importances[param] = abs(corr)
            except:
                continue

    if importances:
        sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:4]
        params = [p[0].replace('_', '\n').title() for p in sorted_params]
        values = [p[1] for p in sorted_params]

        bars = ax.bar(params, values, color='coral', alpha=0.8)
        ax.set_ylabel('Importance', fontsize=9)
        ax.set_title('Parameter Importance', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', fontsize=8)


def _plot_mini_correlation(df: pd.DataFrame, ax) -> None:
    """Helper function to plot correlation heatmap."""
    # Select key columns
    cols = []
    for col in ['learning_rate', 'batch_size', 'epochs', 'val_accuracy', 'test_accuracy']:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    cols.append(col)
            except:
                continue

    if len(cols) < 2:
        ax.text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center', fontsize=10, color='gray')
        ax.set_title('Correlation Matrix', fontsize=11, fontweight='bold')
        ax.axis('off')
        return

    corr_matrix = df[cols].corr()

    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([c.replace('_', '\n') for c in cols], fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels([c.replace('_', ' ') for c in cols], fontsize=8)
    ax.set_title('Correlation Matrix', fontsize=11, fontweight='bold')

    # Add correlation values
    for i in range(len(cols)):
        for j in range(len(cols)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', fontsize=7,
                         color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')


def _plot_performance_distribution(df: pd.DataFrame, ax) -> None:
    """Helper function to plot performance distribution."""
    metric_col = None
    for col in ['val_accuracy', 'best_val_accuracy']:
        if col in df.columns:
            metric_col = col
            break

    if not metric_col:
        ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center', fontsize=10, color='gray')
        ax.set_title('Performance Distribution', fontsize=11, fontweight='bold')
        ax.axis('off')
        return

    # Clean and validate values
    values = pd.to_numeric(df[metric_col], errors='coerce')
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) > 0:
        ax.hist(values, bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {values.mean():.3f}')
        ax.axvline(values.max(), color='green', linestyle='--', linewidth=2,
                  label=f'Best: {values.max():.3f}')
        ax.set_xlabel(metric_col.replace('_', ' ').title(), fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title('Performance Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


def _plot_convergence_analysis(df: pd.DataFrame, ax) -> None:
    """Helper function to plot convergence analysis."""
    if 'epochs_trained' in df.columns:
        epochs = pd.to_numeric(df['epochs_trained'], errors='coerce').dropna()
        if len(epochs) > 0:
            ax.hist(epochs, bins=range(int(epochs.min()), int(epochs.max()) + 2),
                   color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            ax.axvline(epochs.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {epochs.mean():.1f}')
            ax.set_xlabel('Epochs Trained', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_title('Convergence Analysis', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No convergence data', ha='center', va='center')
        ax.set_title('Convergence Analysis', fontsize=11, fontweight='bold')


def _plot_parameter_scatter(df: pd.DataFrame, ax) -> None:
    """Helper function to plot parameter vs performance scatter."""
    metric_col = None
    for col in ['val_accuracy', 'best_val_accuracy']:
        if col in df.columns:
            metric_col = col
            break

    param_col = 'learning_rate' if 'learning_rate' in df.columns else None

    if not metric_col or not param_col:
        ax.text(0.5, 0.5, 'Insufficient data for scatter plot', ha='center', va='center', fontsize=10, color='gray')
        ax.set_title('Parameter vs Performance', fontsize=11, fontweight='bold')
        ax.axis('off')
        return

    try:
        x = pd.to_numeric(df[param_col], errors='coerce')
        y = pd.to_numeric(df[metric_col], errors='coerce')
        valid_mask = x.notna() & y.notna()
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) > 0:
            scatter = ax.scatter(x, y, c=y, cmap='viridis', s=50, alpha=0.6, edgecolors='black')
            ax.set_xlabel(param_col.replace('_', ' ').title(), fontsize=9)
            ax.set_ylabel(metric_col.replace('_', ' ').title(), fontsize=9)
            ax.set_title('Learning Rate vs Performance', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add trend line
            if len(x) > 1:
                z = np.polyfit(np.log10(x), y, 2)
                x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
                y_line = np.polyval(z, np.log10(x_line))
                ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2)
            ax.set_xscale('log')
    except Exception as e:
        ax.text(0.5, 0.5, f'Plot error: {str(e)[:30]}', ha='center', va='center')


def _plot_experiment_timeline(df: pd.DataFrame, ax) -> None:
    """Helper function to plot experiment timeline."""
    if 'timestamp' in df.columns and len(df) > 0:
        try:
            # Parse timestamps
            timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
            valid_timestamps = timestamps[timestamps.notna()]

            if len(valid_timestamps) > 0:
                # Get performance metric
                metric_col = None
                for col in ['val_accuracy', 'best_val_accuracy']:
                    if col in df.columns:
                        metric_col = col
                        break

                if metric_col:
                    y_values = pd.to_numeric(df[metric_col], errors='coerce')
                    valid_mask = timestamps.notna() & y_values.notna()

                    if valid_mask.sum() > 0:
                        x = range(valid_mask.sum())
                        y = y_values[valid_mask].values

                        ax.plot(x, y, 'o-', color='steelblue', markersize=6, linewidth=1.5, alpha=0.7)
                        ax.fill_between(x, y, alpha=0.3, color='steelblue')

                        # Mark best experiment
                        best_idx = np.argmax(y)
                        ax.scatter(best_idx, y[best_idx], color='red', s=100, marker='*',
                                 zorder=5, edgecolors='darkred', linewidth=2)

                        ax.set_xlabel('Experiment Number', fontsize=9)
                        ax.set_ylabel(metric_col.replace('_', ' ').title(), fontsize=9)
                        ax.set_title('Experiment Progress Timeline', fontsize=11, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        return
        except Exception as e:
            logger.debug(f"Timeline plot error: {e}")

    # Fallback if no timestamp data
    ax.text(0.5, 0.5, 'No timeline data available', ha='center', va='center', fontsize=10, color='gray')
    ax.set_title('Experiment Progress Timeline', fontsize=11, fontweight='bold')


def _plot_curved_segment(ax, x1, y1, x2, y2, color, alpha, linewidth, zorder, curve_points=100):
    """
    Helper function to plot a curved line segment between two points using bezier-like curves.

    Args:
        ax: Matplotlib axis
        x1, y1: Starting point coordinates
        x2, y2: Ending point coordinates
        color: Line color
        alpha: Line transparency
        linewidth: Line width
        zorder: Drawing order
        curve_points: Number of interpolation points for the curve
    """
    # Calculate the curve intensity based on vertical distance
    vertical_dist = abs(y2 - y1)

    # Create control points for a smooth S-curve
    # The curve bends outward in the middle section
    if vertical_dist > 0.01:  # Only curve if there's significant vertical change
        # Determine curve direction based on whether line goes up or down
        curve_direction = 1 if y2 > y1 else -1

        # Create 4 control points for cubic bezier-like curve
        # This creates a gentle S-curve between the axes
        t = np.linspace(0, 1, curve_points)

        # Control points for bezier curve
        # P0 is start point, P3 is end point
        # P1 and P2 are control points that define the curve shape
        P0_x, P0_y = x1, y1
        P3_x, P3_y = x2, y2

        # Calculate control points that create an S-curve
        # The curve bulges out to create smooth transitions
        curve_strength = 0.3  # Adjust this to control how curved the lines are

        # For S-curve: first control point pulls in direction of start
        # second control point pulls in direction of end
        P1_x = x1 + (x2 - x1) * 0.3
        P1_y = y1

        P2_x = x1 + (x2 - x1) * 0.7
        P2_y = y2

        # Calculate bezier curve points using cubic bezier formula
        # B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        x_smooth = ((1 - t)**3 * P0_x +
                   3 * (1 - t)**2 * t * P1_x +
                   3 * (1 - t) * t**2 * P2_x +
                   t**3 * P3_x)

        y_smooth = ((1 - t)**3 * P0_y +
                   3 * (1 - t)**2 * t * P1_y +
                   3 * (1 - t) * t**2 * P2_y +
                   t**3 * P3_y)

        # Plot the smooth curve
        ax.plot(x_smooth, y_smooth, color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
    else:
        # For very small changes, just draw a straight line
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)


def plot_parallel_coordinates(tracker: ExperimentTracker,
                             output_file: Optional[str] = None,
                             metric_to_color: str = 'val_accuracy',
                             figsize: Optional[Tuple[int, int]] = None,
                             highlight_top_n: int = 10,
                             equal_visibility: bool = True,
                             jitter_strength: float = 0.02) -> None:
    """
    Create an enhanced parallel coordinates plot for hyperparameter search results.

    This plot shows all hyperparameters and metrics on parallel axes, with each
    experiment represented as a line connecting its values across all axes.
    Lines are colored by performance metric to easily identify best configurations.

    Args:
        tracker: ExperimentTracker instance with results
        output_file: Optional path to save the plot
        metric_to_color: Metric to use for color coding (default: 'val_accuracy')
        figsize: Figure size as (width, height) tuple (auto-sized if None)
        highlight_top_n: Number of top experiments to highlight (default: 10, ignored if equal_visibility=True)
        equal_visibility: If True, all experiments have equal visual weight (default: True)
        jitter_strength: Amount of vertical jitter to apply to prevent overlap (0.0-0.1, default: 0.02)
    """
    # Load data
    if not tracker.experiments and os.path.exists(tracker.results_file):
        df = pd.read_csv(tracker.results_file)
    else:
        df = pd.DataFrame(tracker.experiments)

    if df.empty:
        logger.warning("No results to plot")
        # Create empty plot with message
        if figsize is None:
            figsize = (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No experiment results available for parallel coordinates',
               ha='center', va='center', fontsize=16, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return

    # Calculate F1 scores from existing precision and recall metrics
    if 'avg_precision' in df.columns and 'avg_recall' in df.columns:
        # Calculate F1 score with proper handling of edge cases
        precision = pd.to_numeric(df['avg_precision'], errors='coerce')
        recall = pd.to_numeric(df['avg_recall'], errors='coerce')

        # F1 = 2 * (precision * recall) / (precision + recall)
        # Handle division by zero when precision + recall = 0
        denominator = precision + recall
        df['f1_score'] = np.where(
            denominator > 0,
            2 * (precision * recall) / denominator,
            0.0
        )
        logger.info(f"Calculated F1 scores from avg_precision and avg_recall. Mean F1: {df['f1_score'].mean():.4f}")
    elif 'val_precision' in df.columns and 'val_recall' in df.columns:
        # Fallback to validation precision/recall if available
        precision = pd.to_numeric(df['val_precision'], errors='coerce')
        recall = pd.to_numeric(df['val_recall'], errors='coerce')

        denominator = precision + recall
        df['f1_score'] = np.where(
            denominator > 0,
            2 * (precision * recall) / denominator,
            0.0
        )
        logger.info(f"Calculated F1 scores from val_precision and val_recall. Mean F1: {df['f1_score'].mean():.4f}")
    elif 'test_precision' in df.columns and 'test_recall' in df.columns:
        # Fallback to test precision/recall if available
        precision = pd.to_numeric(df['test_precision'], errors='coerce')
        recall = pd.to_numeric(df['test_recall'], errors='coerce')

        denominator = precision + recall
        df['f1_score'] = np.where(
            denominator > 0,
            2 * (precision * recall) / denominator,
            0.0
        )
        logger.info(f"Calculated F1 scores from test_precision and test_recall. Mean F1: {df['f1_score'].mean():.4f}")

    # Define known hyperparameters explicitly
    known_hyperparams = ['learning_rate', 'batch_size', 'epochs', 'es_patience', 'lr_factor', 'l2_regularization_weight']

    # Define columns to skip entirely
    skip_cols = ['experiment_id', 'timestamp', 'error', 'status', 'training_history',
                 'training_summary', 'test_metrics_available', 'actual_epochs', 'final_lr']

    # Identify columns
    param_cols = []
    metric_cols = []

    for col in df.columns:
        if col in skip_cols:
            continue

        # Check if it's a known hyperparameter
        if col in known_hyperparams:
            # Check if values vary
            try:
                if df[col].nunique() > 1:
                    param_cols.append(col)
            except:
                continue
        else:
            # Everything else that's numeric is a metric
            # This includes: loss, accuracy, precision, recall, f1, gaps, ratios, scores, time, etc.
            try:
                # Try to convert to numeric if not already
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 0:
                    metric_cols.append(col)
            except:
                continue
    
    # Order hyperparameters in a logical sequence
    ordered_param_cols = []
    param_order = ['learning_rate', 'batch_size', 'epochs', 'es_patience', 'lr_factor']
    for param in param_order:
        if param in param_cols:
            ordered_param_cols.append(param)
    # Add any remaining parameters not in the explicit order
    for param in param_cols:
        if param not in ordered_param_cols:
            ordered_param_cols.append(param)
    param_cols = ordered_param_cols

    # Select and order metrics - LIMITED SET FOR READABILITY
    display_metrics = []

    # Priority order for metrics - ONLY THE MOST IMPORTANT ONES
    # This is intentionally limited to 6 key metrics for readability
    metric_priority = [
        # Primary performance metrics (4 metrics)
        'test_accuracy',
        'avg_precision',
        'avg_recall',
        'weighted_f1_score',  # Overall multi-class performance

        # Model health indicators (2 metrics)
        'overfitting_ratio',
        'generalization_score'
    ]

    # Add metrics in priority order - ONLY those in the priority list
    for metric in metric_priority:
        if metric in metric_cols and metric not in display_metrics:
            display_metrics.append(metric)

    # DO NOT add remaining metrics - keep plot readable
    # Skip per-class metrics entirely (precision_*, recall_*, f1_*)
    # The weighted_f1_score captures overall performance across all classes

    if not param_cols:
        logger.warning("No varying hyperparameters found")
        return

    # Combine columns for display (F1 score is now at the rightmost position)
    display_cols = param_cols + display_metrics

    if not display_cols:
        logger.warning("No columns to display in parallel coordinates plot")
        return

    # Auto-size figure based on number of columns with reasonable limits
    if figsize is None:
        width = min(30, max(12, len(display_cols) * 2.2))  # Limit max width
        height = min(15, max(8, 10))  # Standard height
        figsize = (width, height)

    # Create figure with better spacing
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for plotting - only use numeric columns
    plot_df = df[display_cols].copy()

    # Convert to numeric, handle infinite values and drop non-numeric columns
    columns_to_keep = []
    for col in display_cols:
        # Skip boolean columns completely
        if df[col].dtype == bool or df[col].dtype == 'bool':
            continue
        try:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
            plot_df[col] = plot_df[col].replace([np.inf, -np.inf], np.nan)
            # Only keep if we have at least some numeric values
            if plot_df[col].notna().sum() > 0:
                columns_to_keep.append(col)
        except:
            continue

    # Filter to only numeric columns
    if columns_to_keep:
        plot_df = plot_df[columns_to_keep]
        display_cols = columns_to_keep
    else:
        logger.warning("No numeric columns to display in parallel coordinates plot")
        return

    # Modified filtering: Keep rows that have at least some valid data
    # This ensures experiments with F1=0 are preserved even if they have some NaN values
    # Count non-NaN values per row - require at least 30% valid data
    valid_data_mask = plot_df.notna().sum(axis=1) >= max(1, len(display_cols) * 0.3)
    plot_df = plot_df[valid_data_mask]

    if plot_df.empty:
        logger.warning("No valid numeric data to plot")
        # Create empty plot with message
        if figsize is None:
            figsize = (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No valid numeric data for parallel coordinates plot',
               ha='center', va='center', fontsize=16, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        return

    # Store original min/max values for each column (for scaling and labeling)
    col_ranges = {}
    normalized_df = plot_df.copy()
    f1_binned_zero = False  # Track if we binned F1 zeros

    for col in display_cols:
        col_data = plot_df[col].dropna()
        if len(col_data) > 0:
            col_min = col_data.min()
            col_max = col_data.max()

            # Special handling for F1 score with dual-scale normalization
            if col == 'f1_score':
                # Get non-zero values
                non_zero_data = col_data[col_data > 0]
                if len(non_zero_data) > 0:
                    # Find the minimum and maximum non-zero values
                    min_non_zero = non_zero_data.min()
                    max_non_zero = non_zero_data.max()

                    # If we have zeros and non-zeros, create dual-scale visualization
                    if col_min == 0 and min_non_zero > 0:
                        f1_binned_zero = True

                        # Store metadata for axis labeling
                        col_ranges[col] = {
                            'min': 0,  # Keep original min for display
                            'max': col_max,
                            'binned_zero': True,
                            'dual_scale': True,
                            'min_non_zero': min_non_zero,
                            'max_non_zero': max_non_zero,
                            'zero_region_max': 0.15,  # Top of zero region
                            'gap_start': 0.15,  # Start of visual gap
                            'gap_end': 0.20,  # End of visual gap
                            'success_region_min': 0.20  # Bottom of success region
                        }

                        # Apply dual-scale normalization
                        temp_normalized = plot_df[col].copy()

                        # Zero values go to the true zero position (0.0)
                        zero_mask = (plot_df[col] == 0).to_numpy()
                        temp_normalized[zero_mask] = 0.0

                        # Non-zero values are scaled to 0.20-1.0 range (80% of vertical space)
                        non_zero_mask = (plot_df[col] > 0).to_numpy()
                        if max_non_zero > min_non_zero:
                            # Scale non-zero values to expanded range
                            scaled_values = 0.20 + 0.80 * ((plot_df[col][non_zero_mask] - min_non_zero) / (max_non_zero - min_non_zero))
                            temp_normalized[non_zero_mask] = scaled_values
                        else:
                            # All non-zero values are the same
                            temp_normalized[non_zero_mask] = 0.60  # Middle of success region

                        normalized_df[col] = temp_normalized
                    else:
                        # Normal case - no zeros or all zeros
                        col_ranges[col] = {'min': col_min, 'max': col_max}
                        if col_max > col_min:
                            normalized_df[col] = (plot_df[col] - col_min) / (col_max - col_min)
                        else:
                            normalized_df[col] = 0.5
                else:
                    # All values are zero or same
                    col_ranges[col] = {'min': col_min, 'max': col_max}
                    normalized_df[col] = 0.5
            else:
                # Normal handling for other columns
                col_ranges[col] = {'min': col_min, 'max': col_max}
                # Normalize for internal plotting (but keep originals for display)
                if col_max > col_min:
                    normalized_df[col] = (plot_df[col] - col_min) / (col_max - col_min)
                else:
                    # If all values are the same, set to 0.5
                    normalized_df[col] = 0.5
        else:
            col_ranges[col] = {'min': 0, 'max': 1}

    # Set up color mapping based on metric
    # Try to find the color metric in various forms
    color_metric_found = False
    for variant in [metric_to_color, f'best_{metric_to_color}', f'final_{metric_to_color}']:
        if variant in df.columns:
            color_values = pd.to_numeric(df[variant], errors='coerce').values
            color_metric_found = True
            metric_to_color = variant  # Update to the found variant
            break

    if not color_metric_found:
        # Use index if metric not found
        logger.warning(f"Color metric '{metric_to_color}' not found, using index for coloring")
        color_values = np.arange(len(df))
    else:
        # Handle NaN values in color metric more robustly
        valid_mask = ~np.isnan(color_values)
        if valid_mask.any():
            min_val = np.nanmin(color_values)
            color_values = np.nan_to_num(color_values, nan=min_val)
        else:
            logger.warning(f"All values for color metric '{metric_to_color}' are NaN, using index")
            color_values = np.arange(len(df))

    # Special handling for F1 score to create meaningful colorbar
    f1_custom_colorbar = False
    if metric_to_color == 'f1_score':
        # Check if we have both zeros and non-zeros
        non_zero_mask = (color_values > 0).to_numpy() if hasattr(color_values > 0, 'to_numpy') else color_values > 0
        zero_mask = (color_values == 0).to_numpy() if hasattr(color_values == 0, 'to_numpy') else color_values == 0

        if zero_mask.any() and non_zero_mask.any():
            f1_custom_colorbar = True
            # Store original values for colorbar
            original_color_values = color_values.copy()

            # Get the range of non-zero values
            min_non_zero = color_values[non_zero_mask].min()
            max_non_zero = color_values[non_zero_mask].max()

            # Create adjusted color values
            # Map zeros to -1 (will get dark red color)
            # Map non-zeros to 0-1 range for full gradient
            adjusted_color_values = color_values.copy()
            adjusted_color_values[zero_mask] = -1
            if max_non_zero > min_non_zero:
                adjusted_color_values[non_zero_mask] = (color_values[non_zero_mask] - min_non_zero) / (max_non_zero - min_non_zero)
            else:
                adjusted_color_values[non_zero_mask] = 0.5

            # Use adjusted values for coloring
            color_values = adjusted_color_values

            # Create custom normalization for the adjusted range
            norm = Normalize(vmin=-1.2, vmax=1.0)  # Extend below -1 for better red

            # Store metadata for colorbar
            f1_colorbar_info = {
                'min_non_zero': min_non_zero,
                'max_non_zero': max_non_zero,
                'has_zeros': True
            }
        else:
            # Normal handling if no zeros or all zeros
            f1_custom_colorbar = False
            norm = Normalize(vmin=np.nanmin(color_values), vmax=np.nanmax(color_values))
    else:
        # Normal color normalization for other metrics
        unique_colors = np.unique(color_values[~np.isnan(color_values)])
        if len(unique_colors) > 1:
            norm = Normalize(vmin=np.nanmin(color_values), vmax=np.nanmax(color_values))
        elif len(unique_colors) == 1:
            # All values are the same, create a small range around the value
            val = unique_colors[0]
            norm = Normalize(vmin=val - 0.1, vmax=val + 0.1)
        else:
            # No valid values, use default range
            norm = Normalize(vmin=0, vmax=1)

    # Use a diverging colormap for better performance visualization
    metric_lower = metric_to_color.lower()
    if ('accuracy' in metric_lower or 'precision' in metric_lower or
        'recall' in metric_lower or 'f1' in metric_lower or
        'generalization_score' in metric_lower):
        # Higher is better - use green for good, red for bad
        cmap = cm.get_cmap('RdYlGn')
    elif 'loss' in metric_lower or 'gap' in metric_lower:
        # Lower is better - use green for low, red for high
        cmap = cm.get_cmap('RdYlGn_r')
    else:
        # Default colormap
        cmap = cm.get_cmap('coolwarm')

    # Get top experiments for highlighting
    top_indices = []
    if color_metric_found and highlight_top_n > 0:
        # Filter out NaN values before getting top experiments
        valid_df = df[df[metric_to_color].notna()]
        if len(valid_df) > 0:
            top_indices = valid_df.nlargest(min(highlight_top_n, len(valid_df)), metric_to_color).index.tolist()
    
    # Create jittered values to prevent overlap
    # Use deterministic jitter based on experiment index for reproducibility
    np.random.seed(42)  # Set seed for reproducibility
    jittered_df = normalized_df.copy()

    if jitter_strength > 0:
        for i, col in enumerate(display_cols):
            # Identify columns with many repeated values (discrete parameters)
            unique_vals = plot_df[col].dropna().unique()
            total_vals = len(plot_df[col].dropna())

            # Apply jitter to columns with repeated values (less than 50% unique values)
            # or specifically to hyperparameters that are often discrete
            if (len(unique_vals) < total_vals * 0.5 or
                col in ['batch_size', 'epochs', 'es_patience', 'lr_factor'] or
                (col in param_cols and len(unique_vals) <= 10)):

                # Create jitter for each experiment
                # Use a hash of the experiment index to get consistent jitter
                for idx in range(len(jittered_df)):
                    # Create deterministic jitter based on experiment index
                    # This ensures the same experiment always gets the same jitter
                    jitter_seed = hash((idx, i)) % 10000
                    np.random.seed(jitter_seed)
                    jitter = np.random.uniform(-jitter_strength, jitter_strength)

                    # Apply jitter, keeping values within [0, 1] range
                    if not pd.isna(jittered_df.iloc[idx, i]):
                        jittered_val = jittered_df.iloc[idx, i] + jitter
                        jittered_df.iloc[idx, i] = np.clip(jittered_val, 0, 1)

    # Sort experiments by color metric for better layering
    sorted_indices = np.argsort(color_values)

    # Plot experiments with equal or differential visibility based on setting
    if equal_visibility:
        # Equal visibility for all experiments
        for sort_idx, idx in enumerate(sorted_indices):
            if idx >= len(jittered_df):
                continue
            row = jittered_df.iloc[idx]
            values = row.values
            positions = np.arange(len(display_cols))

            # Adjusted visibility settings for better separation
            alpha = 0.6  # Slightly higher alpha for better visibility
            linewidth = 0.8  # Slightly thinner lines to reduce overlap
            # Z-order based on performance (better experiments on top)
            zorder = sort_idx
            color = cmap(norm(color_values[idx]))

            # Plot curved line segments with better NaN handling
            # Find continuous segments and plot them separately
            segments = []
            current_segment = []

            for i in range(len(positions)):
                if not np.isnan(values[i]):
                    current_segment.append(i)
                else:
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = []

            if len(current_segment) >= 2:
                segments.append(current_segment)

            # Plot each continuous segment
            for segment in segments:
                for i in range(len(segment) - 1):
                    idx1, idx2 = segment[i], segment[i+1]
                    _plot_curved_segment(ax, positions[idx1], values[idx1],
                                       positions[idx2], values[idx2],
                                       color, alpha, linewidth, zorder)
    else:
        # Original behavior: highlight top experiments
        # First pass: plot non-top experiments with low alpha
        for idx in sorted_indices:
            if idx >= len(jittered_df) or idx in top_indices:
                continue
            row = jittered_df.iloc[idx]
            values = row.values
            positions = np.arange(len(display_cols))

            # Low visibility for background experiments
            alpha = 0.15  # Slightly more visible
            linewidth = 0.4  # Slightly thicker
            zorder = 1
            color = cmap(norm(color_values[idx]))

            # Plot curved line segments with better NaN handling
            # Find continuous segments and plot them separately
            segments = []
            current_segment = []

            for i in range(len(positions)):
                if not np.isnan(values[i]):
                    current_segment.append(i)
                else:
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = []

            if len(current_segment) >= 2:
                segments.append(current_segment)

            # Plot each continuous segment
            for segment in segments:
                for i in range(len(segment) - 1):
                    idx1, idx2 = segment[i], segment[i+1]
                    _plot_curved_segment(ax, positions[idx1], values[idx1],
                                       positions[idx2], values[idx2],
                                       color, alpha, linewidth, zorder)

        # Second pass: plot top experiments with high visibility
        if top_indices:
            for rank, idx in enumerate(top_indices):
                if idx >= len(jittered_df):
                    continue
                row = jittered_df.iloc[idx]
                values = row.values
                positions = np.arange(len(display_cols))

                # High visibility for top performers
                alpha = 0.85 + 0.15 * (1 - rank / max(len(top_indices), 1))
                linewidth = 2.5
                zorder = 100 + (len(top_indices) - rank)
                color = cmap(norm(color_values[idx]))

                # Plot with curved line segments for smoother appearance
                # Find continuous segments and plot them separately
                segments = []
                current_segment = []

                for i in range(len(positions)):
                    if not np.isnan(values[i]):
                        current_segment.append(i)
                    else:
                        if len(current_segment) >= 2:
                            segments.append(current_segment)
                        current_segment = []

                if len(current_segment) >= 2:
                    segments.append(current_segment)

                # Plot each continuous segment
                for segment in segments:
                    for i in range(len(segment) - 1):
                        idx1, idx2 = segment[i], segment[i+1]
                        _plot_curved_segment(ax, positions[idx1], values[idx1],
                                           positions[idx2], values[idx2],
                                           color, alpha, linewidth, zorder)

                # Add small markers at data points for top experiments
                valid_mask = ~np.isnan(values)
                if np.sum(valid_mask) > 0:
                    ax.scatter(positions[valid_mask], values[valid_mask],
                              color=color, s=15, zorder=zorder+1,
                              edgecolors='white', linewidths=0.5, alpha=alpha)
    
    # Customize axes
    ax.set_xticks(positions)
    
    # Create improved labels with better formatting
    # Track used labels to avoid duplicates
    labels = []
    used_labels = set()

    for col in display_cols:
        # Create concise, readable labels
        if col == 'learning_rate':
            short_name = 'Learning\nRate'
        elif col == 'batch_size':
            short_name = 'Batch\nSize'
        elif col == 'epochs':
            short_name = 'Epochs'
        elif col == 'es_patience':
            short_name = 'Early Stop\nPatience'
        elif col == 'lr_factor':
            short_name = 'LR\nFactor'
        elif col == 'f1_score':
            short_name = 'F1\nScore'
        # Handle various accuracy metrics with unique labels
        elif col == 'best_val_accuracy':
            short_name = 'Best Val\nAccuracy'
        elif col == 'final_val_accuracy':
            short_name = 'Final Val\nAccuracy'
        elif col == 'val_accuracy':
            short_name = 'Val\nAccuracy'
        elif col == 'test_accuracy':
            short_name = 'Test\nAccuracy'
        # Handle various loss metrics with unique labels
        elif col == 'best_val_loss':
            short_name = 'Best Val\nLoss'
        elif col == 'final_val_loss':
            short_name = 'Final Val\nLoss'
        elif col == 'val_loss':
            short_name = 'Val\nLoss'
        elif col == 'best_loss':
            short_name = 'Best\nLoss'
        elif col == 'final_loss':
            short_name = 'Final\nLoss'
        # Handle gap and ratio metrics
        elif col == 'train_val_gap':
            short_name = 'Train-Val\nGap'
        elif col == 'val_test_gap':
            short_name = 'Val-Test\nGap'
        elif col == 'overfitting_ratio':
            short_name = 'Overfitting\nRatio'
        elif col == 'generalization_score':
            short_name = 'Generalization\nScore'
        elif col == 'training_time_seconds':
            short_name = 'Training\nTime (s)'
        elif 'epochs_trained' in col:
            short_name = 'Actual\nEpochs'
        elif col == 'avg_precision':
            short_name = 'Avg\nPrecision'
        elif col == 'avg_recall':
            short_name = 'Avg\nRecall'
        else:
            # Default formatting
            short_name = col.replace('_', ' ').title()
            words = short_name.split()
            if len(words) > 2:
                short_name = '\n'.join(words[:2]) + '\n' + ' '.join(words[2:])
            elif len(words) == 2:
                short_name = '\n'.join(words)

        # Ensure uniqueness
        if short_name in used_labels:
            # Add a suffix to make it unique
            suffix_num = 2
            while f"{short_name} ({suffix_num})" in used_labels:
                suffix_num += 1
            short_name = f"{short_name} ({suffix_num})"

        used_labels.add(short_name)
        labels.append(short_name)

    # Set labels with better rotation for readability
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, len(display_cols) - 0.5)  # Tightly bound x-axis to remove extra space

    # Remove the normalized value label since we'll show actual values
    ax.set_ylabel('')  # Will be replaced with actual value indicators

    # Remove the y-axis ticks and labels (normalized values not meaningful to users)
    ax.set_yticks([])  # Remove all y-axis tick marks
    ax.set_yticklabels([])  # Remove all y-axis tick labels

    # Optionally hide the left spine for a cleaner look
    ax.spines['left'].set_visible(False)
    
    # Add vertical lines for each axis
    for pos in positions:
        ax.axvline(x=pos, color='gray', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add value indicators on axes with improved formatting
    for i, (pos, col) in enumerate(zip(positions, display_cols)):
        # Draw main axis line
        ax.plot([pos, pos], [0, 1], 'k-', linewidth=1.5, alpha=0.7, zorder=998)

        # Add visual separator for F1 score dual-scale
        if col == 'f1_score' and isinstance(col_ranges[col], dict) and 'dual_scale' in col_ranges[col]:
            # Draw separator line between failed and successful regions
            gap_start = col_ranges[col]['gap_start']
            gap_end = col_ranges[col]['gap_end']

            # Draw a dashed line to show the separation
            ax.plot([pos - 0.05, pos + 0.05], [gap_end, gap_end],
                   'r--', linewidth=2, alpha=0.6, zorder=999)

            # Add subtle shading for the failed region
            ax.add_patch(plt.Rectangle((pos - 0.04, 0), 0.08, gap_start,
                                      facecolor='red', alpha=0.05, zorder=1))

            # Add labels for the regions
            ax.text(pos - 0.08, 0.07, 'Failed',
                   rotation=90, va='center', ha='right',
                   fontsize=8, color='red', alpha=0.7, weight='bold')
            ax.text(pos - 0.08, 0.6, 'Successful',
                   rotation=90, va='center', ha='right',
                   fontsize=8, color='green', alpha=0.7, weight='bold')

        # Add tick marks and value labels for each axis
        if col in col_ranges:
            col_min = col_ranges[col]['min']
            col_max = col_ranges[col]['max']

            # Get unique values for this column from the actual data
            unique_values = sorted(plot_df[col].dropna().unique())

            # Determine which values to show based on the number of unique values
            if len(unique_values) <= 10:
                # Show all unique values if there are 10 or fewer
                values_to_show = unique_values
            elif col in param_cols:
                # For hyperparameters with many values, show all if they're discrete settings
                # (like batch sizes or epochs that were actually tested)
                if col in ['batch_size', 'epochs', 'es_patience'] and len(unique_values) <= 20:
                    values_to_show = unique_values
                else:
                    # Show min, max, and evenly distributed intermediate values
                    values_to_show = [unique_values[0], unique_values[-1]]
                    # Add quartiles if they exist
                    if len(unique_values) >= 4:
                        q1_idx = len(unique_values) // 4
                        q2_idx = len(unique_values) // 2
                        q3_idx = 3 * len(unique_values) // 4
                        values_to_show.extend([unique_values[q1_idx], unique_values[q2_idx], unique_values[q3_idx]])
                    values_to_show = sorted(set(values_to_show))
            else:
                # For metrics, show min, max, and key percentiles
                values_to_show = [unique_values[0], unique_values[-1]]
                if len(unique_values) >= 3:
                    median_idx = len(unique_values) // 2
                    values_to_show.append(unique_values[median_idx])
                if len(unique_values) >= 5:
                    q1_idx = len(unique_values) // 4
                    q3_idx = 3 * len(unique_values) // 4
                    values_to_show.extend([unique_values[q1_idx], unique_values[q3_idx]])
                values_to_show = sorted(set(values_to_show))

            # Special handling for F1 score with dual-scale
            if col == 'f1_score' and isinstance(col_ranges[col], dict) and 'dual_scale' in col_ranges[col]:
                # Custom labels for dual-scale F1 score
                # Label for failed experiments (zeros)
                if 0 in unique_values:
                    # Draw tick at the true zero position
                    ax.plot([pos-0.03, pos+0.03], [0.0, 0.0], 'r-', linewidth=1.2, alpha=0.7, zorder=999)
                    val_text = '0 (Failed)'
                    offset = 0.12
                    ax.text(pos + offset, 0.0, val_text,
                           ha='left', va='center',
                           fontsize=7, alpha=0.9, fontweight='bold', color='red',
                           zorder=1000,
                           bbox=dict(boxstyle='round,pad=0.1',
                                    facecolor='#ffeeee', alpha=0.8, edgecolor='red'))

                # Labels for successful experiments (non-zero values)
                # Show min, max, and a few intermediate values
                min_non_zero = col_ranges[col]['min_non_zero']
                max_non_zero = col_ranges[col]['max_non_zero']

                # Calculate positions in normalized space
                success_values_to_show = [min_non_zero, max_non_zero]
                if len(non_zero_data := col_data[col_data > 0]) > 2:
                    # Add median
                    median_val = non_zero_data.median()
                    if median_val not in success_values_to_show:
                        success_values_to_show.insert(1, median_val)
                    # Add quartiles if enough data
                    if len(non_zero_data) >= 10:
                        q1 = non_zero_data.quantile(0.25)
                        q3 = non_zero_data.quantile(0.75)
                        success_values_to_show = sorted(set([min_non_zero, q1, median_val, q3, max_non_zero]))

                # Draw labels for success region
                for j, actual_val in enumerate(success_values_to_show):
                    # Calculate normalized position using dual-scale
                    if max_non_zero > min_non_zero:
                        norm_val = 0.20 + 0.80 * ((actual_val - min_non_zero) / (max_non_zero - min_non_zero))
                    else:
                        norm_val = 0.60

                    # Draw tick mark
                    ax.plot([pos-0.03, pos+0.03], [norm_val, norm_val], 'g-', linewidth=0.8, alpha=0.5, zorder=999)

                    # Format value text
                    val_text = f'{actual_val:.4f}'

                    # Alternate positioning to avoid overlap
                    offset = 0.12 if j % 2 == 0 else -0.12
                    ha = 'left' if offset > 0 else 'right'

                    ax.text(pos + offset, norm_val, val_text,
                           ha=ha, va='center',
                           fontsize=7, alpha=0.9, fontweight='normal',
                           zorder=1000,
                           bbox=dict(boxstyle='round,pad=0.1',
                                    facecolor='#eeffee', alpha=0.7, edgecolor='green'))

                # Skip normal processing for F1 score
                values_to_show = []

            # Draw tick marks and labels for each unique value
            for j, actual_val in enumerate(values_to_show):
                # Calculate normalized position for this value
                # Skip F1 score as it's already handled above with dual-scale
                if col == 'f1_score' and isinstance(col_ranges[col], dict) and 'dual_scale' in col_ranges[col]:
                    continue  # Already handled above with custom dual-scale logic
                elif col_max > col_min:
                    norm_val = (actual_val - col_min) / (col_max - col_min)
                else:
                    norm_val = 0.5

                # Draw subtle tick mark
                ax.plot([pos-0.03, pos+0.03], [norm_val, norm_val], 'k-', linewidth=0.8, alpha=0.5, zorder=999)

                # Format value based on parameter type
                if col == 'learning_rate':
                    if actual_val < 0.01:
                        val_text = f'{actual_val:.1e}'
                    else:
                        val_text = f'{actual_val:.4f}'
                elif col == 'batch_size':
                    val_text = f'{int(actual_val)}'
                elif col == 'epochs':
                    val_text = f'{int(actual_val)}'
                elif col == 'es_patience':
                    val_text = f'{int(actual_val)}'
                elif col == 'lr_factor':
                    val_text = f'{actual_val:.2f}'
                elif col == 'f1_score':
                    # Special formatting for F1 score
                    val_text = f'{actual_val:.3f}'
                elif 'accuracy' in col.lower():
                    # Show as percentage
                    val_text = f'{actual_val*100:.1f}%' if col_max <= 1 else f'{actual_val:.1f}'
                elif 'loss' in col.lower():
                    val_text = f'{actual_val:.4f}'
                else:
                    val_text = f'{actual_val:.3f}'

                # Position labels intelligently to avoid overlap
                # Use alternating offsets for dense labels
                if len(values_to_show) > 5:
                    # For many values, alternate left and right more aggressively
                    offset = 0.12 if j % 2 == 0 else -0.12
                else:
                    # Use different offset patterns for different columns
                    if i < len(param_cols):  # Hyperparameters
                        offset = 0.12 if i % 2 == 0 else -0.12
                    else:  # Metrics
                        offset = -0.12 if (i - len(param_cols)) % 2 == 0 else 0.12

                ha = 'left' if offset > 0 else 'right'

                ax.text(pos + offset, norm_val, val_text,
                       ha=ha, va='center',
                       fontsize=7, alpha=0.9, fontweight='normal',
                       zorder=1000,
                       bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add improved colorbar with performance indicators
    if f1_custom_colorbar and metric_to_color == 'f1_score':
        # Create custom colorbar for F1 scores
        import matplotlib.colors as mcolors
        from matplotlib.patches import Rectangle

        # Create the colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)

        # Remove default ticks
        cbar.set_ticks([])

        # Add custom ticks for F1 score
        # Add tick for failed (zeros)
        cbar.ax.plot([0, 1], [-1, -1], 'k-', linewidth=2, transform=cbar.ax.transData)
        cbar.ax.text(1.3, -1, 'Failed\n(0.000)', transform=cbar.ax.transData,
                    va='center', fontsize=8, color='darkred', weight='bold')

        # Add ticks for successful range
        tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
        tick_labels = []
        for pos in tick_positions:
            # Map position back to actual F1 value
            actual_value = f1_colorbar_info['min_non_zero'] + pos * (f1_colorbar_info['max_non_zero'] - f1_colorbar_info['min_non_zero'])
            tick_labels.append(f'{actual_value:.4f}')
            # Draw tick mark
            cbar.ax.plot([0, 0.2], [pos, pos], 'k-', linewidth=1, transform=cbar.ax.transData)

        # Add labels for the successful range
        for pos, label in zip(tick_positions, tick_labels):
            cbar.ax.text(1.3, pos, label, transform=cbar.ax.transData,
                        va='center', fontsize=8)

        # Add separator line between failed and successful
        cbar.ax.plot([0, 1], [-0.8, -0.8], 'k--', linewidth=1, alpha=0.5, transform=cbar.ax.transData)

        # Add title
        cbar_label = 'F1 Score'
        cbar.ax.text(0.5, 1.1, 'Better →', transform=cbar.ax.transAxes,
                    ha='center', fontsize=9, weight='bold', color='green')
        cbar.ax.text(0.5, -0.35, 'Worse →', transform=cbar.ax.transAxes,
                    ha='center', fontsize=9, color='red')
    else:
        # Standard colorbar for other metrics
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)

        # Format colorbar label based on metric type
        # Metrics where higher is better
        metric_lower = metric_to_color.lower()
        if ('accuracy' in metric_lower or 'precision' in metric_lower or
            'recall' in metric_lower or 'f1' in metric_lower or
            'generalization_score' in metric_lower):
            cbar_label = f'{metric_to_color.replace("_", " ").title()} →'
            cbar.ax.text(0.5, 1.05, 'Better', transform=cbar.ax.transAxes,
                        ha='center', fontsize=9, weight='bold')
            cbar.ax.text(0.5, -0.05, 'Worse', transform=cbar.ax.transAxes,
                        ha='center', fontsize=9)
        else:  # For loss metrics and gaps, lower is better
            cbar_label = f'{metric_to_color.replace("_", " ").title()} →'
            cbar.ax.text(0.5, 1.05, 'Worse', transform=cbar.ax.transAxes,
                        ha='center', fontsize=9)
            cbar.ax.text(0.5, -0.05, 'Better', transform=cbar.ax.transAxes,
                        ha='center', fontsize=9, weight='bold')

    cbar.set_label(cbar_label, fontsize=11, fontweight='bold')
    
    # Add informative title
    title_text = 'Hyperparameter Search Results - Parallel Coordinates Plot\n'
    if len(df) > 0:
        title_text += f'({len(df)} experiments, colored by {metric_to_color.replace("_", " ").title()})'
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    # Add appropriate legend based on visibility mode
    if equal_visibility:
        # Equal visibility mode - no legend needed since all experiments have same visual weight
        pass
    else:
        # Original legend for highlighting mode
        if highlight_top_n > 0 and top_indices:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=cmap(norm(np.max(color_values))), linewidth=2.5,
                      label=f'Top {min(highlight_top_n, len(top_indices))} experiments'),
                Line2D([0], [0], color='gray', linewidth=0.8, alpha=0.3,
                      label='Other experiments')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Always annotate the best experiment (useful information regardless of visibility mode)
    if top_indices and len(top_indices) > 0 and top_indices[0] < len(df):
        best_exp = df.iloc[top_indices[0]]
        annotation_text = 'Best Config:\n'
        for param in param_cols:
            if param in best_exp:
                val = best_exp[param]
                if param == 'learning_rate':
                    annotation_text += f'LR: {val:.4f}\n'
                elif param == 'batch_size':
                    annotation_text += f'Batch: {int(val)}\n'
                elif param == 'epochs':
                    annotation_text += f'Epochs: {int(val)}\n'
        # Add best metric value
        if metric_to_color in best_exp:
            metric_val = best_exp[metric_to_color]
            if 'accuracy' in metric_to_color:
                annotation_text += f'\n{metric_to_color.replace("_", " ").title()}: {metric_val*100:.1f}%'
            else:
                annotation_text += f'\n{metric_to_color.replace("_", " ").title()}: {metric_val:.4f}'

        # Place annotation in upper right
        ax.text(0.98, 0.98, annotation_text,
               transform=ax.transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='lightyellow',
                                   edgecolor='orange',
                                   alpha=0.9))
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.2)
    ax.set_facecolor('#f8f8f8')
    
    # Add clear separator between parameters and metrics
    if param_cols and display_metrics:
        separator_pos = len(param_cols) - 0.5
        ax.axvline(x=separator_pos, color='darkred', alpha=0.7, linestyle='--', linewidth=2)

        # Add background shading for different sections
        ax.axvspan(-0.5, separator_pos, alpha=0.02, color='blue', zorder=0)
        ax.axvspan(separator_pos, len(display_cols) - 0.5, alpha=0.02, color='green', zorder=0)

        # Add section labels
        ax.text(len(param_cols)/2 - 0.5, 1.12, '← Hyperparameters →',
               transform=ax.get_xaxis_transform(), ha='center',
               fontsize=11, color='darkblue', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.text((len(param_cols) + len(display_cols))/2 - 0.5, 1.12, '← Metrics →',
               transform=ax.get_xaxis_transform(), ha='center',
               fontsize=11, color='darkgreen', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Parallel coordinates plot saved to {output_file}")
    else:
        plt.show()