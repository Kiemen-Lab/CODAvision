"""
Generate Visualization Figures from Incomplete Hyperparameter Search Results

This script generates all the same visualization figures that hyperparameter_search.py
would produce, but works with an incomplete hyperparameter search (128 out of 576
experiments completed).

Usage:
    python generate_incomplete_search_figures.py

Output:
    Creates a 'incomplete_search_results' directory containing all visualization figures:
    - hyperparameter_plots.png: Performance metrics vs each hyperparameter
    - parallel_coordinates_plot.png: Interactive visualization of all parameters
    - correlation_heatmap.png: Correlations between parameters and metrics
    - parameter_importance.png: Parameter importance ranking
    - learning_curves.png: Training/validation curves for top experiments
    - summary_dashboard.png: Comprehensive multi-panel visualization
    - summary_report.txt: Text report with best models and statistics
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path to import base modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization functions from hyperparameter_utils
from hyperparameter_utils import (
    ExperimentTracker,
    plot_hyperparameter_results,
    plot_parallel_coordinates,
    plot_correlation_heatmap,
    plot_learning_curves,
    plot_parameter_importance,
    create_summary_dashboard,
    create_summary_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_incomplete_search_data(search_dir: str) -> ExperimentTracker:
    """
    Load the incomplete hyperparameter search data into an ExperimentTracker.

    Args:
        search_dir: Directory containing the hyperparameter search results

    Returns:
        ExperimentTracker with loaded data
    """
    tracker = ExperimentTracker(search_dir)

    # Load the CSV summary if it exists
    csv_path = os.path.join(search_dir, 'results_summary.csv')
    if os.path.exists(csv_path):
        logger.info(f"Loading results from {csv_path}")
        df = pd.read_csv(csv_path)
        tracker.experiments = df.to_dict('records')
        logger.info(f"Loaded {len(tracker.experiments)} experiments from CSV")
    else:
        logger.warning("No results_summary.csv found, will load from individual experiment directories")

        # Load from individual experiment directories
        experiments = []
        exp_dirs = [d for d in os.listdir(search_dir) if d.startswith('experiment_')]

        for exp_dir in sorted(exp_dirs):
            exp_path = os.path.join(search_dir, exp_dir)
            config_path = os.path.join(exp_path, 'config.json')
            metrics_path = os.path.join(exp_path, 'metrics.json')

            if os.path.exists(config_path) and os.path.exists(metrics_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)

                    # Combine config and metrics
                    experiment = {
                        'experiment_id': config['experiment_id'],
                        'timestamp': config.get('timestamp', ''),
                        **config.get('parameters', {}),
                        **metrics
                    }
                    experiments.append(experiment)
                except Exception as e:
                    logger.warning(f"Failed to load experiment {exp_dir}: {str(e)}")

        tracker.experiments = experiments
        logger.info(f"Loaded {len(tracker.experiments)} experiments from individual directories")

    return tracker


def analyze_search_completeness(tracker: ExperimentTracker, expected_total: int = 576) -> dict:
    """
    Analyze the completeness of the hyperparameter search.

    Args:
        tracker: ExperimentTracker with loaded data
        expected_total: Expected total number of experiments

    Returns:
        Dictionary with completeness statistics
    """
    completed = len(tracker.experiments)
    percentage = (completed / expected_total) * 100

    # Analyze parameter coverage
    if tracker.experiments:
        df = pd.DataFrame(tracker.experiments)
        param_coverage = {}

        for col in ['learning_rate', 'batch_size', 'epochs', 'es_patience', 'lr_factor']:
            if col in df.columns:
                param_coverage[col] = {
                    'unique_values': df[col].nunique(),
                    'values': sorted(df[col].unique().tolist())
                }
    else:
        param_coverage = {}

    return {
        'completed_experiments': completed,
        'expected_experiments': expected_total,
        'completion_percentage': percentage,
        'parameter_coverage': param_coverage
    }


def generate_all_figures(tracker: ExperimentTracker, output_dir: str):
    """
    Generate all visualization figures from the hyperparameter search data.

    Args:
        tracker: ExperimentTracker with loaded data
        output_dir: Directory to save the figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Hyperparameter plots with individual points
    logger.info("Generating hyperparameter plots...")
    try:
        plot_hyperparameter_results(
            tracker,
            os.path.join(output_dir, 'hyperparameter_plots.png'),
            show_individual_points=True
        )
        logger.info("✓ Hyperparameter plots saved")
    except Exception as e:
        logger.error(f"Failed to generate hyperparameter plots: {str(e)}")

    # 2. Parallel coordinates plot
    logger.info("Generating parallel coordinates plot...")
    try:
        plot_parallel_coordinates(
            tracker,
            os.path.join(output_dir, 'parallel_coordinates_plot.png'),
            metric_to_color='test_accuracy',  # Use test_accuracy since all experiments have it
            highlight_top_n=10  # Highlight top 10 experiments
        )
        logger.info("✓ Parallel coordinates plot saved")
    except Exception as e:
        logger.error(f"Failed to generate parallel coordinates plot: {str(e)}")

    # 3. Correlation heatmap
    logger.info("Generating correlation heatmap...")
    try:
        plot_correlation_heatmap(
            tracker,
            os.path.join(output_dir, 'correlation_heatmap.png'),
            figsize=(12, 10)
        )
        logger.info("✓ Correlation heatmap saved")
    except Exception as e:
        logger.error(f"Failed to generate correlation heatmap: {str(e)}")

    # 4. Parameter importance plot
    logger.info("Generating parameter importance plot...")
    try:
        plot_parameter_importance(
            tracker,
            os.path.join(output_dir, 'parameter_importance.png'),
            method='correlation',
            figsize=(10, 6)
        )
        logger.info("✓ Parameter importance plot saved")
    except Exception as e:
        logger.error(f"Failed to generate parameter importance plot: {str(e)}")

    # 5. Learning curves (may not have full training history)
    logger.info("Generating learning curves...")
    try:
        plot_learning_curves(
            tracker,
            os.path.join(output_dir, 'learning_curves.png'),
            top_n=5,
            figsize=(14, 8)
        )
        logger.info("✓ Learning curves saved")
    except Exception as e:
        logger.warning(f"Failed to generate learning curves (likely no training history available): {str(e)}")

    # 6. Summary dashboard
    logger.info("Generating summary dashboard...")
    try:
        create_summary_dashboard(
            tracker,
            os.path.join(output_dir, 'summary_dashboard.png'),
            figsize=(20, 16)
        )
        logger.info("✓ Summary dashboard saved")
    except Exception as e:
        logger.error(f"Failed to generate summary dashboard: {str(e)}")

    # 7. Text summary report
    logger.info("Generating summary report...")
    try:
        report = create_summary_report(
            tracker,
            os.path.join(output_dir, 'summary_report.txt')
        )
        logger.info("✓ Summary report saved")

        # Print key findings
        print("\n" + "="*80)
        print("KEY FINDINGS FROM INCOMPLETE HYPERPARAMETER SEARCH")
        print("="*80)

        # Find best configuration
        if tracker.experiments:
            df = pd.DataFrame(tracker.experiments)
            if 'test_accuracy' in df.columns:
                best_idx = df['test_accuracy'].idxmax()
                best_exp = df.iloc[best_idx]

                print(f"\nBest Test Accuracy: {best_exp['test_accuracy']:.4f}")
                print(f"Experiment ID: {best_exp.get('experiment_id', 'N/A')}")
                print("\nBest Hyperparameters:")
                for param in ['learning_rate', 'batch_size', 'epochs', 'es_patience', 'lr_factor']:
                    if param in best_exp:
                        print(f"  {param}: {best_exp[param]}")

                # Show top 5 experiments
                print("\nTop 5 Experiments by Test Accuracy:")
                top_5 = df.nlargest(5, 'test_accuracy')[['experiment_id', 'test_accuracy',
                                                          'learning_rate', 'batch_size', 'epochs']]
                for idx, row in top_5.iterrows():
                    print(f"  {row['experiment_id']}: {row['test_accuracy']:.4f} "
                          f"(lr={row['learning_rate']}, batch={row['batch_size']}, epochs={row['epochs']})")

        print("="*80)

    except Exception as e:
        logger.error(f"Failed to generate summary report: {str(e)}")


def create_extended_report(tracker: ExperimentTracker, completeness_stats: dict, output_dir: str):
    """
    Create an extended report that includes information about the incomplete search.

    Args:
        tracker: ExperimentTracker with loaded data
        completeness_stats: Dictionary with completeness statistics
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, 'extended_summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("INCOMPLETE HYPERPARAMETER SEARCH - EXTENDED ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Completeness statistics
        f.write("-"*40 + "\n")
        f.write("SEARCH COMPLETENESS\n")
        f.write("-"*40 + "\n")
        f.write(f"Completed experiments: {completeness_stats['completed_experiments']}\n")
        f.write(f"Expected experiments: {completeness_stats['expected_experiments']}\n")
        f.write(f"Completion percentage: {completeness_stats['completion_percentage']:.1f}%\n\n")

        # Parameter coverage
        f.write("Parameter Coverage:\n")
        for param, info in completeness_stats['parameter_coverage'].items():
            f.write(f"  {param}: {info['unique_values']} unique values\n")
            f.write(f"    Values tested: {info['values']}\n")

        # Performance statistics
        if tracker.experiments:
            df = pd.DataFrame(tracker.experiments)

            f.write("\n" + "-"*40 + "\n")
            f.write("PERFORMANCE STATISTICS\n")
            f.write("-"*40 + "\n")

            for metric in ['test_accuracy', 'avg_precision', 'avg_recall']:
                if metric in df.columns:
                    f.write(f"\n{metric}:\n")
                    f.write(f"  Mean: {df[metric].mean():.4f}\n")
                    f.write(f"  Std:  {df[metric].std():.4f}\n")
                    f.write(f"  Min:  {df[metric].min():.4f}\n")
                    f.write(f"  Max:  {df[metric].max():.4f}\n")

            # Hyperparameter analysis
            f.write("\n" + "-"*40 + "\n")
            f.write("HYPERPARAMETER IMPACT ANALYSIS\n")
            f.write("-"*40 + "\n")

            if 'test_accuracy' in df.columns:
                for param in ['learning_rate', 'batch_size', 'epochs', 'es_patience', 'lr_factor']:
                    if param in df.columns:
                        grouped = df.groupby(param)['test_accuracy'].agg(['mean', 'std', 'count'])
                        f.write(f"\n{param} impact on test_accuracy:\n")
                        for val, row in grouped.iterrows():
                            f.write(f"  {val}: mean={row['mean']:.4f}, std={row['std']:.4f}, n={int(row['count'])}\n")

            # Recommendations
            f.write("\n" + "-"*40 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            f.write("Based on the incomplete search results:\n\n")

            # Find most promising hyperparameters
            if 'test_accuracy' in df.columns:
                for param in ['learning_rate', 'batch_size', 'epochs']:
                    if param in df.columns:
                        best_val = df.groupby(param)['test_accuracy'].mean().idxmax()
                        best_mean = df.groupby(param)['test_accuracy'].mean().max()
                        f.write(f"• Best {param}: {best_val} (avg accuracy: {best_mean:.4f})\n")

            f.write(f"\nNote: These recommendations are based on only {completeness_stats['completion_percentage']:.1f}% ")
            f.write("of the planned experiments.\n")
            f.write("A complete search may reveal different optimal configurations.\n")

    logger.info(f"Extended report saved to {report_path}")


def main():
    """
    Main function to generate all figures from incomplete hyperparameter search.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths - now relative to script directory
    search_dir = os.path.join(script_dir, 'hyperparameter_search_results')
    output_dir = os.path.join(script_dir, 'incomplete_search_results')

    logger.info("="*60)
    logger.info("Generating figures from incomplete hyperparameter search")
    logger.info("="*60)

    # Check if search directory exists
    if not os.path.exists(search_dir):
        logger.error(f"Search directory not found: {search_dir}")
        return

    # Load the incomplete search data
    logger.info(f"Loading data from {search_dir}...")
    tracker = load_incomplete_search_data(search_dir)

    if not tracker.experiments:
        logger.error("No experiments found in the search directory")
        return

    logger.info(f"Loaded {len(tracker.experiments)} experiments")

    # Analyze completeness
    completeness_stats = analyze_search_completeness(tracker)
    logger.info(f"Search is {completeness_stats['completion_percentage']:.1f}% complete")

    # Generate all figures
    logger.info(f"\nGenerating figures in {output_dir}...")
    generate_all_figures(tracker, output_dir)

    # Create extended report
    create_extended_report(tracker, completeness_stats, output_dir)

    logger.info("\n" + "="*60)
    logger.info("Figure generation complete!")
    logger.info(f"All figures saved to: {output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()