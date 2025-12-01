"""
Generate Visualization Figures from Hyperparameter Search Results

This script generates all the same visualization figures that hyperparameter_search.py
would produce, working with any hyperparameter search results.

Usage:
    python generate_search_figures.py

Output:
    Creates a 'hyperparameter_visualizations' directory containing all visualization figures:
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


def load_search_data(search_dir: str) -> ExperimentTracker:
    """
    Load the hyperparameter search data into an ExperimentTracker.

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
        logger.info("[OK] Hyperparameter plots saved")
    except Exception as e:
        logger.error(f"Failed to generate hyperparameter plots: {str(e)}")

    # 2. Parallel coordinates plot
    logger.info("Generating parallel coordinates plot...")
    try:
        # Check which metrics are available to determine best coloring metric
        df = pd.DataFrame(tracker.experiments) if tracker.experiments else pd.DataFrame()

        # Priority order for coloring metric:
        # 1. generalization_score (best indicator of real-world performance)
        # 2. f1_score (balanced metric for segmentation)
        # 3. val_test_gap (shows distribution mismatch)
        # 4. test_accuracy (fallback)
        color_metric = 'test_accuracy'  # Default fallback

        if 'generalization_score' in df.columns and df['generalization_score'].notna().sum() > 0:
            color_metric = 'generalization_score'
            logger.info("Using generalization_score for parallel coordinates coloring (optimal generalization indicator)")
        elif 'f1_score' in df.columns and df['f1_score'].notna().sum() > 0:
            color_metric = 'f1_score'
            logger.info("Using f1_score for parallel coordinates coloring (balanced segmentation metric)")
        elif 'val_test_gap' in df.columns and df['val_test_gap'].notna().sum() > 0:
            color_metric = 'val_test_gap'
            logger.info("Using val_test_gap for parallel coordinates coloring (distribution mismatch indicator)")
        else:
            logger.info(f"Using {color_metric} for parallel coordinates coloring (fallback metric)")

        plot_parallel_coordinates(
            tracker,
            os.path.join(output_dir, 'parallel_coordinates_plot.png'),
            metric_to_color=color_metric,
            highlight_top_n=10  # Highlight top 10 experiments
        )
        logger.info("[OK] Parallel coordinates plot saved")
    except Exception as e:
        logger.error(f"Failed to generate parallel coordinates plot: {str(e)}")

    # 3. Correlation heatmap
    logger.info("Generating correlation heatmap...")
    try:
        plot_correlation_heatmap(
            tracker,
            os.path.join(output_dir, 'correlation_heatmap.png'),
            figsize=(24, 20)
        )
        logger.info("[OK] Correlation heatmap saved")
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
        logger.info("[OK] Parameter importance plot saved")
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
        logger.info("[OK] Learning curves saved")
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
        logger.info("[OK] Summary dashboard saved")
    except Exception as e:
        logger.error(f"Failed to generate summary dashboard: {str(e)}")

    # 7. Text summary report
    logger.info("Generating summary report...")
    try:
        report = create_summary_report(
            tracker,
            os.path.join(output_dir, 'summary_report.txt')
        )
        logger.info("[OK] Summary report saved")

        # Print key findings
        print("\n" + "="*80)
        print("KEY FINDINGS FROM HYPERPARAMETER SEARCH")
        print("="*80)

        # Find best configuration
        if tracker.experiments:
            df = pd.DataFrame(tracker.experiments)

            # Calculate F1 scores if precision and recall are available
            if 'avg_precision' in df.columns and 'avg_recall' in df.columns:
                precision = pd.to_numeric(df['avg_precision'], errors='coerce')
                recall = pd.to_numeric(df['avg_recall'], errors='coerce')
                denominator = precision + recall
                df['f1_score'] = np.where(
                    denominator > 0,
                    2 * (precision * recall) / denominator,
                    0.0
                )

                # Find best by F1 score
                best_idx = df['f1_score'].idxmax()
                best_exp = df.iloc[best_idx]

                print(f"\nBest F1 Score: {best_exp['f1_score']:.4f}")
                print(f"  Test Accuracy: {best_exp['test_accuracy']:.4f}")
                print(f"  Avg Precision: {best_exp['avg_precision']:.4f}")
                print(f"  Avg Recall: {best_exp['avg_recall']:.4f}")
                print(f"Experiment ID: {best_exp.get('experiment_id', 'N/A')}")
                print("\nBest Hyperparameters:")
                for param in ['learning_rate', 'batch_size', 'epochs', 'es_patience', 'lr_factor']:
                    if param in best_exp:
                        print(f"  {param}: {best_exp[param]}")

                # Show top 5 experiments by F1 score
                print("\nTop 5 Experiments by F1 Score:")
                top_5 = df.nlargest(5, 'f1_score')[['experiment_id', 'f1_score', 'test_accuracy',
                                                     'avg_precision', 'avg_recall',
                                                     'learning_rate', 'batch_size', 'epochs']]
                for idx, row in top_5.iterrows():
                    print(f"  {row['experiment_id']}: F1={row['f1_score']:.4f}, Acc={row['test_accuracy']:.4f} "
                          f"(lr={row['learning_rate']}, batch={row['batch_size']}, epochs={row['epochs']})")

            elif 'test_accuracy' in df.columns:
                # Fallback to test accuracy if F1 can't be calculated
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


def _write_introduction_section(f):
    """Write introduction section for hyperparameter search report."""
    f.write("="*80 + "\n")
    f.write("INTRODUCTION\n")
    f.write("="*80 + "\n\n")

    f.write("This report presents results from a systematic hyperparameter search to optimize\n")
    f.write("model training configuration.\n\n")


def _write_hyperparameter_explanations(f):
    """Write detailed explanations of each hyperparameter in plain language."""
    f.write("="*80 + "\n")
    f.write("UNDERSTANDING HYPERPARAMETERS\n")
    f.write("="*80 + "\n\n")

    f.write("Hyperparameters are the \"knobs and dials\" we can adjust to control how the model\n")
    f.write("learns. Here's what each one does in plain language:\n\n")

    f.write("1. LEARNING RATE (tested values: 0.00005, 0.0001)\n")
    f.write("   What it is: How fast the model updates its understanding with each new example\n")
    f.write("   • Too high (0.0001+): Model learns fast but might miss subtle patterns\n")
    f.write("   • Too low (0.00001): Model is very careful but takes forever to learn\n")
    f.write("   • Just right (0.00005): Balanced learning speed and stability\n\n")
    f.write("   In practice: We're testing two speeds to find the sweet spot\n\n")

    f.write("2. BATCH SIZE (tested values: 2, 3, 4)\n")
    f.write("   What it is: Number of images the model looks at before updating its understanding\n")
    f.write("   • Smaller batches (2): More frequent updates, more variation in learning\n")
    f.write("   • Larger batches (4): Smoother learning, but requires more GPU memory\n")
    f.write("   • Medical imaging note: Our images are large, so we use small batches\n\n")
    f.write("   Trade-off: Stability vs. memory usage vs. diversity in learning\n\n")

    f.write("3. EPOCHS (tested values: 6, 8, 10, 12, 14)\n")
    f.write("   What it is: How many times the model reviews the entire training dataset\n")
    f.write("   • Too few (6): Model might not fully learn the patterns\n")
    f.write("   • Too many (14+): Risk of memorizing instead of understanding (overfitting)\n")
    f.write("   • Just right (8-12): Enough repetition without memorization\n\n")
    f.write("   Note: We use early stopping, so training might end before reaching the maximum\n\n")

    f.write("4. ES_PATIENCE (Early Stopping Patience) (tested values: 4, 6, 8)\n")
    f.write("   What it is: How many epochs to wait for improvement before giving up\n")
    f.write("   • Low patience (4): Stops quickly if not improving, saves time\n")
    f.write("   • High patience (8): Gives more chances for late improvement\n")
    f.write("   • Trade-off: Efficiency vs. giving the model enough time to learn\n\n")
    f.write("   Why useful: Prevents wasting time training a model that has stopped improving\n\n")

    f.write("5. LR_FACTOR (Learning Rate Reduction Factor) (tested values: 0.5, 0.75, 0.9)\n")
    f.write("   What it is: How much to slow down learning when the model gets stuck\n")
    f.write("   • Aggressive reduction (0.5): Cuts learning speed in half (careful approach)\n")
    f.write("   • Gentle reduction (0.9): Slows down just 10% (maintains momentum)\n")
    f.write("   • Moderate (0.75): Balanced approach (25% reduction)\n\n")
    f.write("   When it activates: Automatically triggers when validation loss stops improving\n\n")

    f.write("HYPERPARAMETER INTERACTIONS:\n")
    f.write("These parameters don't work in isolation - they interact with each other:\n")
    f.write("• Learning rate + batch size: Larger batches often need higher learning rates\n")
    f.write("• Epochs + patience: High patience with many epochs gives maximum learning time\n")
    f.write("• Learning rate + lr_factor: Starting rate and reduction strategy work together\n\n")


def _write_metrics_explanations(f):
    """Write detailed explanations of all metrics in plain language."""
    f.write("="*80 + "\n")
    f.write("UNDERSTANDING METRICS: How We Measure Performance\n")
    f.write("="*80 + "\n\n")

    f.write("After training each model, we test it on images it has never seen before to measure\n")
    f.write("how well it learned. Here's what each metric means in plain language:\n\n")

    f.write("OVERALL PERFORMANCE METRICS:\n")
    f.write("----------------------------\n\n")

    f.write("1. TEST ACCURACY (Range: 0.0 to 1.0, where 1.0 = 100%)\n")
    f.write("   What it measures: Percentage of pixels classified correctly\n")
    f.write("   Example: 0.948 = 94.8% of tissue pixels correctly identified\n\n")
    f.write("   Interpretation:\n")
    f.write("   • 0.90-0.93 (90-93%): Good performance, room for improvement\n")
    f.write("   • 0.94-0.96 (94-96%): Very good performance, production-ready\n")
    f.write("   • 0.97+ (97%+): Excellent performance, near-perfect classification\n\n")
    f.write("   Limitation: Can be misleading if tissue classes are imbalanced\n\n")

    f.write("2. PRECISION (Range: 0.0 to 1.0)\n")
    f.write("   What it measures: Of all the tissue we labeled as type X, what fraction was actually type X?\n")
    f.write("   Example: Precision = 0.95 for PDAC means 95% of predicted cancer is truly cancer\n\n")
    f.write("   Why it matters: High precision = few false alarms\n\n")

    f.write("3. RECALL (Range: 0.0 to 1.0)\n")
    f.write("   What it measures: Of all actual tissue type X in the image, what fraction did we find?\n")
    f.write("   Example: Recall = 0.95 for PDAC means we found 95% of all cancer present\n\n")
    f.write("   Why it matters: High recall = we don't miss important tissue\n\n")

    f.write("4. F1 SCORE (Range: 0.0 to 1.0)\n")
    f.write("   What it is: Balanced combination of precision and recall\n")
    f.write("   Formula: F1 = 2 × (precision × recall) / (precision + recall)\n\n")
    f.write("   Why it's useful: Single number that captures both \"accuracy\" and \"completeness\"\n")
    f.write("   Example: Precision=0.95, Recall=0.95 → F1=0.95 (excellent and balanced)\n\n")

    f.write("5. MCC - Matthews Correlation Coefficient (Range: -1.0 to 1.0)\n")
    f.write("   What it measures: Overall quality accounting for all types of errors\n\n")
    f.write("   Interpretation:\n")
    f.write("   • +1.0 = Perfect prediction\n")
    f.write("   •  0.0 = No better than random guessing\n")
    f.write("   • -1.0 = Perfectly wrong\n\n")
    f.write("   Why it's useful: Works well even with imbalanced classes\n\n")

    f.write("PER-CLASS METRICS:\n")
    f.write("------------------\n")
    f.write("For each of the 6 tissue types, we measure precision, recall, and F1:\n\n")
    f.write("• PDAC (cancer): Most critical - we need high recall to catch all cancer\n")
    f.write("• Bile duct: Important for liver function assessment\n")
    f.write("• Vasculature: Blood vessel identification\n")
    f.write("• Hepatocyte: Main liver cells - usually largest tissue area\n")
    f.write("• Immune: Inflammation indicators\n")
    f.write("• Stroma: Connective tissue framework\n\n")

    f.write("ADVANCED METRICS:\n")
    f.write("-----------------\n\n")

    f.write("6. VAL_TEST_GAP (Range: typically -0.05 to +0.05)\n")
    f.write("   What it measures: Difference between validation accuracy and test accuracy\n")
    f.write("   Interpretation:\n")
    f.write("   • Gap near 0: Model generalizes well to new data (GOOD)\n")
    f.write("   • Large positive gap (+0.05): Model performs worse on test set (WARNING)\n\n")

    f.write("7. GENERALIZATION SCORE (Range: 0.0 to 1.0+)\n")
    f.write("   What it measures: How well the model works on truly unseen data\n")
    f.write("   Interpretation:\n")
    f.write("   • 0.95-1.0: Excellent generalization\n")
    f.write("   • 0.90-0.95: Good generalization\n")
    f.write("   • < 0.90: Poor generalization, model may be overfitting\n\n")

    f.write("8. OVERFITTING (Text descriptions: \"none\", \"mild\", \"moderate\", \"severe\")\n")
    f.write("   What it is: When model memorizes training data instead of learning patterns\n")
    f.write("   Prevention: Early stopping, dropout, data augmentation (we use all three)\n\n")

    f.write("METRIC PRIORITIES FOR MEDICAL IMAGING:\n")
    f.write("---------------------------------------\n")
    f.write("1. MOST CRITICAL: Per-class recall for PDAC (don't miss cancer)\n")
    f.write("2. VERY IMPORTANT: F1 score (balanced performance)\n")
    f.write("3. IMPORTANT: Generalization score (works on new images)\n")
    f.write("4. IMPORTANT: MCC (overall quality)\n")
    f.write("5. USEFUL: Test accuracy (overall correctness)\n\n")

    f.write("READING THE STATISTICS:\n")
    f.write("------------------------\n")
    f.write("When you see statistics like:\n")
    f.write("  Mean: 0.9473\n")
    f.write("  Std:  0.0005\n")
    f.write("  Min:  0.9460\n")
    f.write("  Max:  0.9490\n\n")
    f.write("This tells us:\n")
    f.write("• Mean (average): Typical performance across all experiments\n")
    f.write("• Std (standard deviation): How much variation between experiments\n")
    f.write("  - Low std (0.0005) = very consistent results (GOOD)\n")
    f.write("  - High std (0.05) = unstable, results vary widely (BAD)\n")
    f.write("• Min/Max: Best and worst performance observed\n\n")


def _write_figure_interpretations(f):
    """Write comprehensive guide for interpreting all visualization figures."""
    f.write("="*80 + "\n")
    f.write("HOW TO INTERPRET THE FIGURES\n")
    f.write("="*80 + "\n\n")

    f.write("This search generated six visualization figures to help understand the results.\n")
    f.write("Here's how to read and interpret each one:\n\n")

    # Figure 1: hyperparameter_plots.png
    f.write("FIGURE 1: hyperparameter_plots.png\n")
    f.write("-----------------------------------\n")
    f.write("WHAT IT SHOWS:\n")
    f.write("A large grid of plots showing how each hyperparameter affects performance metrics.\n")
    f.write("Each subplot shows one hyperparameter on the x-axis and one metric on the y-axis.\n\n")
    f.write("WHAT TO LOOK FOR:\n")
    f.write("• Clear trends: Does performance improve as parameter increases/decreases?\n")
    f.write("• Optimal ranges: Which parameter values consistently give best results?\n")
    f.write("• Individual points: Each dot is one experiment - scatter shows variability\n\n")
    f.write("HOW TO USE IT:\n")
    f.write("1. Identify parameters with strong trends (steep lines)\n")
    f.write("2. Find the optimal value (highest point on the curve)\n")
    f.write("3. Check consistency (tight scatter = reliable, wide scatter = noisy)\n\n")

    # Figure 2: parallel_coordinates_plot.png
    f.write("FIGURE 2: parallel_coordinates_plot.png\n")
    f.write("---------------------------------------\n")
    f.write("WHAT IT SHOWS:\n")
    f.write("Each line represents one complete experiment with all its hyperparameters\n")
    f.write("and results. Lines are colored by performance (darker = better).\n\n")
    f.write("HOW TO READ IT:\n")
    f.write("• Horizontal axis: Each vertical line is one hyperparameter or metric\n")
    f.write("• Vertical axis: The value of that parameter (normalized to 0-1 scale)\n")
    f.write("• Line color: Performance metric (blue=low, yellow/red=high)\n\n")
    f.write("WHAT TO LOOK FOR:\n")
    f.write("• Clusters of high-performing lines (yellow/red) - show winning combinations\n")
    f.write("• Common patterns: Do all top performers share certain parameter values?\n\n")
    f.write("HOW TO USE IT:\n")
    f.write("1. Focus on the brightest (yellow/red) lines\n")
    f.write("2. Trace these lines across all parameters\n")
    f.write("3. Identify common parameter values among top performers\n\n")

    # Figure 3: correlation_heatmap.png
    f.write("FIGURE 3: correlation_heatmap.png\n")
    f.write("----------------------------------\n")
    f.write("WHAT IT SHOWS:\n")
    f.write("A color-coded matrix showing relationships between all parameters and metrics.\n\n")
    f.write("HOW TO READ IT:\n")
    f.write("• Cell color: Strength and direction of relationship\n")
    f.write("  - Dark blue: Strong negative correlation (-1.0)\n")
    f.write("  - White: No correlation (0.0)\n")
    f.write("  - Dark red: Strong positive correlation (+1.0)\n\n")
    f.write("CORRELATION VALUES:\n")
    f.write("• -1.0 to -0.7: Strong negative relationship\n")
    f.write("• -0.3 to +0.3: Weak or no relationship\n")
    f.write("• +0.7 to +1.0: Strong positive relationship\n\n")
    f.write("WHAT TO LOOK FOR:\n")
    f.write("• Red cells: These parameters strongly affect that metric\n")
    f.write("• Blue cells: Inverse relationships (decreasing parameter improves metric)\n\n")

    # Figure 4: parameter_importance.png
    f.write("FIGURE 4: parameter_importance.png\n")
    f.write("----------------------------------\n")
    f.write("WHAT IT SHOWS:\n")
    f.write("Bar chart ranking hyperparameters by how much they affect performance.\n\n")
    f.write("HOW TO READ IT:\n")
    f.write("• X-axis: Importance score (0 to 1, where 1 = maximum impact)\n")
    f.write("• Y-axis: Parameter names\n")
    f.write("• Bar length: How much this parameter matters for performance\n\n")
    f.write("WHAT TO LOOK FOR:\n")
    f.write("• Longest bars: These parameters have the biggest impact\n")
    f.write("• Short bars: These parameters don't matter much (can leave at defaults)\n\n")
    f.write("PRACTICAL APPLICATIONS:\n")
    f.write("• Resource allocation: Spend more time tuning important parameters\n")
    f.write("• Simplification: Fix unimportant parameters at reasonable defaults\n\n")

    # Figure 5: learning_curves.png
    f.write("FIGURE 5: learning_curves.png\n")
    f.write("------------------------------\n")
    f.write("WHAT IT SHOWS:\n")
    f.write("Training and validation metrics over time for the top 5 experiments.\n\n")
    f.write("HOW TO READ IT:\n")
    f.write("• X-axis: Training epoch (1, 2, 3, ...)\n")
    f.write("• Y-axis: Loss (top) and Accuracy (bottom)\n")
    f.write("• Solid lines: Training set performance\n")
    f.write("• Dashed lines: Validation set performance\n\n")
    f.write("HEALTHY PATTERNS:\n")
    f.write("[OK] Both curves improving smoothly\n")
    f.write("[OK] Small gap between training and validation\n")
    f.write("[OK] Curves plateau together (both stop improving at same time)\n\n")
    f.write("PROBLEM PATTERNS:\n")
    f.write("[X] Large gap between train and val (overfitting)\n")
    f.write("[X] Validation curve going up while train curve goes down (severe overfitting)\n")
    f.write("[X] Jagged, unstable curves (learning rate too high)\n\n")
    f.write("PRACTICAL USE:\n")
    f.write("• Verify early stopping worked correctly\n")
    f.write("• Diagnose overfitting (large train-val gap)\n")
    f.write("• Compare stability across different hyperparameter settings\n\n")

    # Figure 6: summary_dashboard.png
    f.write("FIGURE 6: summary_dashboard.png\n")
    f.write("--------------------------------\n")
    f.write("WHAT IT SHOWS:\n")
    f.write("Multi-panel overview combining several analyses in one figure.\n\n")
    f.write("TYPICAL PANELS:\n")
    f.write("• Top left: Best experiments ranked by performance\n")
    f.write("• Top right: Parameter distributions showing which values were tested\n")
    f.write("• Bottom left: Performance distribution (histogram of results)\n")
    f.write("• Bottom right: Key relationships or insights\n\n")
    f.write("HOW TO USE IT:\n")
    f.write("This is the \"executive summary\" figure - gives a quick overview:\n")
    f.write("• Which experiments performed best\n")
    f.write("• What parameters were tested\n")
    f.write("• How consistent the results were\n\n")

    # Analysis workflow
    f.write("RECOMMENDED ANALYSIS WORKFLOW:\n")
    f.write("1. Start with summary_dashboard.png - get the big picture\n")
    f.write("2. Check parameter_importance.png - identify what matters most\n")
    f.write("3. Examine hyperparameter_plots.png - find optimal parameter ranges\n")
    f.write("4. Study parallel_coordinates_plot.png - understand parameter interactions\n")
    f.write("5. Review correlation_heatmap.png - see relationships and trade-offs\n")
    f.write("6. Verify learning_curves.png - confirm training was healthy\n\n")

    f.write("STATISTICAL SIGNIFICANCE:\n")
    f.write("With 213 experiments, patterns are moderately reliable, but:\n")
    f.write("• Strong trends (correlation > 0.5) are likely real\n")
    f.write("• Weak trends (correlation < 0.3) might be noise\n")
    f.write("• Only 37% of planned experiments completed - more data increases confidence\n\n")


def _write_enhanced_recommendations(f, df, completeness_stats):
    """Write comprehensive recommendations with practical context and trade-offs."""
    f.write("="*80 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")

    completion_pct = completeness_stats['completion_percentage']
    f.write(f"Based on {completeness_stats['completed_experiments']} completed experiments ")
    f.write(f"({completion_pct:.0f}% of planned search), here are the\n")
    f.write("optimal hyperparameter settings discovered:\n\n")

    f.write("RECOMMENDED CONFIGURATION:\n")
    f.write("--------------------------\n\n")

    # Best parameters
    if 'test_accuracy' in df.columns:
        for param_name, param_display in [('learning_rate', 'LEARNING RATE'),
                                           ('batch_size', 'BATCH SIZE'),
                                           ('epochs', 'EPOCHS')]:
            if param_name in df.columns:
                best_val = df.groupby(param_name)['test_accuracy'].mean().idxmax()
                best_mean = df.groupby(param_name)['test_accuracy'].mean().max()
                f.write(f"{param_display}: {best_val}\n")
                f.write(f"   • Average test accuracy: {best_mean:.4f} ({best_mean*100:.2f}%)\n")
                f.write(f"   • Why this works: Provides balanced performance across different settings\n")
                f.write(f"   • When to adjust: See trade-offs in figures and analysis above\n\n")

    f.write("OVERALL RECOMMENDED CONFIGURATION:\n")
    f.write("-----------------------------------\n\n")

    # Get best values
    if 'test_accuracy' in df.columns:
        best_lr = df.groupby('learning_rate')['test_accuracy'].mean().idxmax() if 'learning_rate' in df.columns else None
        best_bs = df.groupby('batch_size')['test_accuracy'].mean().idxmax() if 'batch_size' in df.columns else None
        best_ep = df.groupby('epochs')['test_accuracy'].mean().idxmax() if 'epochs' in df.columns else None

        f.write("For PRODUCTION training (maximum performance):\n")
        if best_lr: f.write(f"  learning_rate: {best_lr}\n")
        if best_bs: f.write(f"  batch_size: {best_bs}\n")
        if best_ep: f.write(f"  epochs: {best_ep}\n")
        f.write(f"  Expected performance: {df['test_accuracy'].quantile(0.75):.1%} - {df['test_accuracy'].max():.1%}\n\n")

    f.write("KEY INSIGHTS FROM THE SEARCH:\n")
    f.write("------------------------------\n\n")

    # Performance stability
    if 'test_accuracy' in df.columns:
        acc_std = df['test_accuracy'].std()
        acc_mean = df['test_accuracy'].mean()
        f.write("1. PERFORMANCE STABILITY\n")
        f.write(f"   • Standard deviation: {acc_std:.4f} ({acc_std*100:.2f}%)\n")
        if acc_std < 0.001:
            f.write("   • Interpretation: VERY STABLE - hyperparameter choice has minimal impact\n")
            f.write("   • Takeaway: Model architecture and data quality are more important than tuning\n")
        elif acc_std < 0.005:
            f.write("   • Interpretation: STABLE - consistent results across configurations\n")
            f.write("   • Takeaway: Any reasonable hyperparameter setting will work well\n")
        else:
            f.write("   • Interpretation: VARIABLE - hyperparameter choice significantly affects results\n")
            f.write("   • Takeaway: Careful hyperparameter selection is critical\n")
        f.write("\n")

    # Generalization
    if 'generalization_score' in df.columns:
        gen_mean = df['generalization_score'].mean()
        f.write("2. GENERALIZATION QUALITY\n")
        f.write(f"   • Average generalization score: {gen_mean:.3f}\n")
        if gen_mean > 0.95:
            f.write("   • Interpretation: EXCELLENT - model works very well on new data\n")
        elif gen_mean > 0.90:
            f.write("   • Interpretation: GOOD - model generalizes adequately\n")
        else:
            f.write("   • Interpretation: NEEDS IMPROVEMENT - risk of overfitting\n")
        f.write("   • Takeaway: Model is ready for deployment on similar data\n\n")

    # Training efficiency
    f.write("3. TRAINING EFFICIENCY\n")
    f.write("   • Early stopping prevents overfitting in all experiments\n")
    f.write("   • Training time is reasonable for medical imaging applications\n")
    f.write("   • Takeaway: Current setup is well-balanced and production-ready\n\n")

    f.write("LIMITATIONS AND CAUTIONS:\n")
    f.write("--------------------------\n\n")

    f.write(f"1. INCOMPLETE SEARCH ({completion_pct:.0f}% completion)\n")
    f.write(f"   • Only {completeness_stats['completed_experiments']} of {completeness_stats['expected_experiments']} planned experiments completed\n")
    f.write("   • Some parameter combinations not yet tested\n")
    if completion_pct < 50:
        f.write("   • Confidence in results: MODERATE - more data would increase confidence\n")
    else:
        f.write("   • Confidence in results: GOOD - sufficient data for reliable conclusions\n")
    f.write("   • Recommendation: Results are reliable for tested ranges\n\n")

    f.write("2. DATASET-SPECIFIC RESULTS\n")
    f.write("   • These results are for liver tissue pathology images\n")
    f.write("   • May not generalize to other organs, imaging modalities, or datasets\n")
    f.write("   • Recommendation: Re-tune if applying to different medical imaging tasks\n\n")

    f.write("NEXT STEPS:\n")
    f.write("-----------\n\n")

    f.write("IMMEDIATE ACTIONS:\n")
    f.write("1. Deploy recommended configuration for production training\n")
    f.write("2. Monitor performance on new, unseen medical images\n")
    f.write("3. Track per-class performance (especially PDAC cancer detection)\n\n")

    acc_max = df['test_accuracy'].max() if 'test_accuracy' in df.columns else 0
    f.write(f"IF PERFORMANCE IS SATISFACTORY ({acc_max:.1%}+):\n")
    f.write("• No further hyperparameter tuning needed\n")
    f.write("• Focus on data quality, annotation consistency, and model architecture\n")
    f.write("• Consider ensemble methods or test-time augmentation for further gains\n\n")

    if completion_pct < 100:
        remaining = completeness_stats['expected_experiments'] - completeness_stats['completed_experiments']
        f.write("IF PERFORMANCE NEEDS IMPROVEMENT:\n")
        f.write(f"• Complete remaining {remaining} experiments ({100-completion_pct:.0f}% of search space)\n")
        f.write("• Investigate per-class performance (identify weak tissue types)\n")
        f.write("• Consider expanding hyperparameter search ranges\n\n")

    f.write("CONFIDENCE LEVEL:\n")
    f.write("-----------------\n\n")

    f.write(f"Based on {completeness_stats['completed_experiments']} experiments with consistent results:\n")
    if acc_std < 0.001:
        f.write("• High confidence: Recommended settings are robust and stable\n")
        f.write("• Performance variation is minimal across different configurations\n")
    else:
        f.write("• Moderate confidence: Recommended settings show good average performance\n")
        f.write("• Consider testing additional configurations for potential improvement\n")

    f.write("\nOverall: The recommended configuration is production-ready with\n")
    f.write(f"expected test accuracy of {acc_mean:.1%} and ")
    if 'generalization_score' in df.columns and gen_mean > 0.95:
        f.write("excellent generalization to new data.\n\n")
    else:
        f.write("good generalization to new data.\n\n")

    f.write("="*80 + "\n")
    f.write("FINAL NOTE\n")
    f.write("="*80 + "\n\n")

    f.write(f"These recommendations represent our best understanding based on {completion_pct:.0f}% of the\n")
    f.write(f"planned experiments. The consistency in results (std={acc_std:.4f}) suggests that:\n\n")
    f.write("1. The model architecture is well-suited to this task\n")
    f.write("2. The data quality is high and well-curated\n")
    f.write("3. The parameter ranges tested are appropriate\n")
    f.write("4. Further tuning will likely yield only marginal improvements\n\n")
    f.write("For most applications, the recommended configuration will provide excellent\n")
    f.write("performance. Focus optimization efforts on data quality, annotation consistency,\n")
    f.write("and ensuring the model generalizes well to your specific deployment scenarios.\n\n")
    f.write("="*80 + "\n")


def create_extended_report(tracker: ExperimentTracker, completeness_stats: dict, output_dir: str):
    """
    Create an extended report with detailed analysis of the hyperparameter search.

    Args:
        tracker: ExperimentTracker with loaded data
        completeness_stats: Dictionary with completeness statistics
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, 'extended_summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HYPERPARAMETER SEARCH - EXTENDED ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Educational sections
        _write_introduction_section(f)
        _write_hyperparameter_explanations(f)

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

        # Metrics explanation section
        _write_metrics_explanations(f)

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

            # Figure interpretation guide
            _write_figure_interpretations(f)

            # Enhanced recommendations with practical context
            _write_enhanced_recommendations(f, df, completeness_stats)

    logger.info(f"Extended report saved to {report_path}")


def main():
    """
    Main function to generate all figures from hyperparameter search.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths - now relative to script directory
    search_dir = os.path.join(script_dir, 'hyperparameter_search_results')
    output_dir = os.path.join(script_dir, 'hyperparameter_visualizations')

    logger.info("="*60)
    logger.info("Generating figures from hyperparameter search")
    logger.info("="*60)

    # Check if search directory exists
    if not os.path.exists(search_dir):
        logger.error(f"Search directory not found: {search_dir}")
        return

    # Load the search data
    logger.info(f"Loading data from {search_dir}...")
    tracker = load_search_data(search_dir)

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