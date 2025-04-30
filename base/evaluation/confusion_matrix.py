"""
Confusion Matrix Visualization for Semantic Segmentation Models

This module provides classes and functions for creating, visualizing, and analyzing
confusion matrices for semantic segmentation model evaluation.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated March 13, 2025
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


class ConfusionMatrixVisualizer:
    """
    Class for creating and visualizing confusion matrices for model evaluation.

    This class handles the calculation of precision, recall, and accuracy from
    raw confusion data, and provides methods for visualizing these metrics using
    customizable heatmaps.
    """

    def __init__(
            self,
            class_names: List[str],
            output_dir: str,
            model_name: str
    ):
        """
        Initialize the confusion matrix visualizer.

        Args:
            class_names: Names of the classes for the confusion matrix
            output_dir: Directory where the visualization will be saved
            model_name: Name of the model for labeling the output file
        """
        self.class_names = class_names
        self.output_dir = output_dir
        self.model_name = model_name

        # Initialize color map for the visualization
        self._initialize_color_map()

    def _initialize_color_map(self) -> None:
        """
        Initialize the color map for visualizing the confusion matrix.

        Creates a red-yellow-green color map for highlighting performance levels.
        """
        colors = np.array([
            (250, 116, 95),  # Red
            (252, 189, 189),  # Light red
            (255, 255, 133),  # Yellow
            (199, 252, 199),  # Light green
            (139, 247, 139)  # Green
        ]) / 255

        positions = [0, 0.5, 0.6, 0.75, 1]
        n_bins = 100
        cmap_name = 'red_yellow_green'

        self.color_map = LinearSegmentedColormap.from_list(
            cmap_name,
            list(zip(positions, colors)),
            N=n_bins
        )

        self.norm = plt.Normalize(60, 100)

    def calculate_metrics(
            self,
            confusion_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Calculate precision, recall, and accuracy from confusion data.

        Args:
            confusion_data: Raw confusion matrix data

        Returns:
            Tuple containing precision, recall, accuracy, and the confusion matrix with metrics
        """
        # Calculate metrics with handling for division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.diag(confusion_data) / np.sum(confusion_data, axis=0)
            recall = np.diag(confusion_data) / np.sum(confusion_data, axis=1)
            accuracy = np.sum(np.diag(confusion_data)) / np.sum(confusion_data)

        # Convert to percentages and handle NaN values
        precision = np.nan_to_num(precision * 100, nan=0.0)
        recall = np.nan_to_num(recall * 100, nan=0.0)
        accuracy = np.round(accuracy * 100, 1)

        # Round to one decimal place
        precision = np.round(precision, 1)
        recall = np.round(recall, 1)

        # Create confusion matrix with metrics
        confusion_with_metrics = np.zeros(
            (confusion_data.shape[0] + 1, confusion_data.shape[1] + 1)
        )
        confusion_with_metrics[:-1, :-1] = confusion_data
        confusion_with_metrics[:-1, -1] = recall
        confusion_with_metrics[-1, :-1] = precision
        confusion_with_metrics[-1, -1] = accuracy

        return precision, recall, accuracy, confusion_with_metrics

    def visualize(
            self,
            confusion_data: np.ndarray,
            fig_size: Tuple[int, int] = (12, 10),
            save_fig: bool = True,
            show_fig: bool = True
    ) -> np.ndarray:
        """
        Visualize the confusion matrix with precision, recall, and accuracy.

        Args:
            confusion_data: Raw confusion matrix data
            fig_size: Size of the figure (width, height)
            save_fig: Whether to save the figure to disk
            show_fig: Whether to display the figure

        Returns:
            Confusion matrix with metrics
        """
        try:
            # Calculate metrics
            precision, recall, accuracy, confusion_with_metrics = self.calculate_metrics(confusion_data)

            # Create figure
            plt.figure(figsize=fig_size)

            # Base heatmap for raw confusion data
            sns.heatmap(
                confusion_with_metrics,
                annot=True,
                fmt='g',
                cmap='Blues',
                xticklabels=self.class_names + ['RECALL'],
                yticklabels=self.class_names + ['PRECISION'],
                cbar=False
            )

            # Create mask for metrics section
            mask = np.zeros_like(confusion_with_metrics, dtype=bool)
            mask[:-1, :-1] = True

            # Overlay heatmap for metrics
            sns.heatmap(
                confusion_with_metrics,
                annot=True,
                fmt='g',
                cmap=self.color_map,
                mask=mask,
                xticklabels=self.class_names + ['RECALL'],
                yticklabels=self.class_names + ['PRECISION'],
                cbar=False,
                norm=self.norm
            )

            # Add rectangle around the accuracy value
            ax = plt.gca()
            rect = Rectangle(
                (confusion_data.shape[1], confusion_data.shape[0]),
                1, 1,
                fill=False,
                edgecolor='black',
                lw=3
            )
            ax.add_patch(rect)

            # Add labels and title
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix', fontweight='bold')

            # Rotate x-axis labels
            plt.setp(
                plt.gca().get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor"
            )
            plt.draw()

            # Position x-axis label at top
            plt.gca().xaxis.set_label_position('top')
            plt.gca().xaxis.tick_top()

            # Adjust layout
            plt.tight_layout()

            # Save figure if requested
            if save_fig:
                output_path = os.path.join(self.output_dir, f'confusion_matrix_{self.model_name}.png')
                plt.savefig(output_path)
                print(f"\nConfusion matrix saved to {output_path}")

            # Show figure if requested
            if show_fig:
                plt.show()
            else:
                plt.close()

            # Print overall accuracy
            print(f"Overall Accuracy: {accuracy}%")

            return confusion_with_metrics

        except Exception as e:
            print(f"Error visualizing confusion matrix: {str(e)}")
            raise


# For backward compatibility
def plot_confusion_matrix(
        confusion_data: np.ndarray,
        classNames: List[str],
        pthDL: str,
        cnn_name: str
) -> np.ndarray:
    """
    Plot and save a confusion matrix for model evaluation.

    This function maintains backward compatibility with the original implementation.

    Args:
        confusion_data: Raw confusion matrix data
        classNames: Names of the classes
        pthDL: Directory where the visualization will be saved
        cnn_name: Name of the model

    Returns:
        Confusion matrix with metrics
    """
    visualizer = ConfusionMatrixVisualizer(
        class_names=classNames,
        output_dir=pthDL,
        model_name=cnn_name
    )

    return visualizer.visualize(confusion_data)