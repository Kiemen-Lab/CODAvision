"""
Evaluation utilities for CODAvision.

This package provides functionality for evaluating and analyzing segmentation results.
"""

from .confusion_matrix import ConfusionMatrixVisualizer, plot_confusion_matrix
from .testing import SegmentationModelTester, test_segmentation_model
from .image_quantification import ImageQuantifier, quantify_images
from .object_quantification import ObjectQuantifier, quantify_objects
from .visualize import plot_cmap_legend