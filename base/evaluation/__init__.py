"""
Evaluation utilities for CODAvision.

This package provides functionality for evaluating and analyzing segmentation results.
"""

from .confusion_matrix import ConfusionMatrixVisualizer, plot_confusion_matrix
from .testing import SegmentationModelTester, test_segmentation_model
from .quantification import ImageQuantifier, quantify_images