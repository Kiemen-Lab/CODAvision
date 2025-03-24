"""
CODAvision: A Python package for computational tissue analysis and segmentation.

This package provides tools for loading, processing, and analyzing histological
images with deep learning-based segmentation methods.

Modules:
    models: Model architectures and training utilities
    data: Data handling and annotation tools
    evaluation: Metrics and testing utilities
    image: Image processing and classification
    utils: General utility functions
"""

# Package version
__version__ = '1.0.0'

# Import key components for easier access
from .save_model_metadata import save_model_metadata
from .determine_optimal_TA import determine_optimal_TA
from .create_training_tiles import create_training_tiles
from .quantify_images import quantify_images
from .quantify_objects import quantify_objects
from .create_output_pdf import create_output_pdf
from .WSI2tif import WSI2tif
from .data.annotation import (
    load_annotation_data,
    save_annotation_mask,
    format_white,
    save_bounding_boxes,
    calculate_tissue_mask,
)

from .models.training import train_segmentation_model_cnns
from .evaluation.testing import test_segmentation_model
from .image.classification import classify_images

# Make core functionality available at package level
__all__ = [
    # Core workflow functions
    'load_annotation_data',
    'save_model_metadata',
    'determine_optimal_TA',
    'create_training_tiles',
    'train_segmentation_model_cnns',
    'test_segmentation_model',
    'classify_images',
    'quantify_images',
    'quantify_objects',
    'create_output_pdf',
    'WSI2tif',

    # Annotation utilities
    'save_annotation_mask',
    'format_white',
    'save_bounding_boxes',
    'calculate_tissue_mask'
]
