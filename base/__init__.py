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

__version__ = '1.0.0'

from .models.utils import save_model_metadata
from .utils.metadata import save_model_metadata_gui
from .determine_optimal_TA import determine_optimal_TA
from .data.tiles import create_training_tiles
from .evaluation.image_quantification import quantify_images
from .evaluation.object_quantification import quantify_objects
from .evaluation.reporting import create_output_pdf
from .image.wsi import WSI2tif, WSI2png
from .data.annotation import (
    load_annotation_data,
    save_annotation_mask,
    format_white,
    save_bounding_boxes,
    calculate_tissue_mask,
    extract_annotation_layers,
)

from .models.training import train_segmentation_model_cnns
from .evaluation.testing import test_segmentation_model
from .image.classification import classify_images
from .image.augmentation import augment_annotation
from .image.utils import create_overlay

__all__ = [
    'load_annotation_data',
    'save_model_metadata',
    'save_model_metadata_gui',
    'determine_optimal_TA',
    'create_training_tiles',
    'train_segmentation_model_cnns',
    'test_segmentation_model',
    'classify_images',
    'quantify_images',
    'quantify_objects',
    'create_output_pdf',
    'WSI2tif',
    'WSI2png',
    'save_annotation_mask',
    'format_white',
    'save_bounding_boxes',
    'calculate_tissue_mask',
    'extract_annotation_layers'
]