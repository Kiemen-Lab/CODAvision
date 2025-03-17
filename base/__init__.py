# Import  modules or functions
from .save_model_metadata import save_model_metadata
from .determine_optimal_TA import determine_optimal_TA
from .load_annotation_data import load_annotation_data
from .create_training_tiles import create_training_tiles
from .models.training import train_segmentation_model_cnns
from .evaluation.testing import test_segmentation_model
from .image.classification import classify_images
from .quantify_images import quantify_images
from .quantify_objects import quantify_objects
from .create_output_pdf import create_output_pdf
from .WSI2tif import WSI2tif

#
# Package version
__version__ = '1.0.0'