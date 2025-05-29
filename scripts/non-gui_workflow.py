"""
Liver Tissue Segmentation Workflow Script

This script orchestrates the complete workflow for training and testing a
tissue segmentation model without requiring a GUI. It handles data preparation,
model training, and evaluation for liver tissue image segmentation.

The workflow:
1. Sets up paths and configuration parameters
2. Saves model metadata
3. Loads annotation data
4. Creates training tiles
5. Trains the segmentation model using CNNs
6. Tests the model on separate test data

Usage:
    Run this script directly to execute the complete workflow
    python non-gui_workflow.py

Authors:
    Valentina Matos (Johns Hopkins - Kiemen/Wirtz Lab)
    Tyler Newton (JHU - DSAI)

Updated: May 2025
"""

import os
import numpy as np
import logging
from datetime import datetime

from base.models.utils import create_initial_model_metadata
from base.data.annotation import load_annotation_data
from base.data.tiles import create_training_tiles
from base.models.training import train_segmentation_model_cnns
from base.evaluation.testing import test_segmentation_model


DEBUG_MODE = False

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Create log filename with timestamp
log_filename = os.path.join(logs_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
# Configure logging to write to both console and file
if DEBUG_MODE:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Console output
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        handlers=[logging.FileHandler(log_filename)]
    )
logging.info(f"Starting tissue segmentation workflow. Log level set to {'DEBUG' if DEBUG_MODE else 'INFO'}. Logging to {log_filename}")

# Set up data paths
pth = '/Users/tnewton3/Desktop/liver_tissue_data'
pthim = os.path.join(pth, '10x')  # Path to 10x magnification images
umpix = 1  # Microns per pixel
pthtest = os.path.join(pth, 'testing_image')  # Path to test dataset
pthtestim = os.path.join(pthtest, '10x')  # Test images at 10x magnification
nm = 'test_model'  # Model name
resolution = '10x'  # Image resolution/magnification

# The WS variable below reflects these settings in the GUI:
# ------------------------------------
# Tab 2: Segmentation Settings
# PDAC: "Remove whitespace"
# bile duct: "Remove whitespace"
# vasculature: "Remove whitespace"
# hepatocyte: "Remove whitespace"
# immune: "Keep tissue and whitespace"
# stroma: "Remove whitespace"
# whitespace: "Keep tissue and whitespace"
#
# Additional settings:
# Add Whitespace to: Set to "whitespace"
# Add Non-whitespace to: Set to "stroma"
# ------------------------------------
# Tab 3: Nesting
# Arrange the layers in this specific order (top to bottom):
# whitespace
# PDAC
# immune
# vasculature
# bile duct
# hepatocyte
# stroma
# ------------------------------------
# Whitespace handling configuration
WS = [[0, 0, 0, 0, 2, 0, 2], # 0: remove whitespace, 1: keep only whitespace, 2: keep both
      [7, 6],                # Classes to which removed whitespace should be added
      [1, 2, 3, 4, 5, 6, 7], # Class renaming order
      [6, 4, 2, 3, 5, 1, 7], # Reverse priority of classes (for overlapping regions)
      []]                    # Classes to delete (empty list means keep all)

# Model and dataset parameters
numclass = max(WS[2])  # Maximum class number
sxy = 1024  # Size of image tiles (1024x1024 pixels)
pthDL = os.path.join(pth, nm)  # Path for model data storage
nblack = numclass + 1  # Index for black color in visualization
nwhite = WS[1][0]  # Index for white color in visualization

# Color map for tissue types visualization (RGB values)
cmap = np.array([[230, 190, 100], # PDAC
                  [65, 155, 210],  # bile duct
                  [145, 35, 35],   # vasculature
                  [158, 24, 118],  # hepatocyte
                  [30, 50, 50],    # immune
                  [235, 188, 215], # stroma
                  [255, 255, 255]]) # whitespace

# Class names for the tissue types being segmented
classNames = ['PDAC', 'bile duct', 'vasculature', 'hepatocyte', 'immune', 'stroma', 'whitespace']
classCheck = []  # Optional list for class validation
ntrain = 15  # Number of training images
nvalidate = np.ceil(ntrain/5)  # Number of validation images (20% of training)
numims = 2  # Number of images to process

# Step 1: Save model configuration and metadata
create_initial_model_metadata(
    pthDL=pthDL,
    pthim=pthim,
    WS=WS,
    nm=nm,
    umpix=umpix,
    cmap=cmap,
    sxy=sxy,
    classNames=classNames,
    ntrain=ntrain,
    nvalidate=nvalidate,
    pthtest=pthtest
    # model_type="DeepLabV3_plus", # Optional: specify if not default
    # batch_size=3 # Optional: specify if not default
)

# Step 2: Load annotation data from existing annotations
[ctlist0, numann0, create_new_tiles] = load_annotation_data(pthDL, pth, pthim, classCheck)

# Step 3: Create training tiles from the annotations
create_training_tiles(pthDL, numann0, ctlist0, create_new_tiles)

# Step 4: Train the segmentation model using CNNs
train_segmentation_model_cnns(pthDL, retrain_model=True)

# Step 5: Test the trained model on separate test images
test_segmentation_model(pthDL, pthtest, pthtestim)