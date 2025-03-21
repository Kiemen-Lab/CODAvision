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

Updated: March 21, 2025
"""

import os
import numpy as np
from base import *

# Set up data paths
pth = '/Users/tnewton3/Desktop/liver_tissue_data'
pthim = os.path.join(pth, '10x')  # Path to 10x magnification images
umpix = 1  # Microns per pixel
pthtest = os.path.join(pth, 'testing_image')  # Path to test dataset
pthtestim = os.path.join(pthtest, '10x')  # Test images at 10x magnification
nm = 'test_model'  # Model name
resolution = '10x'  # Image resolution/magnification

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
save_model_metadata(pthDL, pthim, WS, nm, umpix, cmap, sxy, classNames, ntrain, nvalidate)

# Step 2: Load annotation data from existing annotations
[ctlist0, numann0, create_new_tiles] = load_annotation_data(pthDL, pth, pthim, classCheck)

# Step 3: Create training tiles from the annotations
create_training_tiles(pthDL, numann0, ctlist0, create_new_tiles)

# Step 4: Train the segmentation model using CNNs
train_segmentation_model_cnns(pthDL, retrain_model=True)

# Step 5: Test the trained model on separate test images
test_segmentation_model(pthDL, pthtest, pthtestim)