"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: October 21, 2024
"""
import os.path

from base import *
from CODAGUI_bend import MainWindow
import sys
from PySide6 import QtWidgets

# 1 Execute the GUI
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

# Load and apply the dark theme stylesheet
with open('dark_theme.qss', 'r') as file:
    app.setStyleSheet(file.read())

window = MainWindow()
window.show()
app.exec()

# Load the paths from the GUI
pth = os.path.abspath(window.ui.trianing_LE.text())
pthDL = os.path.abspath(window.get_pthDL())
pthim = os.path.abspath(window.get_pthim())
pthtest = os.path.abspath(window.ui.testing_LE.text())
pthtestim = os.path.abspath(window.get_pthtestim())
nTA = window.TA

# Determine optimal TA
determine_optimal_TA(pthim,nTA)

# 2 load and format annotations from each annotated image
[ctlist0, numann0] = load_annotation_data(pthDL, pth, pthim)

# 3 Make training & validation tiles for model training
create_training_tiles(pthDL, numann0, ctlist0)

# 4 Train model
train_segmentation_model(pthDL)

# 5 Test model
test_segmentation_model(pthDL, pthtest, pthtestim)

# 6 Classify images with pretrained model
classify_images(pthim,pthDL)

# 7 Quantify images
quantify_images(pthDL, pthim)