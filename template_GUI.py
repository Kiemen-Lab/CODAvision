"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: November 15, 2024
"""
import os.path
import pickle
from base import *
from CODAGUI_fend import MainWindow
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
determine_optimal_TA(pthim, nTA)

# 2 load and format annotations from each annotated image
[ctlist0, numann0] = load_annotation_data(pthDL, pth, pthim)

# 3 Make training & validation tiles for model training
create_training_tiles(pthDL, numann0, ctlist0)

# 4 Train model
train_segmentation_model_cnns(pthDL)

# 5 Test model
test_segmentation_model(pthDL, pthtest, pthtestim)

# 6 Classify images with pretrained model
classify_images(pthim, pthDL)

# 7 Quantify images
#quantify_images(pthDL, pthim)

# 8 Object count analysis if annotation classes were selected
pickle_path = os.path.join(pthDL, 'net.pkl')
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

final_df = data['final_df']
model_name = data['nm']
quantpath = os.path.join(pthim, model_name)

# Identify annotation classes for component analysis
tissue = [index + 1 for index, row in final_df.iterrows() if row['Component analysis']]

# Check if the tissue list has elements
if tissue:
    # Call the quantify_objects function
    quantify_objects(pthDL, quantpath, tissue)