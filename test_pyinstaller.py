"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: June 22, 2024
"""
import os
import numpy as np
import warnings
from save_model_metadata import save_model_metadata
from load_annotation_data import load_annotation_data
from train_segmentation_model import train_segmentation_model
from create_training_tiles import create_training_tiles
from test_segmentation_model import test_segmentation_model
from classify_images import classify_images
import sys
from PySide6 import QtWidgets
from CODAGUI_fend import MainWindow
warnings.filterwarnings("ignore")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')

# Execute the GUI
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()

#_______________Variable parametrization from GUI_______________

#Paths
pth = window.ui.trianing_LE.text()
pthtest = window.ui.testing_LE.text()
model_name = window.ui.model_name.text()
resolution = window.ui.resolution_CB.currentText()
pthim = os.path.join(pth, f'{resolution}')
pthDL = os.path.join(pth, model_name)

#Tif resolution
resolution_to_umpix = {"10x": 1, "5x": 2, "16x": 4}
umpix = resolution_to_umpix.get(resolution, 2)  # Default to 2 if resolution not found

# Get the dataframe with annotation information
combined_df = window.combined_df
classNames = combined_df['Combined names'].tolist()
colormap = combined_df['Combined colors'].tolist()

#Training tile size
tile_size = window.tile_size
#Number of training tiles
ntrain = window.ntrain
nvalidate = window.nval
#Number of validations tiles
nval = window.nval
#Number of TA images to evaluate (coming sooon)
# TA = window.TA

#Create WS
df = window.df
layers_to_delete = df.index[~df['Delete layer']].tolist()
layers_to_delete = [i+1 for i in layers_to_delete] #get row index starting from 1
nesting_list = df['Nesting'].tolist()
nesting_list.reverse()
WS = [df['Whitespace Settings'].tolist(),
      [window.add_ws_to, window.add_nonws_to],
      df['Combined layers'].tolist(),
      nesting_list,
      layers_to_delete
      ]


numclass = max(WS[2])
nblack = numclass + 1;nwhite = WS[1][0]

colormap = np.array(colormap)

#Final Parameters
print('Classnames: ', classNames)
print('Colormap: ', colormap)
print(WS)

# 1 save model metadata
save_model_metadata(pthDL, pthim, WS, model_name, umpix, colormap, tile_size, classNames, ntrain, nvalidate)

# 2 load and format annotations from each annotated image
[ctlist0, numann0] = load_annotation_data(pthDL, pth, pthim)

# 3 Make training & validation tiles for model training
create_training_tiles(pthDL, numann0, ctlist0)

# 4 Train model
train_segmentation_model(pthDL)

# 5 Test model
pthtestim = os.path.join(pthtest, f'{resolution}')
test_segmentation_model(pthDL, pthtest, pthtestim)

# 6 Classify images with pretrained model
classify_images(pthim,pthDL)










