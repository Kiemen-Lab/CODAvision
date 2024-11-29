"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: November 15, 2024
"""
import os.path
import shutil
import pickle
from base import *
from CODAGUI_fend import MainWindow
import sys
from PySide6 import QtWidgets
from classify_im_fend import MainWindowClassify

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

if window.classify:
    if window.classification_source == 1:
        with open(window.pth_net, 'rb') as f:
            data = pickle.load(f)
            pthim = data['pthim']
            umpix = data['umpix']
            nm = data['nm']
            final_df = data['final_df']
            model_type = data['model_type']
        umpix_to_resolution = {1: '10x', 2: '5x', 4: '1x'}
        resolution = umpix_to_resolution[umpix]
        pth = ''
        for element in pthim.split(os.sep)[:-1]:
            pth = os.path.join(pth, element)
        window2 = MainWindowClassify(pth, resolution, nm, model_type)
        window2.show()
        app.exec()
    else:
        window2 = MainWindowClassify(window.pthim, window.resolution, window.nm, window.model_type)
        window2.show()
        app.exec()

else:
    # Load the paths from the GUI
    pth = os.path.abspath(window.ui.trianing_LE.text())
    pthDL = os.path.abspath(window.get_pthDL())
    pthim = os.path.abspath(window.get_pthim())
    pthtest = os.path.abspath(window.ui.testing_LE.text())
    pthtestim = os.path.abspath(window.get_pthtestim())
    nTA = window.TA
    umpix = window.umpix
    resolution = window.resolution
    model_type = window.model_type

    already_scaled = True
    # Create tiff images if they don't exist
    print(' ')
    if resolution == 'Custom':
        train_img_type = window.img_type
        test_img_type = window.test_img_type
        scale = float(window.scale)
        uncomp_pth = window.uncomp_train_pth
        uncomp_test_pth = window.uncomp_test_pth
        if already_scaled:
            pthim = uncomp_pth
            pthtestim = uncomp_test_pth
        if not already_scaled: # Additional function i accidentally added, might include it in the future
            WSI2tif(uncomp_pth, resolution, umpix, train_img_type, scale,pth)
    else:
        WSI2tif(pth, resolution, umpix)


    # Determine optimal TA
    determine_optimal_TA(pthim, nTA)

    # 2 load and format annotations from each annotated image
    [ctlist0, numann0, create_new_tiles] = load_annotation_data(pthDL, pth, pthim)

    # 3 Make training & validation tiles for model training
    create_training_tiles(pthDL, numann0, ctlist0, create_new_tiles)

    # 4 Train model
    train_segmentation_model_cnns(pthDL)

    # 5 Test model
    print(' ')
    WSI2tif(pthtest, resolution, umpix)
    if resolution == 'Custom':
        train_img_type = window.img_type
        test_img_type = window.test_img_type
        scale = float(window.scale)
        uncomp_pth = window.uncomp_train_pth
        uncomp_test_pth = window.uncomp_test_pth
        if already_scaled:
            if not os.path.isfile(os.path.join(pthtestim, 'TA', 'TA_cutoff.pkl')):
                try:
                    os.makedirs(os.path.join(pthtestim, 'TA'), exist_ok=True)
                    shutil.copy(os.path.join(pthim, 'TA', 'TA_cutoff.pkl'),
                                os.path.join(pthtestim, 'TA', 'TA_cutoff.pkl'))
                except:
                    print('No TA cutoff file found, using default value')
        if not already_scaled: # Additional function i accidentally added, might include it in the future
            WSI2tif(uncomp_pth, resolution, umpix, train_img_type, scale,pth)
            if not os.path.isfile(os.path.join(pthtest,'Custom_scale_'+scale, 'TA', 'TA_cutoff.pkl')):
                try:
                    os.makedirs(os.path.join(pthtest,'Custom_Scale_'+scale, 'TA'), exist_ok=True)
                    shutil.copy(os.path.join(pth, 'Custom_Scale_'+scale, 'TA', 'TA_cutoff.pkl'),
                                os.path.join(pthtest,'Custom_Scale_'+scale, 'TA', 'TA_cutoff.pkl'))
                except:
                    print('No TA cutoff file found, using default value')
    else:
        WSI2tif(pthtest, resolution, umpix)
        if not os.path.isfile(os.path.join(pthtest,resolution,'TA','TA_cutoff.pkl')):
            try:
                os.makedirs(os.path.join(pthtest,resolution,'TA'), exist_ok=True)
                shutil.copy(os.path.join(pthim,'TA','TA_cutoff.pkl'),os.path.join(pthtest,resolution,'TA','TA_cutoff.pkl'))
            except:
                print('No TA cutoff file found, using default value')

    test_segmentation_model(pthDL, pthtest, pthtestim)

    # 6 Classify images with pretrained model
    classify_images(pthim, pthDL, model_type)

    # 7 Quantify images
    quantify_images(pthDL, pthim)

    # 8 Object count analysis if annotation classes were selected
    pickle_path = os.path.join(pthDL, 'net.pkl')
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    final_df = data['final_df']
    model_name = data['nm']
    classNames = data['classNames']
    quantpath = os.path.join(pthim, 'classification_'+model_name+'_'+model_type)

    # Identify annotation classes for component analysis
    tissues = []
    count = 0
    for index, row in final_df.iterrows():
        if final_df['Delete layer'][index]:
            count += 1
        if row['Component analysis']:
            tissues.append(final_df['Combined layers'][index]-count)
    tissues = list(set(tissues))

    # Check if the tissue list has elements
    for tissue in tissues:
        if not os.path.isfile(os.path.join(quantpath, classNames[tissue-1]+'_count_analysis.csv')):
            # Call the quantify_objects function
            quantify_objects(pthDL, quantpath, tissue)
        else:
            print(f'Object quantification already done for {classNames[tissue-1]}')

    output_path = os.path.join(pthDL, model_type+ 'evaluation_report.pdf')
    confusion_matrix_path = os.path.join(pthDL, 'confusion_matrix_'+model_type+'.jpg')
    color_legend_path = os.path.join(pthDL, 'model_color_legend.jpg')
    check_annotations_path = os.path.join(pth, 'check_annotations')
    check_quant = os.path.join(quantpath, 'image_quantifications.csv')
    check_classification_path = os.path.join(pth, resolution,'classification_'+model_name+'_'+model_type, 'check_classification')
    create_output_pdf(output_path, pthDL, confusion_matrix_path, color_legend_path, check_annotations_path,
                      check_classification_path, check_quant)