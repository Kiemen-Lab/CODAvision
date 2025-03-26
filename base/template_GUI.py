"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: November 15, 2024
"""
import os.path
import shutil
import pickle
from base.CODAGUI_fend import MainWindow
import sys
from PySide6 import QtWidgets
from base.classify_im_fend import MainWindowClassify
from base import *

def CODAVision():
    # 1 Execute the GUI
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Windows")
    # Load and apply the dark theme stylesheet
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir,'dark_theme.qss'), 'r') as file:
        app.setStyleSheet(file.read())

    window = MainWindow()
    window.show()
    app.exec()
    if window.classify:
        if window.classification_source == 1:
            pkl_pth = window.pth_net
        else:
            pkl_pth = os.path.join(window.pthim, window.nm,'net.pkl')
        with open(pkl_pth, 'rb') as f:
            data = pickle.load(f)
            pthim = data['pthim']
            umpix = data['umpix']
            nm = data['nm']
            pthDL = data['pthDL']
            final_df = data['final_df']
            model_type = data['model_type']
            if umpix == 'TBD':
                scale = data['scale']
                uncomp_train = data['uncomp_train_pth']
                uncompt_test = data['uncomp_test_pth']
                create_down = data['create_down']
                downsamp_annotated = data['downsamp_annotated']
        umpix_to_resolution = {1: '10x', 2: '5x', 4: '1x'}
        resolution = umpix_to_resolution.get(umpix, 'TBD')
        if resolution == 'TBD' and create_down:
            pth = ''
            for element in pthDL.split(os.sep)[:-1]:
                pth = os.path.join(pth, element)
            window2 = MainWindowClassify(uncomp_train, nm, model_type, pth)
            window2.show()
            app.exec()
        else:
            window2 = MainWindowClassify(pthim, nm, model_type)
            window2.show()
            app.exec()

    elif window.train:
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

        scale_images = not(window.create_down)
        not_downsamp_annotated = window.downsamp_annotated_images
        # Create tiff images if they don't exist
        print(' ')
        if resolution == 'Custom':
            train_img_type = window.img_type
            test_img_type = window.test_img_type
            scale = float(window.scale)
            if not(not_downsamp_annotated):
                WSI2tif(pth, resolution, umpix, train_img_type, scale, pth)
            else:
                uncomp_pth = window.uncomp_train_pth
                uncomp_test_pth = window.uncomp_test_pth
                if not(scale_images):
                    pthim = uncomp_pth
                    pthtestim = uncomp_test_pth
                if scale_images: # Additional function i accidentally added, might include it in the future
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
        train_segmentation_model_cnns(pthDL, create_new_tiles)

        # 5 Test model
        print(' ')
        if resolution == 'Custom':
            if not(not_downsamp_annotated):
                WSI2tif(pthtest, resolution, umpix, test_img_type, scale, pthtest)
                if not os.path.isfile(os.path.join(pthtestim, 'TA', 'TA_cutoff.pkl')):
                    try:
                        os.makedirs(os.path.join(pthtestim, 'TA'), exist_ok=True)
                        shutil.copy(os.path.join(pthim, 'TA', 'TA_cutoff.pkl'),
                                    os.path.join(pthtestim, 'TA', 'TA_cutoff.pkl'))
                    except:
                        print('No TA cutoff file found, using default value')
            else:
                if not(scale_images):
                    if not os.path.isfile(os.path.join(pthtestim, 'TA', 'TA_cutoff.pkl')):
                        try:
                            os.makedirs(os.path.join(pthtestim, 'TA'), exist_ok=True)
                            shutil.copy(os.path.join(pthim, 'TA', 'TA_cutoff.pkl'),
                                        os.path.join(pthtestim, 'TA', 'TA_cutoff.pkl'))
                        except:
                            print('No TA cutoff file found, using default value')
                if scale_images:
                    WSI2tif(uncomp_test_pth, resolution, umpix, train_img_type, scale, pthtest)
                    if not os.path.isfile(os.path.join(pthtestim, 'TA', 'TA_cutoff.pkl')):
                        try:
                            os.makedirs(os.path.join(pthtestim, 'TA'), exist_ok=True)
                            shutil.copy(os.path.join(pthim, 'TA', 'TA_cutoff.pkl'),
                                        os.path.join(pthtestim, 'TA', 'TA_cutoff.pkl'))
                        except:
                            print('No TA cutoff file found, using default value')
        else:
            WSI2tif(pthtest, resolution, umpix)

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

        output_path = os.path.join(pthDL, model_type+ '_evaluation_report.pdf')
        confusion_matrix_path = os.path.join(pthDL, 'confusion_matrix_'+model_type+'.png')
        color_legend_path = os.path.join(pthDL, 'model_color_legend.jpg')
        check_annotations_path = os.path.join(pth, 'check_annotations')
        check_quant = os.path.join(quantpath, 'image_quantifications.csv')
        check_classification_path = os.path.join(pthim, 'classification_' + model_name + '_' + model_type,
                                                     'check_classification')
        create_output_pdf(output_path, pthDL, confusion_matrix_path, color_legend_path, check_annotations_path,
                          check_classification_path, check_quant)