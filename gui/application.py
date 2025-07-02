"""
CODAvision GUI Application

This module provides the main entry point for the CODAvision GUI application,
orchestrating the workflow for model creation, training, and analysis.
"""

import os
import sys
import time
import pickle
import shutil
from PySide6 import QtWidgets, QtCore

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Import from base package
from base import (
    determine_optimal_TA, load_annotation_data, create_training_tiles,
    train_segmentation_model_cnns, test_segmentation_model, classify_images,
    quantify_images, quantify_objects, create_output_pdf, WSI2tif
)

# Import GUI components
from .components.main_window import MainWindow
from .components.classification_window import MainWindowClassify


def CODAVision():
    """
    Main entry point for the CODAvision GUI application.
    
    This function initializes the application, loads styles, and starts the GUI.
    It also handles the execution flow based on user interactions.
    """
    start_time = time.time()
    times = {}

    # Initialize Qt application
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Windows")
    
    # Load dark theme stylesheet
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'resources', 'dark_theme.qss'), 'r') as file:
        app.setStyleSheet(file.read())

    # Create and show main window
    window = MainWindow()
    window.show()
    app.exec()
    
    # Handle post-GUI actions based on user selections
    if window.classify:
        # Launch classification window for visualizing results
        if window.classification_source == 1:
            pkl_pth = window.pth_net
        else:
            pkl_pth = os.path.join(window.pthim, window.nm, 'net.pkl')
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
        # Execute training workflow
        pth = os.path.abspath(window.ui.trianing_LE.text())
        pthDL = os.path.abspath(window.get_pthDL())
        pthim = os.path.abspath(window.get_pthim())
        pthtest = os.path.abspath(window.ui.testing_LE.text())
        pthtestim = os.path.abspath(window.get_pthtestim())
        nTA = window.TA
        umpix = window.umpix
        resolution = window.resolution
        model_type = window.model_type
        redo = window.redo_TA

        scale_images = not(window.create_down)
        not_downsamp_annotated = window.downsamp_annotated_images

        logger.info("Starting image downsampling process...")
        downsamp_time = time.time()
        if resolution == 'Custom':
            # Handle custom resolution image preparation
            train_img_type = window.img_type
            test_img_type = window.test_img_type
            scale = float(window.scale)
            if not(not_downsamp_annotated):
                WSI2tif(pth, resolution, umpix, train_img_type, scale, pth)
                WSI2tif(pthtest, resolution, umpix, test_img_type, scale, pthtest)
            else:
                uncomp_pth = window.uncomp_train_pth
                uncomp_test_pth = window.uncomp_test_pth
                if not(scale_images):
                    pthim = uncomp_pth
                    pthtestim = uncomp_test_pth
                if scale_images:
                    WSI2tif(uncomp_pth, resolution, umpix, train_img_type, scale, pth)
                    WSI2tif(uncomp_test_pth, resolution, umpix, train_img_type, scale, pthtest)
        else:
            WSI2tif(pth, resolution, umpix)
            WSI2tif(pthtest, resolution, umpix)
        downsamp_time = time.time()-downsamp_time

        # Execute the model training pipeline
        # determine_optimal_TA will handle showing the tissue mask dialog if needed
        determine_optimal_TA(pthim, pthtestim, nTA, redo)
        try:
            os.makedirs(os.path.join(pthtestim, 'TA'), exist_ok=True)
            # The TA file is saved in pthim/TA/, not in the parent directory
            ta_source = os.path.join(pthim, 'TA', 'TA_cutoff.pkl')
            ta_dest = os.path.join(pthtestim, 'TA', 'TA_cutoff.pkl')
            if os.path.exists(ta_source):
                shutil.copy(ta_source, ta_dest)
                logger.debug(f'Copied TA cutoff file from {ta_source} to {ta_dest}')
            else:
                logger.debug(f'TA cutoff file not found at {ta_source}, will be created during processing')
        except Exception as e:
            logger.debug(f'Could not copy TA cutoff file: {e}')
        load_time = time.time()
        [ctlist0, numann0, create_new_tiles] = load_annotation_data(pthDL, pth, pthim)
        load_time = time.time()-load_time
        load_time = str(int(load_time // 3600)) + ':' + str(int((load_time % 3600) // 60)) + ':' + str(
            round(load_time % 60, 2))
        tiles_time = time.time()
        create_training_tiles(pthDL, numann0, ctlist0, create_new_tiles)
        tiles_time = time.time() - tiles_time
        tiles_time = str(int(tiles_time // 3600)) + ':' + str(int((tiles_time % 3600) // 60)) + ':' + str(
            round(tiles_time % 60, 2))
        train_time = time.time()
        train_segmentation_model_cnns(pthDL, create_new_tiles)
        train_time = time.time() - train_time
        train_time = str(int(train_time // 3600)) + ':' + str(int((train_time % 3600) // 60)) + ':' + str(
            round(train_time % 60, 2))

        # Prepare and process test data
        logger.info("Starting test data processing...")
        downsamp_time = str(int(downsamp_time // 3600)) + ':' + str(int((downsamp_time % 3600) // 60)) + ':' + str(
            round(downsamp_time % 60, 2))
        times['Downsampling images'] = downsamp_time
        times['Loading annotations'] = load_time
        times['Creating tiles'] = tiles_time
        times['Training model'] = train_time
        # Test, classify, and quantify results
        test_time = time.time()
        test_segmentation_model(pthDL, pthtest, pthtestim)
        test_time = time.time() - test_time
        test_time = str(int(test_time // 3600)) + ':' + str(int((test_time % 3600) // 60)) + ':' + str(
            round(test_time % 60, 2))
        times['Testing model'] = test_time
        class_time = time.time()
        classify_images(pthim, pthDL, model_type)
        class_time = time.time() - class_time
        class_time = str(int(class_time // 3600)) + ':' + str(int((class_time % 3600) // 60)) + ':' + str(
            round(class_time % 60, 2))
        times['Classifying images'] = class_time
        quant_time = time.time()
        quantify_images(pthDL, pthim)
        quant_time = time.time() - quant_time
        quant_time = str(int(quant_time // 3600)) + ':' + str(int((quant_time % 3600) // 60)) + ':' + str(
            round(quant_time % 60, 2))
        times['Quantifying images'] = quant_time

        # Perform tissue component analysis on specified tissues
        comp_time = time.time()
        pickle_path = os.path.join(pthDL, 'net.pkl')
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        final_df = data['final_df']
        model_name = data['nm']
        classNames = data['classNames']
        quantpath = os.path.join(pthim, 'classification_' + model_name + '_' + model_type)

        # Identify tissues for object quantification
        tissues = []
        count = 0
        for index, row in final_df.iterrows():
            if final_df['Delete layer'][index]:
                count += 1
            if row['Component analysis']:
                tissues.append(final_df['Combined layers'][index] - count)
        tissues = list(set(tissues))

        # Quantify objects for specified tissues
        for tissue in tissues:
            if not os.path.isfile(os.path.join(quantpath, classNames[tissue - 1] + '_count_analysis.csv')):
                quantify_objects(pthDL, quantpath, tissue)
            else:
                logger.info(f'Object quantification already done for {classNames[tissue - 1]}')
        comp_time = time.time()-comp_time
        comp_time = str(int(comp_time // 3600)) + ':' + str(int((comp_time % 3600) // 60)) + ':' + str(
            round(comp_time % 60, 2))
        times['Object quantification'] = comp_time
        total_time = time.time()-start_time
        total_time = str(int(total_time//3600))+':'+str(int((total_time%3600)//60))+':'+str(round(total_time%60,2))
        times['Total time'] = total_time

        # Create output PDF report
        output_path = os.path.join(pthDL, model_type + '_evaluation_report.pdf')
        confusion_matrix_path = os.path.join(pthDL, 'confusion_matrix_' + model_type + '.png')
        color_legend_path = os.path.join(pthDL, 'model_color_legend.jpg')
        check_annotations_path = os.path.join(pth, 'check_annotations')
        check_quant = os.path.join(quantpath, 'image_quantifications.csv')
        check_classification_path = os.path.join(pthim, 'classification_' + model_name + '_' + model_type,
                                                'check_classification')
        create_output_pdf(output_path, pthDL, confusion_matrix_path, color_legend_path, check_annotations_path,
                          check_classification_path, check_quant, times)

    end_time = time.time()
    execution_time = end_time - start_time

    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"Execution time: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")