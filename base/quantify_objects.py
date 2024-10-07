"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: September 11, 2024
"""
import os
import warnings
import numpy as np
import pandas as pd
import pickle
import cv2
from skimage.measure import label
from skimage.morphology import remove_small_objects
warnings.filterwarnings("ignore")


def quantify_objects(pthDL, quantpath, tissue):
    """
    Quantifies the connected components in each classified image (for the specified annotation class) located in the
    specified quantpath, and writes the results to an Excel file named image_quantifications.xlsx in the same
    directory, under the sheet named 'Object analysis'. Not part of the CODA main pipeline.

    Inputs:
    - pthDL (str): Path to the directory containing the model data file 'net.pkl'.
    - quantpath (str): Path to the directory containing the images to be analyzed.
    - tissue (list of int): List of annotation class labels to be analyzed.

    Returns:
    None
    """

    # Load model data
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        classNames = data['classNames']

    # Define the path to the Excel file and the sheet name
    excel_file = os.path.join(quantpath, 'image_quantifications.xlsx')
    sheetName = 'Object analysis'

    # Check if the Excel file exists
    mode = 'w' if not os.path.exists(excel_file) else 'a'

    # Write initial information to the Excel file
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode=mode, if_sheet_exists='overlay') as writer:
        pd.DataFrame([['Model:', pthDL]]).to_excel(writer, sheet_name=sheetName, startrow=0, header=False, index=False)
        pd.DataFrame([['Image location:', quantpath]]).to_excel(writer, sheet_name=sheetName, startrow=1, header=False,
                                                                index=False)

    # Start object analysis
    files = [os.path.join(quantpath, f) for f in os.listdir(quantpath) if f.endswith('.tif')]
    print('_______Starting object analysis________')

    all_props = []

    for im_path in files:
        print(f'Processing image {os.path.basename(im_path)}')
        # Load image
        img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

        # Create a mask with the chosen annotation class to analyze
        for annotation_class in tissue:
            print(f'  Analyzing annotation class: {classNames[annotation_class - 1]}')
            label_mask = img == annotation_class
            # Label objects
            labeled = label(label_mask, connectivity=1)
            labeled = remove_small_objects(labeled, min_size=500, connectivity=1)

            # Get object sizes
            object_ID = 1
            for i in np.unique(labeled):
                if i != 0:
                    all_props.append(
                        [os.path.basename(im_path), f"{classNames[annotation_class - 1]} {object_ID}", np.sum(labeled == i)])
                    object_ID += 1

    # Convert to DataFrame
    props_df = pd.DataFrame(all_props, columns=["Image", "Object ID", "Object Size (pixels)"])
    print(f'DataFrame to be written:\n{props_df}')

    # Write to Excel
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            startrow = writer.sheets[sheetName].max_row
            print(f'Writing DataFrame to Excel at row {startrow}')
            props_df.to_excel(writer, sheet_name=sheetName, startrow=startrow, header=True, index=False)
    except PermissionError as e:
        print(f"PermissionError: {e}")
        return
    except ValueError as e:
        print(f"ValueError: {e}")
        return

    print('_______Object analysis completed________')

    return

# Example usage:
if __name__ == '__main__':
    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\mouse lung\annotations\CODA_python_09_09_2024_GPU'
    quantpath = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\mouse lung\annotations\5x\classification_CODA_python_09_09_2024_GPU'
    tissue = [4]  # array with the annotation label that we want to quantify
    quantify_objects(pthDL, quantpath, tissue)