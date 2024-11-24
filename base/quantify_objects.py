"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: October 22, 2024
"""
import os
import warnings
import numpy as np
import pandas as pd
import pickle
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import remove_small_objects
warnings.filterwarnings("ignore")

def quantify_objects(pthDL, quantpath, tissue):
    """
    Quantifies the connected components in each classified image (for the specified annotation class) located in the
    specified quantpath, and writes the results to separate CSV files for each class in the tissue list.

    Inputs:
    - pthDL (str): Path to the directory containing the model data file 'net.pkl'.
    - quantpath (str): Path to the directory containing the images to be analyzed.
    - tissue (list of int): List of annotation class labels to be analyzed.

    Returns:
    .csv files with the object analysis results for each annotation class in the tissue list.
    """

    # Load model data
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        classNames = data['classNames']

    # Start object analysis
    files = [os.path.join(quantpath, f) for f in os.listdir(quantpath) if f.endswith('.tif')]
    print('_______Starting object analysis________')

    class_name = classNames[tissue-1]
    csv_file = os.path.join(quantpath, f'{class_name}_count_analysis.csv')
    all_props = []

    for im_path in files:
        print(f'Processing image {os.path.basename(im_path)}')
        # Load image
        img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

        print(f'  Analyzing annotation class: {class_name}')
        label_mask = img == tissue
        # Label objects
        labeled = label(label_mask, connectivity=1)
        labeled = remove_small_objects(labeled, min_size=500, connectivity=1)

        # Get object sizes using np.bincount
        object_sizes = np.bincount(labeled.ravel())[1:]  # Exclude background (label 0)
        for object_ID, size in enumerate(object_sizes, start=1):
            if size >= 500:  # Ensure only objects larger than 500 pixels are included
                all_props.append([os.path.basename(im_path), f"{class_name} {object_ID}", size])

    # Convert to DataFrame
    props_df = pd.DataFrame(all_props, columns=["Image", "Object ID", "Object Size (pixels)"])
    print(f'DataFrame to be written for {class_name}:\n{props_df}')

    # Write to CSV
    try:
        props_df.to_csv(csv_file, mode='w', header=True, index=False)
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
    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\mouse lung\test python TA 205\10_22_2024'
    quantpath = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\mouse lung\test python TA 205\5x\classification_10_22_2024'
    tissue = [4]  # array with the annotation label that we want to quantify
    quantify_objects(pthDL, quantpath, tissue)