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
    - tissue int): int of annotation class label to be analyzed.

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
        filtered_labels = [object_ID for object_ID, size in enumerate(object_sizes, start=1) if size >= 500]

        # Sort labels by size in descending order
        sorted_labels = sorted(filtered_labels, key=lambda x: object_sizes[x - 1], reverse=True)

        # Create a new labeled image
        new_labeled = np.zeros_like(labeled)

        # Assign the labels in order of size
        for new_label, label_id in enumerate(sorted_labels, start=1):
            new_labeled[labeled == label_id] = new_label

        # Update the labeled variable to the new labeled image
        labeled = new_labeled

        # Build all_props based on the new labels
        object_sizes = np.bincount(labeled.ravel())[1:]
        for object_ID, size in enumerate(object_sizes, start=1):
            if size >= 500:  # Ensure only objects larger than 500 pixels are included
                all_props.append([os.path.basename(im_path), f"{class_name} {object_ID}", size])

        # ########## Plotting
        #
        # # Plot the labeled object as a mask on top of the original image
        # original_image_path = os.path.join(os.path.dirname(quantpath), os.path.basename(im_path))
        # original_img = cv2.imread(original_image_path)
        #
        # # Create a mask for each filtered label and overlay them with different colors
        # overlay = original_img.copy()
        # colors = plt.cm.get_cmap('tab20', len(filtered_labels))
        #
        # for idx, label_id in enumerate(filtered_labels):
        #     mask = np.zeros_like(original_img)
        #     mask[labeled == label_id] = (np.array(colors(idx)[:3]) * 255).astype(int)  # Different color for each label
        #     overlay = cv2.addWeighted(overlay, 0.7, mask, 0.3, 0)
        #
        # # Identify the biggest labeled object
        # biggest_mask = np.zeros_like(original_img)
        # biggest_mask[labeled == 1] = [255, 0, 0]  # Red color for the biggest object
        #
        # # Overlay the biggest labeled object with the number on top
        # overlay_biggest = cv2.addWeighted(original_img, 0.7, biggest_mask, 0.3, 0)
        #
        # # Plot the biggest labeled object with the number on top
        # plt.imshow(cv2.cvtColor(overlay_biggest, cv2.COLOR_BGR2RGB))
        # plt.title(f'Biggest Labeled Object {class_name}')
        # plt.axis('off')
        # plt.show()



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
    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\mouse lung\annotations\01_30_2025_metsquantification'
    quantpath = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\mouse lung\annotations\5x\classification_01_30_2025_metsquantification_DeepLabV3_plus'
    tissue = 4  # array with the annotation label that we want to quantify
    quantify_objects(pthDL, quantpath, tissue)