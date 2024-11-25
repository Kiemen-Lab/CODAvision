
import os
import numpy as np
import pickle
from tifffile import imread
from skimage.morphology import remove_small_objects
from .load_annotation_data import load_annotation_data
from base.classify_images import classify_images
from PIL import Image
from .Plot_confussion_matrix import plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")

"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: November 14, 2024
"""

def read_image_as_double(file_path):
    try:
        img = Image.open(file_path)
        img = img.convert('L') # Convert image to grayscale (single-channel images)
        return np.array(img).astype(np.double) # Convert to NumPy array and cast to double
    except Exception as e:
        raise RuntimeError(f"Error reading image file {file_path}: {e}")

def test_segmentation_model(pthDL,pthtest, pthtestim, cnn_name):

    print("Testing segmentation model......")

    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        nblack = data['nblack']
        nwhite = data['nwhite']
        classNames = data['classNames']

    pthtestdata = os.path.join(pthtest, 'data py')
    load_annotation_data(pthDL, pthtest, pthtestim)

    pthclassifytest = classify_images(pthtestim, pthDL, cnn_name, color_overlay_HE=True, color_mask=False)

    classNames = classNames[:-1]
    numClasses = nblack - 1

    true_labels = []
    predicted_labels = []

    imlist = os.listdir(pthtestdata)

    for folder in imlist:
        pth_annotation_data = os.path.join(pthtestdata, folder)
        annotation_file_png = os.path.join(pth_annotation_data, 'view_annotations.png')
        annotation_file_raw_png = os.path.join(pth_annotation_data, 'view_annotations_raw.png')
        # annotation_file_png = os.path.join(pth_annotation_data, 'view_annotations.tif')
        # annotation_file_raw_png = os.path.join(pth_annotation_data, 'view_annotations_raw.tif')


        if os.path.exists(os.path.join(pth_annotation_data, 'view_annotations.png')) or os.path.exists(
                os.path.join(pth_annotation_data, 'view_annotations_raw.png')):
            try:
                # Read the image and convert to double
                if os.path.exists(annotation_file_png):
                    J0 = read_image_as_double(annotation_file_png)
                else:
                    J0 = read_image_as_double(annotation_file_raw_png)
            except RuntimeError as e:
                print(e)
                continue

            # Read the imDL image
            imDL = imread(os.path.join(pthclassifytest, folder + '.tif'))
            imDL_array = np.array(imDL)

            # Remove small pixels
            for b in range(0, int(J0.max())):
                tmp = J0 == b
                J0[J0 == b] = 0
                tmp = remove_small_objects(tmp.astype(bool), min_size=25, connectivity=2)
                J0[tmp] = b

            # Get true and predicted class at testing annotation locations
            L = np.where(J0 > 0)

            true_labels = np.concatenate((true_labels, J0[L]))
            predicted_labels = np.concatenate((predicted_labels, imDL_array[L]))


    predicted_labels = np.array(predicted_labels)
    predicted_labels[predicted_labels == nblack] = nwhite
    predicted_labels = predicted_labels



    # Normalize to the minimum number of pixels, rounded to nearest 1000
    label_counts = np.histogram(true_labels, bins=numClasses)[0]
    label_percentages = (label_counts / label_counts.max() * 100).astype(int)
    min_count = label_counts.min()


    # Display number of pixels of each class in testing
    print()
    print('Calculating total number of pixels in the testing dataset...')
    for i, count in enumerate(label_counts):
        if label_percentages[i] == 100:
            print(f"  There are {count} pixels of {classNames[i]}. This is the most common class.")
        else:
            print(f"  There are {count} pixels of {classNames[i]}, {label_percentages[i]}% of the most common class.")

    if 0 in label_counts:
        for i, count in enumerate(label_counts):
            if count == 0:
                print(f"\n No testing annotations exist for class {classNames[i]}.")
        raise ValueError("Cannot make confusion matrix. Please add testing annotations of missing class(es).")

    if min_count < 100:
        min_count = (min_count // 10) * 10
    elif min_count < 1000:
        min_count = (min_count // 100) * 100
    else:
        min_count = (min_count // 1000) * 1000

    for i, count in enumerate(label_counts):
        if count < 15000:
            print(f"\n  Only {count} testing pixels of {classNames[i]} found.")
            print("    We suggest a minimum of 15,000 pixels for a good assessment of model accuracy.")
            print("    Confusion matrix may be misleading.")

    # Normalize Pixel counts
    balanced_true_labels = []
    balanced_predicted_labels = []
    for label in np.unique(true_labels):
        indices = np.where(np.array(true_labels) == label)[0]
        selected_indices = np.random.choice(indices, min_count, replace=False)
        balanced_true_labels.extend(np.array(true_labels)[selected_indices])
        balanced_predicted_labels.extend(np.array(predicted_labels)[selected_indices])

    balanced_true_labels = np.array(balanced_true_labels)
    balanced_predicted_labels = np.array(balanced_predicted_labels)

    # Confusion matrix with equal number of pixels of each class
    confusion_data = np.zeros((int(np.max(balanced_true_labels)), int(np.max(balanced_predicted_labels))))
    for true_label in range(1, int(np.max(balanced_true_labels)) + 1):
        for pred_label in range(1, int(np.max(balanced_predicted_labels)) + 1):
            confusion_data[true_label - 1, pred_label - 1] = np.sum(
                (balanced_true_labels == true_label) &
                (balanced_predicted_labels == pred_label)
            )

    confusion_data[np.isnan(confusion_data)] = 0

    _ = plot_confusion_matrix(confusion_data, classNames, pthDL, cnn_name)
    return

