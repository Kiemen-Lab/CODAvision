"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: April 30, 2024
"""

import pickle
from PIL import Image
# Disable the maximum image pixel check
Image.MAX_IMAGE_PIXELS = None
import os
from PIL import Image
import numpy as np
from skimage import morphology
from skimage.measure import label
import cv2
from concurrent.futures import ThreadPoolExecutor


def save_bounding_boxes(I0, outpth, model_name, numclass):
    """
    This function creates bounding box tiles of all annotations in an image and saves them as separate image files.
    It also saves the number of annotations for each class in a pickle file.

    Args:
        I0 (numpy.ndarray): The input image as a 3D numpy array (height, width, channels).
        outpth (str): The output directory path where the bounding box tiles and pickle file will be saved.
        model_name (str): The name of the model used to create a subdirectory for the bounding box tiles.
        numclass (int): The number of classes or annotations present in the image.

    Returns:
        Numann and ctlist

    """
    print(' 4. of 4. Creating bounding box tiles of all annotations')
    try:
        imlabel = np.array(Image.open(os.path.join(outpth, 'view_annotations.png')))
    except:
        imlabel = np.array(Image.open(os.path.join(outpth, 'view_annotations_raw.png')))

    # Create directories:
    pthbb = os.path.join(outpth, model_name + '_boundbox')
    pthim = os.path.join(pthbb, 'im')
    pthlabel = os.path.join(pthbb, 'label')

    if os.path.isdir(pthim):
        os.rmdir(pthim)
    if os.path.isdir(pthlabel):
        os.rmdir(pthlabel)

    os.makedirs(pthim)
    os.makedirs(pthlabel)

    # Image Processing

    # Perform morphological closing
    tmp = imlabel > 0
    tmp = tmp.astype(np.uint8)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # Larger kernel for closing
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel_large)
    # Fill holes in the binary image
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel_small)
    contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tmp, contours, -1, 255, thickness=cv2.FILLED)
    # Remove small objects with less than 300 pixels
    tmp = morphology.remove_small_objects(tmp.astype(bool), min_size=300)

    L = label(tmp)

    numann = np.zeros((np.max(L), numclass), dtype=np.uint32)

    def create_bounding_box(pk):
        # Create a binary mask for the current component
        tmp = (L == pk)
        a = np.sum(tmp, axis=1)
        b = np.sum(tmp, axis=0)
        rect = [np.nonzero(b)[0][0], np.nonzero(b)[0][-1], np.nonzero(a)[0][0], np.nonzero(a)[0][-1]]

        # Crop the binary mask to the region defined by the bounding box
        tmp = tmp[rect[2]:rect[3], rect[0]:rect[1]]

        # Make label and image bounding boxes
        tmplabel = imlabel[rect[2]:rect[3], rect[0]:rect[1]] * tmp
        tmpim = I0[rect[2]:rect[3], rect[0]:rect[1], :]

        nm = str(pk).zfill(5)
        # Collect images and labels for batch processing
        return nm, tmpim, tmplabel

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(create_bounding_box, pk) for pk in range(1, np.max(L) + 1)]

    count = 0


    for future in futures:
        nm, tmpim, tmplabel = future.result()

        # Save images and labels in batch if needed
        Image.fromarray(tmpim.astype(np.uint8)).save(os.path.join(pthim, f'{nm}.png'))
        Image.fromarray(tmplabel.astype(np.uint8)).save(os.path.join(pthlabel, f'{nm}.png'))

        for anns in range(numclass):
            numann[count, anns] = np.sum(tmplabel == anns + 1)
        count += 1


    ctlist = {
        'tile_name': [f for f in os.listdir(pthim) if f.endswith('.png')],
        'tile_pth': [os.path.dirname(os.path.join(pthim, f)) for f in os.listdir(pthim) if f.endswith('.png')]
    }

    bb = 1  # indicate that xml file is fully analyzed

    annotations_file = os.path.join(outpth, 'annotations.pkl')

    # Save data on .pkl file
    if os.path.join(outpth, 'annotations.pkl'):
        with open(annotations_file, 'rb') as f:
            data = pickle.load(f)
            data['numann'] = numann
            data['ctlist'] = ctlist
            data['bb'] = bb
        with open(annotations_file, 'wb') as f:
            pickle.dump(data, f)
            f.close()
    else:
        data = {'numann': numann, 'ctlist': ctlist, 'bb': bb}  # create dictionary to append data on annotation.pkl file
        with open(annotations_file, 'wb') as f:  # save data to new file 'write binary mode'
            pickle.dump(data, f)
            f.close()
    return numann, ctlist

