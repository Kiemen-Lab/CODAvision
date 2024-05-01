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
from scipy import ndimage
from skimage.measure import label


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
        None. The function saves the bounding box tiles and a pickle file containing annotation information.

    """
    print('4. of 4. Creating bounding box tiles of all annotations')
    try:
        imlabel = np.array(Image.open(os.path.join(outpth, 'view_annotations.tif')))
    except:
        imlabel = np.array(Image.open(os.path.join(outpth, 'view_annotations_raw.tif')))

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
    tmp = ndimage.binary_closing(imlabel > 0, structure=morphology.disk(10))
    # Fill holes in the binary image
    tmp = ndimage.binary_fill_holes(tmp)
    # Remove small objects with less than 300 pixels
    tmp = morphology.remove_small_objects(tmp, min_size=300)

    L = label(tmp)
    numann = np.zeros((np.max(L) + 1, numclass))

    for pk in range(1, np.max(L) + 1):

        # Create a binary mask for the current component
        tmp = (L == pk).astype(float)
        a = np.sum(tmp, axis=1)
        b = np.sum(tmp, axis=0)
        rect = [np.nonzero(b)[0][0], np.nonzero(b)[0][-1], np.nonzero(a)[0][0], np.nonzero(a)[0][-1]]

        # Crop the binary mask to the region defined by the bounding box
        tmp = tmp[rect[2]:rect[3] + 1, rect[0]:rect[1] + 1]

        # Make label and image bounding boxes
        tmplabel = imlabel[rect[2]:rect[3] + 1, rect[0]:rect[1] + 1] * tmp
        tmpim = I0[rect[2]:rect[3] + 1, rect[0]:rect[1] + 1, :]

        nm = str(pk).zfill(5)
        Image.fromarray(tmpim.astype(np.uint8)).save(os.path.join(pthim, f'{nm}.tif'))
        Image.fromarray(tmplabel.astype(np.uint8)).save(os.path.join(pthlabel, f'{nm}.tif'))

        for anns in range(numclass):
            numann[pk, anns] = np.sum(tmplabel == anns)

    ctlist = [f for f in os.listdir(pthim) if f.endswith('.tif')]
    bb = 1  # indicate that xml file is fully analyzed

    annotations_file = os.path.join(outpth, 'annotations.pkl')
    data = {'numann': numann, 'ctlist': ctlist, 'bb': bb}  # create dictionary to append data on annotation.pkl file

    # Save data on .pkl file
    if os.path.join(outpth, 'annotations.pkl'):
        with open(annotations_file, 'ab') as f:  # append new data if the file already exists
            pickle.dump(data, f)
    else:
        with open(annotations_file, 'wb') as f:  # save data to new file 'write binary mode'
            pickle.dump(data, f)

    return numann, ctlist
# # Example usage:
# if __name__ == '__main__':
#     from calculate_tissue_mask import calculate_tissue_mask
#
#     # Inputs
#     WS = [[2, 0, 0, 1, 0, 0, 2, 0, 2, 2, 2, 0, 0], [7, 6], [1, 2, 3, 4, 5, 6, 7, 8, 7, 9, 10, 8, 11],
#           [6, 5, 4, 11, 1, 2, 3, 8, 10, 12, 13, 7, 9], []]
#     imnm = 'SG_013_0061'
#     pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\5x'
#     I0, TA, _ = calculate_tissue_mask(pth, imnm)
#     model_name = '02_23_2024'
#     numclass = max(WS[3])
#     print(f'numclass: {numclass}')
#     pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model'
#     outpth = os.path.join(pth, 'data', imnm, '')
#     print(f'outpth: {outpth}')
#
#     # Function
#     save_bounding_boxes(I0, outpth, model_name, numclass)