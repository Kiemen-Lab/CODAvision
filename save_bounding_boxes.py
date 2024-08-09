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
import time
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
    open_start = time.time()
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
    #elapsed_time = time.time() - open_start
    #print(f'Open image and create directories took {np.floor(elapsed_time / 60)} minutes and {elapsed_time-60*np.floor(elapsed_time / 60)} seconds')
    # Image Processing
    processing_time = time.time()
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

    elapsed_time = time.time() - processing_time
    #print(f'Image processing took {np.floor(elapsed_time / 60)} minutes and {elapsed_time - 60 * np.floor(elapsed_time / 60)} seconds')
    L = label(tmp)

    mask_start = time.time()
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
        #Image.fromarray(tmpim.astype(np.uint8)).save(os.path.join(pthim, f'{nm}.tif'))
        Image.fromarray(tmpim.astype(np.uint8)).save(os.path.join(pthim, f'{nm}.png'))
        #Image.fromarray(tmplabel.astype(np.uint8)).save(os.path.join(pthlabel, f'{nm}.tif'))
        Image.fromarray(tmplabel.astype(np.uint8)).save(os.path.join(pthlabel, f'{nm}.png'))
        for anns in range(numclass):
            numann[count, anns] = np.sum(tmplabel == anns + 1)
        count += 1
    elapsed_time = time.time() - mask_start
    #print(f'Mask creation took {np.floor(elapsed_time / 60)} minutes and {elapsed_time - 60 * np.floor(elapsed_time / 60)} seconds')

    # ctlist = [f for f in os.listdir(pthim) if f.endswith('.tif')]

    #ct_time = time.time()
    ctlist = {
        'tile_name': [f for f in os.listdir(pthim) if f.endswith('.png')],
        'tile_pth': [os.path.dirname(os.path.join(pthim, f)) for f in os.listdir(pthim) if f.endswith('.png')]
    }
    #elapsed_time = time.time() - ct_time
    #print(f'Ct creation took {np.floor(elapsed_time / 60)} minutes and {elapsed_time-60*np.floor(elapsed_time / 60)} seconds')
    bb = 1  # indicate that xml file is fully analyzed

    annotations_file = os.path.join(outpth, 'annotations.pkl')
    data_time = time.time()
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
    elapsed_time = time.time() - data_time
    #print(f'Data save took {np.floor(elapsed_time / 60)} minutes and {elapsed_time-60*np.floor(elapsed_time / 60)} seconds')
    return numann, ctlist


# # Example usage:
# if __name__ == '__main__':
    # from calculate_tissue_mask import calculate_tissue_mask
    #
    # # Inputs
    # WS = [[2, 0, 0, 1, 0, 0, 2, 0, 2, 2, 2, 0, 0], [7, 6], [1, 2, 3, 4, 5, 6, 7, 8, 7, 9, 10, 8, 11],
    #       [6, 5, 4, 11, 1, 2, 3, 8, 10, 12, 13, 7, 9], []]
    # imnm = 'SG_013_0061'
    # pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\5x'
    # I0, TA, _ = calculate_tissue_mask(pth, imnm)
    # model_name = '04_19_2024'
    # numclass = max(WS[3])
    # print(f'numclass: {numclass}')
    # pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model'
    # outpth = os.path.join(pth, 'data py', imnm, '')
    # print(f'outpth: {outpth}')
    #
    # # Function
    # numann , ctlist = save_bounding_boxes(I0, outpth, model_name, numclass)
    # print(f'numann: {len(numann)}')
    # print(f'ctlist: {ctlist}')
    #