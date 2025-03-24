
"""
Author: Jaime Gomez (Johns Hopkins - Wirtz/Kiemen Lab)
Date: April 24, 2024
"""
import os
import numpy as np
from skimage import morphology
import cv2
import pickle as pkl

def calculate_tissue_mask(pth, imnm):
    """
        Reads an image and returns the image as a numpy array and a binary copy of the image's green value after it has been thresholded.

        Parameters:
            - pth (str): The path where the images are located.
            - imnm (str): The name of the image.

        Returns:
            - im0 (np.ndarray): The image as a numpy array.
            - TA (np.ndarray): The binary copy of the image's green value after it has been thresholded.
            - outpth (str): The path where TA has been saved.
           """
    outpth = os.path.join(pth.rstrip(os.path.sep), 'TA')
    if not os.path.isdir(outpth): #Check if the path to the image exists
        os.mkdir(outpth)
    try:
        im0 = cv2.imread(os.path.join(pth, imnm + '.tif'))
        im0 = im0[:,:,::-1] # cv2.imread() reads image in BGR order, so we have to reorder color channels
    except:
        try:
            #im0 = io.imread(os.path.join(pth, imnm + '.jpg')) #Check for jpg images
            im0 = cv2.imread(os.path.join(pth, imnm + '.jpg'))
            im0 = im0[:, :, ::-1]  # cv2.imread() reads image in BGR order, so we have to reorder color channels
        except:
            try:
                #im0 = io.imread(os.path.join(pth, imnm + '.jp2')) #Check for jp2 images
                im0 = cv2.imread(os.path.join(pth, imnm + '.jp2'))
                im0 = im0[:, :, ::-1]  #
            except:
                #im0 = io.imread(os.path.join(pth, imnm + '.png'))  # Check for png images
                im0 = cv2.imread(os.path.join(pth, imnm + '.png'))
                im0 = im0[:, :, ::-1]
    if os.path.isfile(os.path.join(outpth, imnm + '.tif')): # If there already is an TA image in the outpth, load it and return
        TA = cv2.imread(os.path.join(outpth, imnm + '.tif'), cv2.IMREAD_GRAYSCALE)
        print('  Existing TA loaded')
        return im0, TA, outpth

    print('  Calculating TA image')
    if os.path.isfile(os.path.join(outpth, 'TA_cutoff.pkl')): # Check if the TA value has already been calculated
        with open(os.path.join(outpth, 'TA_cutoff.pkl'), 'rb') as f:  #
            data = pkl.load(f)  #
            cts = data['cts']
            mode = data['mode']
        ct=0
        for i in cts:
            for j in i:
                ct += j
            ct = ct/len(i)
    else:
        # ct = 210 #If there is no previous TA value, use 210
        ct = 205

    if mode == 'H&E':
        TA = im0[:, :, 1] < ct # Threshold the image green values
    else:
        TA = im0[:, :, 1] > ct  # Threshold the image green values
    kernel_size = 3
    TA = TA.astype(np.uint8)
    kernel = morphology.disk(kernel_size)  # Larger kernel for closing
    TA = cv2.morphologyEx(TA, cv2.MORPH_CLOSE, kernel.astype(np.uint8))
    TA = morphology.remove_small_objects(TA.astype(bool), min_size=10)
    cv2.imwrite(os.path.join(outpth, imnm + '.tif'), TA.astype(np.uint8))
    return im0, TA, outpth