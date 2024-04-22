import os
import numpy as np
from skimage import io, morphology
import cv2
import scipy
from skimage.morphology import remove_small_objects
import pickle

def calculate_tissue_mask(pth, imnm):
    """
        Reads an image and returns the image as a numpy array and a binary copy of the image's green value after it has been thresholded.

        Parameters:
            - pth (str): The path where the images are located.
            - imnm (str): The name of the image.

        Returns:
            - im0 (np.ndarray): The image as a numpy array.
            - outpth (str): The path where TA has been saved.
           """
    outpth = os.path.join(pth.rstrip('\\'), 'TA')
    if not os.path.isdir(outpth):
        os.mkdir(outpth)
    try:
        im0 = io.imread(os.path.join(pth, imnm + '.tif'))
    except:
        try:
            im0 = io.imread(os.path.join(pth, imnm + '.jpg'))
        except:
            im0 = io.imread(os.path.join(pth, imnm + '.jp2'))
    if os.path.isfile(os.path.join(outpth, imnm + '.tif')):
        TA = cv2.imread(os.path.join(outpth, imnm + '.tif'), cv2.IMREAD_GRAYSCALE)
        print('Existing TA loaded')
        return im0, TA, outpth

    print('     Calculating TA image')
    if os.path.isfile(os.path.join(outpth, 'TA_cutoff.mat')):
        data = scipy.io.loadmat(os.path.join(outpth, 'TA_cutoff.mat'))

    # if os.path.exists(os.path.join(outpth, 'TA_cutoff.pkl')):   # todo: uncomment these two lines and delete the one avobe when TA_cutoff.plk has been created
    #     with open(os.path.join(outpth, 'TA_cutoff.pkl'), 'rb') as f:
    #         try:
    #             data = pickle.load(f)      # todo: check if bugs arise after changing .mat file for .pkl file
    #         except EOFError:
    #             existing_data = {}
    # else:
    #     print('No TA_cuttoff file found')

        cts = data['cts']
        ct=0
        for i in cts:
            for j in i:
                ct += j
            ct = ct/len(i)
    else:
        ct = 210

    TA = im0[:, :, 1] < ct
    kernel_size = 3
    kernel = morphology.disk(kernel_size)
    TA = morphology.binary_closing(TA, kernel)
    min_area = 10
    TA = remove_small_objects(TA, min_size=min_area)
    cv2.imwrite(os.path.join(outpth, imnm + '.tif'), TA.astype(np.uint8))
    return im0, TA, outpth

# imnm = '84 - 2024-02-26 10.33.40'
# pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Python tests'
# calculate_tissue_mask(pth, imnm)