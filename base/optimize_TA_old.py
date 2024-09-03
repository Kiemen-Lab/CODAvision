import os
import numpy as np
from skimage import morphology
import cv2
from scipy.ndimage import label
import time

def optimize_TA_old(pth, imnm):
    """
               Reads an image and returns the image as a numpy array and a binary copy of the image's green value after it has been thresholded.
               Parameters:
               - pth (str): The path where the images are located.
               - imnm (str): The name of the image.
               Returns:
               - im0 (np.ndarray): The image as a numpy array.
               - TA (np.ndarray): The binary copy of the image's green value.
               - outpth (str): The path where TA has been saved.
        """
    outpth = os.path.join(pth.rstrip('\\'), 'TA')
    if not os.path.isdir(outpth):
        os.mkdir(outpth)
    try:
        #im0 = io.imread(os.path.join(pth, imnm + '.tif'))
        im0 = cv2.imread(os.path.join(pth, imnm + '.tif'))
        im0 = im0[:, :, ::-1]  # cv2.imread() reads image in BGR order, so we have to reorder color channels
    except:
        try:
            #im0 = io.imread(os.path.join(pth, imnm + '.jp2'))
            im0 = cv2.imread(os.path.join(pth, imnm + '.jp2'))
            im0 = im0[:, :, ::-1]  # cv2.imread() reads image in BGR order, so we have to reorder color channels
        except:
            #im0 = io.imread(os.path.join(pth, imnm + '.jpg'))
            im0 = cv2.imread(os.path.join(pth, imnm + '.jpg'))
            im0 = im0[:, :, ::-1]  # cv2.imread() reads image in BGR order, so we have to reorder color channels
    if os.path.isfile(os.path.join(outpth, imnm + '.tif')):
        TA = cv2.imread(os.path.join(outpth, imnm + '.tif'), cv2.IMREAD_GRAYSCALE)
        print('Existing TA loaded')
        return im0, TA, outpth

    TA = im0[:, :, 1] < 210
    kernel_size = 3
    kernel = morphology.disk(kernel_size)
    TA = morphology.binary_closing(TA, kernel)
    TA = label(TA, np.ones((3, 3)))[0] >= 4
    cv2.imwrite(os.path.join(outpth, imnm + '.tif'), TA.astype(np.uint8))
    return im0, TA, outpth

# Example usage:
if __name__ == "__main__":
    TA_start = time.time()
    imnm = 'SG_014_0016'
    pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\Jaime\Optimize\5x'
    optimize_TA_old(pth, imnm)
    print(time.time() - TA_start)