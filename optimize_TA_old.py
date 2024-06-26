import os
import numpy as np
from skimage import io, morphology
import cv2
from scipy.ndimage import label

def optimize_TA_old(pth, imnm): # pth: path to image folder    #imnm: image name
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
        im0 = io.imread(os.path.join(pth, imnm + '.tif'))
    except:
        try:
            im0 = io.imread(os.path.join(pth, imnm + '.jp2'))
        except:
            im0 = io.imread(os.path.join(pth, imnm + '.jpg'))
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
# if __name__ == "__main__":
#   imnm = '2024-02-26 10.36.39'
#   pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Python tests\5x'
#   optimize_TA_old(pth, imnm)