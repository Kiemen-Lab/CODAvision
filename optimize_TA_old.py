import os
import numpy as np
from skimage import io, morphology
from skimage.morphology import remove_small_objects
import cv2

def optimize_TA_old(pth, imnm): # pth: path to image folder\        #imnm: image name
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
    min_area = 4
    TA = remove_small_objects(TA, min_size=min_area)
    cv2.imwrite(os.path.join(outpth, imnm + '.tif'), TA.astype(np.uint8))
    return im0, TA, outpth

imnm = '2024-02-26 10.36.39'
pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Python tests'