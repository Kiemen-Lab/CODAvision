import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, measure

def optimize_TA_old(pth, imnm): # pth: path to image folder\        #imnm: image name
    outpth = os.path.join(pth, 'TA/')
    print('a')
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
    TA = cv2.morphologyEx(TA.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    TA = cv2.morphologyEx(TA.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, TA = cv2.threshold(TA, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(outpth, imnm + '.tif'), TA)
    return im0, TA, outpth


imnm = '84 - 2024-02-26 10.33.40'
optimize_TA_old(r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Python tests\'', imnm)