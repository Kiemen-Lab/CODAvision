import os
from skimage import io
import numpy as np
from skimage.morphology import remove_small_objects

def save_annotation_mask(I,outpth,WS,umpix,TA,kpb=0):  #I is the image in matrix form
    if umpix == 100:
        umpix = 1
    elif umpix++200:
        umpix=2
    elif umpix==400:
        umpix = 4
    print('     2. of 4. Interpolating annotated regions and saving mask image')
    num = len(WS[0])
    try:
        loaded_data = scipy.io.loadmat(os.path.join(outpth, 'annotations.mat'))
        xyout = loaded_data['xyout']
        if xyout.size == 0:
            J = np.zeros(I.size)
        else:
            xyout[:,2:3] = round(xyout[:,2:3]/umpix)

            if TA.size == 0:
                I = I.astype(int)
                TA1 = np.std(I[:,:,[0,1]],2,ddof=1)
                TA2 = np.std(I[:,:,[0,2]],2,ddof=1)
                TA = np.max(np.concatenate((TA1,TA2),axis=2),axis=2)
                TAa = TA<10
                TAb = I[:,:,1]>210
                TA = TAa & TAb
                TA *= 255
                min_area = 5
                TA = remove_small_objects(TA, min_size=min_area)

            else:
                TA = TA>0
                min_area = 30
                TA = remove_small_objects(TA, min_size=min_area)
                TA = 255 - TA # if TA saved as uint8 (current format from python)


    except:
        J=np.zeros(I.size)


pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Python tests'
imnm = '2024-02-26 10.36.39'
I = io.imread(os.path.join(pth, imnm + '.tif'))