
import format_white
import os
from skimage import io
import numpy as np
from scipy.ndimage import label, binary_fill_holes
import scipy
import calculate_tissue_mask
from IPython.display import display
import pandas as pd
from skimage.morphology import remove_small_objects
import pickle

def save_annotation_mask(I,outpth,WS,umpix,TA,kpb=0):  #I is the image in matrix form
    if umpix == 100:
        umpix = 1
    elif umpix==200:
        umpix=2
    elif umpix==400:
        umpix = 4
    print('     2. of 4. Interpolating annotated regions and saving mask image')
    num = len(WS[0])
    try:
        with open(os.path.join(outpth, 'annotations.pkl'), 'rb') as f:
            data = pickle.load(f)
            xyout = data['xyout']
        if xyout.size == 0:
            J = np.zeros(I.size)
        else:
            xyout[:,2:3] = round(xyout[:,2:3]/umpix)
            if TA.size > 0:
                TA = TA > 0
                TA = remove_small_objects(TA.astype(bool), min_size=30, connectivity=2)
                TA = np.logical_not(TA)
            else:
                I = I.astype(float)
                TA1 = np.std(I[:, :, [0, 1]], 2, ddof=1)
                TA2 = np.std(I[:, :, [0, 2]], 2, ddof=1)
                TA = np.max(np.concatenate((TA1, TA2), axis=2), axis=2)
                TAa = TA < 10
                TAb = I[:, :, 1] > 210
                TA = TAa & TAb
                TA *= 255
                TA = label(TA, np.ones((3, 3)))[0] >= 5
            Ig = np.flatnonzero(TA)  # Find the linear indices of non-zero elements
            szz = TA.shape  # Get the size of the binary image
            J = [[] for _ in range(num)]

            # interpolate annotation points to make closed objects
            for k in np.unique(xyout[:, 0]):
                Jtmp = np.zeros(szz, dtype=int)
                bwtypek = Jtmp.copy()
                xyz = xyout[xyout[:, 0] == k, :]
                for pp in np.unique(xyz[:, 1]):
                    if pp == 0:
                        continue
                    cc = np.flatnonzero(xyz[:, 1] == pp)
                    xyv = np.vstack((xyz[cc, 2:4], xyz[cc[0], 2:4]))
                    dxyv = np.sqrt(np.sum((xyv[1:, :] - xyv[:-1, :]) ** 2, axis=1))
                    dxyv_nonzero = dxyv != 0
                    xyv = xyv[np.concatenate(([True], dxyv_nonzero)), :]
                    dxyv = dxyv[dxyv_nonzero]
                    dxyv = np.concatenate(([0], dxyv))
                    ssd = np.cumsum(dxyv)
                    ss0 = np.arange(1, np.ceil(ssd.max()) + 0.49, 0.49)
                    xnew = np.interp(ss0, ssd, xyv[:, 0]).round().astype(int)
                    ynew = np.interp(ss0, ssd, xyv[:, 1]).round().astype(int)
                    try:
                        indnew = np.ravel_multi_index((ynew, xnew), szz)
                    except ValueError:
                        print('annotation out of bounds')
                        continue
                    indnew = indnew[~np.isnan(indnew)].astype(int)
                    bwtypek.flat[indnew] = 1
                bwtypek = binary_fill_holes(bwtypek > 0)
                Jtmp[bwtypek == 1] = k
                if not kpb:
                    Jtmp[:400, :] = 0
                    Jtmp[:, :400] = 0
                    Jtmp[-401:, :] = 0
                    Jtmp[:, -401:] = 0
                J[int(k) - 1] = np.flatnonzero(Jtmp == k)

            del bwtypek, Jtmp, xyz
            # format annotations to keep or remove whitespace
            J, ind = format_white.format_white(J, Ig, WS, szz)
            from PIL import Image
            Image.fromarray(np.uint8(J)).save(outpth + 'view_annotations_raw.tif')
    except:
        J=np.zeros(I.size)
    return J


pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\LG HG PanIN project\Jaime\Python tests'
imnm = '2024-02-26 10.36.39'
I = io.imread(os.path.join(pth, imnm + '.tif'))