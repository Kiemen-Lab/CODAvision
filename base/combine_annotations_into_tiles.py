"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 20, 2024
"""

import os
import cv2
import numpy as np
from PIL import Image
from .edit_annotation_tiles import edit_annotations_tiles


def combine_annotations_into_tiles(numann0, numann, percann, imlist, nblack, pthDL, outpth, sxy, stile=10240, nbg=0):
    """
        Combine annotations into large tiles to train the deep neural network

        Inputs:
        - numann0 (numpy array): Initial An array containing the number of pixels per class per bounding box. Each row
                                 is a bounding box and each column is a class.
        - numann (numpy array): Updated numann.
        - percann (numpy array): Percentages of annotations.
        - imlist (list of dicts): List of image tiles with paths.
        - nblack (int): Number of annotation classes + 1.
        - pthDL (str): Pth to the model
        - outpth (str): Output path where the big tiles are going to be saved.
        - sxy (int): Scaling size.
        - stile (int, optional): Size of tiles. Default is 10000.
        - nbg (int, optional): Background number. Default is 0.

        Returns:
        - numann (numpy array): Updated array containing the number of pixels per class per bounding box.
        - percann (numpy array): Updated percentages of annotations.
    """
    stile += 200
    kpall = 1

    # Define folder locations
    outpthim = os.path.join(pthDL, outpth, 'im/')
    outpthlabel = os.path.join(pthDL, outpth, 'label/')
    outpthbg = os.path.join(pthDL, outpth, 'big_tiles/')

    os.makedirs(outpthim, exist_ok=True)
    os.makedirs(outpthlabel, exist_ok=True)
    os.makedirs(outpthbg, exist_ok=True)

    imlistck = [f for f in os.listdir(outpthim) if f.endswith('.png')]
    nm0 = len(imlistck) + 1

    # Create very large blank images
    imH = np.full((stile, stile, 3), nbg, dtype=np.uint8) # create an array with the specified dimensions with the value of ngb as background
    imT = np.zeros((stile, stile), dtype=np.uint8)  # blank mask
    nL = imT.size  # size of the big tile
    ct = np.zeros(numann.shape[1])  # list (size of how many annotations you have)
    sf = np.sum(ct) / nL  # how full is the big tile
    num_classes = len(ct)
    count = 1
    tcount = 1  # is not one because of python indexing starts at 0
    cutoff = 0.55  # Cut off for how full the big tile is (55% full)
    rsf = 10
    type0 = 0

    # Here comes the fun part, tiles are going to be added to the big black tile until the cutoff is achieved ~55%
    while sf < cutoff:
        # choose one of each class in order in a loop
        if count % 10 == 1:
            type_ = tcount-1
            tcount = (tcount % num_classes)+1

        # choose a tile containing the least prevalent class
        else:
            tmp = ct.copy()
            tmp[type0] = np.max(tmp)
            type_ = np.argmin(tmp)

        num = np.where(numann[:, type_] > 0)[0]
        if len(num) == 0:
            numann[:, type_] = numann0[:, type_]
            num = np.where(numann[:, type_] > 0)[0]
        num = np.random.choice(num, size=1, replace=False)
        tile_name = imlist['tile_name'][num[0]]  # chosen random tile to be processed

        tile_path = os.path.join(imlist['tile_pth'][num[0]], tile_name)
        pf = tile_path.rfind('\\', 0, tile_path.rfind('\\'))
        pthlabel = os.path.join(tile_path[0:pf], 'label')

        im = cv2.imread(tile_path)
        im = im[:, :, ::-1]  # cv2.imread() reads image in BGR order, so we have to reorder color channels
        TA = cv2.imread(os.path.join(pthlabel, tile_name), cv2.IMREAD_GRAYSCALE)

        if count % 3 == 1:
            doaug = 1
        else:
            doaug = 0

        im, TA, kp = edit_annotations_tiles(im, TA, doaug, type_, ct, imT.shape[0], kpall)
        numann[num[0], kp - 1] = 0  # kp-1 due to layer index starting in 1 and python index starting in 0
        percann[num[0], kp - 1, 0] += 1
        percann[num[0], kp - 1, 1] = 2
        fx = (TA != 0)  # linear idx of tissue pixels in the mask bb
        if np.sum(fx) < 30:
            print('skipped')
            continue
        # find low density location in large tile to add annotation
        tmp2 = imT[::rsf, ::rsf] > 0
        dist = cv2.distanceTransform((tmp2 <= 0).astype(np.uint8), cv2.DIST_L2, 3)
        dist[:(100/rsf)-1, :] = 0  # Sets the first 20 rows to 0
        dist[:, :(100/rsf)-1] = 0  # Sets the first 20 columns to 0
        dist[-100/rsf:, :] = 0  # Sets the last 20 rows to 0
        dist[:, -100/rsf:] = 0  # Sets the last 20 columns to 0
        xii = np.where(dist == np.max(dist))
        index = np.random.choice(len(xii[0]), size=1, replace=False)
        x = int(xii[0][index[0]]*rsf)
        y = int(xii[1][index[0]]*rsf)
        szz = np.array(TA.shape) - 1
        szzA = szz // 2
        szzB = szz - szzA

        # Ensure szzA and szzB are tuples of integers to be used for slicing afterward
        szzA = tuple(map(int, szzA))
        szzB = tuple(map(int, szzB))

        # x,y - Location in the big tile to add the BB to
        if x + szzA[0]+1 > imT.shape[1]:
            x -= szzA[0]
        if y + szzA[1]+1 > imT.shape[0]: #Since we use y+szzA[1]+1 as the upper bound, we should compare it to imT.shape
            y -= szzA[1]
        if x - szzB[0] < 0: #less than 0 cause indxing starts at 0 not 1 as in MATLAB
            x += szzB[0]
        if y - szzB[1] < 0:
            y += szzB[1]
        tmpT = imT[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1].copy()
        tmpT[fx] = TA[fx]
        tmpH = imH[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1, :].copy()
        tmpH[np.dstack((fx, fx, fx))] = im[np.dstack((fx, fx, fx))]

        imT[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1] = tmpT  # imT is the bigtile mask
        imH[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1, :] = tmpH  # imH is the bigtile HE

        # Update total count
        if count % 2 == 0:
            tmp = imT[100:-100, 100:-100]
            sf = cv2.countNonZero(tmp)/nL
            #print(f' Bigtile occupancy rate: {sf * 100:.2e} %')

        for p in range(numann.shape[1]):
            # print(f'p numann idx: {p}')
            ct[p] += np.sum(tmpT == p+1)

        if count % 150 == 0 or sf > cutoff:
            tmp = np.histogram(imT, bins=np.arange(numann.shape[1] + 2))[0]
            ct = tmp[1:]
            ct[ct == 0] = 1

        count += 1
        type0 = type_

    # cut edges off tile
    imH = imH[100:-100, 100:-100, :].astype(np.uint8)
    imT = imT[100:-100, 100:-100].astype(np.uint8)
    for p in range(nblack - 1):  # the '-1' has to do with python indxing
        ct[p] = np.sum(imT == p)
    imT[imT == 0] = nblack
    imT = imT-1  # We want python index ==> First class label = 0

    # save cutouts to outpth
    sz = imH.shape
    for s1 in range(0, sz[0], sxy):
        for s2 in range(0, sz[1], sxy):
            try:
                imHtmp = imH[s1:s1 + sxy, s2:s2 + sxy, :]
                imTtmp = imT[s1:s1 + sxy, s2:s2 + sxy]
                Image.fromarray(imHtmp).save(os.path.join(outpthim, f"{nm0}.png"))
                Image.fromarray(imTtmp).save(os.path.join(outpthlabel, f"{nm0}.png"))
                nm0 += 1
            except ValueError:
                continue

    # save large tiles
    print('Saving big tiles')
    nm1 = len([f for f in os.listdir(outpthbg) if f.startswith('HE')]) + 1
    Image.fromarray(imH).save(os.path.join(outpthbg, f"HE_tile_{nm1}.jpg"))
    Image.fromarray(imT).save(os.path.join(outpthbg, f"label_tile_{nm1}.jpg"))

    return numann, percann

# # Example usage
#
#if __name__ == '__main__':
#
#     # Pre - inputs
#     pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\Jaime\Optimize'
#     pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\Jaime\Optimize\optimization_06_28_2024'
#     pthim_ann = r'\\10.99.68.52\Kiemendata\Valentina Matos\Jaime\Optimize\5x'
#     classcheck = 0
#     datafile = r'\\10.99.68.52\Kiemendata\Valentina Matos\Jaime\Optimize\optimization_06_28_2024\net.pkl'
##
#     # Inputs
#     import pickle
#     from load_annotation_data import load_annotation_data
##
#     with open(datafile, 'rb') as f:
#         data = pickle.load(f)
#     nblack = data['nblack']
#     sxy = data['sxy']
#     ctlist0, numann0 = load_annotation_data(pthDL, pth, pthim_ann, classcheck)
#     numann0 = np.array(numann0)  # Convert numann0 to a NumPy array
#
#     numann = numann0.copy()
#     percann = np.double(numann0 > 0)
#     percann = np.dstack((percann, percann))
#     percann0 = percann.copy()
#     stile = None
#     nbg = None
##
#     outpth = r'training'
##
##     # Function
##     full_function_start_time = time.time()
##     print('Starting timer for the function call')
#     numann, percann = combine_annotations_into_tiles(numann0, numann, percann, ctlist0, nblack, pthDL, outpth, sxy)
#     end_fucntion_time = time.time() - full_function_start_time
#     hours, re, = divmod(end_fucntion_time, 3600)
#     minutes, seconds = divmod(re, 60)
#     print(f'Function call took {hours}h {minutes}m {seconds}s')
#
#
#
#