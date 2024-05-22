"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 20, 2024
"""

import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from edit_annotation_tiles import edit_annotations_tiles
from PIL import Image
from scipy.ndimage import convolve
import time



def combine_annotations_into_tiles(numann0, numann, percann, imlist, nblack, pthDL, outpth, sxy, stile=10000, nbg=0):
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

    imlistck = [f for f in os.listdir(outpthim) if f.endswith('.tif')]
    nm0 = len(imlistck) + 1

    # Create very large blank images
    imH = np.full((stile, stile, 3), nbg, dtype=np.uint8) # create an array with the specified dimensions with the value of ngb as background
    imT = np.zeros((stile, stile), dtype=np.uint8)  # blank mask
    nL = imT.size  # size of the big tile
    ct = np.zeros(numann.shape[1])  # list (size of how many annotations you have)
    sf = np.sum(ct) / nL  # how full is the big tile

    count = 1
    tcount = 0  # is not one because of python indexing starts at 0
    cutoff = 0.55  # Cut off for how full the big tile is (55% full)
    rsf = 5
    type0 = 0
    h = np.ones((51, 51))
    h[25, 25] = 0
    h = distance_transform_edt(h) < 26

# Here comes the fun part, tiles are going to be added to the big black tile until the cutoff is achieved ~55%
    iteration = 1
    # start iteration timing
    iter_start_time = time.time()
    print('Starting time for the while loop')
    while sf < cutoff:
        print(f'Iteration: {iteration}')
        print(f' Bigtile occupancy rate: {(sf:.2e)*100} %')

        # choose one of each class in order in a loop
        if count % 10 == 1:
            type_ = tcount
            tcount = (tcount % len(ct))
        # choose a tile containing the least prevalent class
        else:

            tmp = ct.copy()
            tmp[type0] = np.max(tmp)
            type_ = np.argmin(tmp)

        print(f" type_: {type_}")

        num = np.where(numann[:, type_] > 0)[0]

        if len(num) == 0:
            numann[:, type_] = numann0[:, type_]
            num = np.where(numann[:, type_] > 0)[0]

        num = np.random.choice(num, size=1, replace=False)

        # chosen random tile to be processed
        tile_name = imlist['tile_name'][num[0]]
        tile_path = os.path.join(imlist['tile_pth'][num[0]], tile_name)
        pf = tile_path.rfind('\\', 0, tile_path.rfind('\\'))
        pthlabel = os.path.join(tile_path[0:pf], 'label')

        im = np.array(Image.open(tile_path))
        TA = np.array(Image.open(os.path.join(pthlabel, tile_name)))

        doaug = (count % 3 == 1)
        im, TA, kp = edit_annotations_tiles(im, TA, doaug, type_, ct, imT.shape[0], kpall)
        numann[num, kp - 1] = 0  # kp-1 due to layer index starting in 1 and python index starting in 0
        percann[num, kp - 1, 0] += 1
        percann[num, kp - 1, 1] = 2
        fx = (TA != 0) #linear idx of tissue pixels in the mask bb
        if np.sum(fx) < 30:
            print('skipped')
            continue

        # find low density location in large tile to add annotation
        tmp = imT > 0
        tmp2 = tmp[::rsf, ::rsf]
        tmp2 = convolve(tmp2.astype(float), h, mode='constant', cval=0.0)
        tmp = distance_transform_edt(tmp2 == 0)
        tmp[:19, :] = 0
        tmp[-20:, :] = 0
        tmp[:, :19] = 0
        tmp[:, -20:] = 0
        xii = np.argmax(tmp)
        xii = np.random.choice(xii, size=1, replace=False)
        x, y = np.unravel_index(xii, tmp.shape)

        x = int(x * rsf)
        y = int(y * rsf)

        szz = np.array(TA.shape) - 1
        szzA = szz // 2
        szzB = szz - szzA

        # Ensure szzA and szzB are tuples of integers to be used for slicing afterward
        szzA = tuple(map(int, szzA))
        szzB = tuple(map(int, szzB))

        #x,y - Location in the big tile to add the BB to
        if x + szzA[0] > imT.shape[1]:
            x -= szzA[0]
        if y + szzA[1] > imT.shape[0]:
            y -= szzA[1]
        if x - szzB[0] < 0:
            x += szzB[0]
        if y - szzB[1] < 0:
            y += szzB[1]

        tmpT = imT[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1].copy() #imT is the bigtile mask
        tmpT[fx] = TA[fx]
        tmpH = imH[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1, :].copy()
        tmpH[np.dstack((fx, fx, fx))] = im[np.dstack((fx, fx, fx))]
        imT[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1] = tmpT
        imH[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1, :] = tmpH #imH is the bigtile HE

        # Update total count
        if count % 2 == 0:
            sf = np.sum(imT > 0) / nL  # print(f"{count} {round(sf * 100)}")

        for p in range(numann.shape[1]):
            # print(f'p numann idx: {p}')
            ct[p] += np.sum(tmpT == p+1)

        if count % 150 == 0 or sf > cutoff:
            tmp = np.histogram(imT, bins=np.arange(numann.shape[1] + 2))[0]
            ct = tmp[1:]
            ct[ct == 0] = 1

        count += 1
        type0 = type_
        iteration += 1

    # End of while loop timer
    end_time = time.time()
    total_time_while = end_time - iter_start_time
    print(f'Total time elapsed for the while loop: {total_time_while}')


    # cut edges off tile
    imH = imH[100:-100, 100:-100, :].astype(np.uint8)
    imT = imT[100:-100, 100:-100].astype(np.uint8)
    for p in range(nblack - 1):  # the '-1' has to do with python indxing
        ct[p] = np.sum(imT == p)
    imT[imT == 0] = nblack

    # save cutouts to outpth
    sz = imH.shape
    for s1 in range(0, sz[0], sxy):
        for s2 in range(0, sz[1], sxy):
            try:
                imHtmp = imH[s1:s1 + sxy, s2:s2 + sxy, :]
                imTtmp = imT[s1:s1 + sxy, s2:s2 + sxy]
            except ValueError:
                continue

            Image.fromarray(imHtmp).save(os.path.join(outpthim, f"{nm0}.tif"))
            Image.fromarray(imTtmp).save(os.path.join(outpthlabel, f"{nm0}.tif"))
            nm0 += 1

    nm1 = len([f for f in os.listdir(outpthbg) if f.startswith('HE')]) + 1

    # save large tiles
    print('Saving big tiles')
    Image.fromarray(imH).save(os.path.join(outpthbg, f"HE_tile_{nm1}.jpg"))
    Image.fromarray(imT).save(os.path.join(outpthbg, f"label_tile_{nm1}.jpg"))

    return numann, percann

# Example usage

if __name__ == '__main__':

    # Pre - inputs
    pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model'
    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\04_19_2024'
    pthim_ann = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\5x'
    classcheck = 0
    datafile = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\04_19_2024\net.pkl'

    # Inputs
    import pickle
    from load_annotation_data import load_annotation_data

    with open(datafile, 'rb') as f:
        data = pickle.load(f)
    nblack = data['nblack']
    sxy = data['sxy']
    ctlist0, numann0 = load_annotation_data(pthDL, pth, pthim_ann, classcheck)
    numann0 = np.array(numann0)  # Convert numann0 to a NumPy array

    numann = numann0.copy()
    percann = np.double(numann0 > 0)
    percann = np.dstack((percann, percann))
    percann0 = percann.copy()
    stile = None
    nbg = None

    outpth = r'training'

    # Function
    full_function_start_time = time.time()
    print('Starting timer for the function call')
    numann, percann = combine_annotations_into_tiles(numann0, numann, percann, ctlist0, nblack, pthDL, outpth, sxy)
    end_fucntion_time = time.time() - full_function_start_time
    hours, re, = divmod(end_fucntion_time, 3600)
    minutes, seconds = divmod(re, 60)
    print(f'Function call took {hours}h {minutes}m {seconds}s')




