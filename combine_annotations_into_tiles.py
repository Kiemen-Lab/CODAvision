import os
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.filters import rank
from skimage.morphology import disk, dilation

def combine_annotations_into_tiles(numann0, numann, percann, imlist, nblack, pthDL, outpth, sxy, stile=10000, nbg=0):
    stile += 200
    kpall = 1

    # define folder locations
    outpthim = os.path.join(pthDL, outpth, 'im')
    outpthlabel = os.path.join(pthDL, outpth, 'label')
    outpthbg = os.path.join(pthDL, outpth, 'big_tiles')
    os.makedirs(outpthim, exist_ok=True)
    os.makedirs(outpthlabel, exist_ok=True)
    os.makedirs(outpthbg, exist_ok=True)
    imlistck = [f for f in os.listdir(outpthim) if f.endswith('.tif')]
    nm0 = len(imlistck) + 1

    # create very large blank images
    imH = np.ones((stile, stile, 3)) * nbg
    imT = np.zeros((stile, stile))
    nL = imT.size
    ct = np.zeros(numann.shape[1])
    sf = np.sum(ct) / nL

    count = 1
    tcount = 1
    cutoff = 0.55
    rsf = 5
    type0 = 1
    h = np.ones((51, 51))
    h[25, 25] = 0
    h = distance_transform_edt(h) < 26

    while sf < cutoff:
        # choose one of each class in order in a loop
        if count % 10 == 1:
            type_ = tcount
            tcount = (tcount % len(ct)) + 1
        # choose a tile containing the least prevalent class
        else:
            tmp = np.sum(ct, axis=0)
            tmp[type0] = np.max(tmp)
            type_ = np.argmin(tmp)

        num = np.where(numann[:, type_] > 0)[0]

        if len(num) == 0:
            numann[:, type_] = numann0[:, type_]
            num = np.where(numann[:, type_] > 0)[0]
        num = np.random.permutation(num)[0]

        # load annotation and mask
        imnm = imlist[num].name
        pthim = os.path.join(imlist[num].folder, '/')
        pf = pthim.rfind('/')
        pthlabel = os.path.join(pthim[:pf], 'label/')

        TA = np.array(Image.open(os.path.join(pthlabel, imnm)))
        im = np.array(Image.open(os.path.join(pthim, imnm)))

        # keep only needed annotation classes
        doaug = (count % 3 == 1)
        im, TA, kp = edit_annotation_tiles(im, TA, doaug, type_, ct, imT.shape[0], kpall)
        numann[num, kp] = 0
        percann[num, kp, 0] += 1
        percann[num, kp, 1] = 2
        fx = (TA != 0)
        if np.sum(fx) < 30:
            print('skipped')
            continue

        # find low density location in large tile to add annotation
        tmp = imT > 0
        tmp2 = tmp[::rsf, ::rsf]
        tmp2 = rank.filter(tmp2.astype(np.float), disk(25), footprint=h)
        tmp = bwdist(tmp2 > 0)
        tmp[:20, :] = 0
        tmp[-20:, :] = 0
        tmp[:, :20] = 0
        tmp[:, -20:] = 0
        xii = np.argmax(tmp)
        x, y = np.unravel_index(xii, tmp.shape)
        x *= rsf
        y *= rsf
        szz = np.array(TA.shape) - 1
        szzA = szz // 2
        szzB = szz - szzA

        if x + szzA[0] > imT.shape[1]:
            x -= szzA[0]
        if y + szzA[1] > imT.shape[0]:
            y -= szzA[1]
        if x - szzB[0] < 0:
            x += szzB[0]
        if y - szzB[1] < 0:
            y += szzB[1]
        tmpT = imT[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1]
        tmpT[fx] = TA[fx]
        tmpH = imH[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1, :]
        tmpH[np.broadcast_arrays(fx, fx, fx)] = im[fx]
        imT[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1] = tmpT
        imH[x - szzB[0]:x + szzA[0] + 1, y - szzB[1]:y + szzA[1] + 1, :] = tmpH

        # update total count
        if count % 2 == 0:
            sf = np.sum(imT > 0) / nL
        for p in range(numann.shape[1]):
            ct[p] += np.sum(tmpT == p)

        if count % 150 == 0 or sf > cutoff:
            tmp = np.histogram(imT, bins=np.arange(numann.shape[1] + 2))[0]
            ct = tmp[1:]
            ct[ct == 0] = 1

        count += 1
        type0 = type_

    # cut edges off tile
    imH = imH[100:-100, 100:-100, :]
    imT = imT[100:-100, 100:-100]
    for p in range(nblack):
        ct[p] = np.sum(imT == p)
    imT[imT == 0] = nblack

    # save cutouts to outpth
    sz = imH.shape
    for s1 in range(0, sz[0], sxy):
        for s2 in range(0, sz[1], sxy):
            try:
                imHtmp = imH[s1:s1 + sxy, s2:s2 + sxy, :]
                imTtmp = imT[s1:s1 + sxy, s2:s2 + sxy]
            except:
                continue
            Image.fromarray(imHtmp).save(os.path.join(outpthim, f'{nm0}.tif'))
            Image.fromarray(imTtmp).save(os.path.join(outpthlabel, f'{nm0}.tif'))
            nm0 += 1

    nm1 = len([f for f in os.listdir(outpthbg) if f.startswith('HE')]) + 1
    Image.fromarray(imH).save(os.path.join(outpthbg, f'HE_tile_{nm1}.jpg'))
    Image.fromarray(imT).save(Image.fromarray(imT).save(os.path.join(outpthbg, f'label_tile_{nm1}.jpg')))