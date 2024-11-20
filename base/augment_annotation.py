"""
Author: Jaime Gomez/Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 14, 2024
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def augment_annotation(imh0, imlabel0, rot=True, sc=True, hue=True, blr=False, rsz=False):
    """
    Randomly augments with rotation, scaling, hue, and blur

    Parameters:
        -imh0 (numpy.ndarray): H&E image tile
        -imlabel0 (numpy.ndarray): mask image for H&E
        -rot (bool, optional): Perform rotation augmentation? Defaults to True.
        -sc (bool, optional): Perform scaling augmentation? Defaults to True.
        -hue (bool, optional): Perform hue augmentation? Defaults to True.
        -blr (bool, optional): Perform blur augmentation? Defaults to False.
        -rsz (bool, optional): Resize images after scaling augmentation? Defaults to False.

    Returns:
        -imh (numpy.ndarray): Augmented H&E image
        -imlabel (numpy.ndarray): Mask of the augmented image
    """
    if imlabel0 is None:
        imlabel0 = imh0.copy()

    imh = imh0.astype(np.float64)
    imlabel = imlabel0.astype(np.float64)
    szz = imh0.shape[0]

    # Random rotation
    if rot:

        angs = np.arange(0, 360, 5)
        angle = np.random.choice(angs)  # Get a random rotation angle
        height, width = imh.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        # Calculate the new bounding box after rotation
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)

        rotation_matrix[0, 2] += (new_width / 2) - width / 2
        rotation_matrix[1, 2] += (new_height / 2) - height / 2

        imh = cv2.warpAffine(imh, rotation_matrix, (new_width, new_height), borderValue=(0, 0, 0),
                             flags=cv2.INTER_NEAREST)
        imlabel = cv2.warpAffine(imlabel, rotation_matrix, (new_width, new_height), borderValue=(0, 0, 0),
                                 flags=cv2.INTER_NEAREST)

    # Random scaling
    if sc:
        scales = np.concatenate((np.arange(0.6, 0.96, 0.01), np.arange(1.1, 1.41, 0.01)))
        ii = np.random.permutation(len(scales))[0] # Get a random scaling factor from the desired value range (scales)
        imh = cv2.resize(imh, None, fx=scales[ii], fy=scales[ii], interpolation=cv2.INTER_NEAREST)
        imlabel = cv2.resize(imlabel, None, fx=scales[ii], fy=scales[ii], interpolation=cv2.INTER_NEAREST)

    # Random hue adjustment
    if hue:
        rd = np.concatenate((np.arange(0.88, 0.99, 0.01), np.arange(1.02, 1.13, 0.01)))
        gr = np.concatenate((np.arange(0.88, 0.99, 0.01), np.arange(1.02, 1.13, 0.01)))
        bl = np.concatenate((np.arange(0.88, 0.99, 0.01), np.arange(1.02, 1.13, 0.01)))
        ird = np.random.permutation(len(rd))[0] # Get a random red hue adjustment from the desired value range (rd)
        igr = np.random.permutation(len(gr))[0] # Get a random green hue adjustment from the desired value range (gr)
        ibl = np.random.permutation(len(bl))[0]  # Get a random blue hue adjustment from the desired value range (bl)

        # Scale red
        imr = 255 - imh[:, :, 0]
        imh[:, :, 0] = 255 - (imr * rd[ird])

        # Scale green
        img = 255 - imh[:, :, 1]
        imh[:, :, 1] = 255 - (img * gr[igr])

        # Scale blue
        imb = 255 - imh[:, :, 2]
        imh[:, :, 2] = 255 - (imb * bl[ibl])

    # Random blurring

    if blr:
        bll = np.ones(50)
        bll[0] = 1.05
        bll[1] = 1.1
        bll[2] = 1.15
        bll[3] = 1.2
        ibl = np.random.randint(len(bll))
        bll = bll[ibl]

        if bll != 1:
            imh = gaussian_filter(imh, sigma=bll)

    # If scaling augmentation was performed, resize images to be correct tilesize
    if rsz:
        szh = imh.shape[0]
        if szh > szz:
            cent = int(np.round(szh / 2))
            sz1 = int((szz - 1) // 2)
            sz2 = int(np.ceil((szz - 1) / 2))
            imh = imh[cent - sz1:cent + sz2, cent - sz1:cent + sz2, :]
            imlabel = imlabel[cent - sz1:cent + sz2, cent - sz1:cent + sz2]
        elif szh < szz:
            tt = szz - szh
            imh = np.pad(imh, ((0, tt), (0, tt), (0, 0)), mode='constant', constant_values=0)
            imlabel = np.pad(imlabel, ((0, tt), (0, tt)), mode='constant', constant_values=0)

    # Remove non-annotated pixels from imH
    tmp = (imlabel != 0)
    tmp = np.dstack((tmp, tmp, tmp))
    imh = imh * tmp

    return imh, imlabel