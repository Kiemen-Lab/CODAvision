"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 14, 2024
"""

import numpy as np
from skimage.morphology import disk, dilation
from base.augment_annotation import augment_annotation
import cv2
import os


def edit_annotations_tiles(im, TA, do_augmentation, class_id, num_pixels_class, big_tile_size, kpall):
    """
        Edit annotation tiles by performing augmentation and adjusting class distribution.

        Parameters:
            - im (numpy.ndarray): Input image.
            - TA (numpy.ndarray): Input label mask.
            - do_augmentation (bool): Flag indicating whether to perform augmentation.
            - class_id (int): ID of the class to adjust distribution for.
            - num_pixels_class (numpy.ndarray): Array containing the pixel counts for each class.
            - big_tile_size (int): Size of the big tile.
            - kpall (int): Flag indicating whether to keep all classes or not.

        Returns:
            - im (numpy.ndarray): Augmented and adjusted image.
            - TA (numpy.ndarray): Augmented and adjusted label mask.
            - kpout (numpy.ndarray): Unique labels after processing.
    """
    if do_augmentation:
        im, TA = augment_annotation(im, TA, 1, 1, 1, 1, 0)
    else:
        im, TA = augment_annotation(im, TA, 1, 1, 0, 0, 0)

    if kpall == 0:
        maxn = num_pixels_class[class_id]
        kp = num_pixels_class <= maxn * 1.05
    else:
        kp = num_pixels_class >= 0

    # Add zero padding to include background class
    kp = np.concatenate(([0], kp))
    tmp = kp[TA.astype(int)]

    # Dilate the mask
    dil = np.random.randint(15) + 15
    tmp = dilation(tmp, disk(dil))

    # Apply the mask to both image and label
    TA = TA * tmp
    for i in range(im.shape[2]):
        im[:, :, i] *= tmp

    # Extract unique labels excluding background
    kpout = np.unique(TA)[1:].astype(int)

    # Crop both image and label to specified dimensions
    p1, p2 = min(big_tile_size, TA.shape[0]), min(big_tile_size, TA.shape[1])
    im = im[0:p1, 0:p2, :]
    TA = TA[0:p1, 0:p2]

    #im= im.astype(np.uint16)
    TA = TA.astype(np.uint8)

    return im, TA, kpout