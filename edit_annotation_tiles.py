"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 14, 2024
"""

import numpy as np
from skimage.morphology import disk, dilation
from augment_annotation import augment_annotation


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

    im = im.astype(np.uint8)
    TA = TA.astype(np.uint8)

    return im, TA, kpout


# ___________________________________________
#
# Example usage
if __name__ == "__main__":
    from load_annotation_data import load_annotation_data
    import os
    import matplotlib.pyplot as plt

    # Pre - inputs
    pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model'
    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\04_19_2024'
    pthim_ann = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\5x'
    classcheck = 0
    _, numann = load_annotation_data(pthDL, pth, pthim_ann, classcheck)

    pthlabel = (r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test '
                r'model\data\SG_014_0016\04_19_2024_boundbox\label')
    pthim = (r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test '
             r'model\data\SG_014_0016\04_19_2024_boundbox\im')
    imnm = '00226.tif'  #image names/idx differ from the ones in matlab, as the tiles are created in a random order
    size_tile = 10200
    imT = np.zeros((size_tile, size_tile), dtype=np.uint8)
    pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model'

    # ________inputs of the function________
    im = np.array(plt.imread(os.path.join(pthim, imnm)), dtype=np.float64)
    # print(f'im: {os.path.join(pthim, imnm)}')
    TA = np.array(plt.imread(os.path.join(pthlabel, imnm)), dtype=np.float64)
    # print(f'TA: {os.path.join(pthlabel, imnm)}')
    do_augmentation = True
    class_id = 1
    big_tile_size = imT.shape[0]
    # print(f'big tile size: {big_tile_size}')
    Shape_numann = 11
    # print(f'Shape Numann: {Shape_numann}')
    num_pixels_class = np.zeros(Shape_numann, dtype=np.int32)
    # print(f'num_pixels_class: {num_pixels_class}')
    kpall = 1

    # Function
    im, TA, kpout = edit_annotations_tiles(im, TA, do_augmentation, class_id, num_pixels_class, big_tile_size, kpall)
    print('\nOUTPUTS:')
    print(f'Unique TA labels (Kpout): {kpout}')
    plt.imshow(im)
    plt.title(imnm + ' augmented tile & edit_tile() processing')
    plt.show()
