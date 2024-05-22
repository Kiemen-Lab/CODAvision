"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 22, 2024
"""

import glob
import numpy as np
import time
from combine_annotations_into_tiles import combine_annotations_into_tiles
import os
import pickle

def create_training_tiles(pthDL, numann0, ctlist0):
    """
    Builds training and validation tiles using the annotation bounding boxes and saves them to the model name folder

    Inputs:
        pthDL (str): Path to the main data directory.
        numann0 (numpy.ndarray): Array containing annotations.
        ctlist0 (list): List of image paths.

    Outputs:
        HE and Label big tiles, they are saved within the function.
        The code returns NONE
    """

    # Load data from net.pkl file
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        sxy, nblack, classNames, ntrain, nvalidate = data['sxy'], data['nblack'], data['classNames'], data['ntrain'], \
        data['nvalidate']

    if classNames[-1] == "black":
        classNames = classNames[:-1]
    print('')

    # Calculate pixel composition for each annotation class
    print('Calculating total number of pixels in the training dataset...')
    count_annotations = sum(numann0)
    annotation_composition = count_annotations / max(count_annotations) * 100
    for b, count in enumerate(annotation_composition):
        if annotation_composition[b] == 100:
            print(f' There are {count} pixels of {classNames[b]}. This is the most common class.')
        else:
            print(
                f' There are {count} pixels of {classNames[b]}, {int(annotation_composition[b])}% of the most common class.')

    # Check for missing annotations
    if 0 in count_annotations:
        raise ValueError(
            'There are no annotations for one or more classes. Please add annotations, check nesting, or remove empty classes.')

    # Build training tiles
    print('')
    print('Building training tiles...')
    numann0 = np.array(numann0)  # Convert numann0 to a NumPy array
    numann = numann0.copy()
    percann = np.double(numann0 > 0)
    percann = np.dstack((percann, percann))
    percann0 = percann.copy()
    ty = 'training\\'
    obg = os.path.join(pthDL, ty, 'big_tiles\\')
    # Generate tiles until enough are made
    train_start = time.time()
    if len(glob.glob(os.path.join(obg, 'HE*.jpg'))) >= ntrain:
        print('  Already done.')
    else:
        while len(glob.glob(os.path.join(obg, 'HE*.jpg'))) < ntrain:
            numann, percann = combine_annotations_into_tiles(numann0, numann, percann, ctlist0, nblack, pthDL, ty, sxy)
            elapsed_time = time.time() - train_start
            print(
                f'  {len(glob.glob(os.path.join(obg, "HE*.jpg")))} of {ntrain} training images completed in {int(elapsed_time / 60)} minutes')

            baseclass1 = np.sum(percann0[:, :, 0])
            usedclass1 = np.sum(percann[:, :, 0])
            baseclass2 = np.sum(percann0[:, :, 1] == 1)
            usedclass2 = np.sum(percann[:, :, 1] == 2)

            tmp1 = usedclass1 / baseclass1 * 100
            tmp2 = usedclass2 / baseclass2 * 100

            for b, class_name in enumerate(classNames):
                print(f'  Used {tmp1[b]:.1f}% counts and {tmp2[b]:.1f}% unique annotations of {class_name}')

    total_time_train_bigtiles = time.time() - train_start
    hours, rem = divmod(total_time_train_bigtiles, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'  Elapsed time to create training big tiles: {hours}h {minutes}m {seconds}s')

    print('')

    # Build validation tiles
    ty = 'validation\\'
    obg = os.path.join(pthDL, ty, 'big_tiles\\')
    numann = numann0.copy()
    percann = (numann0 > 0).astype(float)
    percann = np.dstack((percann, percann))
    percann0 = percann.copy()
    validation_start_time = time.time()

    print('')
    print('Building validation tiles...')
    if len(glob.glob(os.path.join(obg, 'HE*.jpg'))) >= nvalidate:
        print('Already done.')
    else:
        while len(glob.glob(os.path.join(obg, 'HE*.jpg'))) < nvalidate:
            numann, percann = combine_annotations_into_tiles(numann0, numann, percann, ctlist0, nblack, pthDL, ty, sxy)
            elapsed_time = time.time() - validation_start_time
            print(
                f'{len(glob.glob(os.path.join(obg, "HE*.jpg")))} of {nvalidate} validation images completed in {int(elapsed_time / 60)} minutes')

            baseclass1 = np.sum(percann0[:, :, 0])
            usedclass1 = np.sum(percann[:, :, 0])
            baseclass2 = np.sum(percann0[:, :, 1] == 1)
            usedclass2 = np.sum(percann[:, :, 1] == 2)

            tmp1 = usedclass1 / baseclass1 * 100
            tmp2 = usedclass2 / baseclass2 * 100

            for b, class_name in enumerate(classNames):
                print(f'Used {tmp1[b]:.1f}% counts and {tmp2[b]:.1f}% unique annotations of {class_name}')

    total_time_validation_bigtiles = time.time() - validation_start_time
    hours, rem = divmod(total_time_validation_bigtiles, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'Elapsed time to create validation big tiles: {hours}h {minutes}m {seconds}s')

# Example usage
#
# if __name__ == '__main__':
#     # Pre - inputs
#     pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model'
#     pthim_ann = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\5x'
#     classcheck = 0
#
#     from load_annotation_data import load_annotation_data
#
#     # _____________________________________________________________________________
#
#     # Inputs
#     pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\04_19_2024'
#     ctlist0, numann0 = load_annotation_data(pthDL, pth, pthim_ann, classcheck)
#
#     create_training_tiles(pthDL, numann0, ctlist0)
#