"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 3rd, 2024
"""

from base.calculate_tissue_mask import calculate_tissue_mask
from base.check_if_model_parameters_changed import check_if_model_parameters_changed
from .import_xml import import_xml
from .save_annotation_mask import save_annotation_mask
from .save_bounding_boxes import save_bounding_boxes
from .WSI2png import WSI2png
from .WSI2tif import WSI2tif
import os
import pickle
import shutil
import numpy as np
from skimage import io
import time
import warnings


def load_annotation_data(pthDL,pth,pthim,classcheck=0):
    """
      Loads the annotation data from the .xml files, creates tissue mask and bounding boxes and saves the bounding boxes
      as tif images, both for the H&E image and the labeled image

      Parameters:
      pthDL (str): The file path to save the model data.
      pth (str): The file path to the annotations.
      pthim (str): The file path to the tif images of the desired resolution.
      classcheck (float, optional): Used in validate_annotations.

      Returns:
      ctlist0 (list): List containing the bounding boxes filenames and the path to them.
      numann0 (np.ndarray): An array containing the number of pixels per class per bounding box. Each row is a bounding
                            box and each column is a class.
      """


    print(' ')
    print('Importing annotation data...')

    # Turn off user warnings
    warnings.filterwarnings("ignore")

    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        WS = data['WS']
        umpix = data['umpix']
        cmap = data['cmap']
        nm = data['nm']
        nwhite = data['nwhite']


    cmap2 = np.vstack(([0, 0, 0], cmap)) / 255
    numclass = np.max(WS[2])
    imlist = [f for f in os.listdir(pth) if f.endswith('.xml')]
    numann0 = []
    ctlist0 = {'tile_name': [], 'tile_pth': []}
    outim = os.path.join(pth, 'check_annotations')


    # Check that all images exist for all the .cml files contained in the folder
    for imnm in imlist:
        imnm = imnm[:-4]
        tif_file = os.path.join(pthim, f'{imnm}.tif')
        jpg_file = os.path.join(pthim, f'{imnm}.jpg')
        png_file = os.path.join(pth, f'{imnm}.png')
        if not os.path.isfile(tif_file) and not os.path.isfile(jpg_file) and not os.path.isfile(png_file):
            raise FileNotFoundError(f'Cannot find a tif, png or jpg file for xml file: {imnm}.xml')

    # for each annotation file

    for idx, imnm in enumerate(imlist, start=1):
        image_time = time.time()
        print(f'Image {idx} of {len(imlist)}: {imnm[:-4]}')
        imnm = imnm[:-4]
        outpth = os.path.join(pth, 'data py', imnm)
        annotations_file = os.path.join(outpth, 'annotations.pkl')

        # check if model parameters have changed
        reload_xml = check_if_model_parameters_changed(annotations_file, WS, umpix, nwhite, pthim)

        # skip if file hasn't been updated since last load
        if os.path.isfile(annotations_file):
            with open(annotations_file, 'rb') as f:
                data = pickle.load(f)
                dm, bb = data.get('dm', ''), data.get('bb', 0)
        else:
            dm, bb = '', 0

        modification_time = os.path.getmtime(os.path.join(pth, f'{imnm}.xml'))
        date_modified = time.ctime(modification_time)

        create_new_tiles = True
        if dm == str(date_modified) and bb == 1 and not reload_xml:
            print(' annotation data previously loaded')
            with open(annotations_file, 'rb') as f:
                data = pickle.load(f)
                numann, ctlist = data.get('numann', []), data.get('ctlist', [])
            numann0.extend(numann)
            # ctlist0.extend(ctlist)
            #ctlist0 is now a dictionary
            ctlist0['tile_name'].extend(ctlist['tile_name'])
            ctlist0['tile_pth'].extend(ctlist['tile_pth'])
            create_new_tiles = False
            continue


        if os.path.isdir(outpth):
            shutil.rmtree(outpth)
        os.makedirs(outpth)

        # 1 read xml annotation files and save as pkl files

        import_xml(annotations_file, os.path.join(pth, f'{imnm}.xml'), date_modified)

        with open(annotations_file, 'rb') as f:  #
            data = pickle.load(f)  #
            data['WS'] = WS
            data['umpix'] = umpix
            data['nwhite'] = nwhite
            data['pthim'] = pthim
        with open(annotations_file, 'wb') as f:
            pickle.dump(data, f)
            f.close()

        # 2 fill annotation outlines and delete unwanted pixels
        with open(annotations_file, 'rb') as f:  #
            data = pickle.load(f)

        I0, TA, _ = calculate_tissue_mask(pthim, imnm)

        J0 = save_annotation_mask(I0, outpth, WS, umpix, TA, 1)

        io.imsave(os.path.join(outpth, 'view_annotations.png'), J0.astype(np.uint8))

        # show mask in color
        I = I0[::2, ::2, :].astype(np.float64) / 255
        J = J0[::2, ::2].astype(int)
        J1 = cmap2[J, 0]
        J1 = J1.reshape(J.shape)
        J2 = cmap2[J, 1]
        J2 = J2.reshape(J.shape)
        J3 = cmap2[J, 2]
        J3 = J3.reshape(J.shape)
        mask = np.dstack((J1, J2, J3))
        I = (I * 0.5) + (mask * 0.5)
        if create_new_tiles and os.path.isdir(outim):
            shutil.rmtree(outim)
        os.makedirs(outim, exist_ok=True)
        io.imsave(os.path.join(outim, f'{imnm}.png'), (I * 255).astype(np.uint8))

        # create annotation bounding boxes and update data to annotation.pkl file
        numann, ctlist = save_bounding_boxes(I0, outpth, nm, numclass)
        numann0.extend(numann)

        #ctlist0 is now a dictionary
        ctlist0['tile_name'].extend(ctlist['tile_name'])
        ctlist0['tile_pth'].extend(ctlist['tile_pth'])

        print(f' Finished image in {round(time.time() - image_time)} seconds.')
    return ctlist0, numann0, create_new_tiles