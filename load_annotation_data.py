"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: May 3rd, 2024
"""

from calculate_tissue_mask import calculate_tissue_mask
from check_if_model_parameters_changed import check_if_model_parameters_changed
from import_xml import import_xml
from save_annotation_mask import save_annotation_mask
from save_bounding_boxes import save_bounding_boxes
import os
import pickle
import shutil
import numpy as np
from skimage import io
import time
import warnings

def load_annotation_data(pthDL,pth,pthim,classcheck=0):
    print(' ')
    print('Importing annotation data...')

    # Turn off user warnings
    warnings.filterwarnings("ignore")

    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        WS = data['WS']
        umpix = data['umpix']
        cmap = data['cmap']
        classNames = data['classNames']
        nm = data['nm']
        nwhite = data['nwhite']

    cmap2 = np.vstack(([0, 0, 0], cmap)) / 255
    numclass = np.max(WS[2])
    imlist = [f for f in os.listdir(pth) if f.endswith('.xml')]
    numann0 = []
    ctlist0 = []
    outim = os.path.join(pth, 'check_annotations')
    os.makedirs(outim, exist_ok=True)

    # Check that all images exist for all the .cml files contained in the folder
    for imnm in imlist:
        imnm = imnm[:-4]
        tif_file = os.path.join(pthim, f'{imnm}.tif')
        jpg_file = os.path.join(pthim, f'{imnm}.jpg')
        if not os.path.isfile(tif_file) and not os.path.isfile(jpg_file):
            raise FileNotFoundError(f'Cannot find a tif or jpg file for xml file: {imnm}.xml')
    # for each annotation file
    start_time = time.time()  # Capture the start time before the loop
    for idx, imnm in enumerate(imlist, start=1):
        print(f'Image {idx} of {len(imlist)}: {imnm[:-4]}')
        imnm = imnm[:-4]
        outpth = os.path.join(pth, 'data', imnm)
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
        if dm == str(date_modified) and bb == 1 and not reload_xml:
            print(' annotation data previously loaded')
            with open(annotations_file, 'rb') as f:
                data = pickle.load(f)
                numann, ctlist = data.get('numann', []), data.get('ctlist', [])
            numann0.extend(numann)
            ctlist0.extend(ctlist)
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
            data = pickle.load(f)  #
        I0, TA, _ = calculate_tissue_mask(pthim, imnm)
        J0 = save_annotation_mask(I0, outpth, WS, umpix, TA, 1)
        io.imsave(os.path.join(outpth, 'view_annotations.tif'), J0.astype(np.uint8))

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
        # J2 = J0 + 1
        # cmap3 = cmap2[np.min(J2):np.max(J2) + 1, :]
        # mask = cmap3[J2]
        # I = I0[::2, ::2, :].astype(np.float64) * 0.5 + mask[::2, ::2, :] * 0.5
        io.imsave(os.path.join(outim, f'{imnm}.jpg'), (I * 255).astype(np.uint8))

        # create annotation bounding boxes and update data to annotation.pkl file
        numann, ctlist = save_bounding_boxes(I0, outpth, nm, numclass)
        # with open(annotations_file, 'wb') as f:
        #     pickle.dump({'dm': str(date_modified), 'bb': 1, 'numann': numann, 'ctlist': ctlist}, f)
        numann0.extend(numann)
        ctlist0.extend(ctlist)

        print(f' Finished image in {round(time.time() - start_time)} seconds.')
    return ctlist0, numann0

# if __name__ == "__main__":
#     # Example usage
#
#     # Inputs
#     pth = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model'
#     pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\04_19_2024'
#     pthim = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\5x'
#     classcheck = 0
#
#     load_annotation_data(pthDL, pth, pthim,classcheck)