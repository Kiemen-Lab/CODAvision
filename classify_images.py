"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: June 17, 2024
"""

import cv2
import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import time
from scipy.ndimage import binary_fill_holes
from Semanticseg import semantic_seg
from make_overlay import make_overlay

Image.MAX_IMAGE_PIXELS = None
import keras
from make_overlay import decode_segmentation_masks
from tensorflow.keras.models import load_model


def classify_images(pthim, pthDL, color_overlay_HE=True, color_mask=False):
    start_time = time.time()

    # Load the model weights and other relevant data
    model = load_model(os.path.join(pthDL, 'net'))
    with open(os.path.join(pthDL, 'net.pkl'), 'rb') as f:
        data = pickle.load(f)
        classNames = data['classNames']
        # sxy = data['sxy']
        nblack = data['nblack']
        nwhite = data['nwhite']
        cmap = data['cmap']
        nm = data['nm']
        sxy = data['sxy']

    outpth = os.path.join(pthim, 'classification_' + nm)
    os.makedirs(outpth, exist_ok=True)

    b = 100
    imlist = sorted(glob(os.path.join(pthim, '*.png')))
    # If no PNGs found, search for TIFF and JPG files
    if not imlist:
        jpg_files = glob(os.path.join(pthim, "*.jpg"))
        if jpg_files:
            imlist.extend(jpg_files)  # Add full paths of JPGs to list
        tif_files = glob(os.path.join(pthim, '*.tif'))
        if tif_files:
            for file in tif_files:
                im = Image.open(file)
                imname = os.path.basename(file)
                im.save(os.path.join(pthim, imname[:-4] + '.png'))
            png_files = glob(os.path.join(pthim, "*.png"))
            imlist.extend(png_files)
    if not imlist:
        print("No TIFF, PNG or JPG image files found in", pthim)
    print('   ')

    classification_st = time.time()

    for i, img_path in enumerate(imlist):
        img_name = os.path.basename(img_path)
        print(f'  Starting classification of image {i + 1} of {len(imlist)}: {img_name}')
        if os.path.isfile(os.path.join(outpth, img_name[:-4] + ".png")):
            print(f'  Image {img_name} already classified by this model')
            continue
        im = Image.open(os.path.join(pthim, img_name))
        im_array = np.array(im)  # Convert to NumPy array for slicing
        try:
            try:
                TA = Image.open(os.path.join(outpth, 'TA', img_name[:-4] + ".png"))
            except:
                TA = Image.open(os.path.join(outpth, 'TA', img_name[:-4] + ".tif"))
            TA = binary_fill_holes(TA)
        except:
            TA = np.array(im.convert('L')) < 220
            TA = binary_fill_holes(TA.astype(bool))

        # Pad image so we classify all the way to the edge
        im_array = np.pad(im_array, pad_width=((sxy + b, sxy + b), (sxy + b, sxy + b), (0, 0)), mode='constant',
                          constant_values=0)
        TA = np.pad(TA, pad_width=((sxy + b, sxy + b), (sxy + b, sxy + b)), mode='constant', constant_values=True)

        imclassify = np.zeros(TA.shape, dtype=np.uint8)
        sz = np.array(im_array).shape

        # Calculate the total number of tiles
        padded_height, padded_width, _ = np.array(im).shape  # Get the padded image dimensions
        tile_rows = (sz[0]-2*sxy) // (sxy - b * 2) + 1
        tile_cols = (sz[1]-2*sxy) // (sxy - b * 2) + 1
        total_tiles = tile_rows * tile_cols
        count = 1

        for s1 in range(sxy, sz[0]-sxy, sxy - b * 2):
            for s2 in range(sxy, sz[1]-sxy, sxy - b * 2):

                print(f'   Tile: {count} of {total_tiles}')

                tileHE = im_array[s1:s1 + sxy, s2:s2 + sxy, :]
                #tileHE_image = Image.fromarray(tileHE)
                #save_path_HE = os.path.join(save_dirHE,
                #                            f'{img_name}_tileHE_{count}.png')  # You can change the filename as needed
                #tileHE_image.save(save_path_HE)

                tileTA = TA[s1:s1 + sxy, s2:s2 + sxy]
                print(f'   Tile: {count} of {total_tiles} at ({s1}, {s2}), size: {tileHE.shape}')

                #if np.sum(tileTA) < 100:
                #    tileclassify = np.zeros((sxy,sxy))
                #    tileclassify.fill(nblack-1)
                #else:
                tileclassify = semantic_seg(tileHE, image_size=sxy, model=model)

                tileclassify = tileclassify[b:-b, b:-b]

                # Convert to uint8 before saving
                #tileclassify_uint8 = (tileclassify * 10).astype(np.uint8)
                #tilemask_image = Image.fromarray(tileclassify_uint8)
                #save_path_mask = os.path.join(save_dirmask, f'{img_name}_tilemask_{count}.png')
                #tilemask_image.save(save_path_mask)
                imclassify[s1 + b:s1 + sxy - b, s2 + b:s2 + sxy - b] = tileclassify
                count += 1

        # Remove padding
        im_array = im_array[sxy + b:-sxy - b, sxy + b:-sxy - b, :]
        imclassify = imclassify[sxy + b:-sxy - b, sxy + b:-sxy - b]

        imclassify = imclassify +1

        imclassify[np.logical_or(imclassify == nblack, imclassify == 0)] = nwhite  #Change black labels to whitespace

        elapsed_time = round(time.time() - classification_st)
        print(f'Image {i + 1} of {len(imlist)} took {elapsed_time} s')

        # Save Classified Image
        imclassify_PIL = Image.fromarray(imclassify)  # Convert NumPy array to PIL Image
        imclassify_PIL.save(os.path.join(outpth, img_name[:-3] + 'tif'))  # Save as TIFF


        # Make color image overlay on H&E
        if color_overlay_HE:
            imclassify = imclassify-1
            save_path = os.path.join(outpth, 'check_classification')
            _ = make_overlay(img_path, imclassify, colormap=cmap, save_path=save_path)

        if color_mask:
            outpthcolor = os.path.join(outpth, 'color')
            os.makedirs(outpthcolor, exist_ok=True)
            red_channel = cmap[:, 0]
            green_channel = cmap[:, 1]
            blue_channel = cmap[:, 2]

            imcolor = np.dstack((red_channel[imclassify], green_channel[imclassify], blue_channel[imclassify])).astype(
                np.uint8)

            save_file_path = os.path.join(outpthcolor, os.path.basename(img_path))
            cv2.imwrite(save_file_path, imcolor)

        # Display the first image in the series
        if i == 0 and cmap is not None:
            prediction_colormap = decode_segmentation_masks(imclassify, cmap, n_classes=len(classNames)-1)



    end_time = time.time() - start_time
    hours, rem = divmod(end_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'  Total time for classification: {hours}h {minutes}m {seconds}s')

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im_array)
    axs[1].imshow(keras.utils.array_to_img(prediction_colormap))

    for ax in axs:
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.pause(0.2)
    return outpth


#Example:
#if __name__ == '__main__':
#    # Inputs:
#    pthim = r'C:\Users\Valentina\OneDrive - Johns Hopkins\Desktop\test png\PNG'
#    pthDL = r'\\10.99.68.52\Kiemendata\Valentina Matos\coda to python\test model\04_19_2024'
#    color_overlay_HE = False  # set to true in the real code
#    color_mask = False

 #   _ = classify_images(pthim, pthDL, color_overlay_HE=False, color_mask=False)
