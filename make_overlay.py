"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: June 14, 2024
"""

import keras
import numpy as np
import cv2
import os
from Semanticseg import read_image

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb
def get_overlay(image, colored_mask):
    image = keras.utils.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.65, colored_mask, 0.35, 0)
    return overlay

def make_overlay(image_file, prediction_mask, image_size, colormap, save_path):

    os.makedirs(save_path, exist_ok=True)

    image_tensor = read_image(image_file, image_size)
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, n_classes=len(colormap)-1) #Take out the black label
    overlay = get_overlay(image_tensor, prediction_colormap)

    # Save the overlay as jpg
    overlay_image = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    save_file_path = os.path.join(save_path, os.path.basename(image_file))

    cv2.imwrite(save_file_path, overlay_image)

    return overlay

