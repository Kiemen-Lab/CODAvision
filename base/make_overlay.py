"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: August 20, 2024
"""

import keras
import numpy as np
import cv2
import os
import tensorflow as tf

from PIL import Image


def convert_to_array(image_path, prediction_mask):
    # Open the image and convert it to a numpy array
    image = cv2.imread(image_path)
    image = image[:, :, ::-1]

    # Check if the image is too large, and downsample it by 2 if it is
    if image.shape[0] > 20000 or image.shape[1] > 20000:
        # Convert numpy array to PIL Image
        image_pil = Image.fromarray(image)
        # Resize the image
        image_pil = image_pil.resize((image_pil.width // 2, image_pil.height // 2), Image.NEAREST)
        # Convert back to numpy array
        image = np.array(image_pil)

        # Convert prediction_mask to PIL Image and resize if needed
        prediction_mask_pil = Image.fromarray(prediction_mask)
        # Resize the prediction mask
        prediction_mask_pil = prediction_mask_pil.resize((prediction_mask_pil.width // 2, prediction_mask_pil.height // 2), Image.NEAREST)
        # Convert back to numpy array
        prediction_mask = np.array(prediction_mask_pil)

    return image, prediction_mask


def read_image_overlay(image_input):
    try:
        if isinstance(image_input, np.ndarray):
            # If input is an image array
            image = tf.convert_to_tensor(image_input)
        else:
            # If input is a file path
            image = tf.io.read_file(image_input)
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
        return image
    except Exception as e:
        print(f"Error reading image {image_input}: {e}")
        return None

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

def make_overlay(image_path, prediction_mask, colormap, save_path):
    os.makedirs(save_path, exist_ok=True)
    image_array, prediction_mask = convert_to_array(image_path, prediction_mask)
    image_tensor = read_image_overlay(image_array)
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, n_classes=len(colormap)) #Take out the black label
    overlay = get_overlay(image_tensor, prediction_colormap)

    # Save the overlay as jpg
    overlay_image = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    save_file_path = os.path.join(save_path, os.path.basename(image_path)[:-3]+'jpg')

    cv2.imwrite(save_file_path, overlay_image)

    return overlay