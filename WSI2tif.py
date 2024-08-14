
"""
Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: August 14, 2024
"""

from PIL import Image
import numpy as np
import os
import glob
from openslide import OpenSlide


def process_missing_images(pth, pthim, missing_images, umpix):
    """Process missing images by converting .ndpi or .svs files to .tif."""
    for idx, missing_image in enumerate(missing_images):
        print(f"{idx + 1} / {len(missing_images)} processing: {missing_image}")
        try:
            # Open the slide
            slide_path = os.path.join(pth, missing_image + '.ndpi')
            wsi = OpenSlide(slide_path)

            # Read the slide region
            svs_img = wsi.read_region(location=(0, 0), level=0, size=wsi.level_dimensions[0]).convert('RGB')

            # Calculate resize factors
            resize_factor_x = umpix / float(wsi.properties['openslide.mpp-x'])
            resize_factor_y = umpix / float(wsi.properties['openslide.mpp-y'])
            resize_dimension = (
                int(np.ceil(wsi.dimensions[0] / resize_factor_x)),
                int(np.ceil(wsi.dimensions[1] / resize_factor_y))
            )

            # Resize and save the image
            svs_img = svs_img.resize(resize_dimension, resample=Image.NEAREST)
            output_path = os.path.join(pthim, missing_image + '.tif')
            svs_img.save(output_path, resolution=1, resolution_unit=1, quality=100, compression=None)
        except Exception as e:
            print(f"Error processing {missing_image}: {e}")


def WSI2tif(pth, resolution, umpix):
    pthim = os.path.join(pth, f'{resolution}')

    # Ensure the image directory exists
    if not os.path.isdir(pthim):
        os.makedirs(pthim)

    # Get the .tif image names
    image_files_tif = glob.glob(os.path.join(pthim, '*.tif'))
    images_names_tif = {os.path.splitext(os.path.basename(image))[0] for image in image_files_tif}

    # Get the .ndpi and .svs image names
    image_files_wsi = glob.glob(os.path.join(pth, '*.ndpi')) + glob.glob(os.path.join(pth, '*.svs'))
    images_names_wsi = {os.path.splitext(os.path.basename(image))[0] for image in image_files_wsi}

    # Compare image names and process missing images
    if images_names_tif != images_names_wsi:
        missing_images = images_names_wsi - images_names_tif
        process_missing_images(pth, pthim, missing_images, umpix)










