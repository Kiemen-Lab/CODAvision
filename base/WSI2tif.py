"""

Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Date: August 14, 2024
"""

from PIL import Image
import numpy as np
import os
import glob
import pydicom as dicom
# from openslide import OpenSlide

# Add the OpenSlide DLL directory
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()  # Fallback to the current working direc
openslide_path = os.path.join(script_dir, 'OpenSlide bin')

if hasattr(os, 'add_dll_directory'):
    # Python 3.8+
    with os.add_dll_directory(openslide_path):
        from openslide import OpenSlide
else:
    # Earlier Python versions
    if openslide_path not in os.environ['PATH']:
        os.environ['PATH'] = openslide_path + os.pathsep + os.environ['PATH']
    from openslide import OpenSlide

def process_missing_images(pth, pthim, missing_images, umpix):
    """Process missing images by converting .ndpi or .svs files to .tif."""
    for idx, missing_image in enumerate(missing_images):
        print(f"  {idx + 1} / {len(missing_images)} processing: {missing_image}")
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


def WSI2tif(pth, resolution, umpix, image_format = '.ndpi', scale = 0, outpth =''):
    print('Making down-sampled images...')
    if scale == 0:
        pthim = os.path.join(pth, f'{resolution}')

        # Ensure the image directory exists
        if not os.path.isdir(pthim):
            os.makedirs(pthim)

        # Get the .tif image names
        image_files_tif = glob.glob(os.path.join(pthim, '*.tif'))
        images_names_tif = {os.path.splitext(os.path.basename(image))[0] for image in image_files_tif}

        # Get the .ndpi and .svs image names
        image_files_wsi = glob.glob(os.path.join(pth, '*.ndpi')) + glob.glob(os.path.join(pth, '*.svs'))
        if not image_files_wsi:
            print("No .ndpi or .svs files found in the directory.")
            return
        images_names_wsi = {os.path.splitext(os.path.basename(image))[0] for image in image_files_wsi}

        # Compare image names and process missing images
        if images_names_tif != images_names_wsi:
            missing_images = images_names_wsi - images_names_tif
            process_missing_images(pth, pthim, missing_images, umpix)
        else:
            print("  All down-sampled images already exist in the directory.")
    else:
        pthim = os.path.join(outpth, 'Custom_Scale_'+str(scale))

        # Ensure the image directory exists
        if not os.path.isdir(pthim):
            os.makedirs(pthim)

        # Get the .tif image names
        image_files_tif = glob.glob(os.path.join(pthim, '*.tif'))
        images_names_tif = {os.path.splitext(os.path.basename(image))[0] for image in image_files_tif}
        if image_format == '.ndpi' or image_format =='.svs':
            # Get the .ndpi and .svs image names
            image_files_wsi = glob.glob(os.path.join(pth, '*.ndpi')) + glob.glob(os.path.join(pth, '*.svs'))
            if not image_files_wsi:
                print("No .ndpi or .svs files found in the directory.")
                return
            images_names_wsi = {os.path.splitext(os.path.basename(image))[0] for image in image_files_wsi}

            # Compare image names and process missing images
            if images_names_tif != images_names_wsi:
                missing_images = images_names_wsi - images_names_tif
                for idx, missing_image in enumerate(missing_images):
                    print(f"  {idx + 1} / {len(missing_images)} processing: {missing_image}")
                    missing_images = images_names_wsi - images_names_tif

                    slide_path = os.path.join(pth, missing_image + '.ndpi')
                    wsi = OpenSlide(slide_path)

                    # Read the slide region
                    svs_img = wsi.read_region(location=(0, 0), level=0, size=wsi.level_dimensions[0]).convert('RGB')
                    resize_dimension = (
                        int(np.ceil(wsi.dimensions[0] / scale)),
                        int(np.ceil(wsi.dimensions[1] / scale))
                    )

                    # Resize and save the image
                    svs_img = svs_img.resize(resize_dimension, resample=Image.NEAREST)
                    output_path = os.path.join(pthim, missing_image + '.tif')
                    svs_img.save(output_path, resolution=1, resolution_unit=1, quality=100, compression=None)
            else:
                print("  All down-sampled images already exist in the directory.")
        elif image_format == '.dcm':
            # Get the .dcm image names
            image_files_dcm = glob.glob(os.path.join(pth, '*.dcm'))
            if not image_files_dcm:
                print("No .dcm files found in the directory.")
                return
            images_names_dcm = {os.path.splitext(os.path.basename(image))[0] for image in image_files_dcm}
            # Compare image names and process missing images
            if images_names_tif != images_names_dcm:
                missing_images = images_names_dcm - images_names_tif
                for idx, missing_image in enumerate(missing_images):
                    print(f"  {idx + 1} / {len(missing_images)} processing: {missing_image}")

                    # Open the slide
                    image_path = os.path.join(pth, missing_image + '.dcm')
                    ds = dicom.dcmread(image_path)
                    pixel_array_numpy = ds.pixel_array
                    image8b = np.uint8((pixel_array_numpy / np.max(pixel_array_numpy) * 255))
                    resize_dimension = (
                        int(np.ceil(image8b.shape[0] / scale)),
                        int(np.ceil(image8b.shape[1] / scale))
                    )
                    image8b = Image.fromarray(image8b)
                    # Resize and save the image
                    image8b = image8b.resize(resize_dimension, resample=Image.NEAREST)
                    output_path = os.path.join(pthim, missing_image + '.tif')
                    image8b.save(output_path, resolution=1, resolution_unit=1, quality=100, compression=None)
            else:
                print("  All down-sampled images already exist in the directory.")
        else:
            # Get the image names
            image_files_wsi = glob.glob(os.path.join(pth, '*'+image_format))
            if not image_files_wsi:
                print(f"No {image_format} files found in the directory.")
                return
            images_names_wsi = {os.path.splitext(os.path.basename(image))[0] for image in image_files_wsi}
            # Compare image names and process missing images
            if images_names_tif != images_names_wsi:
                missing_images = images_names_wsi - images_names_tif
                for idx, missing_image in enumerate(missing_images):
                    print(f"  {idx + 1} / {len(missing_images)} processing: {missing_image}")
                    try:
                        # Open the slide
                        image_path = os.path.join(pth, missing_image + image_format)
                        image = Image.open(image_path)
                        image = np.array(image)
                        resize_dimension = (
                            int(np.ceil(image.shape[0] / scale)),
                            int(np.ceil(image.shape[1] / scale))
                        )
                        image = Image.fromarray(image)
                        # Resize and save the image
                        image = image.resize(resize_dimension, resample=Image.NEAREST)
                        output_path = os.path.join(pthim, missing_image + '.tif')
                        image.save(output_path, resolution=1, resolution_unit=1, quality=100, compression=None)
                    except Exception as e:
                        print(f"Error processing {missing_image}: {e}")
            else:
                print("  All down-sampled images already exist in the directory.")



if __name__ == '__main__':
    path = r'\\10.99.68.52\Kiemendata\Valentina Matos\tissues for methods paper\slides scanned from bispecific study'
    resolution = '5x'
    umpix = 2
    WSI2tif(path, resolution, umpix)