"""
Whole Slide Image Conversion Utilities

This module provides functionality for converting whole slide images (WSI) between
different formats, primarily for use in tissue analysis pipelines. It supports 
conversion from formats like NDPI and SVS to more common formats like TIFF and PNG.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: April 2025
"""

import os
import glob
import logging
from typing import List, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image
import cv2

# Configure logging
logger = logging.getLogger(__name__)

# Prevent PIL from raising DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None

# Import OpenSlide with appropriate error handling for different environments
try:
    # First try importing OpenSlide normally
    from openslide import OpenSlide

except ImportError:
    try:
        # Try getting script directory
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        # Path to OpenSlide binaries in the "base" subdirectory
        openslide_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'base', 'OpenSlide bin')

        # Add OpenSlide to DLL directory on Windows, or to PATH on other platforms
        if hasattr(os, 'add_dll_directory'):
            # Windows-specific approach for Python 3.8+
            with os.add_dll_directory(openslide_path):
                from openslide import OpenSlide
        else:
            # Alternative approach for other platforms
            if openslide_path not in os.environ['PATH']:
                os.environ['PATH'] = openslide_path + os.pathsep + os.environ['PATH']
            from openslide import OpenSlide

    except ImportError as e:
        logger.warning(f"Failed to import OpenSlide: {e}")
        logger.warning("Some WSI conversion functions may not be available")


class WholeSlideImageConverter:
    """
    Class for converting whole slide images between different formats.
    
    This class provides methods to convert whole slide images from formats like
    NDPI and SVS to more common formats like TIFF and PNG, with options for
    scaling and resolution adjustments.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the WholeSlideImageConverter.
        
        Args:
            verbose: Whether to print status messages during conversion
        """
        self.verbose = verbose
    
    def _log(self, message: str, level: str = "info") -> None:
        """
        Log a message based on verbosity setting.
        
        Args:
            message: The message to log
            level: The log level ("info", "warning", "error")
        """
        if self.verbose:
            print(message)
            
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
    
    def convert_to_png(self, 
                       input_path: str, 
                       resolution: str, 
                       umpix: float) -> str:
        """
        Convert whole slide images to PNG format.
        
        This function creates a subdirectory named after the resolution and converts
        whole slide images (WSI) from NDPI or SVS format to PNG format. It resizes
        the images according to the specified microns per pixel.
        
        Args:
            input_path: Path to the directory containing the WSI files
            resolution: String representing the resolution to use (e.g., '5x')
            umpix: Microns per pixel value for resizing
            
        Returns:
            Path to the directory containing the converted images
        """
        self._log(f'Making down-sampled images in PNG format...')
        
        # Create output directory
        output_path = os.path.join(input_path, f'{resolution}')
        os.makedirs(output_path, exist_ok=True)
        
        # Find existing PNG files
        existing_pngs = glob.glob(os.path.join(output_path, '*.png'))
        existing_names = {os.path.splitext(os.path.basename(f))[0] for f in existing_pngs}
        
        # Find WSI files to convert
        wsi_files = glob.glob(os.path.join(input_path, '*.ndpi')) + glob.glob(os.path.join(input_path, '*.svs'))
        if not wsi_files:
            self._log("No .ndpi or .svs files found in the directory.", level="warning")
            return output_path
            
        wsi_names = {os.path.splitext(os.path.basename(f))[0] for f in wsi_files}
        
        # Determine which files need to be converted
        missing_images = wsi_names - existing_names
        if not missing_images:
            self._log("All images already converted to PNG format.")
            return output_path
            
        # Process missing images
        self._process_missing_images_png(input_path, output_path, missing_images, umpix)
        
        return output_path
    
    def _process_missing_images_png(self, 
                                   input_path: str, 
                                   output_path: str, 
                                   missing_images: set, 
                                   umpix: float) -> None:
        """
        Process images that need to be converted to PNG format.
        
        Args:
            input_path: Directory containing source WSI files
            output_path: Directory where converted PNGs will be saved
            missing_images: Set of filenames (without extension) to convert
            umpix: Microns per pixel value for resizing
        """
        for idx, image_name in enumerate(missing_images):
            self._log(f"  {idx + 1} / {len(missing_images)} processing: {image_name}")
            
            try:
                # Open the slide
                slide_path = os.path.join(input_path, f"{image_name}.ndpi")
                if not os.path.exists(slide_path):
                    slide_path = os.path.join(input_path, f"{image_name}.svs")
                
                wsi = OpenSlide(slide_path)
                
                # Read the whole slide at level 0 (highest resolution)
                slide_img = wsi.read_region(
                    location=(0, 0), 
                    level=0, 
                    size=wsi.level_dimensions[0]
                ).convert('RGB')
                
                # Calculate resize factors
                resize_factor_x = umpix / float(wsi.properties['openslide.mpp-x'])
                resize_factor_y = umpix / float(wsi.properties['openslide.mpp-y'])
                
                # Calculate new dimensions
                new_width = int(np.ceil(wsi.dimensions[0] / resize_factor_x))
                new_height = int(np.ceil(wsi.dimensions[1] / resize_factor_y))
                new_dimensions = (new_width, new_height)
                
                # Resize the image
                slide_img = slide_img.resize(new_dimensions, resample=Image.NEAREST)
                
                # Save as PNG
                output_file = os.path.join(output_path, f"{image_name}.png")
                slide_img.save(
                    output_file, 
                    resolution=1, 
                    resolution_unit=1, 
                    quality=100, 
                    compression=None
                )
                
                self._log(f"  Successfully converted {image_name} to PNG")
                
            except Exception as e:
                self._log(f"Error processing {image_name}: {e}", level="error")
    
    def convert_to_tiff(self,
                        input_path: str, 
                        resolution: str, 
                        umpix: Union[float, str], 
                        image_format: str = '.ndpi', 
                        scale: float = 0, 
                        output_path: str = '') -> str:
        """
        Convert whole slide images to TIFF format.
        
        This function creates a subdirectory and converts whole slide images (WSI)
        from various formats to TIFF format. It supports different input formats and
        provides options for custom scaling and output directory.
        
        Args:
            input_path: Path to the directory containing the WSI files
            resolution: String representing the resolution (e.g., '5x') or 'Custom'
            umpix: Microns per pixel value or 'TBD' for custom scaling
            image_format: Format of the source images (default: '.ndpi')
            scale: Custom scaling factor (used when resolution is 'Custom')
            output_path: Custom output directory (optional)
            
        Returns:
            Path to the directory containing the converted images
        """
        self._log('Making down-sampled images in TIFF format...')
        
        if scale == 0:
            # Standard resolution-based path
            tiff_path = os.path.join(input_path, f'{resolution}')
            os.makedirs(tiff_path, exist_ok=True)
            
            # Find existing TIFF files
            existing_tiffs = glob.glob(os.path.join(tiff_path, '*.tif'))
            existing_names = {os.path.splitext(os.path.basename(f))[0] for f in existing_tiffs}
            
            # Find WSI files to convert
            wsi_files = glob.glob(os.path.join(input_path, '*.ndpi')) + glob.glob(os.path.join(input_path, '*.svs'))
            if not wsi_files:
                self._log("No .ndpi or .svs files found in the directory.", level="warning")
                return tiff_path
                
            wsi_names = {os.path.splitext(os.path.basename(f))[0] for f in wsi_files}
            
            # Determine which files need to be converted
            missing_images = wsi_names - existing_names
            if not missing_images:
                self._log("  All down-sampled images already exist in the directory.")
                return tiff_path
                
            # Process missing images
            self._process_missing_images_tiff(input_path, tiff_path, missing_images, umpix)
            
        else:
            # Custom scale output path
            tiff_path = os.path.join(output_path, f'Custom_Scale_{scale}')
            os.makedirs(tiff_path, exist_ok=True)
            
            # Find existing TIFF files
            existing_tiffs = glob.glob(os.path.join(tiff_path, '*.tif'))
            existing_names = {os.path.splitext(os.path.basename(f))[0] for f in existing_tiffs}
            
            # Process based on image format
            self._process_custom_scale_images(
                input_path, 
                tiff_path, 
                existing_names,
                image_format, 
                scale
            )
            
        return tiff_path
    
    def _process_missing_images_tiff(self,
                                    input_path: str, 
                                    output_path: str, 
                                    missing_images: set, 
                                    umpix: float) -> None:
        """
        Process images that need to be converted to TIFF format.
        
        Args:
            input_path: Directory containing source WSI files
            output_path: Directory where converted TIFFs will be saved
            missing_images: Set of filenames (without extension) to convert
            umpix: Microns per pixel value for resizing
        """
        for idx, image_name in enumerate(missing_images):
            self._log(f"  {idx + 1} / {len(missing_images)} processing: {image_name}")
            
            try:
                # Open the slide
                slide_path = os.path.join(input_path, f"{image_name}.ndpi")
                if not os.path.exists(slide_path):
                    slide_path = os.path.join(input_path, f"{image_name}.svs")
                
                wsi = OpenSlide(slide_path)
                
                # Read the whole slide at level 0 (highest resolution)
                slide_img = wsi.read_region(
                    location=(0, 0), 
                    level=0, 
                    size=wsi.level_dimensions[0]
                ).convert('RGB')
                
                # Calculate resize factors
                resize_factor_x = umpix / float(wsi.properties['openslide.mpp-x'])
                resize_factor_y = umpix / float(wsi.properties['openslide.mpp-y'])
                
                # Calculate new dimensions
                new_width = int(np.ceil(wsi.dimensions[0] / resize_factor_x))
                new_height = int(np.ceil(wsi.dimensions[1] / resize_factor_y))
                new_dimensions = (new_width, new_height)
                
                # Resize the image
                slide_img = slide_img.resize(new_dimensions, resample=Image.NEAREST)
                
                # Save as TIFF
                output_file = os.path.join(output_path, f"{image_name}.tif")
                slide_img.save(
                    output_file, 
                    resolution=1, 
                    resolution_unit=1, 
                    quality=100, 
                    compression=None
                )
                
                self._log(f"  Successfully converted {image_name} to TIFF")
                
            except Exception as e:
                self._log(f"Error processing {image_name}: {e}", level="error")
    
    def _process_custom_scale_images(self,
                                    input_path: str,
                                    output_path: str,
                                    existing_names: set,
                                    image_format: str,
                                    scale: float) -> None:
        """
        Process images with custom scaling factor.
        
        Args:
            input_path: Directory containing source files
            output_path: Directory where converted files will be saved
            existing_names: Set of filenames already converted
            image_format: Format of the source files ('.ndpi', '.dcm', etc.)
            scale: Custom scaling factor
        """
        # Find files with the specified format
        if image_format in ['.ndpi', '.svs']:
            source_files = (glob.glob(os.path.join(input_path, '*.ndpi')) + 
                           glob.glob(os.path.join(input_path, '*.svs')))
            
            if not source_files:
                self._log(f"No {image_format} files found in the directory.", level="warning")
                return
                
            source_names = {os.path.splitext(os.path.basename(f))[0] for f in source_files}
            missing_images = source_names - existing_names
            
            # Process ndpi/svs files
            for idx, image_name in enumerate(missing_images):
                self._log(f"  {idx + 1} / {len(missing_images)} processing: {image_name}")
                
                try:
                    # Find correct extension
                    if os.path.exists(os.path.join(input_path, f"{image_name}.ndpi")):
                        slide_path = os.path.join(input_path, f"{image_name}.ndpi")
                    else:
                        slide_path = os.path.join(input_path, f"{image_name}.svs")
                    
                    wsi = OpenSlide(slide_path)
                    
                    # Read the whole slide
                    slide_img = wsi.read_region(
                        location=(0, 0), 
                        level=0, 
                        size=wsi.level_dimensions[0]
                    ).convert('RGB')
                    
                    # Calculate new dimensions
                    new_width = int(np.ceil(wsi.dimensions[0] / scale))
                    new_height = int(np.ceil(wsi.dimensions[1] / scale))
                    new_dimensions = (new_width, new_height)
                    
                    # Resize the image
                    slide_img = slide_img.resize(new_dimensions, resample=Image.NEAREST)
                    
                    # Save as TIFF
                    output_file = os.path.join(output_path, f"{image_name}.tif")
                    slide_img.save(
                        output_file, 
                        resolution=1, 
                        resolution_unit=1, 
                        quality=100, 
                        compression=None
                    )
                    
                except Exception as e:
                    self._log(f"Error processing {image_name}: {e}", level="error")
                    
        elif image_format == '.dcm':
            # Handle DICOM files
            try:
                import pydicom as dicom
                
                source_files = glob.glob(os.path.join(input_path, '*.dcm'))
                if not source_files:
                    self._log("No .dcm files found in the directory.", level="warning")
                    return
                    
                source_names = {os.path.splitext(os.path.basename(f))[0] for f in source_files}
                missing_images = source_names - existing_names
                
                for idx, image_name in enumerate(missing_images):
                    self._log(f"  {idx + 1} / {len(missing_images)} processing: {image_name}")
                    
                    try:
                        # Read DICOM file
                        image_path = os.path.join(input_path, f"{image_name}.dcm")
                        ds = dicom.dcmread(image_path)
                        pixel_array = ds.pixel_array
                        
                        # Normalize to 8-bit
                        image_8bit = np.uint8((pixel_array / np.max(pixel_array) * 255))
                        
                        # Calculate new dimensions
                        new_width = int(np.ceil(image_8bit.shape[1] / scale))
                        new_height = int(np.ceil(image_8bit.shape[0] / scale))
                        
                        # Convert to PIL Image and resize
                        image_pil = Image.fromarray(image_8bit)
                        image_pil = image_pil.resize((new_width, new_height), resample=Image.NEAREST)
                        
                        # Save as TIFF
                        output_file = os.path.join(output_path, f"{image_name}.tif")
                        image_pil.save(
                            output_file, 
                            resolution=1, 
                            resolution_unit=1, 
                            quality=100, 
                            compression=None
                        )
                        
                    except Exception as e:
                        self._log(f"Error processing {image_name}: {e}", level="error")
                        
            except ImportError:
                self._log("pydicom library not found. Cannot process DICOM files.", level="error")
                
        else:
            # Handle other image formats
            source_files = glob.glob(os.path.join(input_path, f'*{image_format}'))
            if not source_files:
                self._log(f"No {image_format} files found in the directory.", level="warning")
                return
                
            source_names = {os.path.splitext(os.path.basename(f))[0] for f in source_files}
            missing_images = source_names - existing_names
            
            for idx, image_name in enumerate(missing_images):
                self._log(f"  {idx + 1} / {len(missing_images)} processing: {image_name}")
                
                try:
                    # Read image
                    image_path = os.path.join(input_path, f"{image_name}{image_format}")
                    image = Image.open(image_path)
                    image_array = np.array(image)
                    
                    # Calculate new dimensions
                    new_width = int(np.ceil(image_array.shape[1] / scale))
                    new_height = int(np.ceil(image_array.shape[0] / scale))
                    
                    # Resize
                    image = image.resize((new_width, new_height), resample=Image.NEAREST)
                    
                    # Save as TIFF
                    output_file = os.path.join(output_path, f"{image_name}.tif")
                    image.save(
                        output_file, 
                        resolution=1, 
                        resolution_unit=1, 
                        quality=100, 
                        compression=None
                    )
                    
                except Exception as e:
                    self._log(f"Error processing {image_name}: {e}", level="error")


# Legacy functions for backward compatibility
def WSI2png(pth: str, resolution: str, umpix: float) -> str:
    """
    Convert whole slide images to PNG format (legacy function).
    
    Args:
        pth: Path to the directory containing the WSI files
        resolution: String representing the resolution to use (e.g., '5x')
        umpix: Microns per pixel value for resizing
        
    Returns:
        Path to the directory containing the converted images
    """
    converter = WholeSlideImageConverter()
    return converter.convert_to_png(pth, resolution, umpix)


def WSI2tif(pth: str, 
            resolution: str, 
            umpix: Union[float, str], 
            image_format: str = '.ndpi', 
            scale: float = 0, 
            outpth: str = '') -> str:
    """
    Convert whole slide images to TIFF format (legacy function).
    
    Args:
        pth: Path to the directory containing the WSI files
        resolution: String representing the resolution (e.g., '5x') or 'Custom'
        umpix: Microns per pixel value or 'TBD' for custom scaling
        image_format: Format of the source images (default: '.ndpi')
        scale: Custom scaling factor (used when resolution is 'Custom')
        outpth: Custom output directory (optional)
        
    Returns:
        Path to the directory containing the converted images
    """
    converter = WholeSlideImageConverter()
    return converter.convert_to_tiff(pth, resolution, umpix, image_format, scale, outpth)