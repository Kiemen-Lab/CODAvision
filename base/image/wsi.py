"""
Whole Slide Image (WSI) Processing Module for CODAvision

This module provides functionality for converting and downsampling whole slide images
from various formats (NDPI, SVS, DCM) to TIFF format with specified resolutions.
"""

import os
import platform
import ctypes
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import pydicom as dicom

logger = logging.getLogger(__name__)

# Disable PIL image size limit
Image.MAX_IMAGE_PIXELS = None


class OpenSlideLoader:
    """Handles platform-specific loading of OpenSlide library."""
    
    _openslide = None
    
    @classmethod
    def get_openslide(cls):
        """
        Get OpenSlide module, loading it if necessary.
        
        Returns:
            OpenSlide module
            
        Raises:
            ImportError: If OpenSlide cannot be loaded
        """
        if cls._openslide is not None:
            return cls._openslide
            
        try:
            from openslide import OpenSlide
            cls._openslide = OpenSlide
            return OpenSlide
        except ImportError:
            cls._openslide = cls._load_openslide_manually()
            return cls._openslide
    
    @classmethod
    def _load_openslide_manually(cls):
        """Load OpenSlide manually based on platform."""
        system_platform = platform.system()
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
            
        openslide_dll_path = os.path.join(os.path.dirname(script_dir), 'OpenSlide bin')
        
        if system_platform == "Windows":
            return cls._load_windows_openslide(openslide_dll_path)
        elif system_platform == "Darwin":  # macOS
            return cls._load_macos_openslide(openslide_dll_path)
        else:
            raise ImportError(f"Unsupported platform: {system_platform}")
    
    @staticmethod
    def _load_windows_openslide(dll_path: str):
        """Load OpenSlide on Windows."""
        dll_file = os.path.join(dll_path, 'libopenslide-1.dll')
        
        if hasattr(os, 'add_dll_directory'):
            # Python 3.8+
            with os.add_dll_directory(dll_path):
                ctypes.cdll.LoadLibrary(dll_file)
                from openslide import OpenSlide
                return OpenSlide
        else:
            # Older Python versions
            if dll_path not in os.environ['PATH']:
                os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
            ctypes.cdll.LoadLibrary(dll_file)
            from openslide import OpenSlide
            return OpenSlide
    
    @staticmethod
    def _load_macos_openslide(dll_path: str):
        """Load OpenSlide on macOS."""
        dylib_file = os.path.join(dll_path, 'libopenslide.1.dylib')
        try:
            ctypes.cdll.LoadLibrary(dylib_file)
            from openslide import OpenSlide
            return OpenSlide
        except OSError as e:
            raise ImportError(f"Failed to load {dylib_file} on macOS. Ensure it is installed.") from e


class ImageConverter(ABC):
    """Abstract base class for image format converters."""
    
    @abstractmethod
    def can_convert(self, file_path: Path) -> bool:
        """Check if this converter can handle the given file."""
        pass
    
    @abstractmethod
    def convert(self, input_path: Path, output_path: Path, scale: float) -> None:
        """Convert the image to TIFF format with scaling."""
        pass


class WSIConverter(ImageConverter):
    """Converter for Whole Slide Image formats (NDPI, SVS)."""
    
    def __init__(self, umpix: float = 1.0):
        self.umpix = umpix
        self.OpenSlide = OpenSlideLoader.get_openslide()
    
    def can_convert(self, file_path: Path) -> bool:
        """Check if file is a WSI format."""
        return file_path.suffix.lower() in ['.ndpi', '.svs']
    
    def convert(self, input_path: Path, output_path: Path, scale: Optional[float] = None) -> None:
        """Convert WSI to TIFF with optional scaling."""
        wsi = None
        try:
            wsi = self.OpenSlide(str(input_path))

            # Read the full image
            image = wsi.read_region(location=(0, 0), level=0, size=wsi.level_dimensions[0]).convert('RGB')

            # Calculate resize dimensions
            if scale is not None:
                resize_dimension = (
                    int(np.ceil(wsi.dimensions[0] / scale)),
                    int(np.ceil(wsi.dimensions[1] / scale))
                )
            else:
                # Use microns per pixel scaling
                resize_factor_x = self.umpix / float(wsi.properties.get('openslide.mpp-x', 1.0))
                resize_factor_y = self.umpix / float(wsi.properties.get('openslide.mpp-y', 1.0))
                resize_dimension = (
                    int(np.ceil(wsi.dimensions[0] / resize_factor_x)),
                    int(np.ceil(wsi.dimensions[1] / resize_factor_y))
                )

            # Resize and save
            image = image.resize(resize_dimension, resample=Image.NEAREST)
            image.save(str(output_path), resolution=1, resolution_unit=1, quality=100, compression=None)

        except Exception as e:
            logger.error(f"Error converting WSI {input_path}: {e}")
            raise
        finally:
            # Always close the OpenSlide object to free file descriptors
            if wsi is not None:
                wsi.close()


class DICOMConverter(ImageConverter):
    """Converter for DICOM format images."""
    
    def can_convert(self, file_path: Path) -> bool:
        """Check if file is DICOM format."""
        return file_path.suffix.lower() == '.dcm'
    
    def convert(self, input_path: Path, output_path: Path, scale: float) -> None:
        """Convert DICOM to TIFF with scaling."""
        image = None
        try:
            ds = dicom.dcmread(str(input_path))
            pixel_array = ds.pixel_array

            # Normalize to 8-bit
            image_8bit = np.uint8((pixel_array / np.max(pixel_array)) * 255)

            # Calculate resize dimensions
            resize_dimension = (
                int(np.ceil(image_8bit.shape[1] / scale)),
                int(np.ceil(image_8bit.shape[0] / scale))
            )

            # Convert to PIL Image, resize, and save
            image = Image.fromarray(image_8bit)
            image = image.resize(resize_dimension, resample=Image.NEAREST)
            image.save(str(output_path), resolution=1, resolution_unit=1, quality=100, compression=None)

        except Exception as e:
            logger.error(f"Error converting DICOM {input_path}: {e}")
            raise
        finally:
            # Explicitly close the image to release any resources
            if image is not None:
                image.close()


class StandardImageConverter(ImageConverter):
    """Converter for standard image formats (TIFF, JPG, PNG)."""
    
    def can_convert(self, file_path: Path) -> bool:
        """Check if file is a standard image format."""
        return file_path.suffix.lower() in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
    
    def convert(self, input_path: Path, output_path: Path, scale: float) -> None:
        """Convert standard image to TIFF with scaling."""
        try:
            with Image.open(str(input_path)) as image:
                # Calculate resize dimensions
                resize_dimension = (
                    int(np.ceil(image.width / scale)),
                    int(np.ceil(image.height / scale))
                )

                # Resize
                resized_image = image.resize(resize_dimension, resample=Image.NEAREST)

            # Save outside context manager to avoid issues with closed file
            resized_image.save(str(output_path), resolution=1, resolution_unit=1, quality=100, compression=None)

        except Exception as e:
            logger.error(f"Error converting image {input_path}: {e}")
            raise


class WSIProcessor:
    """
    Main class for processing Whole Slide Images.
    
    This class handles the conversion and downsampling of WSI files
    from various formats to TIFF format.
    """
    
    def __init__(self, resolution: str, umpix: float = 1.0):
        """
        Initialize the WSI processor.
        
        Args:
            resolution: Target resolution (e.g., '10x', '5x', '1x', 'Custom')
            umpix: Microns per pixel for resolution-based scaling
        """
        self.resolution = resolution
        self.umpix = umpix
        self.converters = [
            WSIConverter(umpix),
            DICOMConverter(),
            StandardImageConverter()
        ]
    
    def process_directory(
        self,
        source_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        image_format: str = '.ndpi',
        scale: Optional[float] = None,
        image_set_type: Optional[str] = None
    ) -> None:
        """
        Process all images in a directory.
        
        Args:
            source_path: Source directory containing images
            output_path: Output directory for converted images (defaults to resolution subdirectory)
            image_format: Expected image format extension
            scale: Custom scale factor (overrides resolution-based scaling)
            image_set_type: Type of image set being processed ('training' or 'testing')
        """
        source_path = Path(source_path)
        
        # Determine output path
        if output_path is None:
            if scale is not None:
                output_path = source_path / f'Custom_Scale_{scale}'
            else:
                output_path = source_path / self.resolution
        else:
            output_path = Path(output_path)
            if scale is not None:
                output_path = output_path / f'Custom_Scale_{scale}'
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Log with image set type if provided
        if image_set_type:
            logger.info(f'Making down-sampled images for {image_set_type} set...')
        else:
            logger.info('Making down-sampled images...')
        
        # Find images to process
        images_to_process = self._find_images_to_process(source_path, output_path, image_format)
        
        if not images_to_process:
            logger.info("  All down-sampled images already exist in the directory.")
            return
        
        # Process each image
        for idx, image_path in enumerate(images_to_process, 1):
            logger.info(f"  {idx} / {len(images_to_process)} processing: {image_path.name}")
            
            try:
                output_file = output_path / f"{image_path.stem}.tif"
                self._convert_image(image_path, output_file, scale)
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
    
    def _find_images_to_process(
        self,
        source_path: Path,
        output_path: Path,
        image_format: str
    ) -> List[Path]:
        """Find images that need processing."""
        # Get existing output files
        existing_outputs = {f.stem for f in output_path.glob('*.tif')}
        
        # Find source images based on format
        if image_format in ['.ndpi', '.svs']:
            source_images = list(source_path.glob('*.ndpi')) + list(source_path.glob('*.svs'))
        else:
            pattern = f'*{image_format}'
            source_images = list(source_path.glob(pattern))
        
        # Return images that don't have corresponding output
        return [img for img in source_images if img.stem not in existing_outputs]
    
    def _convert_image(self, input_path: Path, output_path: Path, scale: Optional[float]) -> None:
        """Convert a single image using the appropriate converter."""
        for converter in self.converters:
            if converter.can_convert(input_path):
                if isinstance(converter, WSIConverter) and scale is None:
                    # Use resolution-based scaling for WSI
                    converter.convert(input_path, output_path, scale=None)
                else:
                    # Use explicit scale factor
                    converter.convert(input_path, output_path, scale=scale or 1.0)
                return
        
        raise ValueError(f"No converter available for file type: {input_path.suffix}")


def WSI2tif(
    pth: str,
    resolution: str,
    umpix: float,
    image_format: str = '.ndpi',
    scale: float = 0,
    outpth: str = '',
    image_set_type: Optional[str] = None
) -> None:
    """
    Convert and downsample Whole Slide Images to TIFF format.
    
    This function maintains backward compatibility with the original implementation
    while using the refactored class-based approach.
    
    Args:
        pth: Source directory path
        resolution: Target resolution (e.g., '10x', '5x', '1x')
        umpix: Microns per pixel
        image_format: Input image format (default: '.ndpi')
        scale: Custom scale factor (0 for resolution-based scaling)
        outpth: Output directory path (empty string uses default)
        image_set_type: Type of image set being processed ('training' or 'testing')
    """
    processor = WSIProcessor(resolution, umpix)
    
    # Handle scale and output path logic
    if scale == 0:
        processor.process_directory(pth, image_format=image_format, image_set_type=image_set_type)
    else:
        output_path = outpth if outpth else pth
        processor.process_directory(pth, output_path, image_format=image_format, scale=scale, image_set_type=image_set_type)