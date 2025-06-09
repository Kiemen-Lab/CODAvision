"""
Model Metadata Management for CODAvision

This module provides high-level interfaces for managing model metadata,
including creation, updates, and validation of model configuration.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: March 2025
"""

import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd

from .utils import save_model_metadata as _save_model_metadata
from ..evaluation.visualize import plot_cmap_legend

import logging
logger = logging.getLogger(__name__)


class ModelMetadata:
    """
    A class for managing model metadata with validation and processing.
    
    This class encapsulates all model metadata and provides methods for
    processing whitespace settings, managing class names, and saving
    configuration.
    """
    
    def __init__(
        self,
        model_path: str,
        image_path: str,
        test_path: str,
        whitespace_settings: List[Any],
        model_name: str,
        resolution: Union[int, str],
        colormap: np.ndarray,
        tile_size: int,
        class_names: List[str],
        num_train_tiles: int,
        num_validate_tiles: int,
        num_tissue_analysis: int,
        final_df: pd.DataFrame,
        combined_df: pd.DataFrame,
        model_type: str = "DeepLabV3_plus",
        batch_size: int = 3,
        **kwargs
    ):
        """
        Initialize model metadata.
        
        Args:
            model_path: Path where model data will be saved
            image_path: Path to training images
            test_path: Path to test images
            whitespace_settings: List containing whitespace configuration
            model_name: Name of the model
            resolution: Image resolution (1, 2, 4, or 'TBD')
            colormap: Color map array (N x 3)
            tile_size: Size of training tiles
            class_names: List of class names
            num_train_tiles: Number of training tiles
            num_validate_tiles: Number of validation tiles
            num_tissue_analysis: Number of images for tissue analysis
            final_df: DataFrame with final annotation settings
            combined_df: DataFrame with combined layer information
            model_type: Type of model architecture
            batch_size: Batch size for training
            **kwargs: Additional parameters for custom resolution
        """
        self.model_path = Path(model_path)
        self.image_path = image_path
        self.test_path = test_path
        self.whitespace_settings = whitespace_settings
        self.model_name = model_name
        self.resolution = resolution
        self.colormap = colormap
        self.tile_size = tile_size
        self.class_names = list(class_names)
        self.num_train_tiles = num_train_tiles
        self.num_validate_tiles = num_validate_tiles
        self.num_tissue_analysis = num_tissue_analysis
        self.final_df = final_df
        self.combined_df = combined_df
        self.model_type = self._normalize_model_type(model_type)
        self.batch_size = batch_size
        
        # Additional parameters for custom resolution
        self.custom_params = kwargs
        
        # Calculated values
        self.num_black = None
        self.num_white = None
        
    def _normalize_model_type(self, model_type: str) -> str:
        """Normalize model type name for consistency."""
        if '+' in model_type:
            return model_type.replace('+', '_plus')
        return model_type
    
    def process_whitespace_settings(self) -> None:
        """
        Process whitespace settings when layers are deleted.
        
        This method updates the whitespace settings based on deleted layers
        and recalculates the white and black class indices.
        """
        ws = self.whitespace_settings
        
        # Process deleted layers
        deleted_layers = ws[4].copy() if ws[4] else []
        if deleted_layers:
            deleted_layers.sort(reverse=True)
            
            for layer_idx in deleted_layers:
                if layer_idx <= 0:
                    continue
                    
                # Get the combined class number for the deleted layer
                combined_classes = ws[2].copy()
                layer_order = ws[3].copy()
                old_class_num = combined_classes[layer_idx - 1]
                
                # Update combined classes
                combined_classes[layer_idx - 1] = 1
                combined_classes = [
                    n - 1 if n > old_class_num else n 
                    for n in combined_classes
                ]
                
                # Update layer order
                layer_order = [n for n in layer_order if n != layer_idx]
                
                # Update class names and colormap if needed
                if len(self.class_names) == max(ws[2]):
                    indices_to_keep = [
                        i for i in range(len(self.class_names)) 
                        if i + 1 != old_class_num
                    ]
                    self.class_names = [self.class_names[i] for i in indices_to_keep]
                    self.colormap = self.colormap[indices_to_keep]
                
                # Update whitespace settings
                ws[2] = combined_classes
                ws[3] = layer_order
        
        # Calculate white class index
        self._calculate_white_class_index()
    
    def _calculate_white_class_index(self) -> None:
        """Calculate the index of the whitespace class."""
        ws = self.whitespace_settings
        
        # Default value
        self.num_white = -1
        
        # Try to get from whitespace settings
        if (ws[1] and len(ws[1]) > 0 and 
            isinstance(ws[1][0], int) and ws[1][0] > 0):
            
            whitespace_layer_idx = ws[1][0]
            
            if (ws[2] and isinstance(ws[2], list) and 
                0 <= (whitespace_layer_idx - 1) < len(ws[2]) and
                isinstance(ws[2][whitespace_layer_idx - 1], int)):
                
                self.num_white = ws[2][whitespace_layer_idx - 1]
        
        # Fallback: find 'whitespace' in class names
        if self.num_white == -1:
            try:
                self.num_white = self.class_names.index("whitespace") + 1
            except ValueError:
                # Default to last class if whitespace not found
                if self.class_names:
                    self.num_white = len(self.class_names)
                else:
                    self.num_white = 1
                logger.warning(
                    f"Could not determine whitespace class. "
                    f"Defaulting to class {self.num_white}"
                )
    
    def prepare_class_names(self) -> List[str]:
        """
        Prepare class names for saving, ensuring 'black' is included.
        
        Returns:
            List of class names with 'black' as the last element
        """
        prepared_names = list(self.class_names)
        
        # Remove 'black' if it exists (to avoid duplicates)
        if prepared_names and prepared_names[-1].lower() == "black":
            prepared_names = prepared_names[:-1]
        
        # Always add 'black' as the last class
        prepared_names.append("black")
        
        # Calculate number of black class
        self.num_black = len(prepared_names)
        
        return prepared_names
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary format for saving.
        
        Returns:
            Dictionary containing all metadata
        """
        # Prepare class names
        final_class_names = self.prepare_class_names()
        
        # Base metadata
        metadata = {
            "pthim": self.image_path,
            "pthDL": str(self.model_path),
            "pthtest": self.test_path,
            "WS": self.whitespace_settings,
            "nm": self.model_name,
            "umpix": self.resolution,
            "cmap": self.colormap,
            "sxy": self.tile_size,
            "classNames": final_class_names,
            "ntrain": self.num_train_tiles,
            "nblack": self.num_black,
            "nwhite": self.num_white,
            "nvalidate": self.num_validate_tiles,
            "nTA": self.num_tissue_analysis,
            "final_df": self.final_df,
            "combined_df": self.combined_df,
            "model_type": self.model_type,
            "batch_size": self.batch_size
        }
        
        # Add custom resolution parameters if applicable
        if self.resolution == 'TBD':
            metadata.update({
                "uncomp_train_pth": self.custom_params.get('uncomp_train_pth', ''),
                "uncomp_test_pth": self.custom_params.get('uncomp_test_pth', ''),
                "scale": self.custom_params.get('scale', ''),
                "create_down": self.custom_params.get('create_down', False),
                "downsamp_annotated": self.custom_params.get('downsamp_annotated', False)
            })
        
        return metadata
    
    def save(self) -> None:
        """
        Save model metadata to disk.
        
        This method saves the metadata pickle file and generates the
        color map legend plot.
        """
        # Create directory if needed
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info('Saving model metadata and classification colormap...')
        
        # Process whitespace settings
        self.process_whitespace_settings()
        
        # Get metadata dictionary
        metadata = self.to_dict()
        
        # Save metadata using the utility function
        _save_model_metadata(str(self.model_path), metadata)
        
        # Generate and save color map legend
        legend_path = self.model_path / 'model_color_legend.jpg'
        plot_cmap_legend(
            self.colormap, 
            self.class_names,  # Use original class names without 'black'
            save_path=str(legend_path)
        )
        
        logger.info(f'Model metadata saved to {self.model_path / "net.pkl"}')
        logger.info(f'Color map legend saved to {legend_path}')


def save_model_metadata_gui(
    pthDL: str,
    pthim: str,
    pthtest: str,
    WS: List[Any],
    nm: str,
    umpix: Union[int, str],
    cmap: np.ndarray,
    sxy: int,
    classNames: List[str],
    ntrain: int,
    nvalidate: int,
    nTA: int,
    final_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    model_type: str,
    batch_size: int,
    uncomp_train_pth: str = '',
    uncomp_test_pth: str = '',
    scale: str = '',
    create_down: bool = False,
    downsamp_annotated: bool = False
) -> None:
    """
    Save model metadata from GUI parameters.
    
    This function provides a simplified interface for the GUI to save model
    metadata. It handles parameter validation and delegates to the ModelMetadata
    class for processing.
    
    Args:
        pthDL: Path where the model metadata will be saved
        pthim: Path where the images are located
        pthtest: Path to test images
        WS: List containing whitespace settings
        nm: The name of the model
        umpix: Scaling factor (1, 2, 4, or 'TBD')
        cmap: The color map of the model (N x 3 array)
        sxy: Training tile size
        classNames: List of class names
        ntrain: Number of training tiles
        nvalidate: Number of validation tiles
        nTA: Number of tissue analysis images
        final_df: DataFrame with annotation layer settings
        combined_df: DataFrame with combined layer information
        model_type: Model architecture type
        batch_size: Batch size for training
        uncomp_train_pth: Path to uncompressed training images (custom resolution)
        uncomp_test_pth: Path to uncompressed test images (custom resolution)
        scale: Scaling factor (custom resolution)
        create_down: Whether to create downsampled images
        downsamp_annotated: Whether to downsample annotated images
    """
    # Create metadata object
    metadata = ModelMetadata(
        model_path=pthDL,
        image_path=pthim,
        test_path=pthtest,
        whitespace_settings=WS,
        model_name=nm,
        resolution=umpix,
        colormap=cmap,
        tile_size=sxy,
        class_names=classNames,
        num_train_tiles=ntrain,
        num_validate_tiles=nvalidate,
        num_tissue_analysis=nTA,
        final_df=final_df,
        combined_df=combined_df,
        model_type=model_type,
        batch_size=batch_size,
        uncomp_train_pth=uncomp_train_pth,
        uncomp_test_pth=uncomp_test_pth,
        scale=scale,
        create_down=create_down,
        downsamp_annotated=downsamp_annotated
    )
    
    # Save metadata
    metadata.save()