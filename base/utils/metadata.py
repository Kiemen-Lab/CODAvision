"""
Model Metadata Utilities for CODAvision

This module provides functions for saving and handling model metadata used by the
CODAvision application, including configuration, parameters, and visualization.

Author: Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
Updated: April 2025
"""

import os
import pickle
from typing import Dict, List, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from base.utils.visualization import plot_cmap_legend


def save_model_metadata_gui(
    model_path: str,
    image_path: str,
    test_path: str,
    whitespace_settings: List,
    model_name: str,
    scale_factor: Union[str, int, float],
    color_map: np.ndarray,
    image_size: int,
    class_names: List[str],
    num_training_tiles: int,
    num_validation_tiles: int,
    num_tissue_analysis: int,
    final_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    model_type: str,
    batch_size: int,
    uncomp_train_pth: str = '',
    uncomp_test_pth: str = '',
    scale: str = '',
    create_down: str = '',
    downsamp_annotated: str = ''
) -> None:
    """
    Save model metadata to a pickle file and generate a color map legend plot.

    This function saves model configuration, parameters, and visualization settings
    to a pickle file for later use by the CODAvision application. It also creates
    a color map legend visualization based on the provided model configuration.

    Args:
        model_path: Path where the model metadata will be saved
        image_path: Path where the training images are located
        test_path: Path where the testing images are located
        whitespace_settings: List containing whitespace removal options, tissue order, 
                           tissues being deleted, and whitespace distribution
        model_name: Name of the model
        scale_factor: Scaling factor (can be a number or 'TBD' string)
        color_map: Color map array with RGB values for each class
        image_size: Size of training tiles
        class_names: List of class names
        num_training_tiles: Number of training tiles to generate
        num_validation_tiles: Number of validation tiles to generate
        num_tissue_analysis: Number of images to analyze for tissue mask evaluation
        final_df: DataFrame with the final annotation layer settings
        combined_df: DataFrame with the names and colors of layers for classification
        model_type: Type of model architecture to use
        batch_size: Batch size for training
        uncomp_train_pth: Path to uncompressed training images (optional)
        uncomp_test_pth: Path to uncompressed testing images (optional)
        scale: Scale factor for image resizing (optional)
        create_down: Whether to create downsampled images (optional)
        downsamp_annotated: Whether to downsample annotated images (optional)

    Returns:
        None
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    print('Saving model metadata and classification colormap...')

    # Process class names
    if class_names[-1] != "black":
        class_names.append("black")

    if class_names[-1] == "black":
        class_names.pop()

    # Process deleted annotations
    ndelete = whitespace_settings[4].copy()
    if isinstance(ndelete, list):
        ndelete.sort(reverse=True)
        if ndelete:
            for b in ndelete:
                ncombine = whitespace_settings[2].copy()
                nload = whitespace_settings[3].copy()
                oldnum = ncombine[b - 1]
                ncombine[b - 1] = 1
                ncombine = [n - 1 if n > oldnum else n for n in ncombine]
                nload = [n for n in nload if n != b]

                if len(class_names) == max(whitespace_settings[2]):
                    print(class_names)
                    zz = [i for i in range(len(class_names)) if i + 1 not in [oldnum]]
                    class_names = [class_names[i] for i in zz]
                    print(class_names)
                    color_map = color_map[zz]
                    print(color_map)

                whitespace_settings[2] = ncombine

            whitespace_settings[2] = ncombine
            whitespace_settings[3] = nload

    # Set whitespace and black pixel indices
    nwhite = whitespace_settings[2][whitespace_settings[1][0] - 1]
    
    print(f'Max WS[2]: {max(whitespace_settings[2])}')
    print(f'Classnames: {class_names}')
    
    if max(whitespace_settings[2]) != len(class_names):
        raise ValueError('The length of classNames does not match the number of classes specified in WS[2].')
    
    if class_names[-1] != "black":
        class_names.append("black")
    
    nblack = len(class_names)

    # Data file path for saving metadata
    data_file = os.path.join(model_path, 'net.pkl')

    # Fix model type if it contains a plus sign
    if '+' in model_type:
        model_type = model_type.replace('+', '_plus')

    # Update existing data file or create a new one
    if os.path.exists(data_file):
        print('Net file already exists, updating data...')
        try:
            with open(data_file, 'rb') as f:
                existing_data = pickle.load(f)
        except EOFError:
            existing_data = {}

        # Update data based on whether special scale factor is provided
        if scale_factor == 'TBD':
            existing_data.update({
                "pthim": image_path, 
                "pthDL": model_path, 
                "pthtest": test_path, 
                "WS": whitespace_settings, 
                "nm": model_name, 
                "umpix": scale_factor, 
                "cmap": color_map, 
                "sxy": image_size,
                "classNames": class_names, 
                "ntrain": num_training_tiles, 
                "nblack": nblack, 
                "nwhite": nwhite, 
                "final_df": final_df,
                "combined_df": combined_df, 
                "nvalidate": num_validation_tiles, 
                "nTA": num_tissue_analysis, 
                "model_type": model_type,
                "batch_size": batch_size, 
                "uncomp_train_pth": uncomp_train_pth, 
                "uncomp_test_pth": uncomp_test_pth,
                "scale": scale, 
                "create_down": create_down, 
                "downsamp_annotated": downsamp_annotated
            })
        else:
            existing_data.update({
                "pthim": image_path, 
                "pthDL": model_path, 
                "pthtest": test_path, 
                "WS": whitespace_settings, 
                "nm": model_name, 
                "umpix": scale_factor, 
                "cmap": color_map,
                "sxy": image_size,
                "classNames": class_names, 
                "ntrain": num_training_tiles, 
                "nblack": nblack, 
                "nwhite": nwhite, 
                "final_df": final_df,
                "combined_df": combined_df, 
                "nvalidate": num_validation_tiles, 
                "nTA": num_tissue_analysis, 
                "model_type": model_type,
                "batch_size": batch_size
            })

        with open(data_file, 'wb') as f:
            pickle.dump(existing_data, f)
    else:
        print('Creating Net metadata file...')
        with open(data_file, 'wb') as f:
            if scale_factor == 'TBD':
                pickle.dump({
                    "pthim": image_path, 
                    "pthDL": model_path, 
                    "pthtest": test_path, 
                    "WS": whitespace_settings, 
                    "nm": model_name, 
                    "umpix": scale_factor,
                    "cmap": color_map, 
                    "sxy": image_size,
                    "classNames": class_names, 
                    "ntrain": num_training_tiles, 
                    "nblack": nblack, 
                    "nwhite": nwhite,
                    "final_df": final_df,
                    "combined_df": combined_df, 
                    "nvalidate": num_validation_tiles, 
                    "nTA": num_tissue_analysis, 
                    "model_type": model_type,
                    "batch_size": batch_size, 
                    "uncomp_train_pth": uncomp_train_pth, 
                    "uncomp_test_pth": uncomp_test_pth,
                    "scale": scale, 
                    "create_down": create_down, 
                    "downsamp_annotated": downsamp_annotated
                }, f)
            else:
                pickle.dump({
                    "pthim": image_path, 
                    "pthDL": model_path, 
                    "pthtest": test_path, 
                    "WS": whitespace_settings, 
                    "nm": model_name, 
                    "umpix": scale_factor, 
                    "cmap": color_map, 
                    "sxy": image_size,
                    "classNames": class_names, 
                    "ntrain": num_training_tiles, 
                    "nblack": nblack, 
                    "nwhite": nwhite, 
                    "final_df": final_df,
                    "combined_df": combined_df,
                    "nvalidate": num_validation_tiles, 
                    "nTA": num_tissue_analysis, 
                    "model_type": model_type, 
                    "batch_size": batch_size
                }, f)

    # Create and save color map legend plot
    plot_cmap_legend(color_map, class_names)
    plt.savefig(os.path.join(model_path, 'model_color_legend.jpg'))