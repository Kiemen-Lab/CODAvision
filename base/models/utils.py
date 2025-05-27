"""
Model Utilities for Semantic Segmentation

This module provides common utilities for loading, saving, and managing 
semantic segmentation models and their metadata.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: March 2025
"""

import os
import pickle
from typing import Dict, Any, List
import numpy as np
import tensorflow as tf
from base.evaluation.visualize import plot_cmap_legend

# Set up logging
import logging
logger = logging.getLogger(__name__)


def save_model_metadata(model_path: str, metadata: Dict[str, Any]) -> None:
    """
    Save or update model metadata to a pickle file (net.pkl).
    If net.pkl exists, it updates it with new key-value pairs from metadata.
    If a key from metadata already exists in net.pkl, its value is overwritten.
    """
    os.makedirs(model_path, exist_ok=True)
    data_file = os.path.join(model_path, 'net.pkl')

    # Standardize model_type representation
    if 'model_type' in metadata and isinstance(metadata.get('model_type'), str) and '+' in metadata['model_type']:
        metadata['model_type'] = metadata['model_type'].replace('+', '_plus')

    existing_data = {}
    if os.path.exists(data_file):
        try:
            with open(data_file, 'rb') as f:
                existing_data = pickle.load(f)
        except EOFError:
            logger.error(f"Warning: EOFError reading existing metadata file {data_file}. Starting with provided metadata.")
            existing_data = {}  # Or initialize with metadata if update implies base exists
        except Exception as e:
            logger.error(f"Warning: Failed to load existing metadata file {data_file}: {e}. Starting with provided metadata.")
            existing_data = {}

    existing_data.update(metadata)

    try:
        with open(data_file, 'wb') as f:
            pickle.dump(existing_data, f)
    except Exception as e:
        logger.error(f"Critical Error: Failed to save metadata to {data_file}: {e}")
        # Optionally, re-raise or handle more gracefully
        raise


def create_initial_model_metadata(
        pthDL: str,
        pthim: str,
        WS: list,
        nm: str,
        umpix: Any,
        cmap: np.ndarray,
        sxy: int,
        classNames: List[str],
        ntrain: int,
        nvalidate: int,
        model_type: str = "DeepLabV3_plus",  # Default from GUI
        batch_size: int = 3,  # Default from GUI/training
        pthtest: str = None,
        nTA: int = None,
        final_df: Any = None,
        combined_df: Any = None,
        uncomp_train_pth: str = '',
        uncomp_test_pth: str = '',
        scale: Any = '',
        create_down: bool = False,
        downsamp_annotated: bool = False
) -> None:
    """
    Creates and saves the initial model metadata to net.pkl.
    This function incorporates logic from the original base/save_model_metadata.py
    and can handle additional parameters typically set by the GUI.
    """
    if not os.path.isdir(pthDL):
        os.makedirs(pthDL)

    logger.info('Preparing initial model metadata and classification colormap...')

    # Make copies for manipulation if necessary, especially for classNames and WS
    _classNames = list(classNames)
    _WS = [list(w) if isinstance(w, (list, tuple)) else w for w in WS]

    # --- Logic for adjusting WS, classNames, cmap based on ndelete (from original save_model_metadata) ---
    # This part needs careful handling if layers are deleted or combined, as it affects indices.
    # The original logic:
    # ndelete = _WS[4].copy()
    # if isinstance(ndelete, list):
    #     ndelete.sort(reverse=True)
    #     if ndelete:
    #         _cmap_effective = cmap.copy()
    #         _classNames_effective = list(_classNames) # Operate on a copy
    #         ncombine_current = list(_WS[2])
    #         nload_current = list(_WS[3])

    #         for b_del_original_idx in ndelete: # b_del_original_idx is 1-based index of layer to delete
    #             # Find the corresponding entry in ncombine_current, as its indices might have shifted
    #             # This is complex because WS[2] itself changes if items are deleted.
    #             # A robust way is to map original layer indices to current list of classes.
    #             # For simplicity here, assuming _WS reflects changes from GUI if combined_df is used.
    #             # If not, the non-GUI path assumes classNames and cmap are already final *before* this processing.

    #             # The original logic in save_model_metadata.py seems to assume classNames/cmap might need trimming
    #             # based on what 'oldnum' (a combined class index) was associated with a deleted layer.
    #             # This suggests that WS[2] (ncombine) and WS[3] (nload) are based on original layer indices
    #             # and need to be remapped if the actual list of classes/colors changes.

    #             # Given the complexity and potential for GUI to pre-process this,
    #             # we will assume `classNames` and `cmap` passed are what should be used.
    #             # The WS adjustments should mainly focus on re-indexing `ncombine` and `nload`.
    #             # The critical part is that WS[2] must map to the final class indices.
    # pass # Placeholder for complex WS adjustment logic if absolutely needed outside GUI pre-processing.
    # For now, assume WS is passed in a relatively final state regarding combinations,
    # and deletion mainly affects nload and potentially nwhite's calculation.

    # Calculate nwhite (1-based index of the whitespace class in the *final* semantic classes)
    # WS[1][0] is the original 1-based index of the layer that defines whitespace behavior.
    # WS[2] (ncombine) maps original layer indices to their *final* 1-based class indices.
    nwhite = -1
    if _WS[1] and len(_WS[1]) > 0 and isinstance(_WS[1][0], int) and _WS[1][0] > 0:
        original_whitespace_layer_idx = _WS[1][0]
        if (_WS[2] and
                isinstance(_WS[2], list) and
                0 <= (original_whitespace_layer_idx - 1) < len(_WS[2]) and
                isinstance(_WS[2][original_whitespace_layer_idx - 1], int)):
            nwhite = _WS[2][original_whitespace_layer_idx - 1]
        else:
            logger.warning(
                f"Warning: WS[2] seems not correctly formatted or index out of bounds for nwhite calculation. WS[1][0]={original_whitespace_layer_idx}, WS[2]={_WS[2]}")

    if nwhite == -1:  # Fallback if calculation failed
        logger.warning("Warning: Could not determine nwhite from WS. Attempting to find 'whitespace' in classNames.")
        try:
            # classNames here should be the list of semantic classes (before "black" is appended)
            nwhite = _classNames.index("whitespace") + 1  # Find 1-based index
        except ValueError:
            if _classNames:  # Default to last semantic class if "whitespace" not found.
                logger.warning(f"Warning: 'whitespace' not in classNames. Defaulting nwhite to last class: {len(_classNames)}")
                nwhite = len(_classNames)
            else:  # No classes, problematic.
                logger.warning("Warning: No classNames provided. Defaulting nwhite to 1.")
                nwhite = 1

    # Prepare final_class_names for saving (conventionally with "black" as the last one for model output)
    _final_class_names_for_saving = list(_classNames)  # Start with semantic classes
    if not _final_class_names_for_saving or _final_class_names_for_saving[-1].lower() != "black":
        _final_class_names_for_saving.append("black")

    nblack_calculated = len(_final_class_names_for_saving)

    # Construct the metadata dictionary
    metadata_dict = {
        "pthim": pthim,
        "pthDL": pthDL,
        "WS": _WS,
        "nm": nm,
        "umpix": umpix,
        "cmap": cmap,
        "sxy": sxy,
        "classNames": _final_class_names_for_saving,  # Save with "black"
        "ntrain": ntrain,
        "nblack": nblack_calculated,
        "nwhite": nwhite,  # Index within the semantic classes (usually 1 to N, not including black)
        "nvalidate": nvalidate,
        "model_type": model_type,
        "batch_size": batch_size
    }

    if pthtest is not None: metadata_dict["pthtest"] = pthtest
    if nTA is not None: metadata_dict["nTA"] = nTA

    # GUI-specific fields, only add if provided (relevant for GUI workflow)
    if final_df is not None: metadata_dict["final_df"] = final_df
    if combined_df is not None: metadata_dict["combined_df"] = combined_df
    if umpix == 'TBD':  # Indicates custom scaling scenario from GUI
        metadata_dict.update({
            "uncomp_train_pth": uncomp_train_pth,
            "uncomp_test_pth": uncomp_test_pth,
            "scale": scale,
            "create_down": create_down,
            "downsamp_annotated": downsamp_annotated
        })

    # Call the core saving function
    save_model_metadata(model_path=pthDL, metadata=metadata_dict)

    # Plot legend using the semantic class names and their cmap
    plot_save_path = os.path.join(pthDL, 'model_color_legend.png')
    # Pass _classNames (semantic names) and corresponding cmap. plot_cmap_legend handles if len(titles) == len(cmap)+1.
    plot_cmap_legend(cmap, _classNames, save_path=plot_save_path)

    logger.info(f"Initial model metadata saved to {os.path.join(pthDL, 'net.pkl')}")
    logger.info(f"Color map legend saved to {plot_save_path}")


def setup_gpu() -> Dict[str, Any]:
    """
    Configure TensorFlow to use GPU and return GPU information.
    
    Returns:
        Dictionary with GPU information or empty dict if no GPU is available
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    gpu_info = {}
    
    if physical_devices:
        try:
            # Set memory growth to avoid allocating all memory at once
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # Use only the first GPU
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            logger.info(f"TensorFlow is using the following GPU: {logical_devices[0]}")
            
            # Get GPU memory info if possible
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # First GPU
                    gpu_info = {
                        'device': gpu.id,
                        'name': gpu.name,
                        'total_memory': f"{gpu.memoryTotal:.1f} MB",
                        'free_memory': f"{gpu.memoryFree:.1f} MB",
                        'used_memory': f"{gpu.memoryUsed:.1f} MB",
                        'utilization': f"{gpu.load * 100:.1f}%"
                    }
            except ImportError:
                logger.error("GPUtil not available - limited GPU information will be displayed")
                gpu_info = {'device': 'GPU available but detailed info unavailable'}
                
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.warning("No GPU available. Operations will proceed on the CPU.")
        logger.warning("Ensure that the NVIDIA GPU and CUDA are correctly installed if you intended to use a GPU.")
    
    return gpu_info


def calculate_class_weights(mask_list: List[str], num_classes: int) -> np.ndarray:
    """
    Calculate class weights based on pixel frequency in label masks.
    
    Args:
        mask_list: List of paths to mask images
        num_classes: Number of classes
        
    Returns:
        Array of class weights
    """
    class_pixels = np.zeros(num_classes)
    image_pixels = np.zeros(num_classes)
    epsilon = 1e-5  # Prevent division by zero
    
    for mask_path in mask_list:
        # Load mask
        mask = tf.keras.preprocessing.image.load_img(
            mask_path, color_mode='grayscale')
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        mask = mask.astype(int)
        
        # Count pixels per class
        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        for val, count in zip(unique, counts):
            if val < num_classes:
                class_pixels[val] += count
                image_pixels[val] += total_pixels
    
    # Calculate frequencies
    freq = class_pixels / (image_pixels + epsilon)
    
    # Handle invalid values
    freq[np.isinf(freq) | np.isnan(freq)] = epsilon
    
    # Calculate weights using median frequency balancing
    median_freq = np.median(freq)
    class_weights = median_freq / (freq + epsilon)
    
    # Print class distribution information
    logger.info("\nClass frequencies:")
    for i, f in enumerate(freq):
        logger.info(f"Class {i}: {f:.4f}")
    
    logger.info("\nClass weights:")
    for i, w in enumerate(class_weights):
        logger.info(f"Class {i}: {w:.4f}")
    
    return class_weights.astype(np.float32)


def get_model_paths(model_path: str, model_type: str) -> Dict[str, str]:
    """
    Get standard paths for model files.
    
    Args:
        model_path: Base directory for the model
        model_type: Type of model (e.g., 'DeepLabV3_plus', 'UNet')
        
    Returns:
        Dictionary containing paths for model files
    """
    # Standardize model type format
    if '+' in model_type:
        model_type = model_type.replace('+', '_plus')
    
    return {
        'best_model': os.path.join(model_path, f'best_model_{model_type}.keras'),
        'final_model': os.path.join(model_path, f'{model_type}.keras'),
        'logs': os.path.join(model_path, 'logs'),
        'metadata': os.path.join(model_path, 'net.pkl'),
        'train_data': os.path.join(model_path, 'training'),
        'val_data': os.path.join(model_path, 'validation'),
        'confusion_matrix': os.path.join(model_path, f'confusion_matrix_{model_type}.png')
    }