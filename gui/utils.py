"""
GUI Utility Functions for CODAvision

This module provides utility functions that bridge the GUI components
with the core CODAvision functionality.
"""

from typing import List, Union, Any
import numpy as np
import pandas as pd

from base.models.metadata import save_model_metadata_gui


def save_model_metadata_GUI(
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
    create_down: str = '',
    downsamp_annotated: str = ''
) -> None:
    """
    Save model metadata from GUI parameters.
    
    This function maintains backward compatibility with the original GUI
    interface while delegating to the refactored metadata handling system.
    
    See base.models.metadata.save_model_metadata_gui for detailed parameter
    documentation.
    """
    # Convert string boolean parameters to actual booleans
    create_down_bool = bool(create_down) if isinstance(create_down, str) else create_down
    downsamp_annotated_bool = bool(downsamp_annotated) if isinstance(downsamp_annotated, str) else downsamp_annotated
    
    # Delegate to the refactored function
    save_model_metadata_gui(
        pthDL=pthDL,
        pthim=pthim,
        pthtest=pthtest,
        WS=WS,
        nm=nm,
        umpix=umpix,
        cmap=cmap,
        sxy=sxy,
        classNames=classNames,
        ntrain=ntrain,
        nvalidate=nvalidate,
        nTA=nTA,
        final_df=final_df,
        combined_df=combined_df,
        model_type=model_type,
        batch_size=batch_size,
        uncomp_train_pth=uncomp_train_pth,
        uncomp_test_pth=uncomp_test_pth,
        scale=scale,
        create_down=create_down_bool,
        downsamp_annotated=downsamp_annotated_bool
    )