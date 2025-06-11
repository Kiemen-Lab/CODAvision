"""
Tissue Area Threshold Selection - Compatibility Layer

This module maintains backward compatibility by delegating to the GUI
implementation when called from existing code.
"""

import warnings

# Import the core version
from .threshold_core import determine_optimal_TA_core, TissueAreaThresholdSelector

# Defer GUI import to avoid circular dependency
_GUI_AVAILABLE = None
_GUI_IMPORT_ERROR = None

def _check_gui_available():
    """Check if GUI is available, importing it if needed."""
    global _GUI_AVAILABLE, _GUI_IMPORT_ERROR
    
    if _GUI_AVAILABLE is not None:
        return _GUI_AVAILABLE
    
    try:
        from gui.tissue_area.threshold_gui import determine_optimal_TA_gui
        globals()['determine_optimal_TA_gui'] = determine_optimal_TA_gui
        _GUI_AVAILABLE = True
    except ImportError as e:
        _GUI_IMPORT_ERROR = str(e)
        _GUI_AVAILABLE = False
    except Exception as e:
        _GUI_IMPORT_ERROR = str(e)
        _GUI_AVAILABLE = False
    
    return _GUI_AVAILABLE


def determine_optimal_TA(
    *args,
    **kwargs
) -> int:
    """
    Determine optimal tissue area threshold.
    
    This function maintains backward compatibility by supporting both old and new
    calling conventions and delegating to the appropriate implementation.
    
    Old convention (4 positional args):
        pthim: Path to training images
        pthtestim: Path to test images  
        nTA: Number of tissue area thresholds
        redo: Whether to redo (boolean)
        
    New convention (keyword args):
        downsampled_path: Path to downsampled images
        output_path: Path for output files
        test_ta_mode: Mode for testing ('redo' or '')
        display_size: Size of display region
        sample_size: Number of images to sample
        
    Returns:
        Number of thresholds determined
    """
    # Handle old calling convention with positional arguments
    if len(args) == 4:
        pthim, pthtestim, n_ta, redo = args
        # Map old parameters to new ones
        downsampled_path = pthim  # training images path
        output_path = pthtestim   # where to save results
        test_ta_mode = 'redo' if redo else ''
        display_size = kwargs.get('display_size', 600)
        sample_size = kwargs.get('sample_size', n_ta if n_ta > 0 else 20)
    # Handle new calling convention
    elif 'downsampled_path' in kwargs and 'output_path' in kwargs:
        downsampled_path = kwargs['downsampled_path']
        output_path = kwargs['output_path']
        test_ta_mode = kwargs.get('test_ta_mode', '')
        display_size = kwargs.get('display_size', 600)
        sample_size = kwargs.get('sample_size', 20)
    else:
        raise ValueError(
            "Invalid arguments. Expected either:\n"
            "1. Old style: determine_optimal_TA(pthim, pthtestim, nTA, redo)\n"
            "2. New style: determine_optimal_TA(downsampled_path=..., output_path=...)"
        )
    if _check_gui_available():
        # Use GUI version when available
        return determine_optimal_TA_gui(
            downsampled_path=downsampled_path,
            output_path=output_path,
            test_ta_mode=test_ta_mode,
            display_size=display_size,
            sample_size=sample_size
        )
    else:
        # Fall back to core version (no GUI)
        warnings.warn(
            f"GUI not available, using core implementation with default thresholds. "
            f"For interactive threshold selection, ensure GUI modules are available. "
            f"Error: {_GUI_IMPORT_ERROR}",
            UserWarning
        )
        
        thresholds = determine_optimal_TA_core(
            downsampled_path=downsampled_path,
            output_path=output_path,
            test_ta_mode=test_ta_mode,
            display_size=display_size,
            sample_size=sample_size
        )
        
        return len(thresholds.thresholds)


# Export the selector for direct use if needed
__all__ = ['determine_optimal_TA', 'TissueAreaThresholdSelector']