"""
Utility functions for CODAvision.

This package contains utility modules for logging, visualization, and other
helper functions used throughout the codebase.
"""

from .logger import Logger
from .visualization import plot_cmap_legend
from .metadata import save_model_metadata_gui

__all__ = [
    'Logger',
    'plot_cmap_legend',
    'save_model_metadata_gui'
]