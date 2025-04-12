"""
Visualization Utilities for CODAvision

This module provides utility functions for creating visualizations of model data,
including color maps, class distributions, and other visual elements to aid in model
interpretation and documentation.

Authors:
    Valentina Matos (Johns Hopkins - Wirtz/Kiemen Lab)
    Tyler Newton (JHU - DSAI)

Updated: April 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional, Tuple, Any


def plot_cmap_legend(
    cmap: np.ndarray, 
    titles: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot a color map legend with class names.

    This function creates a visualization of a model's color map along with 
    class names for easy reference and inclusion in reports or documentation.

    Args:
        cmap: Color map array where each row is an RGB triple (0-255)
        titles: List of class names corresponding to the color map
        figsize: Optional figure size (width, height) in inches
        title: Optional title for the figure

    Returns:
        Matplotlib figure object containing the color map legend
    """
    # Create an image array to represent the color map
    im = np.zeros((50, 50 * len(cmap), 3), dtype=np.uint8)
    for k in range(len(cmap)):
        tmp = cmap[k].reshape(1, 1, 3)
        tmp = np.tile(tmp, (50, 50, 1))
        im[:, k * 50:(k + 1) * 50, :] = tmp

    # Handle special case where titles has one more element than cmap
    if len(cmap) == len(titles) - 1:
        titles = titles[:-1]

    # Replace spaces with underscores in titles
    titles = [title.replace(' ', '_') for title in titles]

    # Create the figure
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    if titles:
        # Rotate the image for better layout
        im = np.rot90(im)
        
        # Configure plot appearance
        plt.tick_params(axis='both', width=1)
        plt.imshow(im)
        plt.ylim(0, im.shape[0])
        plt.yticks(np.arange(25, im.shape[0], 50), labels=titles[::-1])
        plt.xticks([])
        plt.tick_params(axis='y', length=0)
        plt.tick_params(axis='both', labelsize=15)
        
        # Add title if provided
        if title:
            plt.title(title)
    else:
        # Simple image display if no titles are provided
        plt.imshow(im)

    return fig