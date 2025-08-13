"""
Data Models for Tissue Area Threshold Detection

This module defines the data structures used for tissue area threshold
configuration and management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ThresholdMode(Enum):
    """Enumeration of threshold detection modes."""
    HE = "H&E"
    GRAYSCALE = "Grayscale"
    
    @property
    def default_threshold(self) -> int:
        """Get the default threshold value for this mode."""
        return 205 if self == ThresholdMode.HE else 50
    


@dataclass
class RegionSelection:
    """Represents a selected region in an image."""
    x: int
    y: int
    size: int = 600
    
    def get_bounds(self, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate the bounds for cropping.
        
        Args:
            image_shape: Shape of the image (height, width)
            
        Returns:
            Tuple of (y_start, y_end, x_start, x_end)
        """
        height, width = image_shape[:2]
        half_size = self.size
        
        # Calculate y bounds
        if self.y < half_size:
            y_start, y_end = 0, 2 * half_size
        elif self.y + half_size > height:
            y_start, y_end = height - 2 * half_size, height
        else:
            y_start, y_end = self.y - half_size, self.y + half_size
            
        # Calculate x bounds
        if self.x < half_size:
            x_start, x_end = 0, 2 * half_size
        elif self.x + half_size > width:
            x_start, x_end = width - 2 * half_size, width
        else:
            x_start, x_end = self.x - half_size, self.x + half_size
            
        return y_start, y_end, x_start, x_end


@dataclass
class ImageThresholds:
    """Container for image-specific threshold values."""
    thresholds: Dict[str, int] = field(default_factory=dict)
    mode: ThresholdMode = ThresholdMode.HE
    average_threshold: bool = False
    image_list: List[str] = field(default_factory=list)
    
    def get_threshold(self, image_name: str) -> int:
        """Get threshold for a specific image."""
        if self.average_threshold and self.thresholds:
            return int(np.mean(list(self.thresholds.values())))
        return self.thresholds.get(image_name, self.mode.default_threshold)
    
    def set_threshold(self, image_name: str, threshold: int):
        """Set threshold for a specific image."""
        self.thresholds[image_name] = threshold
        if image_name not in self.image_list:
            self.image_list.append(image_name)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'cts': self.thresholds,
            'mode': self.mode.value,
            'average_TA': self.average_threshold,
            'imlist': self.image_list
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ImageThresholds':
        """Create from dictionary."""
        mode = ThresholdMode.HE if data.get('mode', 'H&E') == 'H&E' else ThresholdMode.GRAYSCALE
        return cls(
            thresholds=data.get('cts', {}),
            mode=mode,
            average_threshold=data.get('average_TA', False),
            image_list=data.get('imlist', [])
        )


@dataclass
class ThresholdConfig:
    """Configuration for threshold detection process."""
    training_path: str
    testing_path: str
    num_images: int = 0  # 0 means process all images
    redo: bool = False
    default_threshold: int = 205
    region_size: int = 600
    
    @property
    def training_images_path(self) -> str:
        """Get the path for training images."""
        return self.training_path
    
    @property
    def testing_images_path(self) -> str:
        """Get the path for testing images."""
        return self.testing_path
    
    @property
    def output_path(self) -> str:
        """Get the output path for threshold data."""
        import os
        return os.path.join(self.training_path, 'TA')
    
    @property
    def threshold_file_path(self) -> str:
        """Get the path to the threshold pickle file."""
        import os
        return os.path.join(self.output_path, 'TA_cutoff.pkl')