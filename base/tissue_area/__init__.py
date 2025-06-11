"""
Tissue Area Threshold Detection Module

This module provides functionality for determining optimal tissue area thresholds
in histological images, particularly for distinguishing tissue from whitespace.
"""

# Don't import from threshold to avoid circular imports
# Users should import directly from base.tissue_area.threshold
from .models import (
    ThresholdConfig,
    ThresholdMode,
    ImageThresholds,
    RegionSelection
)

__all__ = [
    'ThresholdConfig',
    'ThresholdMode',
    'ImageThresholds',
    'RegionSelection'
]