"""
Tissue Area Threshold Detection Module

This module provides functionality for determining optimal tissue area thresholds
in histological images, particularly for distinguishing tissue from whitespace.
"""

from .threshold import TissueAreaThresholdSelector, determine_optimal_TA
from .models import (
    ThresholdConfig,
    ThresholdMode,
    ImageThresholds,
    RegionSelection
)

__all__ = [
    'TissueAreaThresholdSelector',
    'determine_optimal_TA',
    'ThresholdConfig',
    'ThresholdMode',
    'ImageThresholds',
    'RegionSelection'
]