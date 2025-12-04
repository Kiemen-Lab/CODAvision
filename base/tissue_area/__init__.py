"""
Tissue Area Threshold Detection Module

This module provides functionality for determining optimal tissue area thresholds
in histological images, particularly for distinguishing tissue from whitespace.
"""

from .models import (
    ThresholdConfig,
    ThresholdMode,
    ImageThresholds,
    RegionSelection
)

from .threshold import determine_optimal_TA

__all__ = [
    'determine_optimal_TA',
    'ThresholdConfig',
    'ThresholdMode',
    'ImageThresholds',
    'RegionSelection'
]