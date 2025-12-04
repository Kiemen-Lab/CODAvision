"""
Data handling utilities for CODAvision.
"""

from .annotation import (
    load_annotation_data,
    import_xml,
    load_annotations,
    extract_annotation_layers,
    load_xml_annotations,
    save_annotation_mask,
    format_white,
    save_bounding_boxes,
    check_if_model_parameters_changed
)

from .loaders import (
    read_image,
    create_dataset,
    load_model_metadata,
    DataGenerator
)

from .tiles import (
    combine_annotations_into_tiles,
    create_training_tiles,
    create_training_tiles_modern,
    create_training_tiles_legacy
)

# Import and re-export tile generation configuration
from ..config import (
    TileGenerationConfig,
    MODERN_CONFIG,
    LEGACY_CONFIG
)

from ..tissue_area.utils import calculate_tissue_mask
from ..image.utils import convert_to_array