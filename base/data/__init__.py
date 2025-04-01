"""
Data handling utilities for CODAvision.
"""

from .annotation import (
    # XML handling
    load_annotation_data,
    import_xml,
    load_annotations,
    extract_annotation_layers,
    load_xml_annotations,
    
    # Annotation masks
    save_annotation_mask,
    format_white,
    save_bounding_boxes,
    
    # Utilities
    calculate_tissue_mask,
    check_if_model_parameters_changed
)

from .loaders import (
    read_image,
    read_image_overlay,
    create_dataset,
    convert_to_array,
    calculate_tissue_mask,
    load_model_metadata,
    DataGenerator
)

from .tiles import (
    combine_annotations_into_tiles
)