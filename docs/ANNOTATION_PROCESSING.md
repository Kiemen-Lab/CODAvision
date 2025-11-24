# Annotation Processing and Training Tile Generation

This document provides comprehensive documentation of how CODAvision processes annotations from XML files and generates training tiles.

## Table of Contents

1. [Annotation Processing from XML](#1-annotation-processing-from-xml)
   - [XML File Input](#xml-file-input)
   - [Parsing Logic](#parsing-logic)
   - [Data Structures](#data-structures-after-xml-parsing)
   - [Storage Locations](#storage-location-annotationspkl)
   - [Annotation Mask Creation](#annotation-mask-creation)
   - [Bounding Box Tile Creation](#bounding-box-tile-creation)
2. [Training Tile Generation](#2-training-tile-generation)
   - [Input Files](#input-files-required)
   - [Configuration](#tile-generation-configuration)
   - [Workflow](#workflow-create_training_tiles)
   - [Output Structure](#output-structure)
   - [PyTorch Metadata](#pytorch-training-metadata)
3. [Complete Pipeline Summary](#complete-pipeline-summary)
4. [File Paths Reference](#key-file-paths-reference)

---

## 1. Annotation Processing from XML

### XML File Input

**Location**: User-specified annotation directory (e.g., `/path/to/annotations/`)

**File Format**: XML files with `.xml` extension containing:
- Annotation coordinates (polygons)
- Layer/class information
- MicronsPerPixel metadata

**Example XML Structure**:
```xml
<Annotations>
    <MicronsPerPixel>0.25</MicronsPerPixel>
    <Annotation>
        <Regions>
            <Region>
                <Vertices>
                    <Vertex X="1234.5" Y="5678.9"/>
                    <Vertex X="1235.2" Y="5679.1"/>
                    ...
                </Vertices>
            </Region>
        </Regions>
    </Annotation>
</Annotations>
```

### Parsing Logic

**Module**: `base/data/annotation.py`

**Entry Point**: `load_annotation_data()`

#### Processing Flow

1. **XML Discovery**:
   - Scans annotation directory for `.xml` files
   - Filters out hidden files (starting with `_` or `.`)
   - Validates corresponding image files exist (`.tif`, `.jpg`, or `.png`)

2. **XML Reading** (`load_xml_annotations()`):
   - Uses multiple parsing strategies for cross-platform compatibility:
     - `xmltodict` (primary)
     - `ElementTree` (fallback)
     - SAX parser (validation)
     - Regex-based manual extraction (robust fallback)
   - Handles various encodings: UTF-8, UTF-16, Latin-1, Mac Roman, Windows-1252
   - Removes BOM markers and Mac OS X metadata
   - Cleans control characters and normalizes line endings

3. **Data Extraction** (`extract_annotation_coordinates()`):
   - Extracts `MicronsPerPixel` value (stored as `reduce_annotations`)
   - Parses nested structure:
     ```
     Annotations → Annotation (layers) → Regions → Region → Vertices → Vertex (X, Y)
     ```
   - Creates DataFrame with columns: `['Annotation Id', 'Annotation Number', 'X vertex', 'Y vertex']`

#### Error Handling

The parser includes robust error handling for:
- Malformed XML
- Missing fields
- Encoding issues
- Platform-specific formatting

### Data Structures After XML Parsing

#### Intermediate DataFrame (`xyout`)

```python
# Shape: (N, 4) where N = total number of vertices
# Columns: [layer_id, annotation_number, x_coord, y_coord]
# Type: np.ndarray

# Example:
#   layer_id  annotation_number  x_coord    y_coord
#   1.0       1.0                 1234.5    5678.9
#   1.0       1.0                 1235.2    5679.1
#   2.0       1.0                 3456.7    7890.1
```

### Storage Location: `annotations.pkl`

**Path**: `{annotation_path}/data py/{image_basename}/annotations.pkl`

**Structure**:
```python
{
    'xyout': np.ndarray,           # Shape (N, 4): annotation coordinates
                                   # Columns: [layer_id, annotation_number, x, y]

    'reduce_annotations': float,    # MicronsPerPixel value from XML

    'dm': float,                    # File modification timestamp

    'WS': list,                     # Whitespace settings from model
                                   # [keep_per_class, distribute_to, priority, nesting_order]

    'umpix': int,                   # Microns per pixel (scaling factor)

    'nwhite': int,                  # White/background class index

    'pthim': str,                   # Path to source images

    'numann': np.ndarray,           # Shape (M, C): pixel counts per class
                                   # M = number of bounding boxes
                                   # C = number of classes

    'ctlist': dict,                 # Bounding box tile information
                                   # {'tile_name': [...], 'tile_pth': [...]}

    'bb': int                       # Flag: 1 if bounding boxes created, 0 otherwise
}
```

### Annotation Mask Creation

**Function**: `save_annotation_mask()` in `base/data/annotation.py`

#### Process

1. **Coordinate Scaling**:
   - Divides XML coordinates by `scale_factor` or `umpix`
   - Converts to image pixel coordinates

2. **Polygon Interpolation**:
   - Connects vertices with interpolated points (0.49 pixel spacing)
   - Ensures continuous polygon boundaries
   - Fills polygon interiors using `scipy.ndimage.binary_fill_holes()`

3. **Whitespace Handling** (`format_white()`):
   - Applies nesting order from `WS[3]` (priority list)
   - Removes/keeps whitespace per class based on `WS[0]`
   - Distributes whitespace pixels to specified classes `WS[1]`
   - Example: Background pixels can be redistributed to tissue classes

4. **Output Files**:
   - `view_annotations.png`: Full-resolution annotation mask (post-processing)
   - `view_annotations_raw.png`: Pre-whitespace processing mask
   - Visualization overlay saved in `check_annotations/` directory

#### Annotation Mask Format

```python
# Shape: (height, width)
# Type: uint8 or uint16 (depending on number of classes)
# Values: 0, 1, 2, ..., num_classes-1
# 0 = background/whitespace
# 1+ = annotation classes
```

### Bounding Box Tile Creation

**Function**: `save_bounding_boxes()` in `base/data/annotation.py`

#### Process

1. **Connected Component Analysis**:
   - Morphological operations to close gaps and remove noise
   - Labels each connected annotation region
   - Handles multiple disconnected annotations

2. **Bounding Box Extraction**:
   - For each labeled region, finds min/max X and Y coordinates
   - Adds padding if configured
   - Crops image and annotation mask to bounding box

3. **Storage Structure**:
   ```
   {annotation_path}/data py/{image_basename}/{model_name}_boundbox/
       ├── im/          # Cropped images (TIFF)
       │   ├── 00001.tif
       │   ├── 00002.tif
       │   └── ...
       └── label/       # Cropped annotation masks (TIFF)
           ├── 00001.tif
           ├── 00002.tif
           └── ...
   ```

4. **Metadata** (`ctlist`):
   ```python
   {
       'tile_name': ['00001.tif', '00002.tif', ...],
       'tile_pth': ['/path/to/im/', '/path/to/im/', ...]
   }
   ```

5. **Pixel Counts** (`numann`):
   ```python
   # Shape: (num_bounding_boxes, num_classes)
   # Example:
   #        class1  class2  class3  ...
   # bbox1    1234    567     890
   # bbox2    2345    678     901
   # bbox3    3456    789     123
   ```

---

## 2. Training Tile Generation

### Input Files Required

#### From Annotation Processing

1. **`annotations.pkl`** (per image)
   - Contains bounding box metadata
   - Location: `{annotation_path}/data py/{image_basename}/annotations.pkl`

2. **Bounding box images**
   - Format: TIFF (`.tif`)
   - Location: `{model_name}_boundbox/im/*.tif`

3. **Bounding box labels**
   - Format: TIFF (`.tif`)
   - Location: `{model_name}_boundbox/label/*.tif`

#### From Model Configuration

**`net.pkl`** in model root directory:
```python
{
    'sxy': 1024,              # Tile size (e.g., 1024x1024)
    'nblack': 13,             # Number of classes + background
    'classNames': [...],      # List of class names
    'ntrain': 30,             # Number of training big tiles to generate
    'nvalidate': 10,          # Number of validation big tiles to generate
    'WS': [...],              # Whitespace settings
    'cmap': np.ndarray        # Color map for visualization (shape: num_classes x 3)
}
```

### Tile Generation Configuration

**Module**: `base/data/tiles.py`

**Configuration Class**: `TileGenerationConfig`

#### Configuration Sources (Priority Order)

1. **Explicit parameter**: Pass `config` to `create_training_tiles()`
2. **Environment variable**: `CODAVISION_TILE_GENERATION_MODE`
3. **Config file default**: `ModelDefaults.TILE_GENERATION_MODE` in `base/config.py`

#### Two Predefined Modes

##### Modern Mode (Default - CODAvision-style)

```python
MODERN_CONFIG = TileGenerationConfig(
    mode="modern",
    reduction_factor=10,        # Coarser placement optimization
    use_disk_filter=False,      # No disk filter convolution
    crop_rotations=False,       # Keep expanded rotation dimensions
    class_rotation_frequency=5, # Rotate class every 5th iteration
    deterministic_seed=3,       # Reproducible results
    big_tile_size=10240,        # Big tile dimensions
    file_format="png"           # Output format
)
```

**Use When**:
- Starting a new project
- Empirical testing shows better results
- You observe too much black background in tiles
- Migrating from CODAvision

##### Legacy Mode (MATLAB-aligned)

```python
LEGACY_CONFIG = TileGenerationConfig(
    mode="legacy",
    reduction_factor=5,         # Fine placement optimization
    use_disk_filter=True,       # Enable disk filter convolution
    crop_rotations=True,        # MATLAB cropping behavior
    class_rotation_frequency=3, # More frequent class rotation
    deterministic_seed=None,    # Diverse random runs (no seed)
    big_tile_size=10000,        # MATLAB-compatible size
    file_format="tif"           # Lossless TIFF format
)
```

**Use When**:
- Comparing results with MATLAB outputs
- Need MATLAB-aligned behavior
- Code maintainability is priority
- Validating against reference implementation

#### Setting Configuration

**Via Environment Variable**:
```bash
# Use modern mode (default)
export CODAVISION_TILE_GENERATION_MODE=modern
python CODAvision.py

# Use legacy mode
export CODAVISION_TILE_GENERATION_MODE=legacy
python scripts/non-gui_workflow.py
```

**Programmatically**:
```python
from base.data.tiles import (
    create_training_tiles,
    create_training_tiles_modern,
    create_training_tiles_legacy,
    TileGenerationConfig,
    MODERN_CONFIG,
    LEGACY_CONFIG
)

# Use default (modern) mode
create_training_tiles(
    model_path='path/to/model',
    annotations=annotations,
    image_list=image_list,
    create_new_tiles=True
)

# Use legacy mode explicitly
create_training_tiles_legacy(
    model_path='path/to/model',
    annotations=annotations,
    image_list=image_list,
    create_new_tiles=True
)

# Create custom configuration
custom_config = TileGenerationConfig(
    mode="custom",
    reduction_factor=7,
    use_disk_filter=True,
    crop_rotations=False,
    class_rotation_frequency=4,
    deterministic_seed=42,
    big_tile_size=10000,
    file_format="tif"
)

create_training_tiles(
    model_path='path/to/model',
    annotations=annotations,
    image_list=image_list,
    create_new_tiles=True,
    config=custom_config
)
```

### Workflow: `create_training_tiles()`

**Module**: `base/data/tiles.py`

#### Phase 1: Preparation

1. **Load Model Metadata**:
   - Reads `net.pkl` for configuration
   - Extracts tile size, class names, training/validation counts

2. **Validate Annotations**:
   - Checks all annotation files exist
   - Verifies non-zero pixel counts for each class
   - Logs warnings for missing or empty annotations

3. **Calculate Statistics**:
   - Computes class distribution across all annotations
   - Determines sampling strategy for balanced tiles

4. **Set Random Seed** (if deterministic mode enabled):
   - Ensures reproducible tile generation
   - Modern mode uses seed=3, legacy mode uses no seed

#### Phase 2: Training Tiles Generation

**Main Function**: `combine_annotations_into_tiles()`

##### Step 1: Canvas Creation

```python
# Create big tile canvases with margin for rotation
big_tile_size = config.big_tile_size  # 10240 or 10000
margin = 200

image_canvas = np.zeros((big_tile_size + margin, big_tile_size + margin, 3), dtype=np.uint8)
label_canvas = np.zeros((big_tile_size + margin, big_tile_size + margin), dtype=np.uint8)
```

##### Step 2: Iterative Tile Placement

```python
MAX_ITERATIONS = 10000
target_fill_ratio = 0.55

while fill_ratio < target_fill_ratio and iteration < MAX_ITERATIONS:
    # 1. Select class to sample (balanced sampling)
    selected_class = select_class_balanced(class_counts, iteration, config.class_rotation_frequency)

    # 2. Find candidate bounding boxes containing that class
    candidates = find_bounding_boxes_with_class(selected_class, annotations)

    # 3. Load and augment selected bounding box
    image_tile, label_tile = load_bounding_box(random.choice(candidates))
    image_tile, label_tile = augment_tile(image_tile, label_tile, config)

    # 4. Find optimal placement using distance transform
    placement_x, placement_y = find_optimal_placement(label_canvas, label_tile, config)

    # 5. Place tile on canvas
    place_tile(image_canvas, label_canvas, image_tile, label_tile, placement_x, placement_y)

    # 6. Update statistics
    fill_ratio = calculate_fill_ratio(label_canvas)
    class_counts = update_class_counts(label_canvas)
```

##### Step 3: Placement Algorithm

```python
def find_optimal_placement(canvas_mask, tile_mask, config):
    """
    Finds optimal placement for a tile using distance transform.
    Places tiles in center of largest empty areas.
    """
    # Downsample for faster computation
    reduction = config.reduction_factor  # 5 or 10
    downsampled = canvas_mask[::reduction, ::reduction]

    # Optional disk filter convolution (legacy mode only)
    if config.use_disk_filter:
        disk_kernel = create_disk_kernel(51)
        filtered = cv2.filter2D(downsampled, -1, disk_kernel)
    else:
        filtered = downsampled

    # Distance transform to find empty space
    binary_mask = (filtered == 0).astype(np.uint8)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Add padding border to prevent edge placement
    padding = 10
    dist_transform[:padding, :] = 0
    dist_transform[-padding:, :] = 0
    dist_transform[:, :padding] = 0
    dist_transform[:, -padding:] = 0

    # Find maximum distance point (center of largest empty area)
    max_loc = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)

    # Scale back to full resolution
    placement_y = max_loc[0] * reduction
    placement_x = max_loc[1] * reduction

    return placement_x, placement_y
```

##### Step 4: Augmentation

**Function**: `edit_annotation_tiles()` in `base/data/tiles.py`

```python
def edit_annotation_tiles(image, label, config):
    """
    Apply random rotation to tiles.
    Handles different cropping strategies based on config.
    """
    # Random rotation (0°, 90°, 180°, 270°)
    angle = random.choice([0, 90, 180, 270])

    if angle != 0:
        # Rotate both image and label
        image_rotated = rotate(image, angle, preserve_range=True)
        label_rotated = rotate(label, angle, order=0, preserve_range=True)

        if config.crop_rotations:
            # Legacy mode: Crop to original size (MATLAB behavior)
            # May introduce more black pixels
            image_rotated = crop_to_original_size(image_rotated, image.shape)
            label_rotated = crop_to_original_size(label_rotated, label.shape)
        else:
            # Modern mode: Keep expanded dimensions
            # Reduces black background artifacts
            pass
    else:
        image_rotated = image
        label_rotated = label

    return image_rotated, label_rotated
```

##### Step 5: Canvas Splitting

```python
def split_big_tile(big_image, big_label, tile_size, margin=100):
    """
    Splits big tile into smaller training tiles.
    Removes margins and creates grid of tiles.
    """
    # Remove margins
    big_image = big_image[margin:-margin, margin:-margin]
    big_label = big_label[margin:-margin, margin:-margin]

    height, width = big_image.shape[:2]
    tiles = []

    # Grid splitting
    for y in range(0, height - tile_size + 1, tile_size):
        for x in range(0, width - tile_size + 1, tile_size):
            image_tile = big_image[y:y+tile_size, x:x+tile_size]
            label_tile = big_label[y:y+tile_size, x:x+tile_size]

            # Only save tiles with sufficient content
            if has_sufficient_content(label_tile):
                tiles.append((image_tile, label_tile))

    return tiles
```

#### Phase 3: Validation Tiles Generation

- Identical process to training tiles
- Uses fresh copy of annotations (no overlap with training)
- Separate output directory
- Same configuration parameters

### Output Structure

```
{model_path}/
├── training/
│   ├── im/              # Training images
│   │   ├── 1.png        # or 1.tif in legacy mode
│   │   ├── 2.png
│   │   └── ...
│   ├── label/           # Training masks
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── big_tiles/       # Reference big tiles (for visualization)
│       ├── HE_tile_1.png
│       ├── label_tile_1.png
│       ├── HE_tile_2.png
│       ├── label_tile_2.png
│       └── ...
├── validation/
│   ├── im/              # Validation images
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   ├── label/           # Validation masks
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── big_tiles/       # Reference big tiles
│       ├── HE_tile_1.png
│       ├── label_tile_1.png
│       └── ...
├── annotations.pkl      # PyTorch metadata (see below)
└── train_list.pkl       # PyTorch training list (see below)
```

**File Formats**:
- **Modern mode**: `.png` (faster I/O, good compression)
- **Legacy mode**: `.tif` (lossless, MATLAB-compatible)

**Tile Dimensions**:
- **Small tiles**: Configured in `net.pkl` (typically 1024×1024 or 512×512)
- **Big tiles**: Configured in `TileGenerationConfig` (10000×10000 or 10240×10240)

### PyTorch Training Metadata

**Generated by**: `create_training_tiles()` (lines 831-856 in `base/data/tiles.py`)

#### File 1: `annotations.pkl` (model root)

```python
# Lightweight mapping: image_id → image_id
# Purpose: PyTorch dataloader compatibility
{
    '1': '1',
    '2': '2',
    '3': '3',
    ...
    '1000': '1000'
}
```

#### File 2: `train_list.pkl` (model root)

```python
# List of training image IDs
# Purpose: Defines training set for PyTorch dataloaders
['1', '2', '3', ..., '1000']
```

**Usage**:
```python
from base.data.loaders import TrainingTileDataLoader

# PyTorch dataloader automatically uses these files
train_loader = TrainingTileDataLoader(
    model_path='path/to/model',
    batch_size=8,
    subset='training'
)

# Iterates over tiles defined in train_list.pkl
for images, labels in train_loader:
    # images shape: (batch_size, height, width, channels)
    # labels shape: (batch_size, height, width)
    pass
```

---

## Complete Pipeline Summary

### Visual Flow

```
┌─────────────────────────┐
│  XML Files              │
│  (annotations)          │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  load_xml_annotations   │
│  - Parse XML            │
│  - Extract coordinates  │
│  - Handle encoding      │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  DataFrame (xyout)      │
│  [layer, ann#, x, y]    │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  save_annotation_mask   │
│  - Interpolate polygons │
│  - Format whitespace    │
│  - Create masks         │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  Annotation Masks (PNG) │
│  - view_annotations.png │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  save_bounding_boxes    │
│  - Connected components │
│  - Extract bboxes       │
│  - Crop tiles           │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  Bounding Box Tiles     │
│  - im/*.tif (images)    │
│  - label/*.tif (masks)  │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  combine_annotations    │
│  - Load bbox tiles      │
│  - Augment (rotate)     │
│  - Optimal placement    │
│  - Iterative fill       │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  Big Tiles              │
│  (10000×10000)          │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  Split into small tiles │
│  - Remove margins       │
│  - Grid splitting       │
│  - Filter by content    │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  Training Tiles         │
│  - training/im/*.png    │
│  - training/label/*.png │
│  - validation/im/*.png  │
│  - validation/label/*.png│
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  PyTorch Metadata       │
│  - annotations.pkl      │
│  - train_list.pkl       │
└─────────────────────────┘
```

### Processing Stages

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **1. XML Parsing** | `*.xml` files | Multi-strategy parsing, encoding handling | DataFrame (`xyout`) |
| **2. Serialization** | DataFrame | Package with metadata | `annotations.pkl` |
| **3. Mask Creation** | `annotations.pkl` | Polygon interpolation, whitespace formatting | `view_annotations.png` |
| **4. Bbox Extraction** | Annotation masks + images | Connected components, cropping | `boundbox/im/*.tif`, `boundbox/label/*.tif` |
| **5. Big Tile Assembly** | Bounding box tiles | Iterative placement, augmentation | Big tiles (10000×10000) |
| **6. Tile Splitting** | Big tiles | Grid splitting, margin removal | Small tiles (1024×1024) |
| **7. Metadata Generation** | Tile IDs | Create PyTorch-compatible mappings | `annotations.pkl`, `train_list.pkl` |

---

## Key File Paths Reference

### Stage 1: Annotation Processing

| File Type | Path Template | Format | Description |
|-----------|--------------|--------|-------------|
| **Input XML** | `{annotation_path}/*.xml` | XML | Polygon vertices, MicronsPerPixel |
| **Parsed Data** | `{annotation_path}/data py/{image}/annotations.pkl` | Pickle | Coordinates, metadata, statistics |
| **Annotation Mask** | `{annotation_path}/data py/{image}/view_annotations.png` | PNG | Full-resolution annotation mask |
| **Raw Mask** | `{annotation_path}/data py/{image}/view_annotations_raw.png` | PNG | Pre-whitespace processing |
| **Bbox Images** | `{annotation_path}/data py/{image}/{model}_boundbox/im/*.tif` | TIFF | Cropped image tiles |
| **Bbox Labels** | `{annotation_path}/data py/{image}/{model}_boundbox/label/*.tif` | TIFF | Cropped annotation tiles |

### Stage 2: Training Tile Generation

| File Type | Path Template | Format | Description |
|-----------|--------------|--------|-------------|
| **Model Config** | `{model_path}/net.pkl` | Pickle | Tile size, class names, counts |
| **Train Images** | `{model_path}/training/im/*.png` | PNG/TIFF | Training tiles (1024×1024) |
| **Train Labels** | `{model_path}/training/label/*.png` | PNG/TIFF | Training annotation masks |
| **Val Images** | `{model_path}/validation/im/*.png` | PNG/TIFF | Validation tiles |
| **Val Labels** | `{model_path}/validation/label/*.png` | PNG/TIFF | Validation annotation masks |
| **Big Tiles (Train)** | `{model_path}/training/big_tiles/HE_tile_*.png` | PNG/TIFF | Reference composites (10000×10000) |
| **Big Labels (Train)** | `{model_path}/training/big_tiles/label_tile_*.png` | PNG/TIFF | Reference label composites |
| **Big Tiles (Val)** | `{model_path}/validation/big_tiles/HE_tile_*.png` | PNG/TIFF | Validation reference composites |
| **PyTorch Mapping** | `{model_path}/annotations.pkl` | Pickle | Image ID mapping dictionary |
| **PyTorch Train List** | `{model_path}/train_list.pkl` | Pickle | List of training image IDs |

### Configuration Files

| File | Path | Format | Description |
|------|------|--------|-------------|
| **Model Config** | `base/config.py` | Python | `ModelDefaults.TILE_GENERATION_MODE` |
| **Tile Config** | `base/data/tiles.py` | Python | `MODERN_CONFIG`, `LEGACY_CONFIG` |
| **Environment** | Shell | Env Var | `CODAVISION_TILE_GENERATION_MODE` |

---

## Configuration Parameter Reference

| Parameter | Modern | Legacy | Description |
|-----------|--------|--------|-------------|
| `reduction_factor` | 10 | 5 | Downsampling for placement (lower = finer optimization) |
| `use_disk_filter` | False | True | Apply disk filter convolution before distance transform |
| `crop_rotations` | False | True | Crop rotated images to original size (may add black pixels) |
| `class_rotation_frequency` | 5 | 3 | Class rotation frequency (lower = more frequent rotation) |
| `deterministic_seed` | 3 | None | Random seed (None = diverse runs, non-reproducible) |
| `big_tile_size` | 10240 | 10000 | Big tile dimensions (pixels) |
| `file_format` | png | tif | Output file format for tiles |

---

## Additional Notes

### Performance Considerations

- **Bounding box creation**: Faster than full-image processing
- **Disk filter**: Adds ~10% overhead in legacy mode
- **Rotation cropping**: Modern mode reduces black pixels by ~15%
- **File format**: PNG is ~30% faster to read/write than TIFF

### Validation

- Training and validation tiles use completely separate annotations
- No overlap between training and validation sets
- Big tiles saved for visual quality control

### Reproducibility

- Modern mode is deterministic (seed=3)
- Legacy mode is non-deterministic (no seed)
- Use modern mode for reproducible experiments

### Compatibility

- Both modes produce tiles compatible with all CODAvision trainers
- Legacy mode matches MATLAB behavior for validation studies
- Modern mode empirically performs better on most datasets

---

## See Also

- `TILE_GENERATION_ANALYSIS.md` - Detailed comparison of modern vs legacy modes
- `ML_WORKFLOW.md` - Complete machine learning workflow documentation
- `MODEL_PLUGIN_ARCHITECTURE.md` - Model architecture and training details
