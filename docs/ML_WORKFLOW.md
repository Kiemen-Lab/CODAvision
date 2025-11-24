# CODAvision ML Workflow - Technical Reference

## 1. Workflow Overview

CODAvision implements a complete medical image analysis pipeline for pathology and whole slide imaging (WSI). The pipeline supports dual ML frameworks (TensorFlow 2.10, PyTorch) with seamless interoperability.

**Pipeline Stages:**
```
Input Data → Preprocessing → Model Training → Inference → Evaluation
     ↓            ↓              ↓              ↓           ↓
   XML/GeoJSON  Tissue Area   TF/PyTorch    Sliding     Metrics/
   Annotations   Detection     Training      Window    Visualization
                 Tile Gen                  Classification
```

**Framework Support:**
- **TensorFlow 2.10**: Native Keras API, legacy compatibility
- **PyTorch**: 9x faster inference (64 vs 7 img/s), adapter layer for Keras compatibility
- **Interoperability**: Automatic model format detection (.keras, .pth), zero code changes required

---

## 2. Input Data

### Supported Image Formats
- **Whole Slide Images (WSI)**: .svs, .ndpi, .tif (via OpenSlide)
- **Standard Images**: TIFF, PNG, JPEG
- **Medical Formats**: DICOM (via pydicom)

### Annotation Formats
- **Primary**: XML (ASAP format)
- **Secondary**: GeoJSON (QuPath exports, converted to XML)

### Directory Structure
```
project/
├── images/
│   ├── training/
│   │   ├── image1.tif
│   │   └── image2.tif
│   └── testing/
│       └── test_image.tif
├── annotations/
│   ├── image1.xml
│   ├── image2.xml
│   └── data py/              # Auto-generated
│       └── image1/
│           └── annotations.pkl
└── models/
    └── my_model/
        ├── training/
        │   ├── im/           # Training tiles
        │   └── label/        # Label tiles
        └── validation/
```

### Annotation Structure
**XML Format** (`base/data/annotation.py:load_xml_annotations`):
- Hierarchical layers with polygonal regions
- `MicronsPerPixel`: Scale calibration attribute
- `LineColor`: RGB class color (e.g., "#FF0000")
- Vertices stored as (X, Y) coordinate lists
- Pickled output: `annotations/data py/<image_name>/annotations.pkl`

**Key Fields in Pickled Annotations**:
- `xyout`: Dictionary mapping layer names to polygon vertices
- `reduce_annotations`: Downsampling factor for visualization
- `umpix`: Microns per pixel (float)
- `dm`: Modification date
- `WS`: Whitespace settings

---

## 3. Preprocessing Pipeline

### 3.1 Tissue Area Detection
**Module**: `base/tissue_area/threshold.py:determine_optimal_TA`

**Algorithm**:
1. Sample N images from training/testing set (default: 20)
2. Interactive or automatic threshold determination
3. Generate binary tissue masks (tissue=1, background=0)
4. Apply threshold across all images

**Output**:
- Tissue masks: `<image_path>/TA/<image_name>.png|.tif`
- Threshold values: `<image_path>/TA/TA_cutoff.pkl`

**Purpose**: Exclude background regions during tile generation and inference

---

### 3.2 Tile Generation
**Module**: `base/data/tiles.py:create_training_tiles`

**Configuration Modes**:

| Parameter | Modern (Default) | Legacy (MATLAB) | Description |
|-----------|------------------|-----------------|-------------|
| `reduction_factor` | 10 | 5 | Downsampling for placement optimization (lower = finer) |
| `use_disk_filter` | False | True | Apply 51x51 disk filter convolution for placement |
| `crop_rotations` | False | True | Crop rotated images back to original size |
| `class_rotation_frequency` | 5 | 3 | Rotate through classes every N iterations |
| `deterministic_seed` | 3 | None | Random seed (None = diverse runs) |
| `big_tile_size` | 10240 | 10000 | Composite tile dimensions (px) |
| `file_format` | png | tif | Output format |

**Environment Variable**: `CODAVISION_TILE_GENERATION_MODE=modern|legacy`

**Algorithm** (`combine_annotations_into_tiles`):
1. **Initialize Canvas**: Create 10240×10240 (or 10000×10000) blank composite tile
2. **Class Selection**: Rotate through classes every N iterations to ensure balance
3. **Candidate Selection**: Randomly choose bounding box tile containing target class
4. **Augmentation** (configurable):
   - Rotation: 0-355° in 5° steps (72 options)
   - Scaling: 0.60-0.95, 1.10-1.40 in 0.01 steps (71 options)
   - Hue shift: Per-channel 0.88-1.12 (21 options/channel)
   - Blur: Gaussian σ=1.0-1.2 (8% probability)
5. **Placement Optimization**:
   - Downsample mask by reduction factor (5 or 10)
   - Optional disk filter convolution (legacy mode only)
   - Distance transform to find largest empty area
   - 100-pixel padding from edges
6. **Tile Placement**: Center tile at optimal location
7. **Fill Monitoring**: Track fill ratio until 55% threshold
8. **Iteration Limits**:
   - MAX_ITERATIONS = 10,000
   - MAX_CONSECUTIVE_FAILURES = 10
9. **Tile Splitting**: Split big tile into `sxy × sxy` training tiles (e.g., 512×512)
10. **Output**: Save to `training/` or `validation/` directories

**Output Structure**:
```
<model_path>/
  training/
    im/           # Training image tiles (PNG/TIFF)
    label/        # Training label tiles (PNG/TIFF)
    big_tiles/    # Composite tiles (HE_tile_*.png, label_tile_*.png)
  validation/
    im/           # Validation tiles
    label/        # Validation labels
    big_tiles/    # Validation composites
```

**Safeguards**:
- Empty distance transform → fallback to random placement
- Consecutive failure limit → prevents infinite loops
- Max iterations → guarantees termination

---

### 3.3 Data Augmentation
**Module**: `base/image/augmentation.py:augment_image`

**Augmentation Types**:

**Rotation** (if enabled):
- Random angles: 0-355° (5° steps, 72 options)
- Expands canvas to fit rotated image (no clipping)
- Optional cropping: Legacy mode crops to original size, Modern keeps expanded

**Scaling** (if enabled):
- Scale factors: 0.60-0.95, 1.10-1.40 (0.01 steps, 71 options)
- Interpolation: Bilinear (images), Nearest-neighbor (masks)

**Hue Shift** (if enabled):
- Per-channel intensity: 0.88-0.98, 1.02-1.12 (21 options/channel)
- Applied independently to R, G, B

**Blur** (if enabled):
- Gaussian blur: σ ∈ {1.0, 1.05, 1.1, 1.15, 1.2}
- Probability: 4/50 (8%)

**Configuration**: Set frequencies in tile generation config (e.g., rotate every 5 iterations)

---

## 4. Model Architecture

### 4.1 Backbone Options
**Module**: `base/models/backbones.py:model_call`, `base/models/backbones_pytorch.py`

**Available Architectures**:

#### DeepLabV3+ (TensorFlow & PyTorch)
- **Encoder Options**:
  - TensorFlow: ResNet50, ResNet101, Xception
  - PyTorch: ResNet50, ResNet101, MobileNetV2, EfficientNet-B0
- **ASPP Module**: Atrous Spatial Pyramid Pooling with dilation rates [6, 12, 18]
- **Decoder**: Bilinear upsampling (4×) + skip connections from low-level features
- **Output**: Softmax activation (NUM_CLASSES channels)

**Input Specifications**:
- TensorFlow: `(batch, 512, 512, 3)` - NHWC format
- PyTorch: `(batch, 3, 512, 512)` - NCHW format (adapter handles conversion)
- Default IMAGE_SIZE: 512 (configurable)

**Output Shape**: `(batch, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES)`

**L2 Regularization**:
- Applied to all Conv2D layers
- Default weight: 0 (disabled)
- Configured via `ModelDefaults.L2_REGULARIZATION_WEIGHT` in `base/config.py`

#### UNet (TensorFlow Only)
- **Encoder**: 4 downsampling blocks (64→128→256→512 filters)
- **Bottleneck**: 1024 filters
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Softmax activation

---

### 4.2 Framework Adapter Layer
**Module**: `base/models/wrappers.py:PyTorchKerasAdapter`

**Purpose**: Enable PyTorch models to use Keras API with automatic tensor format conversion

**Key Features**:
- **Automatic Tensor Conversion**: NHWC ↔ NCHW transparent conversion
- **Full Keras Compatibility**:
  - `predict(x, batch_size)` - Batch inference
  - `fit(x, y, epochs, batch_size)` - Training loop
  - `compile(optimizer, loss, metrics)` - Configure training
  - `save(filepath)` / `load_weights(filepath)` - Persistence
  - `summary()` - Architecture display
  - `trainable` property - Control training mode

**Device Support**:
- Auto-detection priority: CUDA > MPS > CPU
- Manual specification: `device='cuda'|'mps'|'cpu'`
- Environment override: `CODAVISION_PYTORCH_DEVICE=cuda|mps|cpu`

**Performance** (Apple M3 Max, 512×512 images):
- PyTorch inference: **64 img/s** (9× faster)
- TensorFlow inference: **7 img/s**
- Recommended batch size: 4 (PyTorch), 2 (TensorFlow)

**Advanced Features**:
- Mixed Precision (AMP): `CODAVISION_PYTORCH_AMP=1`
- PyTorch 2.0 Compilation: `CODAVISION_PYTORCH_COMPILE=1`
- Gradient Accumulation: `CODAVISION_GRADIENT_ACCUMULATION_STEPS=4`

**Model Loading** (`base/models/backbones.py:load_model_weights`):
- Auto-detects format: `.pth` (PyTorch) or `.keras` (TensorFlow)
- Priority: Framework env var > best model > final model
- Transparent adapter wrapping for PyTorch models

---

## 5. Training Process

### 5.1 Data Loading
**Module**: `base/data/loaders.py`

**TensorFlow Dataset Pipeline**:
```
Image Loading → Cache (optional) → Shuffle → Batch → Prefetch
```

**Functions**:
- `create_training_dataset()`: Optimized for training (shuffle enabled)
- `create_validation_dataset()`: Optimized for validation (shuffle disabled)

**Optimizations**:
- `num_parallel_calls=tf.data.AUTOTUNE` - Parallel loading
- `prefetch(tf.data.AUTOTUNE)` - Prefetch next batch during training
- `cache()` - In-memory caching for small datasets
- `shuffle(buffer_size=1000, reshuffle_each_iteration=True)` - Training randomization

**TIFF Support**:
- Uses PIL via `tf.py_function` (TensorFlow 2.10 lacks native `decode_tiff`)
- Handles multi-channel TIFF, automatic channel conversion

**Configuration** (`base/config.py:DataConfig`):
- `prefetch_buffer_size`: 2
- `shuffle_buffer_size`: 1000
- `num_parallel_calls`: -1 (AUTOTUNE)

---

### 5.2 TensorFlow Training
**Module**: `base/models/training.py`

**Loss Function** (`WeightedSparseCategoricalCrossentropy`):
- **Per-class mean loss** (MATLAB-aligned for class balance)
- Algorithm:
  1. Compute mean loss for each class separately
  2. Apply normalized class weights
  3. Sum weighted class losses (not average)
- Handles severe class imbalance effectively

**Training Configuration** (`base/config.py:ModelDefaults`):
- Optimizer: Adam (epsilon=1e-8 for numerical stability)
- Learning Rate: 1e-4
- Batch Size: 8
- Epochs: 100
- Validation Frequency: Every **128 iterations** (not epoch-based)
  - Override: `CODAVISION_VALIDATION_FREQUENCY=<int>`
  - Guarantees ≥1 validation per training run

**Learning Rate Schedule** (ReduceLROnPlateau):
- Patience: 1 epoch
- Factor: 0.75 (multiply LR by 0.75 on plateau)
- Minimum: 1e-7

**Early Stopping**:
- Patience: 6 epochs
- Monitor: Validation accuracy or loss
- Restores best weights on termination

**Callbacks**:
- `BatchAccuracyCallback` - Iteration-based validation (every N iterations)
- `RegularizationLossCallback` - Tracks L2 regularization separately
- `ModelCheckpoint` - Saves best and final models
- `EarlyStopping` - Prevents overfitting
- `ReduceLROnPlateau` - Adapts learning rate

**Model Checkpointing**:
- Best model: `<model_path>/<model_type>_best.keras`
- Final model: `<model_path>/<model_type>_final.keras`
- Metadata: `<model_path>/net.pkl` (hyperparameters, class names, color map)

**Multi-GPU Support**:
- Uses `tf.distribute.MirroredStrategy` for data parallelism
- Automatically detected via `GPUtil`

---

### 5.3 PyTorch Training
**Module**: `base/models/training_pytorch.py:DeepLabV3PlusTrainer`

**Key Differences from TensorFlow**:
- Custom training loop (no Keras `fit()`)
- Manual gradient accumulation
- Explicit device management (CUDA/MPS/CPU)
- Built-in validation at specified intervals

**Training Features**:
- Same loss function as TensorFlow (per-class mean loss)
- Validation every N iterations (configurable)
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping with patience
- Model checkpointing (.pth format)

**DataLoaders**:
- PyTorch `DataLoader` with custom `TileDataset`
- Automatic NHWC → NCHW conversion
- Pin memory enabled for faster GPU transfer
- Multi-worker support

**Checkpointing**:
- Best model: `<model_path>/<model_type>_best.pth`
- Final model: `<model_path>/<model_type>_final.pth`
- Metadata: Same `net.pkl` format as TensorFlow

---

## 6. Inference

### 6.1 Image Classification
**Module**: `base/image/classification.py:ImageClassifier`

**Classification Pipeline**:
1. **Model Loading** (`load_model_weights`):
   - Auto-detects `.pth` (PyTorch) or `.keras` (TensorFlow)
   - Wraps PyTorch models with `PyTorchKerasAdapter`
2. **Image Preprocessing**:
   - Pad image with `image_size + 100` pixels (default: 612px)
   - Prevent edge artifacts during sliding window
3. **Sliding Window Inference** (`semantic_seg`):
   - Tile size: `sxy × sxy` (e.g., 512×512, from `net.pkl`)
   - Stride: `sxy / 2` (50% overlap)
   - Batch size: Configurable (default: 4 for PyTorch, 2 for TensorFlow)
4. **Reconstruction**:
   - Stitch tiles back together
   - Average overlapping predictions
   - Remove padding
5. **Post-processing** (optional):
   - Apply tissue mask (exclude background)
   - Remove small objects (<25 pixels)
   - Generate visualizations

**Output Files**:
```
<image_path>/classification_<model_name>_<model_type>/
├── <image_name>.tif              # Single-channel label map (0-indexed)
├── check_classification/
│   └── <image_name>.jpg          # 50% image + 50% color overlay
└── color/
    └── <image_name>.png          # Pure color-coded segmentation
```

**Visualization**:
- Color mapping from `net.pkl` → `cmap` (list of RGB tuples)
- Alpha blending for transparency

---

### 6.2 Semantic Segmentation
**Module**: `base/image/segmentation.py:semantic_seg`

**Technical Details**:
- Processes arbitrarily large images via sliding window
- Returns raw probability maps (H × W × NUM_CLASSES)
- Used internally by `classify_images()`

**Algorithm**:
1. Pad image to multiples of `sxy`
2. Extract overlapping tiles (stride = `sxy / 2`)
3. Batch prediction
4. Reconstruct full image by averaging overlaps
5. Remove padding

---

### 6.3 Whole Slide Image Processing
**Module**: `base/image/wsi.py`

**Features**:
- **OpenSlide Integration**: .svs, .ndpi, .tif formats
- **Multi-resolution Pyramid**: Automatic level selection based on target resolution
- **Tile-based Processing**: Memory-efficient for gigapixel images
- **Level Selection**: Chooses optimal pyramid level for performance

**Usage**:
```python
from base.image.wsi import WSIProcessor
processor = WSIProcessor(wsi_path)
processor.process_at_level(level=0, tile_size=512)
```

---

## 7. Evaluation & Output

### 7.1 Model Testing
**Module**: `base/evaluation/testing.py:SegmentationModelTester`

**Testing Workflow**:
1. **Load Test Annotations**: `load_annotation_data(test_annotation_path)`
2. **Classify Test Images**: `classify_images(test_image_path, model_path)`
3. **Collect Predictions**: Load ground truth and predicted labels
4. **Data Cleaning**: Remove small objects (<25 pixels) from ground truth
5. **Compute Metrics**:
   - Overall accuracy: Correct pixels / total pixels
   - Per-class accuracy: Class-specific metrics
   - Confusion matrix: Pairwise class confusion

**Output**:
- Metrics logged to console and saved to file
- Confusion matrix visualization

---

### 7.2 Confusion Matrix
**Module**: `base/evaluation/confusion_matrix.py:ConfusionMatrixVisualizer`

**Features**:
- Computes confusion matrix from label arrays
- Normalizes by true class (row-wise)
- Generates matplotlib heatmap
- Annotates with counts and percentages

**Output**: `<output_path>/confusion_matrix.png`

**Usage**:
```python
from base.evaluation.confusion_matrix import ConfusionMatrixVisualizer
visualizer = ConfusionMatrixVisualizer(y_true, y_pred, class_names)
visualizer.plot(output_path='results/')
```

---

### 7.3 Visualization
**Module**: `base/evaluation/visualize.py`

**Key Functions**:
- `create_overlay(image, mask, alpha=0.5)`: Blend image with segmentation
- `decode_segmentation_masks(mask, cmap)`: Convert label map to RGB
- `create_color_legend(class_names, cmap)`: Generate class legend

**Color Mapping**:
- Uses `cmap` from `net.pkl` (list of RGB tuples)
- Example: `[(255, 0, 0), (0, 255, 0), (0, 0, 255)]` for 3 classes

**Output Formats**:
- JPEG/PNG overlays
- Pure segmentation masks
- Color legends

---

## 8. Key File Path Reference

| Module | Primary Function/Class | Purpose | Key Config Options |
|--------|------------------------|---------|-------------------|
| `base/data/annotation.py` | `load_annotation_data()` | Parse XML annotations to pickle | N/A |
| `base/data/tiles.py` | `create_training_tiles()` | Generate training tiles | `CODAVISION_TILE_GENERATION_MODE`, `config=TileGenerationConfig()` |
| `base/image/augmentation.py` | `augment_image()` | Apply augmentations | `rotation`, `scaling`, `hue_shift`, `blur`, `crop_rotations` |
| `base/tissue_area/threshold.py` | `determine_optimal_TA()` | Tissue area detection | `num_images`, `test_ta_mode` |
| `base/models/backbones.py` | `model_call()` | Model factory | `name`, `IMAGE_SIZE`, `NUM_CLASSES`, `framework` |
| `base/models/wrappers.py` | `PyTorchKerasAdapter` | PyTorch-Keras bridge | `device`, `CODAVISION_PYTORCH_DEVICE` |
| `base/data/loaders.py` | `create_training_dataset()` | TensorFlow data pipeline | `batch_size`, `shuffle_buffer_size`, `cache` |
| `base/models/training.py` | `BatchAccuracyCallback` | TensorFlow training | `CODAVISION_VALIDATION_FREQUENCY`, `learning_rate`, `epochs` |
| `base/models/training_pytorch.py` | `DeepLabV3PlusTrainer` | PyTorch training | `validation_frequency`, `learning_rate`, `epochs` |
| `base/image/classification.py` | `ImageClassifier` | Image classification | `batch_size`, `apply_tissue_mask` |
| `base/image/segmentation.py` | `semantic_seg()` | Sliding window inference | `image_size`, `batch_size` |
| `base/image/wsi.py` | `WSIProcessor` | WSI processing | `level`, `tile_size` |
| `base/evaluation/testing.py` | `SegmentationModelTester` | Model testing | `test_annotation_path`, `test_image_path` |
| `base/evaluation/confusion_matrix.py` | `ConfusionMatrixVisualizer` | Confusion matrix | `class_names`, `normalize` |
| `base/evaluation/visualize.py` | `create_overlay()` | Result visualization | `alpha`, `cmap` |
| `base/config.py` | `ModelDefaults` | Global configuration | `INPUT_SIZE`, `BATCH_SIZE`, `VALIDATION_FREQUENCY`, `TILE_GENERATION_MODE` |

---

## 9. Configuration Summary

### Environment Variables
| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `CODAVISION_FRAMEWORK` | `pytorch`, `tensorflow` | `tensorflow` | Select ML framework |
| `CODAVISION_TILE_GENERATION_MODE` | `modern`, `legacy` | `modern` | Tile generation preset |
| `CODAVISION_VALIDATION_FREQUENCY` | int | 128 | Iterations between validations |
| `CODAVISION_PYTORCH_DEVICE` | `auto`, `cuda`, `mps`, `cpu` | `auto` | PyTorch device |
| `CODAVISION_PYTORCH_COMPILE` | `0`, `1` | `0` | Enable PyTorch 2.0 compilation |
| `CODAVISION_PYTORCH_AMP` | `0`, `1` | `0` | Enable mixed precision training |
| `CODAVISION_GRADIENT_ACCUMULATION_STEPS` | int | 1 | Gradient accumulation steps |

### Key Hyperparameters (`base/config.py`)
- `INPUT_SIZE`: 512 (image size)
- `BATCH_SIZE`: 8 (training batch size)
- `LEARNING_RATE`: 1e-4 (initial LR)
- `EPOCHS`: 100 (max epochs)
- `VALIDATION_FREQUENCY`: 128 (iterations)
- `L2_REGULARIZATION_WEIGHT`: 0 (disabled)

---

## 10. Technical Insights

### Key Design Decisions
1. **Iteration-Based Validation**: Modern approach (every 128 iterations) vs traditional epoch-based
2. **Per-Class Mean Loss**: True class balancing in severely imbalanced datasets
3. **Dual Framework Support**: Complete TensorFlow ↔ PyTorch interoperability via adapter layer
4. **Configurable Tile Generation**: Two preset modes (modern/legacy) with 8 tunable parameters
5. **Sliding Window Inference**: Memory-efficient processing of arbitrarily large images (gigapixel WSI)
6. **Robust XML Parsing**: Multi-strategy parsing (4 fallback levels) for cross-platform compatibility

### Performance Characteristics
- **PyTorch Inference**: 64 img/s (9× faster than TensorFlow)
- **Recommended Batch Size**: 4 (PyTorch), 2 (TensorFlow)
- **Memory Efficiency**: Sliding window with 50% overlap prevents OOM on large images
- **Multi-GPU**: TensorFlow MirroredStrategy for data parallelism

### Validation
- **Unit Tests**: 71+ tests (pytest suite in `tests/unit/`, `tests/integration/`)
- **Verification Scripts**: `scripts/verify_pytorch_*.py` for comprehensive testing
- **Framework Equivalence**: <1e-5 numerical difference between TensorFlow and PyTorch outputs

---

## Quick Start Example

```python
from base.data.annotation import load_annotation_data
from base.data.tiles import create_training_tiles_modern
from base.models.training_pytorch import DeepLabV3PlusTrainer
from base.image.classification import classify_images

# 1. Load annotations
annotations = load_annotation_data('path/to/annotations')

# 2. Generate tiles (modern mode)
create_training_tiles_modern(
    model_path='models/my_model',
    annotations=annotations,
    image_list=['image1.tif', 'image2.tif'],
    create_new_tiles=True
)

# 3. Train model (PyTorch)
trainer = DeepLabV3PlusTrainer(
    model_path='models/my_model',
    image_size=512,
    num_classes=5
)
trainer.train(epochs=100, learning_rate=1e-4)

# 4. Classify test images
classify_images(
    pthim='path/to/test/images',
    pthDL='models/my_model',
    name='DeepLabV3_plus'
)
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-24
**Codebase Version**: CODAvision with PyTorch support (commit: 7a464db)
