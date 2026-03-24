# CLAUDE.md

## Overview

CODAvision is a biomedical image segmentation platform for semantic segmentation of histopathological whole-slide images (WSI). It uses deep learning with both TensorFlow and PyTorch backends.

| Key Fact | Value |
|---|---|
| Python | `>=3.9, <3.11` (`pyproject.toml:107`) |
| Build system | Hatchling (`pyproject.toml:1-3`) |
| Default framework | PyTorch (`base/config.py:68`) |
| GUI framework | PySide6 (`gui/`) |
| Entry point | `python CODAvision.py` or `CODAvision` CLI command |
| Non-GUI entry | `scripts/non-gui_workflow.py` |

## Quick Reference

```bash
# Run application
python CODAvision.py

# Run all tests
pytest

# Run by category
pytest -m unit
pytest -m integration
pytest -m gui
pytest -m pytorch
pytest -m "not slow"

# Run tests with coverage
pytest --cov=base --cov=gui --cov-report=html

# Run in parallel
pytest -n auto

# Change framework: edit base/config.py line 68
# ModelDefaults.DEFAULT_FRAMEWORK = "pytorch"  # or "tensorflow"
```

### Test Markers (from `pytest.ini`)

`unit` `integration` `gui` `slow` `requires_data` `benchmark` `gpu` `mock_heavy` `critical` `real_tensorflow` `mock_tensorflow` `pytorch` `pytorch_integration` `pytorch_training` `cross_framework` `requires_tensorflow`

## Project Structure

```
CODAvision.py                          # Main entry point → launches GUI

base/                                  # Core business logic
├── __init__.py                        # Public API exports (all 10 pipeline stages)
├── config.py                          # ALL defaults: ModelDefaults, FrameworkConfig, TileGenerationConfig, etc.
├── models/
│   ├── base.py                        # Framework enum, BaseSegmentationModelInterface, detect_framework_availability()
│   ├── backbones.py                   # FACTORY: model_call() routes to TF or PyTorch; unfreeze_model()
│   ├── backbones_tf.py                # TF DeepLabV3+ and UNet (BaseSegmentationModel, resnet50_preprocess)
│   ├── backbones_pytorch.py           # PyTorch DeepLabV3+ only (DeepLabV3PlusModel, gradient checkpointing)
│   ├── wrappers.py                    # PyTorchKerasAdapter, TensorFlowKerasAdapter, load_model(), create_model_adapter()
│   ├── training.py                    # TF trainers: DeepLabV3PlusTrainer, UNetTrainer; train_segmentation_model_cnns() dispatch
│   ├── training_pytorch.py            # PyTorch trainers: PyTorchDeepLabV3PlusTrainer, WeightedCrossEntropyLoss
│   ├── metadata.py                    # ModelMetadata class for net.pkl management
│   └── utils.py                       # save_model_metadata, setup_gpu, calculate_class_weights, get_model_paths
├── data/
│   ├── annotation.py                  # XML annotation parsing (ASAP format), load_annotation_data()
│   ├── tiles.py                       # Training tile generation (class-balanced compositing), create_training_tiles()
│   ├── loaders.py                     # TF dataset creation, load_model_metadata()
│   ├── loaders_pytorch.py             # PyTorch Dataset/DataLoader (num_workers=0 on Windows)
│   └── geojson2xml.py                 # GeoJSON to XML annotation conversion
├── image/
│   ├── classification.py              # ImageClassifier: sliding window inference with overlap
│   ├── segmentation.py                # semantic_seg() low-level inference
│   ├── augmentation.py                # augment_annotation() for tile creation
│   ├── utils.py                       # decode_segmentation_masks, create_overlay, load_image_with_fallback
│   └── wsi.py                         # WSI2tif() whole-slide image downsampling
├── evaluation/
│   ├── testing.py                     # test_segmentation_model() orchestration
│   ├── confusion_matrix.py            # Performance metrics
│   ├── image_quantification.py        # quantify_images() tissue percentages
│   ├── object_quantification.py       # quantify_objects() component analysis
│   ├── pdf_report.py                  # create_output_pdf() report generation
│   └── visualize.py                   # plot_cmap_legend visualization
├── tissue_area/
│   ├── threshold.py                   # determine_optimal_TA() tissue detection
│   ├── threshold_core.py              # Core threshold algorithms
│   ├── models.py                      # Tissue area data models
│   └── utils.py                       # calculate_tissue_mask()
└── utils/
    └── logger.py                      # Logger class: structured JSON logging, GPU monitoring, rotating files

gui/                                   # PySide6 GUI layer
├── __init__.py
├── application.py                     # CODAVision() main function, 10-stage pipeline orchestration
├── utils.py                           # GUI utility functions
├── components/
│   ├── main_window.py                 # MainWindow: 4 tabs (paths, layers, nesting, settings)
│   ├── classification_window.py       # MainWindowClassify: results visualization
│   ├── dialogs.py                     # User interaction dialogs
│   └── ui_definitions.py              # UI layout definitions
├── tissue_area/
│   ├── threshold_gui.py               # GUI for threshold selection
│   └── dialogs.py                     # Tissue area dialogs
└── resources/
    ├── dark_theme.qss                 # Dark theme stylesheet
    └── logoCODAvision.png             # Application logo

scripts/                               # Standalone utilities
├── check_cuda_version.py              # CUDA detection and PyTorch install guidance
├── non-gui_workflow.py                # Programmatic pipeline (no GUI)
├── hyperparameter_search.py           # Hyperparameter search utilities
├── hyperparameter_utils.py            # Search helper functions
├── generate_search_figures.py         # Visualization for search results
└── monitor_resources.py               # System resource monitoring

docs/                                  # Extended documentation
├── ANNOTATION_PROCESSING.md           # Whitespace handling and annotation pipeline detail
├── ML_WORKFLOW.md                     # Machine learning workflow documentation
└── MODEL_PLUGIN_ARCHITECTURE.md       # Framework adapter architecture

tests/                                 # Test suite (pytest)
├── conftest.py                        # Shared fixtures, QT_QPA_PLATFORM=offscreen
├── unit/                              # Unit tests
│   ├── test_l2_regularization.py
│   ├── test_matlab_alignment.py
│   ├── test_pytorch_models.py
│   ├── test_pytorch_training.py
│   ├── test_tile_config.py
│   ├── test_validation_frequency.py
│   └── evaluation/
│       └── test_pdf_report.py
├── integration/                       # Integration tests
│   ├── test_pytorch_workflow.py
│   ├── test_tile_workflow.py
│   └── test_tissue_area_workflow.py
├── test_gui/                          # GUI tests
│   └── mock_dialogs.py
└── test_tissue_area/
    └── test_threshold_functionality.py
```

## Dual-Framework Architecture

### Framework Availability Matrix

| Feature | PyTorch | TensorFlow |
|---|---|---|
| DeepLabV3+ | Yes | Yes |
| UNet | **No** | Yes |
| Model file format | `.pth` | `.keras` |
| Available models | `ModelDefaults.PYTORCH_MODELS` | `ModelDefaults.TENSORFLOW_MODELS` |
| Data loading | `loaders_pytorch.py` (torch DataLoader) | `loaders.py` (tf.data) |
| Training module | `training_pytorch.py` | `training.py` |
| Gradient checkpointing | Yes (auto-enabled on CPU) | No |

### Model Creation Flow

```
model_call(name, IMAGE_SIZE, NUM_CLASSES, framework=None, wrap_with_adapter=True)
    │
    ├─ framework=None → reads ModelDefaults.DEFAULT_FRAMEWORK from config.py
    ├─ detect_framework_availability() → checks imports
    │
    ├─ framework="tensorflow"
    │   ├─ name="DeepLabV3_plus" → TFDeepLabV3Plus(IMAGE_SIZE, NUM_CLASSES, l2).build_model()
    │   └─ name="UNet"           → TFUNet(IMAGE_SIZE, NUM_CLASSES, l2).build_model()
    │   (returns tf.keras.Model directly)
    │
    └─ framework="pytorch"
        ├─ Validates name in ModelDefaults.PYTORCH_MODELS (only DeepLabV3_plus)
        ├─ name="DeepLabV3_plus" → PyTorchDeepLabV3Plus(IMAGE_SIZE, NUM_CLASSES, l2).build_model()
        │
        ├─ wrap_with_adapter=True  → PyTorchKerasAdapter(pytorch_model)  [for inference]
        └─ wrap_with_adapter=False → raw nn.Module                       [for PyTorch-native training]
```

### PyTorchKerasAdapter Pattern (`base/models/wrappers.py`)

The adapter wraps a PyTorch `nn.Module` to provide a Keras-compatible API. This allows the existing TF-based inference pipeline to work with PyTorch models without modification.

**What it does:**
- Converts NHWC (TF format) ↔ NCHW (PyTorch format) automatically on predict/fit
- Provides `.predict()`, `.compile()`, `.fit()`, `.save()`, `.load_weights()` matching Keras API
- Auto-detects device (CUDA → MPS → CPU)
- `.model` attribute exposes the raw `nn.Module` for PyTorch-native operations

**When to use `wrap_with_adapter`:**
- `True` (default): For inference pipeline, ImageClassifier, or anywhere Keras API is expected
- `False`: For PyTorch-native training in `training_pytorch.py` (needs `.to()`, `.train()`, `.parameters()`)

### Training Dispatch (`train_segmentation_model_cnns()` in `base/models/training.py`)

```
train_segmentation_model_cnns(pthDL)
    │
    ├─ Loads net.pkl → gets model_type
    ├─ Reads framework from get_framework_config()
    │
    ├─ pytorch + DeepLabV3_plus → PyTorchDeepLabV3PlusTrainer(pthDL).train()
    ├─ pytorch + UNet           → ValueError (not implemented)
    ├─ tensorflow + DeepLabV3_plus → DeepLabV3PlusTrainer(pthDL).train()
    └─ tensorflow + UNet           → UNetTrainer(pthDL).train()
```

### Training Parameters (from `base/config.py` ModelDefaults)

| Parameter | Value | Config Location |
|---|---|---|
| Default input/tile size | **512** | `ModelDefaults.INPUT_SIZE` |
| Default batch size | **8** (ModelDefaults) / **3** (GUI/metadata) | See Gotchas section |
| Learning rate | **1e-4** | `ModelDefaults.LEARNING_RATE` |
| Epochs | **8** | `ModelDefaults.EPOCHS` |
| Early stopping patience | **6** | `ModelDefaults.ES_PATIENCE` |
| LR reduction patience | **1** | `ModelDefaults.LR_PATIENCE` |
| LR reduction factor | **0.75** | `ModelDefaults.LR_FACTOR` |
| Minimum LR | **1e-7** | `ModelDefaults.MIN_LR` |
| Validation frequency | **128** iterations | `ModelDefaults.VALIDATION_FREQUENCY` |
| Optimizer epsilon | **1e-8** | `ModelDefaults.OPTIMIZER_EPSILON` |

### Image Preprocessing (CRITICAL)

Both TF and PyTorch models apply **identical** in-model preprocessing:

1. Images are fed as **0-255 uint8/float32 RGB** (NOT normalized to [0,1])
2. **RGB → BGR** channel reversal
3. **ImageNet mean subtraction** in BGR order: `[103.939, 116.779, 123.68]`

This is implemented as a layer inside the model:
- TF: `resnet50_preprocess()` in `backbones_tf.py:85-107`
- PyTorch: `preprocess()` in `backbones_pytorch.py:553-567` using `self.imagenet_mean_bgr` registered buffer

## Key Architectural Patterns

### Metadata-Driven Configuration (net.pkl)

All model configurations are stored in `net.pkl` (pickle) in the model directory. Key fields:
`pthDL`, `pthim`, `WS`, `nm`, `umpix`, `sxy` (tile size), `cmap`, `ntrain`, `nvalidate`, `model_type`, `batch_size`, `classNames`, `final_df`, `combined_df`, `scale`, `framework`

Load with: `load_model_metadata(model_path)` from `base/data/loaders.py`

At runtime, **net.pkl values override config.py defaults** (e.g., batch_size, tile_size).

### Whitespace Handling System (WS Parameter)

The `WS` parameter is a 5-element list controlling complex annotation behavior:
```python
WS = [
    [0, 0, 2, ...],  # Per-class: 0=remove whitespace, 1=keep only whitespace, 2=keep both
    [7, 6],          # Class indices to receive removed whitespace
    [1, 2, 3, ...],  # Class renaming/combining indices
    [7, 1, 6, ...],  # Nesting priority (reverse order for overlapping regions)
    []               # Class indices to delete entirely
]
```
See `docs/ANNOTATION_PROCESSING.md` for full documentation.

### Two-Phase Workflow

**Phase 1: GUI Configuration** (`gui/components/main_window.py`)
- User configures paths, resolution, layers, nesting, training parameters
- Saves configuration to `net.pkl` and closes

**Phase 2: Pipeline Execution** (`gui/application.py`)
10 sequential stages:
1. `WSI2tif()` — Downsample images to target resolution
2. `determine_optimal_TA()` — Calculate tissue area thresholds
3. `load_annotation_data()` — Parse XML, create annotation masks
4. `create_training_tiles()` — Generate class-balanced training tiles
5. `train_segmentation_model_cnns()` — Train neural network
6. `test_segmentation_model()` — Evaluate on test set
7. `classify_images()` — Classify training images
8. `quantify_images()` — Calculate tissue percentages
9. `quantify_objects()` — Component analysis
10. `create_output_pdf()` — Generate report

### Sliding Window Inference (`base/image/classification.py`)

- Tile size is configurable (read from net.pkl `sxy`)
- 200-pixel overlap (100px border on each side)
- `step_size = image_size - 200`
- Borders are removed after inference to avoid edge artifacts
- Results are stitched for seamless full-image segmentation

### Class-Based API with Legacy Wrappers

```python
# Modern (preferred)
classifier = ImageClassifier(image_path, model_path, model_type)
classifier.classify(color_overlay=True)

# Legacy wrapper (calls class internally)
classify_images(image_path, model_path, model_type, color_overlay=True)
```

New features should use the class-based approach with backward-compatible function wrappers.

### Logging System (`base/utils/logger.py`)

`Logger` class provides structured JSON logging, GPU monitoring, rotating log files. Uses `RotatingFileHandler` with configurable max bytes and backup count (see `LoggingConfig` in config.py).

## Gotchas and Pitfalls

### Framework Limitations
- **PyTorch does NOT support UNet** — only DeepLabV3+. Check `ModelDefaults.PYTORCH_MODELS`.
- Models trained with one framework cannot be used with the other (`.pth` vs `.keras`).
- `load_model()` in `wrappers.py` raises `NotImplementedError` for `.pth` files — PyTorch models must be manually reconstructed and loaded.

### Image Preprocessing
- Images are **NOT** normalized to [0,1]. They are kept in 0-255 range.
- In-model preprocessing applies RGB→BGR + ImageNet mean subtraction `[103.939, 116.779, 123.68]`.
- Both frameworks implement identical preprocessing to ensure model compatibility.

### Configuration Inconsistencies
- **Batch size**: `ModelDefaults.BATCH_SIZE = 8` but GUI/ModelMetadata defaults to `batch_size=3`. At runtime, net.pkl value wins.
- **Tile size**: `ModelDefaults.INPUT_SIZE = 512` is the default, but net.pkl `sxy` field controls actual tile size.
- **Config at runtime comes from net.pkl**, not from config.py defaults. Always check net.pkl first.
- Tile size must be a **multiple of 32** (model architecture constraint).

### Platform-Specific Behavior
- **Windows**: PyTorch DataLoader uses `num_workers=0` (main process only) due to `spawn` multiprocessing overhead. See `loaders_pytorch.py:28`.
- **macOS Silicon**: Requires `pip install -e ".[macos-silicon]"` for TensorFlow GPU. PyTorch MPS works automatically.

### Training Details
- **Gradient checkpointing IS implemented** in PyTorch (`backbones_pytorch.py:541-574`). Auto-enabled on CPU in `training_pytorch.py:1111-1114`.
- Augmentations are applied during **tile creation** (`base/data/tiles.py`), NOT during training.
- TF UNet uses two-phase transfer learning (freeze then unfreeze encoder). TF DeepLabV3+ uses single-phase.
- PyTorch training uses iteration-based validation (`VALIDATION_FREQUENCY = 128`), not epoch-based.
- Random seed is `42` (`RuntimeConfig.seed` in config.py) for reproducibility.

### Code Organization
- `backbones.py` is the **factory/router** — not where model architectures live. Actual architectures are in `backbones_tf.py` and `backbones_pytorch.py`.
- `training.py` contains both TF trainers AND the `train_segmentation_model_cnns()` dispatch function.
- Re-exports in `backbones.py` provide backward compatibility: `from base.models.backbones import DeepLabV3Plus` works (routes to TF).

## Common Development Tasks

### Adding a New Model Architecture

1. **Base**: Add to `ModelDefaults.PYTORCH_MODELS` or `TENSORFLOW_MODELS` in `config.py`
2. **Architecture**: Implement in `backbones_pytorch.py` or `backbones_tf.py` inheriting from `BaseSegmentationModelInterface` (in `base.py`)
3. **Factory**: Add routing in `model_call()` in `backbones.py`
4. **Training**: Create trainer class in `training_pytorch.py` or `training.py`
5. **Dispatch**: Add case to `train_segmentation_model_cnns()` in `training.py`
6. **GUI**: Update model selection dropdown in `gui/components/main_window.py`

### Adding Augmentations

1. Add augmentation function to `base/image/augmentation.py`
2. Integrate into `base/data/tiles.py` tile creation pipeline
3. Augmentations apply at tile creation time, not during training

### Debugging Training Issues

- **Low accuracy**: Check class balance in training tiles; verify WS parameter; check nesting order (Tab 3)
- **OOM errors**: Reduce batch size in net.pkl or GUI settings; check with `nvidia-smi`; tile size must be multiple of 32
- **Loss not decreasing**: Verify learning rate (default 1e-4); check loss weights from `calculate_class_weights()`; examine training tiles

## Code Conventions

- Python `>=3.9, <3.11` (strict, from `pyproject.toml`)
- PEP 8 style; type hints encouraged
- Descriptive variable names (avoid single letters except math contexts)
- New features: class-based API + backward-compatible function wrapper
- Conditional imports for framework-specific code (`try: import torch ... except ImportError`)
- Public API exported through `base/__init__.py`

## Additional Documentation

- `docs/ANNOTATION_PROCESSING.md` — Whitespace handling deep-dive
- `docs/ML_WORKFLOW.md` — ML training workflow
- `docs/MODEL_PLUGIN_ARCHITECTURE.md` — Framework adapter design
- `README.md` — Installation, GPU setup, CUDA troubleshooting
- Paper: bioRxiv preprint at https://www.biorxiv.org/content/10.1101/2025.04.11.648464v1
