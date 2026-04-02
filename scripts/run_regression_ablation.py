"""
Regression Ablation Experiment: DSAI Branch vs Main Branch

Isolates the root cause of accuracy degradation on the DSAI branch by
reverting each algorithmic change individually while keeping everything
else at DSAI defaults. Each experiment runs as a subprocess for GPU
state isolation.

Experiments:
    dsai_baseline   — DSAI as-is (control)
    main_all        — All main-branch behaviors combined
    exp_a_loss      — Main per-pixel weighted loss only
    exp_b_lr_sched  — Main counter-based LR schedule only
    exp_c_lr_init   — Main initial LR (0.0005) only
    exp_d_val_freq  — Main 3-per-epoch validation only
    exp_e_raw_wts   — Raw (un-normalized) class weights only
    exp_f_epsilon   — Default Adam epsilon (1e-7) only

Usage:
    python scripts/run_regression_ablation.py --dataset liver --gpu 0
    python scripts/run_regression_ablation.py --dataset lungs --gpu 1
    python scripts/run_regression_ablation.py --dataset liver --gpu 0 --experiments dsai_baseline main_all exp_a_loss
"""

import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from glob import glob

# Ensure project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBPROCESS_TIMEOUT = 12 * 60 * 60  # 12 hours per experiment

DATASET_CONFIGS = {
    "liver": {
        "WS": [[0, 0, 0, 0, 2, 0, 2], [7, 6], [1, 2, 3, 4, 5, 6, 7], [6, 4, 2, 3, 5, 1, 7], []],
        "CMAP": np.array([
            [230, 190, 100],
            [65, 155, 210],
            [145, 35, 35],
            [158, 24, 118],
            [30, 50, 50],
            [235, 188, 215],
            [255, 255, 255],
        ]),
        "CLASS_NAMES": ["PDAC", "bile duct", "vasculature", "hepatocyte",
                        "immune", "stroma", "whitespace"],
        "display_class_names": ["PDAC", "bile duct", "vasculature",
                                "hepatocyte", "immune", "stroma"],
        "NTRAIN": 15, "NVALIDATE": 3, "umpix": 1,
        "resolution_subdir": "10x", "test_subdir": "testing_image",
        "default_data_path": r"C:\Users\tnewton\Desktop\liver_tissue_data",
    },
    "lungs": {
        "WS": [[0, 2, 0, 0, 2, 0], [5, 6], [1, 2, 3, 4, 5, 6], [6, 2, 4, 3, 1, 5], []],
        "CMAP": np.array([
            [128, 0, 255],
            [166, 193, 202],
            [255, 0, 0],
            [128, 64, 0],
            [255, 255, 255],
            [255, 128, 192],
        ]),
        "CLASS_NAMES": ["bronchioles", "alveoli", "vasculature", "mets",
                        "whitespace", "collagen"],
        "display_class_names": ["bronchioles", "alveoli", "vasculature",
                                "mets", "collagen"],
        "NTRAIN": 15, "NVALIDATE": 3, "umpix": 2,
        "resolution_subdir": "5x", "test_subdir": "test",
        "default_data_path": r"C:\Users\tnewton\Desktop\lungs_data",
    },
}

# Experiment definitions: name, description, trainer class name, metadata overrides
EXPERIMENTS = [
    ("dsai_baseline", "DSAI as-is (control)",
     "SafeBaselineTrainer", {}),
    ("main_all", "All main-branch behaviors",
     "MainStyleFullTrainer", {"learning_rate": 0.0005}),
    ("exp_a_loss", "Main per-pixel loss only",
     "MainStyleLossTrainer", {}),
    ("exp_b_lr_sched", "Main counter-based LR schedule only",
     "MainStyleLRScheduleTrainer", {}),
    ("exp_c_lr_init", "Main initial LR (0.0005) only",
     "SafeBaselineTrainer", {"learning_rate": 0.0005}),
    ("exp_d_val_freq", "Main 3-per-epoch validation only",
     "MainStyleValidationTrainer", {}),
    ("exp_e_raw_wts", "Raw class weights (no normalization) only",
     "RawClassWeightsTrainer", {}),
    ("exp_f_epsilon", "Default Adam epsilon (1e-7) only",
     "SafeBaselineTrainer", {"optimizer_epsilon": 1e-7}),
]

EXPERIMENT_NAMES = [e[0] for e in EXPERIMENTS]


# ---------------------------------------------------------------------------
# GPU safety
# ---------------------------------------------------------------------------

def check_gpu_safety(gpu_index: str) -> None:
    """Verify the requested GPU is a P4000, not the RTX 6000."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print(f"ERROR: nvidia-smi failed: {result.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: nvidia-smi not found")
        sys.exit(1)

    gpu_info = {}
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx, name, mem = parts[0], parts[1], float(parts[2])
            gpu_info[idx] = {"name": name, "memory_used_mb": mem}

    if gpu_index not in gpu_info:
        print(f"ERROR: GPU index {gpu_index} not found. Available: {list(gpu_info.keys())}")
        sys.exit(1)

    selected = gpu_info[gpu_index]

    print(f"GPU {gpu_index} ({selected['name']}) — OK")


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def find_tile_source(data_path: str, dataset: str) -> str:
    """Find an existing tile directory to reuse."""
    candidates = [
        os.path.join(data_path, "gpu_test_results", "shared_tiles"),
        os.path.join(data_path, "regression_ablation_results", "shared_tiles"),
    ]
    # Add epoch_comparison directories (any framework, any suffix)
    for entry in glob(os.path.join(data_path, "epoch_comparison_results_*")):
        if os.path.isdir(entry):
            # Check epochs_008 first (matches our 8-epoch config)
            for epoch_dir in ["epochs_008", "epochs_025", "epochs_002"]:
                candidates.append(os.path.join(entry, epoch_dir))
            candidates.append(os.path.join(entry, "shared_tiles"))

    for path in candidates:
        if _tiles_exist(path):
            print(f"  Found existing tiles: {path}")
            return path

    return ""


def _tiles_exist(directory: str) -> bool:
    """Check if a directory contains usable tiles."""
    if not os.path.isdir(directory):
        return False
    train_im = os.path.join(directory, "training", "im")
    train_lbl = os.path.join(directory, "training", "label")
    val_im = os.path.join(directory, "validation", "im")
    net_pkl = os.path.join(directory, "net.pkl")
    return (os.path.isdir(train_im) and os.path.isdir(train_lbl)
            and os.path.isdir(val_im) and os.path.isfile(net_pkl))


# ---------------------------------------------------------------------------
# Directory setup helpers
# ---------------------------------------------------------------------------

def _link_dir(src: str, dst: str) -> None:
    """Create a Windows directory junction (or copy as fallback)."""
    if os.path.exists(dst):
        return
    try:
        subprocess.check_call(
            ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  Junction failed, copying {src} -> {dst}")
        shutil.copytree(src, dst)


def setup_experiment_dir(exp_dir: str, tile_source: str) -> None:
    """Set up an experiment directory with tile junctions and metadata copies."""
    os.makedirs(exp_dir, exist_ok=True)

    # Junction-link training and validation
    _link_dir(os.path.join(tile_source, "training"),
              os.path.join(exp_dir, "training"))
    _link_dir(os.path.join(tile_source, "validation"),
              os.path.join(exp_dir, "validation"))

    # Copy metadata files (always overwrite — worker will customize net.pkl)
    for fname in ("net.pkl", "annotations.pkl", "train_list.pkl"):
        src = os.path.join(tile_source, fname)
        dst = os.path.join(exp_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Remove old model/results files from previous runs
    for pattern in ("*.keras", "worker_results.json"):
        for old in glob(os.path.join(exp_dir, pattern)):
            os.remove(old)
    old_best = os.path.join(exp_dir, "best_model_DeepLabV3_plus.keras")
    if os.path.isdir(old_best):
        shutil.rmtree(old_best, ignore_errors=True)


def _serialise(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Tile creation (fallback if no existing tiles found)
# ---------------------------------------------------------------------------

def create_tiles(shared_dir: str, data_path: str, tile_size: int,
                 dataset: str) -> None:
    """Create training/validation tiles from scratch."""
    cfg = DATASET_CONFIGS[dataset]

    from base.models.utils import create_initial_model_metadata, save_model_metadata
    from base.data.annotation import load_annotation_data
    from base.data.tiles import create_training_tiles

    pthim = os.path.join(data_path, cfg["resolution_subdir"])
    pthtest = os.path.join(data_path, cfg["test_subdir"])

    os.makedirs(shared_dir, exist_ok=True)

    print(f"[tiles] Creating metadata in {shared_dir}")
    create_initial_model_metadata(
        pthDL=shared_dir,
        pthim=pthim,
        WS=cfg["WS"],
        nm="regression_ablation",
        umpix=cfg["umpix"],
        cmap=cfg["CMAP"],
        sxy=tile_size,
        classNames=cfg["CLASS_NAMES"],
        ntrain=cfg["NTRAIN"],
        nvalidate=cfg["NVALIDATE"],
        pthtest=pthtest,
        tile_format='png',
    )
    save_model_metadata(shared_dir, {
        "resolution_subdir": cfg["resolution_subdir"],
    })

    print("[tiles] Loading annotation data")
    ctlist0, numann0, create_new_tiles = load_annotation_data(
        shared_dir, data_path, pthim, []
    )

    print("[tiles] Creating training tiles")
    create_training_tiles(shared_dir, numann0, ctlist0, create_new_tiles)
    print("[tiles] Done")


# ===================================================================
# WORKER MODE — runs in subprocess with CUDA_VISIBLE_DEVICES already set
# ===================================================================

def _run_worker(args):
    """Train + test a single experiment configuration."""
    # Force non-interactive backend before any plotting imports
    import matplotlib
    matplotlib.use('Agg')

    import tensorflow as tf
    import keras

    from base.config import ModelDefaults, DataConfig
    ModelDefaults.DEFAULT_FRAMEWORK = "tensorflow"

    from base.models.training import (
        DeepLabV3PlusTrainer,
        WeightedSparseCategoricalCrossentropy,
        BatchAccuracyCallback,
    )
    from base.models.utils import save_model_metadata, calculate_class_weights
    from base.data.loaders import create_dataset, load_model_metadata
    from base.evaluation.testing import test_segmentation_model
    from base.utils.logger import Logger

    experiment_name = args.experiment
    model_dir = args.model_dir
    dataset = args.dataset

    # ----- Look up experiment config -----
    exp_config = None
    for name, desc, trainer_cls_name, meta_overrides in EXPERIMENTS:
        if name == experiment_name:
            exp_config = (name, desc, trainer_cls_name, meta_overrides)
            break
    if exp_config is None:
        print(f"ERROR: Unknown experiment '{experiment_name}'")
        sys.exit(1)

    _, description, trainer_cls_name, meta_overrides = exp_config

    result = {
        "experiment_name": experiment_name,
        "description": description,
        "dataset": dataset,
        "framework": "tensorflow",
        "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "success": False,
        "error": None,
        "train_time_s": None,
        "test_time_s": None,
        "test_metrics": None,
        "overall_accuracy": None,
        "epochs_completed": None,
    }

    try:
        # ----- Update metadata with experiment-specific overrides -----
        batch_size = args.batch_size
        tile_size = getattr(args, 'tile_size', 512)
        # Steps per epoch = num_tiles / batch_size. Validate ~3x per epoch.
        steps_per_epoch = 1500 // batch_size
        val_freq = max(1, steps_per_epoch // 3)
        base_meta = {
            "nm": f"ablation_{experiment_name}",  # Unique name per experiment
            "epochs": 8,
            "batch_size": batch_size,
            "sxy": tile_size,  # Model input size (tiles resized during loading)
            "es_patience": 9999,  # Disable early stopping
            "tile_format": "png",
            "framework": "tensorflow",
            "validation_frequency": val_freq,
        }
        base_meta.update(meta_overrides)
        save_model_metadata(model_dir, base_meta)

        print(f"[worker:{experiment_name}] Metadata overrides: {base_meta}")

        # ================================================================
        # Inline classes — defined here so they're only loaded in workers
        # ================================================================

        # ----- MainStyleLoss: main's per-pixel weighted loss -----
        class MainStyleLoss(tf.keras.losses.Loss):
            """Main branch loss: per-pixel weighting, raw weights, reduce_mean."""
            def __init__(self, class_weights, from_logits=True,
                         reduction='sum_over_batch_size', name=None):
                super().__init__(reduction=reduction, name=name)
                self.class_weights = tf.convert_to_tensor(
                    class_weights, dtype=tf.float32)
                self.from_logits = from_logits

            def call(self, y_true, y_pred):
                y_true = tf.cast(y_true, tf.int32)
                y_true_flat = tf.reshape(y_true, [-1])
                epsilon = 1e-7
                y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
                sample_weights = tf.gather(self.class_weights, y_true_flat)
                losses = tf.keras.losses.sparse_categorical_crossentropy(
                    y_true_flat,
                    tf.reshape(y_pred, [tf.shape(y_true_flat)[0], -1]),
                    from_logits=self.from_logits
                )
                weighted_losses = losses * sample_weights
                return tf.reduce_mean(weighted_losses)

            def get_config(self):
                config = super().get_config()
                config.update({
                    "class_weights": self.class_weights.numpy().tolist(),
                    "from_logits": self.from_logits,
                })
                return config

        # ----- RawWeightLoss: DSAI per-class mean but NO normalization -----
        class RawWeightLoss(tf.keras.losses.Loss):
            """DSAI per-class mean loss algorithm with raw (un-normalized) weights."""
            def __init__(self, class_weights, from_logits=True,
                         reduction='sum_over_batch_size', name=None):
                super().__init__(reduction=reduction, name=name)
                # KEY: raw weights — no normalization
                self.class_weights = tf.convert_to_tensor(
                    class_weights, dtype=tf.float32)
                self.from_logits = from_logits

            def call(self, y_true, y_pred):
                y_true = tf.cast(y_true, tf.int32)
                num_classes = tf.shape(y_pred)[-1]
                y_true_flat = tf.reshape(y_true, [-1])
                y_pred_flat = tf.reshape(y_pred, [-1, num_classes])
                per_pixel_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_true_flat, y_pred_flat, from_logits=self.from_logits
                )

                def compute_class_loss(class_idx):
                    class_mask = tf.cast(
                        tf.equal(y_true_flat, class_idx), tf.float32)
                    num_pixels = tf.reduce_sum(class_mask)
                    masked_loss = per_pixel_loss * class_mask
                    class_mean = tf.cond(
                        num_pixels > 0,
                        lambda: tf.reduce_sum(masked_loss) / num_pixels,
                        lambda: 0.0
                    )
                    return self.class_weights[class_idx] * class_mean

                class_losses = tf.map_fn(
                    compute_class_loss, tf.range(num_classes), dtype=tf.float32
                )
                return tf.reduce_sum(class_losses)

            def get_config(self):
                config = super().get_config()
                config.update({
                    "class_weights": self.class_weights.numpy().tolist(),
                    "from_logits": self.from_logits,
                })
                return config

        # ----- NaNTolerantCallback: DSAI callback but skip NaN termination -----
        class NaNTolerantCallback(BatchAccuracyCallback):
            """BatchAccuracyCallback that logs NaN validation loss but does NOT
            terminate training. Needed for batch_size=1 where validation loss
            can be NaN while model predictions remain valid."""
            def run_validation(self):
                val_loss_total = 0
                val_accuracy_total = 0
                num_batches = 0
                try:
                    for x_val, y_val in self.val_data:
                        y_val = tf.cast(y_val, dtype=tf.int32)
                        val_logits = self._model(x_val, training=False)
                        num_classes = val_logits.shape[-1]
                        val_logits_flat = tf.reshape(val_logits, [-1, num_classes])
                        y_val_flat = tf.reshape(y_val, [-1])
                        predictions = tf.cast(
                            tf.argmax(val_logits_flat, axis=1), tf.int32)
                        val_loss = self.loss_function(y_val_flat, val_logits_flat)
                        val_accuracy = tf.reduce_mean(
                            tf.cast(tf.equal(predictions, y_val_flat), tf.float32))
                        loss_val = tf.reduce_mean(val_loss).numpy()
                        if not (np.isnan(loss_val) or np.isinf(loss_val)):
                            val_loss_total += loss_val
                        val_accuracy_total += val_accuracy.numpy()
                        num_batches += 1
                    val_loss_avg = val_loss_total / max(num_batches, 1)
                    val_accuracy_avg = val_accuracy_total / max(num_batches, 1)
                    # Log NaN but do NOT stop training
                    if np.isnan(val_loss_avg) or np.isinf(val_loss_avg):
                        print(f"  [WARNING] Validation loss is NaN/Inf — continuing training")
                        val_loss_avg = 999.0  # Replace NaN for tracking
                    self.validation_losses.append(val_loss_avg)
                    self.validation_accuracies.append(val_accuracy_avg)
                    if self.logger:
                        self.logger.log_validation_metrics(
                            val_logits, y_val,
                            loss=val_loss_avg, accuracy=val_accuracy_avg)
                    if self.early_stopping:
                        current = (val_loss_avg if self.monitor == 'val_loss'
                                   else val_accuracy_avg)
                        if current is None:
                            return
                        if self.monitor_op(current, self.best):
                            self.best = current
                            self.wait = 0
                            if self.save_best_model:
                                tf.keras.models.save_model(
                                    self._model, self.save_path,
                                    save_format='tf')
                        else:
                            self.wait += 1
                            if self.wait >= self.es_patience:
                                self.stopped_epoch = self.current_epoch
                                self._model.stop_training = True
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(f"Validation failed: {e}")
                    raise

        # ----- PatienceBasedLRCallback: DSAI validation + main LR schedule -----
        class PatienceBasedLRCallback(NaNTolerantCallback):
            """BatchAccuracyCallback with main's counter-based LR schedule.

            Overrides on_epoch_end only. Validation mechanism stays DSAI
            (global iteration-based).
            """
            def on_epoch_end(self, epoch, logs=None):
                self.epoch_wait += 1
                if self.epoch_wait > self.lr_patience and self.reduce_lr:
                    old_lr = float(self._model.optimizer.learning_rate.numpy())
                    new_lr = old_lr * self.lr_factor
                    self._model.optimizer.learning_rate.assign(new_lr)
                    if self.verbose > 0 and self.logger:
                        self.logger.logger.info(
                            f"\nEpoch {epoch + 1}: Reducing learning rate "
                            f"from {old_lr:.6f} to {new_lr:.6f}"
                        )
                    self.epoch_wait = 0

        # ----- PerEpochValidationCallback: main validation + DSAI LR -----
        class PerEpochValidationCallback(keras.callbacks.Callback):
            """Main's per-epoch validation (3/epoch via linspace) with
            DSAI's unconditional LR reduction every epoch."""

            def __init__(self, model, val_data, loss_function, logger=None,
                         num_validations=3, early_stopping=True,
                         reduce_lr_on_plateau=True, monitor='val_accuracy',
                         es_patience=6, lr_patience=1, lr_factor=0.75,
                         verbose=0, save_best_model=True,
                         filepath='best_model.h5'):
                super().__init__()
                self.logger = logger
                self._model = model
                self.loss_function = loss_function
                self.num_validations = num_validations
                self.val_data = val_data

                # Metrics tracking
                self.batch_accuracies = []
                self.batch_numbers = []
                self.batch_losses = []
                self.epoch_indices = []
                self.current_epoch = 0
                self.validation_losses = []
                self.validation_accuracies = []
                self.val_indices = []
                self.validation_steps = []
                self.validation_counter = 0
                self.current_step = 0

                # Early stopping
                self.early_stopping = early_stopping
                self.reduce_lr = reduce_lr_on_plateau
                self.monitor = monitor
                self.es_patience = es_patience
                self.lr_patience = lr_patience
                self.lr_factor = lr_factor
                self.verbose = verbose
                self.mode = 'min' if monitor == 'val_loss' else 'max'
                self.save_best_model = save_best_model
                self.save_path = filepath
                self.monitor_op = np.less if self.mode == 'min' else np.greater
                self.best = np.Inf if self.mode == 'min' else -np.Inf
                self.wait = 0
                self.stopped_epoch = 0

            @property
            def model(self):
                return self._model

            @model.setter
            def model(self, value):
                self._model = value

            def on_epoch_begin(self, epoch, logs=None):
                """Main's on_epoch_begin: linspace validation steps, reset per epoch."""
                self.current_epoch = epoch
                self.epoch_indices.append(epoch)
                self.validation_steps = np.linspace(
                    0, self.params['steps'],
                    self.num_validations + 1, dtype=int
                )[1:]
                self.validation_counter = 0
                self.current_step = 0

            def on_batch_end(self, batch, logs=None):
                """Main's on_batch_end: validate at linspace steps."""
                logs = logs or {}
                self.current_step += 1
                if self.current_step in self.validation_steps:
                    self.run_validation()
                    self.val_indices.append(
                        self.current_step
                        + self.params['steps'] * self.current_epoch
                    )
                accuracy = logs.get('accuracy')
                if accuracy is not None:
                    self.batch_accuracies.append(accuracy)
                    self.batch_numbers.append(
                        self.params['steps'] * self.current_epoch + batch + 1
                    )
                loss = logs.get('loss')
                if loss is not None:
                    self.batch_losses.append(loss)

            def on_epoch_end(self, epoch, logs=None):
                """DSAI's on_epoch_end: unconditional LR reduction every epoch."""
                old_lr = float(self._model.optimizer.learning_rate.numpy())
                new_lr = old_lr * self.lr_factor
                self._model.optimizer.learning_rate.assign(new_lr)
                if self.verbose > 0 and self.logger:
                    self.logger.logger.debug(
                        f"\nEpoch {epoch + 1}: Reducing LR "
                        f"from {old_lr:.6f} to {new_lr:.6f}"
                    )

            def run_validation(self):
                """NaN-tolerant validation logic."""
                val_loss_total = 0
                val_accuracy_total = 0
                num_batches = 0
                try:
                    for x_val, y_val in self.val_data:
                        y_val = tf.cast(y_val, dtype=tf.int32)
                        val_logits = self._model(x_val, training=False)
                        num_classes = val_logits.shape[-1]
                        val_logits_flat = tf.reshape(val_logits, [-1, num_classes])
                        y_val_flat = tf.reshape(y_val, [-1])
                        predictions = tf.cast(
                            tf.argmax(val_logits_flat, axis=1), tf.int32)
                        val_loss = self.loss_function(y_val_flat, val_logits_flat)
                        val_accuracy = tf.reduce_mean(
                            tf.cast(tf.equal(predictions, y_val_flat), tf.float32))
                        loss_val = tf.reduce_mean(val_loss).numpy()
                        if not (np.isnan(loss_val) or np.isinf(loss_val)):
                            val_loss_total += loss_val
                        val_accuracy_total += val_accuracy.numpy()
                        num_batches += 1
                    val_loss_avg = val_loss_total / max(num_batches, 1)
                    val_accuracy_avg = val_accuracy_total / max(num_batches, 1)
                    if np.isnan(val_loss_avg) or np.isinf(val_loss_avg):
                        print(f"  [WARNING] Validation loss NaN — continuing")
                        val_loss_avg = 999.0
                    self.validation_losses.append(val_loss_avg)
                    self.validation_accuracies.append(val_accuracy_avg)
                    if self.early_stopping:
                        current = (val_loss_avg if self.monitor == 'val_loss'
                                   else val_accuracy_avg)
                        if current is None:
                            return
                        if self.monitor_op(current, self.best):
                            self.best = current
                            self.wait = 0
                            if self.save_best_model:
                                tf.keras.models.save_model(
                                    self._model, self.save_path,
                                    save_format='tf')
                        else:
                            self.wait += 1
                            if self.wait >= self.es_patience:
                                self.stopped_epoch = self.current_epoch
                                self._model.stop_training = True
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(f"Validation failed: {e}")
                    raise

        # ----- MainStyleFullCallback: main validation + main LR -----
        class MainStyleFullCallback(PerEpochValidationCallback):
            """Full main-branch callback: linspace validation + counter LR."""

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.epoch_wait = 0
                self.original_lr_patience = self.lr_patience

            def on_epoch_end(self, epoch, logs=None):
                """Main's on_epoch_end: counter-based periodic LR reduction."""
                self.epoch_wait += 1
                if self.epoch_wait > self.lr_patience and self.reduce_lr:
                    old_lr = float(self._model.optimizer.learning_rate.numpy())
                    new_lr = old_lr * self.lr_factor
                    self._model.optimizer.learning_rate.assign(new_lr)
                    if self.verbose > 0 and self.logger:
                        self.logger.logger.info(
                            f"\nEpoch {epoch + 1}: Reducing LR "
                            f"from {old_lr:.6f} to {new_lr:.6f}"
                        )
                    self.epoch_wait = 0

        # ================================================================
        # Trainer subclasses
        # ================================================================

        class SafeBaselineTrainer(DeepLabV3PlusTrainer):
            """DSAI baseline trainer with NaN-tolerant validation callback."""
            def _create_callbacks(self, model):
                if not self.logger:
                    log_dir = os.path.join(self.model_path, 'logs')
                    model_name = self.model_data.get('nm', 'unknown_model')
                    self.logger = Logger(log_dir=log_dir, model_name=model_name)
                    self.logger.log_system_info()
                best_model_path = os.path.join(
                    self.model_path, f"best_model_{self.model_type}.keras")
                callback = NaNTolerantCallback(
                    model=model,
                    val_data=self.val_dataset,
                    loss_function=self.loss_function,
                    logger=self.logger,
                    validation_frequency=self.validation_frequency,
                    early_stopping=True,
                    reduce_lr_on_plateau=True,
                    monitor='val_accuracy',
                    es_patience=self.es_patience,
                    lr_patience=self.lr_patience,
                    lr_factor=self.lr_factor,
                    verbose=1,
                    save_best_model=True,
                    filepath=best_model_path,
                )
                return [callback, keras.callbacks.TerminateOnNaN()]

        class MainStyleLossTrainer(SafeBaselineTrainer):
            """Use main's per-pixel weighted loss."""
            def _create_loss_function(self):
                tile_format = self.model_data.get('tile_format', 'tif')
                train_masks = sorted(glob(os.path.join(
                    self.model_path, 'training', 'label', f'*.{tile_format}')))
                self.class_weights = self._calculate_class_weights(train_masks)
                self.loss_function = MainStyleLoss(
                    class_weights=self.class_weights,
                    from_logits=True,
                    reduction='sum_over_batch_size'
                )
                print(f"  [MainStyleLossTrainer] Using per-pixel weighted loss "
                      f"with raw weights (shape={self.class_weights.shape})")

        class MainStyleLRScheduleTrainer(DeepLabV3PlusTrainer):
            """Use main's counter-based LR schedule."""
            def _create_callbacks(self, model):
                if not self.logger:
                    log_dir = os.path.join(self.model_path, 'logs')
                    model_name = self.model_data.get('nm', 'unknown_model')
                    self.logger = Logger(log_dir=log_dir, model_name=model_name)
                    self.logger.log_system_info()
                best_model_path = os.path.join(
                    self.model_path, f"best_model_{self.model_type}.keras")
                callback = PatienceBasedLRCallback(
                    model=model,
                    val_data=self.val_dataset,
                    loss_function=self.loss_function,
                    logger=self.logger,
                    validation_frequency=self.validation_frequency,
                    early_stopping=True,
                    reduce_lr_on_plateau=True,
                    monitor='val_accuracy',
                    es_patience=self.es_patience,
                    lr_patience=self.lr_patience,
                    lr_factor=self.lr_factor,
                    verbose=1,
                    save_best_model=True,
                    filepath=best_model_path,
                )
                print(f"  [MainStyleLRScheduleTrainer] Using counter-based LR "
                      f"(reduce every {self.lr_patience + 1} epochs)")
                return [callback, keras.callbacks.TerminateOnNaN()]

        class MainStyleValidationTrainer(DeepLabV3PlusTrainer):
            """Use main's 3-per-epoch linspace validation."""
            def _create_callbacks(self, model):
                if not self.logger:
                    log_dir = os.path.join(self.model_path, 'logs')
                    model_name = self.model_data.get('nm', 'unknown_model')
                    self.logger = Logger(log_dir=log_dir, model_name=model_name)
                    self.logger.log_system_info()
                best_model_path = os.path.join(
                    self.model_path, f"best_model_{self.model_type}.keras")
                callback = PerEpochValidationCallback(
                    model=model,
                    val_data=self.val_dataset,
                    loss_function=self.loss_function,
                    logger=self.logger,
                    num_validations=3,
                    early_stopping=True,
                    reduce_lr_on_plateau=True,
                    monitor='val_accuracy',
                    es_patience=self.es_patience,
                    lr_patience=self.lr_patience,
                    lr_factor=self.lr_factor,
                    verbose=1,
                    save_best_model=True,
                    filepath=best_model_path,
                )
                print(f"  [MainStyleValidationTrainer] Using 3-per-epoch "
                      f"linspace validation")
                return [callback, keras.callbacks.TerminateOnNaN()]

        class RawClassWeightsTrainer(SafeBaselineTrainer):
            """Use DSAI per-class mean loss but with raw (un-normalized) weights."""
            def _create_loss_function(self):
                tile_format = self.model_data.get('tile_format', 'tif')
                train_masks = sorted(glob(os.path.join(
                    self.model_path, 'training', 'label', f'*.{tile_format}')))
                self.class_weights = self._calculate_class_weights(train_masks)
                self.loss_function = RawWeightLoss(
                    class_weights=self.class_weights,
                    from_logits=True,
                    reduction='sum_over_batch_size'
                )
                print(f"  [RawClassWeightsTrainer] Using per-class mean loss "
                      f"with RAW weights (no normalization)")

        class MainStyleFullTrainer(DeepLabV3PlusTrainer):
            """Reproduce all main-branch training behaviors."""

            def _prepare_data(self, seed=None):
                """Use main's create_dataset() without shuffle."""
                train_path = self.model_paths['train_data']
                val_path = self.model_paths['val_data']
                tile_format = self.model_data.get('tile_format', 'tif')
                file_pattern = f"*.{tile_format}"

                train_images = sorted(glob(os.path.join(
                    train_path, 'im', file_pattern)))
                train_masks = sorted(glob(os.path.join(
                    train_path, 'label', file_pattern)))
                val_images = sorted(glob(os.path.join(
                    val_path, 'im', file_pattern)))
                val_masks = sorted(glob(os.path.join(
                    val_path, 'label', file_pattern)))

                if not train_images or not train_masks:
                    raise ValueError("No training images or masks found")
                if not val_images or not val_masks:
                    raise ValueError("No validation images or masks found")

                # Main uses create_dataset() — no shuffle, no explicit prefetch
                self.train_dataset = create_dataset(
                    train_images, train_masks,
                    self.image_size, self.batch_size)
                self.val_dataset = create_dataset(
                    val_images, val_masks,
                    self.image_size, self.batch_size)

                self.num_train_samples = len(train_images)
                self.num_val_samples = len(val_images)
                self.train_steps_per_epoch = (
                    self.num_train_samples // self.batch_size)
                self.val_steps_per_epoch = (
                    self.num_val_samples // self.batch_size)

                if self.logger:
                    self.logger.log_dataset_info(
                        self.train_dataset, "Training")
                    self.logger.log_dataset_info(
                        self.val_dataset, "Validation")

                print(f"  [MainStyleFullTrainer] Using create_dataset() "
                      f"(no shuffle), {self.num_train_samples} train, "
                      f"{self.num_val_samples} val")

            def _create_loss_function(self):
                """Use main's per-pixel weighted loss with raw weights."""
                tile_format = self.model_data.get('tile_format', 'tif')
                train_masks = sorted(glob(os.path.join(
                    self.model_path, 'training', 'label', f'*.{tile_format}')))
                self.class_weights = self._calculate_class_weights(train_masks)
                self.loss_function = MainStyleLoss(
                    class_weights=self.class_weights,
                    from_logits=True,
                    reduction='sum_over_batch_size'
                )

            def _create_callbacks(self, model):
                """Use main's linspace validation + counter LR schedule."""
                if not self.logger:
                    log_dir = os.path.join(self.model_path, 'logs')
                    model_name = self.model_data.get('nm', 'unknown_model')
                    self.logger = Logger(
                        log_dir=log_dir, model_name=model_name)
                    self.logger.log_system_info()
                best_model_path = os.path.join(
                    self.model_path,
                    f"best_model_{self.model_type}.keras")
                callback = MainStyleFullCallback(
                    model=model,
                    val_data=self.val_dataset,
                    loss_function=self.loss_function,
                    logger=self.logger,
                    num_validations=3,
                    early_stopping=True,
                    reduce_lr_on_plateau=True,
                    monitor='val_accuracy',
                    es_patience=self.es_patience,
                    lr_patience=1,
                    lr_factor=0.75,
                    verbose=1,
                    save_best_model=True,
                    filepath=best_model_path,
                )
                return [callback, keras.callbacks.TerminateOnNaN()]

            def _compile_model(self, model):
                """Use main's Adam(lr=0.0005) with default epsilon."""
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                    loss=self.loss_function,
                    metrics=["accuracy"],
                )
                print("  [MainStyleFullTrainer] Compiled with Adam "
                      "lr=0.0005, default epsilon")

            def _train_model(self, model, callbacks):
                """Main's model.fit() — no steps_per_epoch."""
                epochs = self.epochs
                print(f"  [MainStyleFullTrainer] Starting training "
                      f"(no steps_per_epoch, epochs={epochs})")
                history = model.fit(
                    self.train_dataset,
                    validation_data=self.val_dataset,
                    callbacks=callbacks,
                    verbose=1,
                    epochs=epochs,
                )
                return history

        # ================================================================
        # Trainer dispatch
        # ================================================================

        TRAINER_MAP = {
            "DeepLabV3PlusTrainer": DeepLabV3PlusTrainer,
            "SafeBaselineTrainer": SafeBaselineTrainer,
            "MainStyleLossTrainer": MainStyleLossTrainer,
            "MainStyleLRScheduleTrainer": MainStyleLRScheduleTrainer,
            "MainStyleValidationTrainer": MainStyleValidationTrainer,
            "RawClassWeightsTrainer": RawClassWeightsTrainer,
            "MainStyleFullTrainer": MainStyleFullTrainer,
        }

        TrainerClass = TRAINER_MAP.get(trainer_cls_name)
        if TrainerClass is None:
            raise ValueError(f"Unknown trainer class: {trainer_cls_name}")

        print(f"[worker:{experiment_name}] Using trainer: {trainer_cls_name}")

        # ----- Train -----
        trainer = TrainerClass(model_dir)
        t0 = time.time()
        trainer.train(seed=42)
        result["train_time_s"] = round(time.time() - t0, 1)
        print(f"[worker:{experiment_name}] Training finished in "
              f"{result['train_time_s']}s")

        # ----- Repair NaN BatchNorm statistics and save final model -----
        # batch_size=1 corrupts BatchNorm moving_variance (single-sample
        # variance is degenerate). Load the final model, fix NaN stats,
        # and re-save as best_model so classification uses valid weights.
        final_model_path = os.path.join(model_dir, "DeepLabV3_plus.keras")
        best_model_path = os.path.join(
            model_dir, "best_model_DeepLabV3_plus.keras")
        if os.path.isfile(final_model_path):
            repair_model = tf.keras.models.load_model(
                final_model_path, compile=False)
            nan_fixed = 0
            for layer in repair_model.layers:
                if hasattr(layer, 'moving_variance'):
                    mv = layer.moving_variance.numpy()
                    if np.isnan(mv).any():
                        layer.moving_variance.assign(
                            np.where(np.isnan(mv), 1.0, mv))
                        nan_fixed += 1
                if hasattr(layer, 'moving_mean'):
                    mm = layer.moving_mean.numpy()
                    if np.isnan(mm).any():
                        layer.moving_mean.assign(
                            np.where(np.isnan(mm), 0.0, mm))
                        nan_fixed += 1
            repair_model.save(best_model_path)
            print(f"[worker:{experiment_name}] Saved repaired model "
                  f"(fixed {nan_fixed} NaN BN layers) -> best_model")

        # ----- Extract training info -----
        with open(os.path.join(model_dir, "net.pkl"), "rb") as f:
            meta = pickle.load(f)
        history = meta.get("history", {})
        if history:
            train_loss = history.get("loss", [])
            result["epochs_completed"] = len(train_loss)

        # ----- Test -----
        pthtest = meta.get("pthtest", "")
        res_subdir = meta.get("resolution_subdir") or "10x"
        pthtestim = os.path.join(pthtest, res_subdir) if pthtest else ""

        if pthtest and os.path.isdir(pthtest):
            print(f"[worker:{experiment_name}] Testing")
            t0 = time.time()
            classification_output_dir = os.path.join(model_dir, "classification_output")
            metrics = test_segmentation_model(
                model_dir, pthtest, pthtestim, show_fig=False,
                classification_output_dir=classification_output_dir)
            result["test_time_s"] = round(time.time() - t0, 1)
            if metrics:
                result["test_metrics"] = _serialise(metrics)
                # Extract overall accuracy from confusion matrix
                cm = metrics.get("confusion_matrix")
                if cm is not None:
                    cm_arr = np.array(cm)
                    total = cm_arr.sum()
                    correct = np.trace(cm_arr)
                    if total > 0:
                        result["overall_accuracy"] = round(
                            100.0 * correct / total, 2)
            print(f"[worker:{experiment_name}] Testing finished in "
                  f"{result['test_time_s']}s, accuracy="
                  f"{result.get('overall_accuracy', 'N/A')}%")
        else:
            print(f"[worker:{experiment_name}] Skipping test (no test path)")

        result["success"] = True

    except Exception as exc:
        import traceback
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"[worker:{experiment_name}] FAILED: {result['error']}")

    # Write results
    out_path = os.path.join(model_dir, "worker_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[worker:{experiment_name}] Results written to {out_path}")


# ===================================================================
# ORCHESTRATOR MODE
# ===================================================================

def _orchestrate(args):
    """Run all experiments sequentially on one GPU."""
    dataset = args.dataset
    gpu = args.gpu
    cfg = DATASET_CONFIGS[dataset]
    data_path = cfg["default_data_path"]

    # GPU safety check
    check_gpu_safety(gpu)

    # Determine which experiments to run
    experiments = args.experiments if args.experiments else EXPERIMENT_NAMES
    for name in experiments:
        if name not in EXPERIMENT_NAMES:
            print(f"ERROR: Unknown experiment '{name}'. "
                  f"Available: {EXPERIMENT_NAMES}")
            sys.exit(1)

    # Find tile source
    print(f"\nLooking for existing tiles for {dataset}...")
    tile_source = find_tile_source(data_path, dataset)
    if not tile_source:
        print("No existing tiles found. Creating new tiles...")
        tile_source = os.path.join(
            data_path, "regression_ablation_results", "shared_tiles")
        create_tiles(tile_source, data_path, 1024, dataset)
        if not _tiles_exist(tile_source):
            print("ERROR: Tile creation failed.")
            sys.exit(1)

    # Results directory
    results_dir = os.path.join(data_path, "regression_ablation_results")
    os.makedirs(results_dir, exist_ok=True)

    script_path = os.path.abspath(__file__)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print(f"Regression Ablation Experiment — {timestamp}")
    print(f"  Dataset:      {dataset}")
    print(f"  Data path:    {data_path}")
    print(f"  Tile source:  {tile_source}")
    print(f"  Output:       {results_dir}")
    batch_size = args.batch_size
    print(f"  GPU:          {gpu}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Tile size:    {args.tile_size} (model input, tiles resized during loading)")
    print(f"  Experiments:  {experiments}")
    print("=" * 70)

    # Run each experiment
    all_results = {}

    for exp_name in experiments:
        exp_dir = os.path.join(results_dir, exp_name)

        # Look up description
        desc = ""
        for name, d, _, _ in EXPERIMENTS:
            if name == exp_name:
                desc = d
                break

        print(f"\n{'—' * 70}")
        print(f"[orchestrator] Experiment: {exp_name} — {desc}")
        print(f"{'—' * 70}")

        # Set up experiment directory
        setup_experiment_dir(exp_dir, tile_source)

        # Launch worker subprocess
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # Prevent cuDNN autotune from corrupting CUDA driver on memory-tight GPUs
        env["TF_CUDNN_USE_AUTOTUNE"] = "0"
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        env["PYTHONUNBUFFERED"] = "1"

        tile_size = args.tile_size
        cmd = [
            sys.executable, script_path,
            "--worker",
            "--experiment", exp_name,
            "--model-dir", exp_dir,
            "--dataset", dataset,
            "--batch-size", str(batch_size),
            "--tile-size", str(tile_size),
        ]
        print(f"  CMD: {' '.join(cmd)}")
        print(f"  CUDA_VISIBLE_DEVICES={gpu}")

        stdout_log = os.path.join(exp_dir, "worker_stdout.log")
        stderr_log = os.path.join(exp_dir, "worker_stderr.log")

        t0 = time.time()
        try:
            with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
                proc = subprocess.run(
                    cmd, env=env, stdout=out, stderr=err,
                    timeout=SUBPROCESS_TIMEOUT)
            wall_time = round(time.time() - t0, 1)

            results_file = os.path.join(exp_dir, "worker_results.json")
            if os.path.isfile(results_file):
                with open(results_file) as f:
                    all_results[exp_name] = json.load(f)
                all_results[exp_name]["wall_time_s"] = wall_time
                all_results[exp_name]["exit_code"] = proc.returncode
            else:
                all_results[exp_name] = {
                    "success": False,
                    "error": f"No results file (exit code {proc.returncode})",
                    "wall_time_s": wall_time,
                    "exit_code": proc.returncode,
                }
        except subprocess.TimeoutExpired:
            wall_time = round(time.time() - t0, 1)
            all_results[exp_name] = {
                "success": False,
                "error": f"Timed out after {SUBPROCESS_TIMEOUT}s",
                "wall_time_s": wall_time,
                "exit_code": -1,
            }
            print(f"[orchestrator] {exp_name} TIMED OUT")

        status = "OK" if all_results[exp_name].get("success") else "FAILED"
        acc = all_results[exp_name].get("overall_accuracy", "N/A")
        print(f"[orchestrator] {exp_name}: {status} — accuracy={acc}% "
              f"(wall {all_results[exp_name]['wall_time_s']}s)")

    # Save combined results
    json_path = os.path.join(results_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate CSV
    csv_path = os.path.join(results_dir, "ablation_comparison.csv")
    _generate_csv(csv_path, all_results, dataset)

    # Print summary
    _print_summary(all_results, experiments, dataset, timestamp)

    print(f"\nAll outputs in: {results_dir}")


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def _generate_csv(csv_path: str, all_results: dict, dataset: str) -> None:
    """Write results to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_name", "dataset", "framework", "accuracy",
            "epochs_completed", "training_time"
        ])
        for name, res in all_results.items():
            writer.writerow([
                name,
                dataset,
                res.get("framework", "tensorflow"),
                res.get("overall_accuracy", ""),
                res.get("epochs_completed", ""),
                res.get("train_time_s", ""),
            ])
    print(f"\nCSV saved: {csv_path}")


def _print_summary(all_results: dict, experiments: list,
                   dataset: str, timestamp: str) -> None:
    """Print a comparison summary table."""
    print(f"\n{'=' * 80}")
    print(f"REGRESSION ABLATION RESULTS — {timestamp}")
    print(f"Dataset: {dataset} | Framework: TensorFlow")
    print(f"{'=' * 80}")

    # Get baseline accuracy
    baseline_acc = None
    if "dsai_baseline" in all_results:
        baseline_acc = all_results["dsai_baseline"].get("overall_accuracy")

    main_all_acc = None
    if "main_all" in all_results:
        main_all_acc = all_results["main_all"].get("overall_accuracy")

    header = (f"{'Experiment':<20s} {'Description':<38s} "
              f"{'Accuracy':>8s} {'Delta':>8s} {'Time(s)':>8s}")
    print(header)
    print("-" * len(header))

    for exp_name in experiments:
        res = all_results.get(exp_name, {})

        # Look up description
        desc = ""
        for name, d, _, _ in EXPERIMENTS:
            if name == exp_name:
                desc = d
                break

        acc = res.get("overall_accuracy")
        train_time = res.get("train_time_s", "")

        if acc is not None and baseline_acc is not None:
            delta = acc - baseline_acc
            delta_str = f"{delta:+.1f}%" if exp_name != "dsai_baseline" else "--"
        elif acc is not None:
            delta_str = "--"
        else:
            delta_str = "ERR"

        acc_str = f"{acc:.1f}%" if acc is not None else "FAILED"
        time_str = f"{train_time}" if train_time else ""

        print(f"{exp_name:<20s} {desc:<38s} {acc_str:>8s} "
              f"{delta_str:>8s} {time_str:>8s}")

    # Root cause analysis
    if baseline_acc is not None and main_all_acc is not None:
        gap = main_all_acc - baseline_acc
        print(f"\nTotal gap (main_all - dsai_baseline): {gap:+.1f}%")

        # Find largest individual contributor
        best_exp = None
        best_delta = 0
        for exp_name in experiments:
            if exp_name in ("dsai_baseline", "main_all"):
                continue
            res = all_results.get(exp_name, {})
            acc = res.get("overall_accuracy")
            if acc is not None:
                delta = acc - baseline_acc
                if delta > best_delta:
                    best_delta = delta
                    best_exp = exp_name

        if best_exp:
            pct_explained = (best_delta / gap * 100) if gap > 0 else 0
            print(f"Largest individual contributor: {best_exp} "
                  f"(+{best_delta:.1f}%, explains {pct_explained:.0f}% of gap)")

    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Regression ablation: DSAI vs main branch")

    parser.add_argument("--dataset", choices=["liver", "lungs"],
                        help="Dataset to use")
    parser.add_argument("--gpu", type=str,
                        help="GPU index (must be a P4000)")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Batch size (default: 3)")
    parser.add_argument("--tile-size", type=int, default=1024,
                        help="Tile size for model input (default: 1024)")
    parser.add_argument("--experiments", nargs="*",
                        help="Subset of experiments to run "
                             f"(default: all). Options: {EXPERIMENT_NAMES}")

    # Worker-mode arguments (internal)
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--experiment", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=str,
                        help=argparse.SUPPRESS)
    # batch-size is defined above as a regular arg

    args = parser.parse_args()

    if args.worker:
        # Worker mode — run single experiment
        if not args.experiment or not args.model_dir or not args.dataset:
            parser.error("Worker mode requires --experiment, --model-dir, "
                         "and --dataset")
        _run_worker(args)
    else:
        # Orchestrator mode
        if not args.dataset or not args.gpu:
            parser.error("Orchestrator mode requires --dataset and --gpu")
        _orchestrate(args)


if __name__ == "__main__":
    main()
