"""
Cross-Version Comparison: Full workflow across code versions, frameworks, and tile modes.

Runs tiles -> train -> test -> confusion matrix for each combination of:
  - Code version: DSAI, main, 62614aa, 9a88b70
  - Dataset: lungs, liver
  - Framework: pytorch, tensorflow (DSAI only has both)
  - Tile mode: modern, legacy (DSAI only), default (others)

Each experiment produces a confusion matrix (same as GUI), a detailed JSON
results file, and a human-readable report.  The orchestrator collects all
confusion matrices into one directory and generates a cross-experiment
summary.

Usage:
    python scripts/run_version_comparison.py --gpu 0 --batch-size 1
    python scripts/run_version_comparison.py --gpu 0 --batch-size 1 --datasets lungs --versions dsai
    python scripts/run_version_comparison.py --gpu 0 --batch-size 1 --versions dsai --tile-modes modern --frameworks tensorflow
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
import traceback as tb_module
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project root (used by orchestrator only; worker uses worktree)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SUBPROCESS_TIMEOUT = 12 * 60 * 60  # 12 hours per experiment

# ---------------------------------------------------------------------------
# Dataset configurations (shared with ablation script)
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "liver": {
        "WS": [[0, 0, 0, 0, 2, 0, 2], [7, 6], [1, 2, 3, 4, 5, 6, 7],
               [6, 4, 2, 3, 5, 1, 7], []],
        "CMAP": np.array([
            [230, 190, 100], [65, 155, 210], [145, 35, 35],
            [158, 24, 118], [30, 50, 50], [235, 188, 215],
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
        "WS": [[0, 2, 0, 0, 2, 0], [5, 6], [1, 2, 3, 4, 5, 6],
               [6, 2, 4, 3, 1, 5], []],
        "CMAP": np.array([
            [128, 0, 255], [166, 193, 202], [255, 0, 0],
            [128, 64, 0], [255, 255, 255], [255, 128, 192],
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

# ---------------------------------------------------------------------------
# Version specifications
# ---------------------------------------------------------------------------
VERSION_SPECS = {
    "dsai": {
        "ref": "DSAI",
        "has_pytorch": True,
        "has_tile_modes": True,
        "frameworks": ["pytorch", "tensorflow"],
        "tile_modes": ["modern", "legacy"],
    },
    "main": {
        "ref": "main",
        "has_pytorch": False,
        "has_tile_modes": False,
        "frameworks": ["tensorflow"],
        "tile_modes": ["default"],
    },
    "62614aa": {
        "ref": "62614aad2ff2aab89b65a482926a9c6ceadba3b4",
        "has_pytorch": False,
        "has_tile_modes": False,
        "frameworks": ["tensorflow"],
        "tile_modes": ["default"],
    },
    "9a88b70": {
        "ref": "9a88b701ca7640a6d963e373159c308d6e66c822",
        "has_pytorch": False,
        "has_tile_modes": False,
        "frameworks": ["tensorflow"],
        "tile_modes": ["default"],
    },
}


# ===================================================================
# Utility helpers
# ===================================================================

def _serialise(obj):
    """Recursively convert numpy types for JSON serialisation."""
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


def _exp_name(version: str, dataset: str, framework: str,
              tile_mode: str) -> str:
    """Build a canonical experiment name."""
    return f"{version}__{dataset}__{framework}__{tile_mode}"


def _get_worktree_path(version: str) -> str:
    """Return the worktree path for a given version."""
    if version == "dsai":
        return _PROJECT_ROOT
    base = os.path.dirname(_PROJECT_ROOT)
    return os.path.join(base, f"CODAvision_vc_{version}")


def _get_commit_hash(worktree: str) -> str:
    """Get the short commit hash for a worktree."""
    try:
        result = subprocess.run(
            ["git", "-C", worktree, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ===================================================================
# Git worktree management
# ===================================================================

def _setup_worktrees(versions: List[str]) -> Dict[str, str]:
    """Create git worktrees for non-DSAI versions. Returns {version: path}."""
    paths = {}
    for v in versions:
        wt = _get_worktree_path(v)
        paths[v] = wt
        if v == "dsai":
            continue
        if os.path.isdir(wt):
            print(f"  Worktree exists: {wt}")
            continue
        ref = VERSION_SPECS[v]["ref"]
        print(f"  Creating worktree for {v} ({ref}) at {wt}")
        subprocess.run(
            ["git", "-C", _PROJECT_ROOT, "worktree", "add", "--detach",
             wt, ref],
            check=True, timeout=60)
    return paths


# ===================================================================
# Experiment matrix
# ===================================================================

def _build_experiment_list(
    versions: List[str],
    datasets: List[str],
    frameworks: Optional[List[str]],
    tile_modes: Optional[List[str]],
) -> List[Dict[str, str]]:
    """Generate the full list of experiments to run."""
    experiments = []
    for v in versions:
        spec = VERSION_SPECS[v]
        v_frameworks = spec["frameworks"]
        v_tile_modes = spec["tile_modes"]
        # Apply user filters
        if frameworks:
            v_frameworks = [f for f in v_frameworks if f in frameworks]
        if tile_modes:
            v_tile_modes = [m for m in v_tile_modes if m in tile_modes]
        for ds in datasets:
            for tm in v_tile_modes:
                for fw in v_frameworks:
                    experiments.append({
                        "version": v,
                        "dataset": ds,
                        "framework": fw,
                        "tile_mode": tm,
                        "name": _exp_name(v, ds, fw, tm),
                    })
    return experiments


def _tile_group_key(exp: Dict[str, str]) -> str:
    """Key for tile sharing: same version + dataset + tile_mode share tiles."""
    return f"{exp['version']}__{exp['dataset']}__{exp['tile_mode']}"


# ===================================================================
# WORKER MODE — runs in subprocess
# ===================================================================

def _run_worker(args):
    """Run a single experiment: create metadata, tiles, train, test."""
    # Elevate process priority on Windows so background/minimized workers
    # bypass Windows 11's aggressive background throttling (which otherwise
    # caps the process at ~5% CPU and turns a 10-minute tile creation into
    # a 2-hour wait). HIGH_PRIORITY_CLASS is chosen over REALTIME because
    # REALTIME can starve the OS.
    try:
        import psutil
        proc = psutil.Process()
        if hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
            print(f"[worker] Process priority set to HIGH (psutil)")
        else:
            proc.nice(-5)  # POSIX fallback
            print(f"[worker] Process priority lowered (POSIX nice -5)")
    except Exception as exc:
        print(f"[worker] Could not elevate process priority: {exc}")

    # Force non-interactive matplotlib BEFORE any plotting imports
    import matplotlib
    matplotlib.use('Agg')

    exp_dir = args.experiment_dir
    worktree = args.worktree
    dataset = args.dataset
    data_path = args.data_path
    framework = args.framework
    tile_mode = args.tile_mode
    batch_size = args.batch_size
    tile_size = args.tile_size
    version_name = args.version_name
    commit = args.commit
    exp_name = args.exp_name
    create_tiles = args.create_tiles

    cfg = DATASET_CONFIGS[dataset]
    pthim = os.path.join(data_path, cfg["resolution_subdir"])
    pthtest = os.path.join(data_path, cfg["test_subdir"])
    pthtestim = os.path.join(pthtest, cfg["resolution_subdir"])

    # Determine tile format
    if tile_mode == "modern":
        tile_format = "png"
    elif tile_mode == "legacy":
        tile_format = "tif"
    else:
        tile_format = "tif"  # old versions always use tif

    result = {
        "experiment_name": exp_name,
        "code_version": version_name,
        "commit": commit,
        "dataset": dataset,
        "framework": framework,
        "tile_mode": tile_mode,
        "success": False,
        "error": None,
        "config": {
            "epochs": 8,
            "batch_size": batch_size,
            "tile_size": tile_size,
            "es_patience": 9999,
            "tile_format": tile_format,
            "model_type": "DeepLabV3_plus",
            "class_names": cfg["display_class_names"],
        },
        "training": {},
        "testing": {},
    }

    try:
        # ----- 1. Set framework + tile mode (DSAI only) -----
        HAS_CONFIG = False
        try:
            from base.config import ModelDefaults
            ModelDefaults.DEFAULT_FRAMEWORK = framework
            if tile_mode != "default":
                ModelDefaults.TILE_GENERATION_MODE = tile_mode
            HAS_CONFIG = True
        except ImportError:
            pass

        # ----- 2. Create metadata via direct pickle -----
        classNames = list(cfg["CLASS_NAMES"])
        if classNames[-1] != "black":
            classNames.append("black")
        nblack = len(classNames)
        nwhite = cfg["WS"][2][cfg["WS"][1][0] - 1]

        epochs_value = 2 if getattr(args, "fast_smoke", False) else 8
        metadata = {
            "pthim": pthim, "pthDL": exp_dir, "WS": cfg["WS"],
            "nm": exp_name, "umpix": cfg["umpix"], "cmap": cfg["CMAP"],
            "sxy": tile_size, "classNames": classNames,
            "ntrain": cfg["NTRAIN"], "nvalidate": cfg["NVALIDATE"],
            "nblack": nblack, "nwhite": nwhite,
            "batch_size": batch_size, "model_type": "DeepLabV3_plus",
            "pthtest": pthtest,
            "resolution_subdir": cfg["resolution_subdir"],
            # DSAI reads these; old versions ignore unknown keys:
            "epochs": epochs_value, "es_patience": 9999,
            "tile_format": tile_format,
            "framework": framework,
        }

        net_pkl = os.path.join(exp_dir, "net.pkl")
        # If net.pkl already exists (shared tiles), update it
        if os.path.isfile(net_pkl):
            with open(net_pkl, "rb") as f:
                existing = pickle.load(f)
            existing.update(metadata)
            metadata = existing
        with open(net_pkl, "wb") as f:
            pickle.dump(metadata, f)

        print(f"[worker:{exp_name}] Metadata created ({len(metadata)} keys)")

        # ----- 3. Create tiles (if needed) -----
        if create_tiles:
            print(f"[worker:{exp_name}] Loading annotations...")
            from base.data.annotation import load_annotation_data
            ctlist0, numann0, need_tiles = load_annotation_data(
                exp_dir, data_path, pthim, [])

            print(f"[worker:{exp_name}] Creating tiles "
                  f"(mode={tile_mode}, format={tile_format})...")
            from base.data.tiles import create_training_tiles
            if HAS_CONFIG and tile_mode != "default":
                from base.config import MODERN_CONFIG, LEGACY_CONFIG
                tile_config = (MODERN_CONFIG if tile_mode == "modern"
                               else LEGACY_CONFIG)
                create_training_tiles(
                    exp_dir, numann0, ctlist0, need_tiles,
                    config=tile_config)
            else:
                create_training_tiles(exp_dir, numann0, ctlist0, need_tiles)
            print(f"[worker:{exp_name}] Tiles created")

        # Count tiles
        train_im_dir = os.path.join(exp_dir, "training", "im")
        val_im_dir = os.path.join(exp_dir, "validation", "im")
        n_train = len(glob(os.path.join(train_im_dir, "*"))) if os.path.isdir(train_im_dir) else 0
        n_val = len(glob(os.path.join(val_im_dir, "*"))) if os.path.isdir(val_im_dir) else 0
        result["config"]["num_train_tiles"] = n_train
        result["config"]["num_val_tiles"] = n_val
        print(f"[worker:{exp_name}] Tiles: {n_train} train, {n_val} val")

        # ----- 4. Train -----
        print(f"[worker:{exp_name}] Training ({framework}, "
              f"batch_size={batch_size})...")
        from base.models.training import train_segmentation_model_cnns
        t0 = time.time()
        train_segmentation_model_cnns(
            exp_dir, retrain_model=True, seed=args.seed,
        )
        train_time = round(time.time() - t0, 1)
        result["training"]["train_time_s"] = train_time
        print(f"[worker:{exp_name}] Training finished in {train_time}s")

        # Extract training history from updated net.pkl
        with open(net_pkl, "rb") as f:
            post_meta = pickle.load(f)
        history = post_meta.get("history", {})
        result["training"]["history"] = _serialise(history)
        if history:
            result["training"]["epochs_completed"] = len(
                history.get("loss", []))
            for key in ("loss", "accuracy", "val_loss", "val_accuracy"):
                vals = history.get(key, [])
                if vals:
                    result["training"][f"final_{key}"] = round(
                        float(vals[-1]), 6)
        result["training"]["training_time_meta"] = post_meta.get(
            "training_time")

        # ----- 5. Test -----
        if pthtest and os.path.isdir(pthtest):
            print(f"[worker:{exp_name}] Testing...")
            t0 = time.time()

            from base.evaluation.testing import SegmentationModelTester
            # Handle DSAI (4 args) vs old (3 args) constructor
            try:
                tester = SegmentationModelTester(
                    exp_dir, pthtest, pthtestim,
                    classification_output_dir=os.path.join(
                        exp_dir, "classification_output"))
            except TypeError:
                tester = SegmentationModelTester(exp_dir, pthtest, pthtestim)

            # Handle DSAI (show_fig param) vs old (no param)
            import inspect
            if "show_fig" in inspect.signature(tester.test).parameters:
                metrics = tester.test(show_fig=False)
            else:
                metrics = tester.test()

            test_time = round(time.time() - t0, 1)
            result["testing"]["test_time_s"] = test_time

            if metrics:
                cm = metrics.get("confusion_matrix")
                cwm = metrics.get("confusion_with_metrics")
                if cm is not None:
                    result["testing"]["confusion_matrix"] = _serialise(cm)
                if cwm is not None:
                    cwm_arr = np.array(cwm)
                    result["testing"]["confusion_with_metrics"] = _serialise(
                        cwm_arr)

                    # Extract per-class metrics
                    display_names = cfg["display_class_names"]
                    n_classes = len(display_names)

                    # Precision: last row, first n columns
                    if cwm_arr.shape[0] > n_classes:
                        prec = cwm_arr[-1, :n_classes]
                        result["testing"]["per_class_precision"] = {
                            name: round(float(prec[i]), 2)
                            for i, name in enumerate(display_names)
                            if i < len(prec)
                        }
                    # Recall: first n rows, last column
                    if cwm_arr.shape[1] > n_classes:
                        rec = cwm_arr[:n_classes, -1]
                        result["testing"]["per_class_recall"] = {
                            name: round(float(rec[i]), 2)
                            for i, name in enumerate(display_names)
                            if i < len(rec)
                        }
                    # F1
                    prec_dict = result["testing"].get("per_class_precision", {})
                    rec_dict = result["testing"].get("per_class_recall", {})
                    f1 = {}
                    for name in display_names:
                        p = prec_dict.get(name, 0)
                        r = rec_dict.get(name, 0)
                        f1[name] = round(2 * p * r / (p + r), 2) if (p + r) > 0 else 0.0
                    result["testing"]["per_class_f1"] = f1

                    # Overall accuracy: last row, last col
                    result["testing"]["overall_accuracy"] = round(
                        float(cwm_arr[-1, -1]), 2)

                    # Also compute from raw confusion matrix
                    if cm is not None:
                        cm_arr = np.array(cm)
                        total = cm_arr.sum()
                        correct = np.trace(cm_arr)
                        if total > 0:
                            result["testing"]["overall_accuracy_raw"] = round(
                                100.0 * correct / total, 2)

            print(f"[worker:{exp_name}] Testing finished in {test_time}s, "
                  f"accuracy={result['testing'].get('overall_accuracy', 'N/A')}%")
        else:
            print(f"[worker:{exp_name}] Skipping test (no test path)")

        result["success"] = True

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = tb_module.format_exc()
        print(f"[worker:{exp_name}] FAILED: {result['error']}")

    # ----- Save results -----
    out_path = os.path.join(exp_dir, "worker_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # ----- Generate experiment report -----
    _write_experiment_report(exp_dir, result, cfg)

    print(f"[worker:{exp_name}] Results written to {out_path}")


def _write_experiment_report(exp_dir: str, result: dict,
                             cfg: dict) -> None:
    """Write a comprehensive human-readable report for one experiment."""
    lines = []
    lines.append("=" * 70)
    lines.append("VERSION COMPARISON EXPERIMENT REPORT")
    lines.append("=" * 70)
    lines.append(f"Experiment:   {result['experiment_name']}")
    lines.append(f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Code Version: {result['code_version']} "
                 f"(commit {result['commit']})")
    lines.append(f"Dataset:      {result['dataset']}")
    lines.append(f"Framework:    {result['framework']}")
    lines.append(f"Tile Mode:    {result['tile_mode']}")
    lines.append(f"Status:       {'SUCCESS' if result['success'] else 'FAILED'}")
    if result.get("error"):
        lines.append(f"Error:        {result['error']}")
    lines.append("")

    # Configuration
    conf = result.get("config", {})
    lines.append("CONFIGURATION")
    lines.append(f"  Model:       {conf.get('model_type', 'DeepLabV3_plus')}")
    lines.append(f"  Epochs:      {conf.get('epochs', 8)} "
                 f"(early stopping disabled)")
    lines.append(f"  Batch Size:  {conf.get('batch_size', 'N/A')}")
    lines.append(f"  Tile Size:   {conf.get('tile_size', 'N/A')}")
    lines.append(f"  Tile Format: {conf.get('tile_format', 'N/A')}")
    lines.append("")

    # Dataset info
    lines.append("DATASET")
    lines.append(f"  Training tiles:   {conf.get('num_train_tiles', 'N/A')}")
    lines.append(f"  Validation tiles: {conf.get('num_val_tiles', 'N/A')}")
    names = conf.get("class_names", cfg.get("display_class_names", []))
    lines.append(f"  Classes:          {', '.join(names)}")
    lines.append("")

    # Training results
    train = result.get("training", {})
    if train:
        lines.append("TRAINING RESULTS")
        t = train.get("train_time_s")
        if t is not None:
            h, rem = divmod(t, 3600)
            m, s = divmod(rem, 60)
            lines.append(f"  Training time:    {int(h)}h {int(m)}m {int(s)}s")
        lines.append(f"  Epochs completed: "
                     f"{train.get('epochs_completed', 'N/A')}")
        for key, label in [("final_loss", "Final train loss"),
                           ("final_accuracy", "Final train accuracy"),
                           ("final_val_loss", "Final val loss"),
                           ("final_val_accuracy", "Final val accuracy")]:
            val = train.get(key)
            if val is not None:
                if "accuracy" in key:
                    lines.append(f"  {label}: {val*100:.1f}%")
                else:
                    lines.append(f"  {label}:    {val:.4f}")

        # Per-epoch history
        hist = train.get("history", {})
        loss_list = hist.get("loss", [])
        if loss_list:
            lines.append("")
            lines.append("  Per-Epoch History:")
            header = f"    {'Epoch':<7} {'Loss':<12} {'Accuracy':<12}"
            vloss = hist.get("val_loss", [])
            vacc = hist.get("val_accuracy", [])
            if vloss:
                header += f" {'Val Loss':<12} {'Val Acc':<12}"
            lines.append(header)
            for i, l in enumerate(loss_list):
                row = f"    {i+1:<7} {l:<12.4f}"
                acc_list = hist.get("accuracy", [])
                row += f" {acc_list[i]*100:<11.1f}%" if i < len(acc_list) else f" {'N/A':<12}"
                if vloss and i < len(vloss):
                    row += f" {vloss[i]:<12.4f}"
                if vacc and i < len(vacc):
                    row += f" {vacc[i]*100:.1f}%"
                lines.append(row)
        lines.append("")

    # Test results
    test = result.get("testing", {})
    if test:
        lines.append("TEST RESULTS")
        tt = test.get("test_time_s")
        if tt is not None:
            lines.append(f"  Test time:        {tt}s")
        oa = test.get("overall_accuracy")
        if oa is not None:
            lines.append(f"  Overall accuracy: {oa:.1f}%")

        prec = test.get("per_class_precision", {})
        rec = test.get("per_class_recall", {})
        f1 = test.get("per_class_f1", {})
        if prec or rec:
            lines.append("")
            lines.append("  Per-Class Metrics:")
            lines.append(f"    {'Class':<16} {'Precision':>10} "
                         f"{'Recall':>10} {'F1':>10}")
            for name in names:
                p = prec.get(name)
                r = rec.get(name)
                f = f1.get(name)
                p_str = f"{p:.1f}%" if p is not None else "N/A"
                r_str = f"{r:.1f}%" if r is not None else "N/A"
                f_str = f"{f:.1f}%" if f is not None else "N/A"
                lines.append(f"    {name:<16} {p_str:>10} "
                             f"{r_str:>10} {f_str:>10}")

        # Raw confusion matrix
        cm = test.get("confusion_matrix")
        if cm is not None:
            lines.append("")
            lines.append("  Confusion Matrix (raw):")
            cm_arr = np.array(cm)
            # Truncate to display classes
            n = min(len(names), cm_arr.shape[0])
            short = [nm[:8] for nm in names[:n]]
            header = "    " + "".join(f"{s:>10}" for s in short)
            lines.append(header)
            for i in range(n):
                row = f"    {short[i]:<10}" + "".join(
                    f"{int(cm_arr[i][j]):>10}" for j in range(n))
                lines.append(row)

    lines.append("")
    lines.append("=" * 70)

    report_path = os.path.join(exp_dir, "experiment_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


# ===================================================================
# ORCHESTRATOR MODE
# ===================================================================

def _orchestrate(args):
    """Set up worktrees, run experiments, collect results."""
    versions = args.versions
    datasets = args.datasets
    gpu = args.gpu
    batch_size = args.batch_size
    tile_size = args.tile_size
    output_dir = args.output_dir
    timeout = args.timeout

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Validate versions
    for v in versions:
        if v not in VERSION_SPECS:
            print(f"ERROR: Unknown version '{v}'. "
                  f"Available: {list(VERSION_SPECS.keys())}")
            sys.exit(1)

    # Build experiment list
    experiments = _build_experiment_list(
        versions, datasets, args.frameworks, args.tile_modes)

    if not experiments:
        print("ERROR: No experiments to run with given filters.")
        sys.exit(1)

    # Set up worktrees
    print("\nSetting up git worktrees...")
    worktree_paths = _setup_worktrees(versions)

    # Get commit hashes
    commit_hashes = {}
    for v, wt in worktree_paths.items():
        commit_hashes[v] = _get_commit_hash(wt)

    # Results directory
    results_dir = os.path.join(output_dir, "version_comparison_results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Version Comparison — {timestamp}")
    print(f"  Output:     {results_dir}")
    print(f"  GPU:        {gpu}")
    print(f"  Batch size: {batch_size}")
    print(f"  Tile size:  {tile_size}")
    print(f"  Versions:   {versions}")
    print(f"  Datasets:   {datasets}")
    print(f"  Experiments: {len(experiments)}")
    for exp in experiments:
        print(f"    - {exp['name']}")
    print("=" * 70)

    # Group experiments by tile-sharing key
    tile_groups = {}
    for exp in experiments:
        key = _tile_group_key(exp)
        tile_groups.setdefault(key, []).append(exp)

    # Track which tile groups have been created
    tile_created = {}
    all_results = {}

    script_path = os.path.abspath(__file__)

    for exp in experiments:
        name = exp["name"]
        version = exp["version"]
        dataset = exp["dataset"]
        framework = exp["framework"]
        tile_mode = exp["tile_mode"]
        worktree = worktree_paths[version]
        commit = commit_hashes.get(version, "unknown")

        tg_key = _tile_group_key(exp)
        needs_tiles = tg_key not in tile_created
        data_path = DATASET_CONFIGS[dataset]["default_data_path"]

        exp_dir = os.path.join(results_dir, name)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n{'—' * 70}")
        print(f"[orchestrator] {name} (tiles={'CREATE' if needs_tiles else 'SHARED'})")
        print(f"{'—' * 70}")

        # Set up tile sharing
        if needs_tiles:
            # This experiment creates tiles
            tile_created[tg_key] = exp_dir
        else:
            # Link tiles from the first experiment in this group
            source_dir = tile_created[tg_key]
            _link_dir(os.path.join(source_dir, "training"),
                      os.path.join(exp_dir, "training"))
            _link_dir(os.path.join(source_dir, "validation"),
                      os.path.join(exp_dir, "validation"))
            for fname in ("annotations.pkl", "train_list.pkl"):
                src = os.path.join(source_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(exp_dir, fname))

        # Build subprocess command
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["TF_CUDNN_USE_AUTOTUNE"] = "0"
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        env["PYTHONUNBUFFERED"] = "1"
        # Critical: worker imports `base` from the worktree
        env["PYTHONPATH"] = worktree

        cmd = [
            sys.executable, script_path,
            "--worker",
            "--experiment-dir", exp_dir,
            "--worktree", worktree,
            "--dataset", dataset,
            "--data-path", data_path,
            "--framework", framework,
            "--tile-mode", tile_mode,
            "--batch-size", str(batch_size),
            "--tile-size", str(tile_size),
            "--version-name", version,
            "--commit", commit,
            "--exp-name", name,
        ]
        if needs_tiles:
            cmd.append("--create-tiles")
        if getattr(args, "seed", None) is not None:
            cmd += ["--seed", str(args.seed)]
            env["PYTHONHASHSEED"] = str(args.seed)

        print(f"  PYTHONPATH={worktree}")
        print(f"  CUDA_VISIBLE_DEVICES={gpu}")

        stdout_log = os.path.join(exp_dir, "worker_stdout.log")
        stderr_log = os.path.join(exp_dir, "worker_stderr.log")

        t0 = time.time()
        try:
            with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
                proc = subprocess.run(
                    cmd, env=env, stdout=out, stderr=err,
                    timeout=timeout)
            wall_time = round(time.time() - t0, 1)

            results_file = os.path.join(exp_dir, "worker_results.json")
            if os.path.isfile(results_file):
                with open(results_file) as f:
                    all_results[name] = json.load(f)
                all_results[name]["wall_time_s"] = wall_time
                all_results[name]["exit_code"] = proc.returncode
            else:
                all_results[name] = {
                    "success": False,
                    "error": f"No results file (exit {proc.returncode})",
                    "wall_time_s": wall_time,
                    "exit_code": proc.returncode,
                }
        except subprocess.TimeoutExpired:
            wall_time = round(time.time() - t0, 1)
            all_results[name] = {
                "success": False,
                "error": f"Timed out after {timeout}s",
                "wall_time_s": wall_time, "exit_code": -1,
            }
            print(f"[orchestrator] {name} TIMED OUT")

        status = "OK" if all_results[name].get("success") else "FAILED"
        acc = all_results[name].get("testing", {}).get(
            "overall_accuracy", "N/A")
        print(f"[orchestrator] {name}: {status} — accuracy={acc}% "
              f"(wall {all_results[name].get('wall_time_s', '?')}s)")

    # ----- Save combined results -----
    json_path = os.path.join(results_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ----- Save config -----
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "gpu": gpu,
            "batch_size": batch_size,
            "tile_size": tile_size,
            "versions": versions,
            "datasets": datasets,
            "experiments": [e["name"] for e in experiments],
            "commit_hashes": commit_hashes,
        }, f, indent=2)

    # ----- Collect confusion matrices -----
    _collect_confusion_matrices(results_dir, experiments, all_results)

    # ----- Generate CSV -----
    csv_path = os.path.join(results_dir, "comparison.csv")
    _generate_csv(csv_path, experiments, all_results)

    # ----- Print summary -----
    _print_summary(results_dir, experiments, all_results, datasets,
                   timestamp, gpu, batch_size, tile_size)

    print(f"\nAll outputs in: {results_dir}")


# ===================================================================
# Results collection and reporting
# ===================================================================

def _collect_confusion_matrices(results_dir: str, experiments: list,
                                all_results: dict) -> None:
    """Copy per-experiment confusion matrices and build grid figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    collected = []
    for exp in experiments:
        name = exp["name"]
        exp_dir = os.path.join(results_dir, name)
        src = os.path.join(exp_dir, "confusion_matrix_DeepLabV3_plus.png")
        if os.path.isfile(src):
            dst = os.path.join(results_dir, f"cm_{name}.png")
            shutil.copy2(src, dst)
            collected.append((name, dst))

    if not collected:
        print("\n  No confusion matrix figures found.")
        return

    print(f"\nCollected {len(collected)} confusion matrices -> {results_dir}")
    for name, path in collected:
        print(f"  cm_{name}.png")

    # Build combined grid
    n = len(collected)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (name, path) in enumerate(collected):
        r, c = divmod(idx, ncols)
        img = mpimg.imread(path)
        axes[r, c].imshow(img)
        # Short title: version / framework / tile_mode
        parts = name.split("__")
        title = f"{parts[0]} / {parts[1]} / {parts[2]} / {parts[3]}"
        axes[r, c].set_title(title, fontsize=9, fontweight='bold')
        axes[r, c].axis('off')

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis('off')

    plt.tight_layout()
    grid_path = os.path.join(results_dir, "confusion_matrices_all.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"  Combined grid -> confusion_matrices_all.png")


def _generate_csv(csv_path: str, experiments: list,
                  all_results: dict) -> None:
    """Write results to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "version", "dataset", "framework", "tile_mode",
            "accuracy", "train_time_s", "test_time_s", "epochs_completed",
        ])
        for exp in experiments:
            name = exp["name"]
            r = all_results.get(name, {})
            writer.writerow([
                name, exp["version"], exp["dataset"],
                exp["framework"], exp["tile_mode"],
                r.get("testing", {}).get("overall_accuracy", ""),
                r.get("training", {}).get("train_time_s", ""),
                r.get("testing", {}).get("test_time_s", ""),
                r.get("training", {}).get("epochs_completed", ""),
            ])
    print(f"\nCSV saved: {csv_path}")


def _print_summary(results_dir: str, experiments: list,
                   all_results: dict, datasets: list,
                   timestamp: str, gpu: str, batch_size: int,
                   tile_size: int) -> None:
    """Print and save a comprehensive cross-experiment summary."""
    lines = []
    sep = "=" * 80
    lines.append(sep)
    lines.append(f"VERSION COMPARISON SUMMARY \u2014 {timestamp}")
    lines.append(f"GPU: {gpu} | Batch: {batch_size} | "
                 f"Tile: {tile_size} | Epochs: 8")
    lines.append(sep)
    lines.append("")

    # ----- Overall results table -----
    lines.append("OVERALL RESULTS")
    header = (f"{'Version':<12} {'Dataset':<8} {'Framework':<12} "
              f"{'TileMode':<10} {'Accuracy':>9} {'Train(s)':>10} "
              f"{'Test(s)':>9}")
    lines.append(header)
    lines.append("-" * len(header))

    for exp in experiments:
        name = exp["name"]
        r = all_results.get(name, {})
        acc = r.get("testing", {}).get("overall_accuracy")
        tt = r.get("training", {}).get("train_time_s")
        tst = r.get("testing", {}).get("test_time_s")
        acc_s = f"{acc:.1f}%" if acc is not None else "FAILED"
        tt_s = f"{tt}" if tt is not None else ""
        tst_s = f"{tst}" if tst is not None else ""
        lines.append(f"{exp['version']:<12} {exp['dataset']:<8} "
                     f"{exp['framework']:<12} {exp['tile_mode']:<10} "
                     f"{acc_s:>9} {tt_s:>10} {tst_s:>9}")

    lines.append("")

    # ----- Framework comparison (DSAI pytorch vs tensorflow) -----
    dsai_exps = [e for e in experiments if e["version"] == "dsai"]
    if len(set(e["framework"] for e in dsai_exps)) > 1:
        lines.append("FRAMEWORK COMPARISON (DSAI — pytorch vs tensorflow)")
        header = f"{'Dataset':<8} {'TileMode':<10} {'PT Acc':>9} {'TF Acc':>9} {'Delta':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for ds in datasets:
            for tm in VERSION_SPECS["dsai"]["tile_modes"]:
                pt_name = _exp_name("dsai", ds, "pytorch", tm)
                tf_name = _exp_name("dsai", ds, "tensorflow", tm)
                pt_acc = all_results.get(pt_name, {}).get(
                    "testing", {}).get("overall_accuracy")
                tf_acc = all_results.get(tf_name, {}).get(
                    "testing", {}).get("overall_accuracy")
                pt_s = f"{pt_acc:.1f}%" if pt_acc is not None else "N/A"
                tf_s = f"{tf_acc:.1f}%" if tf_acc is not None else "N/A"
                delta = ""
                if pt_acc is not None and tf_acc is not None:
                    delta = f"{pt_acc - tf_acc:+.1f}%"
                lines.append(f"{ds:<8} {tm:<10} {pt_s:>9} "
                             f"{tf_s:>9} {delta:>8}")
        lines.append("")

    # ----- Tile mode comparison -----
    if any(e["version"] == "dsai" for e in experiments):
        dsai_modes = set(e["tile_mode"] for e in dsai_exps)
        if len(dsai_modes) > 1:
            lines.append("TILE MODE COMPARISON (DSAI — modern vs legacy)")
            header = f"{'Dataset':<8} {'Framework':<12} {'Modern':>9} {'Legacy':>9} {'Delta':>8}"
            lines.append(header)
            lines.append("-" * len(header))
            for ds in datasets:
                for fw in VERSION_SPECS["dsai"]["frameworks"]:
                    mod = _exp_name("dsai", ds, fw, "modern")
                    leg = _exp_name("dsai", ds, fw, "legacy")
                    m_acc = all_results.get(mod, {}).get(
                        "testing", {}).get("overall_accuracy")
                    l_acc = all_results.get(leg, {}).get(
                        "testing", {}).get("overall_accuracy")
                    m_s = f"{m_acc:.1f}%" if m_acc is not None else "N/A"
                    l_s = f"{l_acc:.1f}%" if l_acc is not None else "N/A"
                    delta = ""
                    if m_acc is not None and l_acc is not None:
                        delta = f"{m_acc - l_acc:+.1f}%"
                    lines.append(f"{ds:<8} {fw:<12} {m_s:>9} "
                                 f"{l_s:>9} {delta:>8}")
            lines.append("")

    # ----- Version comparison (tensorflow, modern/default baseline) -----
    lines.append("VERSION COMPARISON (tensorflow — delta from DSAI)")
    for ds in datasets:
        baseline_name = _exp_name("dsai", ds, "tensorflow", "modern")
        baseline_acc = all_results.get(baseline_name, {}).get(
            "testing", {}).get("overall_accuracy")
        lines.append(f"  Dataset: {ds} "
                     f"(baseline: DSAI/tensorflow/modern "
                     f"= {baseline_acc:.1f}%)" if baseline_acc else
                     f"  Dataset: {ds} (baseline: N/A)")
        header = f"    {'Version':<12} {'Accuracy':>9} {'Delta':>8}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for v in ["dsai", "main", "62614aa", "9a88b70"]:
            if v == "dsai":
                name = _exp_name(v, ds, "tensorflow", "modern")
            else:
                name = _exp_name(v, ds, "tensorflow", "default")
            r = all_results.get(name, {})
            acc = r.get("testing", {}).get("overall_accuracy")
            acc_s = f"{acc:.1f}%" if acc is not None else "N/A"
            delta = ""
            if acc is not None and baseline_acc is not None:
                d = acc - baseline_acc
                delta = f"{d:+.1f}%" if v != "dsai" else "--"
            lines.append(f"    {v:<12} {acc_s:>9} {delta:>8}")
        lines.append("")

    # ----- Per-class recall per dataset -----
    for ds in datasets:
        display_names = DATASET_CONFIGS[ds]["display_class_names"]
        short_names = [n[:8] for n in display_names]
        lines.append(f"PER-CLASS RECALL \u2014 {ds}")
        header = (f"  {'Version':<10} {'FW':<6} {'Mode':<8} " +
                  "".join(f"{s:>10}" for s in short_names))
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for exp in experiments:
            if exp["dataset"] != ds:
                continue
            name = exp["name"]
            r = all_results.get(name, {})
            rec = r.get("testing", {}).get("per_class_recall", {})
            vals = "".join(
                f"{rec.get(n, 0):>9.1f}%" if rec.get(n) is not None
                else f"{'N/A':>10}"
                for n in display_names)
            lines.append(f"  {exp['version']:<10} {exp['framework']:<6} "
                         f"{exp['tile_mode']:<8} {vals}")
        lines.append("")

    # ----- Per-class precision per dataset -----
    for ds in datasets:
        display_names = DATASET_CONFIGS[ds]["display_class_names"]
        short_names = [n[:8] for n in display_names]
        lines.append(f"PER-CLASS PRECISION \u2014 {ds}")
        header = (f"  {'Version':<10} {'FW':<6} {'Mode':<8} " +
                  "".join(f"{s:>10}" for s in short_names))
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for exp in experiments:
            if exp["dataset"] != ds:
                continue
            name = exp["name"]
            r = all_results.get(name, {})
            prec = r.get("testing", {}).get("per_class_precision", {})
            vals = "".join(
                f"{prec.get(n, 0):>9.1f}%" if prec.get(n) is not None
                else f"{'N/A':>10}"
                for n in display_names)
            lines.append(f"  {exp['version']:<10} {exp['framework']:<6} "
                         f"{exp['tile_mode']:<8} {vals}")
        lines.append("")

    lines.append(sep)

    # Print and save
    for line in lines:
        print(line)

    report_path = os.path.join(results_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved: {report_path}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-version comparison: full workflow across "
                    "code versions, frameworks, and tile modes.")

    # Orchestrator args
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU index (default: 0)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--tile-size", type=int, default=1024,
                        help="Tile size for model input (default: 1024)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(
                            os.path.expanduser("~"), "Desktop"),
                        help="Output directory (default: ~/Desktop)")
    parser.add_argument("--timeout", type=int, default=SUBPROCESS_TIMEOUT,
                        help=f"Per-experiment timeout in seconds "
                             f"(default: {SUBPROCESS_TIMEOUT})")

    # Filters
    parser.add_argument(
        "--datasets", nargs="+", default=["lungs", "liver"],
        choices=["lungs", "liver"],
        help="Datasets to run (default: lungs liver)")
    parser.add_argument(
        "--versions", nargs="+",
        default=list(VERSION_SPECS.keys()),
        help="Code versions to run (default: all)")
    parser.add_argument(
        "--frameworks", nargs="+", default=None,
        choices=["pytorch", "tensorflow"],
        help="Framework filter (default: all available per version)")
    parser.add_argument(
        "--tile-modes", nargs="+", default=None,
        choices=["modern", "legacy", "default"],
        help="Tile mode filter (default: all available per version)")

    # Worker args (internal)
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--experiment-dir", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worktree", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--data-path", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--framework", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--tile-mode", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--version-name", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--commit", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--exp-name", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--create-tiles", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for reproducibility "
                             "(propagates through TF and PyTorch trainers)")
    parser.add_argument("--fast-smoke", action="store_true",
                        help="Smoke-test mode: sets epochs=2 so a full "
                             "end-to-end run (tiles + train + test) "
                             "completes in ~25 min instead of ~80 min. "
                             "Used only for verifying instrumentation; "
                             "not for production runs.")

    args = parser.parse_args()

    if args.worker:
        # Worker mode: import from worktree (set via PYTHONPATH env var)
        # Do NOT insert _PROJECT_ROOT — let PYTHONPATH take precedence
        worktree = args.worktree
        if worktree and worktree not in sys.path:
            sys.path.insert(0, worktree)
        _run_worker(args)
    else:
        # Orchestrator mode: use DSAI code
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        _orchestrate(args)


if __name__ == "__main__":
    main()
