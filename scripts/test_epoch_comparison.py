"""
Epoch Comparison Script for DeepLabV3+ Training

Trains DeepLabV3+ at multiple epoch counts and compares segmentation
performance.  Supports both PyTorch and TensorFlow frameworks via --framework.
Each epoch count runs as a separate subprocess so that GPU state is fully reset
between runs (same isolation pattern as test_gpu_configurations.py).

When results exist for the *other* framework, a cross-framework comparison
report and plot are generated automatically.

Supports multiple datasets via --dataset (default: liver).

Default epoch counts: 2, 8, 25, 50, 100

Usage:
    python scripts/test_epoch_comparison.py                              # Liver, PyTorch (default)
    python scripts/test_epoch_comparison.py --dataset lungs              # Lungs dataset
    python scripts/test_epoch_comparison.py --framework tensorflow       # TensorFlow
    python scripts/test_epoch_comparison.py --epochs 2 8 25              # Custom subset
    python scripts/test_epoch_comparison.py --no-early-stopping          # Disable early stopping
    python scripts/test_epoch_comparison.py --gpu 0                      # Select GPU
    python scripts/test_epoch_comparison.py --data-path D:/my_data       # Custom data path
    python scripts/test_epoch_comparison.py --dataset lungs --epochs 2 8 --frameworks pytorch
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime

# Ensure the project root is on sys.path so that `base` is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = None  # resolved from dataset config if not specified
DEFAULT_EPOCH_COUNTS = [2, 8, 25, 50, 100]
DEFAULT_TILE_SIZE = 1024
DEFAULT_BATCH_SIZE = 3
DEFAULT_GPU = "0"
SUBPROCESS_TIMEOUT = 12 * 60 * 60  # 12 hours

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "liver": {
        "WS": [[0, 0, 0, 0, 2, 0, 2], [7, 6], [1, 2, 3, 4, 5, 6, 7], [6, 4, 2, 3, 5, 1, 7], []],
        "CMAP": np.array([
            [230, 190, 100],   # PDAC
            [65,  155, 210],   # bile duct
            [145,  35,  35],   # vasculature
            [158,  24, 118],   # hepatocyte
            [30,   50,  50],   # immune
            [235, 188, 215],   # stroma
            [255, 255, 255],   # whitespace
        ]),
        "CLASS_NAMES": ["PDAC", "bile duct", "vasculature", "hepatocyte", "immune", "stroma", "whitespace"],
        "display_class_names": ["PDAC", "bile duct", "vasculature", "hepatocyte", "immune", "stroma"],
        "NTRAIN": 15, "NVALIDATE": 3, "umpix": 1,
        "resolution_subdir": "10x", "test_subdir": "testing_image",
        "default_data_path": r"C:\Users\tnewton\Desktop\liver_tissue_data",
    },
    "lungs": {
        "WS": [[0, 2, 0, 0, 2, 0], [5, 6], [1, 2, 3, 4, 5, 6], [6, 2, 4, 3, 1, 5], []],
        "CMAP": np.array([
            [128,   0, 255],   # bronchioles
            [166, 193, 202],   # alveoli
            [255,   0,   0],   # vasculature
            [128,  64,   0],   # mets
            [255, 255, 255],   # whitespace
            [255, 128, 192],   # collagen
        ]),
        "CLASS_NAMES": ["bronchioles", "alveoli", "vasculature", "mets", "whitespace", "collagen"],
        "display_class_names": ["bronchioles", "alveoli", "vasculature", "mets", "collagen"],
        "NTRAIN": 15, "NVALIDATE": 3, "umpix": 2,
        "resolution_subdir": "5x", "test_subdir": "test",
        "default_data_path": r"C:\Users\tnewton\Desktop\lungs_data",
    },
}


def _get_dataset_config(dataset_name):
    """Return the configuration dict for a named dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


# ---------------------------------------------------------------------------
# Tile-creation subprocess (--create-tiles)
# ---------------------------------------------------------------------------

def _run_create_tiles(args):
    """Create shared training/validation tiles (runs in its own subprocess)."""
    shared_dir = args.shared_dir
    data_path = args.data_path
    tile_size = args.tile_size
    cfg = _get_dataset_config(args.dataset)

    from base.models.utils import create_initial_model_metadata, save_model_metadata
    from base.data.annotation import load_annotation_data
    from base.data.tiles import create_training_tiles

    pthim = os.path.join(data_path, cfg["resolution_subdir"])
    pthtest = os.path.join(data_path, cfg["test_subdir"])

    print(f"[tiles] Creating metadata in {shared_dir} (dataset={args.dataset})")
    create_initial_model_metadata(
        pthDL=shared_dir,
        pthim=pthim,
        WS=cfg["WS"],
        nm="epoch_comparison",
        umpix=cfg["umpix"],
        cmap=cfg["CMAP"],
        sxy=tile_size,
        classNames=cfg["CLASS_NAMES"],
        ntrain=cfg["NTRAIN"],
        nvalidate=cfg["NVALIDATE"],
        pthtest=pthtest,
        tile_format='png',
    )

    # Store resolution_subdir in net.pkl so workers can construct pthtestim
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


# ---------------------------------------------------------------------------
# Worker subprocess (--worker)
# ---------------------------------------------------------------------------

def _run_worker(args):
    """Train + test a single epoch configuration (runs in its own subprocess).

    CUDA_VISIBLE_DEVICES is already set by the orchestrator *before* this
    process was spawned.
    """
    # Force non-interactive matplotlib backend before framework imports
    import matplotlib
    matplotlib.use('Agg')

    from base.config import ModelDefaults
    ModelDefaults.DEFAULT_FRAMEWORK = args.framework

    from base.models.utils import save_model_metadata
    from base.models.training import train_segmentation_model_cnns
    from base.evaluation.testing import test_segmentation_model

    model_dir = args.model_dir
    epoch_count = args.epoch_count
    es_patience = args.es_patience

    framework = args.framework
    result = {
        "epoch_count": epoch_count,
        "framework": framework,
        "es_patience": es_patience,
        "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "batch_size": args.batch_size,
        "tile_size": args.tile_size,
        "success": False,
        "error": None,
        "train_time_s": None,
        "test_time_s": None,
        "test_metrics": None,
        "actual_epochs_completed": None,
        "early_stopped": False,
        "training_history_summary": None,
    }

    try:
        # Update net.pkl with case-specific parameters
        save_model_metadata(model_dir, {
            "epochs": epoch_count,
            "es_patience": es_patience,
            "batch_size": args.batch_size,
            "framework": framework,
        })

        # --- Train ---
        print(f"[worker:epochs_{epoch_count}] Training (epochs={epoch_count}, "
              f"es_patience={es_patience}, batch_size={args.batch_size})")
        t0 = time.time()
        train_segmentation_model_cnns(model_dir, retrain_model=True)
        result["train_time_s"] = round(time.time() - t0, 1)
        print(f"[worker:epochs_{epoch_count}] Training finished in "
              f"{result['train_time_s']}s")

        # --- Load training history ---
        hist_summary = _extract_history(model_dir, framework, epoch_count)
        if hist_summary:
            result["actual_epochs_completed"] = hist_summary["actual_epochs_completed"]
            result["early_stopped"] = hist_summary["early_stopped"]
            result["training_history_summary"] = hist_summary

        # --- Test ---
        with open(os.path.join(model_dir, "net.pkl"), "rb") as f:
            meta = pickle.load(f)
        pthtest = meta.get("pthtest", "")
        res_subdir = meta.get("resolution_subdir", "10x")
        pthtestim = os.path.join(pthtest, res_subdir) if pthtest else ""

        if pthtest and os.path.isdir(pthtest):
            print(f"[worker:epochs_{epoch_count}] Testing")
            t0 = time.time()
            # Use per-worker classification directory to avoid caching
            # across epoch counts and frameworks
            classification_output_dir = os.path.join(model_dir, "classification_output")
            metrics = test_segmentation_model(
                model_dir, pthtest, pthtestim, show_fig=False,
                classification_output_dir=classification_output_dir
            )
            result["test_time_s"] = round(time.time() - t0, 1)
            if metrics:
                result["test_metrics"] = _serialise(metrics)
            print(f"[worker:epochs_{epoch_count}] Testing finished in "
                  f"{result['test_time_s']}s")
        else:
            print(f"[worker:epochs_{epoch_count}] Skipping test (no test path)")

        result["success"] = True

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[worker:epochs_{epoch_count}] FAILED: {result['error']}")

    # Always write results
    out_path = os.path.join(model_dir, "worker_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[worker:epochs_{epoch_count}] Results written to {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_history(model_dir, framework, epoch_count):
    """Extract training history from either framework's output format.

    PyTorch saves training_history.pkl with keys: train_loss, val_loss, val_accuracy.
    TensorFlow stores history in net.pkl['history'] with keys: loss, val_loss, val_accuracy.

    Returns a normalised dict, or None if no history is available.
    """
    if framework == "pytorch":
        path = os.path.join(model_dir, "training_history.pkl")
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            h = pickle.load(f)
        train_loss = h.get("train_loss", [])
        val_loss = h.get("val_loss", [])
        val_acc = h.get("val_accuracy", [])
    else:  # tensorflow
        pkl_path = os.path.join(model_dir, "net.pkl")
        if not os.path.isfile(pkl_path):
            return None
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        h = data.get("history", {})
        if not h:
            return None
        train_loss = h.get("loss", [])
        val_loss = h.get("val_loss", [])
        val_acc = h.get("val_accuracy", [])

    actual_epochs = len(train_loss)
    return {
        "final_train_loss": float(train_loss[-1]) if train_loss else None,
        "final_val_loss": float(val_loss[-1]) if val_loss else None,
        "best_val_loss": float(min(val_loss)) if val_loss else None,
        "final_val_accuracy": float(val_acc[-1]) if val_acc else None,
        "num_train_loss_points": len(train_loss),
        "num_val_points": len(val_loss),
        "train_loss_curve": [float(v) for v in train_loss],
        "val_loss_curve": [float(v) for v in val_loss],
        "actual_epochs_completed": actual_epochs,
        "early_stopped": actual_epochs < epoch_count,
    }


def _serialise(obj):
    """Recursively convert numpy types so json.dump works."""
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


def _link_dir(src, dst):
    """Create a Windows directory junction (or copy as fallback)."""
    if os.path.exists(dst):
        return
    try:
        subprocess.check_call(
            ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  Junction failed, copying {src} -> {dst}")
        shutil.copytree(src, dst)


def _copy_file(src, dst):
    """Copy a file if it exists and the destination doesn't."""
    if os.path.isfile(src) and not os.path.isfile(dst):
        shutil.copy2(src, dst)


def _launch_subprocess(script, extra_args, env, log_stdout, log_stderr, timeout):
    """Launch a subprocess, capturing output to log files."""
    cmd = [sys.executable, script] + extra_args
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  ENV: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    with open(log_stdout, "w") as out, open(log_stderr, "w") as err:
        proc = subprocess.run(
            cmd, env=env, stdout=out, stderr=err, timeout=timeout
        )
    return proc.returncode


# ---------------------------------------------------------------------------
# Orchestrator (default mode)
# ---------------------------------------------------------------------------

def _orchestrate(args):
    """Main orchestrator: ensure shared tiles exist, then run each epoch count."""
    data_path = args.data_path
    epoch_counts = sorted(args.epochs)
    tile_size = args.tile_size
    batch_size = args.batch_size
    gpu = args.gpu
    no_early_stopping = args.no_early_stopping
    framework = args.framework
    dataset = args.dataset
    cfg = _get_dataset_config(dataset)

    # Compute display class indices (maps display_class_names → CLASS_NAMES indices)
    display_class_names = cfg["display_class_names"]
    display_indices = [cfg["CLASS_NAMES"].index(cn) for cn in display_class_names]

    suffix = f"_{args.results_suffix}" if getattr(args, "results_suffix", "") else ""
    results_dir = os.path.join(data_path, f"epoch_comparison_results_{framework}{suffix}")
    os.makedirs(results_dir, exist_ok=True)

    # Reuse shared tiles from gpu_test_results or the other framework if they
    # exist, otherwise create new ones.  Tiles are framework-agnostic.
    gpu_shared_dir = os.path.join(data_path, "gpu_test_results", "shared_tiles")
    local_shared_dir = os.path.join(results_dir, "shared_tiles")
    other_fw = "tensorflow" if framework == "pytorch" else "pytorch"
    other_shared_dir = os.path.join(
        data_path, f"epoch_comparison_results_{other_fw}{suffix}", "shared_tiles"
    )

    # Determine which shared directory to use
    if _tiles_exist(gpu_shared_dir):
        shared_dir = gpu_shared_dir
        print(f"[orchestrator] Reusing shared tiles from {shared_dir}")
    elif _tiles_exist(local_shared_dir):
        shared_dir = local_shared_dir
        print(f"[orchestrator] Reusing shared tiles from {shared_dir}")
    elif _tiles_exist(other_shared_dir):
        shared_dir = other_shared_dir
        print(f"[orchestrator] Reusing shared tiles from {shared_dir}")
    else:
        shared_dir = local_shared_dir
        # Fall through to tile creation below

    script_path = os.path.abspath(__file__)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    es_patience = 9999 if no_early_stopping else 6

    print("=" * 70)
    print(f"Epoch Comparison Test — {timestamp}")
    print(f"  Dataset:         {dataset}")
    print(f"  Framework:       {framework}")
    print(f"  Data path:       {data_path}")
    print(f"  Output:          {results_dir}")
    print(f"  Epoch counts:    {epoch_counts}")
    print(f"  Tile size:       {tile_size}")
    print(f"  Batch size:      {batch_size}")
    print(f"  GPU:             {gpu}")
    print(f"  Early stopping:  {'disabled (patience=9999)' if no_early_stopping else 'enabled (patience=6)'}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Create shared tiles if needed (CPU-only subprocess)
    # ------------------------------------------------------------------
    if not _tiles_exist(shared_dir):
        os.makedirs(shared_dir, exist_ok=True)
        print("\n[orchestrator] Creating shared tiles …")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only

        rc = _launch_subprocess(
            script_path,
            [
                "--create-tiles",
                "--shared-dir", shared_dir,
                "--data-path", data_path,
                "--tile-size", str(tile_size),
                "--dataset", dataset,
            ],
            env=env,
            log_stdout=os.path.join(results_dir, "tile_creation_stdout.log"),
            log_stderr=os.path.join(results_dir, "tile_creation_stderr.log"),
            timeout=SUBPROCESS_TIMEOUT,
        )
        if rc != 0:
            print(f"[orchestrator] Tile creation failed (exit code {rc}). "
                  "Check tile_creation_stderr.log")
            sys.exit(1)
        print("[orchestrator] Tiles created successfully")

    # ------------------------------------------------------------------
    # Step 2: Run each epoch count as a worker subprocess
    # ------------------------------------------------------------------
    case_results = {}

    for epoch_count in epoch_counts:
        case_name = f"epochs_{epoch_count:03d}"
        case_dir = os.path.join(results_dir, case_name)

        print(f"\n{'—' * 70}")
        print(f"[orchestrator] Case: {case_name}  "
              f"(epochs={epoch_count}, es_patience={es_patience})")
        print(f"{'—' * 70}")

        os.makedirs(case_dir, exist_ok=True)

        # Link shared tiles into case directory
        _link_dir(os.path.join(shared_dir, "training"),
                  os.path.join(case_dir, "training"))
        _link_dir(os.path.join(shared_dir, "validation"),
                  os.path.join(case_dir, "validation"))

        # Copy shared metadata (always overwrite net.pkl for clean base)
        src_pkl = os.path.join(shared_dir, "net.pkl")
        if os.path.isfile(src_pkl):
            shutil.copy2(src_pkl, os.path.join(case_dir, "net.pkl"))
        _copy_file(os.path.join(shared_dir, "annotations.pkl"),
                   os.path.join(case_dir, "annotations.pkl"))
        _copy_file(os.path.join(shared_dir, "train_list.pkl"),
                   os.path.join(case_dir, "train_list.pkl"))

        # Verify critical files
        for required in ("net.pkl", "annotations.pkl", "train_list.pkl"):
            if not os.path.isfile(os.path.join(case_dir, required)):
                print(f"  WARNING: {required} missing in {case_dir}")

        # Launch worker
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

        t0 = time.time()
        try:
            rc = _launch_subprocess(
                script_path,
                [
                    "--worker",
                    "--framework", framework,
                    "--model-dir", case_dir,
                    "--epoch-count", str(epoch_count),
                    "--batch-size", str(batch_size),
                    "--tile-size", str(tile_size),
                    "--es-patience", str(es_patience),
                ],
                env=env,
                log_stdout=os.path.join(case_dir, "worker_stdout.log"),
                log_stderr=os.path.join(case_dir, "worker_stderr.log"),
                timeout=SUBPROCESS_TIMEOUT,
            )
            wall_time = round(time.time() - t0, 1)
            results_file = os.path.join(case_dir, "worker_results.json")
            if os.path.isfile(results_file):
                with open(results_file) as f:
                    case_results[case_name] = json.load(f)
                case_results[case_name]["wall_time_s"] = wall_time
                case_results[case_name]["exit_code"] = rc
            else:
                case_results[case_name] = {
                    "success": False,
                    "error": f"No results file (exit code {rc})",
                    "wall_time_s": wall_time,
                    "exit_code": rc,
                }
        except subprocess.TimeoutExpired:
            wall_time = round(time.time() - t0, 1)
            case_results[case_name] = {
                "success": False,
                "error": f"Timed out after {SUBPROCESS_TIMEOUT}s",
                "wall_time_s": wall_time,
                "exit_code": -1,
            }
            print(f"[orchestrator] {case_name} TIMED OUT")

        status = "SUCCESS" if case_results[case_name].get("success") else "FAILED"
        print(f"[orchestrator] {case_name}: {status} "
              f"(wall {case_results[case_name]['wall_time_s']}s)")

    # ------------------------------------------------------------------
    # Step 3: Save raw JSON results
    # ------------------------------------------------------------------
    json_path = os.path.join(results_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(case_results, f, indent=2)

    # ------------------------------------------------------------------
    # Step 4: Generate summary report
    # ------------------------------------------------------------------
    _generate_summary_report(results_dir, case_results, epoch_counts, timestamp,
                             data_path, tile_size, batch_size, gpu,
                             no_early_stopping, framework,
                             class_names=display_class_names,
                             class_indices=display_indices)

    # ------------------------------------------------------------------
    # Step 5: Generate comparison plot
    # ------------------------------------------------------------------
    _generate_comparison_plot(results_dir, case_results, epoch_counts, framework,
                              class_names=display_class_names,
                              class_indices=display_indices)

    # ------------------------------------------------------------------
    # Step 6: Cross-framework comparison (if the other framework's results exist)
    # ------------------------------------------------------------------
    _generate_cross_framework_comparison(results_dir, framework, case_results,
                                         epoch_counts, data_path,
                                         suffix=getattr(args, "results_suffix", ""),
                                         class_names=display_class_names,
                                         class_indices=display_indices)

    print(f"\nAll outputs in: {results_dir}")


def _tiles_exist(directory):
    """Check whether shared tiles directory has the expected contents."""
    return (
        os.path.isdir(os.path.join(directory, "training"))
        and os.path.isdir(os.path.join(directory, "validation"))
        and os.path.isfile(os.path.join(directory, "annotations.pkl"))
        and os.path.isfile(os.path.join(directory, "train_list.pkl"))
    )


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _generate_summary_report(results_dir, case_results, epoch_counts, timestamp,
                             data_path, tile_size, batch_size, gpu,
                             no_early_stopping, framework,
                             class_names=None, class_indices=None):
    """Generate a human-readable summary_report.txt."""
    lines = []
    lines.append(f"Epoch Comparison Results — {timestamp}")
    lines.append(f"Framework: {framework}")
    lines.append(f"Data path: {data_path}")
    lines.append(f"Tile size: {tile_size}, Batch size: {batch_size}, GPU: {gpu}")
    lines.append(f"Early stopping: {'disabled' if no_early_stopping else 'enabled (patience=6)'}")
    lines.append("")

    # --- Main results table ---
    header = (f"{'Epochs':<8} {'Status':<10} {'Train(s)':<10} {'Test(s)':<10} "
              f"{'Accuracy':<10} {'Early_Stop':<12} {'Actual_Epochs':<15}")
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    for ec in epoch_counts:
        case_name = f"epochs_{ec:03d}"
        r = case_results.get(case_name, {})
        status = "SUCCESS" if r.get("success") else "FAILED"
        train_t = r.get("train_time_s", "—")
        test_t = r.get("test_time_s", "—")

        # Extract accuracy from confusion_with_metrics (last row, last col)
        accuracy = _extract_accuracy(r)
        acc_str = f"{accuracy:.1f}%" if accuracy is not None else "—"

        actual = r.get("actual_epochs_completed", "—")
        early = r.get("early_stopped", False)
        if early and actual is not None:
            es_str = f"Yes({actual})"
        else:
            es_str = "No"

        line = (f"{ec:<8} {status:<10} {str(train_t):<10} {str(test_t):<10} "
                f"{acc_str:<10} {es_str:<12} {str(actual):<15}")
        lines.append(line)

    # --- Per-class metrics tables ---
    successful = [(ec, case_results.get(f"epochs_{ec:03d}", {}))
                  for ec in epoch_counts
                  if case_results.get(f"epochs_{ec:03d}", {}).get("success")]

    if successful:
        lines.append("")
        lines.append("Per-class Recall (%)")
        lines.append("-" * 70)

        # Header row
        hdr = f"{'Epochs':<8}"
        for cn in class_names:
            hdr += f" {cn:<14}"
        lines.append(hdr)

        for ec, r in successful:
            recall = _extract_per_class_recall(r)
            row = f"{ec:<8}"
            if recall is not None:
                for idx in class_indices:
                    val = recall[idx] if idx < len(recall) else 0
                    row += f" {val:<14.1f}"
            else:
                for _ in class_names:
                    row += f" {'—':<14}"
            lines.append(row)

        lines.append("")
        lines.append("Per-class Precision (%)")
        lines.append("-" * 70)
        lines.append(hdr)

        for ec, r in successful:
            precision = _extract_per_class_precision(r)
            row = f"{ec:<8}"
            if precision is not None:
                for idx in class_indices:
                    val = precision[idx] if idx < len(precision) else 0
                    row += f" {val:<14.1f}"
            else:
                for _ in class_names:
                    row += f" {'—':<14}"
            lines.append(row)

    # Print errors
    for ec in epoch_counts:
        case_name = f"epochs_{ec:03d}"
        r = case_results.get(case_name, {})
        if not r.get("success") and r.get("error"):
            lines.append(f"\n  {case_name} error: {r['error']}")

    lines.append("")

    report_path = os.path.join(results_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # Also print to console
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for line in lines:
        print(line)
    print(f"\nReport saved to {report_path}")


def _extract_accuracy(result):
    """Extract overall accuracy from worker result dict."""
    metrics = result.get("test_metrics")
    if not metrics:
        return None
    cwm = metrics.get("confusion_with_metrics")
    if cwm and len(cwm) > 0 and len(cwm[-1]) > 0:
        return float(cwm[-1][-1])
    return None


def _extract_per_class_recall(result):
    """Extract per-class recall (last column, all rows except last)."""
    metrics = result.get("test_metrics")
    if not metrics:
        return None
    cwm = metrics.get("confusion_with_metrics")
    if cwm and len(cwm) > 1:
        return [float(row[-1]) for row in cwm[:-1]]
    return None


def _extract_per_class_precision(result):
    """Extract per-class precision (last row, all columns except last)."""
    metrics = result.get("test_metrics")
    if not metrics:
        return None
    cwm = metrics.get("confusion_with_metrics")
    if cwm and len(cwm) > 0:
        return [float(v) for v in cwm[-1][:-1]]
    return None


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _generate_comparison_plot(results_dir, case_results, epoch_counts,
                              framework="pytorch", class_names=None,
                              class_indices=None):
    """Generate a 4-panel matplotlib comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Collect data for successful cases
    epochs_list = []
    accuracies = []
    train_times = []
    actual_epochs_list = []
    per_class_recalls = {}   # {class_name: [values]}
    loss_curves = {}         # {epoch_count: [losses]}

    for cn in class_names:
        per_class_recalls[cn] = []

    for ec in epoch_counts:
        case_name = f"epochs_{ec:03d}"
        r = case_results.get(case_name, {})
        if not r.get("success"):
            continue

        epochs_list.append(ec)
        accuracies.append(_extract_accuracy(r) or 0)
        train_times.append(r.get("train_time_s", 0))
        actual_epochs_list.append(r.get("actual_epochs_completed", ec))

        recall = _extract_per_class_recall(r)
        for i, cn in enumerate(class_names):
            idx = class_indices[i]
            if recall and idx < len(recall):
                per_class_recalls[cn].append(recall[idx])
            else:
                per_class_recalls[cn].append(0)

        hist = r.get("training_history_summary", {})
        if hist and hist.get("train_loss_curve"):
            loss_curves[ec] = hist["train_loss_curve"]

    if not epochs_list:
        print("[orchestrator] No successful cases — skipping comparison plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fw_label = framework.capitalize() if framework else "PyTorch"
    fig.suptitle(f"Epoch Comparison — {fw_label} DeepLabV3+", fontsize=14, fontweight='bold')
    x_labels = [str(e) for e in epochs_list]
    x_pos = np.arange(len(epochs_list))

    # --- (a) Overall accuracy vs epoch count ---
    ax = axes[0, 0]
    bars = ax.bar(x_pos, accuracies, color='#4C72B0', edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Requested Epochs")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("(a) Overall Accuracy")
    ax.set_ylim(0, max(accuracies) * 1.1 if accuracies else 100)

    # --- (b) Training time vs epoch count ---
    ax = axes[0, 1]
    bars = ax.bar(x_pos, train_times, color='#DD8452', edgecolor='black', linewidth=0.5)
    for bar, tt, ae in zip(bars, train_times, actual_epochs_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{tt:.0f}s\n({ae} ep)", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Requested Epochs")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("(b) Training Time")

    # --- (c) Per-class recall comparison ---
    ax = axes[1, 0]
    n_classes = len(class_names)
    bar_width = 0.8 / n_classes
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    for i, cn in enumerate(class_names):
        offset = (i - n_classes / 2 + 0.5) * bar_width
        ax.bar(x_pos + offset, per_class_recalls[cn], bar_width,
               label=cn, color=colors[i], edgecolor='black', linewidth=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Requested Epochs")
    ax.set_ylabel("Recall (%)")
    ax.set_title("(c) Per-class Recall")
    ax.legend(fontsize=7, loc='lower right')

    # --- (d) Training loss curves ---
    ax = axes[1, 1]
    cmap_lines = plt.cm.viridis(np.linspace(0.1, 0.9, len(loss_curves)))
    for idx, (ec, losses) in enumerate(sorted(loss_curves.items())):
        ax.plot(range(1, len(losses) + 1), losses,
                label=f"{ec} epochs", color=cmap_lines[idx], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("(d) Training Loss Curves")
    if loss_curves:
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(results_dir, "comparison_plot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Cross-framework comparison
# ---------------------------------------------------------------------------

def _generate_cross_framework_comparison(results_dir, framework, case_results,
                                          epoch_counts, data_path, suffix="",
                                          class_names=None, class_indices=None):
    """Generate a side-by-side comparison when results from both frameworks exist.

    Looks for the *other* framework's ``all_results.json`` in
    ``epoch_comparison_results_{other}/``.  If found, produces:
      - cross_framework_comparison.txt  — tabular summary
      - cross_framework_comparison.png  — 4-panel figure
    """
    other = "tensorflow" if framework == "pytorch" else "pytorch"
    sfx = f"_{suffix}" if suffix else ""
    other_dir = os.path.join(data_path, f"epoch_comparison_results_{other}{sfx}")
    other_json = os.path.join(other_dir, "all_results.json")

    # Fallback to unsuffixed if suffixed doesn't exist
    if not os.path.isfile(other_json) and suffix:
        other_dir = os.path.join(data_path, f"epoch_comparison_results_{other}")
        other_json = os.path.join(other_dir, "all_results.json")

    if not os.path.isfile(other_json):
        print(f"[orchestrator] No {other} results found at {other_json} — "
              "skipping cross-framework comparison")
        return

    with open(other_json) as f:
        other_results = json.load(f)

    print(f"[orchestrator] Generating cross-framework comparison "
          f"({framework} vs {other})")

    # --- Build per-epoch-count data ---
    rows = []  # list of dicts for the text table
    for ec in epoch_counts:
        case_name = f"epochs_{ec:03d}"
        cur = case_results.get(case_name, {})
        oth = other_results.get(case_name, {})
        rows.append({
            "epochs": ec,
            f"{framework}_acc": _extract_accuracy(cur),
            f"{other}_acc": _extract_accuracy(oth),
            f"{framework}_time": cur.get("train_time_s"),
            f"{other}_time": oth.get("train_time_s"),
            f"{framework}_actual": cur.get("actual_epochs_completed"),
            f"{other}_actual": oth.get("actual_epochs_completed"),
            f"{framework}_es": cur.get("early_stopped", False),
            f"{other}_es": oth.get("early_stopped", False),
            f"{framework}_recall": _extract_per_class_recall(cur),
            f"{other}_recall": _extract_per_class_recall(oth),
            f"{framework}_loss": (cur.get("training_history_summary") or {}).get("train_loss_curve"),
            f"{other}_loss": (oth.get("training_history_summary") or {}).get("train_loss_curve"),
            f"{framework}_success": cur.get("success", False),
            f"{other}_success": oth.get("success", False),
        })

    # --- Text report ---
    _write_cross_framework_text(results_dir, rows, epoch_counts, framework,
                                 other, class_names, class_indices)

    # --- Plot ---
    _write_cross_framework_plot(results_dir, rows, epoch_counts, framework,
                                 other, class_names, class_indices)


def _write_cross_framework_text(results_dir, rows, epoch_counts, fw_a, fw_b,
                                 class_names, class_indices):
    """Write cross_framework_comparison.txt."""
    lines = []
    lines.append(f"Cross-Framework Comparison: {fw_a} vs {fw_b}")
    lines.append("=" * 80)
    lines.append("")

    # Main results table
    hdr = (f"{'Epochs':<8} "
           f"{'Acc(' + fw_a + ')':<14} {'Acc(' + fw_b + ')':<14} "
           f"{'Time(' + fw_a + ')':<14} {'Time(' + fw_b + ')':<14} "
           f"{'ES(' + fw_a + ')':<10} {'ES(' + fw_b + ')':<10}")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for row in rows:
        ec = row["epochs"]

        def _fmt_acc(val):
            return f"{val:.1f}%" if val is not None else "—"

        def _fmt_time(val):
            return f"{val:.0f}s" if val is not None else "—"

        def _fmt_es(stopped, actual):
            if stopped and actual is not None:
                return f"Yes({actual})"
            return "No"

        lines.append(
            f"{ec:<8} "
            f"{_fmt_acc(row[f'{fw_a}_acc']):<14} "
            f"{_fmt_acc(row[f'{fw_b}_acc']):<14} "
            f"{_fmt_time(row[f'{fw_a}_time']):<14} "
            f"{_fmt_time(row[f'{fw_b}_time']):<14} "
            f"{_fmt_es(row[f'{fw_a}_es'], row[f'{fw_a}_actual']):<10} "
            f"{_fmt_es(row[f'{fw_b}_es'], row[f'{fw_b}_actual']):<10}"
        )

    # Per-class recall comparison (for each epoch count)
    lines.append("")
    lines.append("Per-class Recall Comparison (%)")
    lines.append("-" * 80)

    for row in rows:
        ec = row["epochs"]
        r_a = row[f"{fw_a}_recall"]
        r_b = row[f"{fw_b}_recall"]
        if not (row[f"{fw_a}_success"] or row[f"{fw_b}_success"]):
            continue

        lines.append(f"\n  Epochs = {ec}:")
        hdr2 = f"    {'Class':<14} {fw_a:<12} {fw_b:<12} {'Diff':<12}"
        lines.append(hdr2)

        for i, cn in enumerate(class_names):
            idx = class_indices[i]
            va = r_a[idx] if r_a and idx < len(r_a) else None
            vb = r_b[idx] if r_b and idx < len(r_b) else None
            sa = f"{va:.1f}" if va is not None else "—"
            sb = f"{vb:.1f}" if vb is not None else "—"
            if va is not None and vb is not None:
                sd = f"{va - vb:+.1f}"
            else:
                sd = "—"
            lines.append(f"    {cn:<14} {sa:<12} {sb:<12} {sd:<12}")

    lines.append("")

    path = os.path.join(results_dir, "cross_framework_comparison.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Cross-framework text report saved to {path}")


def _write_cross_framework_plot(results_dir, rows, epoch_counts, fw_a, fw_b,
                                 class_names, class_indices):
    """Write cross_framework_comparison.png — 4-panel figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Cross-Framework Comparison — {fw_a.capitalize()} vs "
                 f"{fw_b.capitalize()} DeepLabV3+",
                 fontsize=14, fontweight='bold')

    # Collect arrays for successful cases common to both frameworks
    common_epochs = []
    acc_a, acc_b = [], []
    time_a, time_b = [], []

    for row in rows:
        ec = row["epochs"]
        # Include epoch if at least one framework succeeded
        if row[f"{fw_a}_success"] or row[f"{fw_b}_success"]:
            common_epochs.append(ec)
            acc_a.append(row[f"{fw_a}_acc"] if row[f"{fw_a}_acc"] is not None else 0)
            acc_b.append(row[f"{fw_b}_acc"] if row[f"{fw_b}_acc"] is not None else 0)
            time_a.append(row[f"{fw_a}_time"] if row[f"{fw_a}_time"] is not None else 0)
            time_b.append(row[f"{fw_b}_time"] if row[f"{fw_b}_time"] is not None else 0)

    if not common_epochs:
        print("[orchestrator] No overlapping successful cases — "
              "skipping cross-framework plot")
        plt.close(fig)
        return

    x_labels = [str(e) for e in common_epochs]
    x_pos = np.arange(len(common_epochs))
    bar_w = 0.35
    color_a = '#4C72B0'
    color_b = '#DD8452'

    # --- (a) Accuracy: grouped bars ---
    ax = axes[0, 0]
    ax.bar(x_pos - bar_w / 2, acc_a, bar_w, label=fw_a.capitalize(),
           color=color_a, edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + bar_w / 2, acc_b, bar_w, label=fw_b.capitalize(),
           color=color_b, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Requested Epochs")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("(a) Overall Accuracy")
    ax.legend(fontsize=9)
    all_acc = acc_a + acc_b
    ax.set_ylim(0, max(all_acc) * 1.12 if all_acc else 100)

    # --- (b) Training time: grouped bars ---
    ax = axes[0, 1]
    ax.bar(x_pos - bar_w / 2, time_a, bar_w, label=fw_a.capitalize(),
           color=color_a, edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + bar_w / 2, time_b, bar_w, label=fw_b.capitalize(),
           color=color_b, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Requested Epochs")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("(b) Training Time")
    ax.legend(fontsize=9)

    # --- (c) Training loss curves: solid=fw_a, dashed=fw_b ---
    ax = axes[1, 0]
    cmap_lines = plt.cm.viridis(np.linspace(0.1, 0.9, len(common_epochs)))
    for idx, row in enumerate(rows):
        ec = row["epochs"]
        if ec not in common_epochs:
            continue
        c = cmap_lines[common_epochs.index(ec)]
        loss_a = row[f"{fw_a}_loss"]
        loss_b = row[f"{fw_b}_loss"]
        if loss_a:
            ax.plot(range(1, len(loss_a) + 1), loss_a,
                    color=c, linewidth=1.5, linestyle='-',
                    label=f"{ec}ep {fw_a}")
        if loss_b:
            ax.plot(range(1, len(loss_b) + 1), loss_b,
                    color=c, linewidth=1.5, linestyle='--',
                    label=f"{ec}ep {fw_b}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"(c) Loss Curves (solid={fw_a}, dashed={fw_b})")
    ax.legend(fontsize=6, ncol=2)

    # --- (d) Per-class recall: grouped bars for epoch_count=8 (or first common) ---
    ax = axes[1, 1]
    # Pick a representative epoch count: prefer 8, then first available
    ref_epoch = 8 if 8 in common_epochs else common_epochs[0]
    ref_row = next(r for r in rows if r["epochs"] == ref_epoch)
    recall_a = ref_row[f"{fw_a}_recall"] or []
    recall_b = ref_row[f"{fw_b}_recall"] or []

    n_classes = len(class_names)
    if n_classes > 0:
        cx = np.arange(n_classes)
        ra = [recall_a[class_indices[i]] if recall_a and class_indices[i] < len(recall_a) else 0
              for i in range(n_classes)]
        rb = [recall_b[class_indices[i]] if recall_b and class_indices[i] < len(recall_b) else 0
              for i in range(n_classes)]
        ax.bar(cx - bar_w / 2, ra, bar_w, label=fw_a.capitalize(),
               color=color_a, edgecolor='black', linewidth=0.5)
        ax.bar(cx + bar_w / 2, rb, bar_w, label=fw_b.capitalize(),
               color=color_b, edgecolor='black', linewidth=0.5)
        ax.set_xticks(cx)
        ax.set_xticklabels(class_names[:n_classes], rotation=30, ha='right',
                           fontsize=8)
        ax.set_ylabel("Recall (%)")
        ax.set_title(f"(d) Per-class Recall (epochs={ref_epoch})")
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(results_dir, "cross_framework_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Cross-framework plot saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare DeepLabV3+ training at different epoch counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Orchestrator args
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()),
                        default="liver",
                        help="Dataset configuration to use (default: %(default)s)")
    parser.add_argument("--data-path", default=None,
                        help="Root data directory (default: resolved from --dataset)")
    parser.add_argument("--epochs", type=int, nargs="+", default=DEFAULT_EPOCH_COUNTS,
                        help="Epoch counts to compare (default: %(default)s)")
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"],
                        default="pytorch",
                        help="Training framework (default: %(default)s)")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                        help="Tile size in pixels (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Training batch size (default: %(default)s)")
    parser.add_argument("--gpu", default=DEFAULT_GPU,
                        help="GPU index for CUDA_VISIBLE_DEVICES (default: %(default)s)")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping (set patience=9999)")
    parser.add_argument("--results-suffix", default="",
                        help="Suffix appended to output directory name "
                             "(e.g., 'v2' -> epoch_comparison_results_pytorch_v2)")

    # Subprocess modes (hidden from user)
    parser.add_argument("--create-tiles", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--shared-dir", help=argparse.SUPPRESS)

    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", help=argparse.SUPPRESS)
    parser.add_argument("--epoch-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--es-patience", type=int, default=6,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Resolve default data path from dataset config if not explicitly provided
    if args.data_path is None:
        args.data_path = _get_dataset_config(args.dataset)["default_data_path"]

    if args.create_tiles:
        _run_create_tiles(args)
    elif args.worker:
        _run_worker(args)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
