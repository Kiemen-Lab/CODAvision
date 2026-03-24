"""
GPU Configuration Test Script

Tests TensorFlow and PyTorch training pipelines across single-GPU and multi-GPU
configurations. Each test case runs as a separate subprocess so that
CUDA_VISIBLE_DEVICES is set before any framework imports (the CUDA runtime
initialises on first import).

Test matrix (default):
  tf_single  — TensorFlow, 1 GPU,  batch_size=3
  tf_multi   — TensorFlow, 3 GPUs, batch_size=3  (MirroredStrategy)
  pt_single  — PyTorch,    1 GPU,  batch_size=3

Usage:
    python scripts/test_gpu_configurations.py
    python scripts/test_gpu_configurations.py --epochs 8
    python scripts/test_gpu_configurations.py --cases tf_single pt_single
    python scripts/test_gpu_configurations.py --data-path D:/my_data
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
# (needed when running as `python scripts/test_gpu_configurations.py`)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = r"C:\Users\tnewton\Desktop\liver_tissue_data"
DEFAULT_EPOCHS = 2
DEFAULT_TILE_SIZE = 1024
SUBPROCESS_TIMEOUT = 4 * 60 * 60  # 4 hours

# Dataset-specific configuration (mirrors non-gui_workflow.py)
WS = [
    [0, 0, 0, 0, 2, 0, 2],
    [7, 6],
    [1, 2, 3, 4, 5, 6, 7],
    [6, 4, 2, 3, 5, 1, 7],
    [],
]
CMAP = np.array([
    [230, 190, 100],   # PDAC
    [65,  155, 210],   # bile duct
    [145,  35,  35],   # vasculature
    [158,  24, 118],   # hepatocyte
    [30,   50,  50],   # immune
    [235, 188, 215],   # stroma
    [255, 255, 255],   # whitespace
])
CLASS_NAMES = [
    "PDAC", "bile duct", "vasculature", "hepatocyte",
    "immune", "stroma", "whitespace",
]
NTRAIN = 15
NVALIDATE = int(np.ceil(NTRAIN / 5))

# Test-case definitions: (name, framework, cuda_devices, batch_size)
TEST_CASES = {
    "tf_single": ("tensorflow", "0",     3),
    "tf_multi":  ("tensorflow", "0,1,2", 3),  # 1024px tiles need batch=1/GPU
    "pt_single": ("pytorch",    "0",     3),
}


# ---------------------------------------------------------------------------
# Tile-creation subprocess (--create-tiles)
# ---------------------------------------------------------------------------

def run_create_tiles(args):
    """Create shared training/validation tiles (runs in its own subprocess)."""
    shared_dir = args.shared_dir
    data_path = args.data_path
    tile_size = args.tile_size

    # Import base modules (no GPU needed for tile creation)
    from base.models.utils import create_initial_model_metadata
    from base.data.annotation import load_annotation_data
    from base.data.tiles import create_training_tiles

    pthim = os.path.join(data_path, "10x")
    pthtest = os.path.join(data_path, "testing_image")

    print(f"[tiles] Creating metadata in {shared_dir}")
    create_initial_model_metadata(
        pthDL=shared_dir,
        pthim=pthim,
        WS=WS,
        nm="gpu_test",
        umpix=1,
        cmap=CMAP,
        sxy=tile_size,
        classNames=CLASS_NAMES,
        ntrain=NTRAIN,
        nvalidate=NVALIDATE,
        pthtest=pthtest,
        tile_format='png',
    )

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

def run_worker(args):
    """Train + test a single GPU configuration (runs in its own subprocess).

    CUDA_VISIBLE_DEVICES is already set by the orchestrator *before* this
    process was spawned, so framework imports here see the correct GPU set.
    """
    # Force non-interactive matplotlib backend before any imports
    # that might trigger Qt/shiboken6 DLL loading (WinError 206 on Windows)
    import matplotlib
    matplotlib.use('Agg')

    from base.config import ModelDefaults
    ModelDefaults.DEFAULT_FRAMEWORK = args.framework

    from base.models.utils import save_model_metadata
    from base.models.training import train_segmentation_model_cnns
    from base.evaluation.testing import test_segmentation_model

    model_dir = args.model_dir
    result = {
        "case": args.case_name,
        "framework": args.framework,
        "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "success": False,
        "error": None,
        "train_time_s": None,
        "test_time_s": None,
        "test_metrics": None,
    }

    try:
        # Update net.pkl with case-specific parameters
        save_model_metadata(model_dir, {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "framework": args.framework,
        })

        # Train
        print(f"[worker:{args.case_name}] Training ({args.framework}, "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}, "
              f"batch_size={args.batch_size}, epochs={args.epochs})")
        t0 = time.time()
        train_segmentation_model_cnns(model_dir, retrain_model=True)
        result["train_time_s"] = round(time.time() - t0, 1)
        print(f"[worker:{args.case_name}] Training finished in {result['train_time_s']}s")

        # Test
        with open(os.path.join(model_dir, "net.pkl"), "rb") as f:
            meta = pickle.load(f)
        pthtest = meta.get("pthtest", "")
        pthtestim = os.path.join(pthtest, "10x") if pthtest else ""

        if pthtest and os.path.isdir(pthtest):
            print(f"[worker:{args.case_name}] Testing")
            t0 = time.time()
            metrics = test_segmentation_model(
                model_dir, pthtest, pthtestim, show_fig=False
            )
            result["test_time_s"] = round(time.time() - t0, 1)
            # Convert numpy values for JSON serialisation
            if metrics:
                result["test_metrics"] = _serialise(metrics)
            print(f"[worker:{args.case_name}] Testing finished in {result['test_time_s']}s")
        else:
            print(f"[worker:{args.case_name}] Skipping test (no test path)")

        result["success"] = True

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[worker:{args.case_name}] FAILED: {result['error']}")

    # Always write results
    out_path = os.path.join(model_dir, "worker_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[worker:{args.case_name}] Results written to {out_path}")


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def orchestrate(args):
    """Main orchestrator: create tiles once, then run each test case."""
    data_path = args.data_path
    epochs = args.epochs
    tile_size = args.tile_size
    cases = args.cases

    results_dir = os.path.join(data_path, "gpu_test_results")
    shared_dir = os.path.join(results_dir, "shared_tiles")
    os.makedirs(shared_dir, exist_ok=True)

    script_path = os.path.abspath(__file__)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print(f"GPU Configuration Test — {timestamp}")
    print(f"  Data path:  {data_path}")
    print(f"  Output:     {results_dir}")
    print(f"  Epochs:     {epochs}")
    print(f"  Tile size:  {tile_size}")
    print(f"  Cases:      {', '.join(cases)}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Create shared tiles (CPU-only subprocess)
    # ------------------------------------------------------------------
    tiles_exist = (
        os.path.isdir(os.path.join(shared_dir, "training"))
        and os.path.isdir(os.path.join(shared_dir, "validation"))
        and os.path.isfile(os.path.join(shared_dir, "annotations.pkl"))
        and os.path.isfile(os.path.join(shared_dir, "train_list.pkl"))
    )
    if tiles_exist:
        print("\n[orchestrator] Shared tiles already exist — skipping creation")
    else:
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
    # Step 2: Run each test case as a worker subprocess
    # ------------------------------------------------------------------
    case_results = {}

    for case_name in cases:
        framework, cuda_devices, batch_size = TEST_CASES[case_name]
        case_dir = os.path.join(results_dir, case_name.replace("_", "_") + "_gpu")

        print(f"\n{'—' * 70}")
        print(f"[orchestrator] Case: {case_name}  "
              f"(framework={framework}, GPUs={cuda_devices}, batch={batch_size})")
        print(f"{'—' * 70}")

        os.makedirs(case_dir, exist_ok=True)

        # Link shared tiles into case directory
        _link_dir(os.path.join(shared_dir, "training"),
                  os.path.join(case_dir, "training"))
        _link_dir(os.path.join(shared_dir, "validation"),
                  os.path.join(case_dir, "validation"))

        # Copy shared metadata files (always overwrite net.pkl so worker
        # starts from a clean base before applying case-specific params)
        src_pkl = os.path.join(shared_dir, "net.pkl")
        if os.path.isfile(src_pkl):
            shutil.copy2(src_pkl, os.path.join(case_dir, "net.pkl"))
        _copy_file(os.path.join(shared_dir, "annotations.pkl"),
                   os.path.join(case_dir, "annotations.pkl"))
        _copy_file(os.path.join(shared_dir, "train_list.pkl"),
                   os.path.join(case_dir, "train_list.pkl"))

        # Verify critical files exist in case directory
        for required in ("net.pkl", "annotations.pkl", "train_list.pkl"):
            if not os.path.isfile(os.path.join(case_dir, required)):
                print(f"  WARNING: {required} missing in {case_dir}")

        # Launch worker
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        t0 = time.time()
        try:
            rc = _launch_subprocess(
                script_path,
                [
                    "--worker",
                    "--case-name", case_name,
                    "--framework", framework,
                    "--batch-size", str(batch_size),
                    "--epochs", str(epochs),
                    "--model-dir", case_dir,
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
    # Step 3: Summary report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    report_lines = []
    report_lines.append(f"GPU Configuration Test Results — {timestamp}")
    report_lines.append(f"Data path: {data_path}")
    report_lines.append(f"Epochs: {epochs}, Tile size: {tile_size}")
    report_lines.append("")

    header = f"{'Case':<15} {'Status':<10} {'Framework':<12} {'GPUs':<8} {'Batch':<6} {'Train(s)':<10} {'Test(s)':<10} {'Wall(s)':<10}"
    sep = "-" * len(header)
    report_lines.append(header)
    report_lines.append(sep)
    print(header)
    print(sep)

    for case_name in cases:
        r = case_results.get(case_name, {})
        status = "SUCCESS" if r.get("success") else "FAILED"
        fw = r.get("framework", "?")
        gpus = r.get("cuda_devices", "?")
        batch = r.get("batch_size", "?")
        train_t = r.get("train_time_s", "—")
        test_t = r.get("test_time_s", "—")
        wall_t = r.get("wall_time_s", "—")

        line = f"{case_name:<15} {status:<10} {fw:<12} {gpus:<8} {str(batch):<6} {str(train_t):<10} {str(test_t):<10} {str(wall_t):<10}"
        report_lines.append(line)
        print(line)

    # Print errors for failed cases
    for case_name in cases:
        r = case_results.get(case_name, {})
        if not r.get("success") and r.get("error"):
            err_line = f"\n  {case_name} error: {r['error']}"
            report_lines.append(err_line)
            print(err_line)

    report_lines.append("")

    report_path = os.path.join(results_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")

    # Save raw JSON results
    json_path = os.path.join(results_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(case_results, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test GPU configurations for TensorFlow and PyTorch training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Orchestrator args
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                        help="Root data directory (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Training epochs per case (default: %(default)s)")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                        help="Tile size in pixels (default: %(default)s)")
    parser.add_argument("--cases", nargs="+",
                        choices=list(TEST_CASES.keys()),
                        default=list(TEST_CASES.keys()),
                        help="Which test cases to run (default: all)")

    # Subprocess modes (used internally, not by the user)
    parser.add_argument("--create-tiles", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--shared-dir", help=argparse.SUPPRESS)

    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--case-name", help=argparse.SUPPRESS)
    parser.add_argument("--framework", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.create_tiles:
        run_create_tiles(args)
    elif args.worker:
        run_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
