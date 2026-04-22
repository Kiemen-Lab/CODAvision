"""
Cross-Framework Epoch Comparison Runner

Thin orchestration script that runs test_epoch_comparison.py for each framework
in sequence, then prints a summary.  The underlying script handles all heavy
lifting (tile creation, training, testing, metrics, plotting) and automatically
generates a cross-framework comparison when results from both frameworks exist.

Supports multiple datasets via --dataset (default: liver).

Usage:
    python scripts/run_cross_framework_comparison.py
    python scripts/run_cross_framework_comparison.py --dataset lungs
    python scripts/run_cross_framework_comparison.py --epochs 2 8 25
    python scripts/run_cross_framework_comparison.py --suffix weight_decay_fix
    python scripts/run_cross_framework_comparison.py --frameworks pytorch
    python scripts/run_cross_framework_comparison.py --data-path D:/data --gpu 1
    python scripts/run_cross_framework_comparison.py --no-early-stopping
    python scripts/run_cross_framework_comparison.py --dataset lungs --epochs 2 8 --frameworks pytorch
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime


DEFAULT_DATA_PATH = None  # resolved from dataset config if not specified
DEFAULT_EPOCH_COUNTS = [2, 8, 25]
DEFAULT_TILE_SIZE = 1024
DEFAULT_BATCH_SIZE = 3
DEFAULT_GPU = "0"
DEFAULT_TIMEOUT = 12 * 60 * 60  # 12 hours per framework

DATASET_DATA_PATHS = {
    "liver": r"C:\Users\tnewton\Desktop\liver_tissue_data",
    "lungs": r"C:\Users\tnewton\Desktop\lungs_data",
}


def _sanitize_suffix(raw: str) -> str:
    """Allow only alphanumeric, underscore, and hyphen; replace spaces."""
    cleaned = raw.strip().replace(" ", "_")
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", cleaned)
    return cleaned


def _format_elapsed(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _build_command(args, framework: str) -> list[str]:
    """Build the subprocess command list for test_epoch_comparison.py."""
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_epoch_comparison.py"
    )
    cmd = [
        sys.executable, script_path,
        "--framework", framework,
        "--dataset", args.dataset,
        "--data-path", args.data_path,
        "--epochs", *[str(e) for e in args.epochs],
        "--tile-size", str(args.tile_size),
        "--batch-size", str(args.batch_size),
        "--gpu", args.gpu,
    ]
    if args.suffix:
        cmd.extend(["--results-suffix", args.suffix])
    if args.no_early_stopping:
        cmd.append("--no-early-stopping")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run epoch comparison across frameworks (sequential)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=list(DATASET_DATA_PATHS.keys()), default="liver",
        help="Dataset configuration to use (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=DEFAULT_EPOCH_COUNTS,
        help="Epoch counts to compare (default: %(default)s)",
    )
    parser.add_argument(
        "--frameworks", nargs="+", default=["pytorch", "tensorflow"],
        choices=["pytorch", "tensorflow"],
        help="Frameworks to run, in order (default: pytorch tensorflow)",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Experiment tag appended to output directory names",
    )
    parser.add_argument(
        "--data-path", default=None,
        help="Root data directory (default: resolved from --dataset)",
    )
    parser.add_argument(
        "--gpu", default=DEFAULT_GPU,
        help="GPU index for CUDA_VISIBLE_DEVICES (default: %(default)s)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help="Tile size in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Training batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--no-early-stopping", action="store_true",
        help="Disable early stopping (set patience=9999)",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help="Per-framework timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    # Sanitize suffix
    args.suffix = _sanitize_suffix(args.suffix)
    sfx = f"_{args.suffix}" if args.suffix else ""

    # Resolve default data path from dataset if not explicitly provided
    if args.data_path is None:
        args.data_path = DATASET_DATA_PATHS[args.dataset]

    # --- Banner ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"Cross-Framework Epoch Comparison — {timestamp}")
    print(f"  Dataset:         {args.dataset}")
    print(f"  Frameworks:      {', '.join(args.frameworks)}")
    print(f"  Epoch counts:    {sorted(args.epochs)}")
    print(f"  Suffix:          {args.suffix or '(none)'}")
    print(f"  Data path:       {args.data_path}")
    print(f"  Tile size:       {args.tile_size}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  GPU:             {args.gpu}")
    print(f"  Early stopping:  {'disabled' if args.no_early_stopping else 'enabled'}")
    print(f"  Timeout:         {_format_elapsed(args.timeout)} per framework")
    print("=" * 70)

    # --- Run each framework ---
    run_results = []  # list of (framework, exit_code, elapsed)

    for i, framework in enumerate(args.frameworks, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(args.frameworks)}] Running {framework.upper()} ...")
        print("=" * 70)

        cmd = _build_command(args, framework)
        print(f"  Command: {' '.join(cmd)}\n")

        t0 = time.time()
        try:
            result = subprocess.run(cmd, timeout=args.timeout)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            print(f"\n[ERROR] {framework} timed out after "
                  f"{_format_elapsed(args.timeout)}")
            exit_code = -1
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] {framework} run interrupted by user")
            exit_code = -2
            run_results.append((framework, exit_code, time.time() - t0))
            break

        elapsed = time.time() - t0
        status = "OK" if exit_code == 0 else f"FAILED (exit {exit_code})"
        print(f"\n[{framework}] {status} in {_format_elapsed(elapsed)}")
        run_results.append((framework, exit_code, elapsed))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Framework':<14} {'Status':<22} {'Elapsed':>10}")
    print(f"  {'-' * 14} {'-' * 22} {'-' * 10}")
    for framework, exit_code, elapsed in run_results:
        if exit_code == 0:
            status = "OK"
        elif exit_code == -1:
            status = "TIMEOUT"
        elif exit_code == -2:
            status = "INTERRUPTED"
        else:
            status = f"FAILED (exit {exit_code})"
        print(f"  {framework:<14} {status:<22} {_format_elapsed(elapsed):>10}")

    # --- Output paths ---
    print(f"\nOutput directories:")
    for framework, _, _ in run_results:
        results_dir = os.path.join(
            args.data_path, f"epoch_comparison_results_{framework}{sfx}"
        )
        exists = os.path.isdir(results_dir)
        marker = "" if exists else " (not created)"
        print(f"  {results_dir}{marker}")

    # Point to cross-framework comparison if it exists
    if len(run_results) >= 2:
        last_fw = run_results[-1][0]
        comp_dir = os.path.join(
            args.data_path, f"epoch_comparison_results_{last_fw}{sfx}"
        )
        comp_txt = os.path.join(comp_dir, "cross_framework_comparison.txt")
        comp_png = os.path.join(comp_dir, "cross_framework_comparison.png")
        if os.path.isfile(comp_txt):
            print(f"\nCross-framework comparison:")
            print(f"  {comp_txt}")
            print(f"  {comp_png}")

    # Exit with non-zero if any framework failed
    any_failed = any(rc != 0 for _, rc, _ in run_results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
