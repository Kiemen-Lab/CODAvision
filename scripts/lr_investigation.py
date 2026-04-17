"""
lr_investigation — phased orchestrator for the liver/tf regression study.

Drives `scripts/run_version_comparison.py --worker` across a matrix of
``(cell, condition, seed)`` runs with deterministic seeding, tile reuse,
and per-phase CSV rollup. Each condition is represented by a git worktree
containing a specific code state, so interventions are checked out not
imported.

Usage:

    # Phase 0 (variance characterization — 40 runs, ~43 GPU-hours)
    python scripts/lr_investigation.py phase0 --gpu 2
    python scripts/lr_investigation.py phase0 --gpu 2 --seeds 1 2
    python scripts/lr_investigation.py phase0 --gpu 2 --cells liver_tf --seeds 1

The orchestrator is idempotent: if ``{run_dir}/worker_results.json`` already
exists for a run, the worker is skipped and the existing result is loaded
for the summary CSV. Delete the run_dir to force a re-run.

Tiles are created once per cell in ``{results_root}/{cell}/_tiles/`` and
junctioned into every run directory for that cell via ``mklink /J`` (Windows).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

RESULTS_ROOT = r"C:\Users\tnewton\Desktop\lr_investigation"

# Where each "condition" lives as a git worktree. Must be set up BEFORE
# running a phase: see setup_worktrees() below for the expected layout.
CONDITION_WORKTREES: Dict[str, str] = {
    # Phase 0
    "baseline": r"C:\Users\tnewton\git\CODAvision_phase0_baseline",
    "lr_fix":   r"C:\Users\tnewton\git\CODAvision_phase0_lr_fix",
    # Phase 5
    "clipping": r"C:\Users\tnewton\git\CODAvision_phase5_clipping",
    # Phase 6 (PyTorch framework-parity fixes, on top of DSAI/clipping)
    "fix1":     r"C:\Users\tnewton\git\CODAvision_phase6_fix1",
    "fix2":     r"C:\Users\tnewton\git\CODAvision_phase6_fix2",
    "fix12":    r"C:\Users\tnewton\git\CODAvision_phase6_fix12",
}

CONDITION_BRANCHES: Dict[str, str] = {
    "baseline": "phase0_baseline",
    "lr_fix":   "phase0_lr_fix",
    "clipping": "phase5_clipping",
    "fix1":     "phase6_fix1",
    "fix2":     "phase6_fix2",
    "fix12":    "phase6_fix12",
}

# Cell definitions: (dataset, framework, tile_mode, batch_size, data_path)
# Batch sizes match the ones used in the original lr_fix_results run so that
# Phase 0 reproduces the exact training conditions that produced the 82.5%
# liver/tf result.
CELLS: Dict[str, Dict] = {
    "lungs_pt": {
        "dataset": "lungs", "framework": "pytorch", "tile_mode": "modern",
        "batch_size": 3, "tile_size": 1024,
        "data_path": r"C:\Users\tnewton\Desktop\lungs_data",
    },
    "lungs_tf": {
        "dataset": "lungs", "framework": "tensorflow", "tile_mode": "modern",
        "batch_size": 3, "tile_size": 1024,
        "data_path": r"C:\Users\tnewton\Desktop\lungs_data",
    },
    "liver_pt": {
        "dataset": "liver", "framework": "pytorch", "tile_mode": "modern",
        "batch_size": 1, "tile_size": 1024,
        "data_path": r"C:\Users\tnewton\Desktop\liver_tissue_data",
    },
    "liver_tf": {
        "dataset": "liver", "framework": "tensorflow", "tile_mode": "modern",
        "batch_size": 2, "tile_size": 1024,
        "data_path": r"C:\Users\tnewton\Desktop\liver_tissue_data",
    },
}

PHASES: Dict[str, Dict] = {
    "phase0": {
        "description": "Variance characterization — baseline vs lr_fix × 4 cells × 5 seeds",
        "cells": list(CELLS.keys()),
        "conditions": ["baseline", "lr_fix"],
        "seeds": [1, 2, 3, 4, 5],
        "bn_diagnostics": False,
    },
    "phase5": {
        "description": "Gradient clipping validation — 4 cells × 3 seeds × clipping",
        "cells": list(CELLS.keys()),
        "conditions": ["clipping"],
        "seeds": [1, 2, 3],
        "bn_diagnostics": False,
    },
    "phase6": {
        "description": "PyTorch framework-parity fixes — 2 PT cells × 3 seeds × {fix1, fix2, fix12}",
        "cells": ["lungs_pt", "liver_pt"],
        "conditions": ["fix1", "fix2", "fix12"],
        "seeds": [1, 2, 3],
        "bn_diagnostics": False,
    },
}


# ===================================================================
# Worktree management
# ===================================================================

def _worktree_exists(path: str) -> bool:
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, ".git")) or (
        os.path.isdir(path) and os.path.isfile(os.path.join(path, ".git"))
    )


def ensure_worktrees(conditions: List[str]) -> None:
    """Create any missing git worktrees for the conditions used by this phase."""
    for cond in conditions:
        if cond not in CONDITION_WORKTREES:
            print(f"ERROR: condition {cond!r} has no worktree path defined")
            sys.exit(1)
        wt = CONDITION_WORKTREES[cond]
        branch = CONDITION_BRANCHES[cond]
        if _worktree_exists(wt):
            print(f"  worktree exists: {wt}")
            continue
        print(f"  creating worktree: {wt}  (branch {branch})")
        subprocess.run(
            ["git", "-C", _PROJECT_ROOT, "worktree", "add", wt, branch],
            check=True, timeout=120,
        )


# ===================================================================
# Tile reuse
# ===================================================================

def _link_dir(src: str, dst: str) -> None:
    """Create a Windows directory junction. Raises on failure."""
    if os.path.exists(dst):
        return
    subprocess.check_call(
        ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _copy_file(src: str, dst: str) -> None:
    if os.path.isfile(src) and not os.path.isfile(dst):
        shutil.copy2(src, dst)


# ===================================================================
# Run execution
# ===================================================================

def _run_dir(phase_dir: str, cell_name: str, seed: int, condition: str) -> str:
    return os.path.join(phase_dir, cell_name, f"seed{seed}", condition)


def _tile_source_marker(phase_dir: str, cell_name: str) -> str:
    """Path to a marker file that records which run directory owns the canonical tiles for this cell."""
    return os.path.join(phase_dir, cell_name, "_tile_source.txt")


def _read_tile_source(phase_dir: str, cell_name: str) -> Optional[str]:
    marker = _tile_source_marker(phase_dir, cell_name)
    if not os.path.isfile(marker):
        return None
    with open(marker) as f:
        return f.read().strip() or None


def _write_tile_source(phase_dir: str, cell_name: str, source: str) -> None:
    marker = _tile_source_marker(phase_dir, cell_name)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write(source)


def launch_worker(
    phase_dir: str,
    cell_name: str,
    cell: Dict,
    seed: int,
    condition: str,
    gpu: str,
    bn_diagnostics: bool,
) -> Dict:
    """
    Launch a single worker run. Handles tile reuse: the first run of each
    cell creates tiles fresh; subsequent runs of the same cell junction
    training/ and validation/ from the first run's directory.

    Returns the parsed worker_results.json or a dict with 'error' set.
    """
    run_dir = _run_dir(phase_dir, cell_name, seed, condition)
    results_path = os.path.join(run_dir, "worker_results.json")

    # Idempotence: skip if already completed
    if os.path.isfile(results_path):
        print(f"  SKIP (exists): {run_dir}")
        with open(results_path) as f:
            return json.load(f)

    worktree = CONDITION_WORKTREES[condition]
    if not os.path.isdir(worktree):
        raise RuntimeError(
            f"Worktree missing for condition {condition!r}: {worktree}\n"
            f"Run ensure_worktrees() first."
        )

    os.makedirs(run_dir, exist_ok=True)

    # Tile sharing: determine whether we need to create tiles or link them.
    tile_source = _read_tile_source(phase_dir, cell_name)
    create_tiles = False
    if tile_source is None:
        # First run for this cell — this run creates tiles fresh
        create_tiles = True
        _write_tile_source(phase_dir, cell_name, run_dir)
        print(f"  first run for {cell_name}; creating tiles in {run_dir}")
    else:
        # Junction tiles from the canonical source
        if os.path.abspath(tile_source) != os.path.abspath(run_dir):
            for sub in ("training", "validation"):
                src = os.path.join(tile_source, sub)
                dst = os.path.join(run_dir, sub)
                if os.path.isdir(src):
                    _link_dir(src, dst)
            for fname in ("annotations.pkl", "train_list.pkl"):
                _copy_file(
                    os.path.join(tile_source, fname),
                    os.path.join(run_dir, fname),
                )

    # Build the subprocess command
    script_path = os.path.join(worktree, "scripts", "run_version_comparison.py")
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["TF_CUDNN_USE_AUTOTUNE"] = "0"
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = worktree
    env["PYTHONHASHSEED"] = str(seed)
    if bn_diagnostics:
        env["BN_DIAGNOSTICS"] = "1"

    cmd = [
        sys.executable, script_path,
        "--worker",
        "--experiment-dir", run_dir,
        "--worktree", worktree,
        "--dataset", cell["dataset"],
        "--data-path", cell["data_path"],
        "--framework", cell["framework"],
        "--tile-mode", cell["tile_mode"],
        "--batch-size", str(cell["batch_size"]),
        "--tile-size", str(cell["tile_size"]),
        "--version-name", "dsai",
        "--commit", CONDITION_BRANCHES[condition],
        "--exp-name", f"{cell_name}_{condition}_seed{seed}",
        "--seed", str(seed),
    ]
    if create_tiles:
        cmd.append("--create-tiles")

    stdout_log = os.path.join(run_dir, "worker_stdout.log")
    stderr_log = os.path.join(run_dir, "worker_stderr.log")

    t0 = time.time()
    print(f"  LAUNCH {cell_name} seed={seed} cond={condition}")
    print(f"         worktree={worktree}")
    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.run(cmd, env=env, stdout=out, stderr=err)
    wall = round(time.time() - t0, 1)

    if os.path.isfile(results_path):
        with open(results_path) as f:
            result = json.load(f)
        result["wall_time_s"] = wall
        result["exit_code"] = proc.returncode
    else:
        result = {
            "success": False,
            "error": f"No worker_results.json (exit {proc.returncode})",
            "wall_time_s": wall,
            "exit_code": proc.returncode,
        }

    status = "OK" if result.get("success") else "FAIL"
    acc = result.get("testing", {}).get("overall_accuracy", "N/A")
    print(f"  [{status}] {cell_name} seed={seed} {condition}: acc={acc}% ({wall}s)")

    return result


# ===================================================================
# CSV rollup
# ===================================================================

CSV_COLS = [
    "phase", "cell", "condition", "seed", "success",
    "accuracy", "train_time_s", "test_time_s", "epochs_completed",
    "final_val_acc", "max_val_loss", "spike_count",
    "wall_time_s", "exit_code", "run_dir",
]


def _summarize_result(phase: str, cell_name: str, condition: str, seed: int,
                      run_dir: str, result: Dict) -> Dict:
    testing = result.get("testing", {}) or {}
    training = result.get("training", {}) or {}
    history = training.get("history", {}) or {}
    val_loss_series = history.get("val_loss", []) or []
    val_acc_series = history.get("val_accuracy", []) or []
    max_vl = max(val_loss_series) if val_loss_series else None
    spike_count = sum(1 for v in val_loss_series if v > 1.0)
    final_va = val_acc_series[-1] if val_acc_series else None

    return {
        "phase": phase,
        "cell": cell_name,
        "condition": condition,
        "seed": seed,
        "success": result.get("success", False),
        "accuracy": testing.get("overall_accuracy"),
        "train_time_s": training.get("train_time_s"),
        "test_time_s": testing.get("test_time_s"),
        "epochs_completed": training.get("epochs_completed"),
        "final_val_acc": final_va,
        "max_val_loss": max_vl,
        "spike_count": spike_count,
        "wall_time_s": result.get("wall_time_s"),
        "exit_code": result.get("exit_code"),
        "run_dir": run_dir,
    }


def append_csv_row(csv_path: str, row: Dict) -> None:
    existed = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if not existed:
            w.writeheader()
        w.writerow(row)


# ===================================================================
# Phase driver
# ===================================================================

def run_phase(
    phase_name: str,
    gpu: str,
    cell_filter: Optional[List[str]] = None,
    seed_filter: Optional[List[int]] = None,
    condition_filter: Optional[List[str]] = None,
) -> None:
    if phase_name not in PHASES:
        print(f"ERROR: unknown phase {phase_name!r}; known: {list(PHASES.keys())}")
        sys.exit(1)

    phase = PHASES[phase_name]
    cells = cell_filter or phase["cells"]
    conditions = condition_filter or phase["conditions"]
    seeds = seed_filter or phase["seeds"]
    bn_diagnostics = phase.get("bn_diagnostics", False)

    phase_dir = os.path.join(RESULTS_ROOT, phase_name)
    os.makedirs(phase_dir, exist_ok=True)
    csv_path = os.path.join(phase_dir, f"{phase_name}_results.csv")

    print("=" * 70)
    print(f"{phase_name}: {phase['description']}")
    print(f"  output:      {phase_dir}")
    print(f"  csv:         {csv_path}")
    print(f"  gpu:         {gpu}")
    print(f"  cells:       {cells}")
    print(f"  conditions:  {conditions}")
    print(f"  seeds:       {seeds}")
    print(f"  bn_diag:     {bn_diagnostics}")
    print("=" * 70)

    ensure_worktrees(conditions)

    # Build the run list — order matters for tile sharing: all runs of the
    # same cell should be contiguous so the tile-source marker logic works.
    total_runs = 0
    for cell_name in cells:
        cell = CELLS[cell_name]
        print(f"\n--- cell: {cell_name} ({cell['dataset']}/{cell['framework']}/bs={cell['batch_size']}) ---")
        for seed in seeds:
            for condition in conditions:
                total_runs += 1
                print(f"\n[{total_runs}] {phase_name} {cell_name} seed={seed} {condition}")
                try:
                    result = launch_worker(
                        phase_dir=phase_dir,
                        cell_name=cell_name,
                        cell=cell,
                        seed=seed,
                        condition=condition,
                        gpu=gpu,
                        bn_diagnostics=bn_diagnostics,
                    )
                except Exception as exc:
                    print(f"  EXCEPTION: {exc}")
                    result = {"success": False, "error": str(exc)}

                row = _summarize_result(
                    phase=phase_name,
                    cell_name=cell_name,
                    condition=condition,
                    seed=seed,
                    run_dir=_run_dir(phase_dir, cell_name, seed, condition),
                    result=result,
                )
                append_csv_row(csv_path, row)

    print("\n" + "=" * 70)
    print(f"{phase_name} complete. Results: {csv_path}")
    print("=" * 70)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="lr_investigation phased orchestrator"
    )
    parser.add_argument("phase", choices=list(PHASES.keys()),
                        help="Which phase to run")
    parser.add_argument("--gpu", type=str, default="2",
                        help="GPU index (default: 2 = RTX 6000)")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Cell filter (default: all cells in phase)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seed filter (default: all seeds in phase)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Condition filter (default: all conditions in phase)")

    args = parser.parse_args()
    run_phase(
        phase_name=args.phase,
        gpu=args.gpu,
        cell_filter=args.cells,
        seed_filter=args.seeds,
        condition_filter=args.conditions,
    )


if __name__ == "__main__":
    main()
