"""
Rebuild the top-level lr_fix_results summary from the per-experiment
worker_results.json files.

The orchestrator overwrites all_results.json / comparison.csv / summary_report.txt
with only the current invocation's experiments. The lr_fix_results tree accumulated
four sub-experiments across multiple orchestrator runs, plus one that crashed and
was re-run directly via the worker. This script merges all four into a single
top-level summary without re-invoking the orchestrator.
"""

import json
import os
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.run_version_comparison import (  # noqa: E402
    _build_experiment_list,
    _collect_confusion_matrices,
    _generate_csv,
    _print_summary,
)

RESULTS_DIR = r"C:\Users\tnewton\Desktop\lr_fix_results\version_comparison_results"


def main() -> None:
    experiments = _build_experiment_list(
        versions=["dsai"],
        datasets=["lungs", "liver"],
        frameworks=["pytorch", "tensorflow"],
        tile_modes=["modern"],
    )

    all_results = {}
    missing = []
    for exp in experiments:
        name = exp["name"]
        path = os.path.join(RESULTS_DIR, name, "worker_results.json")
        if not os.path.isfile(path):
            missing.append(name)
            continue
        with open(path) as f:
            all_results[name] = json.load(f)

    if missing:
        print("Missing worker_results.json for:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote all_results.json ({len(all_results)} experiments)")

    _generate_csv(
        os.path.join(RESULTS_DIR, "comparison.csv"),
        experiments,
        all_results,
    )

    _collect_confusion_matrices(RESULTS_DIR, experiments, all_results)

    _print_summary(
        RESULTS_DIR,
        experiments,
        all_results,
        datasets=["lungs", "liver"],
        timestamp=timestamp,
        gpu="2",
        batch_size=1,
        tile_size=1024,
    )

    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "gpu": "2",
                "batch_size": "1 (pytorch) / 2 (tensorflow)",
                "tile_size": 1024,
                "versions": ["dsai"],
                "datasets": ["lungs", "liver"],
                "experiments": [e["name"] for e in experiments],
                "commit_hashes": {"dsai": "47f53e9"},
                "note": "lr_fix: LEARNING_RATE 1e-4 -> 5e-4, patience-based "
                        "ReduceLROnPlateau, PyTorch lr_factor via "
                        "ModelDefaults.LR_FACTOR (0.75)",
            },
            f,
            indent=2,
        )
    print("Wrote config.json")


if __name__ == "__main__":
    main()
