"""
Generate a markdown summary report for a completed (or in-progress)
lr_investigation phase.

Reads ``phase{N}_results.csv`` produced by scripts/lr_investigation.py and
writes ``phase{N}_report.md`` next to it with: per-cell per-condition mean /
std / min / max, paired baseline-vs-lr_fix t-test where applicable, spike
tallies, and a histogram of liver/tf accuracy (ASCII).

Usage:
    python scripts/lr_investigation_report.py phase0
    python scripts/lr_investigation_report.py phase0 --results-dir C:/Users/tnewton/Desktop/lr_investigation/phase0
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional

DEFAULT_RESULTS_ROOT = r"C:\Users\tnewton\Desktop\lr_investigation"


def _load_csv(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _coerce_float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _aggregate(rows: List[Dict]) -> Dict:
    """Return {(cell, condition): {n, mean_acc, std_acc, min_acc, max_acc, ...}}."""
    buckets: Dict[tuple, Dict] = defaultdict(lambda: {
        "accs": [], "train_times": [], "test_times": [],
        "max_val_losses": [], "spikes": [], "successes": 0, "total": 0,
    })
    for row in rows:
        key = (row["cell"], row["condition"])
        b = buckets[key]
        b["total"] += 1
        if row.get("success") == "True":
            b["successes"] += 1
        acc = _coerce_float(row.get("accuracy"))
        if acc is not None:
            b["accs"].append(acc)
        tt = _coerce_float(row.get("train_time_s"))
        if tt is not None:
            b["train_times"].append(tt)
        tst = _coerce_float(row.get("test_time_s"))
        if tst is not None:
            b["test_times"].append(tst)
        mvl = _coerce_float(row.get("max_val_loss"))
        if mvl is not None:
            b["max_val_losses"].append(mvl)
        sp = _coerce_float(row.get("spike_count"))
        if sp is not None:
            b["spikes"].append(int(sp))

    out: Dict[tuple, Dict] = {}
    for key, b in buckets.items():
        accs = b["accs"]
        summary = {
            "n_runs": b["total"],
            "n_success": b["successes"],
            "n_acc_samples": len(accs),
        }
        if accs:
            summary["mean_acc"] = statistics.mean(accs)
            summary["std_acc"] = statistics.stdev(accs) if len(accs) > 1 else 0.0
            summary["min_acc"] = min(accs)
            summary["max_acc"] = max(accs)
        if b["train_times"]:
            summary["mean_train_s"] = statistics.mean(b["train_times"])
        if b["max_val_losses"]:
            summary["mean_max_val_loss"] = statistics.mean(b["max_val_losses"])
            summary["worst_val_loss"] = max(b["max_val_losses"])
        if b["spikes"]:
            summary["total_spikes"] = sum(b["spikes"])
            summary["runs_with_spike"] = sum(1 for s in b["spikes"] if s > 0)
        out[key] = summary
    return out


def _welch_t(a: List[float], b: List[float]) -> Optional[float]:
    """Welch's t-statistic for two samples (no p-value — report t magnitude only)."""
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = statistics.variance(a)
    vb = statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return float("inf") if ma != mb else 0.0
    return (ma - mb) / se


def _histogram(values: List[float], buckets: int = 10, width: int = 40) -> List[str]:
    if not values:
        return ["    (no data)"]
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [f"    {vmin:6.2f}: {'#' * min(len(values), width)} ({len(values)})"]
    step = (vmax - vmin) / buckets
    counts = [0] * buckets
    for v in values:
        idx = min(int((v - vmin) / step), buckets - 1)
        counts[idx] += 1
    lines = []
    cmax = max(counts)
    for i, c in enumerate(counts):
        lo = vmin + i * step
        hi = vmin + (i + 1) * step
        bar = "#" * int(width * c / cmax) if cmax > 0 else ""
        lines.append(f"    [{lo:6.2f}, {hi:6.2f}): {bar} ({c})")
    return lines


def generate_report(phase_name: str, results_dir: str) -> str:
    csv_path = os.path.join(results_dir, f"{phase_name}_results.csv")
    if not os.path.isfile(csv_path):
        return f"# {phase_name} report\n\nNo results CSV yet at {csv_path}\n"

    rows = _load_csv(csv_path)
    agg = _aggregate(rows)

    lines: List[str] = []
    lines.append(f"# {phase_name} report")
    lines.append("")
    lines.append(f"Source: `{csv_path}`")
    lines.append(f"Total rows: {len(rows)}")
    lines.append("")

    # ----- Per-cell per-condition summary table -----
    lines.append("## Summary by cell × condition")
    lines.append("")
    lines.append("| cell | condition | n | success | mean acc | std | min | max | mean train(s) | worst val_loss | runs with spike |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (cell, cond), s in sorted(agg.items()):
        n = s.get("n_acc_samples", 0)
        suc = f"{s.get('n_success', 0)}/{s.get('n_runs', 0)}"
        mean_s = f"{s['mean_acc']:.2f}%" if "mean_acc" in s else "—"
        std_s = f"{s['std_acc']:.2f}" if "std_acc" in s else "—"
        min_s = f"{s['min_acc']:.2f}%" if "min_acc" in s else "—"
        max_s = f"{s['max_acc']:.2f}%" if "max_acc" in s else "—"
        tt_s = f"{s['mean_train_s']:.0f}" if "mean_train_s" in s else "—"
        worst = f"{s['worst_val_loss']:.3f}" if "worst_val_loss" in s else "—"
        rws = f"{s.get('runs_with_spike', 0)}" if "runs_with_spike" in s else "—"
        lines.append(f"| {cell} | {cond} | {n} | {suc} | {mean_s} | {std_s} | {min_s} | {max_s} | {tt_s} | {worst} | {rws} |")
    lines.append("")

    # ----- Paired baseline vs lr_fix comparison per cell -----
    cells = sorted({k[0] for k in agg})
    conditions = sorted({k[1] for k in agg})
    if "baseline" in conditions and "lr_fix" in conditions:
        lines.append("## Baseline vs lr_fix (per cell)")
        lines.append("")
        lines.append("| cell | baseline n | baseline mean | lr_fix n | lr_fix mean | delta | Welch t |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for cell in cells:
            b = agg.get((cell, "baseline"), {})
            lf = agg.get((cell, "lr_fix"), {})
            b_accs = [
                _coerce_float(r["accuracy"])
                for r in rows
                if r["cell"] == cell and r["condition"] == "baseline"
                and _coerce_float(r["accuracy"]) is not None
            ]
            lf_accs = [
                _coerce_float(r["accuracy"])
                for r in rows
                if r["cell"] == cell and r["condition"] == "lr_fix"
                and _coerce_float(r["accuracy"]) is not None
            ]
            bn = len(b_accs)
            lfn = len(lf_accs)
            bmean = f"{statistics.mean(b_accs):.2f}%" if b_accs else "—"
            lfmean = f"{statistics.mean(lf_accs):.2f}%" if lf_accs else "—"
            if b_accs and lf_accs:
                delta = statistics.mean(lf_accs) - statistics.mean(b_accs)
                delta_s = f"{delta:+.2f}%"
            else:
                delta_s = "—"
            t = _welch_t(lf_accs, b_accs)
            t_s = f"{t:.2f}" if t is not None else "—"
            lines.append(f"| {cell} | {bn} | {bmean} | {lfn} | {lfmean} | {delta_s} | {t_s} |")
        lines.append("")

    # ----- liver_tf accuracy histogram -----
    lt_accs_base = [
        _coerce_float(r["accuracy"])
        for r in rows
        if r["cell"] == "liver_tf" and r["condition"] == "baseline"
        and _coerce_float(r["accuracy"]) is not None
    ]
    lt_accs_lrfix = [
        _coerce_float(r["accuracy"])
        for r in rows
        if r["cell"] == "liver_tf" and r["condition"] == "lr_fix"
        and _coerce_float(r["accuracy"]) is not None
    ]
    if lt_accs_base or lt_accs_lrfix:
        lines.append("## liver_tf accuracy distribution")
        lines.append("")
        if lt_accs_base:
            lines.append("### baseline")
            lines.append("```")
            lines.extend(_histogram(lt_accs_base))
            lines.append("```")
            lines.append("")
        if lt_accs_lrfix:
            lines.append("### lr_fix")
            lines.append("```")
            lines.extend(_histogram(lt_accs_lrfix))
            lines.append("```")
            lines.append("")

    # ----- Per-row details -----
    lines.append("## All runs")
    lines.append("")
    lines.append("| cell | condition | seed | success | acc | train_s | spike_count | max_val_loss |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        acc = r.get("accuracy") or "—"
        tt = r.get("train_time_s") or "—"
        sc = r.get("spike_count") or "0"
        mvl = r.get("max_val_loss") or "—"
        lines.append(
            f"| {r['cell']} | {r['condition']} | {r['seed']} | "
            f"{r.get('success','—')} | {acc} | {tt} | {sc} | {mvl} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="lr_investigation phase report")
    parser.add_argument("phase", help="Phase name (e.g. phase0)")
    parser.add_argument("--results-dir", default=None,
                        help="Override results directory (default: "
                             f"{DEFAULT_RESULTS_ROOT}/<phase>/)")
    parser.add_argument("--print", action="store_true",
                        help="Also print the report to stdout")
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(DEFAULT_RESULTS_ROOT, args.phase)
    report = generate_report(args.phase, results_dir)

    out_path = os.path.join(results_dir, f"{args.phase}_report.md")
    os.makedirs(results_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Wrote {out_path}")

    if args.print:
        print()
        print(report)


if __name__ == "__main__":
    main()
