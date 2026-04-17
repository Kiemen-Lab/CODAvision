"""
Resource monitoring script for CODAvision training sessions.

Samples CPU and GPU usage every 30 seconds for a specified duration,
writing results to a CSV file for later analysis.

Supports multi-GPU systems: by default reports the most active GPU,
or use --gpu N to pin to a specific GPU index.
"""

import csv
import datetime
import os
import subprocess
import sys
import time

import psutil


def get_gpu_stats():
    """Query nvidia-smi for GPU utilization, memory, and temperature for ALL GPUs."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                print(
                    f"Warning: unexpected nvidia-smi output (expected 8 fields, got {len(parts)}): {line}",
                    file=sys.stderr,
                )
                continue
            gpus.append({
                "gpu_index": int(parts[0]),
                "gpu_util_pct": float(parts[1]),
                "gpu_mem_util_pct": float(parts[2]),
                "gpu_mem_used_mb": float(parts[3]),
                "gpu_mem_total_mb": float(parts[4]),
                "gpu_temp_c": float(parts[5]),
                "gpu_power_w": float(parts[6]) if parts[6] != "[N/A]" else 0.0,
                "gpu_name": parts[7],
            })
        return gpus
    except Exception as e:
        print(f"GPU query error: {e}", file=sys.stderr)
        return []


def get_cpu_stats():
    """Get CPU utilization and memory usage."""
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    return {
        "cpu_util_pct": cpu_pct,
        "ram_used_gb": mem.used / (1024 ** 3),
        "ram_total_gb": mem.total / (1024 ** 3),
        "ram_pct": mem.percent,
    }


def monitor(duration_minutes=60, interval_seconds=30, output_path=None, gpu_index=None):
    """Run the monitoring loop.

    Args:
        duration_minutes: How long to monitor.
        interval_seconds: Time between samples.
        output_path: CSV output path (auto-generated if None).
        gpu_index: GPU index to monitor (None = auto-select most active).
    """
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            f"resource_monitor_{timestamp}.csv",
        )
        output_path = os.path.abspath(output_path)

    fieldnames = [
        "timestamp",
        "elapsed_min",
        "cpu_util_pct",
        "ram_used_gb",
        "ram_total_gb",
        "ram_pct",
        "gpu_index",
        "gpu_count",
        "gpu_util_pct",
        "gpu_mem_util_pct",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
        "gpu_temp_c",
        "gpu_power_w",
        "gpu_name",
    ]

    total_samples = int((duration_minutes * 60) / interval_seconds)
    start_time = time.time()

    # Prime the CPU percent counter (first call always returns 0)
    psutil.cpu_percent(interval=None)

    gpu_mode = f"GPU {gpu_index}" if gpu_index is not None else "auto (most active)"
    print(f"Monitoring for {duration_minutes} minutes ({total_samples} samples)")
    print(f"Sampling every {interval_seconds} seconds")
    print(f"GPU selection: {gpu_mode}")
    print(f"Output: {output_path}")
    print("-" * 70)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(total_samples):
            now = datetime.datetime.now()
            elapsed = (time.time() - start_time) / 60.0

            row = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_min": round(elapsed, 2),
            }

            cpu = get_cpu_stats()
            row.update(cpu)

            gpus = get_gpu_stats()
            gpu = None
            if gpus:
                row["gpu_count"] = len(gpus)
                if gpu_index is not None:
                    # Pinned to a specific GPU
                    matches = [g for g in gpus if g["gpu_index"] == gpu_index]
                    if matches:
                        gpu = matches[0]
                    else:
                        print(
                            f"Warning: GPU {gpu_index} not found (available: "
                            f"{[g['gpu_index'] for g in gpus]})",
                            file=sys.stderr,
                        )
                else:
                    # Auto-select the GPU with highest utilization
                    gpu = max(gpus, key=lambda g: g["gpu_util_pct"])

            if gpu:
                row.update(gpu)
            else:
                row["gpu_count"] = row.get("gpu_count", 0)
                for k in fieldnames:
                    if k.startswith("gpu_") and k not in row:
                        row[k] = ""

            writer.writerow(row)
            f.flush()

            # Print a compact status line
            if gpu:
                print(
                    f"[{now.strftime('%H:%M:%S')}] "
                    f"CPU {cpu['cpu_util_pct']:5.1f}%  "
                    f"RAM {cpu['ram_used_gb']:.1f}/{cpu['ram_total_gb']:.0f} GB  "
                    f"GPU{gpu['gpu_index']} {gpu['gpu_util_pct']:5.1f}%  "
                    f"VRAM {gpu['gpu_mem_used_mb']:.0f}/{gpu['gpu_mem_total_mb']:.0f} MB  "
                    f"({len(gpus)} GPUs)",
                    flush=True,
                )
            else:
                print(
                    f"[{now.strftime('%H:%M:%S')}] "
                    f"CPU {cpu['cpu_util_pct']:5.1f}%  "
                    f"RAM {cpu['ram_used_gb']:.1f}/{cpu['ram_total_gb']:.0f} GB  "
                    f"GPU  N/A",
                    flush=True,
                )

            if i < total_samples - 1:
                time.sleep(interval_seconds)

    print("-" * 70)
    print(f"Monitoring complete. Data saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor CPU and GPU resources")
    parser.add_argument(
        "duration", type=int, nargs="?", default=60,
        help="Duration in minutes (default: 60)",
    )
    parser.add_argument(
        "interval", type=int, nargs="?", default=30,
        help="Sample interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index to monitor (default: auto-select most active)",
    )
    args = parser.parse_args()
    monitor(
        duration_minutes=args.duration,
        interval_seconds=args.interval,
        gpu_index=args.gpu,
    )
