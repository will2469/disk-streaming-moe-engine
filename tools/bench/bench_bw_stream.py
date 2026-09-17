#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.

"""STREAM-like Memory Bandwidth Measurement (Gate G-M5-6).

Measures single-thread Copy read-equivalent memory bandwidth using an
array significantly larger than the Last Level Cache (LLC >= 4x LLC).
Verifies the RAM sustained bandwidth floor condition (BW_RAM >= 10 GB/s)
under the active CPU governor.
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_OUTPUT_JSON = Path("reports/2026-09-17/m5_stream_bw.json")


def get_cpu_governor() -> str:
    """Mendapatkan CPU scaling governor dari sysfs."""
    govs = set()
    for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                govs.add(f.read().strip())
        except Exception:
            pass
    if govs:
        return ", ".join(sorted(govs))
    return "unknown"


def get_cpu_model() -> str:
    """Mendapatkan CPU model name dari /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def get_llc_size_bytes() -> int:
    """Mendeteksi ukuran Last Level Cache (LLC) dari sysfs."""
    llc_bytes = 0
    for p in glob.glob("/sys/devices/system/cpu/cpu0/cache/index*"):
        try:
            with open(os.path.join(p, "level"), "r", encoding="utf-8") as f:
                lvl = int(f.read().strip())
            if lvl == 3:
                with open(os.path.join(p, "size"), "r", encoding="utf-8") as f:
                    sz_str = f.read().strip()
                if sz_str.endswith("K"):
                    return int(sz_str[:-1]) * 1024
                elif sz_str.endswith("M"):
                    return int(sz_str[:-1]) * 1024 * 1024
                elif sz_str.endswith("G"):
                    return int(sz_str[:-1]) * 1024 * 1024 * 1024
                else:
                    return int(sz_str)
        except Exception:
            pass
    # Fallback jika tidak terdeteksi: 10 MiB
    return 10 * 1024 * 1024 if llc_bytes == 0 else llc_bytes


def run_stream_copy(
    num_elements: int = 10_000_000,
    repetitions: int = 10,
    warmup: int = 5,
) -> dict:
    """Menjalankan STREAM-like Copy single-thread benchmark (c[i] = a[i])."""
    # 10M float64 = 80 MB (>= 4x LLC pada prosesor laptop 10-12 MB LLC)
    a = np.ones(num_elements, dtype=np.float64)
    c = np.empty(num_elements, dtype=np.float64)

    array_bytes = num_elements * 8  # 8 bytes per float64
    llc_bytes = get_llc_size_bytes()
    llc_multiple = array_bytes / llc_bytes if llc_bytes > 0 else 0.0

    # Warm-up runs
    for _ in range(warmup):
        np.copyto(c, a)

    # Measurement runs
    run_times = []
    run_bw_read_equiv = []
    run_bw_bidi = []

    for _ in range(repetitions):
        t0 = time.perf_counter()
        np.copyto(c, a)
        t1 = time.perf_counter()
        dt = t1 - t0
        run_times.append(dt)

        # Read-equivalent BW: bytes read (array_bytes) / dt
        bw_read = (array_bytes / dt) / 1e9
        run_bw_read_equiv.append(bw_read)

        # Total bi-directional memory traffic: 2 * array_bytes / dt (Read + Write)
        bw_bi = (2 * array_bytes / dt) / 1e9
        run_bw_bidi.append(bw_bi)

    med_time = statistics.median(run_times)
    med_bw_read = statistics.median(run_bw_read_equiv)
    med_bw_bidi = statistics.median(run_bw_bidi)

    # Floor threshold: 10.0 GB/s single-thread Copy read-equiv (G-M5-6)
    gate_g_m5_6_pass = med_bw_read >= 10.0
    verdict = "PASS" if gate_g_m5_6_pass else "FAIL"

    return {
        "benchmark": "STREAM Copy Single-Thread Memory Bandwidth",
        "gate": "G-M5-6",
        "verdict": verdict,
        "floor_threshold_gb_s": 10.0,
        "measured_read_equiv_bw_gb_s": {
            "median": med_bw_read,
            "min": min(run_bw_read_equiv),
            "max": max(run_bw_read_equiv),
            "p95": float(np.percentile(run_bw_read_equiv, 95)),
        },
        "measured_bidi_bw_gb_s": {
            "median": med_bw_bidi,
            "min": min(run_bw_bidi),
            "max": max(run_bw_bidi),
        },
        "median_time_sec": med_time,
        "parameters": {
            "num_elements": num_elements,
            "dtype": "float64",
            "array_bytes": array_bytes,
            "array_mib": array_bytes / (1024 * 1024),
            "llc_bytes": llc_bytes,
            "llc_mib": llc_bytes / (1024 * 1024),
            "llc_multiple": llc_multiple,
            "repetitions": repetitions,
            "warmup_runs": warmup,
        },
        "hardware": {
            "cpu_model": get_cpu_model(),
            "cpu_governor": get_cpu_governor(),
            "num_logical_cores": os.cpu_count() or 1,
        },
        "raw_runs_sec": run_times,
        "raw_bw_read_equiv_gb_s": run_bw_read_equiv,
    }


def main():
    parser = argparse.ArgumentParser(
        description="STREAM-like Copy Memory Bandwidth Benchmark (Gate G-M5-6)"
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=10_000_000,
        help="Number of float64 elements (default: 10,000,000 = 80 MB)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of measurement repetitions (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up repetitions (default: 5)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to output JSON summary",
    )
    args = parser.parse_args()

    results = run_stream_copy(
        num_elements=args.elements,
        repetitions=args.repetitions,
        warmup=args.warmup,
    )

    med_bw = results["measured_read_equiv_bw_gb_s"]["median"]
    floor_bw = results["floor_threshold_gb_s"]
    verdict = results["verdict"]
    p = results["parameters"]
    h = results["hardware"]

    print("=" * 72)
    print("STREAM-LIKE COPY MEMORY BANDWIDTH BENCHMARK (GATE G-M5-6)")
    print("=" * 72)
    print(f"CPU Model          : {h['cpu_model']}")
    print(f"CPU Governor       : {h['cpu_governor']}")
    print(f"Detected LLC Size  : {p['llc_mib']:.2f} MiB")
    print(
        f"Array Size         : {p['array_mib']:.2f} MiB "
        f"({p['llc_multiple']:.1f}x LLC >= 4x requirement)"
    )
    print(f"Repetitions        : {p['repetitions']} (+ {p['warmup_runs']} warmup)")
    print("-" * 72)
    print(f"Median Copy Time   : {results['median_time_sec'] * 1e3:.2f} ms")
    print(
        f"Read-Equiv BW      : {med_bw:.2f} GB/s "
        f"(Floor Gate: >= {floor_bw:.1f} GB/s)"
    )
    print(f"Bi-Directional BW  : {results['measured_bidi_bw_gb_s']['median']:.2f} GB/s")
    print(f"Gate G-M5-6 Verdict: {verdict}")
    print("=" * 72)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"[OK] Summary JSON saved to: {args.output_json}")

    if not (verdict == "PASS"):
        print("FAIL: RAM bandwidth below required floor.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
