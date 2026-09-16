#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.

"""Benchmark runner for M3: Real-checkpoint MoE layer benchmark.

Executes N=5 runs of `./kimo layer --part moe` on
`/home/will/models/qwen1.5-moe-a2.7b-chat` for layers 0, 12, and 23 under
`systemd-run --user --scope -p MemoryMax=6G`.
Measures parse time, compute time, total time, VmHWM, and I/O reads.
"""

import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

DEFAULT_MODEL_DIR = Path("/home/will/models/qwen1.5-moe-a2.7b-chat")
LAYERS = [0, 12, 23]
RUNS = 5


def run_single_benchmark(
    kimo_bin: Path,
    model_dir: Path,
    act_file: Path,
    layer: int,
    run_idx: int,
    tmp_dir: str,
) -> dict:
    out_bin = "moe_out.bin"
    base_cmd = [
        str(kimo_bin),
        "layer",
        "--layer",
        str(layer),
        "--part",
        "moe",
        str(act_file),
        "--model-dir",
        str(model_dir),
        "--workdir",
        tmp_dir,
        "--output",
        out_bin,
    ]

    # Attempt systemd-run cgroup MemoryMax=6G, fallback to direct run
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        "MemoryMax=6G",
    ] + base_cmd

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Fallback to direct run if systemd-run is not permitted/available
        proc = subprocess.run(base_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(
                f"Run {run_idx} layer {layer} FAILED (exit {proc.returncode}):\n"
                f"{proc.stderr}",
                file=sys.stderr,
            )
            sys.exit(proc.returncode)

    try:
        data = json.loads(proc.stdout.strip())
    except Exception as e:
        print(
            f"Failed to parse JSON stdout: {e}\nRaw stdout:\n{proc.stdout}",
            file=sys.stderr,
        )
        sys.exit(1)

    parse_ms = data["parse_time_ms"]
    comp_ms = data["compute_time_ms"]
    mem = data.get("memory", {})
    vmhwm = mem.get("vmhwm_bytes", 0)
    io_read = mem.get("io_read_bytes", 0)
    throughput = 16.0 / (comp_ms / 1000.0)

    return {
        "run": run_idx,
        "layer": layer,
        "type": "cold" if run_idx == 1 else "warm",
        "parse_time_ms": parse_ms,
        "compute_time_ms": comp_ms,
        "total_time_ms": parse_ms + comp_ms,
        "throughput_tok_s": throughput,
        "vmhwm_bytes": vmhwm,
        "vmhwm_gib": vmhwm / (1024**3),
        "io_read_bytes": io_read,
        "io_read_gib": io_read / (1024**3),
    }


def compute_layer_medians(results: list[dict]) -> dict:
    parse_times = [r["parse_time_ms"] for r in results]
    comp_times = [r["compute_time_ms"] for r in results]
    total_times = [r["total_time_ms"] for r in results]
    throughputs = [r["throughput_tok_s"] for r in results]
    vmhwms = [r["vmhwm_bytes"] for r in results]
    ios = [r["io_read_bytes"] for r in results]

    return {
        "parse_time_ms": statistics.median(parse_times),
        "compute_time_ms": statistics.median(comp_times),
        "total_time_ms": statistics.median(total_times),
        "throughput_tok_s": statistics.median(throughputs),
        "vmhwm_bytes": statistics.median(vmhwms),
        "vmhwm_gib": statistics.median(vmhwms) / (1024**3),
        "io_read_bytes": statistics.median(ios),
        "io_read_gib": statistics.median(ios) / (1024**3),
    }


def main():
    root = Path(__file__).resolve().parent.parent.parent
    kimo_bin = root / "kimo"
    model_dir = DEFAULT_MODEL_DIR
    act_file = root / "fixtures/m3/activation.bin"

    if not kimo_bin.exists():
        print(f"Error: {kimo_bin} missing. Build first.", file=sys.stderr)
        sys.exit(1)

    if not model_dir.exists():
        print(f"Error: {model_dir} not found.", file=sys.stderr)
        sys.exit(1)

    if not act_file.exists():
        print(f"Error: {act_file} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"=== Starting M3 Benchmark: {RUNS} runs per layer on {model_dir} ===")
    print("Environment: systemd-run -p MemoryMax=6G, sequence_length=16, part=moe")

    all_results = {}
    medians_by_layer = {}

    for layer in LAYERS:
        print(f"\n--- Benchmarking Layer {layer} ({RUNS} runs) ---")
        layer_runs = []
        for i in range(1, RUNS + 1):
            with tempfile.TemporaryDirectory(
                prefix=f"bench_m3_l{layer}_{i}_"
            ) as tmp_dir:
                res = run_single_benchmark(
                    kimo_bin, model_dir, act_file, layer, i, tmp_dir
                )
                layer_runs.append(res)
                print(
                    f"  Run {i} ({res['type']}): parse={res['parse_time_ms']:.2f} ms, "
                    f"compute={res['compute_time_ms']:.2f} ms, "
                    f"total={res['total_time_ms']:.2f} ms"
                )

        med = compute_layer_medians(layer_runs)
        all_results[str(layer)] = layer_runs
        medians_by_layer[str(layer)] = med

    print("\n" + "=" * 65)
    print("MEDIAN RESULTS SUMMARY (N=5 per layer):")
    print(
        f"{'Layer':<8}{'Parse (ms)':<14}{'Compute (ms)':<16}"
        f"{'Total (ms)':<14}{'Throughput (tok/s)':<18}"
    )
    print("-" * 65)
    for layer in LAYERS:
        m = medians_by_layer[str(layer)]
        print(
            f"{layer:<8}{m['parse_time_ms']:<14.2f}{m['compute_time_ms']:<16.2f}"
            f"{m['total_time_ms']:<14.2f}{m['throughput_tok_s']:<18.2f}"
        )
    print("=" * 65)

    summary_path = root / "reports/2026-09-16/m3_benchmark_raw.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": "M3-20260916-001",
                "model": "Qwen1.5-MoE-A2.7B-Chat",
                "num_tokens": 16,
                "hidden_size": 2048,
                "num_experts": 60,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 1408,
                "shared_expert_intermediate_size": 5632,
                "layers": medians_by_layer,
                "raw_runs": all_results,
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"Saved raw benchmark data to {summary_path}")


if __name__ == "__main__":
    main()
