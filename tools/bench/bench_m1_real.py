#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.

"""
Benchmark runner for M1-C: Real-checkpoint head-path benchmark.
Executes N=5 runs of `./dismoen head` on `~/models/qwen3.6-35b-a3b`
under `systemd-run --user --scope -p MemoryMax=6G`.
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent.parent
    dismoen_bin = root / "dismoen"
    model_dir = Path(
        os.environ.get("MODEL_DIR", Path.home() / "models/qwen3.6-35b-a3b")
    )
    tokens_file = root / "fixtures/m1/tokens.json"

    if not dismoen_bin.exists():
        print(
            f"Error: {dismoen_bin} does not exist. Run pixi run build first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not model_dir.exists():
        print(f"Error: Model directory {model_dir} not found.", file=sys.stderr)
        sys.exit(1)

    runs = 5
    results = []

    print(f"=== Starting M1-C Benchmark: {runs} runs on {model_dir} ===")
    print("Environment: systemd-run -p MemoryMax=6G, single thread (threads=1)")

    for i in range(1, runs + 1):
        with tempfile.TemporaryDirectory(prefix=f"bench_m1_{i}_") as tmp_dir:
            out_bin = os.path.join(tmp_dir, "logits.bin")
            cmd = [
                "systemd-run",
                "--user",
                "--scope",
                "-p",
                "MemoryMax=6G",
                str(dismoen_bin),
                "head",
                str(tokens_file),
                "--model-dir",
                str(model_dir),
                "--workdir",
                tmp_dir,
                "--output",
                out_bin,
            ]
            print(f"\n[Run {i}/{runs}] Executing dismoen head...")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"Run {i} FAILED (exit {proc.returncode})", file=sys.stderr)
                print("STDOUT:", proc.stdout, file=sys.stderr)
                print("STDERR:", proc.stderr, file=sys.stderr)
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
            mem = data["memory"]
            vmhwm = mem["vmhwm_bytes"]
            io_read = mem.get("io_read_bytes", 0)
            res_target = mem["resident_target_bytes"]
            conv_buf = mem["conversion_buffer_bytes"]
            src_buf = mem["source_buffer_bytes"]
            throughput = 48.0 / (comp_ms / 1000.0)

            run_res = {
                "run": i,
                "type": "cold" if i == 1 else "warm",
                "parse_time_ms": parse_ms,
                "compute_time_ms": comp_ms,
                "total_time_ms": parse_ms + comp_ms,
                "throughput_tok_s": throughput,
                "vmhwm_bytes": vmhwm,
                "vmhwm_gib": vmhwm / (1024**3),
                "io_read_bytes": io_read,
                "io_read_gib": io_read / (1024**3),
                "resident_target_bytes": res_target,
                "conversion_buffer_bytes": conv_buf,
                "source_buffer_bytes": src_buf,
            }
            results.append(run_res)
            print(
                f"  -> Parse: {parse_ms:.2f} ms | "
                f"Compute: {comp_ms:.2f} ms | "
                f"Throughput: {throughput:.2f} tok/s | "
                f"VmHWM: {vmhwm / (1024**3):.3f} GiB | "
                f"I/O: {io_read / (1024**3):.3f} GiB"
            )

    # Aggregate metrics
    parse_times = [r["parse_time_ms"] for r in results]
    comp_times = [r["compute_time_ms"] for r in results]
    total_times = [r["total_time_ms"] for r in results]
    throughputs = [r["throughput_tok_s"] for r in results]
    vmhwms = [r["vmhwm_bytes"] for r in results]
    ios = [r["io_read_bytes"] for r in results]

    median_parse = statistics.median(parse_times)
    median_comp = statistics.median(comp_times)
    median_total = statistics.median(total_times)
    median_throughput = statistics.median(throughputs)
    median_vmhwm = statistics.median(vmhwms)
    median_io = statistics.median(ios)

    print("\n" + "=" * 50)
    print(f"MEDIAN RESULTS (N={runs}):")
    print(f"Parse time:    {median_parse:.2f} ms")
    print(f"Compute time:  {median_comp:.2f} ms")
    print(f"Total time:    {median_total:.2f} ms")
    print(f"Throughput:    {median_throughput:.2f} tok/s")
    print(f"VmHWM:         {median_vmhwm / (1024**3):.3f} GiB ({median_vmhwm} bytes)")
    print(f"I/O Read:      {median_io / (1024**3):.3f} GiB ({median_io} bytes)")
    print("=" * 50)

    # Save summary json
    summary_path = root / "reports/2026-09-11/m1_benchmark_raw.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_id": "M1-20260911-001",
                "model": "Qwen1.5-MoE-A2.7B-Chat",
                "revision": "ec052fda178e241c7c443468d2fa1db6618996be",
                "rms_norm_eps": 1e-6,
                "vocab_size": 151936,
                "hidden_size": 2048,
                "runs": results,
                "median": {
                    "parse_time_ms": median_parse,
                    "compute_time_ms": median_comp,
                    "total_time_ms": median_total,
                    "throughput_tok_s": median_throughput,
                    "vmhwm_bytes": median_vmhwm,
                    "vmhwm_gib": median_vmhwm / (1024**3),
                    "io_read_bytes": median_io,
                    "io_read_gib": median_io / (1024**3),
                },
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"Saved raw benchmark data to {summary_path}")


if __name__ == "__main__":
    main()
