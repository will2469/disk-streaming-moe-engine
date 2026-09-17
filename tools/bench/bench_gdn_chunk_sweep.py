#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Benchmark sweep ukuran chunk GDN chunked scan (M8-W2).

Mengukur throughput dan latensi komputasi across chunk sizes:
    C in {64, 128, 256, 512, 1024}
pada sekuens token representatif (default: 1024 token, dk=128, dv=128)
untuk menjustifikasi pemilihan default 512 sesuai spesifikasi M8 (§ Performance).
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="GDN chunk size sweep benchmark (M8-W2)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Panjang sekuens token (default: 1024)",
    )
    parser.add_argument(
        "--dk",
        type=int,
        default=128,
        help="Dimensi key dk (default: 128)",
    )
    parser.add_argument(
        "--dv",
        type=int,
        default=128,
        help="Dimensi value dv (default: 128)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Jumlah layer GDN (default: 2)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Jumlah iterasi per level chunk size (default: 3)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Path file output JSON report (opsional)",
    )
    return parser.parse_args()


def generate_benchmark_fixture(
    fixture_dir: Path, layers: int, dk: int, dv: int, seq_len: int
) -> tuple[Path, Path]:
    """Membangkitkan fixture sintetis untuk benchmark jika belum ada."""
    weights_path = fixture_dir / f"bench_weights_{layers}x{dk}x{dv}.safetensors"
    tokens_path = fixture_dir / f"bench_tokens_{seq_len}.json"

    if not weights_path.exists() or not tokens_path.exists():
        cmd = [
            sys.executable,
            "tools/oracle/generate_gdn_fixture.py",
            "--layers",
            str(layers),
            "--dk",
            str(dk),
            "--dv",
            str(dv),
            "--vocab",
            "512",
            "--seq-len",
            str(seq_len),
            "--seed",
            "42",
            "--output",
            str(weights_path),
            "--tokens-output",
            str(tokens_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            sys.stderr.write(f"Gagal generate benchmark fixture:\n{res.stderr}\n")
            sys.exit(1)

    return tokens_path, weights_path


def run_benchmark_for_chunk(
    tokens_path: Path,
    weights_path: Path,
    output_bin: Path,
    layers: int,
    dk: int,
    dv: int,
    chunk_size: int,
) -> float:
    """Menjalankan satu iterasi benchmark dan mengembalikan wall time (detik)."""
    cmd = [
        "pixi",
        "run",
        "mojo",
        "run",
        "-I",
        "src",
        "tests/integration/run_gdn_scan_test.mojo",
        "--tokens",
        str(tokens_path),
        "--weights",
        str(weights_path),
        "--output",
        str(output_bin),
        "--layers",
        str(layers),
        "--dk",
        str(dk),
        "--dv",
        str(dv),
        "--chunk-size",
        str(chunk_size),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()

    if res.returncode != 0:
        sys.stderr.write(
            f"Error pada chunk_size={chunk_size}:\n{res.stderr}\n{res.stdout}\n"
        )
        sys.exit(1)

    return t1 - t0


def main():
    args = parse_args()
    chunk_sizes = [64, 128, 256, 512, 1024]
    # Filter chunk sizes yang melebihi seq_len
    chunk_sizes = [c for c in chunk_sizes if c <= args.seq_len]

    print(
        f"=== GDN Chunk Size Sweep Benchmark: seq_len={args.seq_len}, "
        f"layers={args.layers}, dk={args.dk}, dv={args.dv} ==="
    )

    with tempfile.TemporaryDirectory(prefix="gdn_bench_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tokens_path, weights_path = generate_benchmark_fixture(
            tmp_dir, args.layers, args.dk, args.dv, args.seq_len
        )
        out_bin = tmp_dir / "bench_out.bin"

        results = []
        for c in chunk_sizes:
            times = []
            # Warm-up run
            run_benchmark_for_chunk(
                tokens_path,
                weights_path,
                out_bin,
                args.layers,
                args.dk,
                args.dv,
                c,
            )

            for _ in range(args.runs):
                t_sec = run_benchmark_for_chunk(
                    tokens_path,
                    weights_path,
                    out_bin,
                    args.layers,
                    args.dk,
                    args.dv,
                    c,
                )
                times.append(t_sec)

            median_sec = sorted(times)[len(times) // 2]
            tok_per_sec = args.seq_len / median_sec if median_sec > 0 else 0.0
            results.append(
                {
                    "chunk_size": c,
                    "median_sec": median_sec,
                    "tok_per_sec": tok_per_sec,
                    "runs": times,
                }
            )

        print(
            "\n| Chunk Size | Median Time (s) | Throughput (tok/s) | Relative Speed |"
        )
        print("|------------|-----------------|--------------------|----------------|")
        base_t = results[0]["median_sec"]
        best_chunk = 512
        best_throughput = 0.0

        for r in results:
            rel_speed = base_t / r["median_sec"] if r["median_sec"] > 0 else 1.0
            print(
                f"| {r['chunk_size']:>10} | {r['median_sec']:>15.4f} | "
                f"{r['tok_per_sec']:>18.1f} | {rel_speed:>13.2f}x |"
            )
            if r["tok_per_sec"] > best_throughput:
                best_throughput = r["tok_per_sec"]
                best_chunk = r["chunk_size"]

        print(
            f"\nOptimal / Recommended Default Chunk Size: {best_chunk} "
            f"(throughput max: {best_throughput:.1f} tok/s)\n"
        )

        if args.output_json:
            out_p = Path(args.output_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "config": {
                            "seq_len": args.seq_len,
                            "layers": args.layers,
                            "dk": args.dk,
                            "dv": args.dv,
                            "runs": args.runs,
                        },
                        "results": results,
                        "best_chunk": best_chunk,
                    },
                    f,
                    indent=2,
                )
            print(f"Benchmark report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
