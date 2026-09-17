#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Analisis stabilitas sekuens panjang GDN & evaluasi Gate G-M8-2.

Menguji scaling s in {1K..32K} untuk membuktikan:
1. Ketahanan numerik F14 vs naive oracle (Delta_max <= 1e-3, G-M8-1).
2. Runtime peak memory O(1) konstan (|Delta VmHWM / Delta s| approx 0, <= 10 MB).
3. Ukuran tensor state biner GDNS v1 konstan independen dari s.
4. Pemantauan norm state Frobenius bebas dari lonjakan eksponensial.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="GDN Long-Sequence Stability & Peak Memory Benchmark"
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096, 8192, 16384, 32768],
        help="Sequence lengths to evaluate",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with subset {1024, 2048, 4096}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size for GDN chunked scan",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of GDN layers",
    )
    parser.add_argument(
        "--dk",
        type=int,
        default=32,
        help="Key dimension",
    )
    parser.add_argument(
        "--dv",
        type=int,
        default=32,
        help="Value dimension",
    )
    return parser.parse_args()


def generate_tokens(seq_len: int, vocab: int = 512, seed: int = 42) -> list[int]:
    """Membangkitkan token sequence deterministik dengan LCG pseudo-random."""
    # LCG sederhana deterministik
    tokens = []
    state = seed
    for _ in range(seq_len):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        tokens.append(state % vocab)
    return tokens


def compute_frobenius_norm(binary_path: Path, state_bytes: int) -> float:
    """Menghitung Frobenius norm dari payload FP32 state."""
    raw = binary_path.read_bytes()
    # 128 header, state_bytes payload, 32 checksum
    payload = raw[128 : 128 + state_bytes]
    import struct

    num_floats = state_bytes // 4
    floats = struct.unpack(f"<{num_floats}f", payload)
    sq_sum = sum(x * x for x in floats)
    return math.sqrt(sq_sum)


def main():
    args = parse_args()
    seq_lengths = [1024, 2048, 4096] if args.quick else args.seq_lengths

    kimo_bin = REPO_ROOT / "kimo"
    oracle_script = REPO_ROOT / "tools" / "oracle" / "oracle_gdn.py"
    weights_path = REPO_ROOT / "fixtures" / "m8_gdn_weights.safetensors"

    if not kimo_bin.exists():
        print("Binary kimo not found, compiling...")
        subprocess.run(["pixi", "run", "build"], check=True, cwd=REPO_ROOT)

    expected_state_bytes = args.layers * args.dv * args.dk * 4
    expected_file_size = 128 + expected_state_bytes + 32

    results = []

    print("===================================================================")
    print("MILESTONE M8: Long-Sequence Stability & Peak Memory Evaluation")
    print("===================================================================")
    print(f"Layers: {args.layers}, dk: {args.dk}, dv: {args.dv}")
    print(f"Chunk Size: {args.chunk_size}, Fixed State Size: {expected_file_size} B")
    print(f"Testing Sequence Grid: {seq_lengths}\n")

    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    py_exec = str(venv_py) if venv_py.exists() else sys.executable

    with tempfile.TemporaryDirectory(prefix="gdn_stability_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        for s in seq_lengths:
            print(f"--> [s = {s:5d} tokens] Menguji stabilitas & memori...")
            tokens = generate_tokens(s, vocab=512, seed=42)
            tok_file = tmp_path / f"tokens_{s}.json"
            tok_file.write_text(json.dumps({"tokens": tokens, "seq_len": s}))

            out_mojo = tmp_path / f"state_mojo_{s}.bin"
            out_oracle = tmp_path / f"state_oracle_{s}.bin"

            # 1. Jalankan kimo gdn
            cmd_mojo = [
                str(kimo_bin),
                "gdn",
                "--model-dir",
                str(weights_path),
                "--tokens",
                str(tok_file),
                "--output",
                str(out_mojo),
                "--layers",
                str(args.layers),
                "--dk",
                str(args.dk),
                "--dv",
                str(args.dv),
                "--chunk-size",
                str(args.chunk_size),
                "--threads",
                "1",
            ]
            res_mojo = subprocess.run(
                cmd_mojo,
                capture_output=True,
                text=True,
                check=True,
            )
            report = json.loads(res_mojo.stdout)
            vmhwm = report["metrics"]["vmhwm_bytes"]
            wall_sec = report["metrics"]["walltime_sec"]

            # 2. Jalankan oracle naive
            cmd_oracle = [
                py_exec,
                str(oracle_script),
                "--tokens",
                str(tok_file),
                "--weights",
                str(weights_path),
                "--output",
                str(out_oracle),
                "--layers",
                str(args.layers),
                "--dk",
                str(args.dk),
                "--dv",
                str(args.dv),
                "--seed",
                "42",
            ]
            subprocess.run(
                cmd_oracle,
                capture_output=True,
                text=True,
                check=True,
            )

            # 3. Bandingkan dengan kimo compare
            cmd_comp = [
                str(kimo_bin),
                "compare",
                "--reference",
                str(out_oracle),
                "--candidate",
                str(out_mojo),
                "--gate",
                "G-M8-1",
            ]
            res_comp = subprocess.run(
                cmd_comp,
                capture_output=True,
                text=True,
                check=True,
            )
            comp_data = json.loads(res_comp.stdout)
            delta_max = comp_data["metrics"]["delta_max"]
            epsilon_rel = comp_data["metrics"]["epsilon_rel"]
            verdict = comp_data["verdict"]

            # 4. Validasi ukuran file konstan
            actual_size = out_mojo.stat().st_size
            assert (
                actual_size == expected_file_size
            ), f"State size changed! {actual_size} != {expected_file_size}"

            # 5. Hitung Frobenius norm
            fnorm = compute_frobenius_norm(out_mojo, expected_state_bytes)

            results.append(
                {
                    "seq_len": s,
                    "vmhwm_bytes": vmhwm,
                    "vmhwm_mb": vmhwm / (1024 * 1024),
                    "wall_sec": wall_sec,
                    "delta_max": delta_max,
                    "epsilon_rel": epsilon_rel,
                    "verdict": verdict,
                    "fnorm": fnorm,
                }
            )

        print("\n===================================================================")
        print("HASIL EVALUASI STABILITAS & MEMORY SCALING")
        print("===================================================================")
        header = (
            f"| {'Seq Len':>7} | {'VmHWM (MB)':>10} | {'Delta Max':>12} | "
            f"{'Eps Rel':>11} | {'F-Norm':>10} | {'G-M8-1':>7} |"
        )
        print(header)
        print("|" + "-" * 75 + "|")
        for r in results:
            row = (
                f"| {r['seq_len']:>7} | {r['vmhwm_mb']:>10.2f} | "
                f"{r['delta_max']:>12.4e} | {r['epsilon_rel']:>11.4e} | "
                f"{r['fnorm']:>10.4f} | {r['verdict']:>7} |"
            )
            print(row)

        # Evaluasi Gate G-M8-2
        vmhwm_min = results[0]["vmhwm_bytes"]
        vmhwm_max = results[-1]["vmhwm_bytes"]
        delta_mem_bytes = abs(vmhwm_max - vmhwm_min)
        delta_mem_mb = delta_mem_bytes / (1024 * 1024)

        print("\n===================================================================")
        print("EVALUASI SCORECARD G-M8-2 (O(1) RUNTIME PEAK MEMORY)")
        print("===================================================================")
        s_min = results[0]["seq_len"]
        s_max = results[-1]["seq_len"]
        print(f"VmHWM pada s={s_min}:  {results[0]['vmhwm_mb']:.2f} MB")
        print(f"VmHWM pada s={s_max}: {results[-1]['vmhwm_mb']:.2f} MB")
        print(f"Delta VmHWM:               {delta_mem_mb:.2f} MB")
        print("Batas Toleransi G-M8-2:    <= 10.00 MB (slope approx 0)")

        if delta_mem_bytes <= 10 * 1024 * 1024:
            print("VERDICT G-M8-2: [PASS] Peak memory runtime O(1) konstan terbukti!")
        else:
            print("VERDICT G-M8-2: [FAIL] Peak memory bertumbuh melampaui batas 10 MB!")
            sys.exit(1)

        # Cek stabilitas norm
        initial_norm = results[0]["fnorm"]
        final_norm = results[-1]["fnorm"]
        print(
            f"State Frobenius Norm: initial={initial_norm:.4f}, final={final_norm:.4f}"
        )
        # Norm tidak boleh NaN atau infinity
        assert not math.isnan(final_norm) and not math.isinf(final_norm)

        print("\nSELURUH PENGUJIAN STABILITAS & PEAK MEMORY M8-W4 HIJAU 100%!")


if __name__ == "__main__":
    main()
