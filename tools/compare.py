#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""CLI Wrapper untuk evaluasi ekivalensi numerik F10 (Gate G-M5-1).

Menghubungkan Mojo engine / Oracle outputs ke `kimo-tools compare`
(atau komputasi F10 fallback deterministik).
Mendukung argumen:
  --mojo / --ref: path binary logits kandidat / mojo
  --oracle / --cand: path binary logits referensi / oracle
  --gate: gate evaluasi (default: G-M5-1)
  --dim: ukuran vocab / dimensi (default: 151936)
"""

import argparse
import json
import math
import os
import subprocess
import sys


def find_kimo_tools() -> str | None:
    """Mencari executable kimo-tools."""
    env_bin = os.environ.get("COMPARE_BIN")
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin

    candidates = [
        os.path.abspath("target/debug/kimo-tools"),
        os.path.abspath("target/release/kimo-tools"),
        os.path.abspath("tools/kimo-tools/target/debug/kimo-tools"),
        os.path.abspath("tools/kimo-tools/target/release/kimo-tools"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    # Cek di PATH
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(path_dir, "kimo-tools")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


def compute_f10_fallback(
    ref_path: str, cand_path: str, vocab_size: int, gate: str
) -> tuple[int, str]:
    """Fallback pure-Python perhitungan F10 jika kimo-tools belum ada."""
    import numpy as np

    if not os.path.exists(ref_path):
        err = {
            "error_type": "FILE_NOT_FOUND",
            "detail": f"file not found: {ref_path}",
            "stage": "compare",
        }
        return 2, json.dumps(err)

    if not os.path.exists(cand_path):
        err = {
            "error_type": "FILE_NOT_FOUND",
            "detail": f"file not found: {cand_path}",
            "stage": "compare",
        }
        return 2, json.dumps(err)

    ref_data = np.fromfile(ref_path, dtype=np.float32)
    cand_data = np.fromfile(cand_path, dtype=np.float32)

    if len(ref_data) != len(cand_data):
        err = {
            "error_type": "LAYOUT_MISMATCH",
            "detail": (
                f"length mismatch: ref has {len(ref_data)} floats, "
                f"cand has {len(cand_data)} floats"
            ),
            "stage": "compare",
        }
        return 2, json.dumps(err)

    if len(ref_data) == 0:
        err = {
            "error_type": "LAYOUT_MISMATCH",
            "detail": "empty data buffer",
            "stage": "compare",
        }
        return 2, json.dumps(err)

    if len(ref_data) % vocab_size != 0:
        err = {
            "error_type": "LAYOUT_MISMATCH",
            "detail": (
                f"total elements {len(ref_data)} not divisible by "
                f"vocab_size {vocab_size}"
            ),
            "stage": "compare",
        }
        return 2, json.dumps(err)

    diff = np.abs(cand_data - ref_data)
    delta_max = float(np.max(diff))

    sum_ref_sq = float(np.sum(ref_data.astype(np.float64) ** 2))
    sum_diff_sq = float(np.sum(diff.astype(np.float64) ** 2))
    epsilon_rel = math.sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else 0.0

    sum_cand_sq = float(np.sum(cand_data.astype(np.float64) ** 2))
    dot_prod = float(np.sum(ref_data.astype(np.float64) * cand_data.astype(np.float64)))
    norm_prod = math.sqrt(sum_ref_sq) * math.sqrt(sum_cand_sq)
    cos_theta = (dot_prod / norm_prod) if norm_prod > 0.0 else 1.0

    num_tokens = len(ref_data) // vocab_size
    ref_2d = ref_data.reshape((num_tokens, vocab_size))
    cand_2d = cand_data.reshape((num_tokens, vocab_size))

    ref_argmax = np.argmax(ref_2d, axis=1)
    cand_argmax = np.argmax(cand_2d, axis=1)
    matched = int(np.sum(ref_argmax == cand_argmax))
    agreement = (matched / num_tokens) * 100.0

    # Cross entropy delta
    def entropy(row):
        m = np.max(row)
        exp = np.exp(row - m)
        p = exp / np.sum(exp)
        return -float(np.sum(p * np.log(p + 1e-12)))

    h_ref = sum(entropy(row) for row in ref_2d) / num_tokens
    h_cand = sum(entropy(row) for row in cand_2d) / num_tokens
    delta_ce = abs(h_cand - h_ref)

    metrics = {
        "delta_max": delta_max,
        "epsilon_rel": epsilon_rel,
        "cos_theta": cos_theta,
        "agreement": agreement,
        "delta_ce": delta_ce,
    }

    # Evaluate gate
    if gate in ("G-M4-1", "G-M5-1"):
        is_pass = (
            delta_max <= 1e-2
            and epsilon_rel <= 1e-4
            and agreement >= 99.9
            and delta_ce <= 0.02
        )
        thresh_str = (
            "delta_max <= 1e-2 && epsilon_rel <= 1e-4 && "
            "agreement >= 99.9 && delta_ce <= 0.02"
        )
    else:
        is_pass = (
            delta_max <= 1e-3 and epsilon_rel <= 1e-4 and abs(agreement - 100.0) < 1e-5
        )
        thresh_str = "delta_max <= 1e-3 && epsilon_rel <= 1e-4 && agreement == 100.0"

    run_id = "M5-F10-001" if gate == "G-M5-1" else "M4-F10-001"
    report = {
        "status": "MATCH" if is_pass else "MISMATCH",
        "run_id": run_id,
        "reference_path": ref_path,
        "candidate_path": cand_path,
        "metrics": metrics,
        "verdict": "PASS" if is_pass else "FAIL",
        "threshold": thresh_str,
        "fail_category": None if is_pass else "numeric-order",
    }
    return (0 if is_pass else 1), json.dumps(report, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="F10 Numerical Equivalence Comparison Wrapper"
    )
    parser.add_argument("--ref", default="", help="Path ke logits referensi")
    parser.add_argument("--cand", default="", help="Path ke logits kandidat")
    parser.add_argument("--oracle", default="", help="Alias untuk logits referensi")
    parser.add_argument("--mojo", default="", help="Alias untuk logits kandidat/mojo")
    parser.add_argument("--gate", default="G-M5-1", help="Gate F10 (default: G-M5-1)")
    parser.add_argument(
        "--dim",
        "--vocab-size",
        dest="vocab_size",
        type=int,
        default=151936,
        help="Ukuran vocab (default: 151936)",
    )
    parser.add_argument("--run-id", default="", help="Custom run ID (opsional)")
    parser.add_argument("--oracle-routing", default="", help="Path routing oracle JSON")
    parser.add_argument("--cand-routing", default="", help="Path routing kandidat JSON")
    parser.add_argument("positional", nargs="*", help="Positional args [ref, cand]")

    args = parser.parse_args()

    ref_path = args.ref or args.oracle
    cand_path = args.cand or args.mojo

    if not ref_path and args.positional:
        ref_path = args.positional[0]
    if not cand_path and len(args.positional) > 1:
        cand_path = args.positional[1]

    if not ref_path or not cand_path:
        sys.stderr.write(
            "Usage: python tools/compare.py --ref <ref.bin> --cand <cand.bin>"
            " [--gate G-M5-1]\n"
        )
        sys.exit(2)

    kimo_tools = find_kimo_tools()
    if kimo_tools:
        cmd = [
            kimo_tools,
            "compare",
            "--ref",
            ref_path,
            "--cand",
            cand_path,
            "--gate",
            args.gate,
            "--dim",
            str(args.vocab_size),
        ]
        if args.run_id:
            cmd.extend(["--run-id", args.run_id])
        if args.oracle_routing and args.cand_routing:
            cmd.extend(
                [
                    "--oracle-routing",
                    args.oracle_routing,
                    "--cand-routing",
                    args.cand_routing,
                ]
            )

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.stdout:
            sys.stdout.write(res.stdout)
        if res.stderr:
            sys.stderr.write(res.stderr)
        sys.exit(res.returncode)
    else:
        # Fallback pure-Python jika binary kimo-tools belum terkompilasi
        rc, out = compute_f10_fallback(ref_path, cand_path, args.vocab_size, args.gate)
        if rc == 2:
            sys.stderr.write(out + "\n")
        else:
            sys.stdout.write(out + "\n")
        sys.exit(rc)


if __name__ == "__main__":
    main()
