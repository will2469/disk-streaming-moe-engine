#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""CLI Wrapper untuk evaluasi ekivalensi numerik F10 (Gate G-M5-1).

Menghubungkan Mojo engine / Oracle outputs ke `dismoen-tools compare`
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


def find_dismoen_tools() -> str | None:
    """Mencari executable dismoen-tools."""
    env_bin = os.environ.get("COMPARE_BIN")
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin

    candidates = [
        os.path.abspath("target/debug/dismoen-tools"),
        os.path.abspath("target/release/dismoen-tools"),
        os.path.abspath("tools/dismoen-tools/target/debug/dismoen-tools"),
        os.path.abspath("tools/dismoen-tools/target/release/dismoen-tools"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    # Cek di PATH
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(path_dir, "dismoen-tools")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


def read_floats_py(path: str):
    """Membaca float32 dari format raw atau GDNS v1 framed binary."""
    import hashlib

    import numpy as np

    with open(path, "rb") as f:
        data = f.read()

    if data.startswith(b"GDNS"):
        if len(data) < 160:
            return None, "GDNS state file shorter than 160 bytes"
        state_bytes = int.from_bytes(data[32:40], "little")
        payload_end = 128 + state_bytes
        if len(data) != payload_end + 32:
            return None, (
                f"GDNS file size mismatch: actual {len(data)}, "
                f"expected {payload_end + 32}"
            )
        hasher = hashlib.sha256()
        hasher.update(data[:payload_end])
        if hasher.digest() != data[payload_end:]:
            return None, "CORRUPT_STATE_CHECKSUM"
        return np.frombuffer(data[128:payload_end], dtype=np.float32), None

    if len(data) % 4 != 0:
        return None, f"file size {len(data)} is not a multiple of 4 bytes"
    return np.frombuffer(data, dtype=np.float32), None


def compute_f10_fallback(
    ref_path: str,
    cand_path: str,
    vocab_size: int,
    gate: str,
    tolerance: float | None = None,
) -> tuple[int, str]:
    """Fallback pure-Python perhitungan F10 jika dismoen-tools belum ada."""
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

    ref_data, err_ref = read_floats_py(ref_path)
    if err_ref:
        err_type = (
            "CORRUPT_STATE_CHECKSUM"
            if err_ref == "CORRUPT_STATE_CHECKSUM"
            else "LAYOUT_MISMATCH"
        )
        return 2, json.dumps(
            {"error_type": err_type, "detail": err_ref, "stage": "compare"}
        )

    cand_data, err_cand = read_floats_py(cand_path)
    if err_cand:
        err_type = (
            "CORRUPT_STATE_CHECKSUM"
            if err_cand == "CORRUPT_STATE_CHECKSUM"
            else "LAYOUT_MISMATCH"
        )
        return 2, json.dumps(
            {"error_type": err_type, "detail": err_cand, "stage": "compare"}
        )

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

    if gate == "G-M8-1" or tolerance is not None:
        if len(ref_data) % vocab_size != 0:
            vocab_size = len(ref_data)

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
    if tolerance is not None or gate == "G-M8-1":
        tol = tolerance if tolerance is not None else 1e-3
        is_pass = delta_max <= tol and epsilon_rel <= 1e-4
        thresh_str = (
            "delta_max <= 1e-3 && epsilon_rel <= 1e-4"
            if abs(tol - 1e-3) < 1e-9
            else f"delta_max <= {tol:e} && epsilon_rel <= 1e-4"
        )
        run_id = "M8-F10-001"
    elif gate == "G-M10-3":
        tol = tolerance if tolerance is not None else 1e-7
        is_pass = delta_max <= tol
        thresh_str = f"delta_max <= {tol:e}"
        run_id = "M10-G3-001"
    elif gate in ("G-M4-1", "G-M5-1"):
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
        run_id = "M5-F10-001" if gate == "G-M5-1" else "M4-F10-001"
    else:
        is_pass = (
            delta_max <= 1e-3 and epsilon_rel <= 1e-4 and abs(agreement - 100.0) < 1e-5
        )
        thresh_str = "delta_max <= 1e-3 && epsilon_rel <= 1e-4 && agreement == 100.0"
        run_id = "M1-F10-001"

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


def build_dismoen_tools_cmd(
    dismoen_tools: str,
    ref_path: str,
    cand_path: str,
    gate: str,
    args: argparse.Namespace,
) -> list[str]:
    """Menyusun argumen command untuk memanggil binary dismoen-tools."""
    cmd = [
        dismoen_tools,
        "compare",
        "--ref",
        ref_path,
        "--cand",
        cand_path,
        "--gate",
        gate,
    ]
    if args.vocab_size is not None:
        cmd.extend(["--dim", str(args.vocab_size)])
    elif gate not in ("G-M8-1", "G-M2-1", "G-M3-1") and args.tolerance is None:
        cmd.extend(["--dim", "151936"])
    if args.tolerance is not None:
        cmd.extend(["--tolerance", str(args.tolerance)])
    if args.output:
        cmd.extend(["--output", args.output])
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
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="F10 Numerical Equivalence Comparison Wrapper"
    )
    parser.add_argument("--ref", default="", help="Path ke logits referensi")
    parser.add_argument("--reference", default="", help="Alias untuk --ref")
    parser.add_argument("--cand", default="", help="Path ke logits kandidat")
    parser.add_argument("--candidate", default="", help="Alias untuk --cand")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Toleransi delta_max (default: None)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Path untuk menyimpan laporan compare JSON",
    )
    parser.add_argument("--oracle", default="", help="Alias untuk logits referensi")
    parser.add_argument("--mojo", default="", help="Alias untuk logits kandidat/mojo")
    parser.add_argument("--gate", default="G-M5-1", help="Gate F10 (default: G-M5-1)")
    parser.add_argument(
        "--dim",
        "--vocab-size",
        dest="vocab_size",
        type=int,
        default=None,
        help="Ukuran vocab (opsional, auto-detect bila tidak diisi)",
    )
    parser.add_argument("--run-id", default="", help="Custom run ID (opsional)")
    parser.add_argument("--oracle-routing", default="", help="Path routing oracle JSON")
    parser.add_argument("--cand-routing", default="", help="Path routing kandidat JSON")
    parser.add_argument("positional", nargs="*", help="Positional args [ref, cand]")

    args = parser.parse_args()

    ref_path = args.ref or args.reference or args.oracle
    cand_path = args.cand or args.candidate or args.mojo

    if not ref_path and args.positional:
        ref_path = args.positional[0]
    if not cand_path and len(args.positional) > 1:
        cand_path = args.positional[1]

    if not ref_path or not cand_path:
        sys.stderr.write(
            "Usage: python tools/compare.py --ref <ref.bin> --cand <cand.bin>"
            " [--gate G-M5-1] [--tolerance <tol>] [--output <path>]\n"
        )
        sys.exit(2)

    gate = args.gate
    if args.tolerance is not None and gate == "G-M5-1":
        gate = "G-M8-1"

    dismoen_tools = find_dismoen_tools()
    if dismoen_tools:
        cmd = build_dismoen_tools_cmd(dismoen_tools, ref_path, cand_path, gate, args)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.stdout:
            sys.stdout.write(res.stdout)
        if res.stderr:
            sys.stderr.write(res.stderr)
        sys.exit(res.returncode)
    else:
        # Fallback pure-Python jika binary dismoen-tools belum terkompilasi
        rc, out = compute_f10_fallback(
            ref_path, cand_path, args.vocab_size, gate, args.tolerance
        )
        if args.output and rc != 2:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out + "\n")
        if rc == 2:
            sys.stderr.write(out + "\n")
        else:
            sys.stdout.write(out + "\n")
        sys.exit(rc)


if __name__ == "__main__":
    main()
