#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
r"""Gate G-M8-1 Certification: 100 Random Sequences Numerical Equivalence.

Menguji 100 sekuens acak deterministik (seed 42) untuk membuktikan ekuivalensi
numerik kernel chunked scan (Mojo dismoen gdn) terhadap Naive Oracle FP32 (PyTorch):
    \Delta_{max} <= 10^{-3}
    \epsilon_{rel} <= 10^{-4}
Wajib mencakup konfigurasi simetris (dk=dv=32) dan asimetris (dk=32, dv=48)
dengan variasi panjang sekuens s in [8, 1024] dan chunk size C in [8, 512]
di bawah single-threaded execution (--threads 1) sesuai Invariant I-8 mojo-1-0.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_gdns_payload(path: str) -> tuple[int, int, int, bytes]:
    """Membaca header dan payload state dari file GDNS v1."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 160:
        raise ValueError(f"File too small for GDNS v1: {len(data)} bytes")
    magic = data[:4]
    if magic != b"GDNS":
        raise ValueError(f"Invalid magic: {magic!r}")
    layers = int.from_bytes(data[48:52], "little")
    dk = int.from_bytes(data[52:56], "little")
    dv = int.from_bytes(data[56:60], "little")
    payload = data[128:-32]
    return layers, dk, dv, payload


def compute_metrics(ref_bytes: bytes, cand_bytes: bytes) -> tuple[float, float]:
    """Menghitung delta_max dan epsilon_rel kanonis sesuai tools/compare.py."""
    import math

    import numpy as np

    if len(ref_bytes) != len(cand_bytes):
        raise ValueError(
            f"Payload size mismatch: {len(ref_bytes)} vs {len(cand_bytes)}"
        )
    ref_arr = np.frombuffer(ref_bytes, dtype=np.float32)
    cand_arr = np.frombuffer(cand_bytes, dtype=np.float32)

    diff = np.abs(cand_arr - ref_arr)
    delta_max = float(np.max(diff))

    sum_ref_sq = float(np.sum(ref_arr.astype(np.float64) ** 2))
    sum_diff_sq = float(np.sum(diff.astype(np.float64) ** 2))
    epsilon_rel = math.sqrt(sum_diff_sq / sum_ref_sq) if sum_ref_sq > 0.0 else 0.0

    return delta_max, epsilon_rel


def run_random100_certification() -> int:
    """Sertifikasi Gate G-M8-1: 100 sekuens acak."""
    sys.path.insert(0, str(REPO_ROOT))
    from tools.oracle.oracle_gdn import (
        load_or_synthesize_weights,
        run_oracle_naive,
        write_gdns_v1,
    )

    dismoen_bin = str(REPO_ROOT / "dismoen")
    if not os.path.exists(dismoen_bin):
        sys.stderr.write(
            "ERROR: dismoen binary tidak ditemukan. Jalankan pixi run build!\n"
        )
        return 1

    fixtures_dir = REPO_ROOT / "fixtures"
    sym_weights_path = str(fixtures_dir / "m8_gdn_weights.safetensors")
    asym_weights_path = str(fixtures_dir / "m8_asym_weights.safetensors")

    print("=" * 72)
    print("MILESTONE M8 WAVE 6: GATE G-M8-1 RANDOM 100 SEQUENCES CERTIFICATION")
    print("=" * 72)
    print("Evaluating 100 random deterministic sequences under --threads 1...")

    # Memuat bobot oracle sekali di memori
    w_sym = load_or_synthesize_weights(sym_weights_path, 2, 32, 32, 512, 42)
    w_asym = load_or_synthesize_weights(asym_weights_path, 2, 32, 48, 512, 42)

    rng = random.Random(42)
    all_results = []
    failed_cases = []

    # Sequence length candidate points
    seq_lengths_pool = [
        8,
        13,
        16,
        25,
        32,
        47,
        64,
        89,
        127,
        128,
        129,
        200,
        255,
        256,
        257,
        337,
        511,
        512,
        513,
        687,
        768,
        1000,
        1024,
    ]
    chunk_sizes_pool = [8, 16, 32, 64, 128, 256, 512]

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx in range(1, 101):
            is_asym = idx > 50
            if is_asym:
                dk, dv = 32, 48
                weights = w_asym
                model_dir_arg = asym_weights_path
                profile_name = "Asymmetric (32x48)"
            else:
                dk, dv = 32, 32
                weights = w_sym
                model_dir_arg = sym_weights_path
                profile_name = "Symmetric  (32x32)"

            # Pilih panjang sekuens dan chunk_size
            seq_len = rng.choice(seq_lengths_pool)
            chunk_size = rng.choice(chunk_sizes_pool)
            # Batasi chunk_size agar tidak melebihi rentang valid
            chunk_size = max(8, min(chunk_size, 512))

            # Generate random tokens [0, 512)
            tokens = [rng.randint(0, 511) for _ in range(seq_len)]

            tok_path = os.path.join(tmpdir, f"tok_{idx}.json")
            with open(tok_path, "w", encoding="utf-8") as f:
                json.dump({"tokens": tokens}, f)

            ref_path = os.path.join(tmpdir, f"ref_{idx}.bin")
            cand_path = os.path.join(tmpdir, f"cand_{idx}.bin")

            # 1. Oracle naive FP32
            ref_tensor = run_oracle_naive(tokens, 2, dk, dv, weights)
            write_gdns_v1(ref_path, ref_tensor, 2, dk, dv)

            # 2. Mojo dismoen gdn
            cmd = [
                dismoen_bin,
                "gdn",
                "--model-dir",
                model_dir_arg,
                "--tokens",
                tok_path,
                "--output",
                cand_path,
                "--layers",
                "2",
                "--dk",
                str(dk),
                "--dv",
                str(dv),
                "--chunk-size",
                str(chunk_size),
                "--threads",
                "1",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[{idx:3d}/100] ERROR: dismoen gdn exit {res.returncode}")
                print(res.stderr)
                failed_cases.append((idx, "DISMOEN_CRASH", 999.0, 999.0))
                continue

            # 3. Bandingkan binary payload
            _, _, _, ref_bytes = load_gdns_payload(ref_path)
            _, _, _, cand_bytes = load_gdns_payload(cand_path)

            delta_max, eps_rel = compute_metrics(ref_bytes, cand_bytes)

            is_pass = delta_max <= 1e-3 and eps_rel <= 1e-4
            verdict = "PASS" if is_pass else "FAIL"

            record = {
                "id": idx,
                "profile": profile_name,
                "dk": dk,
                "dv": dv,
                "seq_len": seq_len,
                "chunk_size": chunk_size,
                "delta_max": delta_max,
                "eps_rel": eps_rel,
                "verdict": verdict,
            }
            all_results.append(record)

            if not is_pass:
                failed_cases.append((idx, "GATE_FAIL", delta_max, eps_rel))

            if idx % 10 == 0 or idx == 1 or idx == 100:
                print(
                    f"--> [{idx:3d}/100] {profile_name} | s={seq_len:4d} | "
                    f"C={chunk_size:3d} | Delta={delta_max:.4e} | Verdict: [{verdict}]"
                )

    print("\n" + "=" * 72)
    print("RINGKASAN SERTIFIKASI GATE G-M8-1 (100 SEKUEN RANDOM)")
    print("=" * 72)
    total_tested = len(all_results)
    total_passed = sum(1 for r in all_results if r["verdict"] == "PASS")
    max_observed_delta = (
        max(r["delta_max"] for r in all_results) if all_results else 0.0
    )
    max_observed_rel = max(r["eps_rel"] for r in all_results) if all_results else 0.0

    print(f"Total Sequences Tested: {total_tested} / 100")
    print(f"Total Passed:           {total_passed} / 100")
    print(f"Max Delta Observed:     {max_observed_delta:.6e} (Threshold <= 1.00e-03)")
    print(f"Max Rel Eps Observed:   {max_observed_rel:.6e} (Threshold <= 1.00e-04)")

    # Simpan laporan markdown scorecard
    report_dir = REPO_ROOT / "reports" / "2026-09-18"
    os.makedirs(report_dir, exist_ok=True)
    report_path = report_dir / "M8-random100-scorecard.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Laporan Sertifikasi Gate G-M8-1: 100 Sekuens Acak\n\n")
        f.write("- **Tanggal**: 2026-09-18\n")
        f.write("- **Gate**: G-M8-1 (Ekuivalensi Numerik Chunked vs Naive Oracle)\n")
        f.write(
            "- **Kriteria**:"
            " $\\Delta_{\\max} \\le 10^{-3}$,"
            " $\\epsilon_{\\text{rel}} \\le 10^{-4}$\n"
        )
        f.write(
            "- **Konfigurasi**: `--threads 1` (single-threaded invariant), seed 42\n"
        )
        f.write(f"- **Total Sekuens**: {total_tested}\n")
        f.write(f"- **Hasil**: {total_passed}/100 PASS\n")
        f.write(f"- **Delta Max Ekstrem**: {max_observed_delta:.6e}\n")
        f.write(f"- **Eps Rel Ekstrem**: {max_observed_rel:.6e}\n\n")
        f.write("## Sampel Hasil Pengujian (10 Interval Terpilih)\n\n")
        hdr = (
            "| ID | Profil | Dims ($d_k \\times d_v$)"
            " | Sekuens ($s$) | Chunk ($C$)"
            " | $\\Delta_{\\max}$"
            " | $\\epsilon_{\\text{rel}}$ | Status |\n"
        )
        f.write(hdr)
        f.write("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for i in [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
            if i < len(all_results):
                r = all_results[i]
                f.write(
                    f"| {r['id']} | {r['profile'].strip()}"
                    f" | {r['dk']}x{r['dv']}"
                    f" | {r['seq_len']}"
                    f" | {r['chunk_size']}"
                    f" | {r['delta_max']:.4e}"
                    f" | {r['eps_rel']:.4e}"
                    f" | **{r['verdict']}** |\n"
                )
        f.write("\n## Kesimpulan Gate G-M8-1\n\n")
        if total_passed == 100:
            f.write(
                "**GATE G-M8-1 VERDICT: [PASS]**"
                " — 100/100 sekuens acak lolos"
                " ekuivalensi numerik.\n"
            )
        else:
            f.write(
                f"**GATE G-M8-1 VERDICT: [FAIL]** — {len(failed_cases)} kasus gagal.\n"
            )

    print(f"Laporan tersimpan di: {report_path}")

    if total_passed == 100:
        print("\nVERDICT GATE G-M8-1: [PASS] (100/100 LULUS PENUH!)")
        return 0
    print(f"\nVERDICT GATE G-M8-1: [FAIL] ({len(failed_cases)} kasus gagal)")
    return 1


if __name__ == "__main__":
    sys.exit(run_random100_certification())
