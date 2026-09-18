#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Stress test boundary chunk & continuation multi-chunk GDN (§ State Continuation).

Matriks uji:
- Chunk sizes: C in {64, 128, 256, 512, 1024}
- Boundary split (s1 + s2 = 1024):
  1. Split simetris rapi: (512, 512)
  2. Split under-cut: (511, 513)
  3. Split over-cut: (513, 511)
  4. Split asimetris prima: (337, 687)

Membuktikan bahwa hukum komposisi operator:
state(seq1 || seq2) == continuation(state(seq1), seq2)
berlaku secara universal di bawah Gate G-M8-1 (Delta_max <= 1e-3, eps_rel <= 1e-4).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def generate_deterministic_tokens(
    total_len: int = 1024,
    vocab: int = 512,
    seed: int = 42,
) -> list[int]:
    """Membangkitkan array token pseudo-random deterministik."""
    tokens = []
    state = seed
    for _ in range(total_len):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        tokens.append(state % vocab)
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="GDN Multi-Chunk & Boundary Stress Test Matrix"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with subset of chunks {64, 512} for fast CI",
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
    args = parser.parse_args()

    dismoen_bin = REPO_ROOT / "dismoen"
    weights_path = REPO_ROOT / "fixtures" / "m8_gdn_weights.safetensors"

    if not dismoen_bin.exists():
        subprocess.run(["pixi", "run", "build"], check=True, cwd=REPO_ROOT)

    chunk_sizes = [64, 512] if args.quick else [64, 128, 256, 512, 1024]
    splits = [
        (512, 512, "Simetris Rapi"),
        (511, 513, "Under-Cut (-1)"),
        (513, 511, "Over-Cut (+1)"),
        (337, 687, "Asimetris Prima"),
    ]

    total_tokens = 1024
    all_tokens = generate_deterministic_tokens(total_tokens, vocab=512, seed=42)

    results = []

    print("===================================================================")
    print("M8-W5: MULTI-CHUNK & BOUNDARY STRESS TEST MATRIX")
    print("===================================================================")
    print(f"Total Sequence Length: {total_tokens} tokens")
    print(f"Testing Chunks:        {chunk_sizes}")
    print(f"Testing Splits:        {[f'{s1}+{s2}' for s1, s2, _ in splits]}\n")

    with tempfile.TemporaryDirectory(prefix="gdn_stress_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        # File combined tokens
        comb_tok_file = tmp_path / "tokens_comb_1024.json"
        comb_tok_file.write_text(
            json.dumps({"tokens": all_tokens, "seq_len": total_tokens})
        )

        for c_val in chunk_sizes:
            # 1. Jalankan Single-pass baseline combined
            state_comb = tmp_path / f"state_comb_C{c_val}.bin"
            cmd_comb = [
                str(dismoen_bin),
                "gdn",
                "--model-dir",
                str(weights_path),
                "--tokens",
                str(comb_tok_file),
                "--output",
                str(state_comb),
                "--layers",
                str(args.layers),
                "--dk",
                str(args.dk),
                "--dv",
                str(args.dv),
                "--chunk-size",
                str(c_val),
                "--threads",
                "1",
            ]
            subprocess.run(cmd_comb, capture_output=True, check=True)

            for s1, s2, label in splits:
                # 2. Siapkan file token s1 dan s2 spesifik untuk split ini
                f_s1 = tmp_path / f"tok_{s1}_{s2}_part1.json"
                f_s2 = tmp_path / f"tok_{s1}_{s2}_part2.json"
                if not f_s1.exists():
                    f_s1.write_text(
                        json.dumps({"tokens": all_tokens[:s1], "seq_len": s1})
                    )
                if not f_s2.exists():
                    f_s2.write_text(
                        json.dumps({"tokens": all_tokens[s1 : s1 + s2], "seq_len": s2})
                    )

                state_s1 = tmp_path / f"state_s1_C{c_val}_{s1}.bin"
                state_cont = tmp_path / f"state_cont_C{c_val}_{s1}_{s2}.bin"

                # Step 1: Prefill seq1 -> simpan state
                cmd_s1 = [
                    str(dismoen_bin),
                    "gdn",
                    "--model-dir",
                    str(weights_path),
                    "--tokens",
                    str(f_s1),
                    "--output",
                    str(state_s1),
                    "--layers",
                    str(args.layers),
                    "--dk",
                    str(args.dk),
                    "--dv",
                    str(args.dv),
                    "--chunk-size",
                    str(c_val),
                    "--threads",
                    "1",
                ]
                subprocess.run(cmd_s1, capture_output=True, check=True)

                # Step 2: Continuation seq2 dari state_s1
                cmd_s2 = [
                    str(dismoen_bin),
                    "gdn",
                    "--model-dir",
                    str(weights_path),
                    "--tokens",
                    str(f_s2),
                    "--state-input",
                    str(state_s1),
                    "--output",
                    str(state_cont),
                    "--layers",
                    str(args.layers),
                    "--dk",
                    str(args.dk),
                    "--dv",
                    str(args.dv),
                    "--chunk-size",
                    str(c_val),
                    "--threads",
                    "1",
                ]
                subprocess.run(cmd_s2, capture_output=True, check=True)

                # Step 3: Compare candidate continuation vs combined baseline
                cmd_comp = [
                    str(dismoen_bin),
                    "compare",
                    "--reference",
                    str(state_comb),
                    "--candidate",
                    str(state_cont),
                    "--gate",
                    "G-M8-1",
                ]
                res_comp = subprocess.run(
                    cmd_comp, capture_output=True, text=True, check=False
                )
                try:
                    comp_data = json.loads(res_comp.stdout)
                    d_max = comp_data["metrics"]["delta_max"]
                    eps_rel = comp_data["metrics"]["epsilon_rel"]
                    verdict = comp_data["verdict"]
                except Exception as e:
                    print("Compare stdout:", res_comp.stdout)
                    print("Compare stderr:", res_comp.stderr)
                    raise e

                results.append(
                    {
                        "chunk": c_val,
                        "split": f"{s1}+{s2}",
                        "label": label,
                        "delta_max": d_max,
                        "eps_rel": eps_rel,
                        "verdict": verdict,
                    }
                )

                print(
                    f"--> [C={c_val:4d} | Split {s1:3d}+{s2:3d} ({label:<16})]"
                    f" Delta_max: {d_max:.4e} | Verdict: [{verdict}]"
                )

    print("\n===================================================================")
    print("MATRIKS HASIL STRESS TEST BOUNDARY CONTINUATION")
    print("===================================================================")
    hdr = (
        f"| {'Chunk':>6} | {'Split':>9} | {'Label':<17} | "
        f"{'Delta Max':>12} | {'Eps Rel':>11} | {'Gate G-M8-1':>11} |"
    )
    print(hdr)
    print("|" + "-" * 78 + "|")
    for r in results:
        row = (
            f"| {r['chunk']:>6d} | {r['split']:>9} | {r['label']:<17} | "
            f"{r['delta_max']:>12.4e} | {r['eps_rel']:>11.4e} | "
            f"{r['verdict']:>11} |"
        )
        print(row)

    all_pass = all(r["verdict"] == "PASS" for r in results)
    print("-------------------------------------------------------------------")
    if all_pass:
        print(
            f"STATUS: [PASS] Seluruh {len(results)} kombinasi boundary stress test"
            " MEMENUHI Gate G-M8-1!"
        )
    else:
        print("STATUS: [FAIL] Ditemukan regresi numerik pada boundary split!")
        sys.exit(1)


if __name__ == "__main__":
    main()
