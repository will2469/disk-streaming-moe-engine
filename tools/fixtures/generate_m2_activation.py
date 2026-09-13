#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generate canonical M2 activation fixture and reference attention outputs.

Pipeline:
  1. Generate deterministic activation tensor: seed 42, shape [16, 2048] fp32 LE.
     Total bytes: 16 * 2048 * 4 = 131,072 bytes.
     Written to: fixtures/m2/activation.bin
  2. Execute PyTorch Oracle for layers 0, 12, 23:
     Written to: fixtures/m2/attn_ref_0.bin, attn_ref_12.bin, attn_ref_23.bin
  3. Hash-pin all generated artifacts in fixtures/m2/SHA256SUMS.

Usage:
  uv run python tools/fixtures/generate_m2_activation.py
"""

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys

DEFAULT_MODEL_DIR = "/home/will/models/qwen1.5-moe-a2.7b-chat"
DEFAULT_OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../fixtures/m2")
)


def generate_activation_tensor(num_tokens: int, hidden_dim: int, seed: int) -> bytes:
    """Generate deterministic fp32 LE activation bytes using seed 42."""
    try:
        import torch

        torch.manual_seed(seed)
        x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32) * 0.1
        return x.detach().cpu().numpy().tobytes()
    except ImportError:
        import random

        rng = random.Random(seed)
        out = bytearray(num_tokens * hidden_dim * 4)
        for i in range(num_tokens * hidden_dim):
            val = rng.gauss(0.0, 0.1)
            struct.pack_into("<f", out, i * 4, val)
        return bytes(out)


def main():
    parser = argparse.ArgumentParser(
        description="Generate M2 Activation and PyTorch Oracle References"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=16,
        help="Sequence length (default: 16)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden dimension (default: 2048)",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Model directory with safetensors (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for fixtures (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--layers",
        default="0,12,23",
        help="Comma-separated list of layers to process (default: '0,12,23')",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    # 1. Generate activation.bin
    act_bytes = generate_activation_tensor(args.num_tokens, args.hidden_dim, args.seed)
    expected_size = args.num_tokens * args.hidden_dim * 4
    if len(act_bytes) != expected_size:
        sys.stderr.write(
            f"ERROR: activation bytes length mismatch: "
            f"{len(act_bytes)} != {expected_size}\n"
        )
        sys.exit(1)

    act_path = os.path.join(args.output_dir, "activation.bin")
    with open(act_path, "wb") as f:
        f.write(act_bytes)
    print(f"[1/3] Generated activation: {act_path} ({len(act_bytes)} bytes)")

    # 2. Run PyTorch Oracle for each layer
    oracle_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../oracle/oracle_layer.py")
    )
    has_model = os.path.isdir(args.model_dir) and (
        os.path.exists(os.path.join(args.model_dir, "model.safetensors.index.json"))
        or os.path.exists(os.path.join(args.model_dir, "model_config.json"))
        or os.path.exists(os.path.join(args.model_dir, "config.json"))
    )

    if not has_model:
        sys.stderr.write(
            f"WARNING: Model directory not found or incomplete: {args.model_dir}\n"
            "Skipping oracle reference generation. "
            "Existing references will be preserved.\n"
        )
    else:
        print(
            f"[2/3] Running PyTorch Oracle on {args.model_dir} for layers {layers}..."
        )
        for lyr in layers:
            ref_path = os.path.join(args.output_dir, f"attn_ref_{lyr}.bin")
            cmd = [
                sys.executable,
                oracle_script,
                "--part",
                "attn",
                "--layer",
                str(lyr),
                "--activation",
                act_path,
                "--model-dir",
                args.model_dir,
                "--output",
                ref_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                sys.stderr.write(
                    f"ERROR: Oracle failed for layer {lyr} with exit "
                    f"code {res.returncode}:\n{res.stderr}\n"
                )
                sys.exit(res.returncode)

            report = json.loads(res.stdout.strip())
            out_sz = report["output_bytes"]
            out_sha = report["sha256"]
            print(
                f"  - Layer {lyr}: wrote {ref_path} ({out_sz} bytes, sha256={out_sha})"
            )

    # 3. Hash-pinning SHA256SUMS
    print("[3/3] Generating fixtures/m2/SHA256SUMS...")
    sums = []
    for fn in sorted(os.listdir(args.output_dir)):
        if fn == "SHA256SUMS":
            continue
        p = os.path.join(args.output_dir, fn)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            sums.append(f"{h}  {fn}")

    sums_path = os.path.join(args.output_dir, "SHA256SUMS")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sums) + "\n")

    print(f"Generated M2 fixture successfully in {args.output_dir}:")
    for s in sums:
        print(f"  {s}")


if __name__ == "__main__":
    main()
