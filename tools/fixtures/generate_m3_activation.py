#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generate canonical M3 activation fixture and reference MoE outputs.

Pipeline:
  1. Generate deterministic activation tensor: seed 42, shape [16, 2048] fp32 LE.
     Total bytes: 16 * 2048 * 4 = 131,072 bytes.
     Written to: fixtures/m3/activation.bin
  2. Execute PyTorch Oracle for layers 0, 12, 23:
     Written to:
       fixtures/m3/moe_ref_0.bin, routing_info_0.json
       fixtures/m3/moe_ref_12.bin, routing_info_12.json
       fixtures/m3/moe_ref_23.bin, routing_info_23.json
  3. Hash-pin all generated artifacts in fixtures/m3/SHA256SUMS.

Usage:
  uv run python tools/fixtures/generate_m3_activation.py
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
    os.path.join(os.path.dirname(__file__), "../../fixtures/m3")
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
        description="Generate M3 Activation and PyTorch MoE Oracle References"
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
            f"[2/3] Running PyTorch MoE Oracle on {args.model_dir} "
            f"for layers {layers}..."
        )
        for lyr in layers:
            ref_path = os.path.join(args.output_dir, f"moe_ref_{lyr}.bin")
            routing_path = os.path.join(args.output_dir, f"routing_info_{lyr}.json")
            cmd = [
                sys.executable,
                oracle_script,
                "--part",
                "moe",
                "--layer",
                str(lyr),
                "--activation",
                act_path,
                "--model-dir",
                args.model_dir,
                "--output",
                ref_path,
                "--routing-output",
                routing_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                sys.stderr.write(
                    f"ERROR: Oracle failed for layer {lyr} with exit "
                    f"code {res.returncode}:\n{res.stderr}\n"
                )
                sys.exit(res.returncode)

            report = json.loads(res.stdout.strip())
            print(
                f"  Layer {lyr}: wrote {ref_path} ({report['output_bytes']} bytes, "
                f"sha256={report['sha256'][:16]}...) and {routing_path}"
            )

    # 3. Compute and write SHA256SUMS for all artifacts in output_dir
    print(f"[3/3] Generating SHA256SUMS in {args.output_dir}...")
    sha_lines = []
    # Deterministic file order: activation.bin, then moe_ref_*.bin, routing_info_*.json
    files_to_hash = ["activation.bin"]
    for lyr in layers:
        files_to_hash.append(f"moe_ref_{lyr}.bin")
    for lyr in layers:
        files_to_hash.append(f"routing_info_{lyr}.json")

    for fname in files_to_hash:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            sha_lines.append(f"{digest}  {fname}\n")

    sums_path = os.path.join(args.output_dir, "SHA256SUMS")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.writelines(sha_lines)

    print(f"Generated {sums_path} with {len(sha_lines)} entries:")
    for line in sha_lines:
        print(f"  {line.strip()}")


if __name__ == "__main__":
    main()
