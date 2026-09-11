#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generate synthetic M1 fixture: canonical model config, tokens, shards,
and reference logits.

Output in fixtures/m1/:
  - model_config.json (canonical, derived from M0 + rms_norm_eps + provenance)
  - tokens.json (3 prompts x 16 tokens)
  - model.safetensors.index.json (index mapping required tensors)
  - fixture-00001-of-00003.safetensors
  - fixture-00002-of-00003.safetensors
  - fixture-00003-of-00003.safetensors
  - logits_ref.bin (reference fp32 logits from PyTorch oracle)
  - SHA256SUMS (hash-pinning all artifacts)

Jalankan: .venv/bin/python3 tools/fixtures/generate_m1_tokens.py
"""

import hashlib
import json
import os
import random
import struct
import subprocess
import sys

SEED = 42
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
M0_DIR = os.path.join(ROOT, "fixtures/m0")
OUT_DIR = os.path.join(ROOT, "fixtures/m1")

DTYPES = {"BF16": 2, "F32": 4}

# Mini config matching M0: L=2, H=2, 8 routed + 1 shared, d_e=64, vocab=512
TENSORS = [
    ("model.embed_tokens.weight", "BF16", [512, 64]),
    ("model.norm.weight", "BF16", [64]),
]
for lyr in (0, 1):
    p = f"model.layers.{lyr}"
    TENSORS += [
        (f"{p}.self_attn.q_proj.weight", "BF16", [64, 64]),
        (f"{p}.self_attn.q_proj.bias", "BF16", [64]),
        (f"{p}.self_attn.k_proj.weight", "BF16", [64, 64]),
        (f"{p}.self_attn.o_proj.weight", "BF16", [64, 64]),
        (f"{p}.mlp.gate.weight", "BF16", [8, 64]),
        (f"{p}.mlp.experts.0.w1.weight", "BF16", [64, 64]),
        (f"{p}.mlp.shared_expert.w1.weight", "BF16", [64, 64]),
        (f"{p}.input_layernorm.weight", "BF16", [64]),
    ]
TENSORS.append(("lm_head.weight", "BF16", [512, 64]))


def float_to_bf16_bytes(val: float) -> bytes:
    u32 = struct.unpack("<I", struct.pack("<f", val))[0]
    u16 = (u32 >> 16) & 0xFFFF
    return struct.pack("<H", u16)


def generate_finite_bf16_blob(n_elements: int, rng: random.Random) -> bytes:
    out = bytearray(n_elements * 2)
    for i in range(n_elements):
        val = rng.uniform(-0.25, 0.25)
        u32 = struct.unpack("<I", struct.pack("<f", val))[0]
        struct.pack_into("<H", out, i * 2, (u32 >> 16) & 0xFFFF)
    return bytes(out)


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Derive canonical model_config.json from M0
    m0_config_path = os.path.join(M0_DIR, "model_config.json")
    with open(m0_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Add canonical rms_norm_eps (1e-6)
    cfg["rms_norm_eps"] = 1e-6
    m1_config_path = os.path.join(OUT_DIR, "model_config.json")
    with open(m1_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    # 2. Generate 3 golden prompts x 16 tokens in range [0..vocab_size)
    vocab_size = cfg["vocab_size"]
    tokens = []
    for _ in range(3):
        prompt = [rng.randint(0, vocab_size - 1) for _ in range(16)]
        tokens.append(prompt)

    tokens_path = os.path.join(OUT_DIR, "tokens.json")
    with open(tokens_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, indent=2)
        f.write("\n")

    # 3. Partition tensors across 3 shards (round-robin)
    by_shard = {0: [], 1: [], 2: []}
    for i, (name, dtype, shape) in enumerate(TENSORS):
        by_shard[i % 3].append((name, dtype, shape))

    weight_map = {}
    for idx in (0, 1, 2):
        items = by_shard[idx]
        hdr, off, blobs = {}, 0, []
        for name, dtype, shape in items:
            n = 1
            for d in shape:
                n *= d
            ln = n * DTYPES[dtype]
            hdr[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [off, off + ln],
            }
            # Generate finite deterministic BF16 values
            blobs.append(generate_finite_bf16_blob(n, rng))
            off += ln
            weight_map[name] = f"fixture-0000{idx + 1}-of-00003.safetensors"

        hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        shard_name = f"fixture-0000{idx + 1}-of-00003.safetensors"
        shard_path = os.path.join(OUT_DIR, shard_name)
        with open(shard_path, "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))

    index_path = os.path.join(OUT_DIR, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": {"total_size": 0}, "weight_map": weight_map},
            f,
            indent=2,
        )
        f.write("\n")

    # 4. Run PyTorch Oracle to generate logits_ref.bin
    logits_ref_path = os.path.join(OUT_DIR, "logits_ref.bin")
    oracle_script = os.path.join(ROOT, "tools/oracle/oracle_head.py")
    cmd = [
        sys.executable,
        oracle_script,
        "--model-dir",
        OUT_DIR,
        "--tokens",
        tokens_path,
        "--output",
        logits_ref_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"Oracle failed with code {res.returncode}:\n{res.stderr}\n")
        sys.exit(1)
    print("Oracle output:", res.stdout.strip())

    # 5. Generate SHA256SUMS for all files in fixtures/m1/
    sums = []
    for fn in sorted(os.listdir(OUT_DIR)):
        if fn == "SHA256SUMS":
            continue
        p = os.path.join(OUT_DIR, fn)
        if os.path.isfile(p):
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            sums.append(f"{h}  {fn}")

    sums_path = os.path.join(OUT_DIR, "SHA256SUMS")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sums) + "\n")

    print(f"Generated M1 fixture successfully in {OUT_DIR}:")
    for s in sums:
        print(f"  {s}")


if __name__ == "__main__":
    main()
