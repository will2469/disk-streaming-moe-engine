#!/usr/bin/env python3
"""Generate synthetic M0 fixture: mini-checkpoint deterministik (seed 42).

Output di fixtures/m0/: 3 shard + model.safetensors.index.json + model_config.json.
Setiap shard F15-valid (kontigu, dtype whitelist, layout eksak) agar lolos check-index.
Jalankan: uv run tools/fixtures/generate_m0_fixture.py
"""

import hashlib
import json
import os
import random
import struct

SEED = 42
OUT = os.path.join(os.path.dirname(__file__), "../../fixtures/m0")
DTYPES = {"BF16": 2, "F32": 4}

# Mini config: L=2, H=2, 8 routed + 1 shared, d_e=64, vocab=512
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

SHARD_OF = {}  # name -> shard idx (round-robin agar merge diuji lintas file)


def main() -> None:
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    by_shard: dict[int, list] = {0: [], 1: [], 2: []}
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
            blobs.append(rng.randbytes(ln))
            off += ln
            weight_map[name] = f"fixture-0000{idx + 1}-of-00003.safetensors"
        hb = json.dumps(hdr, separators=(",", ":")).encode()
        path = os.path.join(OUT, f"fixture-0000{idx + 1}-of-00003.safetensors")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)
    with open(os.path.join(OUT, "model_config.json"), "w") as f:
        json.dump(
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64,
                "vocab_size": 512,
                "norm_topk_prob": False,
            },
            f,
            indent=2,
        )
    sums = []
    for fn in sorted(os.listdir(OUT)):
        if fn == "SHA256SUMS":
            continue  # jangan hash diri sendiri (pecah determinisme antar-run)
        p = os.path.join(OUT, fn)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        sums.append(f"{h}  {fn}")
    open(os.path.join(OUT, "SHA256SUMS"), "w").write("\n".join(sums) + "\n")
    print(f"fixture ok: {len(TENSORS)} tensor, 3 shard di {OUT}")


if __name__ == "__main__":
    main()
