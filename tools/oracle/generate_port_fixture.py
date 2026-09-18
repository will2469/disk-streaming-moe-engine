#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generator untuk fixture sintetis port Qwen3.6 (M9-W4).

Membuat weights Safetensors deterministik dan tokens JSON untuk keperluan
pengujian CI tanpa memerlukan checkpoint model 70 GB (DoD M9).
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Qwen3.6 port weights and tokens fixture"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Jumlah layer (default: 4 untuk pola 3xGDN + 1xGatedAttn)",
    )
    parser.add_argument(
        "--routed-experts",
        type=int,
        default=8,
        help="Jumlah routed experts (default: 8)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=2,
        help="Top-k experts aktif per token (default: 2)",
    )
    parser.add_argument(
        "--inter-dim",
        type=int,
        default=64,
        help="Dimensi intermediate MoE (default: 64)",
    )
    parser.add_argument(
        "--vocab",
        type=int,
        default=1024,
        help="Ukuran vocabulary (default: 1024)",
    )
    parser.add_argument(
        "--dh",
        type=int,
        default=32,
        help="Dimensi per attention head (default: 32)",
    )
    parser.add_argument(
        "--gqa-q",
        type=int,
        default=4,
        help="Jumlah query heads (default: 4)",
    )
    parser.add_argument(
        "--gqa-kv",
        type=int,
        default=1,
        help="Jumlah key/value heads (default: 1)",
    )
    parser.add_argument(
        "--dk",
        type=int,
        default=32,
        help="Dimensi key GDN (default: 32)",
    )
    parser.add_argument(
        "--dv",
        type=int,
        default=32,
        help="Dimensi value GDN (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed deterministik (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fixtures/m9_port_weights.safetensors",
        help="Path output weights safetensors",
    )
    parser.add_argument(
        "--tokens-output",
        type=str,
        default="fixtures/m9_port_tokens.json",
        help="Path output tokens JSON",
    )
    return parser.parse_args()


def generate_tokens(seq_len: int = 8) -> list[int]:
    """Menghasilkan sequence tokens deterministik standar M9."""
    return [1, 23, 45, 67, 89, 101, 123, 145]


def generate_synthetic_port_weights(
    layers: int,
    routed_experts: int,
    inter_dim: int,
    vocab: int,
    dh: int,
    gqa_q: int,
    gqa_kv: int,
    dk: int,
    dv: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Membangkitkan bobot matriks port hybrid sintetis deterministik (BF16)."""
    gen = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}

    hidden_size = gqa_q * dh  # 4 * 32 = 128
    q_dim = gqa_q * dh  # 128
    kv_dim = gqa_kv * dh  # 32
    shared_inter_dim = inter_dim  # 64

    # 1. Embedding & LM Head: [vocab, hidden_size]
    scale_emb = 1.0 / (hidden_size**0.5)
    tensors["model.language_model.embed_tokens.weight"] = (
        torch.randn(vocab, hidden_size, generator=gen, dtype=torch.float32) * scale_emb
    ).to(torch.bfloat16)

    tensors["lm_head.weight"] = (
        torch.randn(vocab, hidden_size, generator=gen, dtype=torch.float32) * scale_emb
    ).to(torch.bfloat16)

    # 2. Final RMSNorm: [hidden_size]
    tensors["model.language_model.norm.weight"] = (
        torch.ones(hidden_size, dtype=torch.float32)
        + torch.randn(hidden_size, generator=gen, dtype=torch.float32) * 0.01
    ).to(torch.bfloat16)

    # 3. 4-Layer Hybrid Transformer Blocks
    for lyr in range(layers):
        pfx = f"model.language_model.layers.{lyr}."

        # Input & Post-Attention Norms
        tensors[pfx + "input_layernorm.weight"] = (
            torch.ones(hidden_size, dtype=torch.float32)
            + torch.randn(hidden_size, generator=gen, dtype=torch.float32) * 0.01
        ).to(torch.bfloat16)

        tensors[pfx + "post_attention_layernorm.weight"] = (
            torch.ones(hidden_size, dtype=torch.float32)
            + torch.randn(hidden_size, generator=gen, dtype=torch.float32) * 0.01
        ).to(torch.bfloat16)

        is_linear_attn = (lyr % 4) != 3

        if is_linear_attn:
            # Token Mixer: Gated DeltaNet (GDN)
            scale_k = 1.0 / (hidden_size**0.5)
            scale_v = 1.0 / (hidden_size**0.5)
            scale_out = 1.0 / (dv**0.5)

            tensors[pfx + "linear_attn.k_proj.weight"] = (
                torch.randn(dk, hidden_size, generator=gen, dtype=torch.float32)
                * scale_k
            ).to(torch.bfloat16)

            tensors[pfx + "linear_attn.v_proj.weight"] = (
                torch.randn(dv, hidden_size, generator=gen, dtype=torch.float32)
                * scale_v
            ).to(torch.bfloat16)

            tensors[pfx + "linear_attn.beta_proj.weight"] = (
                torch.randn(hidden_size, generator=gen, dtype=torch.float32) * 0.05
            ).to(torch.bfloat16)

            tensors[pfx + "linear_attn.out_proj.weight"] = (
                torch.randn(hidden_size, dv, generator=gen, dtype=torch.float32)
                * scale_out
            ).to(torch.bfloat16)
        else:
            # Token Mixer: Gated Attention (GQA)
            scale_q = 1.0 / (hidden_size**0.5)
            scale_kv = 1.0 / (hidden_size**0.5)
            scale_o = 1.0 / (q_dim**0.5)

            tensors[pfx + "self_attn.q_proj.weight"] = (
                torch.randn(q_dim, hidden_size, generator=gen, dtype=torch.float32)
                * scale_q
            ).to(torch.bfloat16)

            tensors[pfx + "self_attn.k_proj.weight"] = (
                torch.randn(kv_dim, hidden_size, generator=gen, dtype=torch.float32)
                * scale_kv
            ).to(torch.bfloat16)

            tensors[pfx + "self_attn.v_proj.weight"] = (
                torch.randn(kv_dim, hidden_size, generator=gen, dtype=torch.float32)
                * scale_kv
            ).to(torch.bfloat16)

            tensors[pfx + "self_attn.gate_proj.weight"] = (
                torch.randn(q_dim, hidden_size, generator=gen, dtype=torch.float32)
                * scale_q
            ).to(torch.bfloat16)

            tensors[pfx + "self_attn.o_proj.weight"] = (
                torch.randn(hidden_size, q_dim, generator=gen, dtype=torch.float32)
                * scale_o
            ).to(torch.bfloat16)

        # Channel Mixer: MoE MLP pada SETIAP transformer block
        scale_router = 1.0 / (hidden_size**0.5)
        tensors[pfx + "mlp.gate.weight"] = (
            torch.randn(routed_experts, hidden_size, generator=gen, dtype=torch.float32)
            * scale_router
        ).to(torch.bfloat16)

        scale_inter = 1.0 / (hidden_size**0.5)
        scale_down = 1.0 / (inter_dim**0.5)

        for e in range(routed_experts):
            tensors[pfx + f"mlp.experts.{e}.gate_proj.weight"] = (
                torch.randn(inter_dim, hidden_size, generator=gen, dtype=torch.float32)
                * scale_inter
            ).to(torch.bfloat16)

            tensors[pfx + f"mlp.experts.{e}.up_proj.weight"] = (
                torch.randn(inter_dim, hidden_size, generator=gen, dtype=torch.float32)
                * scale_inter
            ).to(torch.bfloat16)

            tensors[pfx + f"mlp.experts.{e}.down_proj.weight"] = (
                torch.randn(hidden_size, inter_dim, generator=gen, dtype=torch.float32)
                * scale_down
            ).to(torch.bfloat16)

        # Shared Expert
        scale_sh_down = 1.0 / (shared_inter_dim**0.5)
        tensors[pfx + "mlp.shared_expert.gate_proj.weight"] = (
            torch.randn(
                shared_inter_dim, hidden_size, generator=gen, dtype=torch.float32
            )
            * scale_inter
        ).to(torch.bfloat16)

        tensors[pfx + "mlp.shared_expert.up_proj.weight"] = (
            torch.randn(
                shared_inter_dim, hidden_size, generator=gen, dtype=torch.float32
            )
            * scale_inter
        ).to(torch.bfloat16)

        tensors[pfx + "mlp.shared_expert.down_proj.weight"] = (
            torch.randn(
                hidden_size, shared_inter_dim, generator=gen, dtype=torch.float32
            )
            * scale_sh_down
        ).to(torch.bfloat16)

        tensors[pfx + "mlp.shared_expert_gate.weight"] = (
            torch.randn(hidden_size, generator=gen, dtype=torch.float32) * 0.05
        ).to(torch.bfloat16)

    return tensors


def main():
    args = parse_args()
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    output_path = (
        os.path.join(repo_root, args.output)
        if not os.path.isabs(args.output)
        else args.output
    )
    tokens_output_path = (
        os.path.join(repo_root, args.tokens_output)
        if not os.path.isabs(args.tokens_output)
        else args.tokens_output
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(tokens_output_path)), exist_ok=True)

    # 1. Generate & save tokens JSON
    tokens = generate_tokens(seq_len=8)
    tokens_payload = {
        "tokens": tokens,
        "seq_len": len(tokens),
    }
    with open(tokens_output_path, "w", encoding="utf-8") as f:
        json.dump(tokens_payload, f, indent=2)
        f.write("\n")
    print(f"Generated tokens fixture: {tokens_output_path} ({len(tokens)} tokens)")

    # 2. Generate & save weights Safetensors
    tensors = generate_synthetic_port_weights(
        layers=args.layers,
        routed_experts=args.routed_experts,
        inter_dim=args.inter_dim,
        vocab=args.vocab,
        dh=args.dh,
        gqa_q=args.gqa_q,
        gqa_kv=args.gqa_kv,
        dk=args.dk,
        dv=args.dv,
        seed=args.seed,
    )

    save_file(tensors, output_path)
    file_size = os.path.getsize(output_path)

    # 3. Hitung & simpan SHA-256
    hasher = hashlib.sha256()
    with open(output_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256 = hasher.hexdigest()

    sha_path = output_path + ".sha256"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{sha256}  {os.path.basename(output_path)}\n")

    print(
        f"Generated synthetic weights: {output_path} "
        f"({file_size} bytes, {len(tensors)} tensors, SHA256: {sha256[:16]}...)"
    )

    # Guard DoD: weights harus < 10 MB untuk CI
    if file_size >= 10 * 1024 * 1024:
        sys.stderr.write(f"ERROR: file size {file_size} exceeds 10 MB limit\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
