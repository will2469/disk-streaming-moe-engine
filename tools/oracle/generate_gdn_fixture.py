#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generator untuk fixture sintetis GDN (M8-W1).

Membuat weights safetensors deterministik dan tokens JSON untuk keperluan
pengujian CI tanpa memerlukan checkpoint model besar (DoD M8).
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic GDN weights and tokens fixture"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Jumlah layer GDN (default: 2)",
    )
    parser.add_argument(
        "--dk",
        type=int,
        default=32,
        help="Dimensi key dk (default: 32)",
    )
    parser.add_argument(
        "--dv",
        type=int,
        default=32,
        help="Dimensi value dv (default: 32)",
    )
    parser.add_argument(
        "--vocab",
        type=int,
        default=512,
        help="Ukuran vocabulary (default: 512)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Panjang sequence tokens (default: 16)",
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
        default="fixtures/m8_gdn_weights.safetensors",
        help="Path output weights safetensors",
    )
    parser.add_argument(
        "--tokens-output",
        type=str,
        default="fixtures/m8_tokens.json",
        help="Path output tokens JSON (opsional)",
    )
    return parser.parse_args()


def generate_tokens(seq_len: int, vocab_size: int, seed: int) -> list[int]:
    """Menghasilkan tokens deterministik."""
    if seq_len == 16 and vocab_size >= 312:
        return [
            1,
            23,
            45,
            67,
            89,
            101,
            123,
            145,
            167,
            189,
            201,
            223,
            245,
            267,
            289,
            311,
        ]

    gen = torch.Generator().manual_seed(seed)
    rand_ints = torch.randint(
        low=1,
        high=vocab_size,
        size=(seq_len,),
        generator=gen,
    )
    return [int(x) for x in rand_ints]


def generate_synthetic_weights(
    layers: int,
    dk: int,
    dv: int,
    vocab: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Membangkitkan bobot matriks GDN sintetis."""
    gen = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}

    # Embedding table: [vocab, dk]
    scale_emb = 1.0 / (dk**0.5)
    tensors["embed_tokens.weight"] = (
        torch.randn((vocab, dk), generator=gen, dtype=torch.float32) * scale_emb
    )

    scale_k = 1.0 / (dk**0.5)
    scale_v = 1.0 / (dk**0.5)

    for lyr in range(layers):
        # Key projection: [dk, dk]
        tensors[f"layers.{lyr}.k_proj.weight"] = (
            torch.randn((dk, dk), generator=gen, dtype=torch.float32) * scale_k
        )
        # Value projection: [dv, dk]
        tensors[f"layers.{lyr}.v_proj.weight"] = (
            torch.randn((dv, dk), generator=gen, dtype=torch.float32) * scale_v
        )
        # Beta gate projection: [1, dk]
        tensors[f"layers.{lyr}.beta_proj.weight"] = (
            torch.randn((1, dk), generator=gen, dtype=torch.float32) * scale_k
        )

    return tensors


def main():
    args = parse_args()

    if args.layers <= 0 or args.dk <= 0 or args.dv <= 0 or args.vocab <= 0:
        sys.stderr.write(
            "ERROR: layers, dk, dv, dan vocab harus bilangan bulat positif\n"
        )
        sys.exit(1)

    # Pastikan direktori output ada
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. Bangkitkan bobot safetensors
    weights = generate_synthetic_weights(
        layers=args.layers,
        dk=args.dk,
        dv=args.dv,
        vocab=args.vocab,
        seed=args.seed,
    )
    save_file(weights, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(
        f"Generated weights: {args.output} ({size_mb:.3f} MB, {len(weights)} tensors)"
    )

    # 2. Bangkitkan tokens JSON bila diminta
    if args.tokens_output:
        tok_dir = os.path.dirname(args.tokens_output)
        if tok_dir:
            os.makedirs(tok_dir, exist_ok=True)
        tokens_list = generate_tokens(
            seq_len=args.seq_len,
            vocab_size=args.vocab,
            seed=args.seed,
        )
        token_payload = {
            "tokens": tokens_list,
            "seq_len": len(tokens_list),
        }
        with open(args.tokens_output, "w", encoding="utf-8") as f:
            json.dump(token_payload, f, indent=2)
            f.write("\n")
        print(f"Generated tokens: {args.tokens_output} ({len(tokens_list)} tokens)")


if __name__ == "__main__":
    main()
