#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle MoE Router Slice — PyTorch FP32 Reference (M3-W1).

Pipeline:
  logits = F.linear(x, W_r) = x W_r^T  (shape: [L, num_experts])
  p = F.softmax(logits, dim=-1, dtype=torch.float32) (shape: [L, num_experts])
  topk_probs, topk_indices = torch.topk(p, k=top_k, dim=-1, sorted=True)

Validates:
  - fp32 softmax with numerical stability (max-shift)
  - top-4 selection TANPA renormalisasi (norm_topk_prob=false)
  - sum(topk_probs) <= 1.0 per token
  - 100% deterministic invariant on 256 random inputs
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


def fail(
    error_type: str, detail: str, stage: str = "router", extra: dict = None
) -> None:
    payload = {
        "error_type": error_type,
        "detail": detail,
        "stage": stage,
    }
    if extra:
        payload.update(extra)
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.exit(2)


def compute_router(
    x: torch.Tensor,
    w_router: torch.Tensor,
    top_k: int = 4,
    norm_topk_prob: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hitung router logits, probs, dan top-k.

    Returns:
        logits: [L, num_experts]
        probs: [L, num_experts]
        topk_probs: [L, top_k] (tanpa renormalisasi jika norm_topk_prob=False)
        topk_indices: [L, top_k] (indeks expert, 0-indexed)
    """
    logits = F.linear(x, w_router)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1, sorted=True)

    if norm_topk_prob:
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    return logits, probs, topk_probs, topk_indices


def generate_synthetic_router_fixture(
    seq_len: int = 256,
    hidden: int = 2048,
    num_experts: int = 60,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    # Skala normal untuk aktivasinya
    x = torch.randn(seq_len, hidden, dtype=torch.float32) * 0.1
    w_router = torch.randn(num_experts, hidden, dtype=torch.float32) * 0.02
    return x, w_router


def main():
    parser = argparse.ArgumentParser(
        description="Oracle MoE Router slice (PyTorch fp32)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run deterministic synthetic test"
    )
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dump-dir", type=str, default="", help="Dump binary fixtures if provided"
    )
    args = parser.parse_args()

    x, w_router = generate_synthetic_router_fixture(
        args.seq_len, args.hidden_size, args.num_experts, args.seed
    )
    logits, probs, topk_probs, topk_indices = compute_router(
        x, w_router, top_k=args.top_k, norm_topk_prob=False
    )

    # Validasi properti dasar
    assert logits.shape == (args.seq_len, args.num_experts)
    assert probs.shape == (args.seq_len, args.num_experts)
    assert topk_probs.shape == (args.seq_len, args.top_k)
    assert topk_indices.shape == (args.seq_len, args.top_k)

    # Properti softmax: jumlah seluruh probabilitas == 1.0
    sum_all = probs.sum(dim=-1)
    assert torch.allclose(
        sum_all, torch.ones_like(sum_all), atol=1e-5
    ), "Softmax does not sum to 1.0"

    # Properti norm_topk_prob=false:
    # Jumlah top-k probabilitas < 1.0 (karena num_experts=60 > top_k=4)
    sum_topk = topk_probs.sum(dim=-1)
    assert (sum_topk <= 1.0).all(), "Top-k sum exceeded 1.0"
    assert (
        sum_topk < 1.0
    ).all(), "Top-k sum unexpectedly equal to 1.0 (renormalization leak)"

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        x.numpy().tofile(os.path.join(args.dump_dir, "x.bin"))
        w_router.numpy().tofile(os.path.join(args.dump_dir, "w_router.bin"))
        logits.numpy().tofile(os.path.join(args.dump_dir, "logits_ref.bin"))
        probs.numpy().tofile(os.path.join(args.dump_dir, "probs_ref.bin"))
        topk_probs.numpy().tofile(os.path.join(args.dump_dir, "topk_probs_ref.bin"))
        topk_indices.numpy().astype(np.int32).tofile(
            os.path.join(args.dump_dir, "topk_indices_ref.bin")
        )

        metadata = {
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "seed": args.seed,
            "norm_topk_prob": False,
        }
        with open(os.path.join(args.dump_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dumped router oracle fixtures to {args.dump_dir}")

    if args.test:
        print("Oracle router test passed successfully.")


if __name__ == "__main__":
    main()
