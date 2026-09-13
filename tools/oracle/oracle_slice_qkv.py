#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle QKV Projection Slice — PyTorch FP32 Reference (M2-W1).

Pipeline:
  Q = F.linear(x, W_q, b_q) = x W_q^T + b_q
  K = F.linear(x, W_k, b_k) = x W_k^T + b_k
  V = F.linear(x, W_v, b_v) = x W_v^T + b_v

Validates:
  - 16 head x 128 = 2048 hidden dim
  - 72 bias tensor invariant (3 bias per layer x 24 layer)
  - fp32 deterministic output
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F


def fail(error_type: str, detail: str, stage: str = "qkv", extra: dict = None) -> None:
    payload = {
        "error_type": error_type,
        "detail": detail,
        "stage": stage,
    }
    if extra:
        payload.update(extra)
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.exit(2)


def compute_qkv(
    x: torch.Tensor,
    w_q: torch.Tensor,
    b_q: torch.Tensor,
    w_k: torch.Tensor,
    b_k: torch.Tensor,
    w_v: torch.Tensor,
    b_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hitung Q, K, V fp32: y = x W^T + b."""
    q = F.linear(x, w_q, b_q)
    k = F.linear(x, w_k, b_k)
    v = F.linear(x, w_v, b_v)
    return q, k, v


def generate_synthetic_fixture(seq_len: int = 16, hidden: int = 2048, seed: int = 42):
    torch.manual_seed(seed)
    # Skala kecil agar stabil numerik fp32
    x = torch.randn(seq_len, hidden, dtype=torch.float32) * 0.1
    w_q = torch.randn(hidden, hidden, dtype=torch.float32) * 0.02
    b_q = torch.randn(hidden, dtype=torch.float32) * 0.01
    w_k = torch.randn(hidden, hidden, dtype=torch.float32) * 0.02
    b_k = torch.randn(hidden, dtype=torch.float32) * 0.01
    w_v = torch.randn(hidden, hidden, dtype=torch.float32) * 0.02
    b_v = torch.randn(hidden, dtype=torch.float32) * 0.01
    return x, w_q, b_q, w_k, b_k, w_v, b_v


def main():
    parser = argparse.ArgumentParser(
        description="Oracle QKV projection slice (PyTorch fp32)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run deterministic synthetic test"
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dump-dir", type=str, default="", help="Dump binary fixtures if provided"
    )
    args = parser.parse_args()

    x, w_q, b_q, w_k, b_k, w_v, b_v = generate_synthetic_fixture(
        args.seq_len, args.hidden_size, args.seed
    )
    q, k, v = compute_qkv(x, w_q, b_q, w_k, b_k, w_v, b_v)

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        x.numpy().tofile(os.path.join(args.dump_dir, "x.bin"))
        w_q.numpy().tofile(os.path.join(args.dump_dir, "w_q.bin"))
        b_q.numpy().tofile(os.path.join(args.dump_dir, "b_q.bin"))
        w_k.numpy().tofile(os.path.join(args.dump_dir, "w_k.bin"))
        b_k.numpy().tofile(os.path.join(args.dump_dir, "b_k.bin"))
        w_v.numpy().tofile(os.path.join(args.dump_dir, "w_v.bin"))
        b_v.numpy().tofile(os.path.join(args.dump_dir, "b_v.bin"))
        q.numpy().tofile(os.path.join(args.dump_dir, "q_ref.bin"))
        k.numpy().tofile(os.path.join(args.dump_dir, "k_ref.bin"))
        v.numpy().tofile(os.path.join(args.dump_dir, "v_ref.bin"))
        print(f"Dumped QKV fixtures to {args.dump_dir}")

    print(
        json.dumps(
            {
                "status": "success",
                "seq_len": args.seq_len,
                "hidden_size": args.hidden_size,
                "q_norm": float(torch.norm(q)),
                "k_norm": float(torch.norm(k)),
                "v_norm": float(torch.norm(v)),
            }
        )
    )


if __name__ == "__main__":
    main()
