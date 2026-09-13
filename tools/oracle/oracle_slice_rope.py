#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle RoPE rotate_half Slice — PyTorch FP32 Reference (M2-W2).

Formula F7:
  theta_i = m * omega_i, omega_i = base^(-2i / d_h)
  Rotation pairs: (q_i, q_{i + d_h / 2}) for Q and K with base = 1000000.0.

Validates:
  - Invariant Isometry (Property P-3): ||q'|| == ||q|| per head.
  - Scale invariance: RoPE(c * q) == c * RoPE(q).
  - Position 0 identity: RoPE(q, pos=0) == q.
  - Isolation from interleaved style (rules out FAIL category rope-style).
  - fp32 deterministic output matching Mojo implementation.
"""

import argparse
import json
import os
import sys
import torch


def fail(error_type: str, detail: str, stage: str = "rope", extra: dict = None) -> None:
    payload = {
        "error_type": error_type,
        "detail": detail,
        "stage": stage,
    }
    if extra:
        payload.update(extra)
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.exit(2)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def compute_rope(
    x: torch.Tensor,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    pos_offset: int = 0,
    base: float = 1000000.0,
) -> torch.Tensor:
    """Compute RoPE rotate_half in fp32.

    Input x shape: [seq_len, num_heads * head_dim] or [seq_len, num_heads, head_dim]
    """
    if head_dim % 2 != 0:
        fail("ROPE_ERROR", f"head_dim must be even, got {head_dim}")

    x_reshaped = x.view(seq_len, num_heads, head_dim)

    # inv_freq: [head_dim // 2]
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )

    # positions: [seq_len]
    t = torch.arange(pos_offset, pos_offset + seq_len, dtype=torch.float32)

    # freqs: [seq_len, half_dim]
    freqs = torch.outer(t, inv_freq)

    # emb: [seq_len, head_dim]
    emb = torch.cat((freqs, freqs), dim=-1)

    # cos, sin: [seq_len, 1, head_dim]
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)

    # Apply RoPE
    x_rot = (x_reshaped * cos) + (rotate_half(x_reshaped) * sin)
    return x_rot.view(seq_len, num_heads * head_dim)


def compute_interleaved_rope(
    x: torch.Tensor,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    pos_offset: int = 0,
    base: float = 1000000.0,
) -> torch.Tensor:
    """Compute RoPE interleaved style for difference assertion."""
    x_reshaped = x.view(seq_len, num_heads, head_dim)
    half_dim = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(pos_offset, pos_offset + seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    out = torch.zeros_like(x_reshaped)
    for s in range(seq_len):
        for h in range(num_heads):
            for i in range(half_dim):
                theta = freqs[s, i]
                c = torch.cos(theta)
                s_th = torch.sin(theta)
                x0 = x_reshaped[s, h, 2 * i]
                x1 = x_reshaped[s, h, 2 * i + 1]
                out[s, h, 2 * i] = x0 * c - x1 * s_th
                out[s, h, 2 * i + 1] = x0 * s_th + x1 * c
    return out.view(seq_len, num_heads * head_dim)


def verify_isometry(
    orig: torch.Tensor,
    rotated: torch.Tensor,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    tol: float = 1e-4,
) -> bool:
    """Verifies invariant isometry ||q'||_2 == ||q||_2 per head."""
    o = orig.view(seq_len, num_heads, head_dim)
    r = rotated.view(seq_len, num_heads, head_dim)
    for s in range(seq_len):
        for h in range(num_heads):
            norm_o = torch.norm(o[s, h], p=2).item()
            norm_r = torch.norm(r[s, h], p=2).item()
            diff = abs(norm_o - norm_r)
            max_norm = max(1.0, norm_o)
            if diff > tol * max_norm:
                return False
    return True


def run_tests() -> None:
    """Run comprehensive test suite against RoPE properties."""
    # 1. Identity at pos 0
    x0 = torch.randn(1, 4 * 128, dtype=torch.float32)
    y0 = compute_rope(x0, seq_len=1, num_heads=4, head_dim=128, pos_offset=0)
    assert torch.allclose(x0, y0, atol=1e-6), "RoPE at pos 0 must be identity"

    # 2. Known vector check (d_h = 4, pos = 1)
    x_known = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    y_known = compute_rope(x_known, seq_len=1, num_heads=1, head_dim=4, pos_offset=1)
    expected_known = torch.tensor(
        [[-1.9841106, 1.9959991, 2.4623780, 4.0019979]], dtype=torch.float32
    )
    assert torch.allclose(
        y_known, expected_known, atol=1e-5
    ), f"Known vector mismatch: {y_known} vs {expected_known}"

    # 3. Invariant Isometry (Property P-3) across positions and layers
    for pos in [0, 1, 15, 100, 2048]:
        x_test = torch.randn(3, 16 * 128, dtype=torch.float32)
        y_test = compute_rope(
            x_test, seq_len=3, num_heads=16, head_dim=128, pos_offset=pos
        )
        assert verify_isometry(
            x_test, y_test, seq_len=3, num_heads=16, head_dim=128
        ), f"Isometry failed at pos {pos}"

    # 4. Scale Invariance: RoPE(c * q) == c * RoPE(q)
    c = 3.5
    x_base = torch.randn(2, 8 * 128, dtype=torch.float32)
    y_base = compute_rope(x_base, seq_len=2, num_heads=8, head_dim=128, pos_offset=5)
    y_scaled = compute_rope(
        x_base * c, seq_len=2, num_heads=8, head_dim=128, pos_offset=5
    )
    assert torch.allclose(
        y_scaled, y_base * c, atol=1e-5
    ), "RoPE scale invariance failed"

    # 5. Distinction vs interleaved style
    y_inter = compute_interleaved_rope(
        x_known, seq_len=1, num_heads=1, head_dim=4, pos_offset=1
    )
    diff = (y_known - y_inter).abs().max().item()
    assert (
        diff > 0.5
    ), f"RoPE rotate_half must be distinct from interleaved, got diff {diff}"


def main():
    parser = argparse.ArgumentParser(
        description="Oracle RoPE rotate_half slice (PyTorch fp32)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run comprehensive property tests"
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--pos-offset", type=int, default=0)
    parser.add_argument("--base", type=float, default=1000000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="",
        help="Dump binary fixtures if provided",
    )
    args = parser.parse_args()

    if args.test:
        run_tests()
        print(
            json.dumps(
                {
                    "status": "success",
                    "test": "passed",
                    "formula": "F7",
                    "style": "rotate_half",
                    "invariant": "isometry_P3",
                }
            )
        )
        return

    torch.manual_seed(args.seed)
    hidden = args.num_heads * args.head_dim
    x = torch.randn(args.seq_len, hidden, dtype=torch.float32) * 0.1
    y = compute_rope(
        x,
        args.seq_len,
        args.num_heads,
        args.head_dim,
        args.pos_offset,
        args.base,
    )

    if not verify_isometry(x, y, args.seq_len, args.num_heads, args.head_dim):
        fail("ROPE_ERROR", "Isometry check failed on generated output")

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        x.numpy().tofile(os.path.join(args.dump_dir, "rope_in.bin"))
        y.numpy().tofile(os.path.join(args.dump_dir, "rope_out_ref.bin"))
        print(f"Dumped RoPE fixtures to {args.dump_dir}")

    print(
        json.dumps(
            {
                "status": "success",
                "seq_len": args.seq_len,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "base": args.base,
                "input_norm": float(torch.norm(x)),
                "output_norm": float(torch.norm(y)),
                "isometry_verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
