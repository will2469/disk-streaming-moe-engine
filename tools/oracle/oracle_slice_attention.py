#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle Attention Block Slice — PyTorch FP32 Reference (M2-W3).

Pipeline:
  1. x_norm = rmsnorm(x, norm_gamma, eps=1e-6)
  2. Q, K, V = qkv_project(x_norm, W_q, b_q, W_k, b_k, W_v, b_v)
  3. Q_rot, K_rot = apply_rope(Q, K, pos_offset, base=1000000.0)
  4. attn_out = causal_mha(Q_rot, K_rot, V)
  5. y = o_project(attn_out, W_o, b_o)
  6. y_final = y + x (residual connection)

Validates:
  - Causal masking: triangular mask (output token t depends only on <= t)
  - Numerically stable softmax: exp(z - max z) / sum exp(z - max z)
  - Scale factor: 1.0 / sqrt(head_dim)
  - fp32 deterministic output matching Mojo implementation.
"""

import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F


def fail(
    error_type: str, detail: str, stage: str = "attention", extra: dict = None
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


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm per-token (F6)."""
    var = torch.mean(x**2, dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * gamma


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    x: torch.Tensor,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    pos_offset: int = 0,
    base: float = 1000000.0,
) -> torch.Tensor:
    """Apply RoPE rotate_half F7."""
    x_reshaped = x.view(seq_len, num_heads, head_dim)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(pos_offset, pos_offset + seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
    x_rot = (x_reshaped * emb.cos()) + (rotate_half(x_reshaped) * emb.sin())
    return x_rot.view(seq_len, num_heads * head_dim)


def causal_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-head attention with causal mask and stable softmax."""
    qh = q.view(seq_len, num_heads, head_dim).permute(1, 0, 2)
    kh = k.view(seq_len, num_heads, head_dim).permute(1, 0, 2)
    vh = v.view(seq_len, num_heads, head_dim).permute(1, 0, 2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale

    # Triangular causal mask: 0 if i >= j else -inf
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )
    scores_masked = scores + mask

    # Softmax stabil
    probs = F.softmax(scores_masked, dim=-1)

    # Context output
    out = torch.matmul(probs, vh).permute(1, 0, 2).reshape(seq_len, -1)
    return out, probs


def compute_attention_block(
    x: torch.Tensor,
    norm_gamma: torch.Tensor,
    w_q: torch.Tensor,
    b_q: torch.Tensor,
    w_k: torch.Tensor,
    b_k: torch.Tensor,
    w_v: torch.Tensor,
    b_v: torch.Tensor,
    w_o: torch.Tensor,
    b_o: torch.Tensor | None,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    pos_offset: int = 0,
    base: float = 1000000.0,
) -> torch.Tensor:
    """Full forward attention block pipeline."""
    # 1. RMSNorm
    x_norm = rmsnorm(x, norm_gamma, eps)

    # 2. QKV Projection
    q = F.linear(x_norm, w_q, b_q)
    k = F.linear(x_norm, w_k, b_k)
    v = F.linear(x_norm, w_v, b_v)

    # 3. RoPE
    q_rot = apply_rope(q, seq_len, num_heads, head_dim, pos_offset, base)
    k_rot = apply_rope(k, seq_len, num_heads, head_dim, pos_offset, base)

    # 4. Causal MHA
    attn_out, _ = causal_mha(q_rot, k_rot, v, seq_len, num_heads, head_dim)

    # 5. o_proj
    y = F.linear(attn_out, w_o, b_o)

    # 6. Residual
    return y + x


def run_tests() -> None:
    """Run comprehensive property tests for attention block components."""
    # 1. Softmax Max Shift Invariance
    z = torch.tensor([1.0, 5.0, 2.0, 4.0], dtype=torch.float32)
    p1 = F.softmax(z, dim=-1)
    p2 = F.softmax(z + 1000.0, dim=-1)
    assert torch.allclose(p1, p2, atol=1e-6), "Softmax max-shift invariance failed"

    # 2. Causal Mask Autoregressive Property: future change does not affect past
    seq_len = 3
    num_heads = 1
    head_dim = 4
    q_a = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    k_a = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    v_a = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=torch.float32,
    )

    q_b = q_a.clone()
    k_b = k_a.clone()
    v_b = v_a.clone()
    # Modify token 2 drastically
    q_b[2] = torch.tensor([-5.0, 8.0, -3.0, 2.0])
    k_b[2] = torch.tensor([99.0, -42.0, 13.0, 7.0])
    v_b[2] = torch.tensor([-100.0, 200.0, -300.0, 400.0])

    out_a, probs_a = causal_mha(q_a, k_a, v_a, seq_len, num_heads, head_dim)
    out_b, probs_b = causal_mha(q_b, k_b, v_b, seq_len, num_heads, head_dim)

    # Token 0 and token 1 outputs MUST be identical
    assert torch.allclose(
        out_a[:2], out_b[:2], atol=1e-6
    ), "Causal property failed: past token output changed!"
    # Token 2 output MUST differ
    assert not torch.allclose(
        out_a[2], out_b[2], atol=1e-2
    ), "Token 2 output should differ"

    # Verify upper triangle of probs is strictly zero
    for s in range(seq_len):
        for j in range(s + 1, seq_len):
            assert (
                probs_a[0, s, j].item() == 0.0
            ), f"Future attention leaked at ({s}, {j})"

    # 3. Known MHA Vector Verification
    q_known = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=torch.float32
    )
    k_known = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], dtype=torch.float32
    )
    v_known = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32
    )
    out_known, _ = causal_mha(
        q_known, k_known, v_known, seq_len=2, num_heads=1, head_dim=4
    )
    expected_known = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]], dtype=torch.float32
    )
    assert torch.allclose(
        out_known, expected_known, atol=1e-5
    ), f"Known MHA mismatch: {out_known} vs {expected_known}"

    # 4. Full Attention Block Slice Verification
    x_test = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.5, -0.5]], dtype=torch.float32
    )
    gamma_test = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    eye = torch.eye(4, dtype=torch.float32)
    zero_b = torch.zeros(4, dtype=torch.float32)

    y_block = compute_attention_block(
        x=x_test,
        norm_gamma=gamma_test,
        w_q=eye,
        b_q=zero_b,
        w_k=eye,
        b_k=zero_b,
        w_v=eye,
        b_v=zero_b,
        w_o=eye,
        b_o=None,
        seq_len=2,
        num_heads=1,
        head_dim=4,
        eps=1e-6,
        pos_offset=0,
    )
    expected_block = torch.tensor(
        [
            [1.3651483, 2.7302966, 4.0954452, 5.4605932],
            [0.8598768, -1.5558764, 4.2174449, -0.6550304],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(
        y_block, expected_block, atol=1e-5
    ), f"Full block slice mismatch: {y_block} vs {expected_block}"


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Attention Block slice (PyTorch fp32)"
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
                    "causal_property": "verified",
                    "softmax_stable": "verified",
                    "known_mha": "verified",
                    "full_block": "verified",
                }
            )
        )
        return

    torch.manual_seed(args.seed)
    hidden = args.num_heads * args.head_dim

    x = torch.randn(args.seq_len, hidden, dtype=torch.float32) * 0.1
    norm_gamma = torch.ones(hidden, dtype=torch.float32)
    w_q = torch.randn(hidden, hidden, dtype=torch.float32) * 0.02
    b_q = torch.randn(hidden, dtype=torch.float32) * 0.01
    w_k = torch.randn(hidden, hidden, dtype=torch.float32) * 0.02
    b_k = torch.randn(hidden, dtype=torch.float32) * 0.01
    w_v = torch.randn(hidden, hidden, dtype=torch.float32) * 0.02
    b_v = torch.randn(hidden, dtype=torch.float32) * 0.01
    w_o = torch.randn(hidden, hidden, dtype=torch.float32) * 0.02

    y_final = compute_attention_block(
        x,
        norm_gamma,
        w_q,
        b_q,
        w_k,
        b_k,
        w_v,
        b_v,
        w_o,
        None,
        args.seq_len,
        args.num_heads,
        args.head_dim,
        eps=1e-6,
        pos_offset=args.pos_offset,
        base=args.base,
    )

    if not torch.all(torch.isfinite(y_final)):
        fail("ATTENTION_ERROR", "Non-finite output detected in attention block")

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        x.numpy().tofile(os.path.join(args.dump_dir, "attn_in.bin"))
        y_final.numpy().tofile(os.path.join(args.dump_dir, "attn_ref.bin"))
        print(f"Dumped attention block fixtures to {args.dump_dir}")

    print(
        json.dumps(
            {
                "status": "success",
                "seq_len": args.seq_len,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "input_norm": float(torch.norm(x)),
                "output_norm": float(torch.norm(y_final)),
            }
        )
    )


if __name__ == "__main__":
    main()
