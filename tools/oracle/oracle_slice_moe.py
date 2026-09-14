#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle MoE Experts Slice — PyTorch FP32 Reference (M3-W2).

Pipeline:
  1. Router: logits = x W_r^T, p = softmax(logits, fp32), topk unrenormalized
  2. Routed SwiGLU: E_i(x) = W_down (SiLU(W_gate x) * (W_up x))
  3. Weighted sum: y_routed = sum_{i in A} p_i E_i(x)
  4. Shared expert SwiGLU: E_sh(x) = W_down_sh (SiLU(W_gate_sh x) * (W_up_sh x))
  5. Shared gate: g_sh = sigmoid(x W_shared_gate^T)  (INVARIANT: sigmoid, NOT softmax)
  6. Shared output: y_shared = g_sh * E_sh(x)
  7. Residual: y_final = y_routed + y_shared + x
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


def fail(
    error_type: str, detail: str, stage: str = "swiglu", extra: dict = None
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


def compute_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Hitung SwiGLU(x) = W_down (SiLU(W_gate x) * W_up x).

    x: [L, hidden_dim]
    w_gate: [inter_dim, hidden_dim]
    w_up: [inter_dim, hidden_dim]
    w_down: [hidden_dim, inter_dim]
    Returns: [L, hidden_dim]
    """
    g = F.linear(x, w_gate)
    u = F.linear(x, w_up)
    h = F.silu(g) * u
    y = F.linear(h, w_down)
    return y


def compute_moe_layer(
    x: torch.Tensor,
    w_router: torch.Tensor,
    routed_gates: list[torch.Tensor],
    routed_ups: list[torch.Tensor],
    routed_downs: list[torch.Tensor],
    w_shared_gate_proj: torch.Tensor,
    w_shared_up_proj: torch.Tensor,
    w_shared_down_proj: torch.Tensor,
    w_shared_gate: torch.Tensor,
    top_k: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hitung komputasi MoE layer lengkap:

    routed SwiGLU + shared SwiGLU + sigmoid gate + residual.
    """
    seq_len, hidden_dim = x.shape

    # 1. Router
    logits = F.linear(x, w_router)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1, sorted=True)

    # 2. Routed experts
    y_routed = torch.zeros_like(x)
    for t in range(seq_len):
        x_t = x[t : t + 1]
        accum = torch.zeros(1, hidden_dim, dtype=torch.float32)
        for k in range(top_k):
            idx = int(topk_indices[t, k].item())
            prob = topk_probs[t, k].item()
            e_out = compute_swiglu(
                x_t,
                routed_gates[idx],
                routed_ups[idx],
                routed_downs[idx],
            )
            accum += prob * e_out
        y_routed[t] = accum[0]

    # 3. Shared expert SwiGLU
    e_shared = compute_swiglu(
        x,
        w_shared_gate_proj,
        w_shared_up_proj,
        w_shared_down_proj,
    )

    # 4. Shared sigmoid gate: INVARIANT KERAS - sigmoid(x @ W_sh_gate.T)
    # W_shared_gate: [1, hidden_dim]
    shared_logits = F.linear(x, w_shared_gate)  # [L, 1]
    shared_gate_scores = torch.sigmoid(shared_logits)  # [L, 1]

    # 5. Shared contribution
    y_shared = shared_gate_scores * e_shared  # [L, hidden_dim]

    # 6. Residual connection: y_final = y_routed + y_shared + x
    y_final = y_routed + y_shared + x

    return y_final, y_routed, y_shared, topk_indices, topk_probs


def generate_synthetic_moe_fixture(
    seq_len: int = 16,
    hidden: int = 2048,
    inter_routed: int = 1408,
    inter_shared: int = 5632,
    num_experts: int = 60,
    seed: int = 42,
):
    torch.manual_seed(seed)
    # Skala kecil agar stabil numerik fp32
    x = torch.randn(seq_len, hidden, dtype=torch.float32) * 0.1
    w_router = torch.randn(num_experts, hidden, dtype=torch.float32) * 0.02

    routed_gates = []
    routed_ups = []
    routed_downs = []
    for _ in range(num_experts):
        routed_gates.append(
            torch.randn(inter_routed, hidden, dtype=torch.float32) * 0.01
        )
        routed_ups.append(torch.randn(inter_routed, hidden, dtype=torch.float32) * 0.01)
        routed_downs.append(
            torch.randn(hidden, inter_routed, dtype=torch.float32) * 0.01
        )

    w_shared_gate_proj = torch.randn(inter_shared, hidden, dtype=torch.float32) * 0.01
    w_shared_up_proj = torch.randn(inter_shared, hidden, dtype=torch.float32) * 0.01
    w_shared_down_proj = torch.randn(hidden, inter_shared, dtype=torch.float32) * 0.01
    w_shared_gate = torch.randn(1, hidden, dtype=torch.float32) * 0.02

    return (
        x,
        w_router,
        routed_gates,
        routed_ups,
        routed_downs,
        w_shared_gate_proj,
        w_shared_up_proj,
        w_shared_down_proj,
        w_shared_gate,
    )


def main():
    parser = argparse.ArgumentParser(description="Oracle MoE slice (PyTorch fp32)")
    parser.add_argument(
        "--test", action="store_true", help="Run deterministic synthetic test"
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--inter-routed", type=int, default=1408)
    parser.add_argument("--inter-shared", type=int, default=5632)
    parser.add_argument("--num-experts", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dump-dir", type=str, default="", help="Dump binary fixtures if provided"
    )
    args = parser.parse_args()

    (
        x,
        w_router,
        routed_gates,
        routed_ups,
        routed_downs,
        w_shared_gate_proj,
        w_shared_up_proj,
        w_shared_down_proj,
        w_shared_gate,
    ) = generate_synthetic_moe_fixture(
        args.seq_len,
        args.hidden_size,
        args.inter_routed,
        args.inter_shared,
        args.num_experts,
        args.seed,
    )

    y_final, y_routed, y_shared, topk_indices, topk_probs = compute_moe_layer(
        x,
        w_router,
        routed_gates,
        routed_ups,
        routed_downs,
        w_shared_gate_proj,
        w_shared_up_proj,
        w_shared_down_proj,
        w_shared_gate,
        top_k=args.top_k,
    )

    assert y_final.shape == (args.seq_len, args.hidden_size)
    assert torch.isfinite(y_final).all(), "Non-finite in y_final"
    assert torch.isfinite(y_routed).all(), "Non-finite in y_routed"
    assert torch.isfinite(y_shared).all(), "Non-finite in y_shared"

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        x.numpy().tofile(os.path.join(args.dump_dir, "x.bin"))
        w_router.numpy().tofile(os.path.join(args.dump_dir, "w_router.bin"))
        w_shared_gate.numpy().tofile(os.path.join(args.dump_dir, "w_shared_gate.bin"))
        w_shared_gate_proj.numpy().tofile(
            os.path.join(args.dump_dir, "w_shared_gate_proj.bin")
        )
        w_shared_up_proj.numpy().tofile(
            os.path.join(args.dump_dir, "w_shared_up_proj.bin")
        )
        w_shared_down_proj.numpy().tofile(
            os.path.join(args.dump_dir, "w_shared_down_proj.bin")
        )
        y_final.numpy().tofile(os.path.join(args.dump_dir, "moe_ref.bin"))
        y_routed.numpy().tofile(os.path.join(args.dump_dir, "y_routed_ref.bin"))
        y_shared.numpy().tofile(os.path.join(args.dump_dir, "y_shared_ref.bin"))
        topk_indices.numpy().astype(np.int32).tofile(
            os.path.join(args.dump_dir, "topk_indices_ref.bin")
        )
        topk_probs.numpy().tofile(os.path.join(args.dump_dir, "topk_probs_ref.bin"))

        # Simpan bobot expert terpilih saja untuk efisiensi fixture disk
        selected_set = set(topk_indices.flatten().tolist())
        for idx in selected_set:
            routed_gates[idx].numpy().tofile(
                os.path.join(args.dump_dir, f"expert_{idx}_gate.bin")
            )
            routed_ups[idx].numpy().tofile(
                os.path.join(args.dump_dir, f"expert_{idx}_up.bin")
            )
            routed_downs[idx].numpy().tofile(
                os.path.join(args.dump_dir, f"expert_{idx}_down.bin")
            )

        meta = {
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "inter_routed": args.inter_routed,
            "inter_shared": args.inter_shared,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "seed": args.seed,
            "selected_experts": sorted(list(selected_set)),
        }
        with open(os.path.join(args.dump_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Dumped MoE oracle fixtures to {args.dump_dir}")

    if args.test:
        print("Oracle MoE slice test passed successfully.")


if __name__ == "__main__":
    main()
