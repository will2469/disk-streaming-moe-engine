#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""F9 Diagnostics Runner: Load Balance & Routing Distribution (M3-W5).

Calculates F9 metrics from 256 random inputs:
  - f_i: normalized selection frequency (sum f_i = 1.0)
  - P_i: average router probability (sum P_i = 1.0)
  - L_lb: load-balance loss = N_e * sum(f_i * P_i)
  - CV: coefficient of variation = sigma_f / mu_f
  - hot_experts / cold_experts ranking for M7 page cache / LRU baseline

Usage:
  uv run python tools/bench/f9_diagnostics.py --samples 256 --seed 42
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

DEFAULT_MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/models/qwen3.6-35b-a3b")
)
DEFAULT_OUT_JSON = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../reports/2026-09-16/f9_routing_distribution.json",
    )
)


def compute_layer_f9_stats(
    X: torch.Tensor,
    model_dir: str,
    weight_map: dict[str, str],
    layer: int,
    num_experts: int = 60,
    top_k: int = 4,
) -> dict:
    """Compute F9 statistics for a given layer on input tensor X."""
    r_name = f"model.layers.{layer}.mlp.gate.weight"
    if r_name not in weight_map:
        raise ValueError(f"Router gate {r_name} not found in weight map")

    shard_path = os.path.join(model_dir, weight_map[r_name])
    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        w_router = sf.get_tensor(r_name).float()

    num_samples = X.shape[0]
    logits = torch.nn.functional.linear(X, w_router)
    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1, sorted=True)

    # Invariants verification
    counts = np.zeros(num_experts, dtype=np.int64)
    for t in range(num_samples):
        row_exp = topk_indices[t].tolist()
        row_prob = topk_probs[t].tolist()

        # Invariant 1: exactly top_k experts
        if len(row_exp) != top_k:
            raise AssertionError(
                f"Sample {t}: expected {top_k} experts, got {len(row_exp)}"
            )

        # Invariant 2: IDs within [0, num_experts - 1]
        for eid in row_exp:
            if eid < 0 or eid >= num_experts:
                raise AssertionError(f"Sample {t}: invalid expert ID {eid}")

        # Invariant 3: top_k sum <= 1.0 (unrenormalized)
        if sum(row_prob) > 1.0 + 1e-6:
            raise AssertionError(
                f"Sample {t}: unrenormalized sum {sum(row_prob)} > 1.0"
            )

        for eid in row_exp:
            counts[eid] += 1

    # Frequencies and Probabilities
    total_selections = num_samples * top_k
    f_i = counts / float(total_selections)
    P_i = probs.mean(dim=0).numpy()

    # Invariant 4: sum(f_i) == 1.0
    if abs(np.sum(f_i) - 1.0) > 1e-6:
        raise AssertionError(f"sum(f_i) = {np.sum(f_i)} != 1.0")

    # Load-balance loss: L_lb = N_e * sum(f_i * P_i)
    L_lb = float(num_experts * np.sum(f_i * P_i))

    # Coefficient of variation: CV = sigma_f / mu_f
    mu_f = 1.0 / float(num_experts)
    sigma_f = float(np.std(f_i))
    CV = sigma_f / mu_f

    # Ranking hot & cold experts
    expert_ranking = sorted(range(num_experts), key=lambda i: f_i[i], reverse=True)
    hot_experts = expert_ranking[:8]
    cold_experts = sorted(expert_ranking[-8:])

    expert_stats = []
    for i in range(num_experts):
        expert_stats.append(
            {
                "expert_id": i,
                "selection_count": int(counts[i]),
                "frequency": round(float(f_i[i]), 6),
                "avg_probability": round(float(P_i[i]), 6),
            }
        )

    return {
        "layer": layer,
        "num_experts": num_experts,
        "num_samples": num_samples,
        "top_k": top_k,
        "load_balance_loss": round(L_lb, 6),
        "coefficient_of_variation": round(CV, 6),
        "expert_stats": expert_stats,
        "hot_experts": hot_experts,
        "cold_experts": cold_experts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate F9 Routing Distribution and Load-Balance Metrics"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of random input tokens (default: 256)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
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
        help=f"Model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUT_JSON,
        help=f"Path to write report JSON (default: {DEFAULT_OUT_JSON})",
    )
    args = parser.parse_args()

    index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        sys.stderr.write(f"ERROR: index file not found: {index_path}\n")
        sys.exit(1)

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    torch.manual_seed(args.seed)
    X = torch.randn(args.samples, args.hidden_dim, dtype=torch.float32) * 0.1

    print(
        f"Evaluating F9 diagnostics on {args.samples} random inputs "
        f"(seed={args.seed})..."
    )

    layers_to_eval = [0, 12, 23]
    layers_stats = {}
    for lyr in layers_to_eval:
        st = compute_layer_f9_stats(
            X, args.model_dir, weight_map, lyr, num_experts=60, top_k=4
        )
        layers_stats[f"layer_{lyr}"] = st
        print(
            f"  Layer {lyr:2d}: L_lb = {st['load_balance_loss']:.4f}, "
            f"CV = {st['coefficient_of_variation']:.4f} | "
            f"Hot: {st['hot_experts'][:4]}... Cold: {st['cold_experts'][:4]}..."
        )

    # Primary report matches normatif spec format in M3-moe.md (Layer 0 primary)
    primary_l0 = layers_stats["layer_0"].copy()
    full_report = {
        "description": "F9 Routing Distribution and Load-Balance Baseline",
        "seed": args.seed,
        "num_samples": args.samples,
        "num_experts": 60,
        "top_k": 4,
        "primary_layer_0": primary_l0,
        "all_layers": layers_stats,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Report written successfully to: {args.output_json}")


if __name__ == "__main__":
    main()
