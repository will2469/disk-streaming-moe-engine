# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Pipeline MoE block: router, routed experts SwiGLU, shared expert, dan agregasi."""

from core.config import LoadMemoryTelemetry, ModelConfig
from layers.moe import moe_aggregate_forward, shared_gate_forward
from layers.moe_loader import (
    SharedExpertWeights,
    load_layer_routed_expert_weights,
    load_layer_shared_expert_weights,
)
from layers.router import load_layer_router_weights, router_forward
from layers.router_types import RouterConfig, RoutingInfo
from layers.swiglu import SwigluWeights, swiglu_forward
from std.collections import Dict, List


def forward_moe_block(
    x: List[Float32],
    layer_idx: Int,
    seq_len: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    mut telemetry: LoadMemoryTelemetry,
) raises -> Tuple[List[Float32], RoutingInfo]:
    """Menjalankan full forward satu layer MoE:

    1. Router softmax fp32 + top-k tanpa renormalisasi (F8a, F8b)
    2. Routed experts SwiGLU (F8d)
    3. Shared expert SwiGLU + sigmoid gate (F8c)
    4. Agregasi: y = sum(p_i E_i(x)) + sigma(g_sh) E_sh(x) + x
    """
    var router_cfg = RouterConfig(
        cfg.num_experts, cfg.num_experts_per_tok, cfg.norm_topk_prob
    )
    var w_router = load_layer_router_weights(
        layer_idx, model_root, weight_map, cfg, router_cfg, telemetry
    )
    var routing = router_forward(
        x, w_router, seq_len, cfg.hidden_size, router_cfg, layer_idx
    )

    # Identifikasi unique experts yang terpilih
    var unique_experts = List[Int]()
    for t in range(seq_len):
        ref row = routing.selected_experts[t]
        for k in range(len(row)):
            var e_id = row[k]
            var found = False
            for u in range(len(unique_experts)):
                if unique_experts[u] == e_id:
                    found = True
                    break
            if not found:
                unique_experts.append(e_id)

    # Muat bobot routed experts yang unik
    var loaded_experts = List[SwigluWeights]()
    for u in range(len(unique_experts)):
        var e_id = unique_experts[u]
        var w = load_layer_routed_expert_weights(
            layer_idx,
            e_id,
            model_root,
            weight_map,
            cfg,
            cfg.moe_intermediate_size,
            telemetry,
        )
        loaded_experts.append(w^)

    # Hitung routed experts per token
    var all_routed_outputs = List[List[Float32]]()
    for t in range(seq_len):
        ref row = routing.selected_experts[t]
        var tok_out = List[Float32]()
        tok_out.reserve(cfg.num_experts_per_tok * cfg.hidden_size)

        var x_t = List[Float32]()
        x_t.reserve(cfg.hidden_size)
        var offset = t * cfg.hidden_size
        for d in range(cfg.hidden_size):
            x_t.append(x[offset + d])

        for k in range(cfg.num_experts_per_tok):
            var e_id = row[k]
            var exp_idx = -1
            for u in range(len(unique_experts)):
                if unique_experts[u] == e_id:
                    exp_idx = u
                    break
            var e_out = swiglu_forward(
                x_t, loaded_experts[exp_idx], 1, layer_idx, e_id
            )
            for d in range(cfg.hidden_size):
                tok_out.append(e_out[d])
        all_routed_outputs.append(tok_out^)

    # Muat bobot shared expert & hitung SwiGLU + sigmoid gate
    var shared_weights = load_layer_shared_expert_weights(
        layer_idx,
        model_root,
        weight_map,
        cfg,
        cfg.shared_expert_intermediate_size,
        telemetry,
    )
    var shared_out = swiglu_forward(
        x, shared_weights.swiglu, seq_len, layer_idx, -1
    )
    var shared_gate_scores = shared_gate_forward(
        x,
        shared_weights.w_gate_sh,
        seq_len,
        cfg.hidden_size,
        layer_idx,
        "sigmoid",
    )

    # Agregasi akhir ber-residual
    var out_act = moe_aggregate_forward(
        x,
        all_routed_outputs,
        routing.router_probs,
        shared_out,
        shared_gate_scores,
        seq_len,
        cfg.hidden_size,
        layer_idx,
    )

    return (out_act^, routing^)
