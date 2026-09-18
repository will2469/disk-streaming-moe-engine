# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Loader bobot MoE expert (routed dan shared) dari safetensors shards."""

from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import (
    ShardHeaderCache,
    _load_one_tensor_by_name,
    _load_tensor_slice_by_name,
)
from layers.swiglu import SwigluWeights
from std.collections import Dict, List


@fieldwise_init
struct SharedExpertWeights(Copyable, Movable):
    """Bobot shared expert: SwiGLU + sigmoid gate (F8c)."""

    var swiglu: SwigluWeights  # inter_dim = 5632, hidden_dim = 2048
    var w_gate_sh: List[Float32]  # shape: [hidden_dim] (1 x hidden_dim)


def load_layer_routed_expert_weights(
    layer_idx: Int,
    expert_id: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    inter_dim: Int,
    mut cache: ShardHeaderCache,
    mut telemetry: LoadMemoryTelemetry,
) raises -> SwigluWeights:
    """Memuat bobot SwiGLU untuk satu routed expert (W_gate, W_up, W_down)."""
    if layer_idx < 0 or layer_idx >= cfg.num_hidden_layers:
        raise Error(
            '{"error_type":"LAYER_INVALID","detail":"invalid layer'
            ' index","stage":"experts","layer":'
            + String(layer_idx)
            + "}"
        )

    var pfx_lm = "model.language_model.layers." + String(layer_idx) + ".mlp."
    var pfx_legacy = "model.layers." + String(layer_idx) + ".mlp."
    var pfx = pfx_lm if (pfx_lm + "gate.weight") in weight_map else pfx_legacy

    var req_fused_gu = pfx + "experts.gate_up_proj"
    var req_fused_d = pfx + "experts.down_proj"

    if req_fused_gu in weight_map and req_fused_d in weight_map:
        var hidden = cfg.hidden_size
        var gu_raw = _load_tensor_slice_by_name(
            cache,
            model_root,
            weight_map[req_fused_gu],
            req_fused_gu,
            expert_id,
            2 * inter_dim,
            hidden,
            telemetry,
        )
        var gate_elements = inter_dim * hidden
        var w_gate = List[Float32]()
        w_gate.reserve(gate_elements)
        var w_up = List[Float32]()
        w_up.reserve(gate_elements)

        var p_gu = gu_raw.unsafe_ptr()
        for i in range(gate_elements):
            w_gate.append(p_gu[unsafe_offset=i])
        for i in range(gate_elements):
            w_up.append(p_gu[unsafe_offset=gate_elements + i])

        var w_down = _load_tensor_slice_by_name(
            cache,
            model_root,
            weight_map[req_fused_d],
            req_fused_d,
            expert_id,
            hidden,
            inter_dim,
            telemetry,
        )
        return SwigluWeights(w_gate^, w_up^, w_down^, hidden, inter_dim)

    var prefix = pfx + "experts." + String(expert_id) + "."
    var req_gate = prefix + "gate_proj.weight"
    var req_up = prefix + "up_proj.weight"
    var req_down = prefix + "down_proj.weight"
    if (
        req_gate not in weight_map
        or req_up not in weight_map
        or req_down not in weight_map
    ):
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"expert tensor not in'
            ' weight_map","stage":"experts","layer":'
            + String(layer_idx)
            + ',"expert_id":'
            + String(expert_id)
            + "}"
        )

    var hidden = cfg.hidden_size
    var w_gate = _load_one_tensor_by_name(
        cache,
        model_root,
        weight_map[req_gate],
        req_gate,
        inter_dim,
        hidden,
        True,
        telemetry,
    )
    var w_up = _load_one_tensor_by_name(
        cache,
        model_root,
        weight_map[req_up],
        req_up,
        inter_dim,
        hidden,
        True,
        telemetry,
    )
    var w_down = _load_one_tensor_by_name(
        cache,
        model_root,
        weight_map[req_down],
        req_down,
        hidden,
        inter_dim,
        True,
        telemetry,
    )
    return SwigluWeights(w_gate^, w_up^, w_down^, hidden, inter_dim)


def load_layer_shared_expert_weights(
    layer_idx: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    inter_shared: Int,
    mut cache: ShardHeaderCache,
    mut telemetry: LoadMemoryTelemetry,
) raises -> SharedExpertWeights:
    """Memuat bobot shared expert (W_gate, W_up, W_down) + shared_expert_gate.
    """
    if layer_idx < 0 or layer_idx >= cfg.num_hidden_layers:
        raise Error(
            '{"error_type":"LAYER_INVALID","detail":"invalid layer'
            ' index","stage":"shared","layer":'
            + String(layer_idx)
            + "}"
        )

    var pfx_lm = "model.language_model.layers." + String(layer_idx) + ".mlp."
    var pfx_legacy = "model.layers." + String(layer_idx) + ".mlp."
    var pfx = pfx_lm if (pfx_lm + "gate.weight") in weight_map else pfx_legacy

    var prefix = pfx + "shared_expert."
    var req_gate = prefix + "gate_proj.weight"
    var req_up = prefix + "up_proj.weight"
    var req_down = prefix + "down_proj.weight"
    var req_sh_gate = pfx + "shared_expert_gate.weight"
    if (
        req_gate not in weight_map
        or req_up not in weight_map
        or req_down not in weight_map
        or req_sh_gate not in weight_map
    ):
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"shared expert tensor'
            ' not in weight_map","stage":"shared","layer":'
            + String(layer_idx)
            + "}"
        )

    var hidden = cfg.hidden_size
    var w_gate = _load_one_tensor_by_name(
        cache,
        model_root,
        weight_map[req_gate],
        req_gate,
        inter_shared,
        hidden,
        True,
        telemetry,
    )
    var w_up = _load_one_tensor_by_name(
        cache,
        model_root,
        weight_map[req_up],
        req_up,
        inter_shared,
        hidden,
        True,
        telemetry,
    )
    var w_down = _load_one_tensor_by_name(
        cache,
        model_root,
        weight_map[req_down],
        req_down,
        hidden,
        inter_shared,
        True,
        telemetry,
    )
    var w_sh_gate = _load_one_tensor_by_name(
        cache,
        model_root,
        weight_map[req_sh_gate],
        req_sh_gate,
        hidden,
        1,
        False,
        telemetry,
    )
    var swiglu = SwigluWeights(w_gate^, w_up^, w_down^, hidden, inter_shared)
    return SharedExpertWeights(swiglu^, w_sh_gate^)
