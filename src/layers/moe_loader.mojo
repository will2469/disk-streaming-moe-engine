# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Loader bobot MoE expert (routed dan shared) dari safetensors shards."""

from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import _load_one_tensor_by_name
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
    var prefix = (
        "model.layers."
        + String(layer_idx)
        + ".mlp.experts."
        + String(expert_id)
        + "."
    )
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
        model_root,
        weight_map[req_gate],
        req_gate,
        inter_dim,
        hidden,
        True,
        telemetry,
    )
    var w_up = _load_one_tensor_by_name(
        model_root,
        weight_map[req_up],
        req_up,
        inter_dim,
        hidden,
        True,
        telemetry,
    )
    var w_down = _load_one_tensor_by_name(
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
    var prefix = "model.layers." + String(layer_idx) + ".mlp.shared_expert."
    var req_gate = prefix + "gate_proj.weight"
    var req_up = prefix + "up_proj.weight"
    var req_down = prefix + "down_proj.weight"
    var req_sh_gate = (
        "model.layers." + String(layer_idx) + ".mlp.shared_expert_gate.weight"
    )
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
        model_root,
        weight_map[req_gate],
        req_gate,
        inter_shared,
        hidden,
        True,
        telemetry,
    )
    var w_up = _load_one_tensor_by_name(
        model_root,
        weight_map[req_up],
        req_up,
        inter_shared,
        hidden,
        True,
        telemetry,
    )
    var w_down = _load_one_tensor_by_name(
        model_root,
        weight_map[req_down],
        req_down,
        hidden,
        inter_shared,
        True,
        telemetry,
    )
    var w_sh_gate = _load_one_tensor_by_name(
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
