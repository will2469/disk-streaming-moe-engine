# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""MoE Router: Proyeksi gate, softmax fp32 stabil, dan top-k no-renorm (M3-W1)."""

from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import ShardHeaderCache, _load_one_tensor_by_name
from layers.router_types import RouterConfig, RoutingInfo
from layers.topk import select_topk
from std.builtin.dtype import DType
from std.collections import Dict, List
from std.math import exp, isinf, isnan


def router_project(
    x: List[Float32],
    w_gate: List[Float32],
    seq_len: Int,
    hidden_dim: Int,
    num_experts: Int,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Proyeksi linear router logits: z = x W_r^T, fp32.

    @spec scratch/wave/m3/m3-w1-router.md (F8a)
    """
    if seq_len <= 0 or hidden_dim <= 0 or num_experts <= 0:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"dimensions must be'
            ' positive","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(x) != seq_len * hidden_dim:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"activation length'
            " mismatch: expected "
            + String(seq_len * hidden_dim)
            + " got "
            + String(len(x))
            + '","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(w_gate) != num_experts * hidden_dim:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"weight length'
            " mismatch: expected "
            + String(num_experts * hidden_dim)
            + " got "
            + String(len(w_gate))
            + '","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )

    for i in range(len(x)):
        if isnan(x[i]) or isinf(x[i]):
            raise Error(
                '{"error_type":"ACT_LOAD_FAILED","detail":"non-finite value in'
                ' activation","stage":"router","layer":'
                + String(layer_idx)
                + "}"
            )
    for i in range(len(w_gate)):
        if isnan(w_gate[i]) or isinf(w_gate[i]):
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"non-finite value'
                ' in router weight","stage":"router","layer":'
                + String(layer_idx)
                + "}"
            )

    var out = List[Float32]()
    out.reserve(seq_len * num_experts)

    var p_x = x.unsafe_ptr()
    var p_w = w_gate.unsafe_ptr()

    for t in range(seq_len):
        var x_row = t * hidden_dim
        for e in range(num_experts):
            var w_row = e * hidden_dim
            var acc_simd = SIMD[DType.float32, 16](0.0)
            var k = 0
            while k + 16 <= hidden_dim:
                acc_simd += p_x.unsafe_load[width=16](
                    x_row + k
                ) * p_w.unsafe_load[width=16](w_row + k)
                k += 16
            var acc = acc_simd.reduce_add()
            while k < hidden_dim:
                acc += (
                    p_x[unsafe_offset=x_row + k] * p_w[unsafe_offset=w_row + k]
                )
                k += 1
            if isnan(acc) or isinf(acc):
                raise Error(
                    '{"error_type":"ROUTER_ERROR","detail":"non-finite router'
                    ' logit computed","stage":"router","layer":'
                    + String(layer_idx)
                    + "}"
                )
            out.append(acc)

    return out^


def router_softmax(
    logits: List[Float32], seq_len: Int, num_experts: Int, layer_idx: Int = 0
) raises -> List[Float32]:
    """Softmax numerik stabil per baris token pada logits router: P = softmax(z).

    @spec scratch/wave/m3/m3-w1-router.md (F8a)
    """
    if seq_len <= 0 or num_experts <= 0:
        raise Error(
            '{"error_type":"ROUTER_ERROR","detail":"dimensions must be'
            ' positive","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(logits) != seq_len * num_experts:
        raise Error(
            '{"error_type":"ROUTER_ERROR","detail":"logits length mismatch:'
            " expected "
            + String(seq_len * num_experts)
            + " got "
            + String(len(logits))
            + '","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )

    var probs = List[Float32]()
    probs.reserve(seq_len * num_experts)

    var p_logits = logits.unsafe_ptr()

    for t in range(seq_len):
        var base = t * num_experts
        var max_val = p_logits[unsafe_offset=base]
        for e in range(1, num_experts):
            var v = p_logits[unsafe_offset=base + e]
            if v > max_val:
                max_val = v

        if isnan(max_val) or isinf(max_val):
            raise Error(
                '{"error_type":"ROUTER_ERROR","detail":"non-finite max logit in'
                ' router softmax","stage":"router","layer":'
                + String(layer_idx)
                + "}"
            )

        var sum_exp = Float32(0.0)
        var temp_exp = List[Float32]()
        temp_exp.reserve(num_experts)

        for e in range(num_experts):
            var ev = exp(p_logits[unsafe_offset=base + e] - max_val)
            if isnan(ev) or isinf(ev):
                raise Error(
                    '{"error_type":"ROUTER_ERROR","detail":"non-finite exp in'
                    ' router softmax","stage":"router","layer":'
                    + String(layer_idx)
                    + "}"
                )
            temp_exp.append(ev)
            sum_exp += ev

        if sum_exp <= Float32(0.0) or isnan(sum_exp) or isinf(sum_exp):
            raise Error(
                '{"error_type":"ROUTER_ERROR","detail":"invalid sum_exp in'
                ' router softmax","stage":"router","layer":'
                + String(layer_idx)
                + "}"
            )

        var inv_sum = Float32(1.0) / sum_exp
        for e in range(num_experts):
            var p = temp_exp[e] * inv_sum
            probs.append(p)

    return probs^


def router_forward(
    x: List[Float32],
    w_router: List[Float32],
    seq_len: Int,
    hidden_dim: Int,
    cfg: RouterConfig,
    layer_idx: Int = 0,
) raises -> RoutingInfo:
    """Pipeline forward router lengkap: x -> router_project -> router_softmax -> select_topk.
    """
    cfg.validate()
    var logits = router_project(
        x, w_router, seq_len, hidden_dim, cfg.num_experts, layer_idx
    )
    var probs = router_softmax(logits, seq_len, cfg.num_experts, layer_idx)
    return select_topk(
        probs,
        seq_len,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.norm_topk_prob,
        layer_idx,
    )


def load_layer_router_weights(
    layer_idx: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    router_cfg: RouterConfig,
    mut cache: ShardHeaderCache,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    """Memuat bobot gate router (W_r) untuk satu layer dari shard safetensors.
    """
    if layer_idx < 0 or layer_idx >= cfg.num_hidden_layers:
        raise Error(
            '{"error_type":"LAYER_INVALID","detail":"invalid layer index: '
            + String(layer_idx)
            + " (expected 0.."
            + String(cfg.num_hidden_layers - 1)
            + ')","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )
    var pfx_lm = (
        "model.language_model.layers." + String(layer_idx) + ".mlp.gate.weight"
    )
    var pfx_legacy = "model.layers." + String(layer_idx) + ".mlp.gate.weight"
    var tensor_name = pfx_lm if pfx_lm in weight_map else pfx_legacy
    if tensor_name not in weight_map:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"required router gate'
            " tensor not in weight_map: "
            + tensor_name
            + '","shard":"","tensor_name":"'
            + tensor_name
            + '"}'
        )
    return _load_one_tensor_by_name(
        cache,
        model_root,
        weight_map[tensor_name],
        tensor_name,
        router_cfg.num_experts,
        cfg.hidden_size,
        True,
        telemetry,
    )
