# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""MoE Experts: Shared Expert, Sigmoid Gate (Invariant #2), dan Agregasi (M3-W2)."""

from layers.moe_loader import (
    SharedExpertWeights,
    load_layer_routed_expert_weights,
    load_layer_shared_expert_weights,
)
from layers.swiglu import sigmoid_f32
from std.builtin.dtype import DType
from std.collections import List
from std.math import isinf, isnan


def shared_gate_forward(
    x: List[Float32],
    w_shared_gate: List[Float32],
    seq_len: Int,
    hidden_dim: Int,
    layer_idx: Int = 0,
    gate_mode: String = "sigmoid",
) raises -> List[Float32]:
    """Hitung shared expert gate: g = sigma(W_sh_gate x), fp32.

    @spec docs/milestones/M3-moe.md (Invariant keras #2 / jebakan §2.3)
    WAJIB SIGMOID, bukan softmax dan bukan linear!
    """
    if gate_mode != "sigmoid":
        raise Error(
            '{"error_type":"GATE_ERROR","detail":"shared expert gate must be'
            " sigmoid; non-sigmoid gate forbidden (invariant"
            ' #2)","stage":"shared","layer":'
            + String(layer_idx)
            + "}"
        )
    if seq_len <= 0 or hidden_dim <= 0:
        raise Error(
            '{"error_type":"GATE_ERROR","detail":"dimensions must be'
            ' positive","stage":"shared","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(x) != seq_len * hidden_dim:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"activation length'
            ' mismatch","stage":"shared","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(w_shared_gate) != hidden_dim:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"shared gate weight'
            " length mismatch: expected "
            + String(hidden_dim)
            + " got "
            + String(len(w_shared_gate))
            + '","stage":"shared","layer":'
            + String(layer_idx)
            + "}"
        )

    var p_x = x.unsafe_ptr()
    var p_wg = w_shared_gate.unsafe_ptr()

    var scores = List[Float32]()
    scores.reserve(seq_len)

    for t in range(seq_len):
        var x_row = t * hidden_dim
        var acc_simd = SIMD[DType.float32, 16](0.0)
        var k = 0
        while k + 16 <= hidden_dim:
            acc_simd += p_x.unsafe_load[width=16](x_row + k) * p_wg.unsafe_load[
                width=16
            ](k)
            k += 16
        var logit = acc_simd.reduce_add()
        while k < hidden_dim:
            logit += p_x[unsafe_offset=x_row + k] * p_wg[unsafe_offset=k]
            k += 1

        if isnan(logit) or isinf(logit):
            raise Error(
                '{"error_type":"GATE_ERROR","detail":"non-finite shared gate'
                ' logit","stage":"shared","layer":'
                + String(layer_idx)
                + "}"
            )

        var sig = sigmoid_f32(logit)
        if isnan(sig) or isinf(sig) or sig < Float32(0.0) or sig > Float32(1.0):
            raise Error(
                '{"error_type":"GATE_ERROR","detail":"shared gate score outside'
                ' (0, 1)","stage":"shared","layer":'
                + String(layer_idx)
                + "}"
            )
        scores.append(sig)

    return scores^


def moe_aggregate_forward(
    x: List[Float32],
    routed_outputs: List[List[Float32]],
    router_probs: List[List[Float32]],
    shared_output: List[Float32],
    shared_gate_scores: List[Float32],
    seq_len: Int,
    hidden_dim: Int,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Agregasi MoE: y = sum_{k in top-4} p_k E_{i_k}(x) + sigma(g_sh) E_sh(x) + x.

    @spec scratch/wave/m3/m3-w2-swiglu.md (F8c)
    """
    if seq_len <= 0 or hidden_dim <= 0:
        raise Error(
            '{"error_type":"EXPERT_ERROR","detail":"dimensions must be'
            ' positive","stage":"aggregation","layer":'
            + String(layer_idx)
            + "}"
        )
    if (
        len(x) != seq_len * hidden_dim
        or len(shared_output) != seq_len * hidden_dim
    ):
        raise Error(
            '{"error_type":"EXPERT_ERROR","detail":"length mismatch in'
            ' aggregation","stage":"aggregation","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(shared_gate_scores) != seq_len or len(routed_outputs) != seq_len:
        raise Error(
            '{"error_type":"EXPERT_ERROR","detail":"sequence length'
            ' mismatch","stage":"aggregation","layer":'
            + String(layer_idx)
            + "}"
        )

    var out = List[Float32]()
    out.reserve(seq_len * hidden_dim)

    var p_x = x.unsafe_ptr()
    var p_sh_out = shared_output.unsafe_ptr()

    for t in range(seq_len):
        var x_base = t * hidden_dim
        var gate_sh = shared_gate_scores[t]
        ref r_outs = routed_outputs[t]
        ref probs = router_probs[t]
        var top_k = len(probs)

        for d in range(hidden_dim):
            var routed_sum = Float32(0.0)
            for k in range(top_k):
                var p = probs[k]
                var val = r_outs[k * hidden_dim + d]
                routed_sum += p * val

            var sh_val = gate_sh * p_sh_out[unsafe_offset=x_base + d]
            var res_val = p_x[unsafe_offset=x_base + d]
            var total = routed_sum + sh_val + res_val

            if isnan(total) or isinf(total):
                raise Error(
                    '{"error_type":"EXPERT_ERROR","detail":"non-finite value in'
                    ' aggregation","stage":"aggregation","layer":'
                    + String(layer_idx)
                    + "}"
                )
            out.append(total)

    return out^
