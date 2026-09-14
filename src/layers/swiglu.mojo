# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""SwiGLU activation and projection kernels (F8d) berakselerasi SIMD float32."""

from std.builtin.dtype import DType
from std.collections import List
from std.math import exp, isinf, isnan


def sigmoid_f32(z: Float32) -> Float32:
    """Fungsi Sigmoid numerik stabil fp32: sigma(z) = 1 / (1 + exp(-z)).

    @spec docs/milestones/M3-moe.md (Scope F8c)
    Mencegah overflow/underflow float32 pada nilai ekstrem (|z| > 88).
    """
    if isnan(z):
        return Float32(0.0) / Float32(0.0)
    if z < Float32(-88.0):
        return Float32(0.0)
    if z > Float32(88.0):
        return Float32(1.0)
    if z < Float32(0.0):
        var ez = exp(z)
        return ez / (Float32(1.0) + ez)
    return Float32(1.0) / (Float32(1.0) + exp(-z))


def silu_f32(z: Float32) -> Float32:
    """Fungsi SiLU (Swish-1) numerik stabil fp32: silu(z) = z * sigma(z).

    @spec docs/milestones/M3-moe.md (Scope F8d)
    """
    if isnan(z):
        return Float32(0.0) / Float32(0.0)
    if isinf(z):
        return z if z > Float32(0.0) else Float32(0.0)
    return z * sigmoid_f32(z)


@fieldwise_init
struct SwigluWeights(Copyable, Movable):
    """Bobot proyeksi SwiGLU: W_gate, W_up, W_down (F8d)."""

    var w_gate: List[Float32]  # shape: [inter_dim, hidden_dim]
    var w_up: List[Float32]  # shape: [inter_dim, hidden_dim]
    var w_down: List[Float32]  # shape: [hidden_dim, inter_dim]
    var hidden_dim: Int
    var inter_dim: Int


def swiglu_forward(
    x: List[Float32],
    weights: SwigluWeights,
    seq_len: Int,
    layer_idx: Int = 0,
    expert_id: Int = -1,
) raises -> List[Float32]:
    """Hitung SwiGLU: y = W_down (SiLU(W_gate x) * W_up x).

    @spec scratch/wave/m3/m3-w2-swiglu.md (F8d)
    Urutan: gate proj -> SiLU -> up proj -> element-wise multiply -> down proj.
    """
    var hidden = weights.hidden_dim
    var inter = weights.inter_dim
    if seq_len <= 0 or hidden <= 0 or inter <= 0:
        raise Error(
            '{"error_type":"EXPERT_ERROR","detail":"dimensions must be'
            ' positive","stage":"swiglu","layer":'
            + String(layer_idx)
            + ',"expert_id":'
            + String(expert_id)
            + "}"
        )
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"activation length'
            " mismatch: expected "
            + String(seq_len * hidden)
            + " got "
            + String(len(x))
            + '","stage":"swiglu","layer":'
            + String(layer_idx)
            + ',"expert_id":'
            + String(expert_id)
            + "}"
        )
    if (
        len(weights.w_gate) != inter * hidden
        or len(weights.w_up) != inter * hidden
    ):
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"gate/up weight length'
            ' mismatch","stage":"swiglu","layer":'
            + String(layer_idx)
            + ',"expert_id":'
            + String(expert_id)
            + "}"
        )
    if len(weights.w_down) != hidden * inter:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"down weight length'
            ' mismatch","stage":"swiglu","layer":'
            + String(layer_idx)
            + ',"expert_id":'
            + String(expert_id)
            + "}"
        )

    for i in range(len(x)):
        if isnan(x[i]) or isinf(x[i]):
            raise Error(
                '{"error_type":"ACT_LOAD_FAILED","detail":"non-finite value in'
                ' activation","stage":"swiglu","layer":'
                + String(layer_idx)
                + ',"expert_id":'
                + String(expert_id)
                + "}"
            )

    var p_x = x.unsafe_ptr()
    var p_wg = weights.w_gate.unsafe_ptr()
    var p_wu = weights.w_up.unsafe_ptr()
    var p_wd = weights.w_down.unsafe_ptr()

    var out = List[Float32]()
    out.reserve(seq_len * hidden)

    for t in range(seq_len):
        var x_row = t * hidden
        var h = List[Float32]()
        h.reserve(inter)

        for j in range(inter):
            var w_row = j * hidden

            # 1. Gate projection: g = W_gate x
            var g_simd = SIMD[DType.float32, 16](0.0)
            var k = 0
            while k + 16 <= hidden:
                g_simd += p_x.unsafe_load[width=16](
                    x_row + k
                ) * p_wg.unsafe_load[width=16](w_row + k)
                k += 16
            var g = g_simd.reduce_add()
            while k < hidden:
                g += (
                    p_x[unsafe_offset=x_row + k] * p_wg[unsafe_offset=w_row + k]
                )
                k += 1

            if isnan(g) or isinf(g):
                raise Error(
                    '{"error_type":"EXPERT_ERROR","detail":"non-finite in gate'
                    ' projection","stage":"swiglu","layer":'
                    + String(layer_idx)
                    + ',"expert_id":'
                    + String(expert_id)
                    + "}"
                )

            # 2. SiLU(g)
            var g_act = silu_f32(g)

            # 3. Up projection: u = W_up x
            var u_simd = SIMD[DType.float32, 16](0.0)
            k = 0
            while k + 16 <= hidden:
                u_simd += p_x.unsafe_load[width=16](
                    x_row + k
                ) * p_wu.unsafe_load[width=16](w_row + k)
                k += 16
            var u = u_simd.reduce_add()
            while k < hidden:
                u += (
                    p_x[unsafe_offset=x_row + k] * p_wu[unsafe_offset=w_row + k]
                )
                k += 1

            if isnan(u) or isinf(u):
                raise Error(
                    '{"error_type":"EXPERT_ERROR","detail":"non-finite in up'
                    ' projection","stage":"swiglu","layer":'
                    + String(layer_idx)
                    + ',"expert_id":'
                    + String(expert_id)
                    + "}"
                )

            # 4. Element-wise product: h = g_act * u
            var hj = g_act * u
            if isnan(hj) or isinf(hj):
                raise Error(
                    '{"error_type":"SWIGLU_ERROR","detail":"non-finite in'
                    ' element-wise product","stage":"swiglu","layer":'
                    + String(layer_idx)
                    + ',"expert_id":'
                    + String(expert_id)
                    + "}"
                )
            h.append(hj)

        # 5. Down projection: y = W_down h
        var p_h = h.unsafe_ptr()
        for d in range(hidden):
            var wd_row = d * inter
            var d_simd = SIMD[DType.float32, 16](0.0)
            var m = 0
            while m + 16 <= inter:
                d_simd += p_h.unsafe_load[width=16](m) * p_wd.unsafe_load[
                    width=16
                ](wd_row + m)
                m += 16
            var y_val = d_simd.reduce_add()
            while m < inter:
                y_val += p_h[unsafe_offset=m] * p_wd[unsafe_offset=wd_row + m]
                m += 1

            if isnan(y_val) or isinf(y_val):
                raise Error(
                    '{"error_type":"EXPERT_ERROR","detail":"non-finite in down'
                    ' projection","stage":"swiglu","layer":'
                    + String(layer_idx)
                    + ',"expert_id":'
                    + String(expert_id)
                    + "}"
                )
            out.append(y_val)

    return out^
