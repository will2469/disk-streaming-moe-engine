# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""QKV projection layer dan validasi bias attention (M2-W1)."""

from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import _load_one_tensor_by_name
from layers.qkv_bias import (
    collect_attention_bias_names,
    validate_attention_bias_in_index,
    validate_bias_count,
)
from std.builtin.dtype import DType
from std.collections import Dict, List
from std.math import isinf, isnan


@fieldwise_init
struct QKVWeights(Copyable, Movable):
    """Bobot QKV satu layer: W_q, W_k, W_v [hidden, hidden] + bias [hidden]."""

    var w_q: List[Float32]
    var w_k: List[Float32]
    var w_v: List[Float32]
    var b_q: List[Float32]
    var b_k: List[Float32]
    var b_v: List[Float32]


def qkv_project(
    x: List[Float32],
    w: List[Float32],
    b: List[Float32],
    seq_len: Int,
    hidden: Int,
) raises -> List[Float32]:
    """Proyeksi linear y = xW^T + b untuk satu matriks Q/K/V.

    @spec m2-w1-qkv-bias.md
    """
    if seq_len <= 0 or hidden <= 0:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"seq_len and hidden must'
            ' be positive","stage":"qkv"}'
        )
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"activation length'
            " mismatch: expected "
            + String(seq_len * hidden)
            + " got "
            + String(len(x))
            + '","stage":"qkv"}'
        )
    if len(w) != hidden * hidden:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"weight length'
            " mismatch: expected "
            + String(hidden * hidden)
            + " got "
            + String(len(w))
            + '","stage":"qkv"}'
        )
    if len(b) != hidden:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"bias length'
            " mismatch: expected "
            + String(hidden)
            + " got "
            + String(len(b))
            + '","stage":"qkv"}'
        )

    for i in range(len(x)):
        if isnan(x[i]) or isinf(x[i]):
            raise Error(
                '{"error_type":"ACT_LOAD_FAILED","detail":"non-finite value in'
                ' activation","stage":"qkv"}'
            )
    for i in range(len(w)):
        if isnan(w[i]) or isinf(w[i]):
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"non-finite value'
                ' in weight","stage":"qkv"}'
            )
    for i in range(len(b)):
        if isnan(b[i]) or isinf(b[i]):
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"non-finite value'
                ' in bias","stage":"qkv"}'
            )

    var out = List[Float32]()
    out.reserve(seq_len * hidden)

    var p_x = x.unsafe_ptr()
    var p_w = w.unsafe_ptr()
    var p_b = b.unsafe_ptr()

    for t in range(seq_len):
        var x_row = t * hidden
        for j in range(hidden):
            var w_row = j * hidden
            var acc_simd = SIMD[DType.float32, 16](0.0)
            var k = 0
            while k + 16 <= hidden:
                acc_simd += p_x.unsafe_load[width=16](
                    x_row + k
                ) * p_w.unsafe_load[width=16](w_row + k)
                k += 16
            var acc = acc_simd.reduce_add()
            while k < hidden:
                acc += (
                    p_x[unsafe_offset=x_row + k] * p_w[unsafe_offset=w_row + k]
                )
                k += 1
            var val = acc + p_b[unsafe_offset=j]
            if isnan(val) or isinf(val):
                raise Error(
                    '{"error_type":"ATTENTION_ERROR","detail":"non-finite'
                    ' value in qkv projection","stage":"qkv"}'
                )
            out.append(val)
    return out^


def qkv_forward(
    x: List[Float32],
    weights: QKVWeights,
    seq_len: Int,
    cfg: ModelConfig,
) raises -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """Hitung Q, K, V dari input x menggunakan bobot QKV satu layer."""
    var hidden = cfg.hidden_size
    if cfg.num_attention_heads * cfg.head_dim() != hidden:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"num_attention_heads *'
            ' head_dim != hidden_size","stage":"qkv"}'
        )
    var q = qkv_project(x, weights.w_q, weights.b_q, seq_len, hidden)
    var k = qkv_project(x, weights.w_k, weights.b_k, seq_len, hidden)
    var v = qkv_project(x, weights.w_v, weights.b_v, seq_len, hidden)
    return (q^, k^, v^)


def load_layer_qkv_weights(
    layer_idx: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    mut telemetry: LoadMemoryTelemetry,
) raises -> QKVWeights:
    """Memuat bobot QKV (w_q, w_k, w_v + b_q, b_k, b_v) untuk satu layer dari shard safetensors.

    @spec m2-w1-qkv-bias.md
    """
    if layer_idx < 0 or layer_idx >= cfg.num_hidden_layers:
        raise Error(
            '{"error_type":"LAYER_INVALID","detail":"invalid layer index: '
            + String(layer_idx)
            + " (expected 0.."
            + String(cfg.num_hidden_layers - 1)
            + ')","stage":"qkv","layer":'
            + String(layer_idx)
            + "}"
        )

    var prefix = "model.layers." + String(layer_idx) + ".self_attn."
    var req_wq = prefix + "q_proj.weight"
    var req_bq = prefix + "q_proj.bias"
    var req_wk = prefix + "k_proj.weight"
    var req_bk = prefix + "k_proj.bias"
    var req_wv = prefix + "v_proj.weight"
    var req_bv = prefix + "v_proj.bias"

    var required = List[String]()
    required.append(req_wq)
    required.append(req_bq)
    required.append(req_wk)
    required.append(req_bk)
    required.append(req_wv)
    required.append(req_bv)

    for i in range(len(required)):
        var r = required[i]
        if r not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"required QKV'
                " tensor not in weight_map: "
                + r
                + '","shard":"","tensor_name":"'
                + r
                + '"}'
            )

    var hidden = cfg.hidden_size
    var w_q = _load_one_tensor_by_name(
        model_root, weight_map[req_wq], req_wq, hidden, hidden, True, telemetry
    )
    var b_q = _load_one_tensor_by_name(
        model_root, weight_map[req_bq], req_bq, hidden, 1, False, telemetry
    )
    var w_k = _load_one_tensor_by_name(
        model_root, weight_map[req_wk], req_wk, hidden, hidden, True, telemetry
    )
    var b_k = _load_one_tensor_by_name(
        model_root, weight_map[req_bk], req_bk, hidden, 1, False, telemetry
    )
    var w_v = _load_one_tensor_by_name(
        model_root, weight_map[req_wv], req_wv, hidden, hidden, True, telemetry
    )
    var b_v = _load_one_tensor_by_name(
        model_root, weight_map[req_bv], req_bv, hidden, 1, False, telemetry
    )

    return QKVWeights(w_q^, w_k^, w_v^, b_q^, b_k^, b_v^)
