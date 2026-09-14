# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Blok attention lengkap: RMSNorm -> QKV -> RoPE -> MHA -> o_proj -> Residual (M2)."""

from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import _load_one_tensor_by_name
from layers.mha import mha_forward
from layers.qkv import QKVWeights, load_layer_qkv_weights, qkv_forward
from layers.residual import add_residual
from layers.rmsnorm import rmsnorm
from layers.rope import apply_rope
from std.builtin.dtype import DType
from std.collections import Dict, List
from std.math import isinf, isnan


@fieldwise_init
struct AttentionWeights(Copyable, Movable):
    """Bobot attention satu layer: norm_gamma, QKV (w+b), w_o, b_o."""

    var norm_gamma: List[Float32]
    var qkv: QKVWeights
    var w_o: List[Float32]
    var b_o: List[Float32]


def o_project(
    x: List[Float32],
    w_o: List[Float32],
    b_o: List[Float32],
    seq_len: Int,
    hidden: Int,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Proyeksi linear output attention: y = x W_o^T (+ b_o jika ada).

    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if seq_len <= 0 or hidden <= 0:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"seq_len and hidden must'
            ' be positive","stage":"oproj","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"input length'
            ' mismatch","stage":"oproj","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(w_o) != hidden * hidden:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"w_o length'
            ' mismatch","stage":"oproj","layer":'
            + String(layer_idx)
            + "}"
        )
    var has_bias = len(b_o) > 0
    if has_bias and len(b_o) != hidden:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"b_o length'
            ' mismatch","stage":"oproj","layer":'
            + String(layer_idx)
            + "}"
        )

    for i in range(len(x)):
        if isnan(x[i]) or isinf(x[i]):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' o_proj input","stage":"oproj","layer":'
                + String(layer_idx)
                + "}"
            )
    for i in range(len(w_o)):
        if isnan(w_o[i]) or isinf(w_o[i]):
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"non-finite value'
                ' in w_o","stage":"oproj","layer":'
                + String(layer_idx)
                + "}"
            )
    if has_bias:
        for i in range(len(b_o)):
            if isnan(b_o[i]) or isinf(b_o[i]):
                raise Error(
                    '{"error_type":"WEIGHT_LOAD_FAILED","detail":"non-finite'
                    ' value in b_o","stage":"oproj","layer":'
                    + String(layer_idx)
                    + "}"
                )

    var out = List[Float32]()
    out.resize(seq_len * hidden, Float32(0.0))

    var p_x = x.unsafe_ptr()
    var p_w = w_o.unsafe_ptr()
    var p_b = b_o.unsafe_ptr()
    var p_out = out.unsafe_ptr()

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
            if has_bias:
                acc += p_b[unsafe_offset=j]
            if isnan(acc) or isinf(acc):
                raise Error(
                    '{"error_type":"ATTENTION_ERROR","detail":"non-finite'
                    ' value in o_proj output","stage":"oproj","layer":'
                    + String(layer_idx)
                    + "}"
                )
            p_out[unsafe_offset=x_row + j] = acc
    return out^


def forward_attention_block(
    x: List[Float32],
    weights: AttentionWeights,
    seq_len: Int,
    cfg: ModelConfig,
    eps: Float32,
    pos_offset: Int = 0,
    base: Float32 = Float32(1000000.0),
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Pipeline blok attention utuh: RMSNorm F6 -> QKV (+bias) -> RoPE rotate_half F7 -> MHA Causal -> o_proj (+bias) -> Residual.

    @spec docs/milestones/M2-attention.md
    @spec scratch/wave/m2/m2-w3-attention.md
    """
    var hidden = cfg.hidden_size
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"activation length'
            ' mismatch","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    # 1. RMSNorm per-token (reuse F6 dari M1 tanpa duplikasi)
    var x_norm = List[Float32]()
    x_norm.reserve(seq_len * hidden)
    var tok_vec = List[Float32]()
    tok_vec.resize(hidden, Float32(0.0))
    var p_tok = tok_vec.unsafe_ptr()
    var p_raw_x = x.unsafe_ptr()
    for t in range(seq_len):
        var row = t * hidden
        for k in range(hidden):
            p_tok[unsafe_offset=k] = p_raw_x[unsafe_offset=row + k]
        var normed = rmsnorm(tok_vec, weights.norm_gamma, eps)
        var p_normed = normed.unsafe_ptr()
        for k in range(hidden):
            x_norm.append(p_normed[unsafe_offset=k])

    # 2. QKV Projection (W1)
    var qkv_res = qkv_forward(x_norm, weights.qkv, seq_len, cfg)
    ref q = qkv_res[0]
    ref k = qkv_res[1]
    ref v = qkv_res[2]

    # 3. RoPE rotate_half F7 (W2)
    var rope_res = apply_rope(q, k, seq_len, cfg, pos_offset, base, layer_idx)
    ref q_rot = rope_res[0]
    ref k_rot = rope_res[1]

    # 4. MHA dengan Causal Mask & Softmax Stabil (W3)
    var attn_out = mha_forward(q_rot, k_rot, v, seq_len, cfg, layer_idx)

    # 5. o_proj (W3)
    var y = o_project(
        attn_out, weights.w_o, weights.b_o, seq_len, hidden, layer_idx
    )

    # 6. Residual connection y_final = y + x (W3)
    var y_final = add_residual(y, x, layer_idx)
    return y_final^


def load_layer_attention_weights(
    layer_idx: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    mut telemetry: LoadMemoryTelemetry,
) raises -> AttentionWeights:
    """Memuat seluruh bobot blok attention satu layer dari shard safetensors.

    - input_layernorm.weight
    - QKV weights (W_q, W_k, W_v, b_q, b_k, b_v)
    - o_proj.weight (+ b_o jika ada di checkpoint)
    """
    if layer_idx < 0 or layer_idx >= cfg.num_hidden_layers:
        raise Error(
            '{"error_type":"LAYER_INVALID","detail":"invalid layer index: '
            + String(layer_idx)
            + " (expected 0.."
            + String(cfg.num_hidden_layers - 1)
            + ')","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    var hidden = cfg.hidden_size
    var prefix = "model.layers." + String(layer_idx) + "."
    var norm_name = prefix + "input_layernorm.weight"
    var o_proj_name = prefix + "self_attn.o_proj.weight"
    var o_bias_name = prefix + "self_attn.o_proj.bias"

    if norm_name not in weight_map:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"norm weight not in'
            " weight_map: "
            + norm_name
            + '","stage":"attention","layer":'
            + String(layer_idx)
            + ',"shard":"","tensor_name":"'
            + norm_name
            + '"}'
        )
    if o_proj_name not in weight_map:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"o_proj weight not in'
            " weight_map: "
            + o_proj_name
            + '","stage":"attention","layer":'
            + String(layer_idx)
            + ',"shard":"","tensor_name":"'
            + o_proj_name
            + '"}'
        )

    var norm_gamma = _load_one_tensor_by_name(
        model_root,
        weight_map[norm_name],
        norm_name,
        hidden,
        1,
        False,
        telemetry,
    )
    var qkv = load_layer_qkv_weights(
        layer_idx, model_root, weight_map, cfg, telemetry
    )
    var w_o = _load_one_tensor_by_name(
        model_root,
        weight_map[o_proj_name],
        o_proj_name,
        hidden,
        hidden,
        True,
        telemetry,
    )
    var b_o = List[Float32]()
    if o_bias_name in weight_map:
        b_o = _load_one_tensor_by_name(
            model_root,
            weight_map[o_bias_name],
            o_bias_name,
            hidden,
            1,
            False,
            telemetry,
        )

    return AttentionWeights(norm_gamma^, qkv^, w_o^, b_o^)
