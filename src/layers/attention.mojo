# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Blok attention lengkap: RMSNorm -> QKV -> RoPE -> MHA -> o_proj -> Residual (M2)."""

from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import ShardHeaderCache, _load_one_tensor_by_name
from layers.mha import mha_forward
from layers.qkv import QKVWeights, load_layer_qkv_weights, qkv_forward
from layers.residual import add_residual
from layers.rmsnorm import rmsnorm
from layers.rope import apply_rope
from std.builtin.dtype import DType
from std.collections import Dict, List
from std.math import cos, exp, isinf, isnan, pow, sin, sqrt


struct AttentionWeights(Copyable, Movable):
    """Bobot attention satu layer: norm_gamma, QKV (w+b), w_o, b_o, q_norm, k_norm, is_qwen36.
    """

    var norm_gamma: List[Float32]
    var qkv: QKVWeights
    var w_o: List[Float32]
    var b_o: List[Float32]
    var q_norm: List[Float32]
    var k_norm: List[Float32]
    var is_qwen36: Bool

    def __init__(
        out self,
        norm_gamma: List[Float32],
        qkv: QKVWeights,
        w_o: List[Float32],
        b_o: List[Float32],
        q_norm: List[Float32] = List[Float32](),
        k_norm: List[Float32] = List[Float32](),
        is_qwen36: Bool = False,
    ):
        self.norm_gamma = norm_gamma.copy()
        self.qkv = qkv.copy()
        self.w_o = w_o.copy()
        self.b_o = b_o.copy()
        self.q_norm = q_norm.copy()
        self.k_norm = k_norm.copy()
        self.is_qwen36 = is_qwen36


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


def linear_proj_simd(
    x: List[Float32],
    w: List[Float32],
    b: List[Float32],
    seq_len: Int,
    in_dim: Int,
    out_dim: Int,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Proyeksi linear: y = x @ W^T (+ b) dengan akselerasi SIMD width 16."""
    if len(x) != seq_len * in_dim or len(w) != out_dim * in_dim:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"dimension mismatch in'
            ' linear projection","layer":'
            + String(layer_idx)
            + "}"
        )
    var has_bias = len(b) > 0
    if has_bias and len(b) != out_dim:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"bias mismatch in linear'
            ' projection","layer":'
            + String(layer_idx)
            + "}"
        )
    var out = List[Float32]()
    out.resize(seq_len * out_dim, Float32(0.0))

    var p_x = x.unsafe_ptr()
    var p_w = w.unsafe_ptr()
    var p_b = b.unsafe_ptr()
    var p_out = out.unsafe_ptr()

    for t in range(seq_len):
        var x_off = t * in_dim
        var out_off = t * out_dim
        for o in range(out_dim):
            var w_off = o * in_dim
            var acc_simd = SIMD[DType.float32, 16](0.0)
            var c = 0
            while c + 16 <= in_dim:
                acc_simd += p_x.unsafe_load[width=16](
                    x_off + c
                ) * p_w.unsafe_load[width=16](w_off + c)
                c += 16
            var dot = acc_simd.reduce_add()
            while c < in_dim:
                dot += (
                    p_x[unsafe_offset=x_off + c] * p_w[unsafe_offset=w_off + c]
                )
                c += 1
            if has_bias:
                dot += p_b[unsafe_offset=o]
            if isnan(dot) or isinf(dot):
                raise Error(
                    '{"error_type":"ATTENTION_ERROR","detail":"non-finite value'
                    ' in projection","layer":'
                    + String(layer_idx)
                    + "}"
                )
            p_out[unsafe_offset=out_off + o] = dot
    return out^


def forward_attention_qwen36(
    x: List[Float32],
    weights: AttentionWeights,
    seq_len: Int,
    cfg: ModelConfig,
    eps: Float32,
    pos_offset: Int = 0,
    base: Float32 = Float32(10000000.0),
    layer_idx: Int = 0,
) raises -> List[Float32]:
    var hidden = cfg.hidden_size
    var num_heads = cfg.num_attention_heads
    var head_dim = cfg.head_dim()
    var num_kv_heads = cfg.num_key_value_heads
    var q_proj_dim = 2 * num_heads * head_dim
    var q_dim = num_heads * head_dim
    var kv_dim = num_kv_heads * head_dim

    # 1. RMSNorm per-token
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

    # 2. Linear projections (Q+Gate, K, V)
    var empty_b = List[Float32]()
    var q_proj_out = linear_proj_simd(
        x_norm, weights.qkv.w_q, empty_b, seq_len, hidden, q_proj_dim, layer_idx
    )
    var k_raw = linear_proj_simd(
        x_norm, weights.qkv.w_k, empty_b, seq_len, hidden, kv_dim, layer_idx
    )
    var v_raw = linear_proj_simd(
        x_norm, weights.qkv.w_v, empty_b, seq_len, hidden, kv_dim, layer_idx
    )

    # Split q_proj_out into q and gate
    var q = List[Float32]()
    q.resize(seq_len * q_dim, Float32(0.0))
    var gate = List[Float32]()
    gate.resize(seq_len * q_dim, Float32(0.0))
    var p_qpo = q_proj_out.unsafe_ptr()
    var p_q = q.unsafe_ptr()
    var p_gate = gate.unsafe_ptr()
    for t in range(seq_len):
        var src_off = t * q_proj_dim
        var dst_off = t * q_dim
        for i in range(q_dim):
            p_q[unsafe_offset=dst_off + i] = p_qpo[unsafe_offset=src_off + i]
            p_gate[unsafe_offset=dst_off + i] = p_qpo[
                unsafe_offset=src_off + q_dim + i
            ]

    # 3. QK-Norm per-head
    var p_k = k_raw.unsafe_ptr()
    var p_qn = weights.q_norm.unsafe_ptr()
    var p_kn = weights.k_norm.unsafe_ptr()

    for t in range(seq_len):
        # Q-Norm across 16 heads
        for h in range(num_heads):
            var head_off = t * q_dim + h * head_dim
            var acc_var = Float32(0.0)
            for d in range(head_dim):
                var val = p_q[unsafe_offset=head_off + d]
                acc_var += val * val
            var rsqrt = Float32(1.0) / sqrt(acc_var / Float32(head_dim) + eps)
            for d in range(head_dim):
                p_q[unsafe_offset=head_off + d] = (
                    p_q[unsafe_offset=head_off + d]
                    * rsqrt
                    * p_qn[unsafe_offset=d]
                )

        # K-Norm across 2 KV heads
        for kv in range(num_kv_heads):
            var head_off = t * kv_dim + kv * head_dim
            var acc_var = Float32(0.0)
            for d in range(head_dim):
                var val = p_k[unsafe_offset=head_off + d]
                acc_var += val * val
            var rsqrt = Float32(1.0) / sqrt(acc_var / Float32(head_dim) + eps)
            for d in range(head_dim):
                p_k[unsafe_offset=head_off + d] = (
                    p_k[unsafe_offset=head_off + d]
                    * rsqrt
                    * p_kn[unsafe_offset=d]
                )

    # 4. Partial RoPE (factor 0.25 -> rotary_dim = 64, half_rot = 32)
    var rotary_factor = cfg.partial_rotary_factor
    var rotary_dim = Int(Float32(head_dim) * rotary_factor)
    var half_rot = rotary_dim // 2
    var inv_freq = List[Float32]()
    inv_freq.reserve(half_rot)
    for i in range(half_rot):
        var exp_val = Float32(2 * i) / Float32(rotary_dim)
        inv_freq.append(Float32(1.0) / pow(base, exp_val))

    for t in range(seq_len):
        var m = Float32(pos_offset + t)
        # RoPE on Q
        for h in range(num_heads):
            var head_start = t * q_dim + h * head_dim
            for k in range(half_rot):
                var rot_cos = cos(m * inv_freq[k])
                var rot_sin = sin(m * inv_freq[k])
                var v0 = p_q[unsafe_offset=head_start + k]
                var v1 = p_q[unsafe_offset=head_start + half_rot + k]
                p_q[unsafe_offset=head_start + k] = v0 * rot_cos - v1 * rot_sin
                p_q[unsafe_offset=head_start + half_rot + k] = (
                    v0 * rot_sin + v1 * rot_cos
                )

        # RoPE on K
        for kv in range(num_kv_heads):
            var head_start = t * kv_dim + kv * head_dim
            for k in range(half_rot):
                var rot_cos = cos(m * inv_freq[k])
                var rot_sin = sin(m * inv_freq[k])
                var v0 = p_k[unsafe_offset=head_start + k]
                var v1 = p_k[unsafe_offset=head_start + half_rot + k]
                p_k[unsafe_offset=head_start + k] = v0 * rot_cos - v1 * rot_sin
                p_k[unsafe_offset=head_start + half_rot + k] = (
                    v0 * rot_sin + v1 * rot_cos
                )

    # 5. GQA Causal Self-Attention
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var gqa_group = num_heads // num_kv_heads
    var attn_out = List[Float32]()
    attn_out.resize(seq_len * q_dim, Float32(0.0))
    var p_attn = attn_out.unsafe_ptr()
    var p_v = v_raw.unsafe_ptr()

    var scores = List[Float32]()
    scores.resize(seq_len, Float32(0.0))
    var p_scores = scores.unsafe_ptr()
    var probs = List[Float32]()
    probs.resize(seq_len, Float32(0.0))
    var p_probs = probs.unsafe_ptr()

    for h in range(num_heads):
        var kv_h = h // gqa_group
        for i in range(seq_len):
            var q_off = i * q_dim + h * head_dim
            var max_s = -Float32(1e30)
            for j in range(i + 1):
                var k_off = j * kv_dim + kv_h * head_dim
                var dot_simd = SIMD[DType.float32, 16](0.0)
                var d = 0
                while d + 16 <= head_dim:
                    dot_simd += p_q.unsafe_load[width=16](
                        q_off + d
                    ) * p_k.unsafe_load[width=16](k_off + d)
                    d += 16
                var dot = dot_simd.reduce_add()
                while d < head_dim:
                    dot += (
                        p_q[unsafe_offset=q_off + d]
                        * p_k[unsafe_offset=k_off + d]
                    )
                    d += 1
                var s = dot * scale
                p_scores[unsafe_offset=j] = s
                if s > max_s:
                    max_s = s

            var sum_exp = Float32(0.0)
            for j in range(i + 1):
                var ev = exp(p_scores[unsafe_offset=j] - max_s)
                p_probs[unsafe_offset=j] = ev
                sum_exp += ev
            if sum_exp <= Float32(0.0) or isnan(sum_exp) or isinf(sum_exp):
                raise Error(
                    '{"error_type":"ATTENTION_ERROR","detail":"softmax sum'
                    ' non-positive/non-finite","layer":'
                    + String(layer_idx)
                    + "}"
                )
            var inv_sum = Float32(1.0) / sum_exp
            for j in range(i + 1):
                p_probs[unsafe_offset=j] *= inv_sum

            var out_head_off = i * q_dim + h * head_dim
            for d in range(head_dim):
                var val = Float32(0.0)
                for j in range(i + 1):
                    val += (
                        p_probs[unsafe_offset=j]
                        * p_v[unsafe_offset=j * kv_dim + kv_h * head_dim + d]
                    )
                p_attn[unsafe_offset=out_head_off + d] = val

    # 6. Output Sigmoid Gating
    for idx in range(seq_len * q_dim):
        var raw_g = p_gate[unsafe_offset=idx]
        var sig_g = Float32(1.0) / (Float32(1.0) + exp(-raw_g))
        p_attn[unsafe_offset=idx] *= sig_g

    # 7. Output Projection
    var y = linear_proj_simd(
        attn_out, weights.w_o, weights.b_o, seq_len, q_dim, hidden, layer_idx
    )

    # 8. Residual connection y_final = y + x
    return add_residual(y, x, layer_idx)


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
    """Pipeline blok attention utuh: RMSNorm -> QKV -> RoPE -> MHA -> o_proj -> Residual.
    """
    if weights.is_qwen36:
        return forward_attention_qwen36(
            x, weights, seq_len, cfg, eps, pos_offset, base, layer_idx
        )

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
    mut cache: ShardHeaderCache,
    mut telemetry: LoadMemoryTelemetry,
) raises -> AttentionWeights:
    """Memuat seluruh bobot blok attention satu layer dari shard safetensors.

    Mendukung format Qwen3.6-35B-A3B (QK-Norm, Q+Gate proj, GQA, 0 bias)
    dan format legacy Qwen (QKV bias terpisah).
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
    var prefix = "model.language_model.layers." + String(layer_idx) + "."
    if (prefix + "input_layernorm.weight") not in weight_map:
        prefix = "model.layers." + String(layer_idx) + "."

    var is_qwen36 = (
        prefix + "self_attn.q_norm.weight"
    ) in weight_map or not cfg.attention_bias

    if is_qwen36:
        var norm_name = prefix + "input_layernorm.weight"
        var q_name = prefix + "self_attn.q_proj.weight"
        var k_name = prefix + "self_attn.k_proj.weight"
        var v_name = prefix + "self_attn.v_proj.weight"
        var q_norm_name = prefix + "self_attn.q_norm.weight"
        var k_norm_name = prefix + "self_attn.k_norm.weight"
        var o_name = prefix + "self_attn.o_proj.weight"

        if norm_name not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"norm weight not'
                " in weight_map: "
                + norm_name
                + '","stage":"attention","layer":'
                + String(layer_idx)
                + ',"shard":"","tensor_name":"'
                + norm_name
                + '"}'
            )
        if q_name not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"q_proj weight not'
                " in weight_map: "
                + q_name
                + '","stage":"attention","layer":'
                + String(layer_idx)
                + ',"shard":"","tensor_name":"'
                + q_name
                + '"}'
            )
        if k_name not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"k_proj weight not'
                " in weight_map: "
                + k_name
                + '","stage":"attention","layer":'
                + String(layer_idx)
                + ',"shard":"","tensor_name":"'
                + k_name
                + '"}'
            )
        if v_name not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"v_proj weight not'
                " in weight_map: "
                + v_name
                + '","stage":"attention","layer":'
                + String(layer_idx)
                + ',"shard":"","tensor_name":"'
                + v_name
                + '"}'
            )
        if q_norm_name not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"q_norm weight not'
                " in weight_map: "
                + q_norm_name
                + '","stage":"attention","layer":'
                + String(layer_idx)
                + ',"shard":"","tensor_name":"'
                + q_norm_name
                + '"}'
            )
        if k_norm_name not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"k_norm weight not'
                " in weight_map: "
                + k_norm_name
                + '","stage":"attention","layer":'
                + String(layer_idx)
                + ',"shard":"","tensor_name":"'
                + k_norm_name
                + '"}'
            )
        if o_name not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"o_proj weight not'
                " in weight_map: "
                + o_name
                + '","stage":"attention","layer":'
                + String(layer_idx)
                + ',"shard":"","tensor_name":"'
                + o_name
                + '"}'
            )

        var num_heads = cfg.num_attention_heads
        var head_dim = cfg.head_dim()
        var num_kv_heads = cfg.num_key_value_heads
        var q_dim = 2 * num_heads * head_dim
        var kv_dim = num_kv_heads * head_dim

        var norm_gamma = _load_one_tensor_by_name(
            cache,
            model_root,
            weight_map[norm_name],
            norm_name,
            hidden,
            1,
            False,
            telemetry,
        )
        var w_q = _load_one_tensor_by_name(
            cache,
            model_root,
            weight_map[q_name],
            q_name,
            q_dim,
            hidden,
            True,
            telemetry,
        )
        var w_k = _load_one_tensor_by_name(
            cache,
            model_root,
            weight_map[k_name],
            k_name,
            kv_dim,
            hidden,
            True,
            telemetry,
        )
        var w_v = _load_one_tensor_by_name(
            cache,
            model_root,
            weight_map[v_name],
            v_name,
            kv_dim,
            hidden,
            True,
            telemetry,
        )
        var q_norm = _load_one_tensor_by_name(
            cache,
            model_root,
            weight_map[q_norm_name],
            q_norm_name,
            head_dim,
            1,
            False,
            telemetry,
        )
        var k_norm = _load_one_tensor_by_name(
            cache,
            model_root,
            weight_map[k_norm_name],
            k_norm_name,
            head_dim,
            1,
            False,
            telemetry,
        )
        var w_o = _load_one_tensor_by_name(
            cache,
            model_root,
            weight_map[o_name],
            o_name,
            hidden,
            num_heads * head_dim,
            True,
            telemetry,
        )
        var empty_b = List[Float32]()
        var qkv = QKVWeights(
            w_q=w_q^,
            w_k=w_k^,
            w_v=w_v^,
            b_q=empty_b.copy(),
            b_k=empty_b.copy(),
            b_v=empty_b.copy(),
        )
        return AttentionWeights(
            norm_gamma=norm_gamma^,
            qkv=qkv^,
            w_o=w_o^,
            b_o=empty_b^,
            q_norm=q_norm^,
            k_norm=k_norm^,
            is_qwen36=True,
        )

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
        cache,
        model_root,
        weight_map[norm_name],
        norm_name,
        hidden,
        1,
        False,
        telemetry,
    )
    var qkv = load_layer_qkv_weights(
        layer_idx, model_root, weight_map, cfg, cache, telemetry
    )
    var w_o = _load_one_tensor_by_name(
        cache,
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
            cache,
            model_root,
            weight_map[o_bias_name],
            o_bias_name,
            hidden,
            1,
            False,
            telemetry,
        )

    return AttentionWeights(
        norm_gamma=norm_gamma^,
        qkv=qkv^,
        w_o=w_o^,
        b_o=b_o^,
        q_norm=List[Float32](),
        k_norm=List[Float32](),
        is_qwen36=False,
    )
