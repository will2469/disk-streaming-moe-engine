# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Gated Attention dengan Grouped Query Attention (GQA 16Q/2KV) & Physical 5D KV Cache (M9-W2).

Mengimplementasikan arsitektur Gated Attention:
- 4-cabang proyeksi: Q, K, V, Gate
- GQA 16Q/2KV (repeat 8->1 atau grup H_q // H_kv)
- RoPE per-head untuk Q (H_q) dan K (H_kv)
- Physical 5D KV Cache: [2 (K/V), capacity, L_att, H_kv, d_h]
- Attention dengan causal masking dan softmax stabil numerik
- Element-wise sigmoid gating: Attn_gated = Attn * sigmoid(Gate)
- Output projection W_o (+ b_o)
"""

from core.config import ModelConfig
from std.builtin.dtype import DType
from std.collections import List
from std.math import cos, exp, isinf, isnan, pow, sin, sqrt


@fieldwise_init
struct GatedAttentionWeights(Copyable, Movable):
    """Bobot parameter Gated Attention layer: Q, K, V, Gate, O."""

    var w_q: List[Float32]  # [H_q * d_h, hidden]
    var b_q: List[Float32]  # [H_q * d_h] (opsional jika attention_bias)
    var w_k: List[Float32]  # [H_kv * d_h, hidden]
    var b_k: List[Float32]  # [H_kv * d_h]
    var w_v: List[Float32]  # [H_kv * d_h, hidden]
    var b_v: List[Float32]  # [H_kv * d_h]
    var w_gate: List[Float32]  # [H_q * d_h, hidden]
    var b_gate: List[Float32]  # [H_q * d_h]
    var w_o: List[Float32]  # [hidden, H_q * d_h]
    var b_o: List[Float32]  # [hidden]


struct GatedAttnKVCache(Copyable, Movable):
    """Physical 5D KV Cache kanonis untuk model port M9.

    Shape: [2, capacity, L_att, H_kv, d_h]
    Dimensi terdepan:
    - Index 0 = Key (K)
    - Index 1 = Value (V)
    Formula F2: M_KV = 2 * L_att * H_kv * d_h * capacity * b_KV.
    """

    var capacity: Int
    var current_len: Int
    var l_att: Int
    var h_kv: Int
    var head_dim: Int
    var data: List[Float32]

    def __init__(
        out self, capacity: Int, l_att: Int, h_kv: Int, head_dim: Int
    ) raises:
        if capacity <= 0 or l_att <= 0 or h_kv <= 0 or head_dim <= 0:
            raise Error(
                '{"error_type":"KV_ALLOC_ERROR","detail":"KV cache dimensions'
                ' must be positive"}'
            )
        self.capacity = capacity
        self.current_len = 0
        self.l_att = l_att
        self.h_kv = h_kv
        self.head_dim = head_dim
        var total_elems = 2 * capacity * l_att * h_kv * head_dim
        self.data = List[Float32]()
        self.data.resize(total_elems, Float32(0.0))

    def _offset(
        self, kv_idx: Int, pos: Int, att_idx: Int, head_idx: Int, d: Int
    ) -> Int:
        return (
            ((kv_idx * self.capacity + pos) * self.l_att + att_idx) * self.h_kv
            + head_idx
        ) * self.head_dim + d

    def get_k(self, pos: Int, att_idx: Int, head_idx: Int, d: Int) -> Float32:
        return self.data[self._offset(0, pos, att_idx, head_idx, d)]

    def set_k(
        mut self, pos: Int, att_idx: Int, head_idx: Int, d: Int, val: Float32
    ):
        self.data[self._offset(0, pos, att_idx, head_idx, d)] = val

    def get_v(self, pos: Int, att_idx: Int, head_idx: Int, d: Int) -> Float32:
        return self.data[self._offset(1, pos, att_idx, head_idx, d)]

    def set_v(
        mut self, pos: Int, att_idx: Int, head_idx: Int, d: Int, val: Float32
    ):
        self.data[self._offset(1, pos, att_idx, head_idx, d)] = val

    def store_tokens(
        mut self,
        att_idx: Int,
        pos_offset: Int,
        seq_len: Int,
        k: List[Float32],
        v: List[Float32],
    ) raises:
        """Menyimpan Key dan Value hasil proyeksi ke KV cache pada layer att_idx.
        """
        if att_idx < 0 or att_idx >= self.l_att:
            raise Error(
                '{"error_type":"KV_CACHE_ERROR","detail":"att_idx out of'
                ' range"}'
            )
        if pos_offset < 0 or pos_offset + seq_len > self.capacity:
            raise Error(
                '{"error_type":"KV_CACHE_ERROR","detail":"pos exceeds KV'
                ' capacity"}'
            )
        var kv_dim = self.h_kv * self.head_dim
        if len(k) != seq_len * kv_dim or len(v) != seq_len * kv_dim:
            raise Error(
                '{"error_type":"KV_CACHE_ERROR","detail":"k/v size mismatch"}'
            )

        var p_k = k.unsafe_ptr()
        var p_v = v.unsafe_ptr()

        for t in range(seq_len):
            var pos = pos_offset + t
            for h in range(self.h_kv):
                for d in range(self.head_dim):
                    var in_off = t * kv_dim + h * self.head_dim + d
                    var k_val = p_k[unsafe_offset=in_off]
                    var v_val = p_v[unsafe_offset=in_off]
                    if (
                        isnan(k_val)
                        or isinf(k_val)
                        or isnan(v_val)
                        or isinf(v_val)
                    ):
                        raise Error(
                            '{"error_type":"LAYER_FORWARD_ERROR","detail":"non-finite'
                            ' value in KV cache append"}'
                        )
                    self.set_k(pos, att_idx, h, d, k_val)
                    self.set_v(pos, att_idx, h, d, v_val)

        if pos_offset + seq_len > self.current_len:
            self.current_len = pos_offset + seq_len

    def zero(mut self):
        var total = len(self.data)
        var p = self.data.unsafe_ptr()
        for i in range(total):
            p[unsafe_offset=i] = Float32(0.0)
        self.current_len = 0


def apply_rope_to_heads(
    x: List[Float32],
    seq_len: Int,
    num_heads: Int,
    head_dim: Int,
    pos_offset: Int = 0,
    base: Float32 = Float32(1000000.0),
) raises -> List[Float32]:
    """Menerapkan RoPE half-rotation ke tensor berbentuk [seq_len, num_heads * head_dim].
    """
    if head_dim % 2 != 0:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"head_dim must be even"}'
        )
    var dim = num_heads * head_dim
    if len(x) != seq_len * dim:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"length mismatch in'
            ' apply_rope_to_heads"}'
        )

    var half_dim = head_dim // 2
    var inv_freq = List[Float32]()
    inv_freq.reserve(half_dim)
    for i in range(half_dim):
        var exp_val = Float32(2 * i) / Float32(head_dim)
        inv_freq.append(Float32(1.0) / pow(base, exp_val))

    var out = List[Float32]()
    out.resize(seq_len * dim, Float32(0.0))

    var p_x = x.unsafe_ptr()
    var p_out = out.unsafe_ptr()

    for t in range(seq_len):
        var m = Float32(pos_offset + t)
        var t_offset = t * dim
        for h in range(num_heads):
            var head_start = t_offset + h * head_dim
            for k in range(half_dim):
                var rot_cos = cos(m * inv_freq[k])
                var rot_sin = sin(m * inv_freq[k])
                var val_first = p_x[unsafe_offset=head_start + k]
                var val_second = p_x[unsafe_offset=head_start + half_dim + k]
                p_out[unsafe_offset=head_start + k] = (
                    val_first * rot_cos - val_second * rot_sin
                )
                p_out[unsafe_offset=head_start + half_dim + k] = (
                    val_first * rot_sin + val_second * rot_cos
                )

    return out^


def linear_projection(
    x: List[Float32],
    w: List[Float32],
    b: List[Float32],
    seq_len: Int,
    in_dim: Int,
    out_dim: Int,
) raises -> List[Float32]:
    """Proyeksi linear: y = x @ W^T (+ b)."""
    if len(x) != seq_len * in_dim or len(w) != out_dim * in_dim:
        raise Error(
            '{"error_type":"LAYER_FORWARD_ERROR","detail":"dimension mismatch'
            ' in linear projection"}'
        )
    var has_bias = len(b) > 0
    if has_bias and len(b) != out_dim:
        raise Error(
            '{"error_type":"LAYER_FORWARD_ERROR","detail":"bias length mismatch'
            ' in linear projection"}'
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
                    '{"error_type":"LAYER_FORWARD_ERROR","detail":"non-finite'
                    ' value in linear projection"}'
                )
            p_out[unsafe_offset=out_off + o] = dot

    return out^


def gated_attention_forward(
    x_norm: List[Float32],
    weights: GatedAttentionWeights,
    mut kv_cache: GatedAttnKVCache,
    att_idx: Int,
    pos_offset: Int,
    seq_len: Int,
    cfg: ModelConfig,
    layer_idx: Int = 3,
) raises -> List[Float32]:
    """Forward pass Gated Attention dengan GQA 16Q/2KV & Sigmoid Gating.

    Rumus:
    1. Q = x_norm W_q^T (+ b_q)
       K = x_norm W_k^T (+ b_k)
       V = x_norm W_v^T (+ b_v)
       Gate = x_norm W_gate^T (+ b_gate)
    2. RoPE: Q_rot = rope(Q, H_q, d_h), K_rot = rope(K, H_kv, d_h)
    3. Simpan K_rot dan V ke KV Cache slot att_idx
    4. GQA Attention: softmax(Q_rot K_cache^T / sqrt(d_h) + M) V_cache
    5. Sigmoid gating: Attn_gated = Attn * sigmoid(Gate)
    6. Output projection: y = Attn_gated W_o^T (+ b_o)
    """
    var hidden = cfg.hidden_size
    var h_q = cfg.num_attention_heads
    var h_kv = cfg.num_key_value_heads
    var head_dim = cfg.head_dim()
    var q_dim = h_q * head_dim
    var kv_dim = h_kv * head_dim
    var gqa_group = h_q // h_kv

    # 1. Proyeksi 4-Cabang (Q, K, V, Gate)
    var q_raw = linear_projection(
        x_norm, weights.w_q, weights.b_q, seq_len, hidden, q_dim
    )
    var k_raw = linear_projection(
        x_norm, weights.w_k, weights.b_k, seq_len, hidden, kv_dim
    )
    var v_raw = linear_projection(
        x_norm, weights.w_v, weights.b_v, seq_len, hidden, kv_dim
    )
    var gate_raw = linear_projection(
        x_norm, weights.w_gate, weights.b_gate, seq_len, hidden, q_dim
    )

    # 2. RoPE per-head
    var q_rot = apply_rope_to_heads(
        q_raw, seq_len, h_q, head_dim, pos_offset, Float32(1000000.0)
    )
    var k_rot = apply_rope_to_heads(
        k_raw, seq_len, h_kv, head_dim, pos_offset, Float32(1000000.0)
    )

    # 3. Update KV Cache
    kv_cache.store_tokens(att_idx, pos_offset, seq_len, k_rot, v_raw)

    # 4. GQA Scaled Dot-Product Attention
    var total_seq = pos_offset + seq_len
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    var attn_out = List[Float32]()
    attn_out.resize(seq_len * q_dim, Float32(0.0))
    var p_q = q_rot.unsafe_ptr()
    var p_attn = attn_out.unsafe_ptr()

    var scores = List[Float32]()
    scores.resize(total_seq, Float32(0.0))
    var p_scores = scores.unsafe_ptr()

    var probs = List[Float32]()
    probs.resize(total_seq, Float32(0.0))
    var p_probs = probs.unsafe_ptr()

    for h in range(h_q):
        var kv_h = h // gqa_group
        for i in range(seq_len):
            var current_pos = pos_offset + i
            var q_off = i * q_dim + h * head_dim

            # Hitung scores dot-product Q . K untuk j <= current_pos
            var max_score = -Float32(1e30)
            for j in range(current_pos + 1):
                var acc = Float32(0.0)
                for d in range(head_dim):
                    var q_val = p_q[unsafe_offset=q_off + d]
                    var k_val = kv_cache.get_k(j, att_idx, kv_h, d)
                    acc += q_val * k_val
                var score = acc * scale
                p_scores[unsafe_offset=j] = score
                if score > max_score:
                    max_score = score

            # Stable Softmax
            var sum_exp = Float32(0.0)
            for j in range(current_pos + 1):
                var ev = exp(p_scores[unsafe_offset=j] - max_score)
                p_probs[unsafe_offset=j] = ev
                sum_exp += ev

            if sum_exp <= Float32(0.0) or isnan(sum_exp) or isinf(sum_exp):
                raise Error(
                    '{"error_type":"LAYER_FORWARD_ERROR","detail":"invalid'
                    ' attention softmax sum","layer":'
                    + String(layer_idx)
                    + "}"
                )
            var inv_sum = Float32(1.0) / sum_exp
            for j in range(current_pos + 1):
                p_probs[unsafe_offset=j] *= inv_sum

            # Akumulasi Context Vector Attn = probs @ V
            var out_head_off = i * q_dim + h * head_dim
            for d in range(head_dim):
                var val = Float32(0.0)
                for j in range(current_pos + 1):
                    val += p_probs[unsafe_offset=j] * kv_cache.get_v(
                        j, att_idx, kv_h, d
                    )
                if isnan(val) or isinf(val):
                    raise Error(
                        '{"error_type":"LAYER_FORWARD_ERROR","detail":"non-finite'
                        ' in attention context output","layer":'
                        + String(layer_idx)
                        + "}"
                    )
                p_attn[unsafe_offset=out_head_off + d] = val

    # 5. Sigmoid Gating: Attn_gated = Attn * sigmoid(Gate)
    var p_gate = gate_raw.unsafe_ptr()
    for idx in range(seq_len * q_dim):
        var raw_g = p_gate[unsafe_offset=idx]
        var sig_g = Float32(1.0) / (Float32(1.0) + exp(-raw_g))
        p_attn[unsafe_offset=idx] *= sig_g

    # 6. Output Projection: y = Attn_gated @ W_o^T (+ b_o)
    return linear_projection(
        attn_out, weights.w_o, weights.b_o, seq_len, q_dim, hidden
    )
