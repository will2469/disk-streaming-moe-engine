# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Multi-Head Attention (MHA) dengan causal mask dan softmax stabil fp32."""

from core.config import ModelConfig
from layers.kv_cache import LayerKVCache, SLOT_DIM
from std.builtin.dtype import DType
from std.collections import List
from std.math import exp, isinf, isnan, sqrt


def build_causal_mask(seq_len: Int) raises -> List[Float32]:
    """Bangun matriks causal mask triangular [seq_len, seq_len].

    M[i, j] = 0.0 jika i >= j, else -inf.
    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if seq_len <= 0:
        raise Error(
            '{"error_type":"MASK_ERROR","detail":"seq_len must be'
            ' positive","stage":"attention","layer":0}'
        )
    var mask = List[Float32]()
    mask.resize(seq_len * seq_len, Float32(0.0))
    var p_mask = mask.unsafe_ptr()
    var neg_inf = -Float32(1.0) / Float32(0.0)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                p_mask[unsafe_offset=i * seq_len + j] = neg_inf
            else:
                p_mask[unsafe_offset=i * seq_len + j] = Float32(0.0)
    return mask^


def verify_causal_mask_property(
    probs: List[Float32],
    seq_len: Int,
    num_heads: Int,
    layer_idx: Int = 0,
) raises:
    """Verifikasi bahwa token i HANYA mengattend ke token j <= i (prob[i, j] == 0 untuk j > i).

    @spec scratch/wave/m2/m2-w3-attention.md
    """
    var p_probs = probs.unsafe_ptr()
    for h in range(num_heads):
        var head_offset = h * seq_len * seq_len
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                var val = p_probs[unsafe_offset=head_offset + i * seq_len + j]
                if val != Float32(0.0):
                    raise Error(
                        '{"error_type":"MASK_ERROR","detail":"causal mask'
                        " violation: future attention detected at i="
                        + String(i)
                        + ", j="
                        + String(j)
                        + '","stage":"attention","layer":'
                        + String(layer_idx)
                        + "}"
                    )


def softmax_row_stable(
    scores: List[Float32],
    offset: Int,
    length: Int,
    valid_len: Int,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Softmax stabil numerik dengan pergeseran nilai maksimum: exp(z - max z) / sum exp(z - max z).

    Elemen j >= valid_len diatur ke probabilitas 0.0 secara eksak.
    Deteksi NaN atau unmasked Inf menghasilkan ATTENTION_ERROR.
    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if valid_len <= 0 or valid_len > length:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"invalid'
            ' valid_len","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )
    var p_scores = scores.unsafe_ptr()

    var max_val = p_scores[unsafe_offset=offset]
    for j in range(valid_len):
        var s = p_scores[unsafe_offset=offset + j]
        if isnan(s) or isinf(s):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite score in'
                ' attention logits","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )
        if s > max_val:
            max_val = s

    var sum_exp = Float32(0.0)
    var probs = List[Float32]()
    probs.resize(length, Float32(0.0))
    var p_probs = probs.unsafe_ptr()

    for j in range(valid_len):
        var e = exp(p_scores[unsafe_offset=offset + j] - max_val)
        if isnan(e) or isinf(e):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"overflow in'
                ' softmax exponential","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )
        p_probs[unsafe_offset=j] = e
        sum_exp += e

    if sum_exp <= Float32(0.0) or isnan(sum_exp) or isinf(sum_exp):
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"invalid softmax'
            ' denominator","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    for j in range(valid_len):
        p_probs[unsafe_offset=j] = p_probs[unsafe_offset=j] / sum_exp

    return probs^


def mha_forward(
    q: List[Float32],
    k: List[Float32],
    v: List[Float32],
    seq_len: Int,
    cfg: ModelConfig,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Multi-Head Attention dengan causal mask dan softmax stabil.

    Rumus: softmax(Q K^T / sqrt(d_h) + M) V
    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if seq_len <= 0:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"seq_len must be'
            ' positive","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )
    var hidden = cfg.hidden_size
    var num_heads = cfg.num_attention_heads
    var head_dim = cfg.head_dim()

    if num_heads * head_dim != hidden:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"num_attention_heads *'
            ' head_dim != hidden_size","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )
    if (
        len(q) != seq_len * hidden
        or len(k) != seq_len * hidden
        or len(v) != seq_len * hidden
    ):
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"Q, K, V length'
            ' mismatch","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    for i in range(len(q)):
        if isnan(q[i]) or isinf(q[i]):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' Q","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )
    for i in range(len(k)):
        if isnan(k[i]) or isinf(k[i]):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' K","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )
    for i in range(len(v)):
        if isnan(v[i]) or isinf(v[i]):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' V","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )

    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var out = List[Float32]()
    out.resize(seq_len * hidden, Float32(0.0))

    var p_q = q.unsafe_ptr()
    var p_k = k.unsafe_ptr()
    var p_v = v.unsafe_ptr()
    var p_out = out.unsafe_ptr()

    var scores_row = List[Float32]()
    scores_row.resize(seq_len, Float32(0.0))
    var p_scores = scores_row.unsafe_ptr()

    var all_probs = List[Float32]()
    all_probs.resize(num_heads * seq_len * seq_len, Float32(0.0))
    var p_all_probs = all_probs.unsafe_ptr()

    for h in range(num_heads):
        for i in range(seq_len):
            for j in range(i + 1):
                var q_offset = i * hidden + h * head_dim
                var k_offset = j * hidden + h * head_dim
                var acc_simd = SIMD[DType.float32, 16](0.0)
                var d = 0
                while d + 16 <= head_dim:
                    acc_simd += p_q.unsafe_load[width=16](
                        q_offset + d
                    ) * p_k.unsafe_load[width=16](k_offset + d)
                    d += 16
                var acc = acc_simd.reduce_add()
                while d < head_dim:
                    acc += (
                        p_q[unsafe_offset=q_offset + d]
                        * p_k[unsafe_offset=k_offset + d]
                    )
                    d += 1
                p_scores[unsafe_offset=j] = acc * scale

            var probs_row = softmax_row_stable(
                scores_row, 0, seq_len, i + 1, layer_idx
            )
            var p_prow = probs_row.unsafe_ptr()

            var prob_base = (h * seq_len + i) * seq_len
            for j in range(seq_len):
                p_all_probs[unsafe_offset=prob_base + j] = p_prow[
                    unsafe_offset=j
                ]

            var out_offset = i * hidden + h * head_dim
            for d in range(head_dim):
                var val = Float32(0.0)
                for j in range(i + 1):
                    var v_offset = j * hidden + h * head_dim
                    val += (
                        p_prow[unsafe_offset=j]
                        * p_v[unsafe_offset=v_offset + d]
                    )
                if isnan(val) or isinf(val):
                    raise Error(
                        '{"error_type":"ATTENTION_ERROR","detail":"non-finite'
                        " value in MHA context"
                        ' output","stage":"attention","layer":'
                        + String(layer_idx)
                        + "}"
                    )
                p_out[unsafe_offset=out_offset + d] = val

    verify_causal_mask_property(all_probs, seq_len, num_heads, layer_idx)
    return out^


def mha_decode_step(
    q: List[Float32],
    layer_kv: LayerKVCache,
    cache_len: Int,
    cfg: ModelConfig,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Incremental Multi-Head Attention decode step untuk 1 token pada posisi sequence p.

    q: [hidden_size] float32 query vector token baru pada posisi p.
    layer_kv: LayerKVCache menyimpan K dan V dari slot 0 s/d cache_len - 1 dalam BF16.
    cache_len: jumlah total posisi terisi di cache termasuk token p (cache_len = p + 1).

    Invarian:
    - Tidak ada alokasi buffer baru di dalam decode step (zero heap allocation).
    - Membaca langsung dari pointer BF16 layer_kv.
    - Menghitung attention scores [cache_len] per head, softmax stabil, dan akumulasi context output.
    """
    if cache_len <= 0:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"cache_len must be'
            ' positive","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )
    if cache_len > layer_kv.current_len:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"cache_len exceeds'
            ' layer_kv.current_len","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    var hidden = cfg.hidden_size
    var num_heads = cfg.num_attention_heads
    var head_dim = cfg.head_dim()

    if num_heads * head_dim != hidden:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"num_attention_heads *'
            ' head_dim != hidden_size","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(q) != hidden:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"Q length mismatch'
            ' against hidden_size","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    for i in range(len(q)):
        if isnan(q[i]) or isinf(q[i]):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' Q","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )

    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var out = List[Float32]()
    out.resize(hidden, Float32(0.0))

    var p_q = q.unsafe_ptr()
    var p_k = layer_kv.k.unsafe_ptr()
    var p_v = layer_kv.v.unsafe_ptr()
    var p_out = out.unsafe_ptr()

    var scores_row = List[Float32]()
    scores_row.resize(cache_len, Float32(0.0))
    var p_scores = scores_row.unsafe_ptr()

    for h in range(num_heads):
        var q_head_offset = h * head_dim

        # 1. Hitung attention scores: dot(Q[h], K[j, h]) * scale untuk j in [0, cache_len)
        for j in range(cache_len):
            var k_offset = j * SLOT_DIM + h * head_dim
            var acc = Float32(0.0)
            for d in range(head_dim):
                acc += p_q[unsafe_offset=q_head_offset + d] * Float32(
                    p_k[unsafe_offset=k_offset + d]
                )
            p_scores[unsafe_offset=j] = acc * scale

        # 2. Softmax stabil numerik terhadap panjang cache_len
        var probs = softmax_row_stable(
            scores_row, 0, cache_len, cache_len, layer_idx
        )
        var p_probs = probs.unsafe_ptr()

        # 3. Akumulasi context output O[h] = sum_j probs[j] * V[j, h]
        var out_head_offset = h * head_dim
        for d in range(head_dim):
            var val = Float32(0.0)
            for j in range(cache_len):
                var v_offset = j * SLOT_DIM + h * head_dim + d
                val += p_probs[unsafe_offset=j] * Float32(
                    p_v[unsafe_offset=v_offset]
                )
            if isnan(val) or isinf(val):
                raise Error(
                    '{"error_type":"ATTENTION_ERROR","detail":"non-finite value'
                    ' in incremental MHA output","stage":"attention","layer":'
                    + String(layer_idx)
                    + "}"
                )
            p_out[unsafe_offset=out_head_offset + d] = val

    return out^
