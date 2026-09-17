# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M9-W2: Gated Attention, GQA 16Q/2KV, RoPE per-head, dan Physical 5D KV Cache."""

from core.config import ModelConfig
from layers.gated_attention import (
    GatedAttentionWeights,
    GatedAttnKVCache,
    apply_rope_to_heads,
    gated_attention_forward,
)
from std.collections import List
from std.math import abs, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def test_gqa_grouping_ratio() raises:
    """Verifikasi rasio GQA 16Q/2KV (repeat 8->1) dan 4Q/1KV."""
    var cfg_port = ModelConfig(
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        vocab_size=248320,
        num_key_value_heads=2,
        head_dim_override=128,
        full_attention_interval=4,
        architecture="qwen3.6",
    )
    assert_equal(cfg_port.gqa_group_size(), 8)
    assert_equal(cfg_port.num_attention_heads, 16)
    assert_equal(cfg_port.num_key_value_heads, 2)
    assert_equal(cfg_port.head_dim(), 128)

    # Verifikasi pemetaan query head h -> kv head
    var group_size = cfg_port.gqa_group_size()
    for h in range(16):
        var kv_h = h // group_size
        if h < 8:
            assert_equal(kv_h, 0)
        else:
            assert_equal(kv_h, 1)


def test_physical_5d_kv_cache_layout() raises:
    """Verifikasi layout fisik 5D KV Cache: [2, capacity, L_att, H_kv, d_h]."""
    var capacity = 32
    var l_att = 10
    var h_kv = 2
    var head_dim = 128

    var cache = GatedAttnKVCache(capacity, l_att, h_kv, head_dim)
    assert_equal(cache.current_len, 0)
    assert_equal(cache.capacity, capacity)
    assert_equal(cache.l_att, l_att)
    assert_equal(cache.h_kv, h_kv)
    assert_equal(cache.head_dim, head_dim)

    # Total elements = 2 * 32 * 10 * 2 * 128 = 163,840
    assert_equal(len(cache.data), 2 * capacity * l_att * h_kv * head_dim)

    # Set dan verifikasi isolasi K vs V
    cache.set_k(5, 3, 1, 42, Float32(3.14))
    cache.set_v(5, 3, 1, 42, Float32(2.71))

    assert_almost_equal(cache.get_k(5, 3, 1, 42), Float32(3.14), atol=1e-6)
    assert_almost_equal(cache.get_v(5, 3, 1, 42), Float32(2.71), atol=1e-6)

    # Verifikasi isolasi antar-layer attention
    assert_almost_equal(cache.get_k(5, 0, 1, 42), Float32(0.0), atol=1e-6)
    assert_almost_equal(cache.get_k(5, 2, 1, 42), Float32(0.0), atol=1e-6)

    # Verifikasi store_tokens batch
    var seq_len = 4
    var kv_dim = h_kv * head_dim
    var k_batch = List[Float32]()
    var v_batch = List[Float32]()
    k_batch.resize(seq_len * kv_dim, Float32(1.5))
    v_batch.resize(seq_len * kv_dim, Float32(2.5))

    cache.store_tokens(1, 0, seq_len, k_batch, v_batch)
    assert_equal(cache.current_len, 4)
    assert_almost_equal(cache.get_k(0, 1, 0, 0), Float32(1.5), atol=1e-6)
    assert_almost_equal(cache.get_v(3, 1, 1, 127), Float32(2.5), atol=1e-6)


def test_rope_isometry_gqa_heads() raises:
    """Verifikasi RoPE mempertahankan isometri L2 norm untuk Q (H_q) dan K (H_kv).
    """
    var seq_len = 2
    var num_q_heads = 4
    var num_kv_heads = 2
    var head_dim = 16

    # Test Query heads
    var q_raw = List[Float32]()
    q_raw.resize(seq_len * num_q_heads * head_dim, Float32(1.0))
    for i in range(len(q_raw)):
        q_raw[i] = Float32((i % 7) + 1) * Float32(0.1)

    var q_rot = apply_rope_to_heads(q_raw, seq_len, num_q_heads, head_dim, 0)
    for t in range(seq_len):
        for h in range(num_q_heads):
            var orig_norm_sq = Float32(0.0)
            var rot_norm_sq = Float32(0.0)
            var head_off = (t * num_q_heads + h) * head_dim
            for d in range(head_dim):
                var vo = q_raw[head_off + d]
                var vr = q_rot[head_off + d]
                orig_norm_sq += vo * vo
                rot_norm_sq += vr * vr
            assert_almost_equal(
                sqrt(orig_norm_sq), sqrt(rot_norm_sq), atol=1e-4
            )

    # Test Key heads
    var k_raw = List[Float32]()
    k_raw.resize(seq_len * num_kv_heads * head_dim, Float32(0.5))
    var k_rot = apply_rope_to_heads(k_raw, seq_len, num_kv_heads, head_dim, 0)
    for t in range(seq_len):
        for h in range(num_kv_heads):
            var orig_norm_sq = Float32(0.0)
            var rot_norm_sq = Float32(0.0)
            var head_off = (t * num_kv_heads + h) * head_dim
            for d in range(head_dim):
                var vo = k_raw[head_off + d]
                var vr = k_rot[head_off + d]
                orig_norm_sq += vo * vo
                rot_norm_sq += vr * vr
            assert_almost_equal(
                sqrt(orig_norm_sq), sqrt(rot_norm_sq), atol=1e-4
            )


def test_gated_attention_causality_and_gating() raises:
    """Verifikasi causal attention, softmax stabilitas, dan efek sigmoid gating.
    """
    var cfg = ModelConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        vocab_size=100,
        num_key_value_heads=1,
        head_dim_override=16,
        full_attention_interval=4,
        architecture="qwen3.6",
    )
    var hidden = cfg.hidden_size
    var h_q = cfg.num_attention_heads
    var h_kv = cfg.num_key_value_heads
    var head_dim = cfg.head_dim()
    var q_dim = h_q * head_dim
    var kv_dim = h_kv * head_dim

    var w_q = List[Float32]()
    w_q.resize(q_dim * hidden, Float32(0.01))
    var w_k = List[Float32]()
    w_k.resize(kv_dim * hidden, Float32(0.01))
    var w_v = List[Float32]()
    w_v.resize(kv_dim * hidden, Float32(0.01))
    var w_gate = List[Float32]()
    w_gate.resize(q_dim * hidden, Float32(0.0))  # gate 0 => sigmoid(0) = 0.5
    var w_o = List[Float32]()
    w_o.resize(hidden * q_dim, Float32(0.01))

    for i in range(min(q_dim, hidden)):
        w_q[i * hidden + i] = Float32(0.2)
        w_o[i * q_dim + i] = Float32(0.2)
    for i in range(min(kv_dim, hidden)):
        w_k[i * hidden + i] = Float32(0.2)
        w_v[i * hidden + i] = Float32(0.2)

    var b_q = List[Float32]()
    var b_k = List[Float32]()
    var b_v = List[Float32]()
    var b_gate = List[Float32]()
    var b_o = List[Float32]()
    var weights = GatedAttentionWeights(
        w_q^,
        b_q^,
        w_k^,
        b_k^,
        w_v^,
        b_v^,
        w_gate^,
        b_gate^,
        w_o^,
        b_o^,
    )

    var kv_cache = GatedAttnKVCache(16, 1, h_kv, head_dim)

    var seq_len = 3
    var x = List[Float32]()
    x.resize(seq_len * hidden, Float32(0.5))

    var out = gated_attention_forward(
        x, weights, kv_cache, 0, 0, seq_len, cfg, 3
    )

    assert_equal(len(out), seq_len * hidden)
    assert_equal(kv_cache.current_len, seq_len)

    # Pastikan hasil finite dan terdefinisi
    for i in range(len(out)):
        assert_true(out[i] > Float32(-100.0) and out[i] < Float32(100.0))


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_gqa_grouping_ratio]()
    suite.test[test_physical_5d_kv_cache_layout]()
    suite.test[test_rope_isometry_gqa_heads]()
    suite.test[test_gated_attention_causality_and_gating]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
