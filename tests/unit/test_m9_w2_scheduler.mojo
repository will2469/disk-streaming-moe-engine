# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M9-W2: Hybrid Macro Block Scheduler, Layer Topology, dan GDN State Isolation."""

from core.config import ModelConfig
from layers.gdn import GDNState
from layers.gated_attention import GatedAttnKVCache
from layers.port_scheduler import (
    PortBlockWeights,
    create_synthetic_block_weights,
    forward_port_macro_scheduler,
)
from std.collections import List
from std.math import isfinite
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def test_layer_topology_and_indexing() raises:
    """Verifikasi topologi 40 block hybrid: 30 GDN + 10 GatedAttn, serta mapping indeks unik.
    """
    var cfg = ModelConfig(
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        vocab_size=248320,
        num_key_value_heads=2,
        head_dim_override=128,
        full_attention_interval=4,
        architecture="qwen3.6",
    )

    var gdn_count = 0
    var attn_count = 0
    var seen_gdn_indices = List[Int]()
    seen_gdn_indices.resize(30, -1)
    var seen_attn_indices = List[Int]()
    seen_attn_indices.resize(10, -1)

    for l in range(40):
        if cfg.is_linear_attn_layer(l):
            gdn_count += 1
            var gdn_idx = 3 * (l // 4) + (l % 4)
            assert_true(gdn_idx >= 0 and gdn_idx < 30)
            assert_equal(seen_gdn_indices[gdn_idx], -1)  # Tidak boleh duplikat
            seen_gdn_indices[gdn_idx] = l
        else:
            attn_count += 1
            var att_idx = l // 4
            assert_true(att_idx >= 0 and att_idx < 10)
            assert_equal(seen_attn_indices[att_idx], -1)  # Tidak boleh duplikat
            seen_attn_indices[att_idx] = l

    assert_equal(gdn_count, 30)
    assert_equal(attn_count, 10)

    # Verifikasi seluruh indeks terisi penuh tanpa celah
    for i in range(30):
        assert_true(seen_gdn_indices[i] >= 0)
    for i in range(10):
        assert_true(seen_attn_indices[i] >= 0)


def test_gdn_state_isolation() raises:
    """Verifikasi 30 state GDN S[0..29] independen dan terisolasi ketat (bukan shared state).
    """
    var num_layers = 30
    var dv = 32
    var dk = 32
    var states = GDNState(num_layers, dv, dk)

    # Pastikan inisialisasi awal adalah 0.0
    for l in range(num_layers):
        for r in range(dv):
            for c in range(dk):
                assert_equal(states.get(l, r, c), Float32(0.0))

    # Mutasi state pada layer 0 saja
    states.set(0, 5, 10, Float32(3.1415))

    # Mutasi state pada layer 17 saja
    states.set(17, 2, 4, Float32(2.7182))

    # Verifikasi layer 0 memiliki mutasi
    assert_almost_equal(states.get(0, 5, 10), Float32(3.1415), atol=1e-5)
    # Verifikasi layer 17 memiliki mutasi
    assert_almost_equal(states.get(17, 2, 4), Float32(2.7182), atol=1e-5)

    # Verifikasi layer lainnya sama sekali tidak terpengaruh
    for l in range(num_layers):
        if l == 0 or l == 17:
            continue
        for r in range(dv):
            for c in range(dk):
                assert_equal(states.get(l, r, c), Float32(0.0))


def test_macro_scheduler_execution() raises:
    """Verifikasi eksekusi forward macro scheduler pada topologi mini 4-block.
    """
    var hidden = 64
    var num_layers = 4  # 3 GDN + 1 GatedAttn
    var num_exp = 4
    var top_k = 2
    var seq_len = 4
    var dk = 16
    var dv = 16
    var head_dim = 16
    var num_q_heads = 4
    var num_kv_heads = 1
    var num_gdn = 3
    var num_attn = 1

    var cfg = ModelConfig(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=num_q_heads,
        vocab_size=512,
        num_key_value_heads=num_kv_heads,
        head_dim_override=head_dim,
        num_experts=num_exp,
        num_experts_per_tok=top_k,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        full_attention_interval=4,
        norm_topk_prob=False,
        attention_bias=False,
        architecture="qwen3.6",
    )

    var blocks = List[PortBlockWeights]()
    for l in range(num_layers):
        blocks.append(create_synthetic_block_weights(cfg, l, dv, dk))

    var gdn_states = GDNState(num_gdn, dv, dk)
    var kv_cache = GatedAttnKVCache(16, num_attn, num_kv_heads, head_dim)

    var input_x = List[Float32]()
    input_x.resize(seq_len * hidden, Float32(0.0))
    for i in range(seq_len * hidden):
        input_x[i] = Float32(0.05 * Float32((i % 7) + 1))

    var out = forward_port_macro_scheduler(
        input_x,
        blocks,
        gdn_states,
        kv_cache,
        pos_offset=0,
        seq_len=seq_len,
        cfg=cfg,
        dk=dk,
        dv=dv,
    )

    assert_equal(len(out), seq_len * hidden)
    # Verifikasi seluruh nilai berhingga dan tidak NaN
    for i in range(len(out)):
        assert_true(isfinite(out[i]))
        assert_false(out[i] == Float32(0.0))

    # Verifikasi KV cache bertambah seq_len (karena ada 1 layer attention)
    assert_equal(kv_cache.current_len, seq_len)


def test_macro_scheduler_determinism() raises:
    """Verifikasi determinisme hasil forward macro scheduler."""
    var hidden = 32
    var num_layers = 4
    var num_exp = 2
    var top_k = 1
    var seq_len = 2
    var dk = 8
    var dv = 8
    var head_dim = 8
    var num_q_heads = 4
    var num_kv_heads = 1
    var num_gdn = 3
    var num_attn = 1

    var cfg = ModelConfig(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=num_q_heads,
        vocab_size=256,
        num_key_value_heads=num_kv_heads,
        head_dim_override=head_dim,
        num_experts=num_exp,
        num_experts_per_tok=top_k,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        full_attention_interval=4,
        norm_topk_prob=False,
        attention_bias=False,
        architecture="qwen3.6",
    )

    var blocks1 = List[PortBlockWeights]()
    var blocks2 = List[PortBlockWeights]()
    for l in range(num_layers):
        blocks1.append(create_synthetic_block_weights(cfg, l, dv, dk))
        blocks2.append(create_synthetic_block_weights(cfg, l, dv, dk))

    var gdn1 = GDNState(num_gdn, dv, dk)
    var kv1 = GatedAttnKVCache(16, num_attn, num_kv_heads, head_dim)

    var gdn2 = GDNState(num_gdn, dv, dk)
    var kv2 = GatedAttnKVCache(16, num_attn, num_kv_heads, head_dim)

    var x1 = List[Float32]()
    x1.resize(seq_len * hidden, Float32(0.123))
    var x2 = List[Float32]()
    x2.resize(seq_len * hidden, Float32(0.123))

    var out1 = forward_port_macro_scheduler(
        x1, blocks1, gdn1, kv1, 0, seq_len, cfg, dk, dv
    )
    var out2 = forward_port_macro_scheduler(
        x2, blocks2, gdn2, kv2, 0, seq_len, cfg, dk, dv
    )

    assert_equal(len(out1), len(out2))
    for i in range(len(out1)):
        assert_equal(out1[i], out2[i])


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_layer_topology_and_indexing]()
    suite.test[test_gdn_state_isolation]()
    suite.test[test_macro_scheduler_execution]()
    suite.test[test_macro_scheduler_determinism]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
