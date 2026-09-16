# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M5-W1: KV State Layout, Bound Validation, Memory Budget, dan Lifecycle."""

from layers.kv_cache import (
    BYTES_PER_SLOT_PER_LAYER,
    DEFAULT_MAX_POS,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_LAYERS,
    SLOT_DIM,
    FullKVCache,
    LayerKVCache,
    MemoryBudget,
    compute_kv_cache_bytes,
    validate_context_bounds,
)
from std.collections import List
from std.math import abs
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def test_locked_constants_and_slot_size() raises:
    """Verifikasi konstanta formal terkunci: 8 KiB/slot/layer, F2 total size."""
    assert_equal(NUM_KV_HEADS, 16)
    assert_equal(HEAD_DIM, 128)
    assert_equal(NUM_LAYERS, 24)
    assert_equal(SLOT_DIM, 2048)

    # 1 slot per layer = 2 (K+V) * 16 * 128 * 2 bytes = 8192 bytes (8 KiB)
    assert_equal(BYTES_PER_SLOT_PER_LAYER, 8192)

    # F2: total KV cache bytes @2048 = 24 * 8192 * 2048 = 402,653,184 B (384 MiB)
    var bytes_2k = compute_kv_cache_bytes(2048)
    assert_equal(bytes_2k, 402653184)

    # F2: total KV cache bytes @4096 = 24 * 8192 * 4096 = 805,306,368 B (768 MiB = 0.75 GiB)
    var bytes_4k = compute_kv_cache_bytes(4096)
    assert_equal(bytes_4k, 805306368)


def test_context_bounds_validation_chain() raises:
    """Verifikasi rantai bound: S + N <= ctx <= s_max (exit 2 bila dilanggar).
    """
    # 1. Kasus valid
    validate_context_bounds(16, 64, 2048, DEFAULT_MAX_POS)
    validate_context_bounds(
        1984, 64, 2048, DEFAULT_MAX_POS
    )  # S + N = 2048 == ctx

    # 2. Pelanggaran sisi kiri: S + N > ctx
    var left_failed = False
    try:
        validate_context_bounds(2000, 64, 2048, DEFAULT_MAX_POS)
    except e:
        left_failed = True
        var err_msg = String(e)
        assert_true("M5_ERR_CONTEXT_SIZE" in err_msg)
        assert_true("exceeds allocated context_size" in err_msg)
    assert_true(left_failed)

    # 3. Pelanggaran sisi kanan: ctx > s_max
    var right_failed = False
    try:
        validate_context_bounds(16, 64, 32769, 32768)
    except e:
        right_failed = True
        var err_msg = String(e)
        assert_true("M5_ERR_CONTEXT_SIZE" in err_msg)
        assert_true("exceeds maximum position embeddings" in err_msg)
    assert_true(right_failed)

    # 4. Input tidak valid: S <= 0
    var prompt_zero_failed = False
    try:
        validate_context_bounds(0, 64, 2048, DEFAULT_MAX_POS)
    except e:
        prompt_zero_failed = True
        var err_msg = String(e)
        assert_true("M5_ERR_INPUT" in err_msg)
    assert_true(prompt_zero_failed)

    # 5. Input tidak valid: N <= 0
    var tokens_zero_failed = False
    try:
        validate_context_bounds(16, 0, 2048, DEFAULT_MAX_POS)
    except e:
        tokens_zero_failed = True
        var err_msg = String(e)
        assert_true("M5_ERR_INPUT" in err_msg)
    assert_true(tokens_zero_failed)


def test_memory_budget_arithmetic() raises:
    """Verifikasi dekomposisi M_tensor dan bound proses M_peak_bound <= 5 GiB.
    """
    var b_2k = MemoryBudget(2048)
    assert_equal(b_2k.m_kv_cache, 402653184)
    # M_tensor @2K ≈ 3.82 GiB (4,102,036,224 B)
    assert_true(b_2k.m_tensor > 4100000000 and b_2k.m_tensor < 4110000000)
    assert_true(b_2k.is_within_gate())

    var b_4k = MemoryBudget(4096)
    assert_equal(b_4k.m_kv_cache, 805306368)
    # M_tensor @4K ≈ 4.20 GiB (4,504,820,480 B)
    assert_true(b_4k.m_tensor > 4500000000 and b_4k.m_tensor < 4510000000)
    # M_peak_bound @4K = 4.20 + 0.15 + 0.15 = 4.50 GiB <= 5.0 GiB
    assert_true(
        b_4k.m_peak_bound > 4810000000 and b_4k.m_peak_bound < 4825000000
    )
    assert_true(b_4k.is_within_gate())

    # Cek deteksi lubang akuntansi: observed > bound
    assert_true(b_4k.check_accounting(4819000000))
    assert_false(b_4k.check_accounting(4850000000))  # Lubang akuntansi!


def test_layer_kv_cache_lifecycle() raises:
    """Verifikasi siklus hidup LayerKVCache: prefill [0, S) -> decode append di p -> slice -> clear.
    """
    var capacity = 32
    var layer = LayerKVCache(capacity, 0)
    assert_equal(layer.current_len, 0)
    assert_equal(len(layer.k), capacity * SLOT_DIM)
    assert_equal(len(layer.v), capacity * SLOT_DIM)

    # 1. Prefill 4 token
    var s_prefill = 4
    var k_prompt = List[Float32]()
    var v_prompt = List[Float32]()
    for i in range(s_prefill * SLOT_DIM):
        k_prompt.append(Float32(1.0 + Float32(i) * 0.001))
        v_prompt.append(Float32(2.0 + Float32(i) * 0.001))

    layer.store_prefill(k_prompt, v_prompt, s_prefill)
    assert_equal(layer.current_len, 4)

    # 2. Decode step: append token di posisi p = 4
    var k_tok_0 = List[Float32]()
    var v_tok_0 = List[Float32]()
    for i in range(SLOT_DIM):
        k_tok_0.append(Float32(10.0 + Float32(i) * 0.01))
        v_tok_0.append(Float32(20.0 + Float32(i) * 0.01))

    layer.append_decode_token(k_tok_0, v_tok_0, 4)
    assert_equal(layer.current_len, 5)

    # 3. Posisi salah (p != current_len) wajib error
    var mismatch_failed = False
    try:
        layer.append_decode_token(k_tok_0, v_tok_0, 6)  # expected 5
    except e:
        mismatch_failed = True
        var err_msg = String(e)
        assert_true("Position mismatch" in err_msg)
    assert_true(mismatch_failed)

    # 4. Ambil slice K dan V rentang [0, 5)
    var k_slice = layer.get_k_slice(5)
    var v_slice = layer.get_v_slice(5)
    assert_equal(len(k_slice), 5 * SLOT_DIM)
    assert_equal(len(v_slice), 5 * SLOT_DIM)

    # Verifikasi presisi BF16 roundtrip token pertama prompt
    assert_almost_equal(k_slice[0], Float32(1.0), atol=1e-2)
    assert_almost_equal(v_slice[0], Float32(2.0), atol=1e-2)

    # Verifikasi token decode di posisi 4
    var offset_tok4 = 4 * SLOT_DIM
    assert_almost_equal(k_slice[offset_tok4], Float32(10.0), atol=1e-2)
    assert_almost_equal(v_slice[offset_tok4], Float32(20.0), atol=1e-2)

    # 5. Clear buffer
    layer.clear()
    assert_equal(layer.current_len, 0)
    assert_equal(len(layer.k), 0)
    assert_equal(len(layer.v), 0)


def test_full_kv_cache_24_layers() raises:
    """Verifikasi FullKVCache 24 layer: alokasi statis, prefill, append, dan cleanup.
    """
    var capacity = 16
    var full = FullKVCache(capacity)
    assert_equal(len(full.layers), 24)
    assert_equal(full.current_len(), 0)

    # Buat dummy prefill 2 token
    var s_prefill = 2
    var k_prompt = List[Float32]()
    var v_prompt = List[Float32]()
    for _ in range(s_prefill * SLOT_DIM):
        k_prompt.append(Float32(0.5))
        v_prompt.append(Float32(0.75))

    # Store prefill di seluruh 24 layer
    for l in range(24):
        full.store_prefill_layer(l, k_prompt, v_prompt, s_prefill)

    assert_equal(full.current_len(), 2)

    # Append decode token di p = 2
    var k_tok = List[Float32]()
    var v_tok = List[Float32]()
    for _ in range(SLOT_DIM):
        k_tok.append(Float32(1.5))
        v_tok.append(Float32(2.5))

    for l in range(24):
        full.append_decode_token_layer(l, k_tok, v_tok, 2)

    assert_equal(full.current_len(), 3)

    # Verifikasi slice layer 12
    var k_l12 = full.get_layer_k_slice(12, 3)
    assert_equal(len(k_l12), 3 * SLOT_DIM)
    assert_almost_equal(k_l12[2 * SLOT_DIM], Float32(1.5), atol=1e-2)

    # Cleanup
    full.clear()
    assert_equal(len(full.layers), 0)


def test_kv_oracle_slice_precision() raises:
    """Verifikasi presisi numerik BF16 oracle slice: relative error <= 2^-7 (eps BF16).
    """
    var capacity = 4
    var layer = LayerKVCache(capacity, 0)

    # Test values melintasi dinamika representasi BF16
    var oracle_k = List[Float32]()
    var oracle_v = List[Float32]()
    for i in range(2 * SLOT_DIM):
        var base_val = Float32(1.0) + Float32(i % 100) * Float32(0.125)
        if i % 3 == 0:
            base_val = -base_val
        oracle_k.append(base_val)
        oracle_v.append(base_val * Float32(0.5))

    layer.store_prefill(oracle_k, oracle_v, 2)
    var k_retrieved = layer.get_k_slice(2)
    var v_retrieved = layer.get_v_slice(2)

    assert_equal(len(k_retrieved), 2 * SLOT_DIM)
    assert_equal(len(v_retrieved), 2 * SLOT_DIM)

    # Toleransi BF16 rel_tol = 1/128 ~ 0.0078125
    var max_rel_err_k = Float32(0.0)
    var max_rel_err_v = Float32(0.0)
    for i in range(2 * SLOT_DIM):
        var orig_k = oracle_k[i]
        var got_k = k_retrieved[i]
        var diff_k = abs(orig_k - got_k)
        var rel_k = diff_k / abs(orig_k)
        if rel_k > max_rel_err_k:
            max_rel_err_k = rel_k

        var orig_v = oracle_v[i]
        var got_v = v_retrieved[i]
        var diff_v = abs(orig_v - got_v)
        var rel_v = diff_v / abs(orig_v)
        if rel_v > max_rel_err_v:
            max_rel_err_v = rel_v

    # Rel error harus <= 0.008 (BF16 7-bit mantissa)
    assert_true(max_rel_err_k <= Float32(0.008))
    assert_true(max_rel_err_v <= Float32(0.008))

    layer.clear()


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_locked_constants_and_slot_size]()
    suite.test[test_context_bounds_validation_chain]()
    suite.test[test_memory_budget_arithmetic]()
    suite.test[test_layer_kv_cache_lifecycle]()
    suite.test[test_full_kv_cache_24_layers]()
    suite.test[test_kv_oracle_slice_precision]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
