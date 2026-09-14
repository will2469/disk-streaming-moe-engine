# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk MHA causal mask, softmax row stable, o_proj, dan forward_attention_block (M2-W3)."""

from core.config import ModelConfig
from layers.attention import (
    AttentionWeights,
    forward_attention_block,
    o_project,
)
from layers.mha import (
    build_causal_mask,
    mha_forward,
    softmax_row_stable,
)
from layers.qkv import QKVWeights
from layers.residual import add_residual
from std.collections import List
from std.math import abs, isinf
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)


def test_causal_mask_triangular_structure() raises:
    """Struktur matriks causal mask: 0 pada j <= i, -inf pada j > i."""
    var seq_len = 4
    var mask = build_causal_mask(seq_len)
    assert_equal(len(mask), 16)
    for i in range(seq_len):
        for j in range(seq_len):
            var val = mask[i * seq_len + j]
            if j <= i:
                assert_almost_equal(val, Float32(0.0), atol=1e-6)
            else:
                assert_true(isinf(val) and val < Float32(0.0))


def test_causal_mask_autoregressive_property() raises:
    """Property test causal mask: output token t hanya dari input <= t."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var seq_len = 3
    var hidden = 4

    # Sequence A: [t0, t1, t2]
    var q_a: List[Float32] = [
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    var k_a: List[Float32] = [
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    var v_a: List[Float32] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    ]

    # Sequence B: [t0, t1, t2_prime] (token 0 dan 1 sama, token 2 berbeda)
    var q_b: List[Float32] = [
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        -5.0,
        8.0,
        -3.0,
        2.0,
    ]
    var k_b: List[Float32] = [
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        99.0,
        -42.0,
        13.0,
        7.0,
    ]
    var v_b: List[Float32] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        -100.0,
        200.0,
        -300.0,
        400.0,
    ]

    var out_a = mha_forward(q_a, k_a, v_a, seq_len, cfg)
    var out_b = mha_forward(q_b, k_b, v_b, seq_len, cfg)

    # Output pada token 0 dan token 1 wajib IDENTIK
    for t in range(2):
        for d in range(hidden):
            assert_almost_equal(
                out_a[t * hidden + d], out_b[t * hidden + d], atol=1e-6
            )

    # Output pada token 2 wajib BERBEDA
    var diff_t2 = Float32(0.0)
    for d in range(hidden):
        diff_t2 += abs(out_a[2 * hidden + d] - out_b[2 * hidden + d])
    assert_true(diff_t2 > Float32(1.0))


def test_softmax_stable_max_shift() raises:
    """Property test softmax stabil: invariansi max-shift softmax(z + c) == softmax(z).
    """
    var z: List[Float32] = [1.0, 5.0, 2.0, 4.0]
    var z_shifted: List[Float32] = [1001.0, 1005.0, 1002.0, 1004.0]
    var p1 = softmax_row_stable(z, 0, 4, 4)
    var p2 = softmax_row_stable(z_shifted, 0, 4, 4)
    for j in range(4):
        assert_almost_equal(p1[j], p2[j], atol=1e-6)


def test_softmax_stable_sum_to_one() raises:
    """Probabilitas softmax wajib berjumlah tepat 1.0 pada elemen valid."""
    var z: List[Float32] = [-3.0, 2.0, 0.5, 99.0]
    var p = softmax_row_stable(z, 0, 4, 3)
    var sum_prob = Float32(0.0)
    for j in range(3):
        sum_prob += p[j]
    assert_almost_equal(sum_prob, Float32(1.0), atol=1e-6)
    assert_almost_equal(p[3], Float32(0.0), atol=1e-6)


def test_softmax_stable_nan_inf_rejected() raises:
    """Non-finite value di unmasked logits memicu ATTENTION_ERROR."""
    var nan_z: List[Float32] = [1.0, Float32(0.0) / Float32(0.0), 3.0]
    var r1 = False
    try:
        var _p = softmax_row_stable(nan_z, 0, 3, 3)
    except:
        r1 = True
    assert_true(r1)


def test_mha_forward_known_oracle() raises:
    """Verifikasi analitis MHA dengan bobot deterministik kecil vs PyTorch oracle.
    """
    var cfg = ModelConfig(4, 1, 1, 10)
    var seq_len = 2
    var q: List[Float32] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    var k: List[Float32] = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    var v: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var out = mha_forward(q, k, v, seq_len, cfg)
    assert_equal(len(out), 8)
    assert_almost_equal(out[0], Float32(1.0), atol=1e-5)
    assert_almost_equal(out[1], Float32(2.0), atol=1e-5)
    assert_almost_equal(out[2], Float32(3.0), atol=1e-5)
    assert_almost_equal(out[3], Float32(4.0), atol=1e-5)
    assert_almost_equal(out[4], Float32(3.0), atol=1e-5)
    assert_almost_equal(out[5], Float32(4.0), atol=1e-5)
    assert_almost_equal(out[6], Float32(5.0), atol=1e-5)
    assert_almost_equal(out[7], Float32(6.0), atol=1e-5)


def test_o_project_matmul_and_bias() raises:
    """Proyeksi output o_project: perkalian matriks + penambahan bias opsional.
    """
    var seq_len = 1
    var hidden = 2
    var x: List[Float32] = [2.0, 3.0]
    var w_o: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var b_o: List[Float32] = [0.5, -0.5]
    var empty_b = List[Float32]()

    # Tanpa bias
    var y1 = o_project(x, w_o, empty_b, seq_len, hidden)
    assert_almost_equal(y1[0], Float32(2.0), atol=1e-6)
    assert_almost_equal(y1[1], Float32(3.0), atol=1e-6)

    # Dengan bias
    var y2 = o_project(x, w_o, b_o, seq_len, hidden)
    assert_almost_equal(y2[0], Float32(2.5), atol=1e-6)
    assert_almost_equal(y2[1], Float32(2.5), atol=1e-6)


def test_add_residual() raises:
    """Residual connection y + x dan validasi dimensi."""
    var y: List[Float32] = [1.0, 2.0, 3.0]
    var x: List[Float32] = [0.5, 1.0, 1.5]
    var res = add_residual(y, x)
    assert_equal(len(res), 3)
    assert_almost_equal(res[0], Float32(1.5), atol=1e-6)
    assert_almost_equal(res[1], Float32(3.0), atol=1e-6)
    assert_almost_equal(res[2], Float32(4.5), atol=1e-6)


def test_forward_attention_block_oracle_slice() raises:
    """End-to-end forward attention block vs PyTorch fp32 oracle."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var seq_len = 2
    var eps = Float32(1e-6)

    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.5, -0.5]
    var gamma: List[Float32] = [1.0, 1.0, 1.0, 1.0]

    var eye4 = List[Float32]()
    eye4.resize(16, Float32(0.0))
    eye4[0] = Float32(1.0)
    eye4[5] = Float32(1.0)
    eye4[10] = Float32(1.0)
    eye4[15] = Float32(1.0)

    var zero_bias = List[Float32]()
    zero_bias.resize(4, Float32(0.0))

    var qkv = QKVWeights(
        eye4.copy(),
        eye4.copy(),
        eye4.copy(),
        zero_bias.copy(),
        zero_bias.copy(),
        zero_bias.copy(),
    )
    var empty_b = List[Float32]()
    var weights = AttentionWeights(gamma^, qkv^, eye4^, empty_b^)

    var y_final = forward_attention_block(
        x, weights, seq_len, cfg, eps, pos_offset=0
    )
    assert_equal(len(y_final), 8)

    assert_almost_equal(y_final[0], Float32(1.3651483), atol=1e-5)
    assert_almost_equal(y_final[1], Float32(2.7302966), atol=1e-5)
    assert_almost_equal(y_final[2], Float32(4.0954452), atol=1e-5)
    assert_almost_equal(y_final[3], Float32(5.4605932), atol=1e-5)
    assert_almost_equal(y_final[4], Float32(0.8598768), atol=1e-5)
    assert_almost_equal(y_final[5], Float32(-1.5558764), atol=1e-5)
    assert_almost_equal(y_final[6], Float32(4.2174449), atol=1e-5)
    assert_almost_equal(y_final[7], Float32(-0.6550304), atol=1e-5)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_causal_mask_triangular_structure]()
    suite.test[test_causal_mask_autoregressive_property]()
    suite.test[test_softmax_stable_max_shift]()
    suite.test[test_softmax_stable_sum_to_one]()
    suite.test[test_softmax_stable_nan_inf_rejected]()
    suite.test[test_mha_forward_known_oracle]()
    suite.test[test_o_project_matmul_and_bias]()
    suite.test[test_add_residual]()
    suite.test[test_forward_attention_block_oracle_slice]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
