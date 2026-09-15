# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk proyeksi QKV dan qkv_forward (M2-W1)."""

from core.config import ModelConfig
from layers.qkv import QKVWeights, qkv_forward, qkv_project
from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)


def _make_diag(n: Int, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for r in range(n):
        for c in range(n):
            out.append(scale if r == c else 0.0)
    return out^


def test_qkv_project_identity() raises:
    """Proyeksi dengan weight=identity, bias=0 -> output == input."""
    var hidden = 4
    var seq_len = 2
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var w = _make_diag(4, 1.0)
    var b: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    var out = qkv_project(x, w, b, seq_len, hidden)
    for i in range(seq_len * hidden):
        assert_almost_equal(out[i], x[i], atol=1e-6)


def test_qkv_project_with_bias() raises:
    """Proyeksi identity + bias konstan -> output = input + bias."""
    var hidden = 4
    var seq_len = 1
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var w = _make_diag(4, 1.0)
    var b: List[Float32] = [0.5, 0.5, 0.5, 0.5]
    var out = qkv_project(x, w, b, seq_len, hidden)
    assert_almost_equal(out[0], 1.5, atol=1e-6)
    assert_almost_equal(out[1], 2.5, atol=1e-6)
    assert_almost_equal(out[2], 3.5, atol=1e-6)
    assert_almost_equal(out[3], 4.5, atol=1e-6)


def test_qkv_project_matmul() raises:
    """Proyeksi dengan weight non-trivial, verifikasi manual."""
    var hidden = 2
    var seq_len = 1
    var x: List[Float32] = [1.0, 2.0]
    var w: List[Float32] = [3.0, 4.0, 5.0, 6.0]
    var b: List[Float32] = [0.1, 0.2]
    var out = qkv_project(x, w, b, seq_len, hidden)
    assert_almost_equal(out[0], 11.1, atol=1e-5)
    assert_almost_equal(out[1], 17.2, atol=1e-5)


def test_qkv_project_shape_errors() raises:
    """Uji deteksi kesalahan shape pada x, w, dan b -> ACT_LOAD_FAILED / WEIGHT_LOAD_FAILED.
    """
    var valid_x: List[Float32] = [1.0, 2.0]
    var short_x: List[Float32] = [1.0]
    var valid_w: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var short_w: List[Float32] = [1.0, 0.0]
    var valid_b: List[Float32] = [0.0, 0.0]
    var short_b: List[Float32] = [0.0]

    # x mismatch
    var raised_x = False
    try:
        var _o = qkv_project(short_x, valid_w, valid_b, 1, 2)
    except:
        raised_x = True
    assert_true(raised_x)

    # w mismatch
    var raised_w = False
    try:
        var _o = qkv_project(valid_x, short_w, valid_b, 1, 2)
    except:
        raised_w = True
    assert_true(raised_w)

    # b mismatch
    var raised_b = False
    try:
        var _o = qkv_project(valid_x, valid_w, short_b, 1, 2)
    except:
        raised_b = True
    assert_true(raised_b)


def test_qkv_project_nan_inf() raises:
    """Uji deteksi NaN/Inf pada input/weight/bias proyeksi QKV."""
    var nan = Float32(0.0) / Float32(0.0)
    var inf = Float32(1.0) / Float32(0.0)
    var x_nan: List[Float32] = [nan, 1.0]
    var w_inf: List[Float32] = [1.0, inf, 0.0, 1.0]
    var b_nan: List[Float32] = [0.0, nan]
    var valid: List[Float32] = [1.0, 1.0]
    var valid_w: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var valid_b: List[Float32] = [0.0, 0.0]

    var rx = False
    try:
        var _o = qkv_project(x_nan, valid_w, valid_b, 1, 2)
    except:
        rx = True
    assert_true(rx)

    var rw = False
    try:
        var _o = qkv_project(valid, w_inf, valid_b, 1, 2)
    except:
        rw = True
    assert_true(rw)

    var rb = False
    try:
        var _o = qkv_project(valid, valid_w, b_nan, 1, 2)
    except:
        rb = True
    assert_true(rb)


def test_qkv_forward_multi_token() raises:
    """qkv_forward multi-token seq_len=2, hidden=4, num_heads=2, head_dim=2."""
    var cfg = ModelConfig(4, 1, 2, 64)
    var x: List[Float32] = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]  # seq_len = 2, hidden = 4

    var w_q = _make_diag(4, 1.0)
    var b_q: List[Float32] = [0.1, 0.1, 0.1, 0.1]
    var w_k = _make_diag(4, 2.0)
    var b_k: List[Float32] = [0.2, 0.2, 0.2, 0.2]
    var w_v = _make_diag(4, 3.0)
    var b_v: List[Float32] = [0.3, 0.3, 0.3, 0.3]

    var weights = QKVWeights(w_q^, w_k^, w_v^, b_q^, b_k^, b_v^)
    var res = qkv_forward(x, weights, 2, cfg)
    ref q = res[0]
    ref k = res[1]
    ref v = res[2]

    assert_equal(len(q), 8)
    assert_equal(len(k), 8)
    assert_equal(len(v), 8)

    # Token 0: x = [1, 0, 0, 0]
    assert_almost_equal(q[0], 1.1, atol=1e-5)
    assert_almost_equal(q[1], 0.1, atol=1e-5)
    assert_almost_equal(k[0], 2.2, atol=1e-5)
    assert_almost_equal(k[1], 0.2, atol=1e-5)
    assert_almost_equal(v[0], 3.3, atol=1e-5)
    assert_almost_equal(v[1], 0.3, atol=1e-5)

    # Token 1: x = [0, 1, 0, 0]
    assert_almost_equal(q[4], 0.1, atol=1e-5)
    assert_almost_equal(q[5], 1.1, atol=1e-5)
    assert_almost_equal(k[4], 0.2, atol=1e-5)
    assert_almost_equal(k[5], 2.2, atol=1e-5)
    assert_almost_equal(v[4], 0.3, atol=1e-5)
    assert_almost_equal(v[5], 3.3, atol=1e-5)


def test_qkv_forward_head_dim_config_error() raises:
    """Config dengan num_attention_heads * head_dim != hidden_size -> CONFIG_ERROR.
    """
    var raised = False
    try:
        var cfg = ModelConfig(5, 1, 2, 64)  # 5 // 2 = 2, 2 * 2 = 4 != 5
        var x: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0]
        var w = List[Float32]()
        for _ in range(25):
            w.append(0.0)
        var b: List[Float32] = [0.0, 0.0, 0.0, 0.0, 0.0]
        var weights = QKVWeights(
            w.copy(), w.copy(), w.copy(), b.copy(), b.copy(), b.copy()
        )
        var _res = qkv_forward(x, weights, 1, cfg)
    except:
        raised = True
    assert_true(raised)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_qkv_project_identity]()
    suite.test[test_qkv_project_with_bias]()
    suite.test[test_qkv_project_matmul]()
    suite.test[test_qkv_project_shape_errors]()
    suite.test[test_qkv_project_nan_inf]()
    suite.test[test_qkv_forward_multi_token]()
    suite.test[test_qkv_forward_head_dim_config_error]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
