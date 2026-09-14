# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk rotary position embeddings (RoPE) rotate_half dan isometri (M2-W2)."""

from core.config import ModelConfig
from layers.rope import apply_rope, rope_rotate_half, verify_rope_isometry
from std.collections import List
from std.math import abs, cos, sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)


def test_rope_rotate_half_position_zero() raises:
    """Invariant F7: pada posisi m=0, RoPE adalah identitas: rope(x, 0) == x."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var y = rope_rotate_half(x, 1, cfg, pos_offset=0)
    assert_equal(len(y), 4)
    for i in range(4):
        assert_almost_equal(y[i], x[i], atol=1e-6)


def test_rope_rotate_half_known_vector() raises:
    """Verifikasi analitis vektor d_h=4 pada posisi m=1."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var y = rope_rotate_half(x, 1, cfg, pos_offset=1)
    assert_almost_equal(y[0], Float32(-1.9841106), atol=1e-5)
    assert_almost_equal(y[1], Float32(1.9959991), atol=1e-5)
    assert_almost_equal(y[2], Float32(2.4623778), atol=1e-5)
    assert_almost_equal(y[3], Float32(4.001998), atol=1e-5)


def test_rope_rotate_half_oracle_slice() raises:
    """Oracle slice test: cross-check 2-token batch terhadap PyTorch fp32 formula.
    """
    var cfg = ModelConfig(4, 1, 1, 10)
    var seq_len = 2
    var x: List[Float32] = [
        1.0,
        2.0,
        3.0,
        4.0,
        0.5,
        -1.0,
        2.5,
        -0.5,
    ]
    var y = rope_rotate_half(x, seq_len, cfg, pos_offset=1)
    assert_equal(len(y), 8)
    # Token 0 (pos = 1)
    assert_almost_equal(y[0], Float32(-1.9841106), atol=1e-5)
    assert_almost_equal(y[1], Float32(1.9959991), atol=1e-5)
    assert_almost_equal(y[2], Float32(2.4623780), atol=1e-5)
    assert_almost_equal(y[3], Float32(4.0019979), atol=1e-5)
    # Token 1 (pos = 2)
    assert_almost_equal(y[4], Float32(-2.4813168), atol=1e-5)
    assert_almost_equal(y[5], Float32(-0.9989980), atol=1e-5)
    assert_almost_equal(y[6], Float32(-0.5857184), atol=1e-5)
    assert_almost_equal(y[7], Float32(-0.5019990), atol=1e-5)


def test_rope_rotate_half_isometry_properties() raises:
    """Property test P-3: isometri ||q'|| == ||q|| di berbagai posisi (0, 1, 15, 100, 2048)
    dan layer (0, 12, 23) dengan dimensi kepala aktual d_h=128.
    """
    var cfg = ModelConfig(256, 24, 2, 1000)
    var seq_len = 1
    var hidden = cfg.hidden_size

    var x = List[Float32]()
    x.reserve(hidden)
    for i in range(hidden):
        x.append(Float32((i % 29) - 14) * Float32(0.25))

    var positions: List[Int] = [0, 1, 15, 100, 2048]
    var layers: List[Int] = [0, 12, 23]

    for p in range(len(positions)):
        var pos = positions[p]
        for l in range(len(layers)):
            var layer = layers[l]
            var y = rope_rotate_half(
                x, seq_len, cfg, pos_offset=pos, layer_idx=layer
            )
            verify_rope_isometry(x, y, seq_len, cfg, layer_idx=layer)


def test_rope_rotate_half_linearity() raises:
    """Sifat linear: RoPE(c * q) == c * RoPE(q)."""
    var cfg = ModelConfig(128, 1, 1, 100)
    var seq_len = 1
    var c = Float32(3.5)

    var x1 = List[Float32]()
    var x2 = List[Float32]()
    for i in range(128):
        var val = Float32((i % 17) - 8) * Float32(0.1)
        x1.append(val)
        x2.append(val * c)

    var y1 = rope_rotate_half(x1, seq_len, cfg, pos_offset=7)
    var y2 = rope_rotate_half(x2, seq_len, cfg, pos_offset=7)

    for i in range(128):
        assert_almost_equal(y2[i], y1[i] * c, atol=1e-4)


def test_rope_rotate_half_vs_interleaved_distinction() raises:
    """Verifikasi isolasi gaya rotasi: rotate_half vs interleaved (FAIL category rope-style).

    Perbedaan kedua metode rotasi harus signifikan (> 0.5) untuk mencegah salah gaya.
    """
    var cfg = ModelConfig(4, 1, 1, 10)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var y_half = rope_rotate_half(x, 1, cfg, pos_offset=1)

    var th0 = Float32(1.0)
    var th1 = Float32(0.001)
    var y_inter0 = x[0] * cos(th0) - x[1] * sin(th0)
    var y_inter1 = x[0] * sin(th0) + x[1] * cos(th0)
    var y_inter2 = x[2] * cos(th1) - x[3] * sin(th1)
    var y_inter3 = x[2] * sin(th1) + x[3] * cos(th1)

    var max_diff = Float32(0.0)
    var diff0 = abs(y_half[0] - y_inter0)
    var diff1 = abs(y_half[1] - y_inter1)
    var diff2 = abs(y_half[2] - y_inter2)
    var diff3 = abs(y_half[3] - y_inter3)
    if diff0 > max_diff:
        max_diff = diff0
    if diff1 > max_diff:
        max_diff = diff1
    if diff2 > max_diff:
        max_diff = diff2
    if diff3 > max_diff:
        max_diff = diff3

    assert_true(max_diff > Float32(0.5))


def test_rope_rotate_half_shape_errors() raises:
    """Error handling dimensi: seq_len <= 0, length mismatch, odd head_dim, base <= 0 -> ROPE_ERROR.
    """
    var valid_cfg = ModelConfig(4, 1, 1, 10)
    var odd_cfg = ModelConfig(3, 1, 1, 10)  # head_dim = 3 (ganjil)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var short_x: List[Float32] = [1.0, 2.0, 3.0]

    # seq_len <= 0
    var r1 = False
    try:
        var _y = rope_rotate_half(x, 0, valid_cfg)
    except e:
        r1 = True
    assert_true(r1)

    # length mismatch
    var r2 = False
    try:
        var _y = rope_rotate_half(short_x, 1, valid_cfg)
    except e:
        r2 = True
    assert_true(r2)

    # odd head_dim
    var r3 = False
    try:
        var _y = rope_rotate_half(x, 1, odd_cfg)
    except e:
        r3 = True
    assert_true(r3)

    # base <= 0
    var r4 = False
    try:
        var _y = rope_rotate_half(x, 1, valid_cfg, base=Float32(-1.0))
    except e:
        r4 = True
    assert_true(r4)


def test_rope_rotate_half_nan_inf_detection() raises:
    """Non-finite value di input memicu ROPE_ERROR."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var nan_x: List[Float32] = [
        1.0,
        Float32(0.0) / Float32(0.0),
        3.0,
        4.0,
    ]
    var r1 = False
    try:
        var _y = rope_rotate_half(nan_x, 1, cfg)
    except e:
        r1 = True
    assert_true(r1)

    var inf_x: List[Float32] = [
        1.0,
        Float32(1.0) / Float32(0.0),
        3.0,
        4.0,
    ]
    var r2 = False
    try:
        var _y = rope_rotate_half(inf_x, 1, cfg)
    except e:
        r2 = True
    assert_true(r2)


def test_rope_isometry_violation_error() raises:
    """Pelanggaran invariant isometri memicu ROPE_ERROR eksplisit."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var orig: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var violated: List[Float32] = [1.5, 3.0, 4.5, 6.0]
    var raised = False
    try:
        verify_rope_isometry(orig, violated, 1, cfg, layer_idx=5)
    except e:
        raised = True
    assert_true(raised)


def test_apply_rope_q_and_k() raises:
    """Terapkan RoPE memutar Q dan K secara bersamaan."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var q: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var k: List[Float32] = [0.5, -1.0, 2.5, -0.5]
    var res = apply_rope(q, k, 1, cfg, pos_offset=1)
    ref q_rot = res[0]
    ref k_rot = res[1]
    assert_equal(len(q_rot), 4)
    assert_equal(len(k_rot), 4)
    assert_almost_equal(q_rot[0], Float32(-1.9841106), atol=1e-5)
    assert_almost_equal(k_rot[0], Float32(-1.8335261), atol=1e-5)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_rope_rotate_half_position_zero]()
    suite.test[test_rope_rotate_half_known_vector]()
    suite.test[test_rope_rotate_half_oracle_slice]()
    suite.test[test_rope_rotate_half_isometry_properties]()
    suite.test[test_rope_rotate_half_linearity]()
    suite.test[test_rope_rotate_half_vs_interleaved_distinction]()
    suite.test[test_rope_rotate_half_shape_errors]()
    suite.test[test_rope_rotate_half_nan_inf_detection]()
    suite.test[test_rope_isometry_violation_error]()
    suite.test[test_apply_rope_q_and_k]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
