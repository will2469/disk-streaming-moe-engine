# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk ModelConfig, LoadMemoryTelemetry, dan RMSNorm."""

from cli.config_parser import _parse_strict_float
from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import load_tensor_f32_chunked
from format.reader import read_header
from layers.rmsnorm import rmsnorm
from std.collections import List
from std.math import isinf, isnan, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)


def test_model_config_head_dim() raises:
    var cfg = ModelConfig(2048, 24, 16, 151936)
    assert_equal(cfg.head_dim(), 128)


def test_rmsnorm_known_vector() raises:
    """Oracle sebaris fp32: x=[1,2,3,4], gamma=1, eps=1e-6."""
    var eps = Float32(1e-6)
    print("rms_norm_eps =", eps)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var g: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var y = rmsnorm(x, g, eps)
    assert_almost_equal(y[0], Float32(0.365148365), atol=1e-6)
    assert_almost_equal(y[1], Float32(0.730296731), atol=1e-6)
    assert_almost_equal(y[2], Float32(1.095445037), atol=1e-6)
    assert_almost_equal(y[3], Float32(1.460593462), atol=1e-6)


def test_rmsnorm_gamma_scales() raises:
    """Perkalian gamma: y = normalized ⊙ gamma."""
    var eps = Float32(1e-6)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var g: List[Float32] = [2.0, 0.5, 1.0, 3.0]
    var y = rmsnorm(x, g, eps)
    assert_almost_equal(y[0], Float32(0.730296731), atol=1e-6)
    assert_almost_equal(y[1], Float32(0.365148365), atol=1e-6)
    assert_almost_equal(y[2], Float32(1.095445037), atol=1e-6)
    assert_almost_equal(y[3], Float32(4.381780148), atol=1e-6)


def test_rmsnorm_scale_invariant() raises:
    """Invariant F6: rmsnorm(c*x) == rmsnorm(x) untuk c > 0, gamma = 1."""
    var eps = Float32(1e-6)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var xs: List[Float32] = [2.0, 4.0, 6.0, 8.0]
    var g: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var y1 = rmsnorm(x, g, eps)
    var y2 = rmsnorm(xs, g, eps)
    for i in range(len(y1)):
        assert_almost_equal(y1[i], y2[i], atol=1e-6)


def test_rmsnorm_gamma_one_normalized() raises:
    """Invariant F6: gamma = 1 → RMS(output) ≈ 1."""
    var eps = Float32(1e-6)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var g: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var y = rmsnorm(x, g, eps)
    var acc = Float32(0.0)
    for i in range(len(y)):
        acc += y[i] * y[i]
    var rms = sqrt(acc / Float32(len(y)))
    assert_almost_equal(rms, Float32(1.0), atol=1e-6)


def test_rmsnorm_eps_rejected() raises:
    """Kontrak eps: 0 / negatif / NaN / Inf → CONFIG_ERROR."""
    var x: List[Float32] = [1.0, 2.0]
    var g: List[Float32] = [1.0, 1.0]
    var inf_v = Float32(1e30) * Float32(1e30)
    assert_true(isinf(inf_v))
    var nan_v = inf_v - inf_v
    assert_true(isnan(nan_v))
    var bad: List[Float32] = [Float32(0.0), Float32(-1e-6), nan_v, inf_v]
    for i in range(len(bad)):
        var raised = False
        try:
            var _y = rmsnorm(x, g, bad[i])
        except e:
            raised = True
        assert_true(raised)


def test_strict_float_accepts() raises:
    """Grammar angka JSON valid → nilai tepat (atof aman pasca-validasi)."""
    var cases = List[String]()
    cases.append("0")
    cases.append("-0")
    cases.append("2")
    cases.append("3.25")
    cases.append("-0.5")
    cases.append("1e-06")
    cases.append("1E+10")
    cases.append("100.001")
    var want: List[Float32] = [
        0.0,
        0.0,
        2.0,
        3.25,
        -0.5,
        Float32(1e-06),
        Float32(1e10),
        Float32(100.001),
    ]
    for i in range(len(cases)):
        var v = _parse_strict_float(cases[i], "rms_norm_eps")
        assert_almost_equal(v, want[i], atol=1e-6)


def test_strict_float_rejects() raises:
    """Trailing garbage / NaN-Infinity / grammar non-JSON / overflow → raise."""
    var bad = List[String]()
    bad.append("1.0garbage")
    bad.append("nan")
    bad.append("NaN")
    bad.append("inf")
    bad.append("-Infinity")
    bad.append("INFINITY")
    bad.append("")
    bad.append("-")
    bad.append("1.")
    bad.append(".5")
    bad.append("01")
    bad.append("1e")
    bad.append("1e+")
    bad.append("+1.0")
    bad.append("0x10")
    bad.append("1_000")
    bad.append(" 1.0")
    bad.append("1.0 ")
    bad.append("1e999")
    bad.append("-1e999")
    for i in range(len(bad)):
        var raised = False
        try:
            var _v = _parse_strict_float(bad[i], "rms_norm_eps")
        except e:
            raised = True
        assert_true(raised)


def test_rmsnorm_shape_errors() raises:
    """Input kosong / panjang gamma mismatch → NORM_ERROR."""
    var x: List[Float32] = [1.0, 2.0]
    var g_short: List[Float32] = [1.0]
    var empty: List[Float32] = []
    var raised_mismatch = False
    try:
        var _y = rmsnorm(x, g_short, Float32(1e-6))
    except e:
        raised_mismatch = True
    assert_true(raised_mismatch)
    var raised_empty = False
    try:
        var _z = rmsnorm(empty, empty, Float32(1e-6))
    except e:
        raised_empty = True
    assert_true(raised_empty)


def test_load_tensor_chunked() raises:
    """Pemuatan tensor chunked dari fixture m0 & verifikasi telemetri memori."""
    var path = "fixtures/m0/fixture-00001-of-00003.safetensors"
    var st = read_header(path)
    var telem = LoadMemoryTelemetry()
    var found = False
    for i in range(len(st.entries)):
        ref e = st.entries[i]
        if e.name == "model.embed_tokens.weight":
            found = True
            var t = load_tensor_f32_chunked(st.shard, st.data_base, e, telem)
            assert_equal(len(t), 512 * 64)
            assert_almost_equal(t[0], Float32(1.0189883e35), rtol=1e-5)
            assert_true(telem.source_buffer_bytes <= 64 * 1024 * 1024)
            assert_true(telem.conversion_buffer_bytes <= 64 * 1024 * 1024)
            assert_equal(telem.resident_target_bytes, 512 * 64 * 4)
    assert_true(found)


def test_load_tensor_offset_gates() raises:
    """Meta korup ditolak SEBELUM reserve(): tanpa panic/alokasi liar."""
    var path = "fixtures/m0/fixture-00001-of-00003.safetensors"
    var st = read_header(path)
    var gi = 0
    for i in range(len(st.entries)):
        ref e = st.entries[i]
        if e.name == "model.embed_tokens.weight":
            gi = i
    var good = st.entries[gi].copy()
    var shared = LoadMemoryTelemetry()
    var m1 = good.copy()
    m1.begin = good.end
    m1.end = good.begin
    with assert_raises(contains="OFFSET_OVERFLOW"):
        _ = load_tensor_f32_chunked(path, st.data_base, m1, shared)
    var m2 = good.copy()
    m2.end = st.filesize
    with assert_raises(contains="OFFSET_OVERFLOW"):
        _ = load_tensor_f32_chunked(path, st.data_base, m2, shared)
    var m3 = good.copy()
    m3.end = m3.begin + 1
    with assert_raises(contains="LAYOUT_MISMATCH"):
        _ = load_tensor_f32_chunked(path, st.data_base, m3, shared)
    var m4 = good.copy()
    m4.dtype = "F16"
    with assert_raises(contains="UNKNOWN_DTYPE"):
        _ = load_tensor_f32_chunked(path, st.data_base, m4, shared)
    # Load gagal tak boleh mencemari telemetri bersama.
    assert_equal(shared.resident_target_bytes, 0)
    var t = load_tensor_f32_chunked(path, st.data_base, good.copy(), shared)
    assert_equal(len(t), 512 * 64)
    assert_equal(shared.resident_target_bytes, 512 * 64 * 4)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_model_config_head_dim]()
    suite.test[test_rmsnorm_known_vector]()
    suite.test[test_rmsnorm_gamma_scales]()
    suite.test[test_rmsnorm_scale_invariant]()
    suite.test[test_rmsnorm_gamma_one_normalized]()
    suite.test[test_rmsnorm_eps_rejected]()
    suite.test[test_strict_float_accepts]()
    suite.test[test_strict_float_rejects]()
    suite.test[test_rmsnorm_shape_errors]()
    suite.test[test_load_tensor_offset_gates]()
    suite.test[test_load_tensor_chunked]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
