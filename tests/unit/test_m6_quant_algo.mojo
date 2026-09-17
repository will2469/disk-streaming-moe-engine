# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M6-W2: Algoritma Kuantisasi F11a & Evaluasi Properti."""

from quant.quant_algo import (
    QuantMetrics,
    clip_q4,
    compute_quant_metrics,
    dequantize_tensor_bf16,
    dequantize_tensor_q32,
    quantize_group_f11a,
    quantize_tensor_f11a,
    rne_fp32,
    verify_kerneldomain_property,
    verify_qdomain_property,
)
from std.collections import List
from std.math import cos, sin
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def test_rne_tie_breaking_golden() raises:
    """Verifikasi golden tie-breaking Round-Half-to-Even (RNE) fp32.

    Menjamin pecahan tepat 0.5 dibulatkan ke genap terdekat,
    dan pembatasan clip ke rentang [-7, 7] tidak memancarkan -8.
    """
    assert_equal(rne_fp32(0.0), 0)
    assert_equal(rne_fp32(0.4), 0)
    assert_equal(rne_fp32(0.5), 0)  # 0 genap
    assert_equal(rne_fp32(0.6), 1)
    assert_equal(rne_fp32(1.0), 1)
    assert_equal(rne_fp32(1.4), 1)
    assert_equal(rne_fp32(1.5), 2)  # 2 genap
    assert_equal(rne_fp32(1.6), 2)
    assert_equal(rne_fp32(2.5), 2)  # 2 genap
    assert_equal(rne_fp32(3.5), 4)  # 4 genap
    assert_equal(rne_fp32(4.5), 4)  # 4 genap
    assert_equal(rne_fp32(5.5), 6)  # 6 genap
    assert_equal(rne_fp32(6.5), 6)  # 6 genap

    # Nilai negatif
    assert_equal(rne_fp32(-0.4), 0)
    assert_equal(rne_fp32(-0.5), 0)  # 0 genap
    assert_equal(rne_fp32(-0.6), -1)
    assert_equal(rne_fp32(-1.4), -1)
    assert_equal(rne_fp32(-1.5), -2)  # -2 genap
    assert_equal(rne_fp32(-1.6), -2)
    assert_equal(rne_fp32(-2.5), -2)  # -2 genap
    assert_equal(rne_fp32(-3.5), -4)  # -4 genap
    assert_equal(rne_fp32(-4.5), -4)  # -4 genap
    assert_equal(rne_fp32(-5.5), -6)  # -6 genap
    assert_equal(rne_fp32(-6.5), -6)  # -6 genap

    # Clipping [-7, 7]
    assert_equal(clip_q4(rne_fp32(7.5)), 7)
    assert_equal(clip_q4(rne_fp32(-7.5)), -7)
    assert_equal(clip_q4(rne_fp32(10.0)), 7)
    assert_equal(clip_q4(rne_fp32(-10.0)), -7)


def test_tail_group_rejection() raises:
    """Verifikasi penolakan kontrak tail Opsi A (N % G != 0) dan group_size salah.
    """
    var bad_weights = List[Float32]()
    bad_weights.resize(100, Float32(1.0))

    var threw_tail = False
    try:
        var q = quantize_tensor_f11a(bad_weights, 128)
    except:
        threw_tail = True
    assert_true(threw_tail)

    var ok_weights = List[Float32]()
    ok_weights.resize(128, Float32(1.0))

    var threw_grp = False
    try:
        var q_bad_grp = quantize_tensor_f11a(ok_weights, 512)
    except:
        threw_grp = True
    assert_true(threw_grp)


def test_nan_inf_scale_rejection() raises:
    """Verifikasi penolakan tensor input yang mengandung NaN atau Inf."""
    var nan_weights = List[Float32]()
    nan_weights.resize(128, Float32(1.0))
    nan_weights[10] = Float32(0.0) / Float32(0.0)  # NaN

    var threw_nan = False
    try:
        var q_nan = quantize_tensor_f11a(nan_weights, 128)
    except:
        threw_nan = True
    assert_true(threw_nan)

    var inf_weights = List[Float32]()
    inf_weights.resize(128, Float32(1.0))
    inf_weights[20] = Float32(1.0) / Float32(0.0)  # Inf

    var threw_inf = False
    try:
        var q_inf = quantize_tensor_f11a(inf_weights, 128)
    except:
        threw_inf = True
    assert_true(threw_inf)


def test_zero_group_quantization() raises:
    """Verifikasi kuantisasi grup nol: s_g = 1.0, q = 0, properti lolos."""
    var zero_weights = List[Float32]()
    zero_weights.resize(128, Float32(0.0))

    var q_res = quantize_tensor_f11a(zero_weights, 128)
    assert_equal(q_res.num_groups, 1)
    assert_equal(q_res.scales[0], Float16(1.0))

    for i in range(128):
        assert_equal(q_res.q_weights[i], Int8(0))

    var dequant_32 = dequantize_tensor_q32(q_res.scales, q_res.q_weights, 128)
    var prop_q = verify_qdomain_property(
        zero_weights, dequant_32, q_res.scales, 128
    )
    assert_true(prop_q[0])
    assert_almost_equal(prop_q[1], Float32(0.0), atol=1e-6)


def test_constant_tensor_zero_variance() raises:
    """Verifikasi tensor konstan: var(w) = 0 -> jalur absolut lolos tanpa fudge.
    """
    var const_weights = List[Float32]()
    const_weights.resize(128, Float32(2.5))

    var q_res = quantize_tensor_f11a(const_weights, 128)
    var dequant_32 = dequantize_tensor_q32(q_res.scales, q_res.q_weights, 128)

    var prop_q = verify_qdomain_property(
        const_weights, dequant_32, q_res.scales, 128
    )
    assert_true(prop_q[0])

    var metrics = compute_quant_metrics(const_weights, dequant_32)
    assert_true(metrics.zero_variance)
    assert_equal(metrics.variance, Float32(0.0))
    # Epsilon rel bernilai -1.0 (null) pada variansi nol
    assert_equal(metrics.epsilon_rel, Float32(-1.0))


def test_dual_domain_properties_100_percent() raises:
    """Verifikasi bahwa 100% elemen memenuhi bound properti dua domain.

    Q-domain: |w - w_hat^(32)| <= s_g / 2
    Kernel-domain: |w - w_hat^(bf16)| <= s_g / 2 + |w_hat^(32)| / 256
    """
    var n_elem = 256
    var weights = List[Float32]()
    weights.resize(n_elem, Float32(0.0))

    # Pola nilai acak deterministik di [-3.5, 3.5]
    for i in range(n_elem):
        var v = sin(Float32(i) * 0.15) * 3.5
        weights[i] = v

    var q_res = quantize_tensor_f11a(weights, 128)
    assert_equal(q_res.num_groups, 2)

    var dequant_32 = dequantize_tensor_q32(q_res.scales, q_res.q_weights, 128)
    var dequant_bf16 = dequantize_tensor_bf16(
        q_res.scales, q_res.q_weights, 128
    )

    # 1. Verifikasi properti Q-domain
    var prop_q = verify_qdomain_property(weights, dequant_32, q_res.scales, 128)
    assert_true(prop_q[0])

    # 2. Verifikasi properti Kernel-domain
    var prop_kernel = verify_kerneldomain_property(
        weights, dequant_bf16, dequant_32, q_res.scales, 128
    )
    assert_true(prop_kernel[0])


def test_relative_error_threshold() raises:
    """Verifikasi roundtrip BF16 -> 4-bit -> BF16 memenuhi epsilon_rel <= 10^-2.
    """
    var n_elem = 1024
    var weights = List[Float32]()
    weights.resize(n_elem, Float32(0.0))

    var q_pattern: List[Float32] = [
        -7.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0
    ]
    var p_len = len(q_pattern)

    for i in range(n_elem):
        var q_val = q_pattern[i % p_len]
        var noise = Float32(0.01) * sin(Float32(i))
        weights[i] = q_val * Float32(0.25) + noise

    var q_res = quantize_tensor_f11a(weights, 128)
    var dequant_bf16 = dequantize_tensor_bf16(
        q_res.scales, q_res.q_weights, 128
    )

    var dequant_f32 = List[Float32]()
    dequant_f32.resize(n_elem, Float32(0.0))
    for i in range(n_elem):
        dequant_f32[i] = Float32(dequant_bf16[i])

    var metrics = compute_quant_metrics(weights, dequant_f32)
    assert_false(metrics.zero_variance)
    assert_true(metrics.epsilon_rel >= Float32(0.0))
    # Threshold F11: epsilon_rel <= 0.01 (10^-2)
    assert_true(metrics.epsilon_rel <= Float32(0.01))


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_rne_tie_breaking_golden]()
    suite.test[test_tail_group_rejection]()
    suite.test[test_nan_inf_scale_rejection]()
    suite.test[test_zero_group_quantization]()
    suite.test[test_constant_tensor_zero_variance]()
    suite.test[test_dual_domain_properties_100_percent]()
    suite.test[test_relative_error_threshold]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
