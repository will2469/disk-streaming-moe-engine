# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk SwiGLU, shared expert sigmoid gate, dan MoE aggregation (M3-W2)."""

from core.config import LoadMemoryTelemetry, ModelConfig
from layers.moe import moe_aggregate_forward, shared_gate_forward
from layers.moe_loader import (
    load_layer_routed_expert_weights,
    load_layer_shared_expert_weights,
)
from layers.swiglu import SwigluWeights, sigmoid_f32, silu_f32, swiglu_forward
from std.collections import Dict, List
from std.math import abs
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
)


def test_sigmoid_stable_and_range() raises:
    """Uji kestabilan dan batasan nilai sigmoid fp32."""
    var s0 = sigmoid_f32(0.0)
    assert_true(abs(s0 - 0.5) < 1e-6)

    var s_neg_inf = sigmoid_f32(-100.0)
    assert_equal(s_neg_inf, 0.0)

    var s_pos_inf = sigmoid_f32(100.0)
    assert_equal(s_pos_inf, 1.0)

    var test_vals: List[Float32] = [-10.0, -5.0, -1.0, 0.5, 1.0, 5.0, 10.0]
    for i in range(len(test_vals)):
        var s = sigmoid_f32(test_vals[i])
        assert_true(s > 0.0 and s < 1.0)


def test_silu_known_values() raises:
    """Uji nilai terhitung SiLU(z) = z * sigma(z)."""
    assert_equal(silu_f32(0.0), 0.0)

    var s2 = silu_f32(2.0)
    assert_true(abs(s2 - 1.7615941) < 1e-5)

    var s_neg2 = silu_f32(-2.0)
    assert_true(abs(s_neg2 - (-0.23840584)) < 1e-5)


def test_swiglu_forward_known_values() raises:
    """Uji SwiGLU forward dengan matriks identitas sederhana."""
    var hidden = 2
    var inter = 2
    var x: List[Float32] = [1.0, 2.0]
    var w_gate: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var w_up: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var w_down: List[Float32] = [1.0, 0.0, 0.0, 1.0]

    var weights = SwigluWeights(w_gate^, w_up^, w_down^, hidden, inter)
    var out = swiglu_forward(x, weights, 1, 0, 0)
    assert_equal(len(out), 2)

    var exp0 = silu_f32(1.0) * 1.0
    var exp1 = silu_f32(2.0) * 2.0
    assert_true(abs(out[0] - exp0) < 1e-5)
    assert_true(abs(out[1] - exp1) < 1e-5)


def test_swiglu_shape_errors() raises:
    """Uji deteksi kesalahan dimensi pada SwiGLU."""
    var hidden = 2
    var inter = 2
    var w_gate: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var w_up: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var w_down: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var weights = SwigluWeights(w_gate^, w_up^, w_down^, hidden, inter)

    var raised_x = False
    try:
        var bad_x: List[Float32] = [1.0]
        var _out = swiglu_forward(bad_x, weights, 1, 0, 0)
    except e:
        raised_x = True
    assert_true(raised_x)


def test_swiglu_nan_inf() raises:
    """Uji deteksi nilai non-finite pada input SwiGLU."""
    var hidden = 2
    var inter = 2
    var w_gate: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var w_up: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var w_down: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var weights = SwigluWeights(w_gate^, w_up^, w_down^, hidden, inter)

    var raised_nan = False
    try:
        var bad_x: List[Float32] = [Float32(0.0) / Float32(0.0), 1.0]
        var _out = swiglu_forward(bad_x, weights, 1, 0, 0)
    except e:
        raised_nan = True
    assert_true(raised_nan)


def test_shared_gate_sigmoid_property() raises:
    """Uji properti shared expert gate sigmoid."""
    var hidden = 2
    var seq_len = 2
    var x: List[Float32] = [1.0, 2.0, -1.0, -2.0]
    var w_gate: List[Float32] = [0.5, 0.5]

    var scores = shared_gate_forward(x, w_gate, seq_len, hidden, 0, "sigmoid")
    assert_equal(len(scores), 2)
    var exp0 = sigmoid_f32(1.5)
    assert_true(abs(scores[0] - exp0) < 1e-5)
    var exp1 = sigmoid_f32(-1.5)
    assert_true(abs(scores[1] - exp1) < 1e-5)
    assert_true(scores[0] > 0.0 and scores[0] < 1.0)
    assert_true(scores[1] > 0.0 and scores[1] < 1.0)


def test_shared_gate_error_detection() raises:
    """Invariant Keras #2: gate_mode selain sigmoid WAJIB memicu GATE_ERROR."""
    var hidden = 2
    var seq_len = 1
    var x: List[Float32] = [1.0, 1.0]
    var w_gate: List[Float32] = [0.5, 0.5]

    var raised_linear = False
    try:
        var _s = shared_gate_forward(x, w_gate, seq_len, hidden, 0, "linear")
    except e:
        raised_linear = String(e).find("GATE_ERROR") != -1
    assert_true(raised_linear)

    var raised_softmax = False
    try:
        var _s2 = shared_gate_forward(x, w_gate, seq_len, hidden, 0, "softmax")
    except e:
        raised_softmax = String(e).find("GATE_ERROR") != -1
    assert_true(raised_softmax)


def test_moe_aggregate_weighted_sum_and_residual() raises:
    """Uji agregasi lengkap: y = sum(p_i * E_i(x)) + sigma(g_sh) * E_sh(x) + x.
    """
    var seq_len = 1
    var hidden = 2
    var x: List[Float32] = [1.0, 2.0]

    var r_tok0: List[Float32] = [10.0, 20.0, 30.0, 40.0]
    var routed_outputs = List[List[Float32]]()
    routed_outputs.append(r_tok0^)

    var p_tok0: List[Float32] = [0.3, 0.2]
    var router_probs = List[List[Float32]]()
    router_probs.append(p_tok0^)

    var shared_out: List[Float32] = [5.0, 6.0]
    var shared_gates: List[Float32] = [0.5]

    var y = moe_aggregate_forward(
        x,
        routed_outputs,
        router_probs,
        shared_out,
        shared_gates,
        seq_len,
        hidden,
        0,
    )
    assert_equal(len(y), 2)
    assert_true(abs(y[0] - 12.5) < 1e-5)
    assert_true(abs(y[1] - 19.0) < 1e-5)


def test_property_sigmoid_gate_monotonicity() raises:
    """Property test: fungsi sigmoid strictly monoton dan bounded dalam [0, 1].
    """
    var prev = Float32(0.0)
    for i in range(-50, 51):
        var z = Float32(i) * 0.5
        var s = sigmoid_f32(z)
        assert_true(s >= 0.0 and s <= 1.0)
        if i > -50:
            assert_true(s >= prev)
        prev = s


def test_load_layer_routed_expert_weights_validation() raises:
    """Validasi load_layer_routed_expert_weights index bounds dan tensor hilang.
    """
    var cfg = ModelConfig(2048, 24, 16, 151936)
    var empty_map = Dict[String, String]()
    var telem = LoadMemoryTelemetry()

    var raised_neg = False
    try:
        var _w = load_layer_routed_expert_weights(
            -1, 0, "", empty_map, cfg, 1408, telem
        )
    except e:
        raised_neg = True
    assert_true(raised_neg)

    var raised_hi = False
    try:
        var _w2 = load_layer_routed_expert_weights(
            24, 0, "", empty_map, cfg, 1408, telem
        )
    except e:
        raised_hi = True
    assert_true(raised_hi)

    var raised_miss = False
    try:
        var _w3 = load_layer_routed_expert_weights(
            0, 0, "", empty_map, cfg, 1408, telem
        )
    except e:
        raised_miss = True
    assert_true(raised_miss)


def test_load_layer_shared_expert_weights_validation() raises:
    """Validasi load_layer_shared_expert_weights index bounds dan tensor hilang.
    """
    var cfg = ModelConfig(2048, 24, 16, 151936)
    var empty_map = Dict[String, String]()
    var telem = LoadMemoryTelemetry()

    var raised_neg = False
    try:
        var _w = load_layer_shared_expert_weights(
            -1, "", empty_map, cfg, 5632, telem
        )
    except e:
        raised_neg = True
    assert_true(raised_neg)

    var raised_hi = False
    try:
        var _w2 = load_layer_shared_expert_weights(
            24, "", empty_map, cfg, 5632, telem
        )
    except e:
        raised_hi = True
    assert_true(raised_hi)

    var raised_miss = False
    try:
        var _w3 = load_layer_shared_expert_weights(
            0, "", empty_map, cfg, 5632, telem
        )
    except e:
        raised_miss = True
    assert_true(raised_miss)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_sigmoid_stable_and_range]()
    suite.test[test_silu_known_values]()
    suite.test[test_swiglu_forward_known_values]()
    suite.test[test_swiglu_shape_errors]()
    suite.test[test_swiglu_nan_inf]()
    suite.test[test_shared_gate_sigmoid_property]()
    suite.test[test_shared_gate_error_detection]()
    suite.test[test_moe_aggregate_weighted_sum_and_residual]()
    suite.test[test_property_sigmoid_gate_monotonicity]()
    suite.test[test_load_layer_routed_expert_weights_validation]()
    suite.test[test_load_layer_shared_expert_weights_validation]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
