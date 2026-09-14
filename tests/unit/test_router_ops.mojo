# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk RouterConfig, RoutingInfo, router_project, dan router_softmax (M3-W1)."""

from core.config import _contains
from layers.router import router_project, router_softmax
from layers.router_types import RouterConfig, RoutingInfo
from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)


def test_router_config_validation() raises:
    """Validasi parameter RouterConfig."""
    var valid_cfg = RouterConfig(60, 4, False)
    valid_cfg.validate()

    var zero_experts = RouterConfig(0, 4, False)
    var raised_zero = False
    try:
        zero_experts.validate()
    except e:
        raised_zero = True
    assert_true(raised_zero)

    var invalid_topk = RouterConfig(60, 0, False)
    var raised_topk = False
    try:
        invalid_topk.validate()
    except e:
        raised_topk = True
    assert_true(raised_topk)

    var exceed_topk = RouterConfig(60, 61, False)
    var raised_exceed = False
    try:
        exceed_topk.validate()
    except e:
        raised_exceed = True
    assert_true(raised_exceed)

    var renorm_cfg = RouterConfig(60, 4, True)
    var raised_renorm = False
    try:
        renorm_cfg.validate()
    except e:
        raised_renorm = True
    assert_true(raised_renorm)


def test_routing_info_to_json() raises:
    """Format serialisasi JSON dari RoutingInfo."""
    var sel = List[List[Int]]()
    var p1 = List[Int]()
    p1.append(5)
    p1.append(12)
    sel.append(p1^)

    var probs = List[List[Float32]]()
    var pr1 = List[Float32]()
    pr1.append(Float32(0.35))
    pr1.append(Float32(0.25))
    probs.append(pr1^)

    var info = RoutingInfo(sel^, probs^, 1, 2)
    var json_str = info.to_json()
    assert_true(_contains(json_str, '"selected_experts":[[5,12]]'))
    assert_true(_contains(json_str, '"router_probs":[['))


def test_router_project_known_values() raises:
    """Proyeksi linear z = x W_r^T dengan nilai terdefinisi."""
    var seq_len = 2
    var hidden = 2
    var num_experts = 3
    # x: [2, 2] = [[1, 2], [3, 4]]
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    # W_r: [3, 2] = [[1, 0], [0, 1], [1, 1]]
    var w: List[Float32] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    var logits = router_project(x, w, seq_len, hidden, num_experts)
    assert_equal(len(logits), 6)
    # Token 0: [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1.0, 2.0, 3.0]
    assert_almost_equal(logits[0], Float32(1.0), atol=1e-5)
    assert_almost_equal(logits[1], Float32(2.0), atol=1e-5)
    assert_almost_equal(logits[2], Float32(3.0), atol=1e-5)
    # Token 1: [3*1+4*0, 3*0+4*1, 3*1+4*1] = [3.0, 4.0, 7.0]
    assert_almost_equal(logits[3], Float32(3.0), atol=1e-5)
    assert_almost_equal(logits[4], Float32(4.0), atol=1e-5)
    assert_almost_equal(logits[5], Float32(7.0), atol=1e-5)


def test_router_project_shape_errors() raises:
    """Penolakan error dimensi pada router_project."""
    var valid_x: List[Float32] = [1.0, 2.0]
    var valid_w: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var short_x: List[Float32] = [1.0]
    var short_w: List[Float32] = [1.0, 2.0]

    var raised = False
    try:
        var _out = router_project(short_x, valid_w, 1, 2, 2)
    except e:
        raised = True
    assert_true(raised)

    raised = False
    try:
        var _out2 = router_project(valid_x, short_w, 1, 2, 2)
    except e:
        raised = True
    assert_true(raised)

    raised = False
    try:
        var _out3 = router_project(valid_x, valid_w, 0, 2, 2)
    except e:
        raised = True
    assert_true(raised)


def test_router_project_nan_inf() raises:
    """Penolakan nilai NaN atau Inf pada aktivasi dan bobot router."""
    var nan_x: List[Float32] = [Float32(0.0) / Float32(0.0), 1.0]
    var valid_x: List[Float32] = [1.0, 2.0]
    var valid_w: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var nan_w: List[Float32] = [1.0, Float32(0.0) / Float32(0.0), 0.0, 1.0]

    var raised = False
    try:
        var _out = router_project(nan_x, valid_w, 1, 2, 2)
    except e:
        raised = True
    assert_true(raised)

    raised = False
    try:
        var _out2 = router_project(valid_x, nan_w, 1, 2, 2)
    except e:
        raised = True
    assert_true(raised)


def test_router_softmax_stable_max_shift() raises:
    """Invariansi pergeseran maksimum router softmax: softmax(z + c) == softmax(z).
    """
    var z: List[Float32] = [1.0, 2.0, 5.0, 3.0]
    var z_shifted: List[Float32] = [1001.0, 1002.0, 1005.0, 1003.0]

    var p1 = router_softmax(z, 1, 4)
    var p2 = router_softmax(z_shifted, 1, 4)

    assert_equal(len(p1), 4)
    assert_equal(len(p2), 4)
    for i in range(4):
        assert_almost_equal(p1[i], p2[i], atol=1e-5)


def test_router_softmax_sum_to_one() raises:
    """Probabilitas softmax router wajib berjumlah tepat 1.0 per token."""
    var z: List[Float32] = [-2.0, 0.5, 3.0, 1.2, -0.4]
    var p = router_softmax(z, 1, 5)

    var sum_p = Float32(0.0)
    for i in range(5):
        assert_true(p[i] > Float32(0.0))
        sum_p += p[i]
    assert_almost_equal(sum_p, Float32(1.0), atol=1e-5)


def test_router_softmax_nan_inf() raises:
    """Penolakan NaN/Inf pada logits di router_softmax."""
    var nan_z: List[Float32] = [1.0, Float32(0.0) / Float32(0.0), 2.0]
    var raised = False
    try:
        var _p = router_softmax(nan_z, 1, 3)
    except e:
        raised = True
    assert_true(raised)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_router_config_validation]()
    suite.test[test_routing_info_to_json]()
    suite.test[test_router_project_known_values]()
    suite.test[test_router_project_shape_errors]()
    suite.test[test_router_project_nan_inf]()
    suite.test[test_router_softmax_stable_max_shift]()
    suite.test[test_router_softmax_sum_to_one]()
    suite.test[test_router_softmax_nan_inf]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
