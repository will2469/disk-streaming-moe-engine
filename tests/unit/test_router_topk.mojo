# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk top-k selection, router_forward pipeline, dan weight loading (M3-W1)."""

from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import ShardHeaderCache
from format.file_io import read_small_file
from format.index import parse_index_to_dict
from layers.router import (
    load_layer_router_weights,
    router_forward,
)
from layers.router_types import RouterConfig
from layers.topk import select_topk
from std.collections import Dict, List
from std.math import isfinite
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)


def test_select_topk_ranking() raises:
    """Top-k memilih probabilitas tertinggi secara terurut menurun dan stabil.
    """
    var probs: List[Float32] = [0.05, 0.40, 0.10, 0.30, 0.15]
    var info = select_topk(probs, 1, 5, top_k=3, norm_topk_prob=False)

    assert_equal(info.num_tokens, 1)
    assert_equal(info.top_k, 3)
    ref exp_row = info.selected_experts[0]
    ref prob_row = info.router_probs[0]

    # Urutan seharusnya: expert 1 (0.40), expert 3 (0.30), expert 4 (0.15)
    assert_equal(exp_row[0], 1)
    assert_equal(exp_row[1], 3)
    assert_equal(exp_row[2], 4)

    assert_almost_equal(prob_row[0], Float32(0.40), atol=1e-5)
    assert_almost_equal(prob_row[1], Float32(0.30), atol=1e-5)
    assert_almost_equal(prob_row[2], Float32(0.15), atol=1e-5)


def test_select_topk_no_renorm_sum_less_than_one() raises:
    """Property norm_topk_prob=false: jumlah probabilitas top-4 < 1.0."""
    var num_experts = 60
    var probs = List[Float32]()
    probs.reserve(num_experts)
    for _ in range(num_experts):
        probs.append(Float32(1.0) / Float32(num_experts))

    var info = select_topk(probs, 1, num_experts, top_k=4, norm_topk_prob=False)
    ref prob_row = info.router_probs[0]

    var sum_topk = Float32(0.0)
    for k in range(4):
        sum_topk += prob_row[k]

    # 4/60 = ~0.0667, strictly < 1.0
    assert_almost_equal(sum_topk, Float32(4.0) / Float32(60.0), atol=1e-5)
    assert_true(sum_topk < Float32(1.0))


def test_select_topk_renorm_leak_rejected() raises:
    """Deteksi kebocoran renormalisasi memicu ROUTER_ERROR."""
    var probs: List[Float32] = [0.25, 0.25, 0.25, 0.25, 0.0]

    # Jika norm_topk_prob=true di-pass
    var raised_cfg = False
    try:
        var _info = select_topk(probs, 1, 5, top_k=2, norm_topk_prob=True)
    except e:
        raised_cfg = True
    assert_true(raised_cfg)

    # Jika top-k sengaja direnormalisasi (jumlah = 1.0) padahal unselected ada bobot
    var leaked_probs: List[Float32] = [0.6, 0.4, 0.05, 0.05]
    # top-2 = 0.6 + 0.4 = 1.0, sedangkan sisa = 0.10 > 0.0
    var raised_leak = False
    try:
        var _info2 = select_topk(
            leaked_probs, 1, 4, top_k=2, norm_topk_prob=False
        )
    except e:
        raised_leak = True
    assert_true(raised_leak)


def test_select_topk_shape_errors() raises:
    """Validasi input dimensi pada select_topk."""
    var probs: List[Float32] = [0.5, 0.5]
    var raised = False
    try:
        var _info = select_topk(probs, 1, 2, top_k=3, norm_topk_prob=False)
    except e:
        raised = True
    assert_true(raised)


def test_router_forward_pipeline() raises:
    """Pipeline forward router lengkap dari aktivasi token hingga RoutingInfo.
    """
    var cfg = RouterConfig(5, 2, False)
    var seq_len = 2
    var hidden = 3
    # x: [2, 3]
    var x: List[Float32] = [1.0, 0.5, -0.5, 0.0, 2.0, 1.0]
    # w: [5, 3]
    var w: List[Float32] = [
        0.1,
        0.2,
        0.3,
        -0.1,
        0.5,
        0.0,
        0.3,
        -0.2,
        0.1,
        0.0,
        0.1,
        -0.3,
        0.2,
        0.0,
        0.4,
    ]

    var info = router_forward(x, w, seq_len, hidden, cfg)
    assert_equal(info.num_tokens, 2)
    assert_equal(info.top_k, 2)

    for t in range(2):
        ref exp_row = info.selected_experts[t]
        ref prob_row = info.router_probs[t]
        assert_equal(len(exp_row), 2)
        assert_equal(len(prob_row), 2)
        assert_true(prob_row[0] >= prob_row[1])
        assert_true(prob_row[0] + prob_row[1] <= Float32(1.0))


def test_load_layer_router_weights_validation() raises:
    """Validasi load_layer_router_weights: layer invalid dan tensor hilang."""
    var cfg = ModelConfig(2048, 24, 16, 151936)
    var router_cfg = RouterConfig(60, 4, False)
    var empty_map = Dict[String, String]()
    var cache = ShardHeaderCache()
    var telem = LoadMemoryTelemetry()

    var raised_neg = False
    try:
        var _w = load_layer_router_weights(
            -1, "", empty_map, cfg, router_cfg, cache, telem
        )
    except e:
        raised_neg = True
    assert_true(raised_neg)

    var raised_hi = False
    try:
        var _w2 = load_layer_router_weights(
            24, "", empty_map, cfg, router_cfg, cache, telem
        )
    except e:
        raised_hi = True
    assert_true(raised_hi)

    var raised_miss = False
    try:
        var _w3 = load_layer_router_weights(
            0, "", empty_map, cfg, router_cfg, cache, telem
        )
    except e:
        raised_miss = True
    assert_true(raised_miss)


def test_load_layer_router_weights_fixture_m1() raises:
    """Memuat bobot router nyata dari fixture M1 safetensors."""
    var path_idx = "fixtures/m1/model.safetensors.index.json"
    var weight_map = parse_index_to_dict(path_idx)

    # Fixture M1: hidden_size=64, num_hidden_layers=2, num_experts=8
    var cfg = ModelConfig(64, 2, 2, 512)
    var router_cfg = RouterConfig(8, 2, False)
    var cache = ShardHeaderCache()
    var telem = LoadMemoryTelemetry()

    var w_router = load_layer_router_weights(
        0, "fixtures/m1", weight_map, cfg, router_cfg, cache, telem
    )
    assert_equal(len(w_router), 8 * 64)
    for i in range(len(w_router)):
        assert_true(isfinite(w_router[i]))


def test_property_p_norm_topk_prob_false() raises:
    """Property P: konfigurasi model harus norm_topk_prob=false."""
    var m0_raw = read_small_file("fixtures/m0/model_config.json")
    var m0_str = String(from_utf8_lossy=Span(m0_raw))
    assert_true(m0_str.find('"norm_topk_prob": false') >= 0)

    var m1_raw = read_small_file("fixtures/m1/model_config.json")
    var m1_str = String(from_utf8_lossy=Span(m1_raw))
    assert_true(m1_str.find('"norm_topk_prob": false') >= 0)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_select_topk_ranking]()
    suite.test[test_select_topk_no_renorm_sum_less_than_one]()
    suite.test[test_select_topk_renorm_leak_rejected]()
    suite.test[test_select_topk_shape_errors]()
    suite.test[test_router_forward_pipeline]()
    suite.test[test_load_layer_router_weights_validation]()
    suite.test[test_load_layer_router_weights_fixture_m1]()
    suite.test[test_property_p_norm_topk_prob_false]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
