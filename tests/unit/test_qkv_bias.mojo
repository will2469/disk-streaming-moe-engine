# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk validasi attention bias dan oracle slice QKV (M2-W1)."""

from core.config import LoadMemoryTelemetry, ModelConfig, _contains
from format.file_io import read_small_file
from format.index import parse_index
from layers.qkv import load_layer_qkv_weights, qkv_project
from layers.qkv_bias import (
    collect_attention_bias_names,
    validate_attention_bias_in_index,
    validate_bias_count,
)
from std.collections import Dict, List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def test_validate_bias_count_ok() raises:
    """72 bias untuk 24 layer lolos validasi."""
    var names = List[String]()
    for i in range(72):
        names.append("bias_" + String(i))
    validate_bias_count(names, 24, "test_shard.safetensors")


def test_validate_bias_count_fail() raises:
    """71 bias untuk 24 layer gagal validasi (WEIGHT_LOAD_FAILED)."""
    var names = List[String]()
    for i in range(71):
        names.append("bias_" + String(i))
    var raised = False
    try:
        validate_bias_count(names, 24, "test_shard.safetensors")
    except:
        raised = True
    assert_true(raised)


def test_collect_attention_bias_names() raises:
    """Filter nama tensor yang berakhiran .bias dan memuat self_attn + q/k/v_proj.
    """
    var all_names: List[String] = [
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.input_layernorm.weight",
    ]
    var biases = collect_attention_bias_names(all_names, 1)
    assert_equal(len(biases), 3)
    assert_equal(biases[0], "model.layers.0.self_attn.q_proj.bias")
    assert_equal(biases[1], "model.layers.0.self_attn.k_proj.bias")
    assert_equal(biases[2], "model.layers.0.self_attn.v_proj.bias")


def test_validate_attention_bias_in_index_72() raises:
    """24 layer x 3 bias = 72 bias lengkap -> lolos; 71 bias -> WEIGHT_LOAD_FAILED.
    """
    var full_map = Dict[String, String]()
    for l in range(24):
        var prefix = "model.layers." + String(l) + ".self_attn."
        full_map[prefix + "q_proj.bias"] = "shard-0.safetensors"
        full_map[prefix + "k_proj.bias"] = "shard-0.safetensors"
        full_map[prefix + "v_proj.bias"] = "shard-0.safetensors"
    validate_attention_bias_in_index(full_map, 24)

    # Hapus 1 bias (hanya 71 bias)
    var incomplete_map = Dict[String, String]()
    for l in range(24):
        var prefix = "model.layers." + String(l) + ".self_attn."
        incomplete_map[prefix + "q_proj.bias"] = "shard-0.safetensors"
        if l != 23:
            incomplete_map[prefix + "k_proj.bias"] = "shard-0.safetensors"
        incomplete_map[prefix + "v_proj.bias"] = "shard-0.safetensors"
    var raised = False
    try:
        validate_attention_bias_in_index(incomplete_map, 24)
    except:
        raised = True
    assert_true(raised)


def test_property_p2_config_vs_index() raises:
    """Property P-2 (§2.3): config tidak menulis attention_bias, namun index memiliki 72 bias.
    """
    # 1. Verifikasi model_config.json tidak memuat field attention_bias
    var cfg_bytes = read_small_file("fixtures/m0/model_config.json")
    var cfg_str = String(from_utf8_lossy=Span(cfg_bytes))
    assert_false(_contains(cfg_str, "attention_bias"))

    # 2. Verifikasi index asli fixtures/m0_qwen_index.json memiliki tepat 72 tensor bias attention
    var packed = parse_index("fixtures/m0_qwen_index.json")
    var nn = 0
    var cs = packed[0].as_bytes()
    for ci in range(len(cs)):
        nn = nn * 10 + (Int(cs[ci]) - 48)
    assert_equal(nn, 4659)

    var all_names = List[String]()
    var weight_map = Dict[String, String]()
    for i in range(nn):
        var nm = packed[1 + 2 * i]
        var sf = packed[1 + 2 * i + 1]
        all_names.append(nm)
        weight_map[nm] = sf

    var bias_names = collect_attention_bias_names(all_names, 24)
    assert_equal(len(bias_names), 72)
    validate_bias_count(bias_names, 24, "fixtures/m0_qwen_index.json")
    validate_attention_bias_in_index(weight_map, 24)


def test_load_layer_qkv_weights_validation() raises:
    """Validasi load_layer_qkv_weights: layer invalid dan tensor hilang."""
    var cfg = ModelConfig(64, 24, 2, 512)
    var empty_map = Dict[String, String]()
    var telem = LoadMemoryTelemetry()

    # Layer < 0
    var raised_neg = False
    try:
        var _w = load_layer_qkv_weights(-1, "", empty_map, cfg, telem)
    except:
        raised_neg = True
    assert_true(raised_neg)

    # Layer >= num_hidden_layers
    var raised_high = False
    try:
        var _w2 = load_layer_qkv_weights(24, "", empty_map, cfg, telem)
    except:
        raised_high = True
    assert_true(raised_high)

    # Missing tensor in weight map
    var raised_missing = False
    try:
        var _w3 = load_layer_qkv_weights(0, "", empty_map, cfg, telem)
    except:
        raised_missing = True
    assert_true(raised_missing)


def test_qkv_project_oracle_slice() raises:
    """Oracle slice test: memverifikasi proyeksi linear fp32 y = xW^T + b
    secara presisi terhadap ground truth independen (PyTorch fp32 formula).
    """
    var seq_len = 2
    var hidden = 4
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
    var w_q: List[Float32] = [
        0.1,
        0.2,
        0.3,
        0.4,
        -0.1,
        0.5,
        0.0,
        0.2,
        0.3,
        -0.2,
        0.1,
        0.0,
        0.0,
        0.1,
        -0.3,
        0.2,
    ]
    var b_q: List[Float32] = [0.01, -0.02, 0.03, -0.04]

    var q = qkv_project(x, w_q, b_q, seq_len, hidden)
    assert_equal(len(q), 8)
    assert_almost_equal(q[0], Float32(3.01), atol=1e-5)
    assert_almost_equal(q[1], Float32(1.68), atol=1e-5)
    assert_almost_equal(q[2], Float32(0.23), atol=1e-5)
    assert_almost_equal(q[3], Float32(0.06), atol=1e-5)
    assert_almost_equal(q[4], Float32(0.41), atol=1e-5)
    assert_almost_equal(q[5], Float32(-0.67), atol=1e-5)
    assert_almost_equal(q[6], Float32(0.63), atol=1e-5)
    assert_almost_equal(q[7], Float32(-0.99), atol=1e-5)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_validate_bias_count_ok]()
    suite.test[test_validate_bias_count_fail]()
    suite.test[test_collect_attention_bias_names]()
    suite.test[test_validate_attention_bias_in_index_72]()
    suite.test[test_property_p2_config_vs_index]()
    suite.test[test_load_layer_qkv_weights_validation]()
    suite.test[test_qkv_project_oracle_slice]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
