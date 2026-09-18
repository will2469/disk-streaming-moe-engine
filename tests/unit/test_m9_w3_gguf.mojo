# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M9-W3: Deteksi Format Header, GGUF Parser, dan On-Demand Streaming Reader."""

from core.config import ModelConfig
from core.security_port import (
    validate_memory_budget_port,
    validate_vocab_size_port,
)
from format.format_detector import (
    FORMAT_GGUF,
    FORMAT_SAFETENSORS,
    detect_file_format,
    format_to_string,
)
from format.gguf import (
    GGUFIndex,
    parse_gguf_index,
    stream_gguf_tensor_f32,
)
from std.math import isfinite
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def test_format_detection_gguf_and_safetensors() raises:
    """Verifikasi deteksi format berbasis magic/header (bukan ekstensi)."""
    var gguf_path = "fixtures/m9_port_mini.gguf"
    var fmt_gguf = detect_file_format(gguf_path)
    assert_equal(fmt_gguf, FORMAT_GGUF)
    assert_equal(format_to_string(fmt_gguf), "gguf")

    var st_path = (
        "/home/will/models/qwen3.6-35b-a3b/model-00001-of-00026.safetensors"
    )
    var fmt_st = detect_file_format(st_path)
    assert_equal(fmt_st, FORMAT_SAFETENSORS)
    assert_equal(format_to_string(fmt_st), "safetensors")


def test_gguf_index_and_metadata() raises:
    """Verifikasi parsing index GGUF v3 dan penegakan formula F11b-GGUF exact size match.
    """
    var gguf_path = "fixtures/m9_port_mini.gguf"
    var index = parse_gguf_index(gguf_path)

    assert_equal(index.version, 3)
    assert_equal(index.tensor_count, 27)
    assert_true("general.architecture" in index.metadata)
    assert_equal(index.metadata["general.architecture"], "qwen3.6")

    # Invarian F11b-GGUF: actual file size == expected file size
    assert_equal(index.expected_file_size, 329280)


def test_gguf_on_demand_tensor_streaming() raises:
    """Verifikasi streaming on-demand per tensor via O_DIRECT dan dekuantisasi ke Float32.
    """
    var gguf_path = "fixtures/m9_port_mini.gguf"
    var index = parse_gguf_index(gguf_path)

    # 1. Test F32 tensor: output_norm.weight (128 elems)
    var f32_norm = stream_gguf_tensor_f32(index, "output_norm.weight")
    assert_equal(len(f32_norm), 128)
    for i in range(128):
        assert_almost_equal(f32_norm[i], Float32(1.0), atol=1e-5)

    # 2. Test Q8_0 tensor: token_embd.weight (1024 x 128 = 131072 elems)
    var q8_emb = stream_gguf_tensor_f32(index, "token_embd.weight")
    assert_equal(len(q8_emb), 131072)
    for i in range(min(len(q8_emb), 512)):
        assert_true(isfinite(q8_emb[i]))

    # 3. Test Q4_K tensor: blk.0.linear_attn.k.weight (128 x 32 = 4096 elems)
    var q4_k = stream_gguf_tensor_f32(index, "blk.0.linear_attn.k.weight")
    assert_equal(len(q4_k), 4096)
    for i in range(min(len(q4_k), 512)):
        assert_true(isfinite(q4_k[i]))

    # 4. Test Q3_K tensor: blk.0.ffn_down_exps.0.weight (64 x 128 = 8192 elems)
    var q3_k = stream_gguf_tensor_f32(index, "blk.0.ffn_down_exps.0.weight")
    assert_equal(len(q3_k), 8192)
    for i in range(min(len(q3_k), 512)):
        assert_true(isfinite(q3_k[i]))


def test_security_port_guards() raises:
    """Verifikasi security guard SEC-3 dan SEC-4."""
    # SEC-3 Vocab guard
    validate_vocab_size_port(248320, 248320)
    var caught_vocab_error = False
    try:
        validate_vocab_size_port(1000, 248320)
    except:
        caught_vocab_error = True
    assert_true(caught_vocab_error)

    # SEC-4 Memory budget guard pada mini config
    var cfg_mini = ModelConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=1024,
        num_key_value_heads=1,
        head_dim_override=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        full_attention_interval=4,
        norm_topk_prob=False,
        attention_bias=False,
        architecture="qwen3.6",
    )
    validate_memory_budget_port(cfg_mini, seq_len=8)

    # Test OOM rejection bila alokasi melebihi batas
    var caught_oom = False
    try:
        validate_memory_budget_port(cfg_mini, seq_len=10000000)
    except:
        caught_oom = True
    assert_true(caught_oom)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_format_detection_gguf_and_safetensors]()
    suite.test[test_gguf_index_and_metadata]()
    suite.test[test_gguf_on_demand_tensor_streaming]()
    suite.test[test_security_port_guards]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
