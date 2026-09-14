# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk HeadWeights, embedding lookup, matmul, dan forward head (M1)."""

from core.config import ModelConfig
from layers.head import (
    HeadWeights,
    embedding_lookup,
    forward_head,
    matmul_activation_head,
    validate_logits,
)
from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)


def test_embedding_lookup_valid() raises:
    """Lookup baris embedding menghasilkan vektor yang benar."""
    var vocab = 3
    var hidden = 2
    var table: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var ids: List[Int] = [2, 0, 1]
    var out = embedding_lookup(ids, table, vocab, hidden)
    assert_equal(len(out), 6)
    assert_almost_equal(out[0], Float32(5.0))
    assert_almost_equal(out[1], Float32(6.0))
    assert_almost_equal(out[2], Float32(1.0))
    assert_almost_equal(out[3], Float32(2.0))
    assert_almost_equal(out[4], Float32(3.0))
    assert_almost_equal(out[5], Float32(4.0))


def test_embedding_lookup_out_of_bounds() raises:
    """Token id di luar rentang [0, vocab_size-1] -> TOKEN_INVALID."""
    var table: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var bad_ids: List[Int] = [2]
    var raised = False
    try:
        var _out = embedding_lookup(bad_ids, table, 2, 2)
    except e:
        raised = True
    assert_true(raised)

    var neg_ids: List[Int] = [-1]
    var raised_neg = False
    try:
        var _out2 = embedding_lookup(neg_ids, table, 2, 2)
    except e:
        raised_neg = True
    assert_true(raised_neg)


def test_matmul_activation_head_known() raises:
    """Perkalian aktivasi [2, 2] x head^T [3, 2]^T -> logits [2, 3]."""
    var act: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var head: List[Float32] = [
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    var logits = matmul_activation_head(act, head, 2, 3, 2)
    assert_equal(len(logits), 6)
    assert_almost_equal(logits[0], Float32(1.0), atol=1e-5)
    assert_almost_equal(logits[1], Float32(2.0), atol=1e-5)
    assert_almost_equal(logits[2], Float32(3.0), atol=1e-5)
    assert_almost_equal(logits[3], Float32(3.0), atol=1e-5)
    assert_almost_equal(logits[4], Float32(4.0), atol=1e-5)
    assert_almost_equal(logits[5], Float32(7.0), atol=1e-5)


def test_head_weights_untied() raises:
    """Verifikasi struktur HeadWeights memisahkan alokasi embed dan lm_head."""
    var embed: List[Float32] = [1.0, 2.0]
    var norm: List[Float32] = [1.0, 1.0]
    var head: List[Float32] = [1.0, 2.0]
    var hw = HeadWeights(embed^, norm^, head^)
    assert_true(hw.is_untied())


def test_forward_head_pipeline() raises:
    """Pipeline forward head lengkap: token -> lookup -> norm -> matmul."""
    var cfg = ModelConfig(64, 2, 2, 512)
    var eps = Float32(1e-6)

    # Inisialisasi bobot numerik valid terdefinisi (skala kecil agar stabil)
    var embed = List[Float32]()
    embed.reserve(cfg.vocab_size * cfg.hidden_size)
    for i in range(cfg.vocab_size * cfg.hidden_size):
        embed.append(Float32(0.01) * Float32((i % 17) - 8))

    var norm = List[Float32]()
    norm.reserve(cfg.hidden_size)
    for _ in range(cfg.hidden_size):
        norm.append(Float32(1.0))

    var head = List[Float32]()
    head.reserve(cfg.vocab_size * cfg.hidden_size)
    for k in range(cfg.vocab_size * cfg.hidden_size):
        head.append(Float32(0.01) * Float32((k % 13) - 6))

    var weights = HeadWeights(embed^, norm^, head^)
    assert_true(weights.is_untied())

    # Uji 48 token (3 prompt x 16 token)
    var tokens = List[Int]()
    for p in range(3):
        for t in range(16):
            tokens.append((p * 16 + t) % cfg.vocab_size)

    var logits = forward_head(tokens, weights, cfg, eps)
    validate_logits(logits, 3, 16, cfg.vocab_size)
    assert_equal(len(logits), 3 * 16 * cfg.vocab_size)


def test_validate_logits_ok_and_fails() raises:
    """Validasi format logits (dimensi dan deteksi finite)."""
    var valid = List[Float32]()
    for _ in range(48 * 4):
        valid.append(Float32(0.5))
    validate_logits(valid, 3, 16, 4)

    var invalid_len = List[Float32]()
    invalid_len.append(Float32(1.0))
    var raised_len = False
    try:
        validate_logits(invalid_len, 3, 16, 4)
    except e:
        raised_len = True
    assert_true(raised_len)

    var nan_logits = List[Float32]()
    for _ in range(48 * 4):
        nan_logits.append(Float32(0.5))
    nan_logits[10] = Float32(0.0) / Float32(0.0)
    var raised_nan = False
    try:
        validate_logits(nan_logits, 3, 16, 4)
    except e:
        raised_nan = True
    assert_true(raised_nan)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_embedding_lookup_valid]()
    suite.test[test_embedding_lookup_out_of_bounds]()
    suite.test[test_matmul_activation_head_known]()
    suite.test[test_head_weights_untied]()
    suite.test[test_forward_head_pipeline]()
    suite.test[test_validate_logits_ok_and_fails]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
