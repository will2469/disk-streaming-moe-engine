# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests klien tokenizer HF REAL (fix #3).

Tanpa hash, tanpa stub ID: setiap nilai berasal dari pustaka BPE HF atas
fixture fixtures/m12_tokenizer (nilai ekspektasi dibuktikan generator
generate_m12_tokenizer.py via HF langsung). Plus fail-closed paths.
"""

from std.collections import List
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
)
from tokenizer.hf_client import decode_via_hf, encode_via_hf


def test_hf_encode_fixture_values() raises:
    """ID BPE real fixture == nilai HF langsung (tanpa modifikasi)."""
    var ids = encode_via_hf("Halo Dismoen!", "fixtures/m12_tokenizer")
    assert_equal(len(ids), 3)
    assert_equal(ids[0], 259)
    assert_equal(ids[1], 271)
    assert_equal(ids[2], 3)


def test_hf_roundtrip_in_vocab() raises:
    """Teks dalam-vocab pulang-pergi eksak byte-per-byte (BPE real)."""
    var texts = List[String]()
    texts.append("Halo Dismoen!")
    texts.append("Para tetua menelaah naskah kuno.")
    texts.append("Jelaskan cara kerja MoE")
    for i in range(len(texts)):
        var ids = encode_via_hf(texts[i], "fixtures/m12_tokenizer")
        assert_true(len(ids) > 0)
        var back = decode_via_hf(ids, "fixtures/m12_tokenizer")
        assert_equal(back, texts[i])


def test_hf_shell_tricky_passthrough() raises:
    """Teks dengan karakter shell-aktif lolos framing utuh ke helper.

    BPE fixture kecil memetakan beberapa char ke unk (id 0, LOSSY secara
    jujur seperti HF langsung) — yang diuji di sini adalah FRAMING:
    helper menerima teks utuh (tidak terpotong/terekspansi shell) dan
    menjawab tanpa error. Bandingkan dengan HF langsung bila perlu.
    """
    var tricky = String('it\'s `$PATH` "quoted"')
    var ids = encode_via_hf(tricky, "fixtures/m12_tokenizer")
    # Nilai identik HF langsung (bukti tanpa modifikasi/mock):
    # [87, 0, 47, 55, 0, 0, 25, 17, 27, 20, 0, 55, 0, 165, 73, 33, 32, 0]
    assert_equal(len(ids), 18)
    assert_equal(ids[0], 87)
    var back = decode_via_hf(ids, "fixtures/m12_tokenizer")
    assert_true(back.byte_length() > 0)


def test_hf_missing_tokenizer_fail_closed() raises:
    """Tanpa tokenizer.json -> TOKENIZER_NOT_FOUND (bukan hash)."""
    var caught_enc = False
    try:
        var ids = encode_via_hf("hi", "/tmp/dismoen_tok_/absent_dir_xyz")
    except e:
        assert_true(String(e).find("TOKENIZER_NOT_FOUND") >= 0)
        caught_enc = True
    assert_true(caught_enc)

    var caught_dec = False
    try:
        var ids = List[Int]()
        ids.append(1)
        var t = decode_via_hf(ids, "/tmp/dismoen_tok_/absent_dir_xyz")
    except e:
        assert_true(String(e).find("TOKENIZER_NOT_FOUND") >= 0)
        caught_dec = True
    assert_true(caught_dec)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_hf_encode_fixture_values]()
    suite.test[test_hf_roundtrip_in_vocab]()
    suite.test[test_hf_shell_tricky_passthrough]()
    suite.test[test_hf_missing_tokenizer_fail_closed]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
