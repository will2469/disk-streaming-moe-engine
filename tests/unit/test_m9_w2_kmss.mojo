# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M9-W2: Format Sesi Biner KMSS v1 & Checksum Verification."""

from core.config import ModelConfig
from format.kmss import KmssMetadata, read_kmss_v1, write_kmss_v1
from layers.gated_attention import GatedAttnKVCache
from layers.gdn import GDNState
from std.collections import List
from std.math import abs
from std.os import unlink
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def test_kmss_header_and_roundtrip() raises:
    """Verifikasi serialisasi dan deserialisasi round-trip KMSS v1 serta integritas data.
    """
    var cfg = ModelConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        vocab_size=1000,
        num_key_value_heads=1,
        head_dim_override=32,
        full_attention_interval=4,
        architecture="qwen3.6",
    )

    var l_att = cfg.num_attention_layers()
    var h_kv = cfg.num_key_value_heads
    var head_dim = cfg.head_dim()

    var kv_cache = GatedAttnKVCache(16, l_att, h_kv, head_dim)
    kv_cache.current_len = 3

    # Isi KV Cache dengan data deterministik
    for pos in range(3):
        for att in range(l_att):
            for h in range(h_kv):
                for d in range(head_dim):
                    var kv_val = Float32(pos * 100 + att * 10 + d) * Float32(
                        0.1
                    )
                    kv_cache.set_k(pos, att, h, d, kv_val)
                    kv_cache.set_v(pos, att, h, d, kv_val + Float32(0.5))

    # Isi GDN State
    var gdn_layers = cfg.num_gdn_layers()
    var dv = 16
    var dk = 16
    var gdn_state = GDNState(gdn_layers, dv, dk)
    for lyr in range(gdn_layers):
        for r in range(dv):
            for c in range(dk):
                gdn_state.set(
                    lyr, r, c, Float32(lyr * 100 + r * 10 + c) * Float32(0.01)
                )

    var tokens = List[Int]()
    tokens.append(101)
    tokens.append(202)
    tokens.append(303)

    var session_file = "/tmp/test_kmss_valid.session"
    write_kmss_v1(session_file, kv_cache, gdn_state, tokens, cfg, kv_dtype=1)

    # Baca kembali session file
    var res = read_kmss_v1(session_file)
    var meta = res[0].copy()
    ref restored_kv = res[1]
    ref restored_gdn = res[2]
    ref restored_tokens = res[3]

    # 1. Verifikasi Metadata
    assert_equal(meta.version, 1)
    assert_equal(meta.architecture_id, 2)
    assert_equal(meta.seq_len, 3)
    assert_equal(meta.vocab_size, 1000)
    assert_equal(meta.kv_layers, l_att)
    assert_equal(meta.kv_heads, h_kv)
    assert_equal(meta.head_dim, head_dim)
    assert_equal(meta.kv_dtype, 1)
    assert_equal(meta.gdn_layers, gdn_layers)
    assert_equal(meta.gdn_dv, dv)
    assert_equal(meta.gdn_dk, dk)
    assert_equal(meta.gdn_dtype, 1)

    # 2. Verifikasi KV Cache roundtrip
    assert_equal(restored_kv.current_len, 3)
    for pos in range(3):
        for att in range(l_att):
            for h in range(h_kv):
                for d in range(head_dim):
                    var exp_k = Float32(pos * 100 + att * 10 + d) * Float32(0.1)
                    var exp_v = exp_k + Float32(0.5)
                    assert_almost_equal(
                        restored_kv.get_k(pos, att, h, d), exp_k, atol=1e-6
                    )
                    assert_almost_equal(
                        restored_kv.get_v(pos, att, h, d), exp_v, atol=1e-6
                    )

    # 3. Verifikasi GDN State roundtrip
    for lyr in range(gdn_layers):
        for r in range(dv):
            for c in range(dk):
                var exp_s = Float32(lyr * 100 + r * 10 + c) * Float32(0.01)
                assert_almost_equal(
                    restored_gdn.get(lyr, r, c), exp_s, atol=1e-6
                )

    # 4. Verifikasi Token IDs prefix
    assert_equal(len(restored_tokens), 3)
    assert_equal(restored_tokens[0], 101)
    assert_equal(restored_tokens[1], 202)
    assert_equal(restored_tokens[2], 303)


def test_kmss_checksum_corruption_detection() raises:
    """Verifikasi checksum SHA-256 mendeteksi korupsi 1 byte pada payload KMSS.
    """
    var valid_file = "/tmp/test_kmss_valid.session"
    var corrupt_file = "/tmp/test_kmss_corrupt.session"

    var f = open(valid_file, "r")
    var data = f.read_bytes()
    f.close()

    assert_true(len(data) > 160)

    # Korup 1 byte pada payload (posisi 130)
    var corrupt_data = List[UInt8]()
    corrupt_data.reserve(len(data))
    for i in range(len(data)):
        if i == 130:
            corrupt_data.append(UInt8(Int(data[i]) ^ 0xFF))
        else:
            corrupt_data.append(data[i])

    var f_corrupt = open(corrupt_file, "w")
    f_corrupt.write_bytes(Span(corrupt_data))
    f_corrupt.close()

    var caught = False
    try:
        _ = read_kmss_v1(corrupt_file)
    except e:
        var err_s = String(e)
        if err_s.find("CORRUPT_SESSION_CHECKSUM") >= 0:
            caught = True

    assert_true(caught)


def test_kmss_falsification_magic_and_architecture() raises:
    """Verifikasi falsifikasi: magic rusak atau architecture_id mismatch langsung ditolak.
    """
    var valid_file = "/tmp/test_kmss_valid.session"
    var f = open(valid_file, "r")
    var data = f.read_bytes()
    f.close()

    # 1. Invalid magic (ganti 'KMSS' -> 'XXXX')
    var bad_magic = "/tmp/test_kmss_bad_magic.session"
    var bad_magic_data = List[UInt8]()
    bad_magic_data.reserve(len(data))
    for i in range(len(data)):
        if i < 4:
            bad_magic_data.append(UInt8(0x58))  # 'X'
        else:
            bad_magic_data.append(data[i])

    var f_bm = open(bad_magic, "w")
    f_bm.write_bytes(Span(bad_magic_data))
    f_bm.close()

    var magic_caught = False
    try:
        _ = read_kmss_v1(bad_magic)
    except e:
        var err_s = String(e)
        if err_s.find("FORMAT_MISMATCH") >= 0:
            magic_caught = True
    assert_true(magic_caught)

    # 2. Invalid architecture_id (ganti 2 -> 99)
    var bad_arch = "/tmp/test_kmss_bad_arch.session"
    var bad_arch_data = List[UInt8]()
    bad_arch_data.reserve(len(data))
    for i in range(len(data)):
        if i == 8:
            bad_arch_data.append(UInt8(99))
        else:
            bad_arch_data.append(data[i])

    var f_ba = open(bad_arch, "w")
    f_ba.write_bytes(Span(bad_arch_data))
    f_ba.close()

    var arch_caught = False
    try:
        _ = read_kmss_v1(bad_arch)
    except e:
        var err_s = String(e)
        if err_s.find("CONFIG_MISMATCH") >= 0:
            arch_caught = True
    assert_true(arch_caught)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_kmss_header_and_roundtrip]()
    suite.test[test_kmss_checksum_corruption_detection]()
    suite.test[test_kmss_falsification_magic_and_architecture]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
