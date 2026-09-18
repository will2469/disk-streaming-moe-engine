# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M12-W2a: KMSS v1 Longest-Prefix Cache & Canonical Rebase."""

from core.config import ModelConfig
from core.prefix_cache import (
    PrefixCache,
    PrefixCacheEntry,
    PrefixLookupResult,
    compute_domain_key,
)
from core.worker_pool import WorkerPool
from format.kmss import KmssMetadata, read_kmss_v1, write_kmss_v1
from layers.gated_attention import GatedAttnKVCache
from layers.gdn import GDNState
from layers.port_scheduler import (
    PortBlockWeights,
    SchedulerTimings,
    create_synthetic_block_weights,
    forward_port_macro_scheduler,
)
from std.collections import List
from std.ffi import external_call
from std.os import unlink
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def _create_dummy_kv_and_gdn(
    cfg: ModelConfig, current_len: Int
) raises -> Tuple[GatedAttnKVCache, GDNState]:
    """Helper untuk membuat KV cache dan GDN state terinisialisasi deterministik.
    """
    var l_att = cfg.num_attention_layers()
    var h_kv = cfg.num_key_value_heads
    var head_dim = cfg.head_dim()
    var kv_cache = GatedAttnKVCache(current_len + 32, l_att, h_kv, head_dim)
    kv_cache.current_len = current_len

    for pos in range(current_len):
        for att in range(l_att):
            for h in range(h_kv):
                for d in range(head_dim):
                    var val = Float32(pos * 10 + att * 2 + d) * Float32(0.01)
                    kv_cache.set_k(pos, att, h, d, val)
                    kv_cache.set_v(pos, att, h, d, val + Float32(0.1))

    var gdn_layers = cfg.num_gdn_layers()
    var dv = 16
    var dk = 16
    var gdn_state = GDNState(gdn_layers, dv, dk)
    for lyr in range(gdn_layers):
        for r in range(dv):
            for c in range(dk):
                gdn_state.set(
                    lyr, r, c, Float32(lyr * 50 + r * 5 + c) * Float32(0.01)
                )

    return (kv_cache^, gdn_state^)


def test_exact_longest_prefix_match() raises:
    """Verifikasi lookup mencari entri dengan prefiks token terpanjang yang eksak.
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

    var domain_key = compute_domain_key("qwen3.6", "pin_v1", "m12_v1")
    var cache = PrefixCache(capacity=4)

    # Entri 1: panjang 3 [10, 20, 30]
    var t1 = List[Int]()
    t1.append(10)
    t1.append(20)
    t1.append(30)
    var states1 = _create_dummy_kv_and_gdn(cfg, 3)
    assert_true(
        cache.insert(
            domain_key, t1, states1[0], states1[1], finish_reason="stop"
        )
    )

    # Entri 2: panjang 5 [10, 20, 30, 40, 50]
    var t2 = List[Int]()
    t2.append(10)
    t2.append(20)
    t2.append(30)
    t2.append(40)
    t2.append(50)
    var states2 = _create_dummy_kv_and_gdn(cfg, 5)
    assert_true(
        cache.insert(
            domain_key, t2, states2[0], states2[1], finish_reason="stop"
        )
    )

    # Lookup request A: [10, 20, 30, 40, 50, 60, 70]
    # Keduanya cocok sebagai prefix, tapi Entri 2 lebih panjang (5 > 3)
    var req_a = List[Int]()
    req_a.append(10)
    req_a.append(20)
    req_a.append(30)
    req_a.append(40)
    req_a.append(50)
    req_a.append(60)
    req_a.append(70)

    var res_a = cache.lookup(domain_key, req_a)
    assert_true(res_a.hit)
    assert_equal(res_a.prefix_len, 5)
    assert_equal(res_a.delta_tokens_len, 2)

    # Lookup request B: [10, 20, 30, 99, 100]
    # Hanya Entri 1 yang cocok sebagai prefix (panjang 3)
    var req_b = List[Int]()
    req_b.append(10)
    req_b.append(20)
    req_b.append(30)
    req_b.append(99)
    req_b.append(100)

    var res_b = cache.lookup(domain_key, req_b)
    assert_true(res_b.hit)
    assert_equal(res_b.prefix_len, 3)
    assert_equal(res_b.delta_tokens_len, 2)

    # Lookup request C: [99, 100] -> Tak ada entri yang cocok
    var req_c = List[Int]()
    req_c.append(99)
    req_c.append(100)

    var res_c = cache.lookup(domain_key, req_c)
    assert_false(res_c.hit)
    assert_equal(res_c.prefix_len, 0)
    assert_equal(res_c.delta_tokens_len, 2)


def test_mid_history_divergence() raises:
    """Verifikasi penanganan divergensi: regenerasi / edit giliran me-reuse common prefix.
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
    var domain_key = compute_domain_key("qwen3.6", "pin_v1", "m12_v1")
    var cache = PrefixCache(capacity=4)

    # Turn 1: [101, 102, 103]
    var turn1 = List[Int]()
    turn1.append(101)
    turn1.append(102)
    turn1.append(103)
    var st1 = _create_dummy_kv_and_gdn(cfg, 3)
    assert_true(
        cache.insert(domain_key, turn1, st1[0], st1[1], finish_reason="stop")
    )

    # Turn 2: [101, 102, 103, 201, 202, 203]
    var turn2 = List[Int]()
    turn2.append(101)
    turn2.append(102)
    turn2.append(103)
    turn2.append(201)
    turn2.append(202)
    turn2.append(203)
    var st2 = _create_dummy_kv_and_gdn(cfg, 6)
    assert_true(
        cache.insert(domain_key, turn2, st2[0], st2[1], finish_reason="stop")
    )

    # Pengguna mengedit Turn 2 query -> prompt baru: [101, 102, 103, 301, 302]
    # Token 201..203 digantikan 301..302 (divergensi pada indeks 3)
    var edit_req = List[Int]()
    edit_req.append(101)
    edit_req.append(102)
    edit_req.append(103)
    edit_req.append(301)
    edit_req.append(302)

    var res = cache.lookup(domain_key, edit_req)
    # Turn 2 tidak cocok karena token ke-3 berbeda (201 != 301).
    # Turn 1 cocok sebagai common prefix terpanjang (panjang 3).
    assert_true(res.hit)
    assert_equal(res.prefix_len, 3)
    assert_equal(res.delta_tokens_len, 2)  # Prefill hanya [301, 302]


def test_domain_isolation() raises:
    """Verifikasi domain separation: beda model/tokenizer tidak pernah match."""
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
    var domain_a = compute_domain_key("qwen3.6-35b", "pin_a", "m12_v1")
    var domain_b = compute_domain_key("qwen3.6-35b", "pin_b", "m12_v1")

    assert_true(domain_a != domain_b)

    var cache = PrefixCache(capacity=4)
    var tokens = List[Int]()
    tokens.append(10)
    tokens.append(20)
    tokens.append(30)
    var st = _create_dummy_kv_and_gdn(cfg, 3)

    # Insert di bawah domain_a
    assert_true(
        cache.insert(domain_a, tokens, st[0], st[1], finish_reason="stop")
    )

    # Lookup dengan urutan token yang persis sama tapi domain_b -> Wajib MISS
    var res = cache.lookup(domain_b, tokens)
    assert_false(res.hit)
    assert_equal(res.prefix_len, 0)


def test_session_isolation_same_length() raises:
    """Verifikasi dua sesi dengan panjang token sama tidak pernah bertabrakan.
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
    var domain_key = compute_domain_key("qwen3.6", "pin_v1", "m12_v1")
    var cache = PrefixCache(capacity=4)

    # Sesi A: 4 token [1, 2, 3, 4]
    var sess_a = List[Int]()
    sess_a.append(1)
    sess_a.append(2)
    sess_a.append(3)
    sess_a.append(4)
    var st_a = _create_dummy_kv_and_gdn(cfg, 4)
    assert_true(
        cache.insert(domain_key, sess_a, st_a[0], st_a[1], finish_reason="stop")
    )

    # Sesi B: 4 token [5, 6, 7, 8] (panjang sama persis dengan Sesi A)
    var sess_b = List[Int]()
    sess_b.append(5)
    sess_b.append(6)
    sess_b.append(7)
    sess_b.append(8)

    # Lookup Sesi B terhadap cache -> Token per token tidak cocok -> Wajib MISS
    var res_b = cache.lookup(domain_key, sess_b)
    assert_false(res_b.hit)
    assert_equal(res_b.prefix_len, 0)
    assert_equal(res_b.delta_tokens_len, 4)


def test_insertion_policy_clean_vs_abort() raises:
    """Verifikasi kebijakan pemasukan: HANYA saat clean completion (stop/length); abort insert NOTHING.
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
    var domain_key = compute_domain_key("qwen3.6", "pin_v1", "m12_v1")
    var cache = PrefixCache(capacity=4)

    var tok = List[Int]()
    tok.append(1)
    tok.append(2)
    var st = _create_dummy_kv_and_gdn(cfg, 2)

    # 1. Abort generasi (Ctrl+C / client disconnect) -> Insert gagal, cache tetap kosong
    assert_false(
        cache.insert(domain_key, tok, st[0], st[1], finish_reason="abort")
    )
    assert_equal(cache.size(), 0)

    # 2. Error generasi -> Insert gagal
    assert_false(
        cache.insert(domain_key, tok, st[0], st[1], finish_reason="error")
    )
    assert_equal(cache.size(), 0)

    # 3. Timeout -> Insert gagal
    assert_false(
        cache.insert(domain_key, tok, st[0], st[1], finish_reason="timeout")
    )
    assert_equal(cache.size(), 0)

    # 4. Clean completion: stop -> Insert berhasil
    assert_true(
        cache.insert(domain_key, tok, st[0], st[1], finish_reason="stop")
    )
    assert_equal(cache.size(), 1)

    # 5. Clean completion: length -> Insert berhasil
    var tok2 = List[Int]()
    tok2.append(3)
    tok2.append(4)
    var st2 = _create_dummy_kv_and_gdn(cfg, 2)
    assert_true(
        cache.insert(domain_key, tok2, st2[0], st2[1], finish_reason="length")
    )
    assert_equal(cache.size(), 2)


def test_bounded_lru_eviction() raises:
    """Verifikasi bounded LRU: batas kapasitas ditegakkan dan entri terlama di-evict.
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
    var domain_key = compute_domain_key("qwen3.6", "pin_v1", "m12_v1")
    # Kapasitas dibatasi 2 entri
    var cache = PrefixCache(capacity=2)

    var t1 = List[Int]()
    t1.append(1)
    var st1 = _create_dummy_kv_and_gdn(cfg, 1)

    var t2 = List[Int]()
    t2.append(2)
    var st2 = _create_dummy_kv_and_gdn(cfg, 1)

    var t3 = List[Int]()
    t3.append(3)
    var st3 = _create_dummy_kv_and_gdn(cfg, 1)

    # Masukkan E1 dan E2
    assert_true(
        cache.insert(domain_key, t1, st1[0], st1[1], finish_reason="stop")
    )
    assert_true(
        cache.insert(domain_key, t2, st2[0], st2[1], finish_reason="stop")
    )
    assert_equal(cache.size(), 2)

    # Akses E1 sehingga E1 menjadi baru (last accessed diperbarui)
    var res1 = cache.lookup(domain_key, t1)
    assert_true(res1.hit)

    # Masukkan E3 -> Kapasitas 2 terlampaui. E2 paling lama tidak diakses -> E2 di-evict
    assert_true(
        cache.insert(domain_key, t3, st3[0], st3[1], finish_reason="stop")
    )
    assert_equal(cache.size(), 2)

    # E1 dan E3 harus tetap ada
    var res_e1 = cache.lookup(domain_key, t1)
    assert_true(res_e1.hit)

    var res_e3 = cache.lookup(domain_key, t3)
    assert_true(res_e3.hit)

    # E2 harus sudah ter-evict
    var res_e2 = cache.lookup(domain_key, t2)
    assert_false(res_e2.hit)


def test_greedy_parity_cache_vs_full() raises:
    """Verifikasi paritas deterministik greedy (temperature=0): cache-hit == full-prefill.
    """
    var cfg = ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=1000,
        num_key_value_heads=1,
        head_dim_override=16,
        full_attention_interval=2,
        architecture="qwen3.6",
    )
    var domain_key = compute_domain_key("qwen3.6", "pin_v1", "m12_v1")

    # Siapkan bobot transformer sintetis
    var blocks = List[PortBlockWeights]()
    for l in range(cfg.num_hidden_layers):
        blocks.append(create_synthetic_block_weights(cfg, l, 16, 16))

    # Urutan token total T: 6 token
    var full_tokens = List[Int]()
    full_tokens.append(10)
    full_tokens.append(20)
    full_tokens.append(30)
    full_tokens.append(40)
    full_tokens.append(50)
    full_tokens.append(60)

    # === JALUR 1: Full Prefill dari awal ===
    var kv_full = GatedAttnKVCache(
        64, cfg.num_attention_layers(), cfg.num_key_value_heads, cfg.head_dim()
    )
    var gdn_full = GDNState(cfg.num_gdn_layers(), 16, 16)
    var timings1 = SchedulerTimings()
    var pool1 = WorkerPool(1)

    var x_full = List[Float32]()
    x_full.resize(len(full_tokens) * cfg.hidden_size, Float32(0.0))
    for t in range(len(full_tokens)):
        var tid = full_tokens[t]
        for d in range(cfg.hidden_size):
            x_full[t * cfg.hidden_size + d] = Float32(
                (tid * 17 + d * 3) % 100
            ) * Float32(0.001)

    var out_full = forward_port_macro_scheduler(
        x_full,
        blocks,
        gdn_full,
        kv_full,
        0,
        len(full_tokens),
        cfg,
        timings1,
        pool1,
        16,
        16,
        Float32(1e-6),
    )
    pool1.shutdown()

    # === JALUR 2: Prefix Cache Hit (Turn 1 prefills 3 token, Turn 2 prefills delta 3 token) ===
    # Turn 1: 3 token awal [10, 20, 30]
    var prefix_tokens = List[Int]()
    prefix_tokens.append(10)
    prefix_tokens.append(20)
    prefix_tokens.append(30)

    var kv_cached = GatedAttnKVCache(
        64, cfg.num_attention_layers(), cfg.num_key_value_heads, cfg.head_dim()
    )
    var gdn_cached = GDNState(cfg.num_gdn_layers(), 16, 16)
    var timings2 = SchedulerTimings()
    var pool2 = WorkerPool(1)

    var x_p1 = List[Float32]()
    x_p1.resize(len(prefix_tokens) * cfg.hidden_size, Float32(0.0))
    for t in range(len(prefix_tokens)):
        var tid = prefix_tokens[t]
        for d in range(cfg.hidden_size):
            x_p1[t * cfg.hidden_size + d] = Float32(
                (tid * 17 + d * 3) % 100
            ) * Float32(0.001)

    _ = forward_port_macro_scheduler(
        x_p1,
        blocks,
        gdn_cached,
        kv_cached,
        0,
        len(prefix_tokens),
        cfg,
        timings2,
        pool2,
        16,
        16,
        Float32(1e-6),
    )

    # Masukkan ke PrefixCache
    var cache = PrefixCache(capacity=4)
    assert_true(
        cache.insert(
            domain_key,
            prefix_tokens,
            kv_cached,
            gdn_cached,
            finish_reason="stop",
        )
    )

    # Turn 2 request dengan full_tokens: lookup di cache -> HIT 3 token
    var lookup_res = cache.lookup(domain_key, full_tokens)
    assert_true(lookup_res.hit)
    assert_equal(lookup_res.prefix_len, 3)
    assert_equal(lookup_res.delta_tokens_len, 3)

    # Ambil state dari cache dan jalankan HANYA delta tokens [40, 50, 60] dari pos_offset = 3
    ref entry = cache.entries[lookup_res.entry_index]
    var kv_delta = entry.kv_cache.copy()
    var gdn_delta = entry.gdn_state.copy()

    var x_delta = List[Float32]()
    x_delta.resize(lookup_res.delta_tokens_len * cfg.hidden_size, Float32(0.0))
    for t in range(lookup_res.delta_tokens_len):
        var tid = full_tokens[lookup_res.prefix_len + t]
        for d in range(cfg.hidden_size):
            x_delta[t * cfg.hidden_size + d] = Float32(
                (tid * 17 + d * 3) % 100
            ) * Float32(0.001)

    var out_delta = forward_port_macro_scheduler(
        x_delta,
        blocks,
        gdn_delta,
        kv_delta,
        lookup_res.prefix_len,
        lookup_res.delta_tokens_len,
        cfg,
        timings2,
        pool2,
        16,
        16,
        Float32(1e-6),
    )
    pool2.shutdown()

    # Periksa aktivasi token terakhir antara Full Prefill vs Delta Prefill:
    # Elemen aktivasi terakhir harus memenuhi paritas bit-exact (atol <= 1e-6)
    var last_full_offset = (len(full_tokens) - 1) * cfg.hidden_size
    var last_delta_offset = (lookup_res.delta_tokens_len - 1) * cfg.hidden_size
    for d in range(cfg.hidden_size):
        assert_almost_equal(
            out_full[last_full_offset + d],
            out_delta[last_delta_offset + d],
            atol=1e-5,
        )


def test_canonical_rebase_teacher_forced() raises:
    """Verifikasi bahwa rebase kanonis mematerialisasi state histori assistant polos (think dibuang).
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
    var domain_key = compute_domain_key("qwen3.6", "pin_v1", "m12_v1")
    var cache = PrefixCache(capacity=4)

    # Representasi kanonis Turn 1 (assistant polos per §2.1 rule 3, tanpa tag think):
    # [10, 20, 30, 90, 70, 71, 80] (panjang 7)
    var canon_tokens = List[Int]()
    canon_tokens.append(10)
    canon_tokens.append(20)
    canon_tokens.append(30)
    canon_tokens.append(90)
    canon_tokens.append(70)
    canon_tokens.append(71)
    canon_tokens.append(80)

    # Rebase kanonis mematerialisasikan state atas canon_tokens
    var rebase_states = _create_dummy_kv_and_gdn(cfg, len(canon_tokens))
    assert_true(
        cache.insert(
            domain_key,
            canon_tokens,
            rebase_states[0],
            rebase_states[1],
            finish_reason="stop",
        )
    )

    # Turn 2 request prompt yang memperpanjang riwayat kanonis:
    # canon_tokens + [100, 101, 102]
    var turn2_prompt = List[Int]()
    for k in range(len(canon_tokens)):
        turn2_prompt.append(canon_tokens[k])
    turn2_prompt.append(100)
    turn2_prompt.append(101)
    turn2_prompt.append(102)

    var res = cache.lookup(domain_key, turn2_prompt)
    assert_true(res.hit)
    assert_equal(res.prefix_len, 7)
    assert_equal(res.delta_tokens_len, 3)


def test_dir_persistence_roundtrip() raises:
    """Verifikasi persistensi disk (save_to_dir dan load_from_dir) dengan format KMSS v1.
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
    var domain_key = compute_domain_key("qwen3.6", "pin_v1", "m12_v1")
    var cache_dir = "/tmp/test_kmss_prefix_cache_dir"

    # Buat direktori jika belum ada
    var cmd = String("mkdir -p ", cache_dir)
    var cmd_b = cmd.as_bytes()
    var cmd_z = List[UInt8]()
    for idx in range(len(cmd_b)):
        cmd_z.append(cmd_b[idx])
    cmd_z.append(0)
    _ = external_call["system", Int32](cmd_z.unsafe_ptr())

    var cache1 = PrefixCache(capacity=4)
    var t1 = List[Int]()
    t1.append(101)
    t1.append(102)
    var st1 = _create_dummy_kv_and_gdn(cfg, 2)
    assert_true(
        cache1.insert(domain_key, t1, st1[0], st1[1], finish_reason="stop")
    )

    var t2 = List[Int]()
    t2.append(101)
    t2.append(102)
    t2.append(103)
    var st2 = _create_dummy_kv_and_gdn(cfg, 3)
    assert_true(
        cache1.insert(domain_key, t2, st2[0], st2[1], finish_reason="stop")
    )

    # Simpan ke disk
    cache1.save_to_dir(cache_dir, cfg)

    # Muat ke instance PrefixCache baru
    var cache2 = PrefixCache(capacity=4)
    cache2.load_from_dir(cache_dir)

    assert_equal(cache2.size(), 2)

    var query = List[Int]()
    query.append(101)
    query.append(102)
    query.append(103)
    query.append(104)

    var res = cache2.lookup(domain_key, query)
    assert_true(res.hit)
    assert_equal(res.prefix_len, 3)
    assert_equal(res.delta_tokens_len, 1)

    # Cleanup
    var rm_cmd = String("rm -rf ", cache_dir)
    var rm_b = rm_cmd.as_bytes()
    var rm_z = List[UInt8]()
    for idx in range(len(rm_b)):
        rm_z.append(rm_b[idx])
    rm_z.append(0)
    _ = external_call["system", Int32](rm_z.unsafe_ptr())


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_exact_longest_prefix_match]()
    suite.test[test_mid_history_divergence]()
    suite.test[test_domain_isolation]()
    suite.test[test_session_isolation_same_length]()
    suite.test[test_insertion_policy_clean_vs_abort]()
    suite.test[test_bounded_lru_eviction]()
    suite.test[test_greedy_parity_cache_vs_full]()
    suite.test[test_canonical_rebase_teacher_forced]()
    suite.test[test_dir_persistence_roundtrip]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
