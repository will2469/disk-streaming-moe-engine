# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit Test Suite untuk M7-W2: LRU Cache Expert Engine (Budget, State Machine, Pinning, F9)."""

from format.quant_format import QuantTensorMetadata, u16_to_float16
from io.odirect import ODirectReader
from io.lru_cache import (
    STATE_ABSENT,
    STATE_EVICTING,
    STATE_LOADING,
    STATE_RESIDENT,
    CacheEntry,
    CacheKey,
    LRUCache,
    LRUCacheConfig,
    LRUCacheStats,
    MemoryBudget,
    compute_effective_rho,
    get_f9_hot_experts,
)
from quant.dequant_kernel import dequant_kernel_simd
from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def make_dummy_payload(size: Int, fill: UInt8 = 0x55) -> List[UInt8]:
    """Helper membuat byte payload sintetis untuk pengujian cache."""
    var data = List[UInt8]()
    data.reserve(size)
    for _ in range(size):
        data.append(fill)
    return data^


def test_memory_budget_accounting_and_validation() raises:
    """Menguji penegakan kontrak budget memori sistem (fail-fast M7_ERR_LRU_ALLOC).
    """
    # 1. Budget valid: 2 GB + 1 GB + 512 MB + 512 MB + 1 GB = 5 GB <= 8 GB limit
    var valid_budget = MemoryBudget(
        resident_weights_bytes=2000000000,
        kv_cache_bytes=1000000000,
        io_buffers_bytes=512000000,
        dequant_bytes=512000000,
        runtime_headroom_bytes=1000000000,
        memory_limit_bytes=8000000000,
    )
    valid_budget.validate()
    assert_equal(valid_budget.total_allocated(), 5024000000)

    # 2. Budget invalid: total 9 GB > 8 GB limit -> wajib raise M7_ERR_LRU_ALLOC
    var invalid_budget = MemoryBudget(
        resident_weights_bytes=4000000000,
        kv_cache_bytes=2000000000,
        io_buffers_bytes=1000000000,
        dequant_bytes=1000000000,
        runtime_headroom_bytes=1500000000,
        memory_limit_bytes=8000000000,
    )

    var caught_error = False
    try:
        invalid_budget.validate()
    except e:
        var msg = String(e)
        if "M7_ERR_LRU_ALLOC" in msg:
            caught_error = True

    assert_true(caught_error, "Over-budget harus memicu M7_ERR_LRU_ALLOC")


def test_state_machine_and_single_flight() raises:
    """Menguji state machine 4-state (Absent/Loading/Resident/Evicting) dan konkurensi single-flight.
    """
    var cfg = LRUCacheConfig(capacity_bytes=10000)
    var cache = LRUCache(cfg)

    # 1. Access pada expert baru (0, 5) -> Cache MISS, caller menjadi winner (STATE_ABSENT)
    var st1 = cache.begin_access(0, 5)
    assert_equal(st1, STATE_ABSENT, "Akses pertama harus berstatus MISS")

    # 2. Worker kedua meminta expert yang sama saat masih LOADING -> Mengembalikan STATE_LOADING
    var st2 = cache.begin_access(0, 5)
    assert_equal(
        st2,
        STATE_LOADING,
        "Single-flight: worker kedua harus mendeteksi STATE_LOADING",
    )

    # 3. Finish load oleh worker pemenang -> Menjadi STATE_RESIDENT
    var payload = make_dummy_payload(128, 0x42)
    cache.finish_load(0, 5, payload^)

    # 4. Akses berikutnya -> Cache HIT (STATE_RESIDENT)
    var st3 = cache.begin_access(0, 5)
    assert_equal(st3, STATE_RESIDENT, "Sesudah load selesai harus HIT")

    var data = cache.get_resident_data(0, 5)
    assert_equal(len(data), 128)
    assert_equal(data[0], 0x42)

    # 5. Uji cancel_load pada entry lain yang gagal baca disk
    var st_fail = cache.begin_access(0, 6)
    assert_equal(st_fail, STATE_ABSENT)
    cache.cancel_load(0, 6)
    # Sesudah dibatalkan, akses berikutnya bisa mencoba lagi dari awal
    var st_retry = cache.begin_access(0, 6)
    assert_equal(
        st_retry,
        STATE_ABSENT,
        "Sesudah cancel_load status harus kembali ke ABSENT",
    )


def test_pure_lru_eviction() raises:
    """Menguji eviksi pure LRU berdasarkan timestamp tertua saat kapasitas terlampaui.
    """
    # Kapasitas 1000 bytes
    var cfg = LRUCacheConfig(capacity_bytes=1000)
    var cache = LRUCache(cfg)

    # Masukkan entry A (size 400) dan B (size 400)
    var payA = make_dummy_payload(400, 0x11)
    var payB = make_dummy_payload(400, 0x22)
    cache.finish_load(0, 1, payA^)
    cache.finish_load(0, 2, payB^)
    assert_equal(cache.current_bytes, 800)

    # Akses entry A lagi agar timestamp A lebih baru dari B
    var stA = cache.begin_access(0, 1)
    assert_equal(stA, STATE_RESIDENT)

    # Masukkan entry C (size 400). Total 800 + 400 = 1200 > 1000 -> Harus evict entry B!
    var payC = make_dummy_payload(400, 0x33)
    cache.finish_load(0, 3, payC^)

    # Verifikasi: B ter-evict (size_bytes 0 / ABSENT), A dan C tetap RESIDENT
    assert_equal(cache.current_bytes, 800)
    assert_equal(cache.stats.evictions, 1)

    # Entry A harus tetap ada
    var checkA = cache.begin_access(0, 1)
    assert_equal(checkA, STATE_RESIDENT)

    # Entry C harus ada
    var checkC = cache.begin_access(0, 3)
    assert_equal(checkC, STATE_RESIDENT)

    # Entry B harus sudah hilang (MISS)
    var checkB = cache.begin_access(0, 2)
    assert_equal(checkB, STATE_ABSENT, "Entry B harus sudah dieviksi")


def test_pin_budget_invariant_25_percent() raises:
    """Menguji invarian pin budget 25% (over-budget ditolak; victim unpinned selalu ada).
    """
    # Kapasitas 1000 bytes -> Pin budget = 25% = 250 bytes
    var cfg = LRUCacheConfig(capacity_bytes=1000, pin_budget_ratio=0.25)
    var cache = LRUCache(cfg)
    assert_equal(cfg.pin_budget_bytes(), 250)

    # 1. Pin entry pertama 200 bytes <= 250 bytes -> Diterima
    var pay_pinned = make_dummy_payload(200, 0xAA)
    cache.finish_load(0, 10, pay_pinned^, is_pinned=True)
    assert_equal(cache.pinned_bytes, 200)
    assert_equal(cache.stats.pinned_entries, 1)

    # 2. Coba pin entry kedua 100 bytes -> Total 300 > 250 -> Pin DITOLAK (masuk unpinned)
    var pay_unpinned = make_dummy_payload(100, 0xBB)
    cache.finish_load(0, 11, pay_unpinned^, is_pinned=True)
    # pinned_bytes tidak boleh bertambah
    assert_equal(cache.pinned_bytes, 200)
    assert_equal(cache.stats.pinned_entries, 1)

    # 3. Penuhi cache dengan unpinned data sampai penuh (tambah 600 bytes)
    var pay_fill = make_dummy_payload(600, 0xCC)
    cache.finish_load(0, 12, pay_fill^, is_pinned=False)
    assert_equal(cache.current_bytes, 900)

    # 4. Tambah 300 bytes lagi sehingga melebihi kapasitas (900 + 300 = 1200 > 1000)
    # Entry 10 (pinned) TIDAK BOLEH dieviksi; yang dieviksi harus unpinned (11 atau 12)
    var pay_new = make_dummy_payload(300, 0xDD)
    cache.finish_load(0, 13, pay_new^, is_pinned=False)

    # Entry 10 (pinned) harus tetap RESIDENT
    var check_pinned = cache.begin_access(0, 10)
    assert_equal(
        check_pinned,
        STATE_RESIDENT,
        "Pinned expert tidak boleh ter-evict dalam keadaan apa pun",
    )


def test_selective_prefill_bound() raises:
    """Menguji selective prefill hot experts <= pin_budget 25% (over-budget fail-fast).
    """
    var cfg = LRUCacheConfig(capacity_bytes=1000, pin_budget_ratio=0.25)
    var cache = LRUCache(cfg)

    # 1. Prefill sah: 2 entry x 100 bytes = 200 bytes <= 250 bytes
    var keys_ok = List[CacheKey]()
    keys_ok.append(CacheKey(0, 1))
    keys_ok.append(CacheKey(0, 2))

    var payloads_ok = List[List[UInt8]]()
    payloads_ok.append(make_dummy_payload(100))
    payloads_ok.append(make_dummy_payload(100))

    cache.selective_prefill(keys_ok, payloads_ok^)
    assert_equal(cache.pinned_bytes, 200)
    assert_equal(cache.stats.pinned_entries, 2)

    # 2. Prefill over-budget: 300 bytes > 250 bytes -> wajib fail-fast M7_ERR_LRU_ALLOC
    var cache2 = LRUCache(cfg)
    var keys_bad = List[CacheKey]()
    keys_bad.append(CacheKey(12, 1))
    keys_bad.append(CacheKey(12, 2))

    var payloads_bad = List[List[UInt8]]()
    payloads_bad.append(make_dummy_payload(150))
    payloads_bad.append(make_dummy_payload(150))

    var caught_alloc_err = False
    try:
        cache2.selective_prefill(keys_bad, payloads_bad^)
    except e:
        if "M7_ERR_LRU_ALLOC" in String(e):
            caught_alloc_err = True

    assert_true(
        caught_alloc_err,
        "Prefill yang melanggar batas pin budget wajib memicu M7_ERR_LRU_ALLOC",
    )


def test_corruption_detection_and_atomic_clear() raises:
    """Menguji deteksi korupsi struktural (fault injection), atomic clear(), dan M7_ERR_LRU_CORRUPT.
    """
    var cfg = LRUCacheConfig(capacity_bytes=2000)
    var cache = LRUCache(cfg)

    # Isi dengan 2 entry valid
    var pay1 = make_dummy_payload(200, 0x11)
    var pay2 = make_dummy_payload(200, 0x22)
    cache.finish_load(0, 1, pay1^)
    cache.finish_load(0, 2, pay2^)
    assert_equal(cache.current_bytes, 400)

    # Suntikkan payload korup (magic 0xDEADBEEF)
    var corrupt_payload = List[UInt8]()
    corrupt_payload.append(0xDE)
    corrupt_payload.append(0xAD)
    corrupt_payload.append(0xBE)
    corrupt_payload.append(0xEF)
    for _ in range(60):
        corrupt_payload.append(0x00)

    var caught_corrupt_err = False
    try:
        cache.finish_load(0, 3, corrupt_payload^)
    except e:
        if "M7_ERR_LRU_CORRUPT" in String(e):
            caught_corrupt_err = True

    assert_true(
        caught_corrupt_err,
        "Korupsi struktural wajib memicu error M7_ERR_LRU_CORRUPT",
    )
    # Cache wajib dibersihkan secara atomik (zero bytes resident)
    assert_equal(
        cache.current_bytes, 0, "Cache harus dibersihkan total sesudah korupsi"
    )
    assert_equal(cache.pinned_bytes, 0)


def test_quantized_only_content_and_streaming_dequant() raises:
    """Menguji kebijakan konten quantized-only dan dekuantisasi SIMD FP32 per-use.
    """
    var cfg = LRUCacheConfig(capacity_bytes=50000)
    var cache = LRUCache(cfg)

    # Siapkan payload kuantisasi 4-bit sintetis (1 group = 128 elemen)
    # 1 scale FP16 (2 bytes) + 64 packed bytes (4-bit x 128) = 66 bytes
    var num_elements = 128
    var group_size = 128

    var quant_payload = List[UInt8]()
    # FP16 scale = 1.0 (0x3C00 little endian)
    quant_payload.append(0x00)
    quant_payload.append(0x3C)
    # 64 packed bytes with nibbles (q0=2, q1=3 -> byte = 0x32)
    for _ in range(64):
        quant_payload.append(0x32)

    # Simpan ke cache sebagai quantized bytes
    cache.finish_load(0, 7, quant_payload^)

    # Ambil quantized bytes dari cache (dequant per use)
    var resident_data = cache.get_resident_data(0, 7)
    assert_equal(
        len(resident_data),
        66,
        "Data di cache harus murni quantized bytes (bukan FP32)",
    )

    # Parse scales FP16 dan packed weights
    var scales = List[Float16]()
    var u = UInt16(resident_data[0]) | (UInt16(resident_data[1]) << 8)
    scales.append(u16_to_float16(u))

    var weights_raw = List[UInt8]()
    for i in range(2, len(resident_data)):
        weights_raw.append(resident_data[i])

    # Eksekusi dequant SIMD (M6) on-demand
    var out_bf16 = dequant_kernel_simd(
        scales, weights_raw, num_elements, group_size
    )

    assert_equal(len(out_bf16), num_elements)
    # Verifikasi dekuantisasi: scale=1.0, q0=2 -> 2.0, q1=3 -> 3.0
    assert_almost_equal(Float32(out_bf16[0]), Float32(2.0), atol=1e-3)
    assert_almost_equal(Float32(out_bf16[1]), Float32(3.0), atol=1e-3)
    assert_almost_equal(Float32(out_bf16[126]), Float32(2.0), atol=1e-3)
    assert_almost_equal(Float32(out_bf16[127]), Float32(3.0), atol=1e-3)


def test_f13_stats_and_rho_b_calculation() raises:
    """Menguji pelacakan metrik F13, kalkulasi rho_B, dan serialisasi RFC 8259 JSON.
    """
    var cfg = LRUCacheConfig(capacity_bytes=5000)
    var cache = LRUCache(cfg)

    # Akses 1: Miss (500 bytes dari disk)
    var p1 = make_dummy_payload(500)
    cache.finish_load(0, 1, p1^)

    # Akses 2: Miss (500 bytes dari disk)
    var p2 = make_dummy_payload(500)
    cache.finish_load(0, 2, p2^)

    # Akses 3: Hit (500 bytes dari RAM)
    _ = cache.begin_access(0, 1)

    # Akses 4: Hit (500 bytes dari RAM)
    _ = cache.begin_access(0, 1)

    assert_equal(cache.stats.total_accesses, 4)
    assert_equal(cache.stats.hits, 2)
    assert_equal(cache.stats.misses, 2)
    assert_almost_equal(cache.stats.hit_rate(), 0.5, atol=1e-5)

    # S_RAM = 1000 bytes, S_disk = 1000 bytes
    assert_equal(cache.stats.ram_bytes, 1000)
    assert_equal(cache.stats.disk_bytes, 1000)
    # rho_B = S_RAM / (S_RAM + S_disk) = 1000 / 2000 = 0.5
    assert_almost_equal(cache.stats.rho_b(), 0.5, atol=1e-5)

    # Verifikasi JSON serialization RFC 8259
    var json_str = cache.stats.to_json("RUN-TEST-001")
    assert_true(json_str.find('"run_id":"RUN-TEST-001"') >= 0)
    assert_true(json_str.find('"hits":2') >= 0)
    assert_true(json_str.find('"misses":2') >= 0)
    assert_true(json_str.find('"rho_b":0.5') >= 0)


def test_f9_hot_experts_pinning_integration() raises:
    """Menguji integrasi daftar hot experts baseline F9 M3 dan penyesuaian rho efektif.
    """
    var hot_list = get_f9_hot_experts()
    # Harus mencakup minimal 12 expert representatif dari baseline F9
    assert_true(len(hot_list) >= 12, "Daftar hot experts F9 harus memadai")

    # Verifikasi expert konsisten 17 ada di Layer 0, 12, 23
    var found_17_l0 = False
    var found_17_l12 = False
    var found_17_l23 = False
    for i in range(len(hot_list)):
        var k = hot_list[i].copy()
        if k.expert_id == 17:
            if k.layer_id == 0:
                found_17_l0 = True
            elif k.layer_id == 12:
                found_17_l12 = True
            elif k.layer_id == 23:
                found_17_l23 = True

    assert_true(found_17_l0, "Expert 17 harus ada di Layer 0")
    assert_true(found_17_l12, "Expert 17 harus ada di Layer 12")
    assert_true(found_17_l23, "Expert 17 harus ada di Layer 23")

    # Evaluasi formula rho efektif terkalibrasi CV
    var rho_eff = compute_effective_rho(0.67, 0.3877)
    # 0.67 * (1.0 + 0.1 * 0.3877) = 0.67 * 1.03877 = ~0.696
    assert_true(rho_eff > 0.67, "rho efektif harus lebih tinggi dari hit rate")


def test_real_model_odirect_lru_integration() raises:
    """Menguji integrasi end-to-end: O_DIRECT reader -> LRU Cache -> SIMD dequant pada model riil.
    """
    var model_path = (
        "/home/will/models/qwen1.5-moe-a2.7b-chat-4bit/quant_model.bin"
    )
    var reader = ODirectReader.discover(
        model_path, requested_block_size=4096, queue_depth=8
    )

    # Baca metadata tensor pertama (di offset 256)
    var meta_len_raw = reader.read_logical_payload(256, 4)
    var meta_len = (
        Int(meta_len_raw[0])
        | (Int(meta_len_raw[1]) << 8)
        | (Int(meta_len_raw[2]) << 16)
        | (Int(meta_len_raw[3]) << 24)
    )
    var meta_json = reader.read_logical_payload(260, meta_len)
    var meta = QuantTensorMetadata.from_json_bytes(meta_json)
    var payload_offset = 260 + meta_len
    var total_payload_bytes = meta.scales_bytes() + meta.weights_bytes()

    # Siapkan LRUCache (kapasitas 800 MB, pin budget 200 MB)
    var cfg = LRUCacheConfig(
        capacity_bytes=800 * 1024 * 1024, pin_budget_ratio=0.25
    )
    var cache = LRUCache(cfg)

    # 1. Akses Pertama: MISS
    var st_miss = cache.begin_access(0, 0)
    assert_equal(st_miss, STATE_ABSENT, "Akses pertama harus MISS")

    # Baca via O_DIRECT reader
    var raw_data = reader.read_logical_payload(
        payload_offset, total_payload_bytes
    )
    assert_equal(len(raw_data), total_payload_bytes)

    # Simpan ke cache sebagai pinned expert
    cache.finish_load(0, 0, raw_data^, is_pinned=True)

    # 2. Akses Kedua: HIT
    var st_hit = cache.begin_access(0, 0)
    assert_equal(st_hit, STATE_RESIDENT, "Akses kedua harus HIT")

    # Ambil data dari cache (RAM)
    var cached_data = cache.get_resident_data(0, 0)
    assert_equal(len(cached_data), total_payload_bytes)

    # Eksekusi dequant SIMD dari cached data
    var scales = List[Float16]()
    for g in range(meta.num_groups):
        var u = UInt16(cached_data[g * 2]) | (
            UInt16(cached_data[g * 2 + 1]) << 8
        )
        scales.append(u16_to_float16(u))

    var weights_raw = List[UInt8]()
    var w_start = meta.scales_bytes()
    for i in range(w_start, len(cached_data)):
        weights_raw.append(cached_data[i])

    var out_bf16 = dequant_kernel_simd(
        scales, weights_raw, meta.num_elements(), meta.group_size
    )
    assert_equal(len(out_bf16), meta.num_elements())

    # 3. Verifikasi Statistik F13
    assert_equal(cache.stats.total_accesses, 2)
    assert_equal(cache.stats.hits, 1)
    assert_equal(cache.stats.misses, 1)
    assert_almost_equal(cache.stats.hit_rate(), 0.5, atol=1e-5)
    assert_equal(cache.stats.disk_bytes, total_payload_bytes)
    assert_equal(cache.stats.ram_bytes, total_payload_bytes)
    assert_almost_equal(cache.stats.rho_b(), 0.5, atol=1e-5)
    assert_equal(cache.stats.pinned_entries, 1)

    reader.close()


def main() raises:
    var suite = TestSuite()
    suite.test[test_memory_budget_accounting_and_validation]()
    suite.test[test_state_machine_and_single_flight]()
    suite.test[test_pure_lru_eviction]()
    suite.test[test_pin_budget_invariant_25_percent]()
    suite.test[test_selective_prefill_bound]()
    suite.test[test_corruption_detection_and_atomic_clear]()
    suite.test[test_quantized_only_content_and_streaming_dequant]()
    suite.test[test_f13_stats_and_rho_b_calculation]()
    suite.test[test_f9_hot_experts_pinning_integration]()
    suite.test[test_real_model_odirect_lru_integration]()
    suite^.run()
