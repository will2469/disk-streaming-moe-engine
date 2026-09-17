# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""LRU Cache Expert Engine, Memory Budget Accounting, Pinning 25%, State Machine & Single-Flight (M7)."""

from cli.m7_errors import m7_error_json
from format.types import json_escape
from std.collections import Dict, List

# State Machine 4-State per Entry
comptime STATE_ABSENT = 0
comptime STATE_LOADING = 1
comptime STATE_RESIDENT = 2
comptime STATE_EVICTING = 3


@fieldwise_init
struct CacheKey(Copyable, Movable):
    """Identifier unik untuk expert: (layer_id, expert_id)."""

    var layer_id: Int
    var expert_id: Int

    def to_string(self) -> String:
        return String(self.layer_id) + ":" + String(self.expert_id)

    def copy(self) -> Self:
        return CacheKey(self.layer_id, self.expert_id)


struct CacheEntry(Movable):

    """Struktur satu entry dalam LRU Cache."""

    var layer_id: Int
    var expert_id: Int
    var state: Int
    var data: List[UInt8]
    var size_bytes: Int
    var timestamp: Int
    var access_count: Int
    var pinned: Bool

    def __init__(
        out self,
        layer_id: Int,
        expert_id: Int,
        state: Int = STATE_ABSENT,
        size_bytes: Int = 0,
        pinned: Bool = False,
    ):
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.state = state
        self.data = List[UInt8]()
        self.size_bytes = size_bytes
        self.timestamp = 0
        self.access_count = 0
        self.pinned = pinned


@fieldwise_init
struct MemoryBudget(Copyable, Movable):
    """Budget memori sistem sesuai kontrak: resident + kv + io + dequant + headroom <= limit.
    """

    var resident_weights_bytes: Int
    var kv_cache_bytes: Int
    var io_buffers_bytes: Int
    var dequant_bytes: Int
    var runtime_headroom_bytes: Int
    var memory_limit_bytes: Int

    def total_allocated(self) -> Int:
        return (
            self.resident_weights_bytes
            + self.kv_cache_bytes
            + self.io_buffers_bytes
            + self.dequant_bytes
            + self.runtime_headroom_bytes
        )

    def validate(self) raises:
        """Memvalidasi kontrak budget memori (fail-fast M7_ERR_LRU_ALLOC jika melanggar).
        """
        var total = self.total_allocated()
        if total > self.memory_limit_bytes:
            raise Error(
                m7_error_json(
                    "M7_ERR_LRU_ALLOC",
                    "lru_cache",
                    "system memory budget exceeded limit: total="
                    + String(total)
                    + " bytes > limit="
                    + String(self.memory_limit_bytes)
                    + " bytes",
                    String('{"total":')
                    + String(total)
                    + ',"limit":'
                    + String(self.memory_limit_bytes)
                    + "}",
                )
            )


struct LRUCacheConfig(Copyable, Movable):
    """Konfigurasi operasional LRU Cache."""

    var capacity_bytes: Int
    var pin_budget_ratio: Float64
    var allow_revalidation: Bool

    def __init__(
        out self,
        capacity_bytes: Int,
        pin_budget_ratio: Float64 = 0.25,
        allow_revalidation: Bool = True,
    ):
        self.capacity_bytes = capacity_bytes
        self.pin_budget_ratio = pin_budget_ratio
        self.allow_revalidation = allow_revalidation

    def copy(self) -> Self:
        return LRUCacheConfig(
            self.capacity_bytes, self.pin_budget_ratio, self.allow_revalidation
        )

    def pin_budget_bytes(self) -> Int:
        """Batas maksimum alokasi untuk pinned experts (default 25% kapasitas cache).
        """
        return Int(Float64(self.capacity_bytes) * self.pin_budget_ratio)


struct LRUCacheStats(Copyable, Movable):
    """Statistik operasional cache dan metrik F13 terstandarisasi."""

    var total_accesses: Int
    var hits: Int
    var misses: Int
    var hit_bytes: Int
    var miss_bytes: Int
    var disk_bytes: Int
    var ram_bytes: Int
    var evictions: Int
    var pinned_entries: Int
    var pinned_bytes: Int
    var current_bytes: Int

    def __init__(out self):
        self.total_accesses = 0
        self.hits = 0
        self.misses = 0
        self.hit_bytes = 0
        self.miss_bytes = 0
        self.disk_bytes = 0
        self.ram_bytes = 0
        self.evictions = 0
        self.pinned_entries = 0
        self.pinned_bytes = 0
        self.current_bytes = 0

    def copy(self) -> Self:
        var res = LRUCacheStats()
        res.total_accesses = self.total_accesses
        res.hits = self.hits
        res.misses = self.misses
        res.hit_bytes = self.hit_bytes
        res.miss_bytes = self.miss_bytes
        res.disk_bytes = self.disk_bytes
        res.ram_bytes = self.ram_bytes
        res.evictions = self.evictions
        res.pinned_entries = self.pinned_entries
        res.pinned_bytes = self.pinned_bytes
        res.current_bytes = self.current_bytes
        return res^

    def hit_rate(self) -> Float64:
        """Menghitung Hit Rate diagnostik: hits / total_accesses."""
        if self.total_accesses == 0:
            return 0.0
        return Float64(self.hits) / Float64(self.total_accesses)

    def rho_b(self) -> Float64:
        """Menghitung rasio byte-level F13: S_RAM / (S_RAM + S_disk)."""
        var total = self.ram_bytes + self.disk_bytes
        if total == 0:
            return 0.0
        return Float64(self.ram_bytes) / Float64(total)

    def to_json(self, run_id: String = "M7-RUN") -> String:
        """Membentuk string JSON statistik strict RFC 8259."""
        return String(
            '{"run_id":"',
            json_escape(run_id),
            '","statistics":{"total_accesses":',
            String(self.total_accesses),
            ',"hits":',
            String(self.hits),
            ',"misses":',
            String(self.misses),
            ',"hit_rate":',
            String(self.hit_rate()),
            ',"hit_bytes":',
            String(self.hit_bytes),
            ',"miss_bytes":',
            String(self.miss_bytes),
            ',"disk_bytes":',
            String(self.disk_bytes),
            ',"ram_bytes":',
            String(self.ram_bytes),
            ',"rho_b":',
            String(self.rho_b()),
            ',"evictions":',
            String(self.evictions),
            ',"pinned_entries":',
            String(self.pinned_entries),
            ',"pinned_bytes":',
            String(self.pinned_bytes),
            ',"current_bytes":',
            String(self.current_bytes),
            "}}",
        )


struct LRUCache:
    """Implementasi LRU Cache Expert dengan Pin Budget 25%, Single-Flight, dan Revalidasi Korupsi.
    """

    var config: LRUCacheConfig
    var current_bytes: Int
    var pinned_bytes: Int
    var logical_clock: Int
    var stats: LRUCacheStats
    var key_to_idx: Dict[String, Int]
    var entries: List[CacheEntry]

    def __init__(
        out self,
        config: LRUCacheConfig,
        budget: MemoryBudget = MemoryBudget(0, 0, 0, 0, 0, 100000000000),
    ) raises:
        # Validasi budget memori sistem terlebih dahulu
        budget.validate()

        self.config = config.copy()
        self.current_bytes = 0
        self.pinned_bytes = 0
        self.logical_clock = 0
        self.stats = LRUCacheStats()
        self.key_to_idx = Dict[String, Int]()
        self.entries = List[CacheEntry]()

    def _make_key(self, layer_id: Int, expert_id: Int) -> String:
        return String(layer_id) + ":" + String(expert_id)

    def begin_access(mut self, layer_id: Int, expert_id: Int) raises -> Int:
        """Memulai akses ke expert (Single-Flight contract).

        Mengembalikan salah satu status:
        - STATE_RESIDENT: Cache HIT. Data tersedia di RAM.
        - STATE_LOADING: Sedang di-load oleh thread/worker lain. Panggil lagi sesudah selesai.
        - STATE_ABSENT: Cache MISS. Pemenang CAS/single-flight; wajib membaca disk lalu panggil finish_load.
        """
        var key = self._make_key(layer_id, expert_id)
        if key in self.key_to_idx:
            var idx = self.key_to_idx[key]
            var st = self.entries[idx].state
            if st == STATE_RESIDENT:
                # HIT! Update timestamp dan statistik
                self.logical_clock += 1
                self.entries[idx].timestamp = self.logical_clock
                self.entries[idx].access_count += 1
                self.stats.hits += 1
                self.stats.total_accesses += 1
                self.stats.hit_bytes += self.entries[idx].size_bytes
                self.stats.ram_bytes += self.entries[idx].size_bytes
                return STATE_RESIDENT
            elif st == STATE_LOADING:
                # Sedang di-fetch oleh single-flight leader
                return STATE_LOADING
            elif st == STATE_EVICTING:
                # Sedang di-evict -> diperlakukan sebagai miss, resolve ulang sesudah evict
                return STATE_ABSENT

        # Entry tidak ada atau status ABSENT -> Menangkan single-flight load
        if key not in self.key_to_idx:
            var new_idx = len(self.entries)
            var new_ent = CacheEntry(
                layer_id=layer_id, expert_id=expert_id, state=STATE_LOADING
            )
            self.entries.append(new_ent^)
            self.key_to_idx[key] = new_idx
        else:
            var idx = self.key_to_idx[key]
            self.entries[idx].state = STATE_LOADING

        return STATE_ABSENT

    def get_resident_data(
        self, layer_id: Int, expert_id: Int
    ) raises -> List[UInt8]:
        """Mengambil quantized bytes dari entry yang berstatus RESIDENT."""
        var key = self._make_key(layer_id, expert_id)
        if key not in self.key_to_idx:
            raise Error(
                m7_error_json(
                    "M7_ERR_LRU_ALLOC",
                    "lru_cache",
                    "entry not found for key: " + key,
                )
            )
        var idx = self.key_to_idx[key]
        if self.entries[idx].state != STATE_RESIDENT:
            raise Error(
                m7_error_json(
                    "M7_ERR_LRU_ALLOC",
                    "lru_cache",
                    "entry not in resident state: "
                    + String(self.entries[idx].state),
                )
            )

        var copy = List[UInt8]()
        copy.reserve(len(self.entries[idx].data))
        for i in range(len(self.entries[idx].data)):
            copy.append(self.entries[idx].data[i])
        return copy^

    def cancel_load(mut self, layer_id: Int, expert_id: Int) raises:
        """Membatalkan in-flight load jika pembacaan disk gagal."""
        var key = self._make_key(layer_id, expert_id)
        if key in self.key_to_idx:
            var idx = self.key_to_idx[key]
            if self.entries[idx].state == STATE_LOADING:
                self.entries[idx].state = STATE_ABSENT

    def validate_structural_integrity(
        mut self, data: List[UInt8]
    ) raises -> Bool:
        """Memeriksa integritas struktural payload.

        Jika terjadi korupsi (misal panjang 0 atau marker korupsi terdeteksi):
        memicu atomic clear() pada seluruh cache lalu fail-fast M7_ERR_LRU_CORRUPT.
        """
        if len(data) == 0:
            self.clear()
            raise Error(
                m7_error_json(
                    "M7_ERR_LRU_CORRUPT",
                    "lru_cache",
                    "structural revalidation failed: empty payload data",
                )
            )

        # Cek tanda korupsi eksplisit (fault injection: jika 4 byte pertama adalah 0xDEADBEEF)
        if len(data) >= 4:
            if (
                data[0] == 0xDE
                and data[1] == 0xAD
                and data[2] == 0xBE
                and data[3] == 0xEF
            ):
                self.clear()
                raise Error(
                    m7_error_json(
                        "M7_ERR_LRU_CORRUPT",
                        "lru_cache",
                        (
                            "structural revalidation detected corrupted payload"
                            " magic (0xDEADBEEF)"
                        ),
                    )
                )

        return True

    def _evict_lru_victim(mut self) raises:
        """Mencari dan mengeksekusi eviksi terhadap entry RESIDENT unpinned dengan timestamp tertua.
        """
        var victim_idx = -1
        var oldest_timestamp = -1

        for i in range(len(self.entries)):
            if (
                self.entries[i].state == STATE_RESIDENT
                and not self.entries[i].pinned
            ):
                if (
                    oldest_timestamp == -1
                    or self.entries[i].timestamp < oldest_timestamp
                ):
                    oldest_timestamp = self.entries[i].timestamp
                    victim_idx = i

        if victim_idx == -1:
            raise Error(
                m7_error_json(
                    "M7_ERR_LRU_NO_VICTIM",
                    "lru_cache",
                    "no evictable unpinned victim found in resident cache",
                )
            )

        # Transisi ke STATE_EVICTING
        self.entries[victim_idx].state = STATE_EVICTING
        var freed_bytes = self.entries[victim_idx].size_bytes
        self.entries[victim_idx].data.clear()
        self.entries[victim_idx].size_bytes = 0

        # Transisi ke STATE_ABSENT
        self.entries[victim_idx].state = STATE_ABSENT
        self.current_bytes -= freed_bytes
        self.stats.evictions += 1
        self.stats.current_bytes = self.current_bytes

    def finish_load(
        mut self,
        layer_id: Int,
        expert_id: Int,
        var data: List[UInt8],
        is_pinned: Bool = False,
    ) raises:
        """Menyelesaikan loading disk: eviksi LRU jika over-capacity, penegakan pin budget 25%, dan insersi data.
        """
        var key = self._make_key(layer_id, expert_id)
        var new_size = len(data)

        # 1. Revalidasi struktural jika diaktifkan
        if self.config.allow_revalidation:
            _ = self.validate_structural_integrity(data)

        # 2. Penegakan Pin Budget Invariant 25%: sum(pinned) <= 25% capacity
        var will_pin = False
        if is_pinned:
            var max_pin_budget = self.config.pin_budget_bytes()
            if self.pinned_bytes + new_size <= max_pin_budget:
                will_pin = True
            else:
                # Over-budget pin DITOLAK: expert tetap unpinned, normal LRU
                print(
                    "WARNING: Pin admission rejected for expert ("
                    + String(layer_id)
                    + ", "
                    + String(expert_id)
                    + "): requested pinned "
                    + String(self.pinned_bytes + new_size)
                    + " bytes exceeds pin_budget "
                    + String(max_pin_budget)
                    + " bytes. Kept unpinned."
                )
                will_pin = False

        # 3. Eviksi LRU loop sampai ruang mencukupi
        while self.current_bytes + new_size > self.config.capacity_bytes:
            self._evict_lru_victim()

        # 4. Simpan data ke entry
        var idx: Int
        if key in self.key_to_idx:
            idx = self.key_to_idx[key]
        else:
            idx = len(self.entries)
            var ent = CacheEntry(layer_id=layer_id, expert_id=expert_id)
            self.entries.append(ent^)
            self.key_to_idx[key] = idx

        self.entries[idx].data = data^
        self.entries[idx].size_bytes = new_size
        self.entries[idx].state = STATE_RESIDENT
        self.entries[idx].pinned = will_pin
        self.logical_clock += 1
        self.entries[idx].timestamp = self.logical_clock
        self.entries[idx].access_count = 1

        self.current_bytes += new_size
        if will_pin:
            self.pinned_bytes += new_size
            self.stats.pinned_entries += 1
            self.stats.pinned_bytes = self.pinned_bytes

        self.stats.current_bytes = self.current_bytes
        self.stats.misses += 1
        self.stats.total_accesses += 1
        self.stats.miss_bytes += new_size
        self.stats.disk_bytes += new_size

    def finish_load_size(
        mut self,
        layer_id: Int,
        expert_id: Int,
        size_bytes: Int,
        is_pinned: Bool = False,
    ) raises:
        """Menyelesaikan loading disk berbasis ukuran byte tanpa menyimpan array fisik payload.
        """
        var key = self._make_key(layer_id, expert_id)
        var new_size = size_bytes

        # 1. Penegakan Pin Budget Invariant 25%: sum(pinned) <= 25% capacity
        var will_pin = False
        if is_pinned:
            var max_pin_budget = self.config.pin_budget_bytes()
            if self.pinned_bytes + new_size <= max_pin_budget:
                will_pin = True

        # 2. Eviksi LRU loop sampai ruang mencukupi
        while self.current_bytes + new_size > self.config.capacity_bytes:
            self._evict_lru_victim()

        # 3. Simpan state ke entry
        var idx: Int
        if key in self.key_to_idx:
            idx = self.key_to_idx[key]
        else:
            idx = len(self.entries)
            var ent = CacheEntry(layer_id=layer_id, expert_id=expert_id)
            self.entries.append(ent^)
            self.key_to_idx[key] = idx

        self.entries[idx].size_bytes = new_size
        self.entries[idx].state = STATE_RESIDENT
        self.entries[idx].pinned = will_pin
        self.logical_clock += 1
        self.entries[idx].timestamp = self.logical_clock
        self.entries[idx].access_count = 1

        self.current_bytes += new_size
        if will_pin:
            self.pinned_bytes += new_size
            self.stats.pinned_entries += 1
            self.stats.pinned_bytes = self.pinned_bytes

        self.stats.current_bytes = self.current_bytes
        self.stats.misses += 1
        self.stats.total_accesses += 1
        self.stats.miss_bytes += new_size
        self.stats.disk_bytes += new_size

    def pin_expert(mut self, layer_id: Int, expert_id: Int) raises -> Bool:
        """Mem-pin expert yang sudah berada di cache jika masih dalam batas pin budget 25%.
        """
        var key = self._make_key(layer_id, expert_id)
        if key not in self.key_to_idx:
            return False

        var idx = self.key_to_idx[key]
        if self.entries[idx].state != STATE_RESIDENT:
            return False

        if self.entries[idx].pinned:
            return True  # Sudah ter-pin

        var sz = self.entries[idx].size_bytes
        var max_pin = self.config.pin_budget_bytes()
        if self.pinned_bytes + sz <= max_pin:
            self.entries[idx].pinned = True
            self.pinned_bytes += sz
            self.stats.pinned_entries += 1
            self.stats.pinned_bytes = self.pinned_bytes
            return True
        else:
            # Over-budget ditolak
            return False

    def selective_prefill(
        mut self,
        pinned_keys: List[CacheKey],
        var payloads: List[List[UInt8]],
    ) raises:
        """Prefill selektif hot experts: total bytes wajib <= pin_budget (fail-fast M7_ERR_LRU_ALLOC jika melanggar).
        """
        if len(pinned_keys) != len(payloads):
            raise Error(
                m7_error_json(
                    "M7_ERR_LRU_ALLOC",
                    "lru_cache",
                    "pinned_keys and payloads length mismatch",
                )
            )

        var total_prefill_bytes = 0
        for i in range(len(payloads)):
            total_prefill_bytes += len(payloads[i])

        var max_pin_budget = self.config.pin_budget_bytes()
        if total_prefill_bytes > max_pin_budget:
            raise Error(
                m7_error_json(
                    "M7_ERR_LRU_ALLOC",
                    "lru_cache",
                    "selective prefill bytes exceeds pin_budget: "
                    + String(total_prefill_bytes)
                    + " bytes > "
                    + String(max_pin_budget)
                    + " bytes (25% of capacity)",
                    String('{"prefill_bytes":')
                    + String(total_prefill_bytes)
                    + ',"pin_budget":'
                    + String(max_pin_budget)
                    + "}",
                )
            )

        # Muat dan pin seluruh key yang diberikan
        for i in range(len(pinned_keys)):
            var k = pinned_keys[i].copy()
            var p = payloads[i].copy()
            self.finish_load(k.layer_id, k.expert_id, p^, is_pinned=True)

    def clear(mut self):
        """Membersihkan seluruh entry cache secara atomik."""
        for i in range(len(self.entries)):
            self.entries[i].data.clear()
            self.entries[i].size_bytes = 0
            self.entries[i].state = STATE_ABSENT
            self.entries[i].pinned = False

        self.key_to_idx.clear()
        self.entries.clear()
        self.current_bytes = 0
        self.pinned_bytes = 0
        self.stats.current_bytes = 0
        self.stats.pinned_bytes = 0
        self.stats.pinned_entries = 0


# ==============================================================================
# Integrasi F9 Baseline Routing
# ==============================================================================


def get_f9_hot_experts() -> List[CacheKey]:
    """Mengembalikan daftar hot experts berdasarkan baseline empiris F9 Milestone M3.

    (reports/2026-09-16/f9_routing_baseline.md).
    Top experts dengan access frequency tertinggi di Layer 0, Layer 12, dan Layer 23,
    serta Expert 17 yang konsisten di top 8 seluruh layer.
    """
    var res = List[CacheKey]()
    # Layer 0 hot experts
    res.append(CacheKey(0, 59))
    res.append(CacheKey(0, 36))
    res.append(CacheKey(0, 3))
    res.append(CacheKey(0, 10))

    # Layer 12 hot experts
    res.append(CacheKey(12, 0))
    res.append(CacheKey(12, 16))
    res.append(CacheKey(12, 43))
    res.append(CacheKey(12, 2))

    # Layer 23 hot experts
    res.append(CacheKey(23, 7))
    res.append(CacheKey(23, 41))
    res.append(CacheKey(23, 15))
    res.append(CacheKey(23, 30))

    # Cross-layer consistent hot expert
    res.append(CacheKey(0, 17))
    res.append(CacheKey(12, 17))
    res.append(CacheKey(23, 17))

    return res^


def compute_effective_rho(hit_rate: Float64, cv: Float64) -> Float64:
    """Menghitung estimasi rasio rho efektif F5 terkalibrasi F9 routing CV.

    rho_eff = hit_rate * (1.0 + 0.1 * cv)
    """
    return hit_rate * (1.0 + 0.1 * cv)
