# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""KMSS v1 Longest-Prefix Cache & Canonical Rebase Engine (M12-W2a).

Mengimplementasikan session state continuity multi-turn sesuai M12 §2.2 (Gate G-M12-2):
1. Longest-prefix lookup: mencari entri dengan P terpanjang sehingga P = T[0..|P|]
   secara token-per-token eksak.
2. Domain separation: hash (model_id, tokenizer_pin, template_subset_rev) hanya sebagai
   indeks partisi; perbandingan token eksak adalah satu-satunya bukti reuse.
3. Penanganan divergensi: regenerasi / edit giliran lama me-reuse common prefix dan
   hanya men-delta-prefill dari titik divergensi.
4. Bounded LRU cache: kapasitas berbatas; eviksi entri tertua (least-recently-used)
   hanya mempengaruhi latensi, tidak pernah merusak kebenaran.
5. Insertion policy: HANYA dimasukkan saat penyelesaian bersih (clean completion: stop/length).
   Generasi ter-abort, error, atau timeout tidak memasukkan apa pun.
6. Canonical rebase: teacher-forced prefill atas representasi kanonis (assistant polos,
   tanpa tag think struktural) sebelum dimasukkan ke cache.
7. Paritas greedy: output cache-hit ter-gate bit-exact identik vs full-prefill pada temperature=0.
"""

from core.config import ModelConfig
from format.kmss import KmssMetadata, read_kmss_v1, write_kmss_v1
from format.sha256 import sha256
from layers.gated_attention import GatedAttnKVCache
from layers.gdn import GDNState
from std.collections import List
from std.ffi import external_call
from std.time import perf_counter_ns


def _sha256_hex_lower(data: List[UInt8]) -> String:
    """Menghitung digest SHA-256 dalam representasi 64-karakter lowercase hex.
    """
    var digest = sha256(data)
    var hex_bytes = List[UInt8]()
    hex_bytes.reserve(64)
    for i in range(32):
        var b = Int(digest[i])
        var hi = (b >> 4) & 0x0F
        var lo = b & 0x0F
        hex_bytes.append(UInt8(48 + hi if hi < 10 else 87 + hi))
        hex_bytes.append(UInt8(48 + lo if lo < 10 else 87 + lo))
    return String(from_utf8_lossy=Span(hex_bytes))


def compute_domain_key(
    model_id: String, tokenizer_pin: String, template_subset_rev: String
) -> String:
    """Menghitung kunci domain terpartisi (model_id, tokenizer_pin, template_subset_rev).
    """
    var s = model_id + ":" + tokenizer_pin + ":" + template_subset_rev
    var b = s.as_bytes()
    var input_bytes = List[UInt8]()
    input_bytes.reserve(len(b))
    for i in range(len(b)):
        input_bytes.append(b[i])
    return _sha256_hex_lower(input_bytes)


def domain_key_to_bytes(domain_key: String) -> List[UInt8]:
    """Mengonversi kunci domain hex menjadi 32 bytes binary."""
    var out = List[UInt8]()
    out.reserve(32)
    var b = domain_key.as_bytes()
    for i in range(32):
        if (i * 2 + 1) < len(b):
            var c1 = Int(b[i * 2])
            var c2 = Int(b[i * 2 + 1])
            var v1 = c1 - 48 if c1 < 58 else (c1 - 87 if c1 >= 97 else c1 - 55)
            var v2 = c2 - 48 if c2 < 58 else (c2 - 87 if c2 >= 97 else c2 - 55)
            out.append(UInt8((v1 << 4) | (v2 & 0x0F)))
        else:
            out.append(UInt8(0))
    return out^


@fieldwise_init
struct PrefixCacheEntry(Copyable, Movable):
    """Entri tunggal dalam KMSS Prefix Cache."""

    var entry_id: String
    var domain_key: String
    var tokens: List[Int]
    var kv_cache: GatedAttnKVCache
    var gdn_state: GDNState
    var last_accessed_step: Int
    var is_valid: Bool


@fieldwise_init
struct PrefixLookupResult(Copyable, Movable):
    """Hasil resolusi request terhadap prefix cache."""

    var hit: Bool
    var prefix_len: Int
    var entry_index: Int
    var delta_tokens_len: Int
    var domain_key: String


struct PrefixCache:
    """Bounded LRU Cache untuk session state KMSS v1."""

    var capacity: Int
    var entries: List[PrefixCacheEntry]
    var access_clock: Int

    def __init__(out self, capacity: Int = 8):
        self.capacity = capacity
        self.entries = List[PrefixCacheEntry]()
        self.access_clock = 0

    def size(self) -> Int:
        """Mengembalikan jumlah entri valid yang tersimpan."""
        var count = 0
        for i in range(len(self.entries)):
            if self.entries[i].is_valid:
                count += 1
        return count

    def clear(mut self):
        """Mereset seluruh isi cache."""
        self.entries.clear()
        self.access_clock = 0

    def lookup(
        mut self, domain_key: String, req_tokens: List[Int]
    ) -> PrefixLookupResult:
        """Mencari entri cache dengan prefiks P terpanjang yang sama token-per-token dengan req_tokens.

        Aturan:
        1. Domain key wajib cocok (partisi model/tokenizer/template).
        2. len(P) <= len(req_tokens) dan P == req_tokens[0..|P|] eksak.
        3. Memilih entri dengan |P| maksimal.
        4. Mengupdate last_accessed_step untuk LRU.
        """
        var req_len = len(req_tokens)
        if req_len == 0:
            return PrefixLookupResult(
                hit=False,
                prefix_len=0,
                entry_index=-1,
                delta_tokens_len=0,
                domain_key=domain_key,
            )

        var best_idx = -1
        var max_prefix_len = 0

        for i in range(len(self.entries)):
            ref entry = self.entries[i]
            if not entry.is_valid:
                continue

            # Domain Key check (index partisi model/tokenizer/template)
            if entry.domain_key != domain_key:
                continue

            var entry_tok_len = len(entry.tokens)
            if entry_tok_len == 0 or entry_tok_len > req_len:
                continue

            # Verifikasi token-per-token eksak: P == req_tokens[0..|P|]
            var is_exact_prefix = True
            for k in range(entry_tok_len):
                if entry.tokens[k] != req_tokens[k]:
                    is_exact_prefix = False
                    break

            if is_exact_prefix and entry_tok_len > max_prefix_len:
                max_prefix_len = entry_tok_len
                best_idx = i

        if best_idx >= 0 and max_prefix_len > 0:
            self.access_clock += 1
            self.entries[best_idx].last_accessed_step = self.access_clock
            return PrefixLookupResult(
                hit=True,
                prefix_len=max_prefix_len,
                entry_index=best_idx,
                delta_tokens_len=req_len - max_prefix_len,
                domain_key=domain_key,
            )

        return PrefixLookupResult(
            hit=False,
            prefix_len=0,
            entry_index=-1,
            delta_tokens_len=req_len,
            domain_key=domain_key,
        )

    def insert(
        mut self,
        domain_key: String,
        canonical_tokens: List[Int],
        kv_cache: GatedAttnKVCache,
        gdn_state: GDNState,
        finish_reason: String,
    ) -> Bool:
        """Memasukkan state kanonis ke dalam cache HANYA saat penyelesaian bersih.

        Aturan:
        1. Insert HANYA pada finish_reason == 'stop' atau 'length'.
        2. Abort/error/timeout insert NOTHING.
        3. Jika kapasitas penuh, eviksi LRU victim (last_accessed_step terlama).
        """
        if finish_reason != "stop" and finish_reason != "length":
            return False

        if len(canonical_tokens) == 0:
            return False

        self.access_clock += 1

        # Cek apakah entri dengan domain dan token identik sudah ada -> update
        for i in range(len(self.entries)):
            ref e = self.entries[i]
            if e.is_valid and e.domain_key == domain_key:
                if len(e.tokens) == len(canonical_tokens):
                    var is_matched = True
                    for k in range(len(canonical_tokens)):
                        if e.tokens[k] != canonical_tokens[k]:
                            is_matched = False
                            break
                    if is_matched:
                        e.kv_cache = kv_cache.copy()
                        e.gdn_state = gdn_state.copy()
                        e.last_accessed_step = self.access_clock
                        return True

        # Jika belum ada dan kapasitas penuh -> eviksi LRU victim
        if len(self.entries) >= self.capacity:
            var lru_idx = 0
            var oldest_step = self.entries[0].last_accessed_step
            for i in range(1, len(self.entries)):
                if self.entries[i].last_accessed_step < oldest_step:
                    oldest_step = self.entries[i].last_accessed_step
                    lru_idx = i

            var eid = String("entry_", self.access_clock)
            self.entries[lru_idx] = PrefixCacheEntry(
                entry_id=eid,
                domain_key=domain_key,
                tokens=canonical_tokens.copy(),
                kv_cache=kv_cache.copy(),
                gdn_state=gdn_state.copy(),
                last_accessed_step=self.access_clock,
                is_valid=True,
            )
            return True

        # Tambahkan entri baru jika kapasitas masih tersedia
        var eid = String("entry_", self.access_clock)
        self.entries.append(
            PrefixCacheEntry(
                entry_id=eid,
                domain_key=domain_key,
                tokens=canonical_tokens.copy(),
                kv_cache=kv_cache.copy(),
                gdn_state=gdn_state.copy(),
                last_accessed_step=self.access_clock,
                is_valid=True,
            )
        )
        return True

    def save_to_dir(self, cache_dir: String, cfg: ModelConfig) raises:
        """Menyimpan seluruh entri cache yang valid ke direktori sebagai file KMSS v1 biner.
        """
        var manifest_path = String(cache_dir, "/manifest.txt")
        var manifest_content = String("")

        for i in range(len(self.entries)):
            ref e = self.entries[i]
            if not e.is_valid:
                continue
            var kmss_name = String(e.entry_id, ".kmss")
            var kmss_path = String(cache_dir, "/", kmss_name)
            var d_bytes = domain_key_to_bytes(e.domain_key)
            write_kmss_v1(
                kmss_path,
                e.kv_cache,
                e.gdn_state,
                e.tokens,
                cfg,
                manifest_hash=d_bytes,
            )
            manifest_content += (
                e.entry_id
                + " "
                + String(e.last_accessed_step)
                + " "
                + e.domain_key
                + " "
                + kmss_name
                + "\n"
            )

        var f_man = open(manifest_path, "w")
        f_man.write(manifest_content)
        f_man.close()

    def load_from_dir(mut self, cache_dir: String) raises:
        """Memuat entri cache yang tersimpan pada direktori."""
        var manifest_path = String(cache_dir, "/manifest.txt")
        var f_man: FileHandle
        try:
            f_man = open(manifest_path, "r")
        except:
            return

        var raw_text = f_man.read()
        f_man.close()

        self.entries.clear()
        var lines = raw_text.split("\n")
        for idx in range(len(lines)):
            var line = lines[idx].strip()
            if line.byte_length() == 0:
                continue
            var parts = line.split(" ")
            if len(parts) < 4:
                continue
            var entry_id = String(parts[0])
            var step_val = 0
            try:
                step_val = Int(String(parts[1]))
            except:
                pass
            var domain_key = String(parts[2])
            var kmss_name = String(parts[3])
            var kmss_path = String(cache_dir, "/", kmss_name)

            try:
                var res = read_kmss_v1(kmss_path)
                var restored_kv = res[1].copy()
                var restored_gdn = res[2].copy()
                var restored_tokens = res[3].copy()
                self.entries.append(
                    PrefixCacheEntry(
                        entry_id=entry_id,
                        domain_key=domain_key,
                        tokens=restored_tokens^,
                        kv_cache=restored_kv^,
                        gdn_state=restored_gdn^,
                        last_accessed_step=step_val,
                        is_valid=True,
                    )
                )
                if step_val > self.access_clock:
                    self.access_clock = step_val
            except:
                # Lewati entri yang rusak atau tidak dapat dibaca
                pass
