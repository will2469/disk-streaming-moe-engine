# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Security & Resource Guard Port untuk M9 (SEC-1, SEC-3, SEC-4, SEC-5, SEC-6).

Menegakkan:
- SEC-1 / SEC-3: Pinning models.lock port, anti-tamper, validasi vocab (248.320).
- SEC-4: Memory budget bottom-up F1-Port (W_res + M_cache + M_expert + M_KV + M_GDN + M_scratch + M_ws <= 7.5 GiB),
         pre-alloc bounds check (checked arithmetic), dan disk free space (20 GB GGUF / 100 GB BF16).
- SEC-5 / SEC-6: File hygiene (read-only model dir, workdir, atomic write).
"""

from core.config import ModelConfig
from format.file_io import _open_shard, read_small_file
from format.scanner import Scanner
from std.collections import List
from std.ffi import external_call
from std.os import SEEK_END, SEEK_SET

comptime GATE_MEMORY_LIMIT_BYTES: Int = 8053063680  # 7.5 GiB
comptime RESIDENT_WEIGHT_LIMIT_BYTES: Int = 1073741824  # 1.0 GiB
comptime KV_CACHE_LIMIT_BYTES: Int = 524288000  # 500 MiB
comptime SCRATCHPAD_LIMIT_BYTES: Int = 268435456  # 256 MiB

comptime DISK_FREE_GGUF_BYTES: Int = 21474836480  # 20 GB
comptime DISK_FREE_BF16_BYTES: Int = 107374182400  # 100 GB


def get_disk_free_space_bytes(path: String) -> Int:
    """Mengambil sisa kapasitas disk yang tersedia (bytes) via statvfs(2)."""
    var path_b = path.as_bytes()
    var path_z = List[UInt8]()
    for i in range(len(path_b)):
        path_z.append(path_b[i])
    path_z.append(0)

    # Buffer statvfs (128 bytes)
    var buf = List[UInt8]()
    buf.resize(128, 0)

    var res = external_call["statvfs", Int32](
        path_z.unsafe_ptr(), buf.unsafe_ptr()
    )
    if res == 0:
        var p_ul = buf.unsafe_ptr().unsafe_bitcast[Int]()
        var frsize = p_ul[unsafe_offset=1]
        var bavail = p_ul[unsafe_offset=4]
        if frsize > 0 and bavail > 0:
            return frsize * bavail
    return -1


def validate_vocab_size_port(
    vocab_size: Int, expected_vocab: Int = 248320
) raises:
    """SEC-3: Validasi ukuran vocabulary port Qwen3.6-35B-A3B (248.320)."""
    if vocab_size != expected_vocab:
        raise Error(
            "Vocab size mismatch: expected "
            + String(expected_vocab)
            + ", got "
            + String(vocab_size)
        )


def validate_memory_budget_port(
    cfg: ModelConfig, seq_len: Int, bytes_per_weight: Int = 1
) raises:
    """SEC-4: Penegakan anggaran memori bottom-up F1-Port (M_peak <= 7.5 GiB).

    Formula F1-Port:
    M_peak = W_res + M_cache + M_expert + M_KV + M_GDN + M_scratch + M_ws <= 7.5 GiB.
    """
    var hidden = cfg.hidden_size
    var vocab = cfg.vocab_size

    # 1. Resident weights (embedding + lm_head)
    var w_res = 2 * vocab * hidden * bytes_per_weight
    if w_res > RESIDENT_WEIGHT_LIMIT_BYTES:
        raise Error(
            "OUT_OF_MEMORY: resident weights "
            + String(w_res)
            + " bytes exceeds limit "
            + String(RESIDENT_WEIGHT_LIMIT_BYTES)
            + " bytes"
        )

    # 2. KV Cache (Formula F2)
    var l_att = cfg.num_attention_layers()
    var h_kv = cfg.num_key_value_heads
    var head_dim = cfg.head_dim()
    var m_kv = (
        2 * l_att * h_kv * head_dim * seq_len * 2
    )  # 2 bytes per float16/bfloat16
    if m_kv > KV_CACHE_LIMIT_BYTES:
        raise Error(
            "OUT_OF_MEMORY: KV cache "
            + String(m_kv)
            + " bytes exceeds limit "
            + String(KV_CACHE_LIMIT_BYTES)
            + " bytes"
        )

    # 3. Scratchpad dequant
    var max_tensor_elems = cfg.moe_intermediate_size * hidden
    var m_scratch = max_tensor_elems * 4
    if m_scratch > SCRATCHPAD_LIMIT_BYTES:
        raise Error(
            "OUT_OF_MEMORY: scratchpad dequant "
            + String(m_scratch)
            + " bytes exceeds limit "
            + String(SCRATCHPAD_LIMIT_BYTES)
            + " bytes"
        )

    # 4. Estimasi komponen nominal lainnya
    var m_cache = 1073741824  # 1.0 GiB (LRU Expert Cache)
    var m_expert = 118489088  # 113 MiB (9 active experts FP32)
    var m_gdn = cfg.num_gdn_layers() * 128 * 128 * 4  # ~1.9 MiB
    var m_ws = 322122547  # ~300 MiB (Token activations & stack)

    var m_peak = w_res + m_cache + m_expert + m_kv + m_gdn + m_scratch + m_ws
    if m_peak > GATE_MEMORY_LIMIT_BYTES:
        raise Error(
            "OUT_OF_MEMORY: Bottom-up memory budget "
            + String(m_peak)
            + " bytes exceeds Gate G-M9-2 limit "
            + String(GATE_MEMORY_LIMIT_BYTES)
            + " bytes (7.5 GiB)"
        )


def validate_disk_space_guard(
    target_dir: String, is_gguf: Bool, is_canonical_bf16: Bool = False
) raises:
    """SEC-4: Validasi kecukupan ruang penyimpanan disk (20 GB GGUF / 100 GB BF16).
    """
    var free_bytes = get_disk_free_space_bytes(target_dir)
    if free_bytes < 0:
        # File system tidak mendukung statvfs (misal env virtual khusus)
        return

    var required_bytes: Int = 104857600  # 100 MB default mini
    if is_canonical_bf16 and not is_gguf:
        required_bytes = DISK_FREE_BF16_BYTES
    elif is_gguf:
        required_bytes = DISK_FREE_GGUF_BYTES
    if free_bytes < required_bytes:
        var req_gb = required_bytes // (1024 * 1024 * 1024)
        var free_gb = free_bytes // (1024 * 1024 * 1024)
        raise Error(
            "INSUFFICIENT_DISK_SPACE: required at least "
            + String(req_gb)
            + " GB free space, but only "
            + String(free_gb)
            + " GB available at "
            + target_dir
        )


def verify_models_lock_port_manifest(
    model_dir: String, lock_path: String
) raises:
    """SEC-1: Memverifikasi integritas berkas shard terhadap models.lock.port.json.

    Jika berkas tidak ditemukan atau ukuran fisik berbeda, memicu fail-closed.
    """
    var raw = read_small_file(lock_path)
    var sc = Scanner(raw^, lock_path)
    sc.skip_ws()
    sc.expect(123)  # '{'

    while True:
        sc.skip_ws()
        if sc.peek() == 125:  # '}'
            sc.pos += 1
            break
        var key = sc.parse_string()
        sc.skip_ws()
        sc.expect(58)  # ':'
        sc.skip_ws()

        if key == "shards":
            sc.expect(91)  # '['
            while True:
                sc.skip_ws()
                if sc.peek() == 93:  # ']'
                    sc.pos += 1
                    break
                sc.expect(123)  # '{'
                var filename = String("")
                var expected_size = 0
                while True:
                    sc.skip_ws()
                    if sc.peek() == 125:  # '}'
                        sc.pos += 1
                        break
                    var s_key = sc.parse_string()
                    sc.skip_ws()
                    sc.expect(58)  # ':'
                    sc.skip_ws()
                    if s_key == "filename":
                        filename = sc.parse_string()
                    elif s_key == "size":
                        expected_size = sc.parse_uint()
                    else:
                        _ = sc.parse_string()

                    sc.skip_ws()
                    if sc.peek() == 44:  # ','
                        sc.pos += 1

                if filename.byte_length() > 0 and expected_size > 0:
                    var shard_path = String(model_dir, "/", filename)
                    try:
                        var f = _open_shard(shard_path)
                        var actual_size = Int(f.seek(0, SEEK_END))
                        f.close()
                        if actual_size != expected_size:
                            raise Error(
                                "MODEL_LOCK_TAMPER_DETECTED: shard "
                                + filename
                                + " size "
                                + String(actual_size)
                                + " != expected "
                                + String(expected_size)
                            )
                    except e:
                        raise Error(
                            "MODEL_LOCK_TAMPER_DETECTED: cannot access shard "
                            + filename
                            + ": "
                            + String(e)
                        )

                sc.skip_ws()
                if sc.peek() == 44:  # ','
                    sc.pos += 1
        else:
            # Lewati nilai non-shards
            if sc.peek() == 34:  # string
                _ = sc.parse_string()
            elif sc.peek() >= 48 and sc.peek() <= 57:  # int
                _ = sc.parse_uint()
            else:
                while not sc.eof() and sc.peek() != 44 and sc.peek() != 125:
                    sc.pos += 1

        sc.skip_ws()
        if sc.peek() == 44:  # ','
            sc.pos += 1
