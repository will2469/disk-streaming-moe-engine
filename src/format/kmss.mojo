# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Serialisasi dan deserialisasi session state kanonis KMSS v1 (M9-W2).

Format berkas KMSS v1:
- Header 128B:
  - [0..4]   : Magic b"KMSS"
  - [4..8]   : Version = 1 (UInt32 LE)
  - [8..12]  : Architecture ID = 2 (UInt32 LE: ARCH_QWEN36_HYBRID_35B)
  - [12..44] : Model manifest hash (32 bytes)
  - [44..48] : seq_len (UInt32 LE)
  - [48..52] : vocab_size (UInt32 LE)
  - [52..56] : kv_layers (UInt32 LE)
  - [56..60] : kv_heads (UInt32 LE)
  - [60..64] : head_dim (UInt32 LE)
  - [64..68] : kv_dtype (UInt32 LE: 1=FP32, 2=BF16)
  - [68..72] : gdn_layers (UInt32 LE)
  - [72..76] : gdn_dv (UInt32 LE)
  - [76..80] : gdn_dk (UInt32 LE)
  - [80..84] : gdn_dtype (UInt32 LE: 1=FP32)
  - [84..92] : kv_bytes (UInt64 LE)
  - [92..100]: gdn_bytes (UInt64 LE)
  - [100..108]: token_bytes (UInt64 LE)
  - [108..128]: reserved padding (20 bytes 0x00)
- Payload:
  - [128 .. 128 + kv_bytes]                       : 5D KV Cache
  - [128 + kv_bytes .. 128 + kv_bytes + gdn_bytes]: 3D GDN States
  - [128 + kv_bytes + gdn_bytes .. payload_end]   : Token IDs prefix (UInt32 LE)
- Trailing 32B: SHA-256 checksum atas header (128B) || payload
"""

from core.config import ModelConfig
from format.sha256 import sha256
from layers.gated_attention import GatedAttnKVCache
from layers.gdn import GDNState
from std.collections import List
from std.ffi import external_call
from std.os import SEEK_END, SEEK_SET


def _pack_u32_le(val: UInt32, mut buf: List[UInt8], offset: Int):
    buf[offset] = UInt8(Int(val & 0xFF))
    buf[offset + 1] = UInt8(Int((val >> UInt32(8)) & 0xFF))
    buf[offset + 2] = UInt8(Int((val >> UInt32(16)) & 0xFF))
    buf[offset + 3] = UInt8(Int((val >> UInt32(24)) & 0xFF))


def _pack_u64_le(val: UInt64, mut buf: List[UInt8], offset: Int):
    for i in range(8):
        var shift = i * 8
        buf[offset + i] = UInt8(Int((val >> UInt64(shift)) & 0xFF))


def _unpack_u32_le(buf: List[UInt8], offset: Int) -> UInt32:
    return (
        UInt32(buf[offset])
        | (UInt32(buf[offset + 1]) << 8)
        | (UInt32(buf[offset + 2]) << 16)
        | (UInt32(buf[offset + 3]) << 24)
    )


def _unpack_u64_le(buf: List[UInt8], offset: Int) -> UInt64:
    var res: UInt64 = 0
    for i in range(8):
        var shift = i * 8
        res |= UInt64(buf[offset + i]) << UInt64(shift)
    return res


@fieldwise_init
struct KmssMetadata(Copyable, Movable):
    var version: UInt32
    var architecture_id: UInt32
    var seq_len: Int
    var vocab_size: Int
    var kv_layers: Int
    var kv_heads: Int
    var head_dim: Int
    var kv_dtype: Int  # 1=FP32, 2=BF16
    var gdn_layers: Int
    var gdn_dv: Int
    var gdn_dk: Int
    var gdn_dtype: Int  # 1=FP32
    var kv_bytes: Int
    var gdn_bytes: Int
    var token_bytes: Int


def write_kmss_v1(
    path: String,
    kv_cache: GatedAttnKVCache,
    gdn_state: GDNState,
    tokens: List[Int],
    cfg: ModelConfig,
    kv_dtype: Int = 1,  # 1 = FP32, 2 = BF16
    manifest_hash: List[UInt8] = List[UInt8](),
) raises:
    """Menyimpan session state KMSS v1 secara atomik (tmp + rename) dengan SHA-256.
    """
    var seq_len = kv_cache.current_len
    if seq_len < len(tokens):
        seq_len = len(tokens)

    var kv_layers = cfg.num_attention_layers()
    var kv_heads = cfg.num_key_value_heads
    var head_dim = cfg.head_dim()
    var b_kv = 4 if kv_dtype == 1 else 2

    var kv_bytes = 2 * seq_len * kv_layers * kv_heads * head_dim * b_kv
    var gdn_layers = gdn_state.layers
    var gdn_dv = gdn_state.dv
    var gdn_dk = gdn_state.dk
    var gdn_bytes = gdn_layers * gdn_dv * gdn_dk * 4
    var token_bytes = seq_len * 4

    var header = List[UInt8]()
    header.resize(128, UInt8(0))

    # Magic "KMSS"
    header[0] = UInt8(0x4B)  # 'K'
    header[1] = UInt8(0x4D)  # 'M'
    header[2] = UInt8(0x53)  # 'S'
    header[3] = UInt8(0x53)  # 'S'

    # Version = 1
    _pack_u32_le(UInt32(1), header, 4)

    # Architecture ID = 2 (ARCH_QWEN36_HYBRID_35B)
    _pack_u32_le(UInt32(2), header, 8)

    # Manifest hash (32 bytes)
    for i in range(32):
        if i < len(manifest_hash):
            header[12 + i] = manifest_hash[i]
        else:
            header[12 + i] = UInt8(0)

    # Dimensions
    _pack_u32_le(UInt32(seq_len), header, 44)
    _pack_u32_le(UInt32(cfg.vocab_size), header, 48)
    _pack_u32_le(UInt32(kv_layers), header, 52)
    _pack_u32_le(UInt32(kv_heads), header, 56)
    _pack_u32_le(UInt32(head_dim), header, 60)
    _pack_u32_le(UInt32(kv_dtype), header, 64)
    _pack_u32_le(UInt32(gdn_layers), header, 68)
    _pack_u32_le(UInt32(gdn_dv), header, 72)
    _pack_u32_le(UInt32(gdn_dk), header, 76)
    _pack_u32_le(UInt32(1), header, 80)  # gdn_dtype = 1 (FP32)

    _pack_u64_le(UInt64(kv_bytes), header, 84)
    _pack_u64_le(UInt64(gdn_bytes), header, 92)
    _pack_u64_le(UInt64(token_bytes), header, 100)

    # 1. Serialisasi KV Cache Payload
    var kv_payload = List[UInt8]()
    kv_payload.resize(kv_bytes, UInt8(0))

    var p_f32_out = kv_payload.unsafe_ptr().unsafe_bitcast[Float32]()
    var p_bf_out = kv_payload.unsafe_ptr().unsafe_bitcast[BFloat16]()
    var elem_idx = 0

    for kv_idx in range(2):
        for pos in range(seq_len):
            for att in range(kv_layers):
                for h in range(kv_heads):
                    for d in range(head_dim):
                        var val: Float32
                        if kv_idx == 0:
                            val = kv_cache.get_k(pos, att, h, d)
                        else:
                            val = kv_cache.get_v(pos, att, h, d)

                        if kv_dtype == 1:
                            p_f32_out[unsafe_offset=elem_idx] = val
                        else:
                            p_bf_out[unsafe_offset=elem_idx] = BFloat16(val)
                        elem_idx += 1

    # 2. Serialisasi GDN States Payload
    var gdn_payload = List[UInt8]()
    gdn_payload.resize(gdn_bytes, UInt8(0))
    var p_gdn_bytes = gdn_state.data.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(gdn_bytes):
        gdn_payload[i] = p_gdn_bytes[unsafe_offset=i]

    # 3. Serialisasi Token IDs Prefix
    var token_payload = List[UInt8]()
    token_payload.resize(token_bytes, UInt8(0))
    for i in range(seq_len):
        var tid = UInt32(0)
        if i < len(tokens):
            tid = UInt32(tokens[i])
        _pack_u32_le(tid, token_payload, i * 4)

    # 4. Gabungkan semua bytes untuk Checksum SHA-256
    var all_bytes = List[UInt8]()
    var total_size_no_hash = 128 + kv_bytes + gdn_bytes + token_bytes
    all_bytes.reserve(total_size_no_hash)

    for i in range(128):
        all_bytes.append(header[i])
    for i in range(kv_bytes):
        all_bytes.append(kv_payload[i])
    for i in range(gdn_bytes):
        all_bytes.append(gdn_payload[i])
    for i in range(token_bytes):
        all_bytes.append(token_payload[i])

    var digest = sha256(all_bytes)

    # 5. Atomic Write (temp file + rename)
    var tmp_path = String(path, ".tmp")
    try:
        var f = open(tmp_path, "w")
        f.write_bytes(Span(header))
        f.write_bytes(Span(kv_payload))
        f.write_bytes(Span(gdn_payload))
        f.write_bytes(Span(token_payload))
        f.write_bytes(Span(digest))
        f.close()
    except:
        raise Error(
            '{"error_type":"OUTPUT_WRITE_FAILED","detail":"cannot write temp'
            " KMSS file: "
            + tmp_path
            + '"}'
        )

    var p_tmp = tmp_path.as_bytes()
    var p_tmp_z = List[UInt8]()
    for i in range(len(p_tmp)):
        p_tmp_z.append(p_tmp[i])
    p_tmp_z.append(0)

    var p_dst = path.as_bytes()
    var p_dst_z = List[UInt8]()
    for i in range(len(p_dst)):
        p_dst_z.append(p_dst[i])
    p_dst_z.append(0)

    var ret = external_call["rename", Int32](
        p_tmp_z.unsafe_ptr(), p_dst_z.unsafe_ptr()
    )
    if ret != 0:
        raise Error(
            '{"error_type":"OUTPUT_WRITE_FAILED","detail":"atomic rename failed'
            " for KMSS: "
            + path
            + '"}'
        )


def read_kmss_v1(
    path: String,
) raises -> Tuple[KmssMetadata, GatedAttnKVCache, GDNState, List[Int]]:
    """Membaca session state KMSS v1 dan memverifikasi integritas checksum SHA-256.
    """
    var f = open(path, "r")
    var total_size = Int(f.seek(0, SEEK_END))
    _ = f.seek(0, SEEK_SET)

    if total_size < 160:
        f.close()
        raise Error(
            '{"error_type":"CORRUPT_SESSION_CHECKSUM","detail":"file too small'
            ' for KMSS v1"}'
        )

    var header = f.read_bytes(128)
    if (
        header[0] != UInt8(0x4B)
        or header[1] != UInt8(0x4D)
        or header[2] != UInt8(0x53)
        or header[3] != UInt8(0x53)
    ):
        f.close()
        raise Error(
            '{"error_type":"FORMAT_MISMATCH","detail":"invalid KMSS magic'
            ' bytes"}'
        )

    var version = _unpack_u32_le(header, 4)
    if version != 1:
        f.close()
        raise Error(
            '{"error_type":"VERSION_MISMATCH","detail":"unsupported KMSS'
            ' version"}'
        )

    var arch_id = _unpack_u32_le(header, 8)
    if arch_id != 2:
        f.close()
        raise Error(
            '{"error_type":"CONFIG_MISMATCH","detail":"unsupported KMSS'
            ' architecture_id, expected 2"}'
        )

    var seq_len = Int(_unpack_u32_le(header, 44))
    var vocab_size = Int(_unpack_u32_le(header, 48))
    var kv_layers = Int(_unpack_u32_le(header, 52))
    var kv_heads = Int(_unpack_u32_le(header, 56))
    var head_dim = Int(_unpack_u32_le(header, 60))
    var kv_dtype = Int(_unpack_u32_le(header, 64))
    var gdn_layers = Int(_unpack_u32_le(header, 68))
    var gdn_dv = Int(_unpack_u32_le(header, 72))
    var gdn_dk = Int(_unpack_u32_le(header, 76))
    var gdn_dtype = Int(_unpack_u32_le(header, 80))

    var kv_bytes = Int(_unpack_u64_le(header, 84))
    var gdn_bytes = Int(_unpack_u64_le(header, 92))
    var token_bytes = Int(_unpack_u64_le(header, 100))

    var b_kv = 4 if kv_dtype == 1 else 2
    var expected_kv_bytes = 2 * seq_len * kv_layers * kv_heads * head_dim * b_kv
    if kv_bytes != expected_kv_bytes:
        f.close()
        raise Error(
            '{"error_type":"LAYOUT_MISMATCH","detail":"header kv_bytes mismatch'
            ' dimensions"}'
        )

    var expected_gdn_bytes = gdn_layers * gdn_dv * gdn_dk * 4
    if gdn_bytes != expected_gdn_bytes:
        f.close()
        raise Error(
            '{"error_type":"LAYOUT_MISMATCH","detail":"header gdn_bytes'
            ' mismatch dimensions"}'
        )

    var expected_token_bytes = seq_len * 4
    if token_bytes != expected_token_bytes:
        f.close()
        raise Error(
            '{"error_type":"LAYOUT_MISMATCH","detail":"header token_bytes'
            ' mismatch dimensions"}'
        )

    var total_payload = kv_bytes + gdn_bytes + token_bytes
    if total_size != 128 + total_payload + 32:
        f.close()
        raise Error(
            '{"error_type":"LAYOUT_MISMATCH","detail":"total file size mismatch'
            ' expected 160 + payload"}'
        )

    var kv_raw = f.read_bytes(kv_bytes)
    var gdn_raw = f.read_bytes(gdn_bytes)
    var token_raw = f.read_bytes(token_bytes)
    var stored_digest = f.read_bytes(32)
    f.close()

    # Verifikasi Checksum SHA-256
    var all_bytes = List[UInt8]()
    all_bytes.reserve(128 + total_payload)
    for i in range(128):
        all_bytes.append(header[i])
    for i in range(kv_bytes):
        all_bytes.append(kv_raw[i])
    for i in range(gdn_bytes):
        all_bytes.append(gdn_raw[i])
    for i in range(token_bytes):
        all_bytes.append(token_raw[i])

    var computed_digest = sha256(all_bytes)
    for i in range(32):
        if computed_digest[i] != stored_digest[i]:
            raise Error(
                '{"error_type":"CORRUPT_SESSION_CHECKSUM","detail":"checksum'
                ' mismatch in KMSS file"}'
            )

    # Deserialisasi Metadata
    var meta = KmssMetadata(
        version,
        arch_id,
        seq_len,
        vocab_size,
        kv_layers,
        kv_heads,
        head_dim,
        kv_dtype,
        gdn_layers,
        gdn_dv,
        gdn_dk,
        gdn_dtype,
        kv_bytes,
        gdn_bytes,
        token_bytes,
    )

    # Deserialisasi KV Cache
    var kv_cache_capacity = seq_len + 128
    if kv_cache_capacity < 512:
        kv_cache_capacity = 512
    var kv_cache = GatedAttnKVCache(
        kv_cache_capacity, kv_layers, kv_heads, head_dim
    )
    var p_f32_src = kv_raw.unsafe_ptr().unsafe_bitcast[Float32]()
    var p_bf_src = kv_raw.unsafe_ptr().unsafe_bitcast[BFloat16]()
    var elem_idx = 0
    for kv_idx in range(2):
        for pos in range(seq_len):
            for att in range(kv_layers):
                for h in range(kv_heads):
                    for d in range(head_dim):
                        var val: Float32
                        if kv_dtype == 1:
                            val = p_f32_src[unsafe_offset=elem_idx]
                        else:
                            val = Float32(p_bf_src[unsafe_offset=elem_idx])
                        elem_idx += 1

                        if kv_idx == 0:
                            kv_cache.set_k(pos, att, h, d, val)
                        else:
                            kv_cache.set_v(pos, att, h, d, val)
    kv_cache.current_len = seq_len

    # Deserialisasi GDN State
    var gdn_state = GDNState(gdn_layers, gdn_dv, gdn_dk)
    var p_dst_gdn = gdn_state.data.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(gdn_bytes):
        p_dst_gdn[unsafe_offset=i] = gdn_raw[i]

    # Deserialisasi Token IDs
    var tokens = List[Int]()
    tokens.reserve(seq_len)
    for i in range(seq_len):
        var tid = Int(_unpack_u32_le(token_raw, i * 4))
        tokens.append(tid)

    return (meta.copy(), kv_cache^, gdn_state^, tokens^)
