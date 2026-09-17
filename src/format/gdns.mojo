# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Serialisasi dan deserialisasi format framed binary GDNS v1 kanonis (M8)."""

from format.sha256 import sha256
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


def write_gdns_v1(
    path: String,
    state: GDNState,
    manifest_hash: List[UInt8] = List[UInt8](),
) raises:
    """Menuliskan GDNState ke format biner framed GDNS v1 secara atomic (temp file + rename).

    Struktur berkas:
    1. Header 128B (magic, version, arch_id, dtype, layers, dv, dk, state_bytes, manifest_hash)
    2. Payload state FP32 row-major (layers * dv * dk * 4 bytes)
    3. Trailing SHA-256 checksum (32 bytes) dari header || payload
    """
    var state_bytes = state.layers * state.dv * state.dk * 4
    var header = List[UInt8]()
    header.resize(128, UInt8(0))

    # Magic "GDNS"
    header[0] = UInt8(0x47)
    header[1] = UInt8(0x44)
    header[2] = UInt8(0x4E)
    header[3] = UInt8(0x53)

    # Version = 1 (UInt32 LE)
    _pack_u32_le(UInt32(1), header, 4)

    # Architecture ID = 1 (ARCH_QWEN_GDN)
    _pack_u32_le(UInt32(1), header, 8)

    # dtype = 1 (FP32)
    _pack_u32_le(UInt32(1), header, 12)

    # Dims: layers, dv, dk
    _pack_u32_le(UInt32(state.layers), header, 16)
    _pack_u32_le(UInt32(state.dv), header, 20)
    _pack_u32_le(UInt32(state.dk), header, 24)

    # Reserved1 = 0
    _pack_u32_le(UInt32(0), header, 28)

    # Total state_bytes (UInt64 LE)
    _pack_u64_le(UInt64(state_bytes), header, 32)

    # Model manifest hash (32 bytes)
    for i in range(32):
        if i < len(manifest_hash):
            header[40 + i] = manifest_hash[i]
        else:
            header[40 + i] = UInt8(0)

    # Salin float payload ke raw bytes LE
    var payload_bytes = List[UInt8]()
    payload_bytes.resize(state_bytes, UInt8(0))
    var p_bytes = state.data.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(state_bytes):
        payload_bytes[i] = p_bytes[unsafe_offset=i]

    # Hitung SHA-256 checksum: SHA-256(header || payload)
    var all_bytes = List[UInt8]()
    all_bytes.reserve(128 + state_bytes)
    for i in range(128):
        all_bytes.append(header[i])
    for i in range(state_bytes):
        all_bytes.append(payload_bytes[i])

    var digest = sha256(all_bytes)

    # Tulis ke temporary file lalu rename (atomic write)
    var tmp_path = String(path, ".tmp")
    try:
        var f = open(tmp_path, "w")
        f.write_bytes(Span(header))
        f.write_bytes(Span(payload_bytes))
        f.write_bytes(Span(digest))
        f.close()
    except:
        raise Error(
            '{"error_type":"OUTPUT_WRITE_FAILED","detail":"cannot write temp'
            " GDNS file: "
            + tmp_path
            + '"}'
        )

    # C_rename
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
            " to: "
            + path
            + '"}'
        )


def read_gdns_v1(path: String) raises -> GDNState:
    """Membaca berkas GDNS v1 dan memverifikasi integritas checksum SHA-256."""
    var f = open(path, "r")
    var total_size = Int(f.seek(0, SEEK_END))
    _ = f.seek(0, SEEK_SET)

    if total_size < 160:
        f.close()
        raise Error(
            '{"error_type":"CORRUPT_STATE_CHECKSUM","detail":"file too small'
            ' for GDNS v1"}'
        )

    var header = f.read_bytes(128)
    if (
        header[0] != UInt8(0x47)
        or header[1] != UInt8(0x44)
        or header[2] != UInt8(0x4E)
        or header[3] != UInt8(0x53)
    ):
        f.close()
        raise Error(
            '{"error_type":"LAYOUT_MISMATCH","detail":"invalid GDNS magic'
            ' bytes"}'
        )

    var version = _unpack_u32_le(header, 4)
    if version != 1:
        f.close()
        raise Error(
            '{"error_type":"LAYOUT_MISMATCH","detail":"unsupported GDNS'
            ' version"}'
        )

    var layers = Int(_unpack_u32_le(header, 16))
    var dv = Int(_unpack_u32_le(header, 20))
    var dk = Int(_unpack_u32_le(header, 24))
    var state_bytes = Int(_unpack_u64_le(header, 32))

    var expected_payload = layers * dv * dk * 4
    if state_bytes != expected_payload:
        f.close()
        raise Error(
            '{"error_type":"LAYOUT_MISMATCH","detail":"header state_bytes'
            ' mismatch dimensions"}'
        )

    if total_size != 160 + state_bytes:
        f.close()
        raise Error(
            '{"error_type":"LAYOUT_MISMATCH","detail":"file size mismatch'
            ' expected 160 + state_bytes"}'
        )

    var payload = f.read_bytes(state_bytes)
    var stored_digest = f.read_bytes(32)
    f.close()

    # Verifikasi checksum SHA-256
    var all_bytes = List[UInt8]()
    all_bytes.reserve(128 + state_bytes)
    for i in range(128):
        all_bytes.append(header[i])
    for i in range(state_bytes):
        all_bytes.append(payload[i])

    var computed_digest = sha256(all_bytes)
    for i in range(32):
        if computed_digest[i] != stored_digest[i]:
            raise Error(
                '{"error_type":"CORRUPT_STATE_CHECKSUM","detail":"checksum'
                ' mismatch in GDNS file"}'
            )

    # Ekstrak Float32 payload
    var state = GDNState(layers, dv, dk)
    var p_dst_bytes = state.data.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(state_bytes):
        p_dst_bytes[unsafe_offset=i] = payload[i]

    return state^
