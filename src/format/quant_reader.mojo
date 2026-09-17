# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Reader berkas kuantisasi 4-bit M6 dengan pengerasan parser SEC-4.

Invariants SEC-4 (docs/milestones/M6-quantizer.md § Parser Hardening):
- num_tensors <= CONFIGURED_MAX_TENSORS (100.000)
- name_len <= CONFIGURED_MAX_NAME (512), ndim <= CONFIGURED_MAX_NDIM (8)
- num_elements dihitung via aritmetika checked anti-overflow (safe_multiply_int)
- num_groups == N / G eksak (N % G == 0), checked division
- scales_bytes == num_groups * 2, weights_bytes == N / 2 eksak
- Region antar-record berurutan dan strictly non-overlapping (menutup aliasing)
- file_size == header.total_bytes == 256 + sum(record_lengths) eksak
"""

from format.quant_format import (
    CONFIGURED_MAX_NAME,
    CONFIGURED_MAX_NDIM,
    CONFIGURED_MAX_TENSORS,
    QUANT_HEADER_SIZE,
    QuantHeader,
    QuantTensorMetadata,
    float16_to_u16,
    is_allowed_group_size,
    safe_multiply_int,
    u16_to_float16,
    validate_quant_header,
    validate_tensor_meta,
)
from quant.dequant_kernel import dequant_kernel_simd, dequant_kernel_simd_f32
from std.collections import Dict, List
from std.os import SEEK_END, SEEK_SET


@fieldwise_init
struct QuantTensorEntry(Copyable, Movable):
    """Informasi lokasi tensor terkuantisasi di dalam file berkas .kimo.bin."""

    var meta: QuantTensorMetadata
    var record_offset: Int
    var payload_offset: Int
    var scales_offset: Int
    var data_offset: Int
    var scales_bytes: Int
    var weights_bytes: Int
    var total_record_bytes: Int


@fieldwise_init
struct QuantModelIndex(Movable):
    """Indeks pencarian tensor dalam model quant biner dengan validasi SEC-4."""

    var header: QuantHeader
    var entries: List[QuantTensorEntry]
    var tensor_map: Dict[String, Int]
    var file_path: String


def scan_quant_file(file_path: String) raises -> QuantModelIndex:
    """Memindai berkas biner kuantisasi dan memvalidasi seluruh invarian SEC-4.

    Header dipindai terlebih dahulu, diikuti validasi seluruh rentang record
    tanpa tumpang tindih (non-overlapping) dan penegakan batas ukuran eksak.
    """
    var f = open(file_path, "r")
    var file_size = Int(f.seek(0, SEEK_END))
    if file_size < QUANT_HEADER_SIZE:
        f.close()
        raise Error(
            "scan_quant_file: file truncated, size "
            + String(file_size)
            + " < header size "
            + String(QUANT_HEADER_SIZE)
        )

    _ = f.seek(0, SEEK_SET)
    var hdr_bytes = f.read_bytes(QUANT_HEADER_SIZE)
    var header: QuantHeader
    try:
        header = QuantHeader.from_bytes(hdr_bytes)
    except e:
        f.close()
        raise Error(
            "scan_quant_file: failed parsing quant header: " + String(e)
        )

    try:
        validate_quant_header(header, file_size)
    except e:
        f.close()
        raise Error(
            "scan_quant_file: header validation failed (SEC-4): " + String(e)
        )

    var entries = List[QuantTensorEntry]()
    var tensor_map = Dict[String, Int]()
    var cur_off = QUANT_HEADER_SIZE

    for t_idx in range(header.num_tensors):
        if cur_off + 4 > file_size:
            f.close()
            raise Error(
                "scan_quant_file: framing truncated before meta_len at tensor "
                + String(t_idx)
            )

        _ = f.seek(cur_off, SEEK_SET)
        var len_bytes = f.read_bytes(4)
        var meta_len = (
            Int(len_bytes[0])
            | (Int(len_bytes[1]) << 8)
            | (Int(len_bytes[2]) << 16)
            | (Int(len_bytes[3]) << 24)
        )

        if meta_len <= 0 or cur_off + 4 + meta_len > file_size:
            f.close()
            raise Error(
                "scan_quant_file: invalid meta_len ("
                + String(meta_len)
                + ") at offset "
                + String(cur_off)
            )

        var json_bytes = f.read_bytes(meta_len)
        var meta: QuantTensorMetadata
        try:
            meta = QuantTensorMetadata.from_json_bytes(json_bytes)
        except e:
            f.close()
            raise Error(
                "scan_quant_file: failed parsing tensor metadata JSON at"
                " tensor "
                + String(t_idx)
                + ": "
                + String(e)
            )

        try:
            validate_tensor_meta(meta)
        except e:
            f.close()
            raise Error(
                "scan_quant_file: tensor metadata failed SEC-4 validation: "
                + String(e)
            )

        var s_bytes = meta.scales_bytes()
        var w_bytes = meta.weights_bytes()
        var payload_bytes = s_bytes + w_bytes
        var rec_total_bytes = 4 + meta_len + payload_bytes

        # Region non-overlap & boundary check
        if cur_off + rec_total_bytes > file_size:
            f.close()
            raise Error(
                "scan_quant_file: tensor payload truncated: "
                + meta.name
                + " (offset "
                + String(cur_off + rec_total_bytes)
                + " > "
                + String(file_size)
                + ")"
            )

        var payload_offset = cur_off + 4 + meta_len
        var scales_offset = payload_offset + meta.scale_offset
        var data_offset = payload_offset + meta.data_offset

        var entry = QuantTensorEntry(
            meta=meta.copy(),
            record_offset=cur_off,
            payload_offset=payload_offset,
            scales_offset=scales_offset,
            data_offset=data_offset,
            scales_bytes=s_bytes,
            weights_bytes=w_bytes,
            total_record_bytes=rec_total_bytes,
        )

        tensor_map[meta.name] = len(entries)
        entries.append(entry^)
        cur_off += rec_total_bytes

    if cur_off != file_size:
        f.close()
        raise Error(
            "scan_quant_file: trailing data or size mismatch (read "
            + String(cur_off)
            + " != file_size "
            + String(file_size)
            + ")"
        )

    f.close()
    return QuantModelIndex(
        header=header^,
        entries=entries^,
        tensor_map=tensor_map^,
        file_path=file_path,
    )


def pread_tensor_quant(
    file_path: String, entry: QuantTensorEntry
) raises -> Tuple[List[Float16], List[UInt8]]:
    """Membaca data kuantisasi mentah (skala FP16 dan bobot 4-bit ter-pack) dari disk.
    """
    var f = open(file_path, "r")
    _ = f.seek(entry.scales_offset, SEEK_SET)
    var s_raw = f.read_bytes(entry.scales_bytes)

    _ = f.seek(entry.data_offset, SEEK_SET)
    var w_raw = f.read_bytes(entry.weights_bytes)
    f.close()

    var num_groups = entry.meta.num_groups
    var scales = List[Float16]()
    scales.resize(num_groups, Float16(0.0))
    for g in range(num_groups):
        var u = UInt16(s_raw[g * 2]) | (UInt16(s_raw[g * 2 + 1]) << 8)
        scales[g] = u16_to_float16(u)

    var packed = List[UInt8]()
    packed.resize(len(w_raw), 0)
    for i in range(len(w_raw)):
        packed[i] = w_raw[i]

    return (scales^, packed^)


def pread_and_dequant_tensor(
    file_path: String, entry: QuantTensorEntry
) raises -> List[BFloat16]:
    """Membaca tensor dari disk dan mendekuantisasi ke List[BFloat16] via kernel SIMD.
    """
    var raw_tuple = pread_tensor_quant(file_path, entry)
    var scales = raw_tuple[0].copy()
    var packed = raw_tuple[1].copy()

    return dequant_kernel_simd(
        scales,
        packed,
        entry.meta.num_elements(),
        entry.meta.group_size,
    )


def pread_and_dequant_tensor_f32(
    file_path: String, entry: QuantTensorEntry
) raises -> List[Float32]:
    """Membaca tensor dari disk dan mendekuantisasi ke List[Float32] untuk layer forward.
    """
    var raw_tuple = pread_tensor_quant(file_path, entry)
    var scales = raw_tuple[0].copy()
    var packed = raw_tuple[1].copy()

    return dequant_kernel_simd_f32(
        scales,
        packed,
        entry.meta.num_elements(),
        entry.meta.group_size,
    )
