# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Pemuatan tensor chunked dari file safetensors (BF16/F32 -> F32 resident)."""

from core.config import CHUNK_MAX_BYTES, LoadMemoryTelemetry
from format import (
    STHeader,
    TensorMeta,
    _numel_or_fail,
    decode_bf16_le,
    decode_f32_le,
    error_json,
    read_header,
)
from format.file_io import resolve_within_root
from std.collections import Dict, List
from std.math import min
from std.os import SEEK_END, SEEK_SET


def load_tensor_f32_chunked(
    shard_path: String,
    data_base: Int,
    meta: TensorMeta,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    """Pemuatan tensor chunked BF16/F32 -> F32 resident tanpa double-residency.
    Invariants: filesize actual check, byte-size match, no overflow.
    """
    var element_size: Int
    if meta.dtype == "BF16":
        element_size = 2
    elif meta.dtype == "F32":
        element_size = 4
    else:
        raise Error(
            error_json(
                "UNKNOWN_DTYPE",
                String("unsupported dtype: ", meta.dtype),
                shard_path,
                meta.name,
            )
        )

    var f = open(shard_path, "r")
    var filesize = Int(f.seek(0, SEEK_END))
    if meta.begin < 0 or meta.end < meta.begin:
        f.close()
        raise Error(
            error_json(
                "OFFSET_OVERFLOW",
                String("BEGIN/END invalid: ", meta.begin, "..", meta.end),
                shard_path,
                meta.name,
            )
        )
    if data_base < 0 or data_base > filesize:
        f.close()
        raise Error(
            error_json(
                "OFFSET_OVERFLOW",
                "data_base di luar filesize",
                shard_path,
                meta.name,
            )
        )
    if meta.end > filesize - data_base:
        f.close()
        raise Error(
            error_json(
                "OFFSET_OVERFLOW",
                "akhir buffer + data_base melebihi filesize",
                shard_path,
                meta.name,
            )
        )
    var total_bytes = meta.end - meta.begin
    telemetry.logical_bytes_read += total_bytes
    if total_bytes % element_size != 0:
        f.close()
        raise Error(
            error_json(
                "LAYOUT_MISMATCH",
                "panjang buffer bukan kelipatan element_size",
                shard_path,
                meta.name,
            )
        )
    var num_elements = total_bytes // element_size
    # product(shape)*es == total_bytes via bentuk DIVISI (tanpa multiply
    # yang bisa overflow): _numel_or_fail menolak dim negatif/overflow.
    # Reader menjamin ini untuk meta-nya; gate ini untuk meta rakitan.
    var numel = _numel_or_fail(meta.shape, shard_path, meta.name)
    if numel != num_elements:
        f.close()
        raise Error(
            error_json(
                "LAYOUT_MISMATCH",
                "shape/product mismatch tensor bytes",
                shard_path,
                meta.name,
            )
        )

    var out = List[Float32]()
    out.reserve(num_elements)

    _ = f.seek(data_base + meta.begin, SEEK_SET)

    var bytes_remaining = total_bytes
    while bytes_remaining > 0:
        # Invariant: bytes_remaining selalu kelipatan element_size (gate
        # modulo di atas untuk nilai awal; tiap iterasi mengurangkan
        # len(chunk_bytes) == to_read yang kelipatan element_size, karena
        # short read selalu raise). Pembulatan ke bawah di sini hanya
        # membatasi ukuran chunk baca — tak pernah menyembunyikan metadata
        # invalid. Bila invariant jebol (to_read == 0) → gagal tertutup,
        # bukan baca chunk tak-selaras.
        var to_read = (
            min(bytes_remaining, CHUNK_MAX_BYTES) // element_size
        ) * element_size
        if to_read == 0:
            f.close()
            raise Error(
                error_json(
                    "LAYOUT_MISMATCH",
                    "chunk tak-selaras: invariant kelipatan element_size jebol",
                    shard_path,
                    meta.name,
                )
            )

        var chunk_bytes = f.read_bytes(to_read)
        if len(chunk_bytes) < to_read:
            f.close()
            raise Error(
                error_json(
                    "INVALID_HEADER",
                    "truncated tensor data",
                    shard_path,
                    meta.name,
                )
            )

        if len(chunk_bytes) > telemetry.source_buffer_bytes:
            telemetry.source_buffer_bytes = len(chunk_bytes)

        var chunk_elements = len(chunk_bytes) // element_size
        var conv_bytes = chunk_elements * 4
        if conv_bytes > telemetry.conversion_buffer_bytes:
            telemetry.conversion_buffer_bytes = conv_bytes

        # Satu primitive decode (format.types): tanpa bitcast pointer,
        # tanpa asumsi alignment/endianness host. Eksak untuk BF16/F32.
        if meta.dtype == "BF16":
            for i in range(chunk_elements):
                var o = i * 2
                out.append(decode_bf16_le(chunk_bytes[o], chunk_bytes[o + 1]))
        else:
            for i in range(chunk_elements):
                var o = i * 4
                out.append(
                    decode_f32_le(
                        chunk_bytes[o],
                        chunk_bytes[o + 1],
                        chunk_bytes[o + 2],
                        chunk_bytes[o + 3],
                    )
                )

        bytes_remaining -= len(chunk_bytes)

    # Akuntansi resident HANYA setelah read/konversi sukses penuh, sebesar
    # output aktual. Klaim sebelum read berbohong bila tensor truncated
    # (claimed allocation vs actual resident).
    telemetry.resident_target_bytes += len(out) * 4
    f.close()
    return out^


def load_tensor_slice_f32(
    shard_path: String,
    data_base: Int,
    meta: TensorMeta,
    slice_idx: Int,
    slice_elements: Int,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    """Pemuatan irisan (slice) tensor 3D tanpa memuat seluruh tensor ke RAM.
    Offset slice: meta.begin + slice_idx * (slice_elements * element_size).
    """
    var element_size: Int
    if meta.dtype == "BF16":
        element_size = 2
    elif meta.dtype == "F32":
        element_size = 4
    else:
        raise Error(
            error_json(
                "UNKNOWN_DTYPE",
                String("unsupported dtype: ", meta.dtype),
                shard_path,
                meta.name,
            )
        )

    var f = open(shard_path, "r")
    var filesize = Int(f.seek(0, SEEK_END))
    if meta.begin < 0 or meta.end < meta.begin:
        f.close()
        raise Error(
            error_json(
                "OFFSET_OVERFLOW",
                String("BEGIN/END invalid: ", meta.begin, "..", meta.end),
                shard_path,
                meta.name,
            )
        )
    if data_base < 0 or data_base > filesize:
        f.close()
        raise Error(
            error_json(
                "OFFSET_OVERFLOW",
                "data_base di luar filesize",
                shard_path,
                meta.name,
            )
        )

    var slice_bytes = slice_elements * element_size
    var slice_begin = meta.begin + slice_idx * slice_bytes
    var slice_end = slice_begin + slice_bytes

    if slice_idx < 0 or slice_begin < meta.begin or slice_end > meta.end:
        f.close()
        raise Error(
            error_json(
                "OFFSET_OVERFLOW",
                String("slice range invalid: ", slice_begin, "..", slice_end),
                shard_path,
                meta.name,
            )
        )

    if slice_end > filesize - data_base:
        f.close()
        raise Error(
            error_json(
                "OFFSET_OVERFLOW",
                "akhir slice buffer + data_base melebihi filesize",
                shard_path,
                meta.name,
            )
        )

    telemetry.logical_bytes_read += slice_bytes

    var out = List[Float32]()
    out.reserve(slice_elements)

    _ = f.seek(data_base + slice_begin, SEEK_SET)

    var bytes_remaining = slice_bytes
    while bytes_remaining > 0:
        var to_read = (
            min(bytes_remaining, CHUNK_MAX_BYTES) // element_size
        ) * element_size
        if to_read == 0:
            f.close()
            raise Error(
                error_json(
                    "LAYOUT_MISMATCH",
                    "chunk tak-selaras: invariant kelipatan element_size jebol",
                    shard_path,
                    meta.name,
                )
            )

        var chunk_bytes = f.read_bytes(to_read)
        if len(chunk_bytes) < to_read:
            f.close()
            raise Error(
                error_json(
                    "INVALID_HEADER",
                    "truncated tensor data in slice",
                    shard_path,
                    meta.name,
                )
            )

        if len(chunk_bytes) > telemetry.source_buffer_bytes:
            telemetry.source_buffer_bytes = len(chunk_bytes)

        var chunk_elements = len(chunk_bytes) // element_size
        var conv_bytes = chunk_elements * 4
        if conv_bytes > telemetry.conversion_buffer_bytes:
            telemetry.conversion_buffer_bytes = conv_bytes

        if meta.dtype == "BF16":
            for i in range(chunk_elements):
                var o = i * 2
                out.append(decode_bf16_le(chunk_bytes[o], chunk_bytes[o + 1]))
        else:
            for i in range(chunk_elements):
                var o = i * 4
                out.append(
                    decode_f32_le(
                        chunk_bytes[o],
                        chunk_bytes[o + 1],
                        chunk_bytes[o + 2],
                        chunk_bytes[o + 3],
                    )
                )

        bytes_remaining -= len(chunk_bytes)

    telemetry.resident_target_bytes += len(out) * 4
    f.close()
    return out^


struct ShardHeaderCache(Movable):
    """Header shard yang sudah dibaca: satu file dibaca+parse sekali lalu
    dipakai N tensor (anti IO amplification untuk disk-streaming engine).

    Lifecycle: load shard → read header once → lookup → load N tensor.
    Scope = satu invocation (dibuat di command handler, di-thread via `mut`
    seperti telemetry). maps sejajar headers: HeaderIndex tensor_name →
    entry, tanpa linear scan per tensor.
    """

    var files: List[String]
    var headers: List[STHeader]
    var maps: List[Dict[String, Int]]
    var total_header_bytes: Int

    def __init__(out self):
        self.files = List[String]()
        self.headers = List[STHeader]()
        self.maps = List[Dict[String, Int]]()
        self.total_header_bytes = 0

    def get_or_read(mut self, shard_path: String) raises -> Int:
        for i in range(len(self.files)):
            if self.files[i] == shard_path:
                return i
        var st = read_header(shard_path)
        self.total_header_bytes += st.data_base
        var pos = Dict[String, Int]()
        for k in range(len(st.entries)):
            pos[st.entries[k].name] = k
        self.files.append(shard_path)
        self.maps.append(pos^)
        self.headers.append(st^)
        return len(self.files) - 1


def read_shard_header(
    model_root: String, shard_file: String
) raises -> STHeader:
    # SATU-SATUNYA jalan membentuk path shard dari (model_root, nama index):
    # resolve aman (containment) + baca. Join lexical dilarang.
    var shard_path = resolve_within_root(model_root, shard_file)
    return read_header(shard_path)


def _load_one_tensor_by_name(
    mut cache: ShardHeaderCache,
    model_root: String,
    shard_file: String,
    tensor_name: String,
    dim0: Int,
    dim1: Int,
    is_2d: Bool,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    """Helper pemuatan satu tensor dengan verifikasi bentuk 1D/2D dan telemetri.

    Header dibaca sekali per file via cache (bukan per tensor).
    """
    var shard_path = resolve_within_root(model_root, shard_file)
    var idx = cache.get_or_read(shard_path)
    ref hdr = cache.headers[idx]
    ref pos = cache.maps[idx]
    if tensor_name not in pos:
        raise Error(
            error_json(
                "WEIGHT_LOAD_FAILED",
                String("tensor ", tensor_name, " not found in shard header"),
                shard_path,
                tensor_name,
            )
        )
    var gi = pos[tensor_name]
    ref meta = hdr.entries[gi]
    if is_2d:
        if (
            len(meta.shape) != 2
            or meta.shape[0] != dim0
            or meta.shape[1] != dim1
        ):
            raise Error(
                error_json(
                    "WEIGHT_LOAD_FAILED",
                    String(
                        "weight shape mismatch: expected [",
                        dim0,
                        ", ",
                        dim1,
                        "] got length ",
                        len(meta.shape),
                    ),
                    shard_path,
                    tensor_name,
                )
            )
    else:
        var valid_1d = (len(meta.shape) == 1 and meta.shape[0] == dim0) or (
            len(meta.shape) == 2
            and meta.shape[0] == 1
            and meta.shape[1] == dim0
        )
        if not valid_1d:
            raise Error(
                error_json(
                    "WEIGHT_LOAD_FAILED",
                    String(
                        "1D tensor shape mismatch: expected [",
                        dim0,
                        "] got length ",
                        len(meta.shape),
                    ),
                    shard_path,
                    tensor_name,
                )
            )
    return load_tensor_f32_chunked(
        shard_path, hdr.data_base, meta.copy(), telemetry
    )


def _load_tensor_slice_by_name(
    mut cache: ShardHeaderCache,
    model_root: String,
    shard_file: String,
    tensor_name: String,
    slice_idx: Int,
    slice_dim0: Int,
    slice_dim1: Int,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    """Helper pemuatan irisan satu expert dari tensor 3D [num_experts, slice_dim0, slice_dim1].
    """
    var shard_path = resolve_within_root(model_root, shard_file)
    var idx = cache.get_or_read(shard_path)
    ref hdr = cache.headers[idx]
    ref pos = cache.maps[idx]
    if tensor_name not in pos:
        raise Error(
            error_json(
                "WEIGHT_LOAD_FAILED",
                String("tensor ", tensor_name, " not found in shard header"),
                shard_path,
                tensor_name,
            )
        )
    var gi = pos[tensor_name]
    ref meta = hdr.entries[gi]
    if (
        len(meta.shape) != 3
        or meta.shape[1] != slice_dim0
        or meta.shape[2] != slice_dim1
    ):
        raise Error(
            error_json(
                "WEIGHT_LOAD_FAILED",
                String(
                    "3D tensor shape mismatch: expected [N, ",
                    slice_dim0,
                    ", ",
                    slice_dim1,
                    "] got length ",
                    len(meta.shape),
                ),
                shard_path,
                tensor_name,
            )
        )
    if slice_idx < 0 or slice_idx >= meta.shape[0]:
        raise Error(
            error_json(
                "WEIGHT_LOAD_FAILED",
                String("slice index out of bounds: ", slice_idx),
                shard_path,
                tensor_name,
            )
        )
    var slice_elements = slice_dim0 * slice_dim1
    return load_tensor_slice_f32(
        shard_path,
        hdr.data_base,
        meta.copy(),
        slice_idx,
        slice_elements,
        telemetry,
    )
