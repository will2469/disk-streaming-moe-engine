# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Pemuatan tensor chunked dari file safetensors (BF16/F32 -> F32 resident)."""

from core.config import CHUNK_MAX_BYTES, LoadMemoryTelemetry
from format import TensorMeta, read_header
from std.builtin.dtype import DType
from std.collections import List
from std.math import min
from std.os import SEEK_SET


def load_tensor_f32_chunked(
    shard_path: String,
    data_base: Int,
    meta: TensorMeta,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    """Pemuatan tensor chunked BF16/F32 -> F32 resident tanpa double-residency.

    @spec m1-w2-embed-lmhead.md (§ Anggaran memori M1)
    """
    var total_bytes = meta.end - meta.begin
    var num_elements: Int
    var element_size: Int
    if meta.dtype == "BF16":
        element_size = 2
        num_elements = total_bytes // 2
    elif meta.dtype == "F32":
        element_size = 4
        num_elements = total_bytes // 4
    else:
        raise Error(
            '{"error_type":"UNKNOWN_DTYPE","detail":"unsupported dtype: '
            + meta.dtype
            + '","shard":"'
            + shard_path
            + '","tensor_name":"'
            + meta.name
            + '"}'
        )

    var out = List[Float32]()
    out.reserve(num_elements)
    telemetry.resident_target_bytes += num_elements * 4

    var f = open(shard_path, "r")
    _ = f.seek(data_base + meta.begin, SEEK_SET)

    var bytes_remaining = total_bytes
    while bytes_remaining > 0:
        var to_read = min(bytes_remaining, CHUNK_MAX_BYTES)
        to_read = (to_read // element_size) * element_size
        if to_read == 0:
            to_read = bytes_remaining

        var chunk_bytes = f.read_bytes(to_read)
        if len(chunk_bytes) < to_read:
            f.close()
            raise Error(
                '{"error_type":"INVALID_HEADER","detail":"truncated tensor'
                ' data","shard":"'
                + shard_path
                + '","tensor_name":"'
                + meta.name
                + '"}'
            )

        if len(chunk_bytes) > telemetry.source_buffer_bytes:
            telemetry.source_buffer_bytes = len(chunk_bytes)

        var chunk_elements = len(chunk_bytes) // element_size
        var conv_bytes = chunk_elements * 4
        if conv_bytes > telemetry.conversion_buffer_bytes:
            telemetry.conversion_buffer_bytes = conv_bytes

        if meta.dtype == "BF16":
            var p_u8 = chunk_bytes.unsafe_ptr()
            var p_bf = p_u8.unsafe_bitcast[Scalar[DType.bfloat16]]()
            for i in range(chunk_elements):
                out.append(p_bf[unsafe_offset=i].cast[DType.float32]())
        else:
            var p_u8 = chunk_bytes.unsafe_ptr()
            var p_f32 = p_u8.unsafe_bitcast[Scalar[DType.float32]]()
            for i in range(chunk_elements):
                out.append(p_f32[unsafe_offset=i])

        bytes_remaining -= len(chunk_bytes)

    f.close()
    return out^


def _load_one_tensor_by_name(
    model_root: String,
    shard_file: String,
    tensor_name: String,
    dim0: Int,
    dim1: Int,
    is_2d: Bool,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    """Helper pemuatan satu tensor dengan verifikasi bentuk 1D/2D dan telemetri.
    """
    var shard_path = (
        String(model_root, "/", shard_file) if model_root != "" else shard_file
    )
    var header = read_header(shard_path)
    var found = False
    var meta = TensorMeta("", "", List[Int](), 0, 0)
    for i in range(len(header.entries)):
        ref e = header.entries[i]
        if e.name == tensor_name:
            meta = e.copy()
            found = True
            break
    if not found:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"tensor '
            + tensor_name
            + ' not found in shard header","shard":"'
            + shard_path
            + '","tensor_name":"'
            + tensor_name
            + '"}'
        )
    if is_2d:
        if (
            len(meta.shape) != 2
            or meta.shape[0] != dim0
            or meta.shape[1] != dim1
        ):
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"weight shape'
                " mismatch: expected ["
                + String(dim0)
                + ", "
                + String(dim1)
                + "] got length "
                + String(len(meta.shape))
                + '","shard":"'
                + shard_path
                + '","tensor_name":"'
                + tensor_name
                + '"}'
            )
    else:
        var valid_1d = (len(meta.shape) == 1 and meta.shape[0] == dim0) or (
            len(meta.shape) == 2
            and meta.shape[0] == 1
            and meta.shape[1] == dim0
        )
        if not valid_1d:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"1D tensor shape'
                " mismatch: expected ["
                + String(dim0)
                + "] got length "
                + String(len(meta.shape))
                + '","shard":"'
                + shard_path
                + '","tensor_name":"'
                + tensor_name
                + '"}'
            )
    return load_tensor_f32_chunked(
        shard_path, header.data_base, meta, telemetry
    )
