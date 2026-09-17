# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Reference Python implementation for M6 custom 4-bit quant file format."""

import json
import struct
from typing import Any

import numpy as np

QUANT_HEADER_SIZE = 256
QUANT_DEFAULT_GROUP_SIZE = 128
QUANT_FORMAT_NAME = "4-bit per-group"
QUANT_SCALE_DTYPE = "FP16"
QUANTIZED_DTYPE_NAME = "4-bit"


def pack_4bit_pair(w0: int, w1: int) -> int:
    """Pack two signed integers in range [-8, 7] into a single Little-Endian byte.

    w0 is stored in the low nibble (bits 0-3), w1 in the high nibble (bits 4-7).
    """
    if not (-8 <= w0 <= 7) or not (-8 <= w1 <= 7):
        raise ValueError(f"Values must be in [-8, 7], got w0={w0}, w1={w1}")
    u0 = w0 & 0x0F
    u1 = w1 & 0x0F
    return (u1 << 4) | u0


def unpack_4bit_pair(b: int) -> tuple[int, int]:
    """Unpack a single Little-Endian byte into two signed integers in [-8, 7]."""
    u0 = b & 0x0F
    u1 = (b >> 4) & 0x0F
    w0 = u0 - 16 if u0 >= 8 else u0
    w1 = u1 - 16 if u1 >= 8 else u1
    return w0, w1


def compute_fp16_scale_ceil(max_abs: float) -> float:
    """Compute FP16 scale s_g = ceil_FP16(max_abs / 7.0).

    Zero-group returns 1.0 (with q=0).
    Ensures float(s_g) * 7.0 >= max_abs strictly without saturation.
    """
    if max_abs <= 0.0:
        return 1.0
    target = max_abs / 7.0
    s16 = np.float16(target)
    if float(s16) * 7.0 < max_abs:
        u16 = s16.view(np.uint16)
        if u16 >= 0x7C00:
            raise ValueError(f"Scale overflow in FP16 for max_abs={max_abs}")
        s16 = (u16 + np.uint16(1)).view(np.float16)
    return float(s16)


def calculate_tensor_quant_size(
    shape: list[int], group_size: int = QUANT_DEFAULT_GROUP_SIZE
) -> tuple[int, int, int, int]:
    """Calculate (num_groups, scales_bytes, weights_bytes, total_bytes)."""
    n_elem = 1
    for d in shape:
        n_elem *= d
    num_groups = (n_elem + group_size - 1) // group_size
    scales_bytes = num_groups * 2
    weights_bytes = (n_elem + 1) // 2
    total_bytes = scales_bytes + weights_bytes
    return num_groups, scales_bytes, weights_bytes, total_bytes


def make_quant_header(
    model: str,
    num_tensors: int,
    total_bytes: int,
    group_size: int = QUANT_DEFAULT_GROUP_SIZE,
    version: int = 1,
) -> bytes:
    """Create a 256-byte space-padded JSON header."""
    hdr_dict = {
        "version": version,
        "model": model,
        "quantization": {
            "format": QUANT_FORMAT_NAME,
            "group_size": group_size,
            "scale_dtype": QUANT_SCALE_DTYPE,
        },
        "num_tensors": num_tensors,
        "total_bytes": total_bytes,
    }
    raw = json.dumps(hdr_dict, separators=(",", ":")).encode("utf-8")
    if len(raw) > QUANT_HEADER_SIZE:
        raise ValueError(
            f"Header JSON ({len(raw)} bytes) exceeds limit ({QUANT_HEADER_SIZE} bytes)"
        )
    return raw.ljust(QUANT_HEADER_SIZE, b" ")


def parse_quant_header(raw: bytes) -> dict[str, Any]:
    """Parse and validate 256-byte quant header."""
    if len(raw) != QUANT_HEADER_SIZE:
        raise ValueError(f"Header must be 256 bytes, got {len(raw)}")
    text = raw.decode("utf-8").strip()
    data = json.loads(text)
    if data.get("version") != 1:
        raise ValueError(f"Unsupported version: {data.get('version')}")
    q_spec = data.get("quantization", {})
    if q_spec.get("format") != QUANT_FORMAT_NAME:
        raise ValueError(f"Unsupported format: {q_spec.get('format')}")
    if q_spec.get("scale_dtype") != QUANT_SCALE_DTYPE:
        raise ValueError(f"Unsupported scale_dtype: {q_spec.get('scale_dtype')}")
    g_sz = q_spec.get("group_size", 0)
    if g_sz <= 0 or (g_sz & (g_sz - 1)) != 0:
        raise ValueError(f"group_size must be positive power of 2, got {g_sz}")
    return data


def make_tensor_record(
    name: str,
    shape: list[int],
    scales: list[float],
    weights: list[int],
    dtype: str = "BF16",
    group_size: int = QUANT_DEFAULT_GROUP_SIZE,
) -> bytes:
    """Construct complete tensor record bytes."""
    num_groups, scales_bytes, weights_bytes, _ = calculate_tensor_quant_size(
        shape, group_size
    )
    meta = {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "quantized_dtype": QUANTIZED_DTYPE_NAME,
        "group_size": group_size,
        "num_groups": num_groups,
        "scale_offset": 0,
        "data_offset": scales_bytes,
    }
    meta_raw = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack("<I", len(meta_raw))

    # Scales in FP16 LE
    scales_raw = bytearray()
    for s in scales:
        scales_raw.extend(struct.pack("<e", s))

    # Packed weights
    packed_raw = bytearray()
    for i in range(0, len(weights), 2):
        w0 = weights[i]
        w1 = weights[i + 1] if i + 1 < len(weights) else 0
        packed_raw.append(pack_4bit_pair(w0, w1))

    return prefix + meta_raw + bytes(scales_raw) + bytes(packed_raw)


def read_tensor_record(
    raw: bytes, offset: int
) -> tuple[dict[str, Any], list[float], list[int], int]:
    """Read a tensor record from raw bytes at offset.

    Returns (metadata, scales, weights, new_offset).
    """
    if offset + 4 > len(raw):
        raise ValueError("Truncated file: cannot read meta_len")
    meta_len = struct.unpack("<I", raw[offset : offset + 4])[0]
    offset += 4
    if offset + meta_len > len(raw):
        raise ValueError("Truncated file: cannot read metadata JSON")
    meta = json.loads(raw[offset : offset + meta_len].decode("utf-8"))
    offset += meta_len

    num_groups = meta["num_groups"]
    scales_len = num_groups * 2
    if offset + scales_len > len(raw):
        raise ValueError("Truncated file: cannot read scales")
    scales = []
    for i in range(num_groups):
        s = struct.unpack("<e", raw[offset + i * 2 : offset + (i + 1) * 2])[0]
        if np.isnan(s) or np.isinf(s) or s <= 0.0:
            raise ValueError(f"Invalid non-finite or non-positive scale: {s}")
        scales.append(float(s))
    offset += scales_len

    num_elem = 1
    for d in meta["shape"]:
        num_elem *= d
    packed_len = (num_elem + 1) // 2
    if offset + packed_len > len(raw):
        raise ValueError("Truncated file: cannot read packed weights")
    weights = []
    for i in range(packed_len):
        b = raw[offset + i]
        w0, w1 = unpack_4bit_pair(b)
        weights.append(w0)
        if len(weights) < num_elem:
            weights.append(w1)
    offset += packed_len

    return meta, scales, weights, offset
