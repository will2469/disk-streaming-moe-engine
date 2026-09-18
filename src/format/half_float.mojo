# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Modul utilitas netral untuk operasi Float16, bit-packing, dan group scaling."""

from std.collections import List
from std.math import isinf, isnan

comptime QUANT_DEFAULT_GROUP_SIZE: Int = 128


def safe_multiply_int(a: Int, b: Int) raises -> Int:
    """Checked multiplication preventing integer overflow (SEC-4)."""
    if a < 0 or b < 0:
        raise Error("safe_multiply_int: negative operands not allowed")
    if a == 0 or b == 0:
        return 0
    var max_int = 0x7FFFFFFFFFFFFFFF
    if a > max_int // b:
        raise Error("safe_multiply_int: integer multiplication overflow")
    return a * b


def is_allowed_group_size(group_size: Int) -> Bool:
    """Himpunan izin G in {32, 64, 128, 256}."""
    return (
        group_size == 32
        or group_size == 64
        or group_size == 128
        or group_size == 256
    )


def pack_4bit_pair(w0: Int8, w1: Int8) raises -> UInt8:
    """Mengemas dua bobot bertanda [-7, 7] ke dalam 1 byte Little-Endian.

    w0 berada di low nibble (bits 0-3), w1 berada di high nibble (bits 4-7).
    Nibble 0x8 (-8) reserved/invalid: encoder tak pernah memancarkannya.
    """
    if Int(w0) < -7 or Int(w0) > 7 or Int(w1) < -7 or Int(w1) > 7:
        raise Error("pack_4bit_pair: values must be in [-7, 7] (0x8 reserved)")
    var u0 = UInt8(Int(w0) & 0x0F)
    var u1 = UInt8(Int(w1) & 0x0F)
    return (u1 << 4) | u0


def unpack_4bit_pair(b: UInt8) raises -> Tuple[Int8, Int8]:
    """Membongkar 1 byte Little-Endian menjadi dua bobot bertanda [-7, 7].

    Menolak nibble reserved 0x8 (-8) dengan Error.
    """
    var u0 = Int(b & 0x0F)
    var u1 = Int((b >> 4) & 0x0F)
    if u0 == 8 or u1 == 8:
        raise Error("unpack_4bit_pair: reserved nibble 0x8 (-8)")
    var w0 = Int8(u0 - 16 if u0 >= 8 else u0)
    var w1 = Int8(u1 - 16 if u1 >= 8 else u1)
    return (w0, w1)


def float16_to_u16(val: Float16) -> UInt16:
    """Mengonversi bit representasi Float16 ke UInt16."""
    var buf = List[UInt8]()
    buf.resize(2, 0)
    buf.unsafe_ptr().unsafe_bitcast[Float16]()[unsafe_offset=0] = val
    return buf.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=0]


def u16_to_float16(u: UInt16) -> Float16:
    """Mengonversi bit UInt16 ke Float16."""
    var buf = List[UInt8]()
    buf.resize(2, 0)
    buf.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=0] = u
    return buf.unsafe_ptr().unsafe_bitcast[Float16]()[unsafe_offset=0]


def compute_fp16_scale_ceil(max_abs: Float32) raises -> Float16:
    """Menghitung skala s_g = ceil_FP16(max_abs / 7.0).

    Bila max_abs == 0, mengembalikan 1.0 (grup nol, q = 0).
    Pembulatan ke atas menjamin Float32(s_g) * 7.0 >= max_abs (tanpa saturasi).
    Menolak NaN dan Inf dengan Error (M6_ERR_QUANT).
    """
    if isnan(max_abs) or isinf(max_abs):
        raise Error("compute_fp16_scale_ceil: NaN or Inf encountered in scale")
    if max_abs <= 0.0:
        return Float16(1.0)

    var target = max_abs / 7.0
    var s = Float16(target)
    if Float32(s) * 7.0 < max_abs:
        var u = float16_to_u16(s)
        if u >= 0x7C00:  # FP16 Infinity
            raise Error("FP16 scale overflow (exceeds finite range)")
        u += 1
        s = u16_to_float16(u)
    return s
