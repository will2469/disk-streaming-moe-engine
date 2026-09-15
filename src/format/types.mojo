# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Tipe data, konstanta, dan error handler untuk safetensors (F15, C1)."""

from std.collections import List


def _hex_digit(d: Int) -> UInt8:
    if d < 10:
        return UInt8(48 + d)
    return UInt8(87 + d)


def json_escape(s: String) -> String:
    # Invariant protokol: SEMUA string dinamis wajib lewat sini sebelum masuk
    # JSON. Escape '"', '\\', dan SEMUA kontrol < 0x20 (nama file Linux bisa
    # memuat newline/tab; tanpa ini output JSON invalid).
    var sl = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(sl)):
        var b = Int(sl[i])
        if b == 34 or b == 92:
            out.append(92)
            out.append(UInt8(b))
        elif b == 10:
            out.append(92)
            out.append(110)
        elif b == 13:
            out.append(92)
            out.append(114)
        elif b == 9:
            out.append(92)
            out.append(116)
        elif b == 8:
            out.append(92)
            out.append(98)
        elif b == 12:
            out.append(92)
            out.append(102)
        elif b < 32:
            out.append(92)
            out.append(117)
            out.append(48)
            out.append(48)
            out.append(_hex_digit(b // 16))
            out.append(_hex_digit(b % 16))
        else:
            out.append(UInt8(b))
    return String(from_utf8_lossy=Span(out))


comptime HEADER_MAX = 100000000
comptime TENSOR_MAX = 100000


@fieldwise_init
struct STError(Copyable, Movable, Writable):
    """Error terstruktur (C1: stderr JSON, bukan panic)."""

    var code: String
    var detail: String
    var shard: String
    var tensor: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            '{"error_type":"',
            json_escape(self.code),
            '","detail":"',
            json_escape(self.detail),
            '","shard":"',
            json_escape(self.shard),
            '","tensor_name":"',
            json_escape(self.tensor),
            '"}',
        )


@fieldwise_init
struct TensorMeta(Copyable, Movable):
    """Satu entri tensor: nama + aktual dari header (dtype/shape/offsets)."""

    var name: String
    var dtype: String
    var shape: List[Int]
    var begin: Int
    var end: Int


@fieldwise_init
struct STHeader(Movable):
    """Header terparse + bukti F15 satu shard."""

    var shard: String
    var filesize: Int
    var data_base: Int
    var header_len: Int
    var entries: List[TensorMeta]
    var bytes_header_read: Int


def _fail(code: String, detail: String, shard: String, tensor: String) raises:
    raise Error(String(STError(code, detail, shard, tensor)))


def error_json(
    error_type: String, detail: String, shard: String, tensor_name: String
) -> String:
    # SATU-SATUNYA cara membangun error JSON 4-field: keempat string dinamis
    # selalu di-escape di sini. Jangan susun JSON error manual per-fungsi
    # (bug class: nama tensor/path eksternal merusak JSON).
    return String(
        '{"error_type":"',
        json_escape(error_type),
        '","detail":"',
        json_escape(detail),
        '","shard":"',
        json_escape(shard),
        '","tensor_name":"',
        json_escape(tensor_name),
        '"}',
    )


def decode_f32_le(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> Float32:
    # SATU-SATUNYA primitive decode byte representation: 4 byte
    # little-endian → Float32. Kontrak format = LE IEEE (konvensi
    # safetensors); tanpa bitcast pointer sehingga tanpa asumsi alignment
    # buffer maupun endianness host. Eksak: tiap nilai f32 terwakili persis
    # di f64 dan scaling 2^k persis (tanpa overflow/underflow di rentang f64).
    var bits = (
        UInt32(b0) | (UInt32(b1) << 8) | (UInt32(b2) << 16) | (UInt32(b3) << 24)
    )
    var sign = Float64(1.0)
    if (bits >> 31) != UInt32(0):
        sign = Float64(-1.0)
    var exp = Int((bits >> 23) & UInt32(255))
    var frac = Float64(bits & UInt32(8388607))
    if exp == 255:
        if frac == Float64(0.0):
            return Float32(sign * (Float64(1e308) * Float64(10.0)))
        return Float32(Float64(0.0) / Float64(0.0))
    var mant = frac
    var e = -126 - 23
    if exp != 0:
        mant = frac + Float64(8388608.0)
        e = exp - 127 - 23
    var v = sign * mant
    var k = e
    while k >= 10:
        v *= Float64(1024.0)
        k -= 10
    while k <= -10:
        v /= Float64(1024.0)
        k += 10
    while k > 0:
        v *= Float64(2.0)
        k -= 1
    while k < 0:
        v /= Float64(2.0)
        k += 1
    return Float32(v)


def decode_bf16_le(lo: UInt8, hi: UInt8) -> Float32:
    # BF16 LE (2 byte) = paruh atas representasi f32: bit f32 = [hi lo 00 00].
    # Setiap nilai BF16 terwakili persis di f32 → komposisi ini eksak.
    return decode_f32_le(0, 0, lo, hi)


def _dtype_size(dtype: String) -> Int:
    """Ukuran elemen tipe data dalam byte."""
    if dtype == "BF16" or dtype == "F16":
        return 2
    if dtype == "F32":
        return 4
    if dtype == "F64":
        return 8
    return -1


def _dtype_or_fail(dtype: String, shard: String, tensor: String) raises -> Int:
    var sz = _dtype_size(dtype)
    if sz < 0:
        raise Error(
            String(
                STError(
                    "UNKNOWN_DTYPE",
                    String("dtype asing: ", dtype),
                    shard,
                    tensor,
                )
            )
        )
    return sz


def _numel_or_fail(
    shape: List[Int], shard: String, tensor: String
) raises -> Int:
    """Validasi dan hitung total elemen shape tensor."""
    var n = 1
    for i in range(len(shape)):
        var d = shape[i]
        if d < 0:
            raise Error(
                String(
                    STError("INVALID_HEADER", "dimensi negatif", shard, tensor)
                )
            )
        if d > 0 and n > 4611686018427387903 // d:
            raise Error(
                String(
                    STError("INVALID_HEADER", "numel overflow", shard, tensor)
                )
            )
        n = n * d
    return n
