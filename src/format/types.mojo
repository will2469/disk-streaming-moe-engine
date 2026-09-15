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
