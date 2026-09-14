# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Pembaca shard safetensors byte-level dan verifikasi header F15."""

from format.file_io import _open_shard, read_small_file
from format.scanner import Scanner
from format.types import (
    HEADER_MAX,
    STError,
    STHeader,
    TENSOR_MAX,
    TensorMeta,
    _dtype_or_fail,
    _numel_or_fail,
)
from std.collections import List
from std.os import SEEK_END, SEEK_SET


def read_header(path: String) raises -> STHeader:
    """Parse header satu shard + tegakkan F15a/b/c. Tidak membaca payload."""
    var f = _open_shard(path)
    var filesize = Int(f.seek(0, SEEK_END))
    _ = f.seek(0, SEEK_SET)
    var prefix = f.read_bytes(8)
    if len(prefix) < 8:
        f.close()
        raise Error(
            String(STError("INVALID_HEADER", "file < 8 byte", path, ""))
        )
    var header_len = 0
    var mult = 1
    for i in range(8):
        header_len += Int(prefix[i]) * mult
        mult = mult * 256
    if header_len > HEADER_MAX:
        f.close()
        raise Error(
            String(
                STError(
                    "INVALID_HEADER",
                    String("header_len melebihi 100 MB: ", header_len),
                    path,
                    "",
                )
            )
        )
    var data_base = 8 + header_len
    if data_base > filesize:
        f.close()
        raise Error(
            String(
                STError("INVALID_HEADER", "header melebihi filesize", path, "")
            )
        )
    var hbytes = f.read_bytes(header_len)
    f.close()
    if len(hbytes) < header_len:
        raise Error(
            String(STError("INVALID_HEADER", "header terpotong", path, ""))
        )
    var sc = Scanner(hbytes^, path)
    sc.skip_ws()
    sc.expect(123)
    var names = List[String]()
    var metas = List[TensorMeta]()
    while True:
        sc.skip_ws()
        if sc.eof():
            raise Error(
                String(
                    STError("JSON_PARSE_ERROR", "objek tak berakhir", path, "")
                )
            )
        if sc.peek() == 125:
            sc.pos += 1
            break
        var key = sc.parse_string()
        sc.expect(58)
        if key == "__metadata__":
            sc.skip_value()
        else:
            for i in range(len(names)):
                if names[i] == key:
                    raise Error(
                        String(
                            STError(
                                "DUPLICATE_JSON_KEY",
                                String("kunci ganda: ", key),
                                path,
                                key,
                            )
                        )
                    )
            names.append(key)
            sc.skip_ws()
            sc.expect(123)
            var dtype = String("")
            var has_dtype = False
            var shape = List[Int]()
            var has_shape = False
            var begin = -1
            var end = -1
            var has_off = False
            while True:
                sc.skip_ws()
                if sc.eof():
                    raise Error(
                        String(
                            STError(
                                "JSON_PARSE_ERROR",
                                "objek tensor putus",
                                path,
                                key,
                            )
                        )
                    )
                if sc.peek() == 125:
                    sc.pos += 1
                    break
                var field = sc.parse_string()
                sc.expect(58)
                if field == "dtype":
                    dtype = sc.parse_string()
                    has_dtype = True
                elif field == "shape":
                    shape = sc.parse_int_array()
                    has_shape = True
                elif field == "data_offsets":
                    sc.skip_ws()
                    sc.expect(91)
                    begin = sc.parse_uint()
                    sc.skip_ws()
                    sc.expect(44)
                    var e2 = sc.parse_uint()
                    sc.skip_ws()
                    sc.expect(93)
                    end = e2
                    has_off = True
                else:
                    sc.skip_value()
                sc.skip_ws()
                if sc.eof():
                    raise Error(
                        String(
                            STError(
                                "JSON_PARSE_ERROR",
                                "objek tensor putus",
                                path,
                                key,
                            )
                        )
                    )
                var c = sc.peek()
                sc.pos += 1
                if c == 125:
                    break
                if c != 44:
                    raise Error(
                        String(
                            STError(
                                "JSON_PARSE_ERROR",
                                "harap , atau }",
                                path,
                                key,
                            )
                        )
                    )
            if not has_dtype or not has_shape or not has_off:
                raise Error(
                    String(STError("INVALID_HEADER", "field kurang", path, key))
                )
            var sz = _dtype_or_fail(dtype, path, key)
            if begin < 0 or end < begin:
                raise Error(
                    String(
                        STError(
                            "OFFSET_OVERFLOW",
                            String(
                                "BEGIN/END invalid: ",
                                begin,
                                "..",
                                end,
                            ),
                            path,
                            key,
                        )
                    )
                )
            if end > filesize - data_base:
                raise Error(
                    String(
                        STError(
                            "OFFSET_OVERFLOW",
                            String(
                                "akhir buffer ",
                                end,
                                " + data_base ",
                                data_base,
                                " = file ",
                                end + data_base,
                                " > filesize ",
                                filesize,
                            ),
                            path,
                            key,
                        )
                    )
                )
            var numel = _numel_or_fail(shape, path, key)
            if end - begin != numel * sz:
                raise Error(
                    String(
                        STError(
                            "LAYOUT_MISMATCH",
                            String(
                                "len ",
                                end - begin,
                                " != numel*size ",
                                numel * sz,
                            ),
                            path,
                            key,
                        )
                    )
                )
            var m = TensorMeta(key, dtype, shape^, begin, end)
            metas.append(m^)
        sc.skip_ws()
        if sc.eof():
            raise Error(
                String(STError("JSON_PARSE_ERROR", "objek putus", path, ""))
            )
        var sep = sc.peek()
        sc.pos += 1
        if sep == 125:
            break
        if sep != 44:
            raise Error(
                String(STError("JSON_PARSE_ERROR", "harap , atau }", path, ""))
            )
    if len(metas) > TENSOR_MAX:
        raise Error(
            String(STError("INVALID_HEADER", "tensor > 100000", path, ""))
        )
    # F15b: sortir menurut BEGIN, buffer penuh tanpa lubang/overlap
    var order = List[Int]()
    for i in range(len(metas)):
        order.append(i)
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if metas[order[j]].begin < metas[order[i]].begin:
                var t = order[i]
                order[i] = order[j]
                order[j] = t
    var prev_end = 0
    for k in range(len(order)):
        ref m = metas[order[k]]
        if m.begin != prev_end:
            raise Error(
                String(
                    STError(
                        "OFFSET_OVERFLOW",
                        String(
                            "lubang/overlap di ",
                            m.begin,
                            " (harap ",
                            prev_end,
                            ")",
                        ),
                        path,
                        m.name,
                    )
                )
            )
        prev_end = m.end
    if prev_end != filesize - data_base:
        raise Error(
            String(
                STError(
                    "OFFSET_OVERFLOW",
                    String(
                        "buffer tak penuh: akhir ",
                        prev_end,
                        " != ",
                        filesize - data_base,
                    ),
                    path,
                    "",
                )
            )
        )
    var h = STHeader(
        path, filesize, data_base, header_len, metas^, 8 + header_len
    )
    return h^
