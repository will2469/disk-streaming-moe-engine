# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Safetensors multi-shard reader — header parse + predikat F15 (M0-W1).

Kontrak: hanya byte `[0, data_base)` yang dibaca (kontrak I/O M0).
`data_offsets` bersifat relatif terhadap `data_base = 8 + header_len` [R7].
"""

from std.os import SEEK_END, SEEK_SET

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
            self.code,
            '","detail":"',
            self.detail,
            '","shard":"',
            self.shard,
            '","tensor_name":"',
            self.tensor,
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
    if dtype == "BF16":
        return 2
    if dtype == "F16":
        return 2
    if dtype == "F32":
        return 4
    if dtype == "F64":
        return 8
    return -1


struct Scanner(Movable):
    """Pemindai JSON byte-level untuk subset header safetensors."""

    var buf: List[UInt8]
    var pos: Int
    var shard: String

    def __init__(out self, var buf: List[UInt8], shard: String):
        self.buf = buf^
        self.pos = 0
        self.shard = shard

    def eof(self) -> Bool:
        return self.pos >= len(self.buf)

    def peek(self) -> Int:
        return Int(self.buf[self.pos])

    def skip_ws(mut self):
        while not self.eof():
            var b = self.peek()
            if b == 32 or b == 10 or b == 13 or b == 9:
                self.pos += 1
            else:
                break

    def expect(mut self, want: Int) raises:
        self.skip_ws()
        if self.eof() or self.peek() != want:
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR",
                        String("expected byte ", want, " at ", self.pos),
                        self.shard,
                        "",
                    )
                )
            )
        self.pos += 1

    def parse_uint(mut self) raises -> Int:
        self.skip_ws()
        var v = 0
        var ndigits = 0
        while not self.eof():
            var b = self.peek()
            if b < 48 or b > 57:
                break
            if v > 900719925474099:
                raise Error(
                    String(
                        STError(
                            "INVALID_HEADER", "integer overflow", self.shard, ""
                        )
                    )
                )
            v = v * 10 + (b - 48)
            ndigits += 1
            self.pos += 1
        if ndigits == 0:
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR",
                        String("expected integer at ", self.pos),
                        self.shard,
                        "",
                    )
                )
            )
        return v

    def parse_string(mut self) raises -> String:
        # consumes opening quote; handles escapes incl \uXXXX (BMP)
        self.expect(34)
        var out = List[UInt8]()
        while True:
            if self.eof():
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            "unterminated string",
                            self.shard,
                            "",
                        )
                    )
                )
            var b = self.peek()
            if b == 34:
                self.pos += 1
                break
            if b == 92:
                self.pos += 1
                if self.eof():
                    raise Error(
                        String(
                            STError(
                                "JSON_PARSE_ERROR",
                                "bad escape",
                                self.shard,
                                "",
                            )
                        )
                    )
                var e = self.peek()
                self.pos += 1
                if e == 34:
                    out.append(34)
                elif e == 92:
                    out.append(92)
                elif e == 47:
                    out.append(47)
                elif e == 98:
                    out.append(8)
                elif e == 102:
                    out.append(12)
                elif e == 110:
                    out.append(10)
                elif e == 114:
                    out.append(13)
                elif e == 116:
                    out.append(9)
                elif e == 117:
                    var cp = self.parse_hex4()
                    # encode BMP codepoint as UTF-8
                    if cp < 128:
                        out.append(UInt8(cp))
                    elif cp < 2048:
                        out.append(UInt8(192 + cp // 64))
                        out.append(UInt8(128 + cp % 64))
                    else:
                        out.append(UInt8(224 + cp // 4096))
                        out.append(UInt8(128 + (cp // 64) % 64))
                        out.append(UInt8(128 + cp % 64))
                else:
                    raise Error(
                        String(
                            STError(
                                "JSON_PARSE_ERROR",
                                "bad escape",
                                self.shard,
                                "",
                            )
                        )
                    )
            else:
                out.append(UInt8(b))
                self.pos += 1
        return String(from_utf8_lossy=Span(out))

    def parse_hex4(mut self) raises -> Int:
        var v = 0
        for _ in range(4):
            if self.eof():
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR", "bad \\u escape", self.shard, ""
                        )
                    )
                )
            var b = self.peek()
            self.pos += 1
            var d = -1
            if b >= 48 and b <= 57:
                d = b - 48
            elif b >= 65 and b <= 70:
                d = b - 55
            elif b >= 97 and b <= 102:
                d = b - 87
            if d < 0:
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR", "bad \\u escape", self.shard, ""
                        )
                    )
                )
            v = v * 16 + d
        return v

    def skip_value(mut self) raises:
        # lewati satu nilai JSON arbitrer (untuk __metadata__ / field tak dikenal)
        self.skip_ws()
        if self.eof():
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR", "unexpected eof", self.shard, ""
                    )
                )
            )
        var b = self.peek()
        if b == 34:
            _ = self.parse_string()
        elif b == 123 or b == 91:
            var open_b = b
            var close_b = 125
            if b == 91:
                close_b = 93
            self.pos += 1
            var depth = 1
            while depth > 0:
                if self.eof():
                    raise Error(
                        String(
                            STError(
                                "JSON_PARSE_ERROR",
                                "unterminated composite",
                                self.shard,
                                "",
                            )
                        )
                    )
                var c = self.peek()
                self.pos += 1
                if c == 34:
                    self.pos -= 1
                    _ = self.parse_string()
                elif c == open_b:
                    depth += 1
                elif c == close_b:
                    depth -= 1
                elif c == 92:
                    # backslash di luar string: invalid, tapi hitung aman
                    pass
        elif (b >= 48 and b <= 57) or b == 45:
            if b == 45:
                self.pos += 1
            _ = self.parse_uint()
        elif b == 116:
            self.expect_literal("true")
        elif b == 102:
            self.expect_literal("false")
        elif b == 110:
            self.expect_literal("null")
        else:
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR",
                        String("bad value at ", self.pos),
                        self.shard,
                        "",
                    )
                )
            )

    def expect_literal(mut self, word: String) raises:
        var wb = word.as_bytes()
        for i in range(len(wb)):
            if self.eof() or self.peek() != Int(wb[i]):
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            String("bad literal at ", self.pos),
                            self.shard,
                            "",
                        )
                    )
                )
            self.pos += 1

    def parse_int_array(mut self) raises -> List[Int]:
        var out = List[Int]()
        self.expect(91)
        self.skip_ws()
        if not self.eof() and self.peek() == 93:
            self.pos += 1
            return out^
        while True:
            var v = self.parse_uint()
            out.append(v)
            self.skip_ws()
            if self.eof():
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            "unterminated array",
                            self.shard,
                            "",
                        )
                    )
                )
            var b = self.peek()
            self.pos += 1
            if b == 93:
                break
            if b != 44:
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            "expected , or ]",
                            self.shard,
                            "",
                        )
                    )
                )
        return out^


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


def _open_shard(path: String) raises -> FileHandle:
    try:
        return open(path, "r")
    except:
        raise Error(
            String(
                STError(
                    "FILE_NOT_FOUND",
                    String("tidak bisa open: ", path),
                    path,
                    "",
                )
            )
        )


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
            # DUPLICATE_JSON_KEY: kunci ganda level sintaks
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
