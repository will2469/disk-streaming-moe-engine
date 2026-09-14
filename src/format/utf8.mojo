# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Utilitas encoding dan decoding UTF-8 serta escape sequence JSON safetensors."""

from format.types import STError
from std.collections import List


def append_utf8_cp(mut out: List[UInt8], cp: Int, shard: String) raises:
    """Encode unicode codepoint ke byte sequence UTF-8."""
    if cp < 128:
        out.append(UInt8(cp))
    elif cp < 2048:
        out.append(UInt8(192 + cp // 64))
        out.append(UInt8(128 + cp % 64))
    elif cp < 65536:
        out.append(UInt8(224 + cp // 4096))
        out.append(UInt8(128 + (cp // 64) % 64))
        out.append(UInt8(128 + cp % 64))
    elif cp < 1114112:
        out.append(UInt8(240 + cp // 262144))
        out.append(UInt8(128 + (cp // 4096) % 64))
        out.append(UInt8(128 + (cp // 64) % 64))
        out.append(UInt8(128 + cp % 64))
    else:
        raise Error(
            String(STError("JSON_PARSE_ERROR", "codepoint liar", shard, ""))
        )


def parse_hex4_at(buf: List[UInt8], mut pos: Int, shard: String) raises -> Int:
    """Parse 4 digit heksadesimal dari buffer."""
    var v = 0
    for _ in range(4):
        if pos >= len(buf):
            raise Error(
                String(STError("JSON_PARSE_ERROR", "bad \\u escape", shard, ""))
            )
        var b = Int(buf[pos])
        pos += 1
        var d = -1
        if b >= 48 and b <= 57:
            d = b - 48
        elif b >= 65 and b <= 70:
            d = b - 55
        elif b >= 97 and b <= 102:
            d = b - 87
        if d < 0:
            raise Error(
                String(STError("JSON_PARSE_ERROR", "bad \\u escape", shard, ""))
            )
        v = v * 16 + d
    return v


def parse_json_escape(
    mut out: List[UInt8], buf: List[UInt8], mut pos: Int, shard: String
) raises:
    """Parse satu escape sequence JSON sesudah backslash '\\'."""
    if pos >= len(buf):
        raise Error(
            String(
                STError(
                    "JSON_PARSE_ERROR",
                    "bad escape",
                    shard,
                    "",
                )
            )
        )
    var e = Int(buf[pos])
    pos += 1
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
        var cp = parse_hex4_at(buf, pos, shard)
        if cp >= 55296 and cp <= 56319:
            if (
                pos + 1 >= len(buf)
                or Int(buf[pos]) != 92
                or Int(buf[pos + 1]) != 117
            ):
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            "lone high surrogate",
                            shard,
                            "",
                        )
                    )
                )
            pos += 2
            var lo = parse_hex4_at(buf, pos, shard)
            if lo < 56320 or lo > 57343:
                raise Error(
                    String(
                        STError(
                            "JSON_PARSE_ERROR",
                            "bad low surrogate",
                            shard,
                            "",
                        )
                    )
                )
            cp = 65536 + (cp - 55296) * 1024 + (lo - 56320)
        elif cp >= 56320 and cp <= 57343:
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR",
                        "lone low surrogate",
                        shard,
                        "",
                    )
                )
            )
        append_utf8_cp(out, cp, shard)
    else:
        raise Error(
            String(
                STError(
                    "JSON_PARSE_ERROR",
                    "bad escape",
                    shard,
                    "",
                )
            )
        )
