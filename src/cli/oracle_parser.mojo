# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Parser JSON routing oracle untuk verifikasi inline G-M3-2."""

from cli.errors import fail_layer
from cli.sys_utils import str_to_float
from format.file_io import read_small_file
from format.scanner import Scanner
from std.collections import List


@fieldwise_init
struct OracleRoutingData(Copyable, Movable):
    var selected_experts: List[List[Int]]
    var router_probs: List[List[Float32]]


def _skip_to_key(mut sc: Scanner, target_key: String) raises -> Bool:
    """Mencari key tertentu di dalam JSON object saat ini."""
    while not sc.eof():
        sc.skip_ws()
        if sc.eof() or sc.peek() == 125:  # '}'
            return False
        if sc.peek() == 34:  # '"'
            var k_str = sc.parse_string()
            sc.skip_ws()
            sc.expect(58)  # ':'
            sc.skip_ws()
            if k_str == target_key:
                return True
            # Lewati nilai jika bukan objek bersarang yang dicari
            if sc.peek() == 123:  # '{'
                sc.pos += 1
                if _skip_to_key(sc, target_key):
                    return True
            elif sc.peek() == 91:  # '['
                _skip_composite(sc)
            else:
                _skip_scalar(sc)
        else:
            sc.pos += 1
    return False


def _skip_composite(mut sc: Scanner):
    var depth = 0
    while not sc.eof():
        var b = sc.peek()
        sc.pos += 1
        if b == 91 or b == 123:
            depth += 1
        elif b == 93 or b == 125:
            depth -= 1
            if depth <= 0:
                break


def _skip_scalar(mut sc: Scanner):
    while not sc.eof():
        var b = sc.peek()
        if b == 44 or b == 125 or b == 93 or b == 32 or b == 10 or b == 13:
            break
        sc.pos += 1


def _parse_2d_ints(mut sc: Scanner) raises -> List[List[Int]]:
    var out = List[List[Int]]()
    sc.expect(91)  # '['
    sc.skip_ws()
    if not sc.eof() and sc.peek() == 93:
        sc.pos += 1
        return out^
    while not sc.eof():
        sc.skip_ws()
        sc.expect(91)  # '['
        var row = List[Int]()
        sc.skip_ws()
        if not sc.eof() and sc.peek() == 93:
            sc.pos += 1
        else:
            while not sc.eof():
                var is_neg = False
                if sc.peek() == 45:
                    is_neg = True
                    sc.pos += 1
                var v = sc.parse_uint()
                row.append(-v if is_neg else v)
                sc.skip_ws()
                var b = sc.peek()
                sc.pos += 1
                if b == 93:
                    break
                if b != 44:
                    raise Error("expected , or ] in 2d ints")
        out.append(row^)
        sc.skip_ws()
        var b2 = sc.peek()
        sc.pos += 1
        if b2 == 93:
            break
        if b2 != 44:
            raise Error("expected , or ] in outer array")
    return out^


def _parse_2d_floats(mut sc: Scanner) raises -> List[List[Float32]]:
    var out = List[List[Float32]]()
    sc.expect(91)  # '['
    sc.skip_ws()
    if not sc.eof() and sc.peek() == 93:
        sc.pos += 1
        return out^
    while not sc.eof():
        sc.skip_ws()
        sc.expect(91)  # '['
        var row = List[Float32]()
        sc.skip_ws()
        if not sc.eof() and sc.peek() == 93:
            sc.pos += 1
        else:
            while not sc.eof():
                sc.skip_ws()
                var num_chars = List[UInt8]()
                while not sc.eof():
                    var c = sc.peek()
                    if (
                        (c >= 48 and c <= 57)
                        or c == 46
                        or c == 45
                        or c == 43
                        or c == 101
                        or c == 69
                    ):
                        num_chars.append(UInt8(c))
                        sc.pos += 1
                    else:
                        break
                var s = String(from_utf8_lossy=Span(num_chars))
                row.append(str_to_float(s))
                sc.skip_ws()
                var b = sc.peek()
                sc.pos += 1
                if b == 93:
                    break
                if b != 44:
                    raise Error("expected , or ] in 2d floats")
        out.append(row^)
        sc.skip_ws()
        var b2 = sc.peek()
        sc.pos += 1
        if b2 == 93:
            break
        if b2 != 44:
            raise Error("expected , or ] in outer float array")
    return out^


def parse_oracle_routing_json(
    path: String, layer_val: Int
) raises -> OracleRoutingData:
    """Membaca file JSON routing oracle (selected_experts dan router_probs)."""
    var raw_bytes = List[UInt8]()
    try:
        raw_bytes = read_small_file(path)
    except e:
        fail_layer(
            "FILE_NOT_FOUND",
            "cannot read oracle routing file: " + String(e),
            "router",
            layer_val,
        )

    var experts = List[List[Int]]()
    var probs = List[List[Float32]]()

    # Pass 1: selected_experts
    var sc1 = Scanner(raw_bytes.copy(), path)
    if _skip_to_key(sc1, "selected_experts"):
        experts = _parse_2d_ints(sc1)

    # Pass 2: router_probs
    var sc2 = Scanner(raw_bytes^, path)
    if _skip_to_key(sc2, "router_probs"):
        probs = _parse_2d_floats(sc2)

    if len(experts) == 0:
        fail_layer(
            "ROUTER_ERROR",
            "missing or empty selected_experts in oracle routing file",
            "router",
            layer_val,
        )

    return OracleRoutingData(experts^, probs^)
