# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Parser untuk index file safetensors (model.safetensors.index.json)."""

from format.reader import read_small_file
from format.scanner import Scanner
from format.types import TENSOR_MAX, json_escape
from std.collections import Dict, List


def parse_index(path: String) raises -> List[String]:
    """Parse index json dan kembalikan [count, names[0], files[0], ...]."""
    var raw = read_small_file(path)
    var sc = Scanner(raw^, path, allow_float=True)
    var names = List[String]()
    var files = List[String]()
    var seen = Dict[String, Int]()
    sc.skip_ws()
    sc.expect(123)
    while True:
        sc.skip_ws()
        if sc.eof():
            raise Error(
                String(
                    (
                        '{"error_type":"JSON_PARSE_ERROR","detail":"index'
                        ' cut","shard":"'
                    ),
                    path,
                    '","tensor_name":""}',
                )
            )
        if sc.peek() == 125:
            sc.pos += 1
            break
        var key = sc.parse_string()
        sc.expect(58)
        if key == "metadata":
            sc.skip_value()
        elif key == "weight_map":
            sc.skip_ws()
            sc.expect(123)
            while True:
                sc.skip_ws()
                if sc.eof():
                    raise Error(
                        String(
                            (
                                '{"error_type":"JSON_PARSE_ERROR","detail":"weight_map'
                                ' cut","shard":"'
                            ),
                            path,
                            '","tensor_name":""}',
                        )
                    )
                if sc.peek() == 125:
                    sc.pos += 1
                    break
                var nm = sc.parse_string()
                sc.expect(58)
                var fname = sc.parse_string()
                if nm in seen:
                    raise Error(
                        String(
                            (
                                '{"error_type":"DUPLICATE_JSON_KEY","detail":"dup'
                                ' weight_map key","shard":"'
                            ),
                            path,
                            '","tensor_name":"',
                            json_escape(nm),
                            '"}',
                        )
                    )
                seen[nm] = len(names)
                names.append(nm)
                files.append(fname)
                if len(names) > TENSOR_MAX:
                    raise Error(
                        String(
                            (
                                '{"error_type":"INVALID_HEADER","detail":"weight_map'
                                ' > 100000 entri","shard":"'
                            ),
                            path,
                            '","tensor_name":""}',
                        )
                    )
                sc.skip_ws()
                if sc.eof():
                    raise Error(
                        String(
                            (
                                '{"error_type":"JSON_PARSE_ERROR","detail":"weight_map'
                                ' cut","shard":"'
                            ),
                            path,
                            '","tensor_name":""}',
                        )
                    )
                var c = sc.peek()
                sc.pos += 1
                if c == 125:
                    break
                if c != 44:
                    raise Error(
                        String(
                            (
                                '{"error_type":"JSON_PARSE_ERROR","detail":"expect'
                                ' , or }","shard":"'
                            ),
                            path,
                            '","tensor_name":""}',
                        )
                    )
        else:
            sc.skip_value()
        sc.skip_ws()
        if sc.eof():
            raise Error(
                String(
                    (
                        '{"error_type":"JSON_PARSE_ERROR","detail":"index'
                        ' cut","shard":"'
                    ),
                    path,
                    '","tensor_name":""}',
                )
            )
        var sep = sc.peek()
        sc.pos += 1
        if sep == 125:
            break
        if sep != 44:
            raise Error(
                String(
                    (
                        '{"error_type":"JSON_PARSE_ERROR","detail":"expect , or'
                        ' }","shard":"'
                    ),
                    path,
                    '","tensor_name":""}',
                )
            )
    var packed = List[String]()
    packed.append(String(len(names)))
    for i in range(len(names)):
        packed.append(names[i])
        packed.append(files[i])
    return packed^


def parse_index_to_dict(path: String) raises -> Dict[String, String]:
    """Parse index json langsung ke Dict mapping tensor_name -> shard_file."""
    var packed = parse_index(path)
    var num_map = 0
    var cs = packed[0].as_bytes()
    for ci in range(len(cs)):
        num_map = num_map * 10 + (Int(cs[ci]) - 48)
    var weight_map = Dict[String, String]()
    for w in range(num_map):
        weight_map[packed[1 + 2 * w]] = packed[1 + 2 * w + 1]
    return weight_map^
