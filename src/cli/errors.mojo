# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Utilitas penanganan error JSON dan path string untuk Kimo CLI."""

from format.types import json_escape
from std.collections import List
from std.sys.terminate import exit


def eprint_json(msg: String) raises:
    # /dev/stderr dibuka append (tanpa truncate: O_TRUNC di pipe -> ENXIO).
    # Bila device tak ada, fallback stdout agar error tetap terlihat.
    try:
        var e = open("/dev/stderr", "a")
        e.write_all(msg.as_bytes())
        e.close()
    except:
        print(msg)


def err_json(
    code: String, detail: String, shard: String, tensor: String
) -> String:
    # SEMUA field lolos json_escape (detail/shard/tensor bisa dari path CLI).
    return String(
        '{"error_type":"',
        json_escape(code),
        '","detail":"',
        json_escape(detail),
        '","shard":"',
        json_escape(shard),
        '","tensor_name":"',
        json_escape(tensor),
        '"}',
    )


def basename(path: String) -> String:
    var cut = -1
    var bl = path.as_bytes()
    for i in range(len(bl)):
        if Int(bl[i]) == 47:
            cut = i
    if cut < 0:
        return path
    var out = List[UInt8]()
    for i in range(cut + 1, len(bl)):
        out.append(bl[i])
    return String(from_utf8_lossy=Span(out))


def dirname(path: String) -> String:
    var cut = -1
    var bl = path.as_bytes()
    for i in range(len(bl)):
        if Int(bl[i]) == 47:
            cut = i
    if cut < 0:
        return ""
    var out = List[UInt8]()
    for i in range(cut):
        out.append(bl[i])
    return String(from_utf8_lossy=Span(out))


def fail(code: String, detail: String, shard: String, tensor: String) raises:
    eprint_json(err_json(code, detail, shard, tensor))
    exit(2)


def err_layer_json(
    code: String, detail: String, stage: String, layer: Int
) -> String:
    return String(
        '{"error_type":"',
        json_escape(code),
        '","detail":"',
        json_escape(detail),
        '","stage":"',
        json_escape(stage),
        '","layer":',
        String(layer),
        "}",
    )


def fail_layer(code: String, detail: String, stage: String, layer: Int) raises:
    eprint_json(err_layer_json(code, detail, stage, layer))
    exit(2)
