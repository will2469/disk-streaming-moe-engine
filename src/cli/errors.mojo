# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Utilitas penanganan error JSON dan path string untuk Kimo CLI."""

from format.types import error_json, json_escape
from std.collections import List
from std.ffi import external_call
from std.sys.terminate import exit


def eprint_json(msg: String) raises:
    # Kontrak machine-readable: stdout = protocol/result saja; error JSON
    # TIDAK PERNAH ke stdout (konsumen result=$(kimo ...) wajib steril).
    # /dev/stderr dibuka append (tanpa truncate: O_TRUNC di pipe -> ENXIO).
    # Bila /dev tak ada (chroot dsb) tapi fd 2 masih terbuka, tulis langsung
    # via write(2). Dua-duanya gagal → diam; exit code caller tetap
    # mensinyalkan failure. Selalu akhiri satu newline.
    var out = msg
    var mb = msg.as_bytes()
    if len(mb) == 0 or mb[len(mb) - 1] != 10:
        out = String(msg, "\n")
    try:
        var e = open("/dev/stderr", "a")
        e.write_all(out.as_bytes())
        e.close()
        return
    except:
        pass
    # write(2) tak raise; gagal → abaikan (caller tetap exit non-nol).
    var ob = out.as_bytes()
    var n = len(ob)
    var buf = List[UInt8]()
    for i in range(n):
        buf.append(ob[i])
    _ = external_call["write", Int](2, buf.unsafe_ptr(), n)


def err_json(
    code: String, detail: String, shard: String, tensor: String
) -> String:
    # Delegasi ke helper kanonik (satu implementasi escaping, di format).
    return error_json(code, detail, shard, tensor)


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


def err_routing_violation_json(
    detail: String, stage: String, layer: Int, token_index: Int
) -> String:
    return String(
        '{"error_type":"ROUTING_VIOLATION","detail":"',
        json_escape(detail),
        '","stage":"',
        json_escape(stage),
        '","layer":',
        String(layer),
        ',"token_index":',
        String(token_index),
        "}",
    )


def fail_routing_violation(
    detail: String, stage: String, layer: Int, token_index: Int
) raises:
    eprint_json(err_routing_violation_json(detail, stage, layer, token_index))
    exit(1)
