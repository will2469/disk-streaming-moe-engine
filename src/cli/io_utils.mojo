# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Operasi file atomik dan validasi binary aktivasi untuk Kimo CLI."""

from cli.errors import err_layer_json, fail_layer
from cli.sys_utils import c_realpath, c_rename, c_unlink
from format.file_io import read_small_file
from std.builtin.dtype import DType
from std.collections import List
from std.math import abs, isinf, isnan


def atomic_write_logits(target_path: String, logits: List[Float32]) raises:
    var tmp_path = String(target_path, ".tmp.bin")
    try:
        var f = open(tmp_path, "w")
        var p_u8 = logits.unsafe_ptr().unsafe_bitcast[UInt8]()
        var span = Span(unsafe_ptr=p_u8, length=len(logits) * 4)
        f.write_bytes(span)
        f.close()
    except:
        _ = c_unlink(tmp_path)
        raise Error(
            '{"error_type":"OUTPUT_WRITE_FAILED","detail":"cannot write tmp'
            " file: "
            + tmp_path
            + '","stage":"output"}'
        )
    var ret = c_rename(tmp_path, target_path)
    if ret != 0:
        _ = c_unlink(tmp_path)
        raise Error(
            '{"error_type":"OUTPUT_WRITE_FAILED","detail":"atomic rename'
            " failed from "
            + tmp_path
            + " to "
            + target_path
            + '","stage":"output"}'
        )


def load_and_validate_activation(
    path: String,
    layer_idx: Int,
    expected_tokens: Int,
    hidden_dim: Int,
) raises -> List[Float32]:
    var real = c_realpath(path)
    if real == "":
        fail_layer(
            "FILE_NOT_FOUND",
            "activation file not found: " + path,
            "attention",
            layer_idx,
        )

    var raw_bytes = List[UInt8]()
    try:
        raw_bytes = read_small_file(path)
    except e:
        fail_layer(
            "ACT_LOAD_FAILED",
            "cannot read activation file: " + String(e),
            "attention",
            layer_idx,
        )

    var expected_elements = expected_tokens * hidden_dim
    var expected_bytes = expected_elements * 4
    if len(raw_bytes) != expected_bytes:
        fail_layer(
            "ACT_LOAD_FAILED",
            String(
                "activation size mismatch: expected ",
                expected_bytes,
                " bytes (shape [",
                expected_tokens,
                ", ",
                hidden_dim,
                "]), got ",
                len(raw_bytes),
            ),
            "attention",
            layer_idx,
        )

    var out = List[Float32]()
    out.reserve(expected_elements)
    var p_u8 = raw_bytes.unsafe_ptr()
    var p_f32 = p_u8.unsafe_bitcast[Scalar[DType.float32]]()
    for i in range(expected_elements):
        var v = p_f32[unsafe_offset=i]
        if isnan(v) or isinf(v):
            fail_layer(
                "ACT_LOAD_FAILED",
                "activation contains non-finite values (NaN or Inf) at index "
                + String(i),
                "attention",
                layer_idx,
            )
        if abs(v) > Float32(1e6):
            fail_layer(
                "ACT_LOAD_FAILED",
                "activation value out of reasonable range at index "
                + String(i),
                "attention",
                layer_idx,
            )
        out.append(v)
    return out^


def atomic_write_attn_output(
    target_path: String, output: List[Float32], layer_idx: Int
) raises:
    var tmp_path = String(target_path, ".tmp.bin")
    try:
        var f = open(tmp_path, "w")
        var p_u8 = output.unsafe_ptr().unsafe_bitcast[UInt8]()
        var span = Span(unsafe_ptr=p_u8, length=len(output) * 4)
        f.write_bytes(span)
        f.close()
    except:
        _ = c_unlink(tmp_path)
        raise Error(
            err_layer_json(
                "OUTPUT_WRITE_FAILED",
                "cannot write tmp file: " + tmp_path,
                "output",
                layer_idx,
            )
        )
    var ret = c_rename(tmp_path, target_path)
    if ret != 0:
        _ = c_unlink(tmp_path)
        raise Error(
            err_layer_json(
                "OUTPUT_WRITE_FAILED",
                "atomic rename failed from " + tmp_path + " to " + target_path,
                "output",
                layer_idx,
            )
        )


def validate_layer_output(
    out_act: List[Float32], act: List[Float32], layer_val: Int, hidden_size: Int
) raises:
    if len(out_act) != 16 * hidden_size:
        fail_layer(
            "ATTENTION_ERROR", "output size mismatch", "attention", layer_val
        )
    var max_diff = Float32(0.0)
    for idx in range(len(out_act)):
        var val = out_act[idx]
        if isnan(val) or isinf(val):
            fail_layer(
                "ATTENTION_ERROR",
                "output contains non-finite values at index " + String(idx),
                "attention",
                layer_val,
            )
        var diff = abs(val - act[idx])
        if diff > max_diff:
            max_diff = diff
    if max_diff == Float32(0.0):
        fail_layer(
            "ATTENTION_ERROR",
            "residual check failed: output is identical to input",
            "residual",
            layer_val,
        )
