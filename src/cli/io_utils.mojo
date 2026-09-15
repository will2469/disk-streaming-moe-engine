# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Operasi file atomik dan validasi binary aktivasi untuk Kimo CLI."""

from cli.errors import err_layer_json, fail_layer
from cli.sys_utils import (
    c_close_fd,
    c_fsync_fd,
    c_open_tmp_excl,
    c_realpath,
    c_rename,
    c_unlink,
    c_write_f32_fd_all,
    make_unique_tmp_path,
)
from format.file_io import read_small_file
from std.collections import List
from std.math import abs, isinf, isnan


def _write_tmp_secure(
    target_path: String, data: List[Float32]
) raises -> String:
    # Tulis data ke tmp unik via O_CREAT|O_EXCL|O_NOFOLLOW, fsync, close.
    # Return tmp_path (satu direktori dengan target -> rename tetap atomik).
    # Raise Error plain bila gagal; caller unlink sisa dan raise JSON final.
    var fd = -1
    var tmp_path = String("")
    var attempt = 0
    while attempt < 8:
        tmp_path = make_unique_tmp_path(target_path, attempt)
        fd = c_open_tmp_excl(tmp_path)
        if fd >= 0:
            break
        attempt += 1
    if fd < 0:
        raise Error(
            String("cannot create tmp file securely for: ", target_path)
        )
    if not c_write_f32_fd_all(fd, data):
        _ = c_close_fd(fd)
        _ = c_unlink(tmp_path)
        raise Error(String("cannot write tmp file: ", tmp_path))
    if c_fsync_fd(fd) != 0:
        _ = c_close_fd(fd)
        _ = c_unlink(tmp_path)
        raise Error(String("cannot fsync tmp file: ", tmp_path))
    _ = c_close_fd(fd)
    return tmp_path


def atomic_write_logits(target_path: String, logits: List[Float32]) raises:
    var tmp_path = String("")
    try:
        tmp_path = _write_tmp_secure(target_path, logits)
    except:
        if tmp_path != "":
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


def _decode_f32_le(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> Float32:
    # Decode eksplisit 4 byte little-endian → Float32. Kontrak format
    # aktivasi = LE fp32 (konvensi safetensors); tanpa bitcast pointer
    # sehingga tanpa asumsi alignment buffer maupun endianness host.
    # Eksak: tiap nilai f32 terwakili persis di f64 dan scaling 2^k persis
    # (tanpa overflow/underflow di rentang f64).
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
    for i in range(expected_elements):
        var o = i * 4
        var v = _decode_f32_le(
            raw_bytes[o], raw_bytes[o + 1], raw_bytes[o + 2], raw_bytes[o + 3]
        )
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
    var tmp_path = String("")
    try:
        tmp_path = _write_tmp_secure(target_path, output)
    except:
        if tmp_path != "":
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


def check_layer_output_sanity(
    out_act: List[Float32], act: List[Float32], layer_val: Int, hidden_size: Int
) raises:
    # Sanity check residual, BUKAN bukti numerical correctness: ukuran pas,
    # finite, dan tidak identik dengan input (blok attention ber-residual
    # wajib mengubah aktivasi). Output yang sepenuhnya salah tapi finite
    # tetap lolos di sini; correctness numerik digate oleh oracle-compare
    # eksternal (tests/integration/test_m2_w5_compare.sh + Rust compare),
    # bukan oleh fungsi ini.
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
