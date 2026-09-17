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
from format.types import decode_f32_le
from std.collections import List
from std.ffi import external_call
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


def load_and_validate_activation(
    path: String,
    layer_idx: Int,
    expected_tokens: Int,
    hidden_dim: Int,
    stage: String = "attention",
) raises -> List[Float32]:
    var real = c_realpath(path)
    if real == "":
        fail_layer(
            "FILE_NOT_FOUND",
            "activation file not found: " + path,
            stage,
            layer_idx,
        )

    var raw_bytes = List[UInt8]()
    try:
        raw_bytes = read_small_file(path)
    except e:
        fail_layer(
            "ACT_LOAD_FAILED",
            "cannot read activation file: " + String(e),
            stage,
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
            stage,
            layer_idx,
        )

    var out = List[Float32]()
    out.reserve(expected_elements)
    for i in range(expected_elements):
        var o = i * 4
        var v = decode_f32_le(
            raw_bytes[o], raw_bytes[o + 1], raw_bytes[o + 2], raw_bytes[o + 3]
        )
        if isnan(v) or isinf(v):
            fail_layer(
                "ACT_LOAD_FAILED",
                "activation contains non-finite values (NaN or Inf) at index "
                + String(i),
                stage,
                layer_idx,
            )
        if abs(v) > Float32(1e6):
            fail_layer(
                "ACT_LOAD_FAILED",
                "activation value out of reasonable range at index "
                + String(i),
                stage,
                layer_idx,
            )
        out.append(v)
    return out^


def atomic_write_layer_output(
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


def atomic_write_attn_output(
    target_path: String, output: List[Float32], layer_idx: Int
) raises:
    atomic_write_layer_output(target_path, output, layer_idx)


def atomic_write_moe_output(
    target_path: String, output: List[Float32], layer_idx: Int
) raises:
    atomic_write_layer_output(target_path, output, layer_idx)


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


def check_moe_output_sanity(
    out_act: List[Float32], act: List[Float32], layer_val: Int, hidden_size: Int
) raises:
    if len(out_act) != 16 * hidden_size:
        fail_layer("EXPERT_ERROR", "output size mismatch", "output", layer_val)
    var max_diff = Float32(0.0)
    for idx in range(len(out_act)):
        var val = out_act[idx]
        if isnan(val) or isinf(val):
            fail_layer(
                "EXPERT_ERROR",
                "output contains non-finite values at index " + String(idx),
                "output",
                layer_val,
            )
        var diff = abs(val - act[idx])
        if diff > max_diff:
            max_diff = diff
    if max_diff == Float32(0.0):
        fail_layer(
            "EXPERT_ERROR",
            "residual check failed: output is identical to input",
            "residual",
            layer_val,
        )


def parse_flat_u32_tokens(
    raw: List[UInt8], tokens_path: String
) raises -> List[Int]:
    """Parse JSON array of unsigned 32-bit integer token IDs."""
    var n = len(raw)
    var pos = 0

    # Lewati whitespace di awal
    while pos < n and (
        raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
    ):
        pos += 1

    if pos >= n or raw[pos] != 91:  # '['
        raise Error("tokens JSON must start with '['")
    pos += 1

    var out = List[Int]()

    # Cek array kosong ']'
    while pos < n and (
        raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
    ):
        pos += 1

    if pos < n and raw[pos] == 93:  # ']'
        pos += 1
        while pos < n and (
            raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
        ):
            pos += 1
        if pos < n:
            raise Error("trailing characters after closing ']'")
        return out^

    while True:
        while pos < n and (
            raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
        ):
            pos += 1

        if pos >= n:
            raise Error("unterminated tokens array (missing ']')")

        var b = raw[pos]
        if b == 91:
            raise Error("nested array not allowed, expected flat u32 array")
        if b == 45:
            raise Error("negative integer not allowed, expected unsigned u32")
        if b == 34:
            raise Error("string element not allowed, expected unsigned u32")
        if b < 48 or b > 57:
            raise Error("expected unsigned integer digit")

        # Cek leading zero
        if (
            b == 48
            and pos + 1 < n
            and (raw[pos + 1] >= 48 and raw[pos + 1] <= 57)
        ):
            raise Error("leading zero in integer is not valid JSON")

        var val = 0
        while pos < n and (raw[pos] >= 48 and raw[pos] <= 57):
            var d = Int(raw[pos] - 48)
            if val > 429496729 or (val == 429496729 and d > 5):
                raise Error("token integer exceeds u32 range")
            val = val * 10 + d
            pos += 1

        # Cek float
        if pos < n and (raw[pos] == 46 or raw[pos] == 101 or raw[pos] == 69):
            raise Error("float not allowed, expected integer")

        out.append(val)

        while pos < n and (
            raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
        ):
            pos += 1

        if pos >= n:
            raise Error("unterminated tokens array (missing ']')")

        var next_c = raw[pos]
        pos += 1
        if next_c == 93:  # ']'
            break
        elif next_c == 44:  # ','
            var p_peek = pos
            while p_peek < n and (
                raw[p_peek] == 32
                or raw[p_peek] == 9
                or raw[p_peek] == 10
                or raw[p_peek] == 13
            ):
                p_peek += 1
            if p_peek < n and raw[p_peek] == 93:
                raise Error("trailing comma not allowed in JSON array")
        else:
            raise Error("expected ',' or ']' after token integer")

    while pos < n and (
        raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
    ):
        pos += 1

    if pos < n:
        raise Error("trailing characters after closing ']'")

    return out^


def atomic_write_tokens_json(target_path: String, tokens: List[Int]) raises:
    """Atomic write list of token IDs to formatted JSON file with rollback protection.
    """
    var tmp_path: String
    var attempt = 0
    while attempt < 8:
        tmp_path = make_unique_tmp_path(target_path, attempt)
        var fd = c_open_tmp_excl(tmp_path)
        if fd >= 0:
            _ = c_close_fd(fd)
            var json_str = String("[\n")
            for i in range(len(tokens)):
                json_str += String("  ", tokens[i])
                if i + 1 < len(tokens):
                    json_str += String(",\n")
                else:
                    json_str += String("\n")
            json_str += String("]\n")
            try:
                var f = open(tmp_path, "w")
                f.write(json_str)
                f.close()
            except:
                _ = c_unlink(tmp_path)
                raise Error(String("cannot write tmp tokens file: ", tmp_path))
            var ren_ret = c_rename(tmp_path, target_path)
            if ren_ret != 0:
                _ = c_unlink(tmp_path)
                raise Error(
                    String(
                        "atomic rename failed from ",
                        tmp_path,
                        " to ",
                        target_path,
                    )
                )
            return
        attempt += 1
    raise Error(
        String("cannot create tmp tokens file securely for: ", target_path)
    )
