# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Parser konfigurasi model dan token JSON untuk Kimo CLI."""

from cli.sys_utils import str_to_float
from core.config import ModelConfig
from format.file_io import read_small_file
from format.scanner import Scanner
from format.types import STError
from std.collections import List


def _parse_signed_int_array(mut sc: Scanner) raises -> List[Int]:
    var out = List[Int]()
    sc.expect(91)
    sc.skip_ws()
    if not sc.eof() and sc.peek() == 93:
        sc.pos += 1
        return out^
    while True:
        sc.skip_ws()
        var is_neg = False
        if not sc.eof() and sc.peek() == 45:
            is_neg = True
            sc.pos += 1
        var v = sc.parse_uint()
        if is_neg:
            out.append(-v)
        else:
            out.append(v)
        sc.skip_ws()
        if sc.eof():
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR",
                        "unterminated array",
                        sc.shard,
                        "",
                    )
                )
            )
        var b = sc.peek()
        sc.pos += 1
        if b == 93:
            break
        if b != 44:
            raise Error(
                String(
                    STError(
                        "JSON_PARSE_ERROR",
                        "expected , or ]",
                        sc.shard,
                        "",
                    )
                )
            )
    return out^


def parse_tokens_json(path: String, vocab_size: Int) raises -> List[Int]:
    var raw: List[UInt8]
    try:
        raw = read_small_file(path)
    except:
        raise Error(
            '{"error_type":"FILE_NOT_FOUND","detail":"cannot open tokens file: '
            + path
            + '","shard":"'
            + path
            + '","tensor_name":""}'
        )
    var sc = Scanner(raw^, path)
    sc.skip_ws()
    try:
        sc.expect(91)  # '['
    except:
        raise Error(
            '{"error_type":"JSON_PARSE_ERROR","detail":"tokens.json must start'
            ' with [","shard":"'
            + path
            + '","tensor_name":""}'
        )
    var all_tokens = List[Int]()
    var prompt_idx = 0
    while True:
        sc.skip_ws()
        if sc.peek() == 93:
            sc.pos += 1
            break
        var prompt_tokens: List[Int]
        try:
            prompt_tokens = _parse_signed_int_array(sc)
        except e:
            raise Error(
                '{"error_type":"JSON_PARSE_ERROR","detail":"malformed array in'
                " prompt "
                + String(prompt_idx)
                + '","shard":"'
                + path
                + '","tensor_name":""}'
            )
        if len(prompt_tokens) != 16:
            raise Error(
                '{"error_type":"TOKEN_INVALID","detail":"prompt length must be'
                " 16, got "
                + String(len(prompt_tokens))
                + '","stage":"embedding","prompt_idx":'
                + String(prompt_idx)
                + "}"
            )
        for token_pos in range(len(prompt_tokens)):
            var tid = prompt_tokens[token_pos]
            if tid < 0 or tid >= vocab_size:
                raise Error(
                    '{"error_type":"TOKEN_INVALID","detail":"Token ID '
                    + String(tid)
                    + " out of bounds [0, "
                    + String(vocab_size)
                    + ")"
                    + '","stage":"embedding","prompt_idx":'
                    + String(prompt_idx)
                    + ',"token_pos":'
                    + String(token_pos)
                    + ',"token_id":'
                    + String(tid)
                    + "}"
                )
            all_tokens.append(tid)
        prompt_idx += 1
        sc.skip_ws()
        if sc.peek() == 93:
            sc.pos += 1
            break
        if sc.peek() == 44:
            sc.pos += 1
        else:
            raise Error(
                '{"error_type":"JSON_PARSE_ERROR","detail":"expected , or ] in'
                ' tokens.json","shard":"'
                + path
                + '","tensor_name":""}'
            )
    if prompt_idx != 3:
        raise Error(
            '{"error_type":"TOKEN_INVALID","detail":"expected 3 prompts, got '
            + String(prompt_idx)
            + '","stage":"embedding"}'
        )
    return all_tokens^


def _find_config_int(raw: String, field: String) raises -> Int:
    var key = String('"', field, '"')
    var idx = raw.find(key)
    if idx < 0:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"missing required field: '
            + field
            + '","stage":"config"}'
        )
    var after = idx + len(key.as_bytes())
    var colon = raw.find(":", after)
    if colon < 0:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"malformed config near '
            + field
            + '","stage":"config"}'
        )
    var rb = raw.as_bytes()
    var start = colon + 1
    while start < len(rb) and (
        rb[start] == 32 or rb[start] == 9 or rb[start] == 10 or rb[start] == 13
    ):
        start += 1
    var end = start
    while end < len(rb) and (rb[end] >= 48 and rb[end] <= 57):
        end += 1
    if end == start:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"non-integer value for '
            + field
            + '","stage":"config"}'
        )
    var val_str = raw[byte=start:end]
    return Int(val_str)


def _find_config_int_optional(
    raw: String, field: String, default_val: Int
) -> Int:
    var key = String('"', field, '"')
    var idx = raw.find(key)
    if idx < 0:
        return default_val
    var after = idx + len(key.as_bytes())
    var colon = raw.find(":", after)
    if colon < 0:
        return default_val
    var rb = raw.as_bytes()
    var start = colon + 1
    while start < len(rb) and (
        rb[start] == 32 or rb[start] == 9 or rb[start] == 10 or rb[start] == 13
    ):
        start += 1
    var end = start
    while end < len(rb) and (rb[end] >= 48 and rb[end] <= 57):
        end += 1
    if end == start:
        return default_val
    var val_str = raw[byte=start:end]
    try:
        return Int(val_str)
    except:
        return default_val


def _find_config_float(raw: String, field: String) raises -> Float32:
    var key = String('"', field, '"')
    var idx = raw.find(key)
    if idx < 0:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"missing required field: '
            + field
            + '","stage":"config"}'
        )
    var after = idx + len(key.as_bytes())
    var colon = raw.find(":", after)
    if colon < 0:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"malformed config near '
            + field
            + '","stage":"config"}'
        )
    var rb = raw.as_bytes()
    var start = colon + 1
    while start < len(rb) and (
        rb[start] == 32 or rb[start] == 9 or rb[start] == 10 or rb[start] == 13
    ):
        start += 1
    var end = start
    while end < len(rb) and (
        rb[end] != 44
        and rb[end] != 125
        and rb[end] != 32
        and rb[end] != 10
        and rb[end] != 13
    ):
        end += 1
    if end == start:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"non-numeric float value for'
            " "
            + field
            + '","stage":"config"}'
        )
    var val_str = String(raw[byte=start:end])
    return str_to_float(val_str)


def parse_model_config(path: String) raises -> Tuple[ModelConfig, Float32]:
    var raw_bytes: List[UInt8]
    try:
        raw_bytes = read_small_file(path)
    except:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"cannot open'
            " model_config.json: "
            + path
            + '","stage":"config"}'
        )
    var raw = String(from_utf8_lossy=Span(raw_bytes))

    var hidden_size = _find_config_int(raw, "hidden_size")
    var vocab_size = _find_config_int(raw, "vocab_size")
    var num_hidden_layers = _find_config_int_optional(
        raw, "num_hidden_layers", 28
    )
    var num_attention_heads = _find_config_int_optional(
        raw, "num_attention_heads", 16
    )
    var eps = _find_config_float(raw, "rms_norm_eps")

    if eps <= Float32(0.0):
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"rms_norm_eps must be > 0",'
            '"stage":"config"}'
        )

    var cfg = ModelConfig(
        hidden_size, num_hidden_layers, num_attention_heads, vocab_size
    )
    return (cfg^, eps)
