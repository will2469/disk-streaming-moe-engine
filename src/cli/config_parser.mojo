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
from std.math import isinf, isnan


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
        if sc.eof():
            raise Error(
                '{"error_type":"JSON_PARSE_ERROR","detail":"unterminated'
                ' array","shard":"'
                + sc.shard
                + '","tensor_name":""}'
            )
        if sc.peek() == 45:
            is_neg = True
            sc.pos += 1
        # Grammar integer JSON: tanpa '+', tanpa leading zero.
        if not sc.eof() and sc.peek() == 48 and sc.pos + 1 < len(sc.buf):
            var c1 = Int(sc.buf[sc.pos + 1])
            if c1 >= 48 and c1 <= 57:
                raise Error(
                    '{"error_type":"JSON_PARSE_ERROR","detail":"leading zero'
                    ' in integer","shard":"'
                    + sc.shard
                    + '","tensor_name":""}'
                )
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
    # Kontrak probe M2 (bukan invariant engine): tepat 3 prompt × 16 token.
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
        if sc.eof():
            raise Error(
                '{"error_type":"JSON_PARSE_ERROR","detail":"unterminated'
                ' tokens.json","shard":"'
                + path
                + '","tensor_name":""}'
            )
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
        if sc.eof():
            raise Error(
                '{"error_type":"JSON_PARSE_ERROR","detail":"unterminated'
                ' tokens.json","shard":"'
                + path
                + '","tensor_name":""}'
            )
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
    # Setelah ']' penutup hanya whitespace yang sah; sampah trailing ditolak
    # (sebelumnya [[...],[...],[...]]GARBAGE diterima diam-diam).
    sc.skip_ws()
    if not sc.eof():
        raise Error(
            '{"error_type":"JSON_PARSE_ERROR","detail":"trailing characters'
            ' after tokens array","shard":"'
            + path
            + '","tensor_name":""}'
        )
    return all_tokens^


def _fail_config(detail: String) raises:
    raise Error(
        '{"error_type":"CONFIG_ERROR","detail":"'
        + detail
        + '","stage":"config"}'
    )


def _skip_cfg_ws(raw: String, pos: Int) -> Int:
    var rb = raw.as_bytes()
    var p = pos
    while p < len(rb) and (
        rb[p] == 32 or rb[p] == 9 or rb[p] == 10 or rb[p] == 13
    ):
        p += 1
    return p


def _scan_cfg_string(raw: String, pos: Int) raises -> Int:
    # pos menunjuk byte '"'; return indeks setelah '"' penutup.
    var rb = raw.as_bytes()
    var p = pos + 1
    while True:
        if p >= len(rb):
            _fail_config("unterminated string in config")
        var c = rb[p]
        if c == 34:
            return p + 1
        if c == 92:
            p += 1
            if p >= len(rb):
                _fail_config("unterminated string in config")
            p += 1
        elif c < 32:
            _fail_config("raw control character in config string")
        else:
            p += 1


def _scan_cfg_composite(raw: String, pos: Int) raises -> Int:
    # Lewati nilai objek/array tak dikenal (depth-balanced, string-aware).
    # Hanya untuk skip: ketidakseimbangan apa pun → CONFIG_ERROR (fail-closed).
    var rb = raw.as_bytes()
    var depth = 0
    var p = pos
    while p < len(rb):
        var c = rb[p]
        if c == 34:
            p = _scan_cfg_string(raw, p)
        elif c == 123 or c == 91:
            depth += 1
            p += 1
        elif c == 125 or c == 93:
            depth -= 1
            p += 1
            if depth == 0:
                return p
        else:
            p += 1
    _fail_config("unterminated composite in config")
    return p


def _scan_cfg_value_end(raw: String, pos: Int) raises -> Int:
    var rb = raw.as_bytes()
    var c = rb[pos]
    if c == 34:
        return _scan_cfg_string(raw, pos)
    if c == 123 or c == 91:
        return _scan_cfg_composite(raw, pos)
    var end = pos
    while end < len(rb) and (
        rb[end] != 44
        and rb[end] != 125
        and rb[end] != 32
        and rb[end] != 9
        and rb[end] != 10
        and rb[end] != 13
    ):
        end += 1
    return end


def _parse_config_object(raw: String) raises -> Dict[String, String]:
    # Parse objek JSON top-level → {key: substring nilai mentah}.
    # Hanya kunci depth-1 yang diakui: {"foo": {"hidden_size": 1}} TIDAK
    # mengeset hidden_size. Kunci duplikat → error (bukan occurrence
    # pertama/terakhir diam-diam). Sampah setelah '}' → error.
    var fields = Dict[String, String]()
    var rb = raw.as_bytes()
    var p = 0
    if len(rb) >= 3 and rb[0] == 239 and rb[1] == 187 and rb[2] == 191:
        p = 3
    p = _skip_cfg_ws(raw, p)
    if p >= len(rb) or rb[p] != 123:
        _fail_config("config must be a JSON object")
    p += 1
    p = _skip_cfg_ws(raw, p)
    if p < len(rb) and rb[p] == 125:
        p += 1
    else:
        while True:
            p = _skip_cfg_ws(raw, p)
            if p >= len(rb) or rb[p] != 34:
                _fail_config("expected string key in config")
            var kend = _scan_cfg_string(raw, p)
            var key = String(raw[byte = p + 1 : kend - 1])
            p = _skip_cfg_ws(raw, kend)
            if p >= len(rb) or rb[p] != 58:
                _fail_config("expected : in config")
            p = _skip_cfg_ws(raw, p + 1)
            if p >= len(rb):
                _fail_config("unexpected end in config")
            var vend = _scan_cfg_value_end(raw, p)
            if key in fields:
                _fail_config("duplicate field in config")
            fields[key] = String(raw[byte=p:vend])
            p = _skip_cfg_ws(raw, vend)
            if p >= len(rb):
                _fail_config("unterminated object in config")
            if rb[p] == 44:
                p += 1
            elif rb[p] == 125:
                p += 1
                break
            else:
                _fail_config("expected , or } in config")
    p = _skip_cfg_ws(raw, p)
    if p != len(rb):
        _fail_config("trailing characters after config object")
    return fields^


def _find_config_int(fields: Dict[String, String], field: String) raises -> Int:
    if field not in fields:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"missing required field: '
            + field
            + '","stage":"config"}'
        )
    var val = fields[field]
    var vb = val.as_bytes()
    if len(vb) == 0:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"non-integer value for '
            + field
            + '","stage":"config"}'
        )
    for i in range(len(vb)):
        if vb[i] < 48 or vb[i] > 57:
            raise Error(
                '{"error_type":"CONFIG_ERROR","detail":"non-integer value for '
                + field
                + '","stage":"config"}'
            )
    try:
        return Int(val)
    except:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"non-integer value for '
            + field
            + '","stage":"config"}'
        )


def _find_config_int_optional(
    fields: Dict[String, String], field: String, default_val: Int
) raises -> Int:
    # Tidak ada → default. Ada tapi invalid → error (fail-closed, bukan
    # pretend-missing).
    if field not in fields:
        return default_val
    return _find_config_int(fields, field)


def _skip_ascii_digits(val_str: String, pos: Int) -> Int:
    var b = val_str.as_bytes()
    var p = pos
    while p < len(b) and b[p] >= 48 and b[p] <= 57:
        p += 1
    return p


def _parse_strict_float(val_str: String, field: String) raises -> Float32:
    # Grammar angka JSON strict: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
    # harus konsumsi SELURUH token. Ini menolak "1.0garbage" (atof diam-diam
    # memotongnya jadi 1.0) dan literal non-JSON "nan"/"inf"/"Infinity".
    # atof hanya dipakai SETELAH grammar valid, lalu hasil wajib finite:
    # "1e999" grammatically valid tapi overflow jadi +inf.
    var b = val_str.as_bytes()
    var n = len(b)
    var pos = 0
    if pos < n and b[pos] == 45:
        pos += 1
    var ok = True
    if pos >= n:
        ok = False
    elif b[pos] == 48:
        pos += 1
    elif b[pos] >= 49 and b[pos] <= 57:
        pos = _skip_ascii_digits(val_str, pos)
    else:
        ok = False
    if ok and pos < n and b[pos] == 46:
        pos += 1
        var fstart = pos
        pos = _skip_ascii_digits(val_str, pos)
        if pos == fstart:
            ok = False
    if ok and pos < n and (b[pos] == 101 or b[pos] == 69):
        pos += 1
        if pos < n and (b[pos] == 43 or b[pos] == 45):
            pos += 1
        var estart = pos
        pos = _skip_ascii_digits(val_str, pos)
        if pos == estart:
            ok = False
    if not ok or pos != n:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"non-numeric float value for'
            " "
            + field
            + '","stage":"config"}'
        )
    var v = str_to_float(val_str)
    if isnan(v) or isinf(v):
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"non-finite float value for '
            + field
            + '","stage":"config"}'
        )
    return v


def _find_config_float(
    fields: Dict[String, String], field: String
) raises -> Float32:
    if field not in fields:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"missing required field: '
            + field
            + '","stage":"config"}'
        )
    return _parse_strict_float(fields[field], field)


def parse_model_config(path: String) raises -> Tuple[ModelConfig, Float32]:
    return parse_model_config_adapter(path, "trial")


def parse_model_config_adapter(
    path: String, expected_arch: String
) raises -> Tuple[ModelConfig, Float32]:
    """Parse model config dengan deteksi arsitektur dan mismatch detector (M9).

    Exit 2 (M9_ERR_ARCHITECTURE) bila --architecture mismatch terhadap checkpoint.
    Exit 3 (M9_ERR_CONFIG) bila dimensi arsitektur (vocab/layer/expert/topk) mismatch.
    """
    if expected_arch != "trial" and expected_arch != "qwen3.6":
        raise Error(
            '{"error_code":2,"error_type":"ARCHITECTURE_ERROR","detail":"unsupported'
            " architecture: "
            + expected_arch
            + '","stage":"config"}'
        )

    var raw_bytes: List[UInt8]
    try:
        raw_bytes = read_small_file(path)
    except:
        raise Error(
            '{"error_code":1,"error_type":"CONFIG_ERROR","detail":"cannot open'
            " config file: "
            + path
            + '","stage":"config"}'
        )
    var raw = String(from_utf8_lossy=Span(raw_bytes))
    var fields = _parse_config_object(raw)

    # Deteksi nested text_config (seperti pada Qwen3.6-35B-A3B)
    if "text_config" in fields:
        var text_raw = fields["text_config"]
        var tb = text_raw.as_bytes()
        if len(tb) > 2 and tb[0] == 123 and tb[len(tb) - 1] == 125:
            var text_fields = _parse_config_object(text_raw)
            var top_model_type = (
                fields["model_type"] if "model_type" in fields else ""
            )
            fields = text_fields^
            if "model_type" not in fields and top_model_type != "":
                fields["model_type"] = top_model_type

    var model_type = fields["model_type"] if "model_type" in fields else ""
    var raw_vocab = _find_config_int_optional(fields, "vocab_size", -1)
    var raw_experts = _find_config_int_optional(fields, "num_experts", -1)

    # Validasi Mismatch Arsitektur (Exit 2)
    if expected_arch == "trial":
        if (
            model_type == '"qwen3_5_moe"'
            or model_type == '"qwen3_5_moe_text"'
            or model_type == '"qwen3.6"'
            or raw_vocab == 248320
            or raw_experts == 256
        ):
            raise Error(
                '{"error_code":2,"error_type":"ARCHITECTURE_MISMATCH","detail":"checkpoint'
                " is Qwen3.6 architecture, but --architecture trial was"
                ' specified","stage":"config"}'
            )
    elif expected_arch == "qwen3.6":
        if (
            model_type == '"qwen2_moe"'
            or raw_vocab == 151936
            or (raw_experts == 60 and raw_vocab != 1024)
        ):
            raise Error(
                '{"error_code":2,"error_type":"ARCHITECTURE_MISMATCH","detail":"checkpoint'
                " is Trial architecture, but --architecture qwen3.6 was"
                ' specified","stage":"config"}'
            )

    var hidden_size = _find_config_int(fields, "hidden_size")
    var vocab_size = _find_config_int(fields, "vocab_size")
    var def_layers = 24 if expected_arch == "trial" else 40
    var num_hidden_layers = _find_config_int_optional(
        fields, "num_hidden_layers", def_layers
    )
    var num_attention_heads = _find_config_int_optional(
        fields, "num_attention_heads", 16
    )
    var eps = _find_config_float(fields, "rms_norm_eps")

    if isnan(eps) or isinf(eps) or eps <= Float32(0.0):
        raise Error(
            '{"error_code":3,"error_type":"CONFIG_ERROR","detail":"rms_norm_eps'
            ' must be finite and > 0","stage":"config"}'
        )

    var def_experts = 60 if expected_arch == "trial" else 256
    var def_topk = 4 if expected_arch == "trial" else 8
    var def_inter = 1408 if expected_arch == "trial" else 512
    var def_shared = 5632 if expected_arch == "trial" else 512

    var num_experts = _find_config_int_optional(
        fields, "num_experts", def_experts
    )
    var num_experts_per_tok = _find_config_int_optional(
        fields, "num_experts_per_tok", def_topk
    )
    var moe_intermediate_size = _find_config_int_optional(
        fields, "moe_intermediate_size", def_inter
    )
    var shared_expert_intermediate_size = _find_config_int_optional(
        fields, "shared_expert_intermediate_size", def_shared
    )

    var def_kv_heads = num_attention_heads if expected_arch == "trial" else 2
    var num_key_value_heads = _find_config_int_optional(
        fields, "num_key_value_heads", def_kv_heads
    )

    var head_dim_override = _find_config_int_optional(fields, "head_dim", 0)
    var full_attention_interval = _find_config_int_optional(
        fields, "full_attention_interval", 4
    )

    var attention_bias = False
    if "attention_bias" in fields:
        var ab = fields["attention_bias"]
        if ab == "true":
            attention_bias = True
        elif ab == "false":
            attention_bias = False
    elif expected_arch == "trial":
        attention_bias = True

    var norm_topk_prob = False
    if "norm_topk_prob" in fields:
        var val = fields["norm_topk_prob"]
        if val == "true":
            norm_topk_prob = True
        elif val == "false":
            norm_topk_prob = False
        else:
            raise Error(
                '{"error_code":3,"error_type":"CONFIG_ERROR","detail":"norm_topk_prob'
                ' must be boolean","stage":"config"}'
            )

    # Mismatch Detector (Exit 3): Validasi Dimensi Arsitektur Normatif
    if expected_arch == "qwen3.6":
        var is_canonical = (
            num_experts == 256
            or vocab_size == 248320
            or num_hidden_layers == 40
        )
        var is_mini = (
            num_experts == 8 or vocab_size == 1024 or num_hidden_layers == 4
        )
        if is_canonical:
            if num_experts != 256:
                raise Error(
                    '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"canonical'
                    " Qwen3.6 requires num_experts=256, got "
                    + String(num_experts)
                    + '","stage":"config"}'
                )
            if num_experts_per_tok != 8:
                raise Error(
                    '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"canonical'
                    " Qwen3.6 requires num_experts_per_tok=8, got "
                    + String(num_experts_per_tok)
                    + '","stage":"config"}'
                )
            if vocab_size != 248320:
                raise Error(
                    '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"canonical'
                    " Qwen3.6 requires vocab_size=248320, got "
                    + String(vocab_size)
                    + '","stage":"config"}'
                )
            if num_hidden_layers != 40:
                raise Error(
                    '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"canonical'
                    " Qwen3.6 requires num_hidden_layers=40, got "
                    + String(num_hidden_layers)
                    + '","stage":"config"}'
                )
            if num_key_value_heads != 2:
                raise Error(
                    '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"canonical'
                    " Qwen3.6 requires num_key_value_heads=2 (GQA 16Q/2KV),"
                    " got "
                    + String(num_key_value_heads)
                    + '","stage":"config"}'
                )
        elif is_mini:
            if num_experts != 8:
                raise Error(
                    '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"synthetic'
                    " mini port requires num_experts=8, got "
                    + String(num_experts)
                    + '","stage":"config"}'
                )
            if num_experts_per_tok != 2:
                raise Error(
                    '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"synthetic'
                    " mini port requires num_experts_per_tok=2, got "
                    + String(num_experts_per_tok)
                    + '","stage":"config"}'
                )
            if vocab_size != 1024:
                raise Error(
                    '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"synthetic'
                    " mini port requires vocab_size=1024, got "
                    + String(vocab_size)
                    + '","stage":"config"}'
                )
            if num_hidden_layers != 4:
                raise Error(
                    '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"synthetic'
                    " mini port requires num_hidden_layers=4, got "
                    + String(num_hidden_layers)
                    + '","stage":"config"}'
                )
        else:
            raise Error(
                '{"error_code":3,"error_type":"CONFIG_MISMATCH","detail":"Qwen3.6'
                " config does not match canonical (40L/256E/248K) or mini"
                ' (4L/8E/1K) dimensions","stage":"config"}'
            )

    var cfg = ModelConfig(
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        vocab_size,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        shared_expert_intermediate_size,
        norm_topk_prob,
        architecture=expected_arch,
        num_key_value_heads=num_key_value_heads,
        head_dim_override=head_dim_override,
        full_attention_interval=full_attention_interval,
        attention_bias=attention_bias,
    )
    return (cfg^, eps)
