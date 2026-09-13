# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Kimo CLI — check-index (M0), head path (M1), and layer attention (M2)."""

from model import (
    AttentionWeights,
    HeadWeights,
    LoadMemoryTelemetry,
    ModelConfig,
    forward_attention_block,
    forward_head,
    load_layer_attention_weights,
    load_tensor_f32_chunked,
    validate_bias_count,
    validate_logits,
)
from safetensors import (
    STError,
    Scanner,
    STHeader,
    TensorMeta,
    json_escape,
    parse_index,
    read_header,
    read_small_file,
)
from std.builtin.dtype import DType
from std.collections import Dict, List
from std.ffi import external_call
from std.math import abs, isfinite, isinf, isnan
from std.sys.arg import argv
from std.sys.terminate import exit
from std.time import perf_counter_ns


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


def cmd_check_index(shards: List[String]) raises:
    var t0 = perf_counter_ns()
    if len(shards) < 1:
        fail("USAGE", "butuh N≥1 path shard", "", "")
    var d = dirname(shards[0])
    var index_path = String(d, "/model.safetensors.index.json")
    if d == "":
        index_path = String("model.safetensors.index.json")
    var packed = List[String]()
    try:
        packed = parse_index(index_path)
    except e:
        eprint_json(String(e))
        exit(2)
    var nn = 0
    # packed[0] = count as String -> parse manual
    var cs = packed[0].as_bytes()
    for i in range(len(cs)):
        nn = nn * 10 + (Int(cs[i]) - 48)
    var wnames = List[String]()
    var wfiles = List[String]()
    var wpos = Dict[String, Int]()
    for i in range(nn):
        wnames.append(packed[1 + 2 * i])
        wfiles.append(packed[1 + 2 * i + 1])
        wpos[wnames[i]] = i
    # baca header semua shard
    var headers = List[STHeader]()
    for i in range(len(shards)):
        try:
            var st = read_header(shards[i])
            headers.append(st^)
        except e:
            eprint_json(String(e))
            exit(2)
    # scope deterministik: full iff semua file unik weight_map ada di argumen
    var supplied = List[String]()
    for i in range(len(shards)):
        supplied.append(basename(shards[i]))
    var full = True
    for i in range(nn):
        var found = False
        for j in range(len(supplied)):
            if supplied[j] == wfiles[i]:
                found = True
        if not found:
            full = False
    var scope = String("subset")
    if full:
        scope = String("full")
    # peta global nama -> file aktual + deteksi DUPLICATE_TENSOR_NAME (O(1) via Dict)
    var gnames = List[String]()
    var gfiles = List[String]()
    var gpos = Dict[String, Int]()
    var mismatch = List[String]()
    for i in range(len(headers)):
        var base = basename(shards[i])
        ref st = headers[i]
        for k in range(len(st.entries)):
            ref e = st.entries[k]
            if e.name in gpos:
                var seen = gpos[e.name]
                mismatch.append(
                    String(
                        '{"tensor_name":"',
                        json_escape(e.name),
                        '","kind":"DUPLICATE_TENSOR_NAME","files":["',
                        gfiles[seen],
                        '","',
                        base,
                        '"]}',
                    )
                )
            else:
                gpos[e.name] = len(gnames)
                gnames.append(e.name)
                gfiles.append(base)
    # compare vs weight_map (mode full + subset; subset tak boleh sembunyikan salah tempat).
    # total_tensors = len(weight_map) SELALU; assessed = yang dinilai; matched ⊆ assessed.
    var assessed = 0
    var matched = 0
    for i in range(nn):
        var gi = -1
        if wnames[i] in gpos:
            gi = gpos[wnames[i]]
        var exp_in = False
        for j in range(len(supplied)):
            if supplied[j] == wfiles[i]:
                exp_in = True
        if exp_in:
            assessed += 1
            if gi < 0:
                mismatch.append(
                    String(
                        '{"tensor_name":"',
                        json_escape(wnames[i]),
                        '","kind":"MISSING_IN_SHARD","expected_shard":"',
                        wfiles[i],
                        '"}',
                    )
                )
            elif gfiles[gi] != wfiles[i]:
                mismatch.append(
                    String(
                        '{"tensor_name":"',
                        json_escape(wnames[i]),
                        '","kind":"WRONG_SHARD","expected_shard":"',
                        wfiles[i],
                        '","found_shard":"',
                        gfiles[gi],
                        '"}',
                    )
                )
            else:
                matched += 1
        elif gi >= 0:
            # subset: harapan di luar subset, tapi fisik ditemukan di pasokan -> tetap mismatch
            assessed += 1
            mismatch.append(
                String(
                    '{"tensor_name":"',
                    json_escape(wnames[i]),
                    '","kind":"WRONG_SHARD","expected_shard":"',
                    wfiles[i],
                    '","found_shard":"',
                    gfiles[gi],
                    '"}',
                )
            )
        # else: di luar subset dan tak ditemukan -> tidak dinilai
    # MISSING_IN_INDEX: nama di header pasokan tapi ∉ weight_map
    for g in range(len(gnames)):
        if gnames[g] not in wpos:
            mismatch.append(
                String(
                    '{"tensor_name":"',
                    json_escape(gnames[g]),
                    '","kind":"MISSING_IN_INDEX","found_shard":"',
                    gfiles[g],
                    '"}',
                )
            )
    var ms = Float64(perf_counter_ns() - t0) / 1000000.0
    var sup_json = String("")
    for j in range(len(supplied)):
        if j > 0:
            sup_json += ","
        sup_json += String('"', supplied[j], '"')
    var mm_json = String("")
    for j in range(len(mismatch)):
        if j > 0:
            mm_json += ","
        mm_json += mismatch[j]
    var status = String("mismatch")
    if len(mismatch) == 0:
        status = String("match")
    print(
        String(
            '{"status":"',
            status,
            '","scope":"',
            scope,
            '","supplied_shards":[',
            sup_json,
            '],"total_tensors":',
            nn,
            ',"assessed_tensors":',
            assessed,
            ',"matched_tensors":',
            matched,
            ',"mismatches":[',
            mm_json,
            '],"parse_time_ms":',
            ms,
            "}",
        )
    )
    if len(mismatch) == 0:
        exit(0)
    exit(1)


def c_rename(oldpath: String, newpath: String) -> Int:
    var old_s = oldpath.as_bytes()
    var new_s = newpath.as_bytes()
    var old_z = List[UInt8]()
    for i in range(len(old_s)):
        old_z.append(old_s[i])
    old_z.append(0)
    var new_z = List[UInt8]()
    for j in range(len(new_s)):
        new_z.append(new_s[j])
    new_z.append(0)
    return Int(
        external_call["rename", Int32](old_z.unsafe_ptr(), new_z.unsafe_ptr())
    )


def c_unlink(path: String) -> Int:
    var p = path.as_bytes()
    var p_z = List[UInt8]()
    for i in range(len(p)):
        p_z.append(p[i])
    p_z.append(0)
    return Int(external_call["unlink", Int32](p_z.unsafe_ptr()))


def c_realpath(path: String) -> String:
    var p = path.as_bytes()
    var p_z = List[UInt8]()
    for i in range(len(p)):
        p_z.append(p[i])
    p_z.append(0)
    var buf = List[UInt8]()
    for _ in range(4096):
        buf.append(0)
    var res = external_call["realpath", Int](p_z.unsafe_ptr(), buf.unsafe_ptr())
    if res == 0:
        return ""
    var n = 0
    while n < 4096 and buf[n] != 0:
        n += 1
    var out = List[UInt8]()
    for i in range(n):
        out.append(buf[i])
    return String(from_utf8_lossy=Span(out))


def str_to_float(s: String) -> Float32:
    var sb = s.as_bytes()
    var z = List[UInt8]()
    for i in range(len(sb)):
        z.append(sb[i])
    z.append(0)
    var f64 = external_call["atof", Float64](z.unsafe_ptr())
    return Float32(f64)


def get_vmhwm_bytes() -> Int:
    try:
        var f = open("/proc/self/status", "r")
        var content = f.read()
        f.close()
        var lines = content.split("\n")
        for i in range(len(lines)):
            var line = lines[i]
            if line.startswith("VmHWM:"):
                var parts = line.split()
                if len(parts) >= 2:
                    return Int(parts[1]) * 1024
    except:
        pass
    return 0


def get_proc_io_read_bytes() -> Int:
    try:
        var f = open("/proc/self/io", "r")
        var content = f.read()
        f.close()
        var lines = content.split("\n")
        for i in range(len(lines)):
            var line = lines[i]
            if line.startswith("read_bytes:"):
                var parts = line.split()
                if len(parts) >= 2:
                    return Int(parts[1])
    except:
        pass
    return 0


def _in_list(list: List[String], item: String) -> Bool:
    for i in range(len(list)):
        if list[i] == item:
            return True
    return False


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


def cmd_head(args: List[String]) raises:
    if len(args) < 3:
        fail(
            "USAGE",
            (
                "pakai: kimo head tokens.json (--model-dir <dir> | <shard...>)"
                " [--output <path>] [--workdir <dir>]"
            ),
            "",
            "",
        )
    var tokens_path = String(args[2])
    var model_dir = String("")
    var output_file = String("logits_mojo.bin")
    var workdir = String("")
    var supplied_shards = List[String]()

    var i = 3
    while i < len(args):
        var a = String(args[i])
        if a == "--model-dir":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --model-dir", "", "")
            model_dir = String(args[i + 1])
            i += 2
        elif a == "--output":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --output", "", "")
            output_file = String(args[i + 1])
            i += 2
        elif a == "--workdir":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --workdir", "", "")
            workdir = String(args[i + 1])
            i += 2
        elif a.startswith("-"):
            fail("USAGE", "unknown option: " + a, "", "")
        else:
            supplied_shards.append(a)
            i += 1

    var workdir_canon = c_realpath(workdir if workdir != "" else ".")
    if workdir_canon == "":
        fail(
            "OUTPUT_WRITE_FAILED",
            "workdir does not exist: " + (workdir if workdir != "" else "."),
            "",
            "",
        )

    var target_output = output_file
    if not output_file.startswith("/"):
        target_output = String(workdir_canon, "/", output_file)

    var out_parent = dirname(target_output)
    if out_parent == "":
        out_parent = workdir_canon
    var parent_canon = c_realpath(out_parent)
    if parent_canon == "" or not parent_canon.startswith(workdir_canon):
        fail(
            "OUTPUT_WRITE_FAILED",
            "output path escapes workdir: " + target_output,
            "",
            "",
        )

    var model_root: String
    var is_model_dir_mode = False
    if model_dir != "":
        model_root = model_dir
        is_model_dir_mode = True
    else:
        if len(supplied_shards) < 1:
            fail(
                "USAGE",
                "butuh --model-dir <dir> atau N>=1 shard safetensors",
                "",
                "",
            )
        var r0 = dirname(supplied_shards[0])
        if r0 == "":
            r0 = "."
        for k in range(len(supplied_shards)):
            var rk = dirname(supplied_shards[k])
            if rk == "":
                rk = "."
            if rk != r0:
                fail(
                    "FILE_NOT_FOUND",
                    "all supplied shards must reside in the same directory",
                    supplied_shards[k],
                    "",
                )
        model_root = r0

    var t_parse0 = perf_counter_ns()

    var index_path = String(model_root, "/model.safetensors.index.json")
    var packed = parse_index(index_path)
    var num_map = 0
    var cs = packed[0].as_bytes()
    for ci in range(len(cs)):
        num_map = num_map * 10 + (Int(cs[ci]) - 48)
    var weight_map = Dict[String, String]()
    for w in range(num_map):
        weight_map[packed[1 + 2 * w]] = packed[1 + 2 * w + 1]

    var config_path = String(model_root, "/model_config.json")
    if c_realpath(config_path) == "":
        var alt_cfg = String(model_root, "/config.json")
        if c_realpath(alt_cfg) != "":
            config_path = alt_cfg
    var cfg_tuple = parse_model_config(config_path)
    var cfg = cfg_tuple[0].copy()
    var eps = cfg_tuple[1]

    var req_embed = "model.embed_tokens.weight"
    var req_norm = "model.norm.weight"
    var req_head = "lm_head.weight"

    if (
        req_embed not in weight_map
        or req_norm not in weight_map
        or req_head not in weight_map
    ):
        fail(
            "WEIGHT_LOAD_FAILED",
            "Required tensors not present in index weight_map",
            "",
            "",
        )

    var shard_embed_name = weight_map[req_embed]
    var shard_norm_name = weight_map[req_norm]
    var shard_head_name = weight_map[req_head]

    var shard_embed_path = String(model_root, "/", shard_embed_name)
    var shard_norm_path = String(model_root, "/", shard_norm_name)
    var shard_head_path = String(model_root, "/", shard_head_name)

    if not is_model_dir_mode:
        for si in range(len(supplied_shards)):
            var real_p = c_realpath(supplied_shards[si])
            if real_p == "":
                fail(
                    "FILE_NOT_FOUND",
                    "supplied shard not found on disk: " + supplied_shards[si],
                    supplied_shards[si],
                    "",
                )

        var supplied_bases = List[String]()
        for si in range(len(supplied_shards)):
            supplied_bases.append(basename(supplied_shards[si]))

        var missing_tensors = List[String]()
        var expected_shards = List[String]()
        if not _in_list(supplied_bases, shard_embed_name):
            missing_tensors.append(req_embed)
            if not _in_list(expected_shards, shard_embed_name):
                expected_shards.append(shard_embed_name)
        if not _in_list(supplied_bases, shard_norm_name):
            missing_tensors.append(req_norm)
            if not _in_list(expected_shards, shard_norm_name):
                expected_shards.append(shard_norm_name)
        if not _in_list(supplied_bases, shard_head_name):
            missing_tensors.append(req_head)
            if not _in_list(expected_shards, shard_head_name):
                expected_shards.append(shard_head_name)

        if len(missing_tensors) > 0:
            var mt_json = String("")
            for mi in range(len(missing_tensors)):
                if mi > 0:
                    mt_json += ","
                mt_json += String('"', missing_tensors[mi], '"')
            var es_json = String("")
            for ei in range(len(expected_shards)):
                if ei > 0:
                    es_json += ","
                es_json += String('"', expected_shards[ei], '"')
            eprint_json(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"Required tensors'
                " not covered by supplied"
                ' shards","stage":"embedding","missing_tensors":['
                + mt_json
                + '],"expected_shards":['
                + es_json
                + "]}"
            )
            exit(2)

        # Check existence on disk of all supplied shards
        for si in range(len(supplied_shards)):
            var sp = supplied_shards[si]
            if c_realpath(sp) == "":
                fail("FILE_NOT_FOUND", "shard file not found: " + sp, sp, "")
    else:
        # Check existence of required shards in model_dir
        if c_realpath(shard_embed_path) == "":
            fail(
                "FILE_NOT_FOUND",
                "shard file not found: " + shard_embed_path,
                shard_embed_path,
                "",
            )
        if c_realpath(shard_norm_path) == "":
            fail(
                "FILE_NOT_FOUND",
                "shard file not found: " + shard_norm_path,
                shard_norm_path,
                "",
            )
        if c_realpath(shard_head_path) == "":
            fail(
                "FILE_NOT_FOUND",
                "shard file not found: " + shard_head_path,
                shard_head_path,
                "",
            )

    var h_embed = read_header(shard_embed_path)
    var h_norm = read_header(shard_norm_path)
    var h_head = read_header(shard_head_path)

    var meta_embed = TensorMeta("", "", List[Int](), 0, 0)
    var found_embed = False
    for i in range(len(h_embed.entries)):
        ref e = h_embed.entries[i]
        if e.name == req_embed:
            meta_embed = e.copy()
            found_embed = True
            break
    if not found_embed:
        fail(
            "WEIGHT_LOAD_FAILED",
            "tensor missing from shard header",
            shard_embed_path,
            req_embed,
        )

    var meta_norm = TensorMeta("", "", List[Int](), 0, 0)
    var found_norm = False
    for i in range(len(h_norm.entries)):
        ref e = h_norm.entries[i]
        if e.name == req_norm:
            meta_norm = e.copy()
            found_norm = True
            break
    if not found_norm:
        fail(
            "WEIGHT_LOAD_FAILED",
            "tensor missing from shard header",
            shard_norm_path,
            req_norm,
        )

    var meta_head = TensorMeta("", "", List[Int](), 0, 0)
    var found_head = False
    for i in range(len(h_head.entries)):
        ref e = h_head.entries[i]
        if e.name == req_head:
            meta_head = e.copy()
            found_head = True
            break
    if not found_head:
        fail(
            "WEIGHT_LOAD_FAILED",
            "tensor missing from shard header",
            shard_head_path,
            req_head,
        )

    if (
        len(meta_embed.shape) != 2
        or meta_embed.shape[0] != cfg.vocab_size
        or meta_embed.shape[1] != cfg.hidden_size
    ):
        fail(
            "WEIGHT_LOAD_FAILED",
            "embed shape mismatch vs config",
            shard_embed_path,
            req_embed,
        )
    if len(meta_norm.shape) != 1 or meta_norm.shape[0] != cfg.hidden_size:
        fail(
            "WEIGHT_LOAD_FAILED",
            "norm shape mismatch vs config",
            shard_norm_path,
            req_norm,
        )
    if (
        len(meta_head.shape) != 2
        or meta_head.shape[0] != cfg.vocab_size
        or meta_head.shape[1] != cfg.hidden_size
    ):
        fail(
            "WEIGHT_LOAD_FAILED",
            "head shape mismatch vs config",
            shard_head_path,
            req_head,
        )

    var tokens = parse_tokens_json(tokens_path, cfg.vocab_size)

    var telemetry = LoadMemoryTelemetry()
    var embed_t = load_tensor_f32_chunked(
        shard_embed_path, h_embed.data_base, meta_embed, telemetry
    )
    var norm_t = load_tensor_f32_chunked(
        shard_norm_path, h_norm.data_base, meta_norm, telemetry
    )
    var head_t = load_tensor_f32_chunked(
        shard_head_path, h_head.data_base, meta_head, telemetry
    )
    var weights = HeadWeights(embed_t^, norm_t^, head_t^)

    var parse_time_ms = Float64(perf_counter_ns() - t_parse0) / 1000000.0

    var t_comp0 = perf_counter_ns()
    var logits = forward_head(tokens, weights, cfg, eps)
    validate_logits(logits, 3, 16, cfg.vocab_size)
    var compute_time_ms = Float64(perf_counter_ns() - t_comp0) / 1000000.0

    atomic_write_logits(target_output, logits)

    var vmhwm = get_vmhwm_bytes()
    if vmhwm == 0:
        vmhwm = (
            telemetry.resident_target_bytes + telemetry.conversion_buffer_bytes
        )

    print(
        String(
            '{"status":"success","num_prompts":3,"tokens_per_prompt":16,"num_tokens_total":48,"vocab_size":',
            cfg.vocab_size,
            ',"output_file":"',
            json_escape(output_file),
            '","workdir":"',
            json_escape(workdir_canon),
            '","parse_time_ms":',
            parse_time_ms,
            ',"compute_time_ms":',
            compute_time_ms,
            ',"memory":{"resident_target_bytes":',
            telemetry.resident_target_bytes,
            ',"conversion_buffer_bytes":',
            telemetry.conversion_buffer_bytes,
            ',"source_buffer_bytes":',
            telemetry.source_buffer_bytes,
            ',"vmhwm_bytes":',
            vmhwm,
            ',"io_read_bytes":',
            get_proc_io_read_bytes(),
            "}}",
        )
    )
    exit(0)


def cmd_layer(args: List[String]) raises:
    var layer_str = String("")
    var model_dir = String("")
    var output_file = String("attn_output.bin")
    var workdir = String("")
    var positionals = List[String]()

    var i = 2
    while i < len(args):
        var a = String(args[i])
        if a == "--layer":
            if i + 1 >= len(args):
                fail_layer(
                    "LAYER_INVALID",
                    "missing argument for --layer",
                    "attention",
                    -1,
                )
            layer_str = String(args[i + 1])
            i += 2
        elif a == "--model-dir":
            if i + 1 >= len(args):
                fail_layer(
                    "WEIGHT_LOAD_FAILED",
                    "missing argument for --model-dir",
                    "attention",
                    -1,
                )
            model_dir = String(args[i + 1])
            i += 2
        elif a == "--output":
            if i + 1 >= len(args):
                fail_layer(
                    "OUTPUT_WRITE_FAILED",
                    "missing argument for --output",
                    "output",
                    -1,
                )
            output_file = String(args[i + 1])
            i += 2
        elif a == "--workdir":
            if i + 1 >= len(args):
                fail_layer(
                    "OUTPUT_WRITE_FAILED",
                    "missing argument for --workdir",
                    "output",
                    -1,
                )
            workdir = String(args[i + 1])
            i += 2
        elif a.startswith("-"):
            fail_layer("LAYER_INVALID", "unknown option: " + a, "attention", -1)
        else:
            positionals.append(a)
            i += 1

    if layer_str == "":
        fail_layer(
            "LAYER_INVALID",
            "missing required --layer argument",
            "attention",
            -1,
        )

    var layer_val = 0
    var is_neg = False
    var lb = List[UInt8]()
    var sb = layer_str.as_bytes()
    for b_idx in range(len(sb)):
        lb.append(sb[b_idx])
    var start_k = 0
    if len(lb) > 0 and Int(lb[0]) == 45:  # '-'
        is_neg = True
        start_k = 1
    if len(lb) == 0 or (is_neg and len(lb) == 1):
        fail_layer(
            "LAYER_INVALID",
            "invalid layer number format: " + layer_str,
            "attention",
            -1,
        )
    for k in range(start_k, len(lb)):
        var c = Int(lb[k])
        if c < 48 or c > 57:
            fail_layer(
                "LAYER_INVALID",
                "invalid layer number format: " + layer_str,
                "attention",
                -1,
            )
        layer_val = layer_val * 10 + (c - 48)
    if is_neg:
        layer_val = -layer_val

    if layer_val != 0 and layer_val != 12 and layer_val != 23:
        fail_layer(
            "LAYER_INVALID",
            "layer number invalid (must be 0, 12, or 23 for M2): "
            + String(layer_val),
            "attention",
            layer_val,
        )

    if len(positionals) < 1:
        fail_layer(
            "ACT_LOAD_FAILED",
            "missing activation input file argument",
            "attention",
            layer_val,
        )

    var activation_path = positionals[0]
    var supplied_shards = List[String]()
    for si in range(1, len(positionals)):
        supplied_shards.append(positionals[si])

    var workdir_canon = c_realpath(workdir if workdir != "" else ".")
    if workdir_canon == "":
        fail_layer(
            "OUTPUT_WRITE_FAILED",
            "workdir does not exist: " + (workdir if workdir != "" else "."),
            "output",
            layer_val,
        )

    var target_output = output_file
    if not output_file.startswith("/"):
        target_output = String(workdir_canon, "/", output_file)

    var out_parent = dirname(target_output)
    if out_parent == "":
        out_parent = workdir_canon
    var parent_canon = c_realpath(out_parent)
    if parent_canon == "" or not parent_canon.startswith(workdir_canon):
        fail_layer(
            "OUTPUT_WRITE_FAILED",
            "output path escapes workdir: " + target_output,
            "output",
            layer_val,
        )

    var model_root: String
    var is_model_dir_mode = False
    if model_dir != "":
        if c_realpath(model_dir) == "":
            fail_layer(
                "FILE_NOT_FOUND",
                "model directory not found: " + model_dir,
                "attention",
                layer_val,
            )
        model_root = model_dir
        is_model_dir_mode = True
    else:
        if len(supplied_shards) < 1:
            fail_layer(
                "WEIGHT_LOAD_FAILED",
                (
                    "must provide --model-dir <dir> or at least one shard"
                    " safetensors"
                ),
                "attention",
                layer_val,
            )
        var r0 = dirname(supplied_shards[0])
        if r0 == "":
            r0 = "."
        for k in range(len(supplied_shards)):
            var rk = dirname(supplied_shards[k])
            if rk == "":
                rk = "."
            if rk != r0:
                fail_layer(
                    "FILE_NOT_FOUND",
                    "all supplied shards must reside in the same directory",
                    "attention",
                    layer_val,
                )
        model_root = r0

    var t_parse0 = perf_counter_ns()

    var index_path = String(model_root, "/model.safetensors.index.json")
    if c_realpath(index_path) == "":
        fail_layer(
            "FILE_NOT_FOUND",
            "index file not found: " + index_path,
            "attention",
            layer_val,
        )

    var packed = parse_index(index_path)
    var num_map = 0
    var cs = packed[0].as_bytes()
    for ci in range(len(cs)):
        num_map = num_map * 10 + (Int(cs[ci]) - 48)
    var weight_map = Dict[String, String]()
    for w in range(num_map):
        weight_map[packed[1 + 2 * w]] = packed[1 + 2 * w + 1]

    var config_path = String(model_root, "/model_config.json")
    if c_realpath(config_path) == "":
        var alt_cfg = String(model_root, "/config.json")
        if c_realpath(alt_cfg) != "":
            config_path = alt_cfg
    if c_realpath(config_path) == "":
        fail_layer(
            "FILE_NOT_FOUND",
            "config file not found in: " + model_root,
            "attention",
            layer_val,
        )

    var cfg_tuple = parse_model_config(config_path)
    var cfg = cfg_tuple[0].copy()
    var eps = cfg_tuple[1]

    if layer_val >= cfg.num_hidden_layers:
        fail_layer(
            "LAYER_INVALID",
            String(
                "layer index ",
                layer_val,
                " exceeds num_hidden_layers ",
                cfg.num_hidden_layers,
            ),
            "attention",
            layer_val,
        )

    if cfg.num_hidden_layers == 24:
        var bias_count = 0
        for l_i in range(cfg.num_hidden_layers):
            var pfx = "model.layers." + String(l_i) + ".self_attn."
            if (pfx + "q_proj.bias") in weight_map:
                bias_count += 1
            if (pfx + "k_proj.bias") in weight_map:
                bias_count += 1
            if (pfx + "v_proj.bias") in weight_map:
                bias_count += 1
        if bias_count != 72:
            fail_layer(
                "WEIGHT_LOAD_FAILED",
                String(
                    "attention bias count mismatch: expected 72, got ",
                    bias_count,
                ),
                "attention",
                layer_val,
            )

    var prefix = "model.layers." + String(layer_val) + "."
    var req_norm = prefix + "input_layernorm.weight"
    var req_wq = prefix + "self_attn.q_proj.weight"
    var req_bq = prefix + "self_attn.q_proj.bias"
    var req_wk = prefix + "self_attn.k_proj.weight"
    var req_bk = prefix + "self_attn.k_proj.bias"
    var req_wv = prefix + "self_attn.v_proj.weight"
    var req_bv = prefix + "self_attn.v_proj.bias"
    var req_wo = prefix + "self_attn.o_proj.weight"

    var req_list = List[String]()
    req_list.append(req_norm)
    req_list.append(req_wq)
    req_list.append(req_bq)
    req_list.append(req_wk)
    req_list.append(req_bk)
    req_list.append(req_wv)
    req_list.append(req_bv)
    req_list.append(req_wo)

    for ri in range(len(req_list)):
        var rn = req_list[ri]
        if rn not in weight_map:
            fail_layer(
                "WEIGHT_LOAD_FAILED",
                "required tensor not in weight_map: " + rn,
                "attention",
                layer_val,
            )

    if not is_model_dir_mode:
        for si in range(len(supplied_shards)):
            var sp = supplied_shards[si]
            if c_realpath(sp) == "":
                fail_layer(
                    "FILE_NOT_FOUND",
                    "supplied shard not found on disk: " + sp,
                    "attention",
                    layer_val,
                )

        var supplied_bases = List[String]()
        for si in range(len(supplied_shards)):
            supplied_bases.append(basename(supplied_shards[si]))

        var missing_tensors = List[String]()
        var expected_shards = List[String]()
        for ri in range(len(req_list)):
            var rn = req_list[ri]
            var sh_name = weight_map[rn]
            if not _in_list(supplied_bases, sh_name):
                missing_tensors.append(rn)
                if not _in_list(expected_shards, sh_name):
                    expected_shards.append(sh_name)

        if len(missing_tensors) > 0:
            var mt_json = String("")
            for mi in range(len(missing_tensors)):
                if mi > 0:
                    mt_json += ","
                mt_json += String('"', missing_tensors[mi], '"')
            var es_json = String("")
            for ei in range(len(expected_shards)):
                if ei > 0:
                    es_json += ","
                es_json += String('"', expected_shards[ei], '"')
            eprint_json(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"Required tensors'
                ' not covered by supplied shards","stage":"attention","layer":'
                + String(layer_val)
                + ',"missing_tensors":['
                + mt_json
                + '],"expected_shards":['
                + es_json
                + "]}"
            )
            exit(2)

    var act = load_and_validate_activation(
        activation_path, layer_val, 16, cfg.hidden_size
    )

    var telemetry = LoadMemoryTelemetry()
    var weights = load_layer_attention_weights(
        layer_val, model_root, weight_map, cfg, telemetry
    )
    var parse_time_ms = Float64(perf_counter_ns() - t_parse0) / 1000000.0

    var t_comp0 = perf_counter_ns()
    var out_act = forward_attention_block(
        act,
        weights,
        16,
        cfg,
        eps,
        pos_offset=0,
        base=Float32(1000000.0),
        layer_idx=layer_val,
    )
    var compute_time_ms = Float64(perf_counter_ns() - t_comp0) / 1000000.0

    if len(out_act) != 16 * cfg.hidden_size:
        fail_layer(
            "ATTENTION_ERROR",
            "output size mismatch",
            "attention",
            layer_val,
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

    atomic_write_attn_output(target_output, out_act, layer_val)

    print(
        String(
            '{"status":"success","layer":',
            String(layer_val),
            ',"num_tokens":16,"output_file":"',
            json_escape(output_file),
            '","parse_time_ms":',
            String(parse_time_ms),
            ',"compute_time_ms":',
            String(compute_time_ms),
            "}",
        )
    )
    exit(0)


def main() raises:
    var args = argv()
    if len(args) < 2:
        fail("USAGE", "pakai: kimo (check-index|head|layer) ...", "", "")
    var cmd = String(args[1])
    if cmd == "check-index":
        var shards = List[String]()
        for i in range(2, len(args)):
            shards.append(String(args[i]))
        cmd_check_index(shards^)
    elif cmd == "head":
        var pass_args = List[String]()
        for i in range(len(args)):
            pass_args.append(String(args[i]))
        try:
            cmd_head(pass_args^)
        except e:
            var err_s = String(e)
            if err_s.startswith("{"):
                eprint_json(err_s)
            else:
                fail("INTERNAL_ERROR", err_s, "", "")
            exit(2)
    elif cmd == "layer":
        var pass_args = List[String]()
        for i in range(len(args)):
            pass_args.append(String(args[i]))
        try:
            cmd_layer(pass_args^)
        except e:
            var err_s = String(e)
            if err_s.startswith("{"):
                eprint_json(err_s)
            else:
                fail_layer("INTERNAL_ERROR", err_s, "attention", -1)
            exit(2)
    else:
        fail("USAGE", String("subcommand tak dikenal: ", cmd), "", "")
