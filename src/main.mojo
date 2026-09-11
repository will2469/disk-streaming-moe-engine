# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Kimo CLI — check-index (M0) dan head path (M1)."""

from model import (
    HeadWeights,
    LoadMemoryTelemetry,
    ModelConfig,
    forward_head,
    load_tensor_f32_chunked,
    validate_logits,
)
from safetensors import (
    STError,
    Scanner,
    STHeader,
    TensorMeta,
    json_escape,
    read_header,
)
from std.collections import Dict, List
from std.ffi import external_call
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


def read_small_file(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var n = Int(f.seek(0, 2))
    _ = f.seek(0, 0)
    if n > 100000000:
        f.close()
        raise Error(
            String(
                (
                    '{"error_type":"INVALID_HEADER","detail":"index too'
                    ' large","shard":"'
                ),
                path,
                '","tensor_name":""}',
            )
        )
    var out = f.read_bytes(n)
    f.close()
    if len(out) < n:
        raise Error(
            String(
                (
                    '{"error_type":"INVALID_HEADER","detail":"index'
                    ' truncated","shard":"'
                ),
                path,
                '","tensor_name":""}',
            )
        )
    return out^


def parse_index(path: String) raises -> List[String]:
    # return [names..., files...] sejajar: names[i] <-> files[i]
    var raw = read_small_file(path)
    var sc = Scanner(raw^, path)
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
                if len(names) > 100000:
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


def fail(code: String, detail: String, shard: String, tensor: String) raises:
    eprint_json(err_json(code, detail, shard, tensor))
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


def main() raises:
    var args = argv()
    if len(args) < 2:
        fail("USAGE", "pakai: kimo (check-index|head) ...", "", "")
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
    else:
        fail("USAGE", String("subcommand tak dikenal: ", cmd), "", "")
