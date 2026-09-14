# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""POSIX C FFI dan utilitas sistem untuk Kimo CLI."""

from cli.errors import basename, dirname, eprint_json, fail, fail_layer
from std.collections import Dict, List
from std.ffi import external_call
from std.sys.terminate import exit


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


def resolve_target_output(
    output_file: String, workdir: String, layer_idx: Int = -1
) raises -> Tuple[String, String]:
    var workdir_canon = c_realpath(workdir if workdir != "" else ".")
    if workdir_canon == "":
        if layer_idx >= 0:
            fail_layer(
                "OUTPUT_WRITE_FAILED",
                "workdir does not exist: "
                + (workdir if workdir != "" else "."),
                "output",
                layer_idx,
            )
        else:
            fail(
                "OUTPUT_WRITE_FAILED",
                "workdir does not exist: "
                + (workdir if workdir != "" else "."),
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
        if layer_idx >= 0:
            fail_layer(
                "OUTPUT_WRITE_FAILED",
                "output path escapes workdir: " + target_output,
                "output",
                layer_idx,
            )
        else:
            fail(
                "OUTPUT_WRITE_FAILED",
                "output path escapes workdir: " + target_output,
                "",
                "",
            )

    return (target_output, workdir_canon)


def validate_shards_coverage(
    supplied_shards: List[String],
    req_list: List[String],
    weight_map: Dict[String, String],
    stage: String,
    layer_idx: Int = -1,
) raises:
    for si in range(len(supplied_shards)):
        var sp = supplied_shards[si]
        if c_realpath(sp) == "":
            if layer_idx >= 0:
                fail_layer(
                    "FILE_NOT_FOUND",
                    "supplied shard not found on disk: " + sp,
                    stage,
                    layer_idx,
                )
            else:
                fail(
                    "FILE_NOT_FOUND",
                    "supplied shard not found on disk: " + sp,
                    sp,
                    "",
                )

    var supplied_bases = List[String]()
    for si in range(len(supplied_shards)):
        supplied_bases.append(basename(supplied_shards[si]))

    var missing_tensors = List[String]()
    var expected_shards = List[String]()
    for ri in range(len(req_list)):
        var rn = req_list[ri]
        if rn in weight_map:
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
        var err: String
        if layer_idx >= 0:
            err = String(
                (
                    '{"error_type":"WEIGHT_LOAD_FAILED","detail":"Required'
                    ' tensors not covered by supplied shards","stage":"'
                ),
                stage,
                '","layer":',
                String(layer_idx),
                ',"missing_tensors":[',
                mt_json,
                '],"expected_shards":[',
                es_json,
                "]}",
            )
        else:
            err = String(
                (
                    '{"error_type":"WEIGHT_LOAD_FAILED","detail":"Required'
                    ' tensors not covered by supplied shards","stage":"'
                ),
                stage,
                '","missing_tensors":[',
                mt_json,
                '],"expected_shards":[',
                es_json,
                "]}",
            )
        eprint_json(err)
        exit(2)


def resolve_layer_model_root(
    model_dir: String, supplied_shards: List[String], layer_val: Int
) raises -> Tuple[String, Bool]:
    if model_dir != "":
        if c_realpath(model_dir) == "":
            fail_layer(
                "FILE_NOT_FOUND",
                "model directory not found: " + model_dir,
                "attention",
                layer_val,
            )
        return (model_dir, True)
    if len(supplied_shards) < 1:
        fail_layer(
            "WEIGHT_LOAD_FAILED",
            "must provide --model-dir <dir> or at least one shard safetensors",
            "attention",
            layer_val,
        )
    var r0 = dirname(supplied_shards[0])
    var r0_dir = r0 if r0 != "" else "."
    for k in range(len(supplied_shards)):
        var rk = dirname(supplied_shards[k])
        if (rk if rk != "" else ".") != r0_dir:
            fail_layer(
                "FILE_NOT_FOUND",
                "all supplied shards must reside in the same directory",
                "attention",
                layer_val,
            )
    return (r0_dir, False)
