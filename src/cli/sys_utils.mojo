# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""POSIX C FFI dan utilitas sistem untuk Kimo CLI."""

from cli.errors import basename, dirname, eprint_json, fail, fail_layer
from format.file_io import c_realpath, path_is_within
from format.types import json_escape
from std.collections import Dict, List
from std.ffi import external_call
from std.sys.terminate import exit
from std.time import perf_counter_ns


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


# Flags open(2) Linux (<fcntl.h>, bits/fcntl-linux.h):
# O_WRONLY=1, O_CREAT=64, O_EXCL=128, O_NOFOLLOW=131072.
# Kombinasi O_CREAT|O_EXCL|O_NOFOLLOW untuk tmp file:
# - menolak symlink pre-planted (gagal ELOOP, tidak follow),
# - menolak timpa file existing (gagal EEXIST, tidak truncate).
# Mode 0600=384: tmp hanya r/w owner sampai rename atomik.
def c_getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def c_open_tmp_excl(path: String) -> Int:
    # openat(AT_FDCWD, ...) == open() tapi menghindari bentrok signature
    # external_call["open", ...] milik stdlib. AT_FDCWD=-100.
    var sb = path.as_bytes()
    var z = List[UInt8]()
    for i in range(len(sb)):
        z.append(sb[i])
    z.append(0)
    var fd = external_call["openat", Int32](-100, z.unsafe_ptr(), 131265, 384)
    return Int(fd)


def c_write_f32_fd_all(fd: Int, data: List[Float32]) -> Bool:
    # Single write(2) fail-closed: ke regular file yang baru dibuat,
    # kernel transfer penuh atau gagal (ENOSPC/EINTR/dsb) — caller unlink
    # tmp dan lapor OUTPUT_WRITE_FAILED. Tanpa aritmetika pointer.
    var n_bytes = len(data) * 4
    if n_bytes == 0:
        return True
    var p_u8 = data.unsafe_ptr().unsafe_bitcast[UInt8]()
    var ret = external_call["write", Int](fd, p_u8, n_bytes)
    return Int(ret) == n_bytes


def c_fsync_fd(fd: Int) -> Int:
    return Int(external_call["fsync", Int32](Int32(fd)))


def c_close_fd(fd: Int) -> Int:
    return Int(external_call["close", Int32](Int32(fd)))


def make_unique_tmp_path(target_path: String, attempt: Int) -> String:
    # target.tmp.<pid>.<ns>.<attempt>: unik per proses (pid), per waktu
    # (ns), dan per retry (attempt). Attacker tak bisa prediksi nama untuk
    # pre-plant symlink; dua proses konkuren ke target sama tak saling timpa.
    var pid = c_getpid()
    var ns = perf_counter_ns()
    return String(target_path, ".tmp.", pid, ".", ns, ".", attempt)


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


def _is_within_workdir(parent_canon: String, workdir_canon: String) -> Bool:
    # Delegasi ke SATU-SATUNYA implementasi containment (format.path_is_within).
    return path_is_within(workdir_canon, parent_canon)


def resolve_target_output(
    output_file: String, workdir: String, layer_idx: Int = -1
) raises -> Tuple[String, String]:
    # SECURITY: pola check-then-use (realpath → banding string → open).
    # O_EXCL|O_NOFOLLOW + nama tmp unik menutup symlink-plant dan
    # tabrakan konkuren; residual = parent dir diganti (swap/symlink) di
    # jendela mikrodetik antara validasi dan create — diterima untuk threat
    # model CLI (filesystem lokal non-hostile). Filesystem hostile/shared
    # butuh operasi FD penuh (openat2 RESOLVE_BENEATH + renameat).
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
    if parent_canon == "" or not _is_within_workdir(
        parent_canon, workdir_canon
    ):
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
        # Invariant escape global: tiap string dinamis (nama tensor/shard
        # dari file eksternal) lewat json_escape sebelum masuk JSON.
        var mt_json = String("")
        for mi in range(len(missing_tensors)):
            if mi > 0:
                mt_json += ","
            mt_json += String('"', json_escape(missing_tensors[mi]), '"')
        var es_json = String("")
        for ei in range(len(expected_shards)):
            if ei > 0:
                es_json += ","
            es_json += String('"', json_escape(expected_shards[ei]), '"')
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
    # Bandingkan canonical path seperti cmd_head: lexical "a/../a" vs "a"
    # sama; symlink tampak-sama dibedakan. "" == "" lolos ke FILE_NOT_FOUND.
    var r0c = c_realpath(r0_dir)
    for k in range(len(supplied_shards)):
        var rk = dirname(supplied_shards[k])
        if c_realpath(rk if rk != "" else ".") != r0c:
            fail_layer(
                "FILE_NOT_FOUND",
                "all supplied shards must reside in the same directory",
                "attention",
                layer_val,
            )
    return (r0c if r0c != "" else r0_dir, False)
