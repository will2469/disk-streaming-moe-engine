# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""POSIX C FFI dan utilitas sistem untuk Dismoen CLI."""

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


def c_link(oldpath: String, newpath: String) -> Int:
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
        external_call["link", Int32](old_z.unsafe_ptr(), new_z.unsafe_ptr())
    )


def c_mkdir(path: String, mode: Int = 493) -> Int:
    # mkdirat(AT_FDCWD, ...) == mkdir() tapi menghindari bentrok signature
    # external_call["mkdir", ...] milik stdlib. AT_FDCWD=-100.
    var p = path.as_bytes()
    var p_z = List[UInt8]()
    for i in range(len(p)):
        p_z.append(p[i])
    p_z.append(0)
    return Int(
        external_call["mkdirat", Int32](
            Int32(-100), p_z.unsafe_ptr(), Int32(mode)
        )
    )


def c_rmdir(path: String) -> Int:
    var p = path.as_bytes()
    var p_z = List[UInt8]()
    for i in range(len(p)):
        p_z.append(p[i])
    p_z.append(0)
    return Int(external_call["rmdir", Int32](p_z.unsafe_ptr()))


def c_access_w(path: String) -> Bool:
    # W_OK = 2 in POSIX unistd.h
    var p = path.as_bytes()
    var p_z = List[UInt8]()
    for i in range(len(p)):
        p_z.append(p[i])
    p_z.append(0)
    var ret = external_call["access", Int32](p_z.unsafe_ptr(), Int32(2))
    return ret == 0


def c_access_r(path: String) -> Bool:
    # R_OK = 4 in POSIX unistd.h
    var p = path.as_bytes()
    var p_z = List[UInt8]()
    for i in range(len(p)):
        p_z.append(p[i])
    p_z.append(0)
    var ret = external_call["access", Int32](p_z.unsafe_ptr(), Int32(4))
    return ret == 0


def get_file_size(path: String) -> Int:
    try:
        var f = open(path, "r")
        var sz = Int(f.seek(0, 2))  # SEEK_END = 2
        f.close()
        return sz
    except:
        return -1


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


def c_write_f32_fd_n(fd: Int, data: List[Float32], count: Int) -> Bool:
    var n_bytes = count * 4
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


def _get_cgroup_subpath() -> String:
    try:
        var f = open("/proc/self/cgroup", "r")
        var content = f.read()
        f.close()
        var lines = content.split("\n")
        for i in range(len(lines)):
            var line = lines[i]
            if line.startswith("0::"):
                var sub = line[byte=3:]
                return String(sub)
    except:
        pass
    return ""


def get_cgroup_peak_bytes() -> Int:
    var sub = _get_cgroup_subpath()
    var paths = List[String]()
    if sub.byte_length() > 0:
        paths.append(String("/sys/fs/cgroup", sub, "/memory.peak"))
    paths.append("/sys/fs/cgroup/memory.peak")
    for i in range(len(paths)):
        try:
            var f = open(paths[i], "r")
            var content = f.read()
            f.close()
            var s = String(content.strip())
            if s.byte_length() > 0:
                return Int(s)
        except:
            pass
    return 0


def get_cgroup_oom_kills() -> Int:
    var sub = _get_cgroup_subpath()
    var paths = List[String]()
    if sub.byte_length() > 0:
        paths.append(String("/sys/fs/cgroup", sub, "/memory.events"))
    paths.append("/sys/fs/cgroup/memory.events")
    for i in range(len(paths)):
        try:
            var f = open(paths[i], "r")
            var content = f.read()
            f.close()
            var lines = content.split("\n")
            for j in range(len(lines)):
                var line = lines[j]
                if line.startswith("oom_kill "):
                    var parts = line.split()
                    if len(parts) >= 2:
                        return Int(parts[1])
        except:
            pass
    return 0


def get_current_yyyymmdd() -> String:
    var t_buf = List[Int]()
    t_buf.append(0)
    _ = external_call["time", Int](t_buf.unsafe_ptr())
    var tm_buf = List[Int32]()
    for _ in range(14):
        tm_buf.append(0)
    _ = external_call["localtime_r", Int](
        t_buf.unsafe_ptr(), tm_buf.unsafe_ptr()
    )
    var year = Int(tm_buf[5]) + 1900
    var month = Int(tm_buf[4]) + 1
    var day = Int(tm_buf[3])

    var y_str = String(year)
    var m_str = String(month)
    if month < 10:
        m_str = String("0", month)
    var d_str = String(day)
    if day < 10:
        d_str = String("0", day)
    return String(y_str, m_str, d_str)


def allocate_run_id_and_dir(
    workdir_canon: String, custom_run_id: String = "", prefix: String = "M4"
) raises -> Tuple[String, String]:
    var runs_parent = String(workdir_canon, "/runs")
    _ = c_mkdir(runs_parent)

    if custom_run_id.byte_length() > 0:
        var run_dir = String(runs_parent, "/", custom_run_id)
        _ = c_mkdir(run_dir)
        return (custom_run_id, run_dir)

    var ymd = get_current_yyyymmdd()
    for n in range(1, 1000):
        var n_str = String(n)
        if n < 10:
            n_str = String("00", n)
        elif n < 100:
            n_str = String("0", n)
        var run_id = String(prefix, "-", ymd, "-", n_str)
        var run_dir = String(runs_parent, "/", run_id)
        var ret = c_mkdir(run_dir)
        if ret == 0:
            return (run_id, run_dir)

    raise Error("Failed to allocate unique run_id in " + runs_parent)


def cleanup_run_resources(run_dir: String, tmp_files: List[String]):
    for i in range(len(tmp_files)):
        var tf = tmp_files[i]
        if tf.byte_length() > 0:
            _ = c_unlink(tf)
    if run_dir.byte_length() > 0:
        _ = c_rmdir(run_dir)


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
