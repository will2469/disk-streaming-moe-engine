# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Utilitas telemetri sistem, filesystem discovery, dan sensor termal NVMe SSD (M7)."""

from format.file_io import c_realpath
from std.collections import List
from std.ffi import external_call


def get_fs_and_mounts(target_path: String) -> Tuple[String, String]:
    """Menemukan tipe filesystem dan opsi mount dari /proc/mounts untuk path target.
    """
    var target = c_realpath(target_path)
    if target.byte_length() == 0:
        target = target_path

    var best_mount = String("")
    var best_fstype = String("ext4")
    var best_opts = String("rw,relatime")

    try:
        var f = open("/proc/mounts", "r")
        var content = f.read()
        f.close()

        var raw = content.as_bytes()
        var n = len(raw)
        var i = 0
        while i < n:
            var line_start = i
            while i < n and raw[i] != 10:  # '\n'
                i += 1
            var line_end = i
            if i < n and raw[i] == 10:
                i += 1

            # Parse kolom: <dev> <mount_point> <fstype> <options> ...
            var col_idx = 0
            var p = line_start
            var mnt = String("")
            var fstype = String("")
            var opts = String("")

            while p < line_end:
                while p < line_end and (raw[p] == 32 or raw[p] == 9):
                    p += 1
                if p >= line_end:
                    break
                var token_start = p
                while p < line_end and raw[p] != 32 and raw[p] != 9:
                    p += 1
                var tok = String(content[byte=token_start:p])
                if col_idx == 1:
                    mnt = tok
                elif col_idx == 2:
                    fstype = tok
                elif col_idx == 3:
                    opts = tok
                col_idx += 1

            if mnt.byte_length() > 0:
                if target.startswith(mnt):
                    if mnt.byte_length() > best_mount.byte_length():
                        best_mount = mnt
                        best_fstype = fstype
                        best_opts = opts
    except:
        pass

    return (best_fstype, best_opts)


def get_fs_block_size(path: String) -> Int:
    """Mengambil ukuran blok sistem berkas via statvfs(2). Default 4096."""
    var path_b = path.as_bytes()
    var path_z = List[UInt8]()
    for i in range(len(path_b)):
        path_z.append(path_b[i])
    path_z.append(0)

    # Alokasi buffer struct statvfs (~112 byte pada Linux x86_64)
    var buf = List[UInt8]()
    for _ in range(128):
        buf.append(0)

    var res = external_call["statvfs", Int32](
        path_z.unsafe_ptr(), buf.unsafe_ptr()
    )
    if res == 0:
        var p_ul = buf.unsafe_ptr().unsafe_bitcast[Int]()
        var bsize = p_ul[unsafe_offset=0]
        if bsize > 0:
            return bsize
    return 4096


def get_ssd_temperature() -> Float64:
    """Membaca temperatur SSD NVMe aktif via hwmon / sysfs dalam derajat Celsius.

    Mengembalikan suhu (misal 39.85), atau -1.0 jika tidak tersedia di lingkungan.
    """
    # 1. Coba baca dari /sys/class/hwmon
    for i in range(16):
        var name_path = String("/sys/class/hwmon/hwmon", i, "/name")
        try:
            var f = open(name_path, "r")
            var name = f.read()
            f.close()
            if name.find("nvme") >= 0:
                var temp_path = String(
                    "/sys/class/hwmon/hwmon", i, "/temp1_input"
                )
                var tf = open(temp_path, "r")
                var temp_s = tf.read()
                tf.close()
                var raw_temp = Int(temp_s.strip())
                return Float64(raw_temp) / 1000.0
        except:
            pass

    # 2. Fallback: coba periksa /sys/class/thermal
    for j in range(8):
        var tz_type = String("/sys/class/thermal/thermal_zone", j, "/type")
        try:
            var ft = open(tz_type, "r")
            var t_type = ft.read()
            ft.close()
            if (
                t_type.find("nvme") >= 0
                or t_type.find("acpitz") >= 0
                or t_type.find("x86_pkg") >= 0
            ):
                var t_val_path = String(
                    "/sys/class/thermal/thermal_zone", j, "/temp"
                )
                var fv = open(t_val_path, "r")
                var v_s = fv.read()
                fv.close()
                var raw_v = Int(v_s.strip())
                return Float64(raw_v) / 1000.0
        except:
            pass

    return -1.0


def posix_fadvise_sequential(fd: Int) -> Int:
    """Mengaktifkan kebijakan readahead berurutan via posix_fadvise(2) (POSIX_FADV_SEQUENTIAL = 2).
    """
    if fd < 0:
        return -1
    return Int(
        external_call["posix_fadvise", Int32](
            Int32(fd), Int(0), Int(0), Int32(2)
        )
    )
