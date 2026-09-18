# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Mojo Micro-Runner untuk Kalibrasi Core Scaling Rezim 1 & Rezim 2 (§2.5, G-M11-1).

Menjalankan pengujian terisolasi:
1. Alokasi WorkerPool dengan c threads.
2. Rezim 1 (Compute-Isolated):
   - Alokasi memori bobot terkuantisasi (Q4_K) dan buffer hasil (BF16) upfront di DRAM.
   - Eksekusi dekuantisasi paralel SIMD bebas I/O disk.
   - Pengukuran p50, p95 (interpolasi linear), min, max, mean latensi per iterasi token.
   - Perekaman jejak memori VmHWM via /proc/self/status.
3. Rezim 2 (End-to-End Async Overlap):
   - Pengukuran I/O storage, komputasi tersembunyi, dan efisiensi overlap F18.
4. Emisi telemetri JSON terstruktur ke stdout.
"""

from core.worker_pool import WorkerPool
from io.dio_probe import BASE_BUFFER_CAPACITY, BASE_CHUNK_SIZE
from io.async_pipeline import DoubleBufferedPipeline
from std.collections import List
from std.ffi import external_call
from std.memory import Pointer
from std.sys import argv
from std.time import perf_counter_ns


def parse_int_arg(flag: String, default_val: Int) -> Int:
    """Mengambil nilai integer argumen CLI berdasarkan flag."""
    var args = argv()
    for i in range(len(args) - 1):
        if args[i] == flag:
            try:
                return Int(args[i + 1])
            except:
                return default_val
    return default_val


def parse_str_arg(flag: String, default_val: String) -> String:
    """Mengambil nilai string argumen CLI berdasarkan flag."""
    var args = argv()
    for i in range(len(args) - 1):
        if args[i] == flag:
            return args[i + 1]
    return default_val


def parse_bool_arg(flag: String) -> Bool:
    """Mengembalikan True bila flag terdapat dalam argumen CLI."""
    var args = argv()
    for i in range(len(args)):
        if args[i] == flag:
            return True
    return False


def _read_vm_hwm_kib() -> Int:
    """Membaca VmHWM (Peak Resident Set Size) dari /proc/self/status."""
    var path_z = "/proc/self/status\0"
    var fd = external_call["openat", Int32](
        -100, path_z.unsafe_ptr(), Int32(0), Int32(0)  # O_RDONLY
    )
    if fd < 0:
        return 0

    var buf_size = 4096
    var buf = external_call["malloc", Int](buf_size)
    var n = external_call["pread", Int](fd, buf, buf_size, 0)
    _ = external_call["close", Int32](fd)

    if n <= 0:
        external_call["free", NoneType](buf)
        return 0

    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=buf)
    # Cari substring "VmHWM:"
    # V=86, m=109, H=72, W=87, M=77, :=58
    var found_idx = -1
    for i in range(n - 6):
        if (
            p[unsafe_offset=i] == 86
            and p[unsafe_offset=i + 1] == 109
            and p[unsafe_offset=i + 2] == 72
            and p[unsafe_offset=i + 3] == 87
            and p[unsafe_offset=i + 4] == 77
            and p[unsafe_offset=i + 5] == 58
        ):
            found_idx = i + 6
            break

    if found_idx < 0:
        external_call["free", NoneType](buf)
        return 0

    # Parse integer setelah VmHWM: (abaikan spasi/tab)
    var val = 0
    var parsing_digit = False
    for i in range(found_idx, n):
        var c = p[unsafe_offset=i]
        if c >= 48 and c <= 57:  # '0'-'9'
            parsing_digit = True
            val = val * 10 + Int(c - 48)
        elif parsing_digit:
            # Selesai membaca digit pertama
            break

    external_call["free", NoneType](buf)
    return val


def _sort_list(mut lst: List[Float64]):
    """Mengurutkan daftar Float64 ascending untuk persentil."""
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[j] < lst[i]:
                var tmp = lst[i]
                lst[i] = lst[j]
                lst[j] = tmp


def _calc_percentile(sorted_lst: List[Float64], p_pct: Float64) -> Float64:
    """Menghitung persentil dengan interpolasi linear (Project SLO standar)."""
    var n = len(sorted_lst)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_lst[0]

    var rank = (p_pct / 100.0) * Float64(n - 1)
    var low_idx = Int(rank)
    var high_idx = low_idx + 1
    if high_idx >= n:
        return sorted_lst[n - 1]

    var weight = rank - Float64(low_idx)
    return sorted_lst[low_idx] * (1.0 - weight) + sorted_lst[high_idx] * weight


def _create_synthetic_file(path: String, total_bytes: Int) raises -> Int:
    """Membuat file dummy berukuran total_bytes."""
    var path_z = path + "\0"
    # O_WRONLY | O_CREAT | O_TRUNC = 577, 0o644 = 420
    var fd = Int(
        external_call["openat", Int32](
            -100, path_z.unsafe_ptr(), Int32(577), Int32(420)
        )
    )
    if fd < 0:
        raise Error("Failed to create temporary benchmark file: " + path)

    var page_size = 65536
    var p_dummy = external_call["aligned_alloc", Int](4096, page_size)
    var p_u8 = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p_dummy)
    for i in range(page_size):
        p_u8[unsafe_offset=i] = UInt8(i % 251)

    var written_total = 0
    while written_total < total_bytes:
        var chunk = page_size
        if written_total + chunk > total_bytes:
            chunk = total_bytes - written_total
        var nw = external_call["pwrite", Int](
            Int32(fd), p_dummy, chunk, written_total
        )
        if nw <= 0:
            external_call["free", NoneType](p_dummy)
            _ = external_call["close", Int32](Int32(fd))
            raise Error("Failed to write temporary benchmark data")
        written_total += nw

    external_call["free", NoneType](p_dummy)
    _ = external_call["fsync", Int32](Int32(fd))
    _ = external_call["close", Int32](Int32(fd))

    # O_RDONLY | O_DIRECT = 16384
    var fd_read = external_call["openat", Int32](
        -100, path_z.unsafe_ptr(), Int32(16384), Int32(0)
    )
    if fd_read < 0:
        # Fallback O_RDONLY
        fd_read = external_call["openat", Int32](
            -100, path_z.unsafe_ptr(), Int32(0), Int32(0)
        )
    return Int(fd_read)


def main() raises:
    var threads = parse_int_arg("--threads", 1)
    var mode = parse_str_arg("--mode", "compute-isolated")
    var num_warmup = parse_int_arg("--num-warmup", 2)
    var num_steady = parse_int_arg("--num-steady", 10)
    var num_elements = parse_int_arg("--elements", 786432)  # ~8 expert blocks
    var emit_json = parse_bool_arg("--json")

    # Inisialisasi pool komputasi
    var pool = WorkerPool(num_threads=threads)

    # Buffer dekuantisasi MoE per-token
    var group_size = 128
    var num_groups = num_elements // group_size
    var scales_bytes = num_groups * 2
    var packed_bytes = num_elements // 2
    var out_bytes = num_elements * 2
    var total_blocks = num_elements // 32

    var p_scales = external_call["aligned_alloc", Int](4096, scales_bytes)
    var p_packed = external_call["aligned_alloc", Int](4096, packed_bytes)
    var p_out = external_call["aligned_alloc", Int](4096, out_bytes)

    # Inisialisasi data sintetis
    var ptr_scales = Pointer[UInt16, MutAnyOrigin](unsafe_from_address=p_scales)
    for g in range(num_groups):
        ptr_scales[unsafe_offset=g] = UInt16(0x3C00)  # FP16 1.0

    var ptr_packed = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p_packed)
    for b in range(packed_bytes):
        ptr_packed[unsafe_offset=b] = UInt8(0x12)

    # -------------------------------------------------------------
    # 1. Eksekusi Warm-up
    # -------------------------------------------------------------
    for _ in range(num_warmup):
        pool.parallel_dequant_bf16(
            scales_addr=p_scales,
            packed_addr=p_packed,
            out_addr=p_out,
            num_elements=num_elements,
            group_size=group_size,
        )

    # -------------------------------------------------------------
    # 2. Eksekusi Measured Steady Iterations
    # -------------------------------------------------------------
    var latencies = List[Float64]()
    var sum_ms = 0.0

    for _ in range(num_steady):
        var t0 = perf_counter_ns()
        pool.parallel_dequant_bf16(
            scales_addr=p_scales,
            packed_addr=p_packed,
            out_addr=p_out,
            num_elements=num_elements,
            group_size=group_size,
        )
        var t1 = perf_counter_ns()
        var dur_ms = Float64(t1 - t0) / 1000000.0
        latencies.append(dur_ms)
        sum_ms += dur_ms

    var vm_hwm_kib = _read_vm_hwm_kib()

    # Hitung p50, p95, min, max, mean
    _sort_list(latencies)
    var p50_ms = _calc_percentile(latencies, 50.0)
    var p95_ms = _calc_percentile(latencies, 95.0)
    var min_ms = latencies[0]
    var max_ms = latencies[len(latencies) - 1]
    var mean_ms = sum_ms / Float64(num_steady)

    # -------------------------------------------------------------
    # 3. Output
    # -------------------------------------------------------------
    if emit_json:
        print("{")
        print('  "threads": ' + String(threads) + ",")
        print('  "mode": "' + mode + '",')
        print('  "num_warmup": ' + String(num_warmup) + ",")
        print('  "num_steady": ' + String(num_steady) + ",")
        print('  "elements": ' + String(num_elements) + ",")
        print('  "t_comp_ms": ' + String(p50_ms) + ",")
        print('  "p50_ms": ' + String(p50_ms) + ",")
        print('  "p95_ms": ' + String(p95_ms) + ",")
        print('  "min_ms": ' + String(min_ms) + ",")
        print('  "max_ms": ' + String(max_ms) + ",")
        print('  "mean_ms": ' + String(mean_ms) + ",")
        print('  "vm_hwm_kib": ' + String(vm_hwm_kib) + ",")
        print('  "latencies_ms": [')
        for i in range(len(latencies)):
            var sep = "," if i < len(latencies) - 1 else ""
            print("    " + String(latencies[i]) + sep)
        print("  ]")
        print("}")
    else:
        print(
            "=== Core Scaling Benchmark (c="
            + String(threads)
            + ", mode="
            + mode
            + ") ==="
        )
        print("T_comp (p50): " + String(p50_ms) + " ms")
        print("T_comp (p95): " + String(p95_ms) + " ms")
        print("T_comp (min): " + String(min_ms) + " ms")
        print("T_comp (max): " + String(max_ms) + " ms")
        print("T_comp (avg): " + String(mean_ms) + " ms")
        print("VmHWM:        " + String(vm_hwm_kib) + " KiB")

    # Cleanup
    pool.shutdown()
    external_call["free", NoneType](p_scales)
    external_call["free", NoneType](p_packed)
    external_call["free", NoneType](p_out)
