# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Mojo Micro-Runner untuk Kalibrasi Rezim 2: End-to-End Async Overlap (§2.5, Gate G-M11-2).

Menjalankan:
1. Alokasi WorkerPool dengan c threads dan DoubleBufferedPipeline dengan N_in_flight in [2, 4].
2. Pengukuran T_IO murni (streaming storage tanpa intervensi komputasi).
3. Pengukuran T_comp(c) murni (dekuantisasi paralel SIMD 4-bit ke BF16 via worker pool).
4. Kalkulasi T_seq(c) = T_IO + T_comp(c) (Formula F16 naif sekuensial).
5. Fase Warm-up (Fill): N_warmup iterasi token dibuang sesuai protokol P2 (§2.5).
6. Fase Steady-State: N_steady iterasi token diukur latensinya saat I/O dan komputasi tumpang-tindih.
   Mengekstrak T_step_overlap(c) steady-state untuk evaluasi Formula F18.
7. Emisi telemetri JSON (T_IO, T_comp, T_seq, T_overlap, BW_eff, E_overlap).
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


def _create_synthetic_chunk_file(path: String, total_bytes: Int) raises -> Int:
    """Membuat file sementara berukuran total_bytes dengan data ter-align."""
    var path_z = path + "\0"
    # O_WRONLY | O_CREAT | O_TRUNC = 577, 0o644 = 420
    var fd = external_call["openat", Int32](
        -100, path_z.unsafe_ptr(), Int32(577), Int32(420)
    )
    if fd < 0:
        raise Error("Failed to create file: " + path)

    var block_size = 65536
    var p_buf = external_call["aligned_alloc", Int](4096, block_size)
    var ptr = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p_buf)
    for i in range(block_size):
        ptr[unsafe_offset=i] = UInt8((i % 240) + 1)

    var remaining = total_bytes
    var offset = 0
    while remaining > 0:
        var chunk = remaining if remaining < block_size else block_size
        var written = external_call["pwrite", Int](
            Int32(fd), p_buf, chunk, offset
        )
        if written <= 0:
            break
        offset += written
        remaining -= written

    # Sync dirty pages sebelum dibaca
    _ = external_call["fdatasync", Int32](Int32(fd))
    _ = external_call["close", Int32](Int32(fd))
    external_call["free", NoneType](p_buf)

    # O_RDONLY | O_DIRECT = 16384
    var fd_read = external_call["openat", Int32](
        -100, path_z.unsafe_ptr(), Int32(16384), Int32(0)
    )
    if fd_read < 0:
        fd_read = external_call["openat", Int32](
            -100, path_z.unsafe_ptr(), Int32(0), Int32(0)
        )
    return Int(fd_read)


def _sort_list(mut lst: List[Float64]):
    """Mengurutkan daftar Float64 ascending untuk perhitungan median."""
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[j] < lst[i]:
                var tmp = lst[i]
                lst[i] = lst[j]
                lst[j] = tmp


def main() raises:
    var threads = parse_int_arg("--threads", 1)
    var n_in_flight = parse_int_arg("--n-in-flight", 2)
    var num_slots = parse_int_arg("--num-slots", 8)
    var num_warmup = parse_int_arg("--warmup", 3)
    var num_steady = parse_int_arg("--steady", 10)
    var file_arg = parse_str_arg("--file", "")
    var emit_json = parse_bool_arg("--json")

    # Invarian N_in_flight in [2, 4]
    if n_in_flight < 2:
        n_in_flight = 2
    elif n_in_flight > 4:
        n_in_flight = 4

    # Chunk size profil MoE: 853 * 4096 = 3,493,888 B (~3.33 MiB)
    var chunk_size = BASE_CHUNK_SIZE
    var total_layer_bytes = chunk_size * num_slots  # ~27.95 MiB

    var is_temp_file = False
    var temp_file_path = "/tmp/dismoen_m11_overlap_bench.bin"
    var fd: Int

    if file_arg != "":
        var path_z = file_arg + "\0"
        # O_RDONLY | O_DIRECT = 0x4000 = 16384
        fd = Int(
            external_call["openat", Int32](
                -100, path_z.unsafe_ptr(), Int32(16384), Int32(0)
            )
        )
        if fd < 0:
            # Fallback ke buffered O_RDONLY jika filesystem tidak mendukung O_DIRECT
            fd = Int(
                external_call["openat", Int32](
                    -100, path_z.unsafe_ptr(), Int32(0), Int32(0)
                )
            )
        if fd < 0:
            raise Error("Failed to open specified file: " + file_arg)

    else:
        fd = _create_synthetic_chunk_file(temp_file_path, total_layer_bytes)
        is_temp_file = True

    # Inisialisasi pipeline double-buffering dan pool komputasi
    var pipeline = DoubleBufferedPipeline(
        n_in_flight=n_in_flight,
        required_align=4096,
        buffer_capacity=64 * 1024 * 1024,
        num_slots_per_stage=num_slots,
        chunk_size=chunk_size,
    )

    var pool = WorkerPool(num_threads=threads)

    # Buffer komputasi dekuantisasi MoE per-token (~786K elemen ~ 8 pakar layer)
    var num_elements = 786432  # 786,432 elements (6144 groups of 128)
    var group_size = 128
    var num_groups = num_elements // group_size
    var scales_bytes = num_groups * 2
    var packed_bytes = num_elements // 2
    var out_bytes = num_elements * 2

    var p_scales = external_call["aligned_alloc", Int](4096, scales_bytes)
    var p_packed = external_call["aligned_alloc", Int](4096, packed_bytes)
    var p_out = external_call["aligned_alloc", Int](4096, out_bytes)

    # Inisialisasi data sintetis dekuantisasi
    var ptr_scales = Pointer[UInt16, MutAnyOrigin](unsafe_from_address=p_scales)
    for g in range(num_groups):
        ptr_scales[unsafe_offset=g] = UInt16(0x3C00)  # FP16 1.0

    var ptr_packed = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p_packed)
    for b in range(packed_bytes):
        ptr_packed[unsafe_offset=b] = UInt8(0x12)  # Nibbles +1 dan +2

    # -------------------------------------------------------------
    # 1. Ukur T_IO murni (5 warm-up, 15 measured runs, ambil median)
    # -------------------------------------------------------------
    for _ in range(5):
        var w_jobs = List[Int]()
        for s in range(num_slots):
            w_jobs.append(pipeline.dispatch_chunk_io(s, fd, s * chunk_size, s))
        for s in range(num_slots):
            pipeline.wait_chunk_io(s, w_jobs[s])
        pipeline.ring_b.reset_all()

    var io_runs = List[Float64]()
    for _ in range(15):
        var t0 = perf_counter_ns()
        var job_ids = List[Int]()
        for s in range(num_slots):
            job_ids.append(pipeline.dispatch_chunk_io(s, fd, s * chunk_size, s))
        for s in range(num_slots):
            pipeline.wait_chunk_io(s, job_ids[s])
        var t1 = perf_counter_ns()
        io_runs.append(Float64(t1 - t0) / 1000000.0)
        pipeline.ring_b.reset_all()

    _sort_list(io_runs)
    var t_io_ms = io_runs[7]

    # -------------------------------------------------------------
    # 2. Ukur T_comp(c) murni via WorkerPool (3 warm-up, 11 measured)
    # -------------------------------------------------------------
    for _ in range(3):
        pool.parallel_dequant_bf16(
            scales_addr=p_scales,
            packed_addr=p_packed,
            out_addr=p_out,
            num_elements=num_elements,
            group_size=group_size,
        )

    var comp_runs = List[Float64]()
    for _ in range(11):
        var t0 = perf_counter_ns()
        pool.parallel_dequant_bf16(
            scales_addr=p_scales,
            packed_addr=p_packed,
            out_addr=p_out,
            num_elements=num_elements,
            group_size=group_size,
        )
        var t1 = perf_counter_ns()
        comp_runs.append(Float64(t1 - t0) / 1000000.0)

    _sort_list(comp_runs)
    var t_comp_ms = comp_runs[5]
    var t_seq_ms = t_io_ms + t_comp_ms

    # -------------------------------------------------------------
    # 3. Overlap Priming & Warm-up (Fill Phase) — DIABAIKAN (§2.5)
    # -------------------------------------------------------------
    pipeline.ring_a.reset_all()
    pipeline.ring_b.reset_all()
    pipeline.active_compute_stage = 0
    pipeline.active_io_stage = 1

    var init_jobs = List[Int]()
    for s in range(num_slots):
        init_jobs.append(pipeline.dispatch_chunk_io(s, fd, s * chunk_size, s))
    for s in range(num_slots):
        pipeline.wait_chunk_io(s, init_jobs[s])

    pipeline.flip_stages()
    pipeline.acquire_all_compute()

    for _ in range(num_warmup):
        var jobs = List[Int]()
        for s in range(num_slots):
            jobs.append(pipeline.dispatch_chunk_io(s, fd, s * chunk_size, s))

        pool.parallel_dequant_bf16(
            scales_addr=p_scales,
            packed_addr=p_packed,
            out_addr=p_out,
            num_elements=num_elements,
            group_size=group_size,
        )

        pipeline.release_all_compute()

        for s in range(num_slots):
            pipeline.wait_chunk_io(s, jobs[s])

        pipeline.flip_stages()
        pipeline.acquire_all_compute()

    # -------------------------------------------------------------
    # 4. Protokol Rezim 2: Steady-State Tokens (§2.5) — DIUKUR
    # -------------------------------------------------------------
    var steady_overlap_latencies = List[Float64]()

    for _ in range(num_steady):
        var t_step_start = perf_counter_ns()

        var t0 = perf_counter_ns()
        var jobs = List[Int]()
        for s in range(num_slots):
            jobs.append(pipeline.dispatch_chunk_io(s, fd, s * chunk_size, s))
        var t1 = perf_counter_ns()

        pool.parallel_dequant_bf16(
            scales_addr=p_scales,
            packed_addr=p_packed,
            out_addr=p_out,
            num_elements=num_elements,
            group_size=group_size,
        )
        var t2 = perf_counter_ns()

        pipeline.release_all_compute()

        for s in range(num_slots):
            pipeline.wait_chunk_io(s, jobs[s])
        var t3 = perf_counter_ns()

        steady_overlap_latencies.append(Float64(t3 - t_step_start) / 1000000.0)

        pipeline.flip_stages()
        pipeline.acquire_all_compute()

    pipeline.release_all_compute()

    # -------------------------------------------------------------
    # 5. Kalkulasi Metrik Rezim 2 & Formula F18
    # -------------------------------------------------------------
    _sort_list(steady_overlap_latencies)
    var t_overlap_ms = steady_overlap_latencies[
        len(steady_overlap_latencies) // 2
    ]

    var min_c_io = t_io_ms if t_io_ms < t_comp_ms else t_comp_ms
    var hidden_ms = t_seq_ms - t_overlap_ms
    var e_overlap_pct = (hidden_ms / min_c_io) * 100.0
    if e_overlap_pct > 100.0:
        e_overlap_pct = 100.0
    elif e_overlap_pct < 0.0:
        e_overlap_pct = 0.0

    var bw_eff_mbs = (
        Float64(total_layer_bytes) / (t_io_ms / 1000.0)
    ) / 1000000.0

    # -------------------------------------------------------------
    # 6. Emisi Output
    # -------------------------------------------------------------
    if emit_json:
        print("{")
        print('  "threads": ' + String(threads) + ",")
        print('  "n_in_flight": ' + String(n_in_flight) + ",")
        print('  "num_slots": ' + String(num_slots) + ",")
        print('  "chunk_size": ' + String(chunk_size) + ",")
        print('  "total_bytes": ' + String(total_layer_bytes) + ",")
        print('  "num_warmup": ' + String(num_warmup) + ",")
        print('  "num_steady": ' + String(num_steady) + ",")
        print('  "t_io_ms": ' + String(t_io_ms) + ",")
        print('  "t_comp_ms": ' + String(t_comp_ms) + ",")
        print('  "t_seq_ms": ' + String(t_seq_ms) + ",")
        print('  "t_overlap_ms": ' + String(t_overlap_ms) + ",")
        print('  "bw_eff_mbs": ' + String(bw_eff_mbs) + ",")
        print('  "e_overlap_pct": ' + String(e_overlap_pct) + ",")
        print(
            '  "gate_g_m11_2_pass": '
            + ("true" if e_overlap_pct >= 80.0 else "false")
        )
        print("}")
    else:
        print(
            "=== Rezim 2 Steady-State Overlap (c="
            + String(threads)
            + ", N_in_flight="
            + String(n_in_flight)
            + ") ==="
        )
        print("T_IO (murni):       " + String(t_io_ms) + " ms")
        print("T_comp (murni):     " + String(t_comp_ms) + " ms")
        print("T_seq (naif):       " + String(t_seq_ms) + " ms")
        print("T_overlap (riil):   " + String(t_overlap_ms) + " ms")
        print("BW_eff:             " + String(bw_eff_mbs) + " MB/s")
        print("E_overlap (F18):    " + String(e_overlap_pct) + " %")
        print(
            "Gate G-M11-2:       "
            + ("PASS" if e_overlap_pct >= 80.0 else "FAIL")
        )

    # Cleanup resources
    pool.shutdown()
    pipeline.shutdown()
    external_call["free", NoneType](p_scales)
    external_call["free", NoneType](p_packed)
    external_call["free", NoneType](p_out)

    if fd >= 0:
        _ = external_call["close", Int32](Int32(fd))
    if is_temp_file:
        var path_z = temp_file_path + "\0"
        _ = external_call["unlink", Int32](path_z.unsafe_ptr())
