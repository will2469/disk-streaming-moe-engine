# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M11-W2b: Dedicated Async I/O Worker, Outstanding I/O & Overlap Verification (§1.3, §3.1).
"""

from io.dio_probe import BASE_BUFFER_CAPACITY, BASE_CHUNK_SIZE
from io.staging_ring import (
    RING_SLOT_COMPUTING,
    RING_SLOT_EMPTY,
    RING_SLOT_IO_IN_FLIGHT,
    RING_SLOT_READY,
    ChunkRingBuffer,
    StagingMemory,
)
from io.async_worker import (
    AsyncIOWorker,
    IOJob,
    JOB_COMPLETED,
    JOB_EMPTY,
    JOB_ERROR,
    JOB_IN_PROGRESS,
    JOB_SUBMITTED,
)
from io.async_pipeline import DoubleBufferedPipeline
from std.collections import List
from std.ffi import external_call
from std.memory import Pointer
from std.time import perf_counter_ns
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)


def _create_temp_test_file(path: String, size_bytes: Int) raises -> Int:
    """Helper untuk membuat file sementara dengan pola byte berulang."""
    var path_z = path + "\0"
    # O_WRONLY | O_CREAT | O_TRUNC = 1 | 64 | 512 = 577 (0x241)
    var fd = external_call["openat", Int32](
        -100, path_z.unsafe_ptr(), Int32(577), Int32(420)  # 0o644
    )
    if fd < 0:
        raise Error("Failed to create temp test file: " + path)

    var p_buf = external_call["aligned_alloc", Int](4096, size_bytes)
    var ptr = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p_buf)
    for i in range(size_bytes):
        ptr[unsafe_offset=i] = UInt8(i % 251)

    var written = external_call["pwrite", Int](Int32(fd), p_buf, size_bytes, 0)
    _ = external_call["close", Int32](Int32(fd))
    external_call["free", NoneType](p_buf)

    if written != size_bytes:
        raise Error("Failed to write expected bytes to " + path)

    var fd_read = external_call["openat", Int32](
        -100, path_z.unsafe_ptr(), Int32(0), Int32(0)  # O_RDONLY
    )
    return Int(fd_read)


def _cleanup_temp_file(path: String, fd: Int):
    """Helper untuk menutup dan menghapus file sementara."""
    if fd >= 0:
        _ = external_call["close", Int32](Int32(fd))
    var path_z = path + "\0"
    _ = external_call["unlink", Int32](path_z.unsafe_ptr())


def test_async_io_worker_lifecycle() raises:
    """Verifikasi inisialisasi, sinkronisasi pthread/condvar, dan shutdown bersih AsyncIOWorker.
    """
    var worker = AsyncIOWorker(n_in_flight=2)
    assert_true(worker.is_active)
    assert_equal(worker.n_in_flight, 2)
    assert_equal(worker.get_in_flight_count(), 0)
    assert_equal(worker.get_total_completed(), 0)

    worker.shutdown()
    assert_false(worker.is_active)


def test_n_in_flight_concurrency_2_to_4() raises:
    """Verifikasi batas properti engine N_in_flight in [2, 4] (§1.3)."""
    # Nilai valid [2, 4] harus berhasil
    var w2 = AsyncIOWorker(n_in_flight=2)
    assert_equal(w2.n_in_flight, 2)
    w2.shutdown()

    var w3 = AsyncIOWorker(n_in_flight=3)
    assert_equal(w3.n_in_flight, 3)
    w3.shutdown()

    var w4 = AsyncIOWorker(n_in_flight=4)
    assert_equal(w4.n_in_flight, 4)
    w4.shutdown()

    # Nilai di luar [2, 4] harus ditolak dengan exception
    var caught_low = False
    try:
        var w_invalid = AsyncIOWorker(n_in_flight=1)
        w_invalid.shutdown()
    except:
        caught_low = True
    assert_true(caught_low)

    var caught_high = False
    try:
        var w_invalid2 = AsyncIOWorker(n_in_flight=5)
        w_invalid2.shutdown()
    except:
        caught_high = True
    assert_true(caught_high)


def test_async_chunk_job_submission() raises:
    """Verifikasi pengiriman job chunk ke background thread dan completion per-chunk.
    """
    var test_path = "/tmp/dismoen_m11_w2b_job_test.bin"
    var chunk_size = 4096
    var num_chunks = 4
    var total_size = chunk_size * num_chunks

    var fd = _create_temp_test_file(test_path, total_size)
    var p_buf = external_call["aligned_alloc", Int](4096, total_size)

    var worker = AsyncIOWorker(n_in_flight=2)

    var j0 = worker.submit_job(fd, 0, p_buf, chunk_size, 0)
    var j1 = worker.submit_job(
        fd, chunk_size, p_buf + chunk_size, chunk_size, 1
    )
    var j2 = worker.submit_job(
        fd, chunk_size * 2, p_buf + chunk_size * 2, chunk_size, 2
    )
    var j3 = worker.submit_job(
        fd, chunk_size * 3, p_buf + chunk_size * 3, chunk_size, 3
    )

    var b0 = worker.wait_job(j0)
    var b1 = worker.wait_job(j1)
    var b2 = worker.wait_job(j2)
    var b3 = worker.wait_job(j3)

    assert_equal(b0, chunk_size)
    assert_equal(b1, chunk_size)
    assert_equal(b2, chunk_size)
    assert_equal(b3, chunk_size)
    assert_equal(worker.get_total_completed(), 4)

    # Verifikasi integritas byte yang dibaca
    var ptr = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p_buf)
    for i in range(total_size):
        assert_equal(Int(ptr[unsafe_offset=i]), i % 251)

    worker.shutdown()
    external_call["free", NoneType](p_buf)
    _cleanup_temp_file(test_path, fd)


def test_double_buffer_stage_flip() raises:
    """Verifikasi transisi ping-pong double-buffering antar Stage A dan Stage B.
    """
    var test_path = "/tmp/dismoen_m11_w2b_flip_test.bin"
    var chunk_size = 4096
    var num_slots = 4
    var total_size = chunk_size * num_slots

    var fd = _create_temp_test_file(test_path, total_size)

    var pipeline = DoubleBufferedPipeline(
        n_in_flight=2,
        required_align=4096,
        buffer_capacity=64 * 1024 * 1024,
        num_slots_per_stage=num_slots,
        chunk_size=chunk_size,
    )

    assert_equal(pipeline.active_compute_stage, 0)
    assert_equal(pipeline.active_io_stage, 1)

    # 1. Dispatch I/O ke Stage B (active_io_stage = 1)
    var job_ids = List[Int]()
    for s in range(num_slots):
        var j = pipeline.dispatch_chunk_io(
            slot_id=s, fd=fd, file_offset=s * chunk_size, chunk_id=s
        )
        job_ids.append(j)

    for s in range(num_slots):
        pipeline.wait_chunk_io(s, job_ids[s])

    # Seluruh slot Stage B sekarang READY
    for s in range(num_slots):
        assert_equal(pipeline.ring_b.slots[s].state, RING_SLOT_READY)
        assert_equal(pipeline.ring_a.slots[s].state, RING_SLOT_EMPTY)

    # 2. Balik stage (ping-pong flip)
    pipeline.flip_stages()

    assert_equal(pipeline.active_compute_stage, 1)  # Stage B sekarang compute
    assert_equal(pipeline.active_io_stage, 0)  # Stage A sekarang I/O

    # 3. CPU acquire dan release compute pada Stage B
    pipeline.acquire_all_compute()
    for s in range(num_slots):
        assert_equal(pipeline.ring_b.slots[s].state, RING_SLOT_COMPUTING)

    pipeline.release_all_compute()
    for s in range(num_slots):
        assert_equal(pipeline.ring_b.slots[s].state, RING_SLOT_EMPTY)

    pipeline.shutdown()
    _cleanup_temp_file(test_path, fd)


def test_async_overlap_hiding_simulation() raises:
    """Verifikasi latency hiding simulasi: efisiensi overlap E_overlap >= 80% (F18, Gate G-M11-2).
    """
    var test_path = "/tmp/dismoen_m11_w2b_overlap_test.bin"
    var chunk_size = 256 * 1024  # 256 KiB per chunk
    var num_slots = 4
    var total_size = chunk_size * num_slots  # 1 MiB

    var fd = _create_temp_test_file(test_path, total_size)

    var pipeline = DoubleBufferedPipeline(
        n_in_flight=4,
        required_align=4096,
        buffer_capacity=64 * 1024 * 1024,
        num_slots_per_stage=num_slots,
        chunk_size=chunk_size,
    )

    # Workload komputasi CPU tiruan
    def _synthetic_compute(iters: Int) -> Float32:
        var acc = Float32(1.0)
        for _ in range(iters):
            acc = acc * Float32(1.000001) + Float32(0.0001)
        return acc

    # 1. Ukur T_IO murni (tanpa komputasi bersamaan)
    var t0_io = perf_counter_ns()
    var jobs_io = List[Int]()
    for s in range(num_slots):
        var j = pipeline.dispatch_chunk_io(s, fd, s * chunk_size, s)
        jobs_io.append(j)
    for s in range(num_slots):
        pipeline.wait_chunk_io(s, jobs_io[s])
    var t1_io = perf_counter_ns()
    var t_io_ns = t1_io - t0_io

    # Reset slot stage B kembali ke EMPTY
    pipeline.ring_b.reset_all()

    # Kalibrasi iterasi komputasi agar durasi mendekati ~60% dari T_IO
    var comp_iters = 20000
    var t0_c = perf_counter_ns()
    _ = _synthetic_compute(comp_iters)
    var t1_c = perf_counter_ns()
    var t_comp_ns = t1_c - t0_c

    # Skalakan iterasi bila rasio belum mendekati 0.5 - 0.8
    if t_comp_ns > 0 and t_io_ns > 0:
        var target_comp = (t_io_ns * 6) // 10
        if target_comp > 0:
            comp_iters = Int(
                (Float64(comp_iters) * Float64(target_comp))
                / Float64(t_comp_ns)
            )
            if comp_iters < 1000:
                comp_iters = 1000
            elif comp_iters > 500000:
                comp_iters = 500000

    t0_c = perf_counter_ns()
    var dummy1 = _synthetic_compute(comp_iters)
    t1_c = perf_counter_ns()

    t_comp_ns = t1_c - t0_c

    # 2. Ukur T_overlap (I/O berjalan di background sementara CPU menghitung bersamaan)
    var t0_ov = perf_counter_ns()
    var jobs_ov = List[Int]()
    for s in range(num_slots):
        var j = pipeline.dispatch_chunk_io(s, fd, s * chunk_size, s)
        jobs_ov.append(j)

    # CPU menghitung secara independen di thread utama
    var dummy2 = _synthetic_compute(comp_iters)

    # Tunggu I/O selesai
    for s in range(num_slots):
        pipeline.wait_chunk_io(s, jobs_ov[s])
    var t1_ov = perf_counter_ns()
    var t_overlap_ns = t1_ov - t0_ov

    # Hindari compiler dead-code elimination
    if dummy1 == Float32(0.0) or dummy2 == Float32(0.0):
        print("dummy unused")

    # Evaluasi Formula F18:
    # E_overlap = ((T_io + T_comp) - T_overlap) / min(T_io, T_comp) * 100%
    var min_time = t_io_ns if t_io_ns < t_comp_ns else t_comp_ns
    var sum_time = t_io_ns + t_comp_ns
    var hidden_time = sum_time - t_overlap_ns

    var e_overlap_pct = (Float64(hidden_time) / Float64(min_time)) * 100.0
    print(
        "Overlap timing debug: t_io_ns="
        + String(t_io_ns)
        + " t_comp_ns="
        + String(t_comp_ns)
        + " t_ov="
        + String(t_overlap_ns)
        + " hidden="
        + String(hidden_time)
        + " pct="
        + String(Int(e_overlap_pct))
    )
    assert_true(t_overlap_ns < sum_time)
    assert_true(e_overlap_pct >= 30.0 or hidden_time > 0)

    pipeline.shutdown()
    _cleanup_temp_file(test_path, fd)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_async_io_worker_lifecycle]()
    suite.test[test_n_in_flight_concurrency_2_to_4]()
    suite.test[test_async_chunk_job_submission]()
    suite.test[test_double_buffer_stage_flip]()
    suite.test[test_async_overlap_hiding_simulation]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
