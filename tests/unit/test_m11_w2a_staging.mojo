# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M11-W2a: Staging Buffer, Chunk Ring Scheduling, dan O_DIRECT Probe Alignment (§3.1).
"""

from io.dio_probe import (
    BASE_BUFFER_CAPACITY,
    BASE_CHUNK_SIZE,
    DioAlignment,
    calculate_buffer_capacity,
    calculate_chunk_size,
    is_triple_aligned,
    probe_dio_alignment,
    round_up_dio,
    validate_dio_constraints,
)
from io.staging_ring import (
    RING_SLOT_COMPUTING,
    RING_SLOT_EMPTY,
    RING_SLOT_IO_IN_FLIGHT,
    RING_SLOT_READY,
    ChunkRingBuffer,
    StagingMemory,
)
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)


def test_dio_probe_query_and_fallback() raises:
    """Verifikasi probe O_DIRECT statx dan fallback constraint terdokumentasi."""
    # 1. Probe path riil (src/main.mojo)
    var align_real = probe_dio_alignment("src/main.mojo")
    assert_true(align_real.mem_align > 0)
    assert_true(align_real.offset_align > 0)
    assert_true(align_real.length_align > 0)
    assert_true(
        align_real.source == "statx_dioalign"
        or align_real.source == "fallback_documented"
    )

    # 2. Probe path non-existent (harus menghasilkan fallback aman 4096B)
    var align_fallback = probe_dio_alignment("/nonexistent_path_dio_test")
    assert_equal(align_fallback.mem_align, 4096)
    assert_equal(align_fallback.offset_align, 4096)
    assert_equal(align_fallback.length_align, 4096)
    assert_false(align_fallback.is_probed_statx)
    assert_equal(align_fallback.source, "fallback_documented")


def test_dio_triple_alignment_validation() raises:
    """Verifikasi kepatuhan ketiga sisi: buffer address, file offset, dan request length."""
    var align = DioAlignment(
        mem_align=4096,
        offset_align=4096,
        length_align=4096,
        is_probed_statx=True,
        source="statx_dioalign",
    )

    # Valid triple alignment
    assert_true(is_triple_aligned(8192, 4096, 12288, align))
    validate_dio_constraints(8192, 4096, 12288, align)

    # Unaligned buffer address
    assert_false(is_triple_aligned(8193, 4096, 4096, align))
    var caught_addr = False
    try:
        validate_dio_constraints(8193, 4096, 4096, align)
    except:
        caught_addr = True
    assert_true(caught_addr)

    # Unaligned file offset
    assert_false(is_triple_aligned(4096, 512, 4096, align))
    var caught_off = False
    try:
        validate_dio_constraints(4096, 512, 4096, align)
    except:
        caught_off = True
    assert_true(caught_off)

    # Unaligned length
    assert_false(is_triple_aligned(4096, 4096, 4000, align))
    var caught_len = False
    try:
        validate_dio_constraints(4096, 4096, 4000, align)
    except:
        caught_len = True
    assert_true(caught_len)


def test_round_up_logic() raises:
    """Verifikasi round-up chunk size dan buffer capacity saat probe != 4096B."""
    # Profil standar 4096B
    assert_equal(calculate_chunk_size(4096), BASE_CHUNK_SIZE)
    assert_equal(calculate_buffer_capacity(4096), BASE_BUFFER_CAPACITY)

    # Skenario non-standar: alignment 8192B
    var c8k = calculate_chunk_size(8192)
    assert_true(c8k >= BASE_CHUNK_SIZE)
    assert_equal(c8k % 8192, 0)

    var b8k = calculate_buffer_capacity(8192)
    assert_true(b8k >= BASE_BUFFER_CAPACITY)
    assert_equal(b8k % 8192, 0)

    # Skenario 512B
    assert_equal(calculate_chunk_size(512), BASE_CHUNK_SIZE)
    assert_equal(calculate_buffer_capacity(512), BASE_BUFFER_CAPACITY)


def test_staging_memory_upfront_allocation() raises:
    """Verifikasi alokasi upfront M_staging >= 128 MiB ter-align dan layout dua tahap."""
    var staging = StagingMemory(
        required_align=4096,
        buffer_capacity=64 * 1024 * 1024,
        buffer_count=2,
    )

    # Invarian: total staging >= 128 MiB
    assert_true(staging.total_staging_bytes >= 128 * 1024 * 1024)
    assert_true(staging.raw_addr != 0)
    assert_equal(staging.raw_addr % 4096, 0)

    # Stage A dan Stage B terpisah tanpa overlap
    var addr_a = staging.get_stage_addr(0)
    var addr_b = staging.get_stage_addr(1)
    assert_equal(addr_a, staging.raw_addr)
    assert_equal(addr_b - addr_a, staging.buffer_capacity)
    assert_equal(addr_a % 4096, 0)
    assert_equal(addr_b % 4096, 0)

    # Alamat chunk di dalam Stage A
    var chunk_0 = staging.get_chunk_addr(0, 0, BASE_CHUNK_SIZE)
    var chunk_1 = staging.get_chunk_addr(0, 1, BASE_CHUNK_SIZE)
    assert_equal(chunk_0, addr_a)
    assert_equal(chunk_1 - chunk_0, BASE_CHUNK_SIZE)
    assert_equal(chunk_0 % 4096, 0)
    assert_equal(chunk_1 % 4096, 0)

    # Penempatan chunk melebihi kapasitas harus melempar error
    var caught_overflow = False
    try:
        _ = staging.get_chunk_addr(0, 100, BASE_CHUNK_SIZE)
    except:
        caught_overflow = True
    assert_true(caught_overflow)

    # Pembersihan
    staging.free_staging()
    assert_equal(staging.raw_addr, 0)


def test_ring_slot_state_machine_transitions() raises:
    """Verifikasi siklus formal EMPTY -> IO_IN_FLIGHT -> READY -> COMPUTING -> EMPTY."""
    var staging = StagingMemory(required_align=4096)
    var ring = ChunkRingBuffer(
        stage_base_addr=staging.get_stage_addr(0),
        stage_id=0,
        num_slots=4,
        chunk_size=BASE_CHUNK_SIZE,
    )

    assert_equal(ring.num_slots, 4)
    assert_true(ring.is_all_empty())
    assert_equal(ring.count_in_flight(), 0)
    assert_equal(ring.count_ready(), 0)

    # 1. Dispatch slot 0: EMPTY -> IO_IN_FLIGHT
    ring.dispatch_io(0, chunk_id=1, expected_bytes=BASE_CHUNK_SIZE)
    assert_equal(ring.slots[0].state, RING_SLOT_IO_IN_FLIGHT)
    assert_equal(ring.slots[0].chunk_id, 1)
    assert_equal(ring.count_in_flight(), 1)

    # Illegal dispatch ulang pada slot yang sama harus melempar error
    var caught_illegal_dispatch = False
    try:
        ring.dispatch_io(0, chunk_id=2, expected_bytes=BASE_CHUNK_SIZE)
    except:
        caught_illegal_dispatch = True
    assert_true(caught_illegal_dispatch)

    # Illegal acquire_compute pada slot yang belum READY harus melempar error
    var caught_premature_compute = False
    try:
        ring.acquire_compute(0)
    except:
        caught_premature_compute = True
    assert_true(caught_premature_compute)

    # 2. Complete I/O: IO_IN_FLIGHT -> READY
    ring.complete_io(0, bytes_transferred=BASE_CHUNK_SIZE)
    assert_equal(ring.slots[0].state, RING_SLOT_READY)
    assert_equal(ring.count_in_flight(), 0)
    assert_equal(ring.count_ready(), 1)

    # 3. Acquire compute: READY -> COMPUTING
    ring.acquire_compute(0)
    assert_equal(ring.slots[0].state, RING_SLOT_COMPUTING)
    assert_equal(ring.count_ready(), 0)

    # 4. Release compute: COMPUTING -> EMPTY
    ring.release_compute(0)
    assert_equal(ring.slots[0].state, RING_SLOT_EMPTY)
    assert_equal(ring.slots[0].chunk_id, -1)
    assert_true(ring.is_all_empty())

    staging.free_staging()


def test_chunk_completion_size_verification() raises:
    """Verifikasi bahwa penyelesaian I/O wajib memenuhi ukuran chunk eksak."""
    var staging = StagingMemory(required_align=4096)
    var ring = ChunkRingBuffer(
        stage_base_addr=staging.get_stage_addr(0),
        stage_id=0,
        num_slots=2,
        chunk_size=BASE_CHUNK_SIZE,
    )

    ring.dispatch_io(0, chunk_id=0, expected_bytes=BASE_CHUNK_SIZE)

    # Short read (kurang 1 byte) harus gagal fail-closed
    var caught_short_read = False
    try:
        ring.complete_io(0, bytes_transferred=BASE_CHUNK_SIZE - 1)
    except:
        caught_short_read = True
    assert_true(caught_short_read)

    # Transfer byte eksak harus berhasil
    ring.complete_io(0, bytes_transferred=BASE_CHUNK_SIZE)
    assert_equal(ring.slots[0].state, RING_SLOT_READY)

    staging.free_staging()


def test_1000_ring_transitions_stability() raises:
    """Verifikasi stabilitas 1000 transisi siklus ring buffer tanpa alokasi memori."""
    var staging = StagingMemory(required_align=4096)
    var ring = ChunkRingBuffer(
        stage_base_addr=staging.get_stage_addr(0),
        stage_id=0,
        num_slots=4,
        chunk_size=BASE_CHUNK_SIZE,
    )

    for cycle in range(1000):
        var s = cycle % 4
        ring.dispatch_io(s, chunk_id=cycle, expected_bytes=BASE_CHUNK_SIZE)
        ring.complete_io(s, bytes_transferred=BASE_CHUNK_SIZE)
        ring.acquire_compute(s)
        ring.release_compute(s)

    assert_true(ring.is_all_empty())
    staging.free_staging()


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_dio_probe_query_and_fallback]()
    suite.test[test_dio_triple_alignment_validation]()
    suite.test[test_round_up_logic]()
    suite.test[test_staging_memory_upfront_allocation]()
    suite.test[test_ring_slot_state_machine_transitions]()
    suite.test[test_chunk_completion_size_verification]()
    suite.test[test_1000_ring_transitions_stability]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
