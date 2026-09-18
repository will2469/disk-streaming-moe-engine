# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Formal Staging Buffer & Chunk-Level Ring Scheduling (§3.1, M11-W2a).

Mengimplementasikan:
1. Alokasi memori staging M_staging >= 64 MiB x 2 = 128 MiB upfront, required-aligned.
2. Dua staging buffers (Buffer A dan Buffer B, buffer_count = 2) sebagai slot tahap pipeline.
3. Invarian §3.1: Ring slots TIDAK mengalokasikan memori dinamis di luar staging memory.
4. Ring slot state machine: EMPTY -> IO_IN_FLIGHT -> READY -> COMPUTING -> EMPTY.
5. Kepemilikan eksklusif: Thread I/O (tulis saat IO_IN_FLIGHT) vs CPU workers (baca saat COMPUTING).
6. Sinyal completion per-chunk (bytes_transferred == chunk_size).
7. Profil target S_chunk = 3,493,888 B (853 * 4096).
"""

from io.dio_probe import (
    BASE_BUFFER_CAPACITY,
    BASE_CHUNK_SIZE,
    calculate_buffer_capacity,
    calculate_chunk_size,
    round_up_dio,
)
from std.collections import List
from std.ffi import external_call

# State Machine Constants (§3.1 Ring Slots)
comptime RING_SLOT_EMPTY = 0
comptime RING_SLOT_IO_IN_FLIGHT = 1
comptime RING_SLOT_READY = 2
comptime RING_SLOT_COMPUTING = 3


def ring_slot_state_name(state: Int) -> String:
    """Mengonversi kode status slot ring ke nama string representatif."""
    if state == RING_SLOT_EMPTY:
        return "EMPTY"
    if state == RING_SLOT_IO_IN_FLIGHT:
        return "IO_IN_FLIGHT"
    if state == RING_SLOT_READY:
        return "READY"
    if state == RING_SLOT_COMPUTING:
        return "COMPUTING"
    return "UNKNOWN"


@fieldwise_init
struct RingSlot(Copyable, Movable):
    """Slot penjadwalan chunk dalam chunk-level ring buffer (§3.1)."""

    var slot_id: Int
    var state: Int
    var stage_id: Int  # 0 = Buffer A, 1 = Buffer B
    var chunk_id: Int  # ID chunk logis (mis. expert index 0..7)
    var staging_addr: Int  # Alamat memori di dalam StagingMemory (required-aligned)
    var expected_bytes: Int
    var bytes_transferred: Int


struct StagingMemory:
    """Alokasi memori staging upfront ter-align (buffer_capacity x buffer_count >= 128 MiB).
    """

    var raw_addr: Int
    var buffer_capacity: Int
    var buffer_count: Int
    var total_staging_bytes: Int
    var required_align: Int

    def __init__(
        out self,
        required_align: Int = 4096,
        buffer_capacity: Int = BASE_BUFFER_CAPACITY,
        buffer_count: Int = 2,
    ) raises:
        """Mengalokasikan memori staging upfront ter-align.

        Invarian §3.1:
        - buffer_capacity >= 64 MiB (round-up ke kelipatan required_align)
        - buffer_count == 2 (Stage A dan Stage B)
        - total_staging_bytes >= 128 MiB
        - raw_addr != 0 dan raw_addr % required_align == 0
        """
        self.required_align = required_align if required_align > 0 else 4096
        self.buffer_capacity = calculate_buffer_capacity(self.required_align)
        if buffer_capacity > self.buffer_capacity:
            self.buffer_capacity = round_up_dio(
                buffer_capacity, self.required_align
            )

        self.buffer_count = buffer_count if buffer_count >= 2 else 2
        self.total_staging_bytes = self.buffer_capacity * self.buffer_count

        self.raw_addr = external_call["aligned_alloc", Int](
            self.required_align, self.total_staging_bytes
        )
        if self.raw_addr == 0:
            raise Error(
                "StagingMemory: failed to allocate upfront aligned staging"
                " buffer of "
                + String(self.total_staging_bytes)
                + " bytes"
            )

    def get_stage_addr(self, stage_id: Int) raises -> Int:
        """Mendapatkan pointer awal untuk slot tahap stage_id (0 = Buffer A, 1 = Buffer B).
        """
        if stage_id < 0 or stage_id >= self.buffer_count:
            raise Error(
                "StagingMemory: stage_id out of range: "
                + String(stage_id)
                + " (buffer_count="
                + String(self.buffer_count)
                + ")"
            )
        return self.raw_addr + stage_id * self.buffer_capacity

    def get_chunk_addr(
        self, stage_id: Int, chunk_idx: Int, chunk_size: Int
    ) raises -> Int:
        """Mendapatkan alamat memori untuk chunk_idx di dalam stage_id."""
        var stage_base = self.get_stage_addr(stage_id)
        var offset = chunk_idx * chunk_size
        if offset + chunk_size > self.buffer_capacity:
            raise Error(
                "StagingMemory: chunk placement exceeds buffer_capacity: offset"
                " "
                + String(offset + chunk_size)
                + " > "
                + String(self.buffer_capacity)
            )
        return stage_base + offset

    def free_staging(mut self):
        """Membebaskan memori staging upfront."""
        if self.raw_addr != 0:
            external_call["free", NoneType](self.raw_addr)
            self.raw_addr = 0


struct ChunkRingBuffer:
    """State machine penjadwalan chunk-level ring buffer (§3.1)."""

    var num_slots: Int
    var chunk_size: Int
    var stage_id: Int
    var slots: List[RingSlot]

    def __init__(
        out self,
        stage_base_addr: Int,
        stage_id: Int = 0,
        num_slots: Int = 4,
        chunk_size: Int = BASE_CHUNK_SIZE,
    ):
        """Menginisialisasi ring slots yang dipetakan ke staging buffer stage_base_addr.
        """
        self.num_slots = num_slots if num_slots > 0 else 4
        self.chunk_size = chunk_size
        self.stage_id = stage_id
        self.slots = List[RingSlot]()

        for i in range(self.num_slots):
            var slot_addr = stage_base_addr + i * self.chunk_size
            self.slots.append(
                RingSlot(
                    slot_id=i,
                    state=RING_SLOT_EMPTY,
                    stage_id=stage_id,
                    chunk_id=-1,
                    staging_addr=slot_addr,
                    expected_bytes=self.chunk_size,
                    bytes_transferred=0,
                )
            )

    def dispatch_io(
        mut self, slot_id: Int, chunk_id: Int, expected_bytes: Int
    ) raises:
        """Transisi EMPTY -> IO_IN_FLIGHT (Ownership: Thread I/O hak tulis eksklusif).
        """
        if slot_id < 0 or slot_id >= self.num_slots:
            raise Error("ChunkRingBuffer: invalid slot_id " + String(slot_id))

        if self.slots[slot_id].state != RING_SLOT_EMPTY:
            raise Error(
                "ChunkRingBuffer: illegal dispatch on non-empty slot "
                + String(slot_id)
                + " (state="
                + ring_slot_state_name(self.slots[slot_id].state)
                + ")"
            )

        self.slots[slot_id].state = RING_SLOT_IO_IN_FLIGHT
        self.slots[slot_id].chunk_id = chunk_id
        self.slots[slot_id].expected_bytes = expected_bytes
        self.slots[slot_id].bytes_transferred = 0

    def complete_io(mut self, slot_id: Int, bytes_transferred: Int) raises:
        """Transisi IO_IN_FLIGHT -> READY (Verifikasi completion per-chunk)."""
        if slot_id < 0 or slot_id >= self.num_slots:
            raise Error("ChunkRingBuffer: invalid slot_id " + String(slot_id))

        if self.slots[slot_id].state != RING_SLOT_IO_IN_FLIGHT:
            raise Error(
                "ChunkRingBuffer: illegal complete_io on slot "
                + String(slot_id)
                + " (expected IO_IN_FLIGHT, got "
                + ring_slot_state_name(self.slots[slot_id].state)
                + ")"
            )

        # Invarian completion per-chunk: bytes_transferred == chunk_size
        if bytes_transferred != self.slots[slot_id].expected_bytes:
            raise Error(
                "ChunkRingBuffer: chunk completion size mismatch on slot "
                + String(slot_id)
                + ": transferred "
                + String(bytes_transferred)
                + " B, expected "
                + String(self.slots[slot_id].expected_bytes)
                + " B"
            )

        self.slots[slot_id].state = RING_SLOT_READY
        self.slots[slot_id].bytes_transferred = bytes_transferred

    def acquire_compute(mut self, slot_id: Int) raises:
        """Transisi READY -> COMPUTING (Ownership: CPU Worker hak baca eksklusif).
        """
        if slot_id < 0 or slot_id >= self.num_slots:
            raise Error("ChunkRingBuffer: invalid slot_id " + String(slot_id))

        if self.slots[slot_id].state != RING_SLOT_READY:
            raise Error(
                "ChunkRingBuffer: illegal acquire_compute on slot "
                + String(slot_id)
                + " (expected READY, got "
                + ring_slot_state_name(self.slots[slot_id].state)
                + ")"
            )

        self.slots[slot_id].state = RING_SLOT_COMPUTING

    def release_compute(mut self, slot_id: Int) raises:
        """Transisi COMPUTING -> EMPTY (Slot kembali siap untuk I/O baru)."""
        if slot_id < 0 or slot_id >= self.num_slots:
            raise Error("ChunkRingBuffer: invalid slot_id " + String(slot_id))

        if self.slots[slot_id].state != RING_SLOT_COMPUTING:
            raise Error(
                "ChunkRingBuffer: illegal release_compute on slot "
                + String(slot_id)
                + " (expected COMPUTING, got "
                + ring_slot_state_name(self.slots[slot_id].state)
                + ")"
            )

        self.slots[slot_id].state = RING_SLOT_EMPTY
        self.slots[slot_id].chunk_id = -1
        self.slots[slot_id].bytes_transferred = 0

    def find_empty_slot(self) -> Int:
        """Mencari slot kosong pertama, atau -1 jika seluruh slot penuh."""
        for i in range(self.num_slots):
            if self.slots[i].state == RING_SLOT_EMPTY:
                return i
        return -1

    def find_ready_slot(self) -> Int:
        """Mencari slot siap komputasi pertama, atau -1 jika belum ada."""
        for i in range(self.num_slots):
            if self.slots[i].state == RING_SLOT_READY:
                return i
        return -1

    def count_in_flight(self) -> Int:
        """Menghitung jumlah I/O in-flight saat ini."""
        var count = 0
        for i in range(self.num_slots):
            if self.slots[i].state == RING_SLOT_IO_IN_FLIGHT:
                count += 1
        return count

    def count_ready(self) -> Int:
        """Menghitung jumlah slot yang sudah siap untuk dikomputasi."""
        var count = 0
        for i in range(self.num_slots):
            if self.slots[i].state == RING_SLOT_READY:
                count += 1
        return count

    def is_all_empty(self) -> Bool:
        """Memeriksa apakah seluruh slot ring dalam keadaan EMPTY."""
        for i in range(self.num_slots):
            if self.slots[i].state != RING_SLOT_EMPTY:
                return False
        return True

    def reset_all(mut self):
        """Mereset seluruh slot kembali ke EMPTY."""
        for i in range(self.num_slots):
            self.slots[i].state = RING_SLOT_EMPTY
            self.slots[i].chunk_id = -1
            self.slots[i].bytes_transferred = 0
