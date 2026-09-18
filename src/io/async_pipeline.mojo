# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Double-Buffered Asynchronous Pipeline Coordinator (§1.4, §3.1, Gate G-M11-2).

Mengimplementasikan:
1. Pipelining asinkron ping-pong Stage A dan Stage B (double-buffered).
2. Integrasi StagingMemory upfront (>= 128 MiB, required-aligned), ChunkRingBuffer, dan AsyncIOWorker.
3. Transisi slot ring non-blocking dan verifikasi completion per-chunk.
4. Isolasi state: Stage aktif dihitung oleh CPU WorkerPool, Stage latar belakang diisi asinkron oleh AsyncIOWorker.
5. Invarian flip_stages: memastikan seluruh I/O stage selesai (READY) dan seluruh komputasi stage selesai (EMPTY) sebelum flipping.
"""

from io.dio_probe import BASE_BUFFER_CAPACITY, BASE_CHUNK_SIZE
from io.staging_ring import (
    RING_SLOT_COMPUTING,
    RING_SLOT_EMPTY,
    RING_SLOT_IO_IN_FLIGHT,
    RING_SLOT_READY,
    ChunkRingBuffer,
    StagingMemory,
    ring_slot_state_name,
)
from io.async_worker import AsyncIOWorker, IOJob
from std.collections import List


struct DoubleBufferedPipeline:
    """Koordinator Double-Buffered Asynchronous Streaming Pipeline (§3.1, Gate G-M11-2).
    """

    var staging: StagingMemory
    var ring_a: ChunkRingBuffer
    var ring_b: ChunkRingBuffer
    var worker: AsyncIOWorker
    var active_compute_stage: Int  # 0 = Buffer A, 1 = Buffer B
    var active_io_stage: Int  # 1 = Buffer B, 0 = Buffer A
    var num_slots_per_stage: Int
    var chunk_size: Int
    var n_in_flight: Int
    var is_active: Bool

    def __init__(
        out self,
        n_in_flight: Int = 2,
        required_align: Int = 4096,
        buffer_capacity: Int = BASE_BUFFER_CAPACITY,
        num_slots_per_stage: Int = 8,
        chunk_size: Int = BASE_CHUNK_SIZE,
    ) raises:
        """Menginisialisasi pipeline double-buffered dengan memori staging upfront ter-align.

        Invarian:
        - n_in_flight in [2, 4] (§1.3 properti engine).
        - StagingMemory >= 128 MiB (buffer_count = 2).
        - Dua ChunkRingBuffer independen untuk Buffer A dan Buffer B.
        """
        if n_in_flight < 2 or n_in_flight > 4:
            raise Error(
                "DoubleBufferedPipeline: n_in_flight must be in [2, 4], got "
                + String(n_in_flight)
            )

        self.num_slots_per_stage = (
            num_slots_per_stage if num_slots_per_stage > 0 else 8
        )
        self.chunk_size = chunk_size
        self.n_in_flight = n_in_flight
        self.active_compute_stage = 0
        self.active_io_stage = 1
        self.is_active = False

        # Alokasi upfront staging memory (2 x 64 MiB >= 128 MiB)
        self.staging = StagingMemory(
            required_align=required_align,
            buffer_capacity=buffer_capacity,
            buffer_count=2,
        )

        var addr_a = self.staging.get_stage_addr(0)
        var addr_b = self.staging.get_stage_addr(1)

        self.ring_a = ChunkRingBuffer(
            stage_base_addr=addr_a,
            stage_id=0,
            num_slots=self.num_slots_per_stage,
            chunk_size=self.chunk_size,
        )

        self.ring_b = ChunkRingBuffer(
            stage_base_addr=addr_b,
            stage_id=1,
            num_slots=self.num_slots_per_stage,
            chunk_size=self.chunk_size,
        )

        self.worker = AsyncIOWorker(n_in_flight=self.n_in_flight)
        self.is_active = True

    def dispatch_chunk_io(
        mut self, slot_id: Int, fd: Int, file_offset: Int, chunk_id: Int
    ) raises -> Int:
        """Mengirimkan permintaan pembacaan chunk pada stage I/O aktif ke AsyncIOWorker.

        Transisi ring slot: EMPTY -> IO_IN_FLIGHT.
        Mengembalikan job_id unik.
        """
        if not self.is_active:
            raise Error("DoubleBufferedPipeline: pipeline is inactive")

        var dest_addr: Int
        if self.active_io_stage == 0:
            self.ring_a.dispatch_io(slot_id, chunk_id, self.chunk_size)
            dest_addr = self.ring_a.slots[slot_id].staging_addr
        else:
            self.ring_b.dispatch_io(slot_id, chunk_id, self.chunk_size)
            dest_addr = self.ring_b.slots[slot_id].staging_addr

        return self.worker.submit_job(
            fd=fd,
            file_offset=file_offset,
            dest_addr=dest_addr,
            chunk_size=self.chunk_size,
            slot_id=slot_id,
        )

    def wait_chunk_io(mut self, slot_id: Int, job_id: Int) raises:
        """Menunggu job I/O selesai dan memverifikasi completion per-chunk.

        Transisi ring slot: IO_IN_FLIGHT -> READY.
        """
        var bytes_transferred = self.worker.wait_job(job_id)
        if self.active_io_stage == 0:
            self.ring_a.complete_io(slot_id, bytes_transferred)
        else:
            self.ring_b.complete_io(slot_id, bytes_transferred)

    def wait_stage_io_all(
        mut self, job_ids: List[Int], slot_ids: List[Int]
    ) raises:
        """Menunggu seluruh job I/O pada stage aktif selesai dan menandai READY seluruh slot.
        """
        self.worker.wait_all()
        for i in range(len(slot_ids)):
            var s = slot_ids[i]
            var j = job_ids[i]
            var bytes_transferred = self.worker.wait_job(j)
            if self.active_io_stage == 0:
                self.ring_a.complete_io(s, bytes_transferred)
            else:
                self.ring_b.complete_io(s, bytes_transferred)

    def acquire_compute(mut self, slot_id: Int) raises -> Int:
        """Mengambil slot pada stage komputasi aktif untuk diproses CPU.

        Transisi ring slot: READY -> COMPUTING.
        Mengembalikan alamat buffer slot staging.
        """
        if self.active_compute_stage == 0:
            self.ring_a.acquire_compute(slot_id)
            return self.ring_a.slots[slot_id].staging_addr
        else:
            self.ring_b.acquire_compute(slot_id)
            return self.ring_b.slots[slot_id].staging_addr

    def release_compute(mut self, slot_id: Int) raises:
        """Melepaskan slot komputasi setelah selesai diproses CPU.

        Transisi ring slot: COMPUTING -> EMPTY.
        """
        if self.active_compute_stage == 0:
            self.ring_a.release_compute(slot_id)
        else:
            self.ring_b.release_compute(slot_id)

    def acquire_all_compute(mut self) raises:
        """Mengambil seluruh slot pada stage komputasi aktif ke status COMPUTING.
        """
        for s in range(self.num_slots_per_stage):
            if self.active_compute_stage == 0:
                self.ring_a.acquire_compute(s)
            else:
                self.ring_b.acquire_compute(s)

    def release_all_compute(mut self) raises:
        """Melepaskan seluruh slot pada stage komputasi aktif kembali ke EMPTY.
        """
        for s in range(self.num_slots_per_stage):
            if self.active_compute_stage == 0:
                self.ring_a.release_compute(s)
            else:
                self.ring_b.release_compute(s)

    def get_compute_stage_addr(self) raises -> Int:
        """Mendapatkan alamat memori awal stage komputasi aktif."""
        return self.staging.get_stage_addr(self.active_compute_stage)

    def get_io_stage_addr(self) raises -> Int:
        """Mendapatkan alamat memori awal stage I/O aktif."""
        return self.staging.get_stage_addr(self.active_io_stage)

    def flip_stages(mut self) raises:
        """Membalik stage A dan B (ping-pong switch) pada batas layer/langkah.

        Invarian:
        - Seluruh slot pada stage komputasi aktif harus sudah EMPTY.
        - Seluruh slot pada stage I/O aktif harus sudah READY.
        """
        # Verifikasi konsistensi state sebelum flip
        if self.active_compute_stage == 0:
            for s in range(self.num_slots_per_stage):
                var st = self.ring_a.slots[s].state
                if st != RING_SLOT_EMPTY:
                    raise Error(
                        "DoubleBufferedPipeline: cannot flip stages; compute"
                        " stage slot "
                        + String(s)
                        + " is not EMPTY (state="
                        + ring_slot_state_name(st)
                        + ")"
                    )
        else:
            for s in range(self.num_slots_per_stage):
                var st = self.ring_b.slots[s].state
                if st != RING_SLOT_EMPTY:
                    raise Error(
                        "DoubleBufferedPipeline: cannot flip stages; compute"
                        " stage slot "
                        + String(s)
                        + " is not EMPTY (state="
                        + ring_slot_state_name(st)
                        + ")"
                    )

        if self.active_io_stage == 0:
            for s in range(self.num_slots_per_stage):
                var st = self.ring_a.slots[s].state
                if st != RING_SLOT_READY:
                    raise Error(
                        "DoubleBufferedPipeline: cannot flip stages; I/O stage"
                        " slot "
                        + String(s)
                        + " is not READY (state="
                        + ring_slot_state_name(st)
                        + ")"
                    )
        else:
            for s in range(self.num_slots_per_stage):
                var st = self.ring_b.slots[s].state
                if st != RING_SLOT_READY:
                    raise Error(
                        "DoubleBufferedPipeline: cannot flip stages; I/O stage"
                        " slot "
                        + String(s)
                        + " is not READY (state="
                        + ring_slot_state_name(st)
                        + ")"
                    )

        # Lakukan pembalikan stage ping-pong
        var old_compute = self.active_compute_stage
        self.active_compute_stage = self.active_io_stage
        self.active_io_stage = old_compute

    def shutdown(mut self):
        """Menghentikan worker thread I/O dan membebaskan memori staging upfront.
        """
        if self.is_active:
            self.worker.shutdown()
            self.staging.free_staging()
            self.is_active = False
