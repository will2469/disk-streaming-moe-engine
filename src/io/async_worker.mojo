# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Dedicated Engine-Level Asynchronous I/O Worker & Outstanding I/O Pool (§1.3, §3.1, M11-W2b).

Mengimplementasikan:
1. Thread I/O berdedikasi berbasis POSIX pthread terpisah dari pool komputasi CPU.
2. Software pipelining: O_DIRECT / buffered pread blocking ke buffer staging tahap berikutnya.
3. Properti engine N_in_flight in [2, 4] untuk konkurensi I/O outstanding.
4. Sinkronisasi thread-safe bebas alokasi dinamis pada background thread (zero-alloc pthread loop).
5. Sinyal completion per-chunk untuk integrasi dengan ChunkRingBuffer.
"""

from std.collections import List
from std.ffi import external_call
from std.memory import Pointer

# Konstanta Status Job I/O
comptime JOB_EMPTY = 0
comptime JOB_SUBMITTED = 1
comptime JOB_IN_PROGRESS = 2
comptime JOB_COMPLETED = 3
comptime JOB_ERROR = 4

# Batas Maksimum Antrean Job Ring I/O
comptime MAX_IO_JOBS = 256
comptime JOB_TABLE_OFFSET = 32
comptime JOB_STRIDE = 8


@fieldwise_init
struct IOJob(Copyable, Movable):
    """Deskriptor pekerjaan I/O chunk asinkron (§3.1)."""

    var job_id: Int
    var fd: Int
    var file_offset: Int
    var dest_addr: Int
    var chunk_size: Int
    var slot_id: Int
    var status: Int
    var bytes_read: Int

    def is_completed(self) -> Bool:
        """Memeriksa apakah job telah selesai dibaca."""
        return self.status == JOB_COMPLETED

    def is_error(self) -> Bool:
        """Memeriksa apakah job mengalami kesalahan I/O."""
        return self.status == JOB_ERROR


def _io_worker_thread_loop(p_arg: Pointer[Int, MutAnyOrigin]) -> Int:
    """Loop eksekusi utama thread pekerja I/O latar belakang (zero-alloc)."""
    var shared_addr = p_arg[unsafe_offset=0]
    var p_shared = Pointer[Int, MutAnyOrigin](unsafe_from_address=shared_addr)
    var mutex_addr = Int(p_shared.unsafe_offset(0))
    var cond_work_addr = Int(p_shared.unsafe_offset(8))
    var cond_done_addr = Int(p_shared.unsafe_offset(16))

    while True:
        _ = external_call["pthread_mutex_lock", Int32](mutex_addr)

        # Tunggu pekerjaan baru atau sinyal keluar
        while (
            p_shared[unsafe_offset=24] == 0
            and p_shared[unsafe_offset=27] == p_shared[unsafe_offset=28]
        ):
            _ = external_call["pthread_cond_wait", Int32](
                cond_work_addr, mutex_addr
            )

        # Jika bendera keluar aktif dan seluruh antrean kosong
        if (
            p_shared[unsafe_offset=24] != 0
            and p_shared[unsafe_offset=27] == p_shared[unsafe_offset=28]
        ):
            _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)
            break

        var tail = p_shared[unsafe_offset=28]
        var max_jobs = p_shared[unsafe_offset=26]
        var slot_idx = tail % max_jobs
        var base = JOB_TABLE_OFFSET + slot_idx * JOB_STRIDE

        var fd = p_shared[unsafe_offset=base + 1]
        var file_offset = p_shared[unsafe_offset=base + 2]
        var dest_addr = p_shared[unsafe_offset=base + 3]
        var chunk_size = p_shared[unsafe_offset=base + 4]
        var slot_id = p_shared[unsafe_offset=base + 5]

        p_shared[unsafe_offset=base + 0] = JOB_IN_PROGRESS

        # Lepas mutex selama pemanggilan I/O disk blocking agar CPU thread pool tidak terhambat
        _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)

        # Eksekusi pread blocking di luar lock (O_DIRECT / buffer stage berikutnya)
        var total_read = 0
        var io_err = 0
        while total_read < chunk_size:
            var n = external_call["pread", Int](
                Int32(fd),
                dest_addr + total_read,
                chunk_size - total_read,
                file_offset + total_read,
            )
            if n == 0:
                break  # EOF terdeteksi
            if n < 0:
                io_err = 1
                break
            total_read += n

        # Ambil kembali mutex untuk memperbarui status dan broadcast penyelesaian
        _ = external_call["pthread_mutex_lock", Int32](mutex_addr)
        if io_err != 0:
            p_shared[unsafe_offset=base + 0] = JOB_ERROR
            p_shared[unsafe_offset=25] = 1  # Global error flag
        else:
            p_shared[unsafe_offset=base + 0] = JOB_COMPLETED

        p_shared[unsafe_offset=base + 6] = total_read
        p_shared[unsafe_offset=28] = tail + 1
        p_shared[unsafe_offset=30] += 1

        # Broadcast sinyal penyelesaian ke thread koordinator
        _ = external_call["pthread_cond_broadcast", Int32](cond_done_addr)
        _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)

    return 0


struct AsyncIOWorker:
    """Dedicated Engine-Level Asynchronous I/O Worker (§3.1, Gate G-M11-2).

    Mengelola antrean chunk I/O outstanding dengan konkurensi N_in_flight in [2, 4].
    """

    var n_in_flight: Int
    var is_active: Bool
    var shared_mem: List[Int]
    var worker_args: List[Int]
    var thread_ids: List[Int]

    def __init__(out self, n_in_flight: Int = 2) raises:
        """Menginisialisasi thread I/O dedikasi dan struktur sinkronisasi POSIX.

        Invarian:
        - n_in_flight in [2, 4] (§1.3 properti engine).
        - Upfront allocation shared control table (zero-alloc di hot loop).
        """
        if n_in_flight < 2 or n_in_flight > 4:
            raise Error(
                "AsyncIOWorker: n_in_flight must be in [2, 4] (§1.3 engine"
                " property), got "
                + String(n_in_flight)
            )

        self.n_in_flight = n_in_flight
        self.is_active = False

        # shared_mem layout (320 Int = 2560 B):
        # [0..7]:   pthread_mutex_t (40-64 B)
        # [8..15]:  pthread_cond_t cond_work (48-64 B)
        # [16..23]: pthread_cond_t cond_done (48-64 B)
        # [24]:     exit_flag (0 / 1)
        # [25]:     error_flag (0 / 1)
        # [26]:     max_jobs (MAX_IO_JOBS = 32)
        # [27]:     head (job counter / submit index)
        # [28]:     tail (worker progress index)
        # [29]:     n_in_flight
        # [30]:     total_completed
        # [31]:     is_active
        # [32..]:   Job table entries (8 Int per job)
        self.shared_mem = List[Int]()
        for _ in range(32 + MAX_IO_JOBS * JOB_STRIDE):
            self.shared_mem.append(0)

        self.worker_args = List[Int]()
        self.worker_args.append(0)

        self.thread_ids = List[Int]()
        self.thread_ids.append(0)

        var p_shared = self.shared_mem.unsafe_ptr()
        var mutex_addr = Int(p_shared.unsafe_offset(0))
        var cond_work_addr = Int(p_shared.unsafe_offset(8))
        var cond_done_addr = Int(p_shared.unsafe_offset(16))

        _ = external_call["pthread_mutex_init", Int32](mutex_addr, Int(0))
        _ = external_call["pthread_cond_init", Int32](cond_work_addr, Int(0))
        _ = external_call["pthread_cond_init", Int32](cond_done_addr, Int(0))

        p_shared[unsafe_offset=24] = 0  # exit_flag = 0
        p_shared[unsafe_offset=25] = 0  # error_flag = 0
        p_shared[unsafe_offset=26] = MAX_IO_JOBS
        p_shared[unsafe_offset=27] = 0  # head = 0
        p_shared[unsafe_offset=28] = 0  # tail = 0
        p_shared[unsafe_offset=29] = self.n_in_flight
        p_shared[unsafe_offset=30] = 0  # total_completed = 0
        p_shared[unsafe_offset=31] = 1  # is_active = 1

        self.worker_args[0] = Int(p_shared)
        var p_arg = self.worker_args.unsafe_ptr()
        var p_tid = self.thread_ids.unsafe_ptr()

        var rc = external_call["pthread_create", Int32](
            p_tid, Int(0), _io_worker_thread_loop, p_arg
        )
        if rc != 0:
            raise Error(
                "AsyncIOWorker: failed to spawn dedicated I/O pthread (rc="
                + String(rc)
                + ")"
            )

        self.is_active = True

    def submit_job(
        mut self,
        fd: Int,
        file_offset: Int,
        dest_addr: Int,
        chunk_size: Int,
        slot_id: Int = -1,
    ) raises -> Int:
        """Mengirimkan permintaan pembacaan chunk ke thread I/O latar belakang.

        Mengembalikan job_id unik (monotonic).
        """
        if not self.is_active:
            raise Error("AsyncIOWorker: cannot submit job to inactive worker")

        var p_shared = self.shared_mem.unsafe_ptr()
        var mutex_addr = Int(p_shared.unsafe_offset(0))
        var cond_work_addr = Int(p_shared.unsafe_offset(8))
        var cond_done_addr = Int(p_shared.unsafe_offset(16))
        var max_jobs = p_shared[unsafe_offset=26]

        _ = external_call["pthread_mutex_lock", Int32](mutex_addr)

        # Tunggu bila antrean penuh (backpressure)
        while (
            p_shared[unsafe_offset=27] - p_shared[unsafe_offset=28] >= max_jobs
        ):
            _ = external_call["pthread_cond_wait", Int32](
                cond_done_addr, mutex_addr
            )

        var job_id = p_shared[unsafe_offset=27]
        var slot_idx = job_id % max_jobs
        var base = JOB_TABLE_OFFSET + slot_idx * JOB_STRIDE

        p_shared[unsafe_offset=base + 0] = JOB_SUBMITTED
        p_shared[unsafe_offset=base + 1] = fd
        p_shared[unsafe_offset=base + 2] = file_offset
        p_shared[unsafe_offset=base + 3] = dest_addr
        p_shared[unsafe_offset=base + 4] = chunk_size
        p_shared[unsafe_offset=base + 5] = slot_id
        p_shared[unsafe_offset=base + 6] = 0
        p_shared[unsafe_offset=base + 7] = job_id

        p_shared[unsafe_offset=27] = job_id + 1

        # Bangunkan thread I/O
        _ = external_call["pthread_cond_signal", Int32](cond_work_addr)
        _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)

        return job_id

    def wait_job(self, job_id: Int) raises -> Int:
        """Menunggu hingga job_id tertentu selesai dieksekusi.

        Mengembalikan jumlah byte yang berhasil dibaca.
        """
        var p_shared = self.shared_mem.unsafe_ptr()
        var mutex_addr = Int(p_shared.unsafe_offset(0))
        var cond_done_addr = Int(p_shared.unsafe_offset(16))
        var max_jobs = p_shared[unsafe_offset=26]
        var slot_idx = job_id % max_jobs
        var base = JOB_TABLE_OFFSET + slot_idx * JOB_STRIDE

        _ = external_call["pthread_mutex_lock", Int32](mutex_addr)
        while True:
            var status = p_shared[unsafe_offset=base + 0]
            var stored_id = p_shared[unsafe_offset=base + 7]
            if stored_id == job_id:
                if status == JOB_COMPLETED:
                    var bytes_read = p_shared[unsafe_offset=base + 6]
                    _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)
                    return bytes_read
                elif status == JOB_ERROR:
                    _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)
                    raise Error(
                        "AsyncIOWorker: I/O error occurred executing job "
                        + String(job_id)
                    )
            _ = external_call["pthread_cond_wait", Int32](
                cond_done_addr, mutex_addr
            )

    def wait_all(self) raises:
        """Menunggu hingga seluruh job yang telah disubmit selesai dibaca."""
        var p_shared = self.shared_mem.unsafe_ptr()
        var mutex_addr = Int(p_shared.unsafe_offset(0))
        var cond_done_addr = Int(p_shared.unsafe_offset(16))

        _ = external_call["pthread_mutex_lock", Int32](mutex_addr)
        while p_shared[unsafe_offset=28] < p_shared[unsafe_offset=27]:
            _ = external_call["pthread_cond_wait", Int32](
                cond_done_addr, mutex_addr
            )
        var has_err = p_shared[unsafe_offset=25]
        _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)

        if has_err != 0:
            raise Error(
                "AsyncIOWorker: one or more I/O jobs failed during wait_all"
            )

    def is_job_completed(self, job_id: Int) -> Bool:
        """Pemeriksaan non-blocking apakah job_id telah mencapai status COMPLETED atau ERROR.
        """
        var p_shared = self.shared_mem.unsafe_ptr()
        var mutex_addr = Int(p_shared.unsafe_offset(0))
        var max_jobs = p_shared[unsafe_offset=26]
        var slot_idx = job_id % max_jobs
        var base = JOB_TABLE_OFFSET + slot_idx * JOB_STRIDE

        _ = external_call["pthread_mutex_lock", Int32](mutex_addr)
        var status = p_shared[unsafe_offset=base + 0]
        var stored_id = p_shared[unsafe_offset=base + 7]
        _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)

        return stored_id == job_id and (
            status == JOB_COMPLETED or status == JOB_ERROR
        )

    def get_in_flight_count(self) -> Int:
        """Menghitung jumlah chunk I/O outstanding saat ini."""
        var p_shared = self.shared_mem.unsafe_ptr()
        var mutex_addr = Int(p_shared.unsafe_offset(0))

        _ = external_call["pthread_mutex_lock", Int32](mutex_addr)
        var head = p_shared[unsafe_offset=27]
        var tail = p_shared[unsafe_offset=28]
        _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)

        return head - tail

    def get_total_completed(self) -> Int:
        """Mendapatkan total kumulatif job yang telah selesai diproses."""
        var p_shared = self.shared_mem.unsafe_ptr()
        var mutex_addr = Int(p_shared.unsafe_offset(0))

        _ = external_call["pthread_mutex_lock", Int32](mutex_addr)
        var count = p_shared[unsafe_offset=30]
        _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)

        return count

    def shutdown(mut self):
        """Menghentikan thread pekerja I/O dedikasi dan melepaskan resource POSIX.
        """
        if self.is_active:
            var p_shared = self.shared_mem.unsafe_ptr()
            var mutex_addr = Int(p_shared.unsafe_offset(0))
            var cond_work_addr = Int(p_shared.unsafe_offset(8))
            var cond_done_addr = Int(p_shared.unsafe_offset(16))

            _ = external_call["pthread_mutex_lock", Int32](mutex_addr)
            p_shared[unsafe_offset=24] = 1  # exit_flag = 1
            _ = external_call["pthread_cond_signal", Int32](cond_work_addr)
            _ = external_call["pthread_mutex_unlock", Int32](mutex_addr)

            _ = external_call["pthread_join", Int32](self.thread_ids[0], Int(0))

            _ = external_call["pthread_mutex_destroy", Int32](mutex_addr)
            _ = external_call["pthread_cond_destroy", Int32](cond_work_addr)
            _ = external_call["pthread_cond_destroy", Int32](cond_done_addr)

            self.is_active = False
