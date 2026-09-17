# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""O_DIRECT Reader, Dynamic Alignment Discovery, Triple Alignment, dan Bounded Staging Buffer (M7)."""

from cli.m7_errors import m7_error_json
from format.types import json_escape
from std.collections import List
from std.ffi import external_call
from std.math import max

# Konstanta POSIX Linux x86_64
comptime O_RDONLY = 0
comptime O_DIRECT = 16384  # 0x4000 di Linux x86_64

# Errno Linux
comptime EINTR = 4
comptime EIO = 5
comptime EAGAIN = 11
comptime EACCES = 13
comptime EINVAL = 22
comptime ENOSPC = 28


@fieldwise_init
struct ReadToken(Copyable, Movable):
    """Token identitas operasi I/O asinkron dua-fase."""

    var token_id: Int
    var logical_offset: Int
    var logical_length: Int
    var phys_start: Int
    var phys_len: Int
    var staging_addr: Int
    var is_completed: Bool


struct ODirectReader:
    """Reader O_DIRECT dengan alignment discovery, concurrency contract, dan bounded buffer.
    """

    var path: String
    var fd: Int
    var dio_alignment: Int
    var is_odirect: Bool
    var queue_depth: Int
    var block_size: Int
    var outstanding_count: Int
    var max_outstanding_observed: Int
    var total_bytes_read: Int
    var read_count: Int
    var next_token_id: Int

    def __init__(
        out self,
        path: String,
        fd: Int,
        dio_alignment: Int,
        is_odirect: Bool,
        queue_depth: Int = 16,
        block_size: Int = 4096,
    ):
        self.path = path
        self.fd = fd
        self.dio_alignment = dio_alignment
        self.is_odirect = is_odirect
        self.queue_depth = queue_depth
        self.block_size = block_size
        self.outstanding_count = 0
        self.max_outstanding_observed = 0
        self.total_bytes_read = 0
        self.read_count = 0
        self.next_token_id = 1

    @staticmethod
    def discover(
        path: String,
        requested_block_size: Int = 4096,
        queue_depth: Int = 16,
        force_buffered: Bool = False,
    ) raises -> ODirectReader:
        """Menemukan dio_alignment via read-probe nyata pada kandidat [512, 4096].
        """
        if queue_depth <= 0:
            raise Error(
                m7_error_json(
                    "M7_ERR_ODIRECT_ALIGNMENT",
                    "io_direct",
                    "queue-depth must be positive: " + String(queue_depth),
                )
            )

        var path_b = path.as_bytes()
        var path_z = List[UInt8]()
        for i in range(len(path_b)):
            path_z.append(path_b[i])
        path_z.append(0)

        # Jika force_buffered diminta oleh konfigurasi
        if force_buffered:
            var fd_buf = Int(
                external_call["openat", Int32](
                    Int32(-100), path_z.unsafe_ptr(), Int32(O_RDONLY), Int32(0)
                )
            )
            if fd_buf < 0:
                raise Error(
                    m7_error_json(
                        "M7_ERR_ODIRECT_EIO",
                        "io_direct",
                        "cannot open path for buffered read: " + path,
                    )
                )
            return ODirectReader(
                path=path,
                fd=fd_buf,
                dio_alignment=512,
                is_odirect=False,
                queue_depth=queue_depth,
                block_size=requested_block_size,
            )

        # Discovery Kandidat [512, 4096]
        var candidates = List[Int]()
        candidates.append(512)
        candidates.append(4096)

        var chosen_alignment = -1
        var chosen_fd = -1

        for c_idx in range(len(candidates)):
            var cand = candidates[c_idx]
            var fd_probe = Int(
                external_call["openat", Int32](
                    Int32(-100), path_z.unsafe_ptr(), Int32(O_DIRECT), Int32(0)
                )
            )
            if fd_probe < 0:
                continue  # Platform tidak mendukung O_DIRECT atau flag ini

            # Lakukan probe read nyata
            var p_addr = external_call["aligned_alloc", Int](cand, cand)
            if p_addr == 0:
                _ = external_call["close", Int32](Int32(fd_probe))
                continue

            var n_read = external_call["pread", Int](
                Int32(fd_probe), p_addr, cand, 0
            )
            external_call["free", NoneType](p_addr)

            if n_read >= 0:
                # Sukses membaca kandidat ini
                chosen_alignment = cand
                chosen_fd = fd_probe
                break
            else:
                _ = external_call["close", Int32](Int32(fd_probe))
                continue

        # Bila seluruh kandidat gagal probe O_DIRECT, fallback ke buffered (hanya pada fase probe!)
        if chosen_alignment < 0:
            var fd_buf = Int(
                external_call["openat", Int32](
                    Int32(-100), path_z.unsafe_ptr(), Int32(O_RDONLY), Int32(0)
                )
            )

            if fd_buf < 0:
                raise Error(
                    m7_error_json(
                        "M7_ERR_ODIRECT_EIO",
                        "io_direct",
                        "all O_DIRECT probes failed and buffered open failed: "
                        + path,
                    )
                )
            return ODirectReader(
                path=path,
                fd=fd_buf,
                dio_alignment=4096,
                is_odirect=False,
                queue_depth=queue_depth,
                block_size=requested_block_size,
            )

        # Validasi relasi block-size vs dio_alignment
        if requested_block_size < chosen_alignment or (
            requested_block_size % chosen_alignment != 0
        ):
            _ = external_call["close", Int32](Int32(chosen_fd))
            raise Error(
                m7_error_json(
                    "M7_ERR_ODIRECT_ALIGNMENT",
                    "io_direct",
                    "block-size "
                    + String(requested_block_size)
                    + " must be >= dio_alignment "
                    + String(chosen_alignment)
                    + " and a multiple of it",
                )
            )

        return ODirectReader(
            path=path,
            fd=chosen_fd,
            dio_alignment=chosen_alignment,
            is_odirect=True,
            queue_depth=queue_depth,
            block_size=requested_block_size,
        )

    def close(mut self):
        """Menutup file descriptor."""
        if self.fd >= 0:
            _ = external_call["close", Int32](Int32(self.fd))
            self.fd = -1

    def pread_o_direct_span(
        self,
        p_addr: Int,
        phys_start: Int,
        phys_len: Int,
        max_span_retries: Int = 3,
    ) raises -> Int:
        """Membaca span fisik yang dijamin selaras (phys_start % A == 0, phys_len % A == 0).
        """
        var A = self.dio_alignment
        if not self.is_odirect:
            # Jalur buffered read langsung
            var total_read = 0
            while total_read < phys_len:
                var n = external_call["pread", Int](
                    Int32(self.fd),
                    p_addr + total_read,
                    phys_len - total_read,
                    phys_start + total_read,
                )
                if n == 0:
                    break
                if n < 0:
                    raise Error(
                        m7_error_json(
                            "M7_ERR_ODIRECT_EIO",
                            "io_direct",
                            "buffered pread failed with error code: "
                            + String(n),
                        )
                    )
                total_read += n
            return total_read

        # Jalur O_DIRECT dengan perlindungan short-read span
        var total_read = 0
        var span_retries = 0

        while total_read < phys_len:
            var n = external_call["pread", Int](
                Int32(self.fd),
                p_addr + total_read,
                phys_len - total_read,
                phys_start + total_read,
            )
            if n == 0:
                break  # EOF terdeteksi
            if n < 0:
                # Kegagalan I/O pada physical span selaras (EINVAL / EIO)
                raise Error(
                    m7_error_json(
                        "M7_ERR_ODIRECT_EIO",
                        "io_direct",
                        "O_DIRECT pread failed with error code: " + String(n),
                    )
                )

            if n % A != 0:
                # UNEXPECTED short read: sisa tak representable via O_DIRECT.
                # Dilarang menerbitkan read lanjutan misaligned; ulangi SPAN penuh.
                if span_retries >= max_span_retries:
                    raise Error(
                        m7_error_json(
                            "M7_ERR_ODIRECT_SHORT_READ",
                            "io_direct",
                            "unrecoverable unaligned short read after "
                            + String(max_span_retries)
                            + " retries (got "
                            + String(n)
                            + " bytes, alignment="
                            + String(A)
                            + ")",
                        )
                    )
                total_read = 0
                span_retries += 1
                continue

            total_read += n

        return total_read

    def submit_read(
        mut self, logical_offset: Int, logical_length: Int
    ) raises -> ReadToken:
        """Tahap submit API dua-fase (non-blocking / enqueue read request)."""
        if self.outstanding_count >= self.queue_depth:
            raise Error(
                m7_error_json(
                    "M7_ERR_ODIRECT_ALIGNMENT",
                    "io_direct",
                    "queue depth exceeded: outstanding="
                    + String(self.outstanding_count)
                    + " >= max="
                    + String(self.queue_depth),
                )
            )

        var A = self.dio_alignment
        var phys_start = (logical_offset // A) * A
        var phys_end = ((logical_offset + logical_length + A - 1) // A) * A
        var phys_len = phys_end - phys_start
        var alloc_size = ((phys_len + A - 1) // A) * A

        var staging_addr = external_call["aligned_alloc", Int](A, alloc_size)
        if staging_addr == 0:
            raise Error(
                m7_error_json(
                    "M7_ERR_LRU_ALLOC",
                    "io_direct",
                    "failed to allocate aligned staging buffer: "
                    + String(alloc_size)
                    + " bytes",
                )
            )

        self.outstanding_count += 1
        if self.outstanding_count > self.max_outstanding_observed:
            self.max_outstanding_observed = self.outstanding_count

        var tok = ReadToken(
            token_id=self.next_token_id,
            logical_offset=logical_offset,
            logical_length=logical_length,
            phys_start=phys_start,
            phys_len=phys_len,
            staging_addr=staging_addr,
            is_completed=False,
        )
        self.next_token_id += 1
        return tok^

    def complete_read(mut self, var tok: ReadToken) raises -> List[UInt8]:
        """Tahap completion API dua-fase (reap completed buffer dan ekstraksi slice logis).
        """
        if tok.staging_addr == 0:
            raise Error(
                m7_error_json(
                    "M7_ERR_ODIRECT_EIO",
                    "io_direct",
                    "invalid staging address in completion token",
                )
            )

        var bytes_read: Int
        try:
            bytes_read = self.pread_o_direct_span(
                tok.staging_addr, tok.phys_start, tok.phys_len
            )
        except e:
            external_call["free", NoneType](tok.staging_addr)
            tok.staging_addr = 0
            self.outstanding_count -= 1
            raise e

        var slice_offset = tok.logical_offset - tok.phys_start
        var required_bytes = slice_offset + tok.logical_length

        if bytes_read < required_bytes:
            external_call["free", NoneType](tok.staging_addr)
            tok.staging_addr = 0
            self.outstanding_count -= 1
            raise Error(
                m7_error_json(
                    "M7_ERR_FORMAT_ALIGNMENT",
                    "io_direct",
                    "physical span read truncated: got "
                    + String(bytes_read)
                    + ", required "
                    + String(required_bytes),
                )
            )

        var out = List[UInt8]()
        out.reserve(tok.logical_length)
        for _ in range(tok.logical_length):
            out.append(0)

        _ = external_call["memcpy", Int](
            out.unsafe_ptr(),
            tok.staging_addr + slice_offset,
            tok.logical_length,
        )

        external_call["free", NoneType](tok.staging_addr)
        tok.staging_addr = 0
        tok.is_completed = True

        self.outstanding_count -= 1
        self.total_bytes_read += tok.logical_length
        self.read_count += 1

        return out^

    def read_logical_payload(
        mut self, logical_offset: Int, logical_length: Int
    ) raises -> List[UInt8]:
        """Membaca payload logis (unaligned M6 v1) secara langsung melalui staging span selaras.
        """
        var tok = self.submit_read(logical_offset, logical_length)
        return self.complete_read(tok^)
