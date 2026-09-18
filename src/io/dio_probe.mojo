# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""O_DIRECT Alignment Prober & Dynamic Constraints (§3.1 & P1-3, M11-W2a).

Mengimplementasikan:
1. Probe alignment O_DIRECT (A_mem, A_off, A_len) via syscall Linux `statx` dengan `STATX_DIOALIGN` (Linux >= 6.1).
2. Fallback constraint terdokumentasi (expected profile 4096B) bila statx tidak tersedia atau filesystem tidak mendukung.
3. Validasi ketiga sisi alignment secara independen: alamat buffer, offset berkas, dan panjang request.
4. Round-up otomatis chunk size dan buffer capacity bila alignment hasil probe bukan 4096B.
"""

from std.collections import List
from std.ffi import external_call
from std.math import max
from std.memory import Pointer

# Konstanta Linux statx
comptime STATX_DIOALIGN = 0x00002000
comptime AT_FDCWD = -100

# Default target profile (4096B expected profile, bukan syarat keras)
comptime DEFAULT_TARGET_ALIGN = 4096
comptime BASE_CHUNK_SIZE = 3493888  # 853 * 4096 B (~3.33 MiB per Q3_K expert)
comptime BASE_BUFFER_CAPACITY = 67108864  # 64 MiB


@fieldwise_init
struct DioAlignment(Copyable, Movable):
    """Hasil probe alignment O_DIRECT (A_mem, A_off, A_len) per §3.1 & P1-3."""

    var mem_align: Int  # A_mem: memory buffer alignment
    var offset_align: Int  # A_off: file offset alignment
    var length_align: Int  # A_len: transfer length alignment
    var is_probed_statx: Bool
    var source: String  # "statx_dioalign" atau "fallback_documented"

    def required_alignment(self) -> Int:
        """Alignment maksimum yang mengikat ketiga sisi."""
        return max(self.mem_align, max(self.offset_align, self.length_align))


def round_up_dio(val: Int, align: Int) -> Int:
    """Membulatkan val ke atas ke kelipatan terdekat dari align."""
    if align <= 1 or val <= 0:
        return val
    if val % align == 0:
        return val
    return ((val + align - 1) // align) * align


def calculate_chunk_size(required_align: Int) -> Int:
    """Menghitung ukuran chunk per-expert ter-align.

    Jika required_align == 4096, mengembalikan profil target 3,493,888 B (853 * 4096).
    Jika required_align berbeda, melakukan round-up ke kelipatan required_align.
    """
    return round_up_dio(BASE_CHUNK_SIZE, required_align)


def calculate_buffer_capacity(required_align: Int) -> Int:
    """Menghitung kapasitas 1 staging buffer ter-align (minimal 64 MiB)."""
    return round_up_dio(BASE_BUFFER_CAPACITY, required_align)


def probe_dio_alignment(path: String = ".") -> DioAlignment:
    """Mem-probe alignment O_DIRECT (A_mem, A_off, A_len) untuk path target.

    Langkah probe (P1-3):
    1. Mencoba syscall statx dengan STATX_DIOALIGN (Linux >= 6.1).
    2. Bila didukung dan flag STATX_DIOALIGN diset di mask hasil, membaca
       stx_dio_mem_align dan stx_dio_offset_align.
    3. Bila gagal atau flag tidak diset, fallback ke expected profile (4096B)
       terdokumentasi.
    """
    var path_b = path.as_bytes()
    var path_z = List[UInt8]()
    for i in range(len(path_b)):
        path_z.append(path_b[i])
    path_z.append(0)

    # Buffer struct statx (256 bytes)
    var statxbuf = List[UInt8]()
    statxbuf.resize(256, UInt8(0))
    var p_buf = statxbuf.unsafe_ptr()

    var res = external_call["statx", Int32](
        Int32(AT_FDCWD),
        path_z.unsafe_ptr(),
        Int32(0),
        UInt32(STATX_DIOALIGN),
        p_buf,
    )

    if res == 0:
        var p_u32 = Pointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(p_buf)
        )
        var mask = p_u32[unsafe_offset=0]
        if (mask & UInt32(STATX_DIOALIGN)) != 0:
            var m_align = Int(p_u32[unsafe_offset=38])  # stx_dio_mem_align
            var off_align = Int(p_u32[unsafe_offset=39])  # stx_dio_offset_align

            # Nilai validasi sanity
            if m_align > 0 and off_align > 0:
                return DioAlignment(
                    mem_align=m_align,
                    offset_align=off_align,
                    length_align=off_align,
                    is_probed_statx=True,
                    source="statx_dioalign",
                )

    # Fallback constraint terdokumentasi (P1-3 butir 2-3)
    return DioAlignment(
        mem_align=DEFAULT_TARGET_ALIGN,
        offset_align=DEFAULT_TARGET_ALIGN,
        length_align=DEFAULT_TARGET_ALIGN,
        is_probed_statx=False,
        source="fallback_documented",
    )


def validate_dio_constraints(
    addr: Int, offset: Int, length: Int, align: DioAlignment
) raises:
    """Memvalidasi bahwa ketiga sisi (buffer, offset, length) memenuhi alignment O_DIRECT masing-masing.

    Pelanggaran salah satu sisi menghasilkan error konfigurasi (EINVAL).
    """
    if align.mem_align > 0 and addr % align.mem_align != 0:
        raise Error(
            "DIO_ALIGN_ERROR: buffer address "
            + String(addr)
            + " not aligned to A_mem="
            + String(align.mem_align)
        )
    if align.offset_align > 0 and offset % align.offset_align != 0:
        raise Error(
            "DIO_ALIGN_ERROR: file offset "
            + String(offset)
            + " not aligned to A_off="
            + String(align.offset_align)
        )
    if align.length_align > 0 and length % align.length_align != 0:
        raise Error(
            "DIO_ALIGN_ERROR: request length "
            + String(length)
            + " not aligned to A_len="
            + String(align.length_align)
        )


def is_triple_aligned(
    addr: Int, offset: Int, length: Int, align: DioAlignment
) -> Bool:
    """Memeriksa apakah ketiga sisi memenuhi syarat O_DIRECT tanpa melempar exception.
    """
    if align.mem_align > 0 and addr % align.mem_align != 0:
        return False
    if align.offset_align > 0 and offset % align.offset_align != 0:
        return False
    if align.length_align > 0 and length % align.length_align != 0:
        return False
    return True
