# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Deteksi format berkas model berbasis magic number / header (bukan ekstensi).

Sesuai kontrak M9 (docs/milestones/M9-port.md § Weight loading):
- GGUF: magic 0x46554747 ('GGUF'), version uint32.
- Safetensors: uint64 header_len <= 100 MB diikuti byte '{' (0x7B).
- Mismatch / unknown magic memicu penolakan fail-closed (exit 5 / IO_ERROR).
"""

from format.file_io import _open_shard
from format.types import HEADER_MAX
from std.os import SEEK_END, SEEK_SET

comptime FORMAT_UNKNOWN: Int = 0
comptime FORMAT_SAFETENSORS: Int = 1
comptime FORMAT_GGUF: Int = 2

# Magic 'GGUF' dalam ASCII bytes
comptime GGUF_MAGIC_0: UInt8 = 0x47  # 'G'
comptime GGUF_MAGIC_1: UInt8 = 0x47  # 'G'
comptime GGUF_MAGIC_2: UInt8 = 0x55  # 'U'
comptime GGUF_MAGIC_3: UInt8 = 0x46  # 'F'


def format_to_string(fmt: Int) -> String:
    if fmt == FORMAT_SAFETENSORS:
        return "safetensors"
    elif fmt == FORMAT_GGUF:
        return "gguf"
    return "unknown"


def detect_file_format(path: String) raises -> Int:
    """Mendeteksi format berkas secara deterministik dari 8+ byte pertama."""
    var f = _open_shard(path)
    var filesize = Int(f.seek(0, SEEK_END))
    _ = f.seek(0, SEEK_SET)

    if filesize < 8:
        f.close()
        raise Error(
            "FORMAT_ERROR: file size "
            + String(filesize)
            + " < 8 bytes (too short): "
            + path
        )

    var prefix = f.read_bytes(8)

    # 1. Cek Magic GGUF: 'G' 'G' 'U' 'F'
    if (
        prefix[0] == GGUF_MAGIC_0
        and prefix[1] == GGUF_MAGIC_1
        and prefix[2] == GGUF_MAGIC_2
        and prefix[3] == GGUF_MAGIC_3
    ):
        var ver = 0
        var mult = 1
        for i in range(4, 8):
            ver += Int(prefix[i]) * mult
            mult *= 256
        f.close()
        if ver != 2 and ver != 3:
            raise Error(
                "FORMAT_ERROR: unsupported GGUF version "
                + String(ver)
                + " in "
                + path
            )
        return FORMAT_GGUF

    # 2. Cek Safetensors: u64 header_len <= 100 MB dan byte ke-8 adalah '{' (123)
    var header_len = 0
    var mult_st = 1
    for i in range(8):
        header_len += Int(prefix[i]) * mult_st
        mult_st *= 256

    if (
        header_len > 0
        and header_len <= HEADER_MAX
        and (8 + header_len) <= filesize
    ):
        # Baca 1 byte berikutnya untuk memeriksa karakter awal JSON '{'
        var next_byte = f.read_bytes(1)
        f.close()
        if len(next_byte) == 1 and next_byte[0] == 123:  # '{'
            return FORMAT_SAFETENSORS

    f.close()
    raise Error(
        "FORMAT_ERROR: unrecognized format or corrupted magic header in " + path
    )
