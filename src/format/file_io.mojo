# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""File I/O helpers untuk membaca shard dan index safetensors."""

from format.types import HEADER_MAX, STError
from std.collections import List
from std.os import SEEK_END, SEEK_SET


def _open_shard(path: String) raises -> FileHandle:
    try:
        return open(path, "r")
    except:
        raise Error(
            String(
                STError(
                    "FILE_NOT_FOUND",
                    String("tidak bisa open: ", path),
                    path,
                    "",
                )
            )
        )


def read_small_file(path: String) raises -> List[UInt8]:
    """Membaca file kecil (misal JSON atau header) seluruhnya ke memory."""
    var f = open(path, "r")
    var n = Int(f.seek(0, SEEK_END))
    _ = f.seek(0, SEEK_SET)
    if n > HEADER_MAX:
        f.close()
        raise Error(
            String(
                (
                    '{"error_type":"INVALID_HEADER","detail":"index too'
                    ' large","shard":"'
                ),
                path,
                '","tensor_name":""}',
            )
        )
    var out = f.read_bytes(n)
    f.close()
    if len(out) < n:
        raise Error(
            String(
                (
                    '{"error_type":"INVALID_HEADER","detail":"index'
                    ' truncated","shard":"'
                ),
                path,
                '","tensor_name":""}',
            )
        )
    return out^
