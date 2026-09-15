# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""File I/O helpers untuk membaca shard dan index safetensors."""

from format.types import HEADER_MAX, STError, error_json
from std.collections import List
from std.ffi import external_call
from std.os import SEEK_END, SEEK_SET


def c_realpath(path: String) -> String:
    # Dipindah dari cli.sys_utils: lapisan core/format butuh canonicalize
    # tanpa inversi layer (cli -> core -> format satu arah).
    var p = path.as_bytes()
    var p_z = List[UInt8]()
    for i in range(len(p)):
        p_z.append(p[i])
    p_z.append(0)
    var buf = List[UInt8]()
    for _ in range(4096):
        buf.append(0)
    var res = external_call["realpath", Int](p_z.unsafe_ptr(), buf.unsafe_ptr())
    if res == 0:
        return ""
    var n = 0
    while n < 4096 and buf[n] != 0:
        n += 1
    var out = List[UInt8]()
    for i in range(n):
        out.append(buf[i])
    return String(from_utf8_lossy=Span(out))


def path_is_within(root_canon: String, path_canon: String) -> Bool:
    # SATU-SATUNYA implementasi containment path komponen: sama persis atau
    # prefix root + "/". Menolak sibling prefix (/a/workevil vs /a/work).
    if path_canon == root_canon:
        return True
    if root_canon == "/":
        return path_canon.startswith("/")
    return path_canon.startswith(String(root_canon, "/"))


def resolve_within_root(model_root: String, rel_path: String) raises -> String:
    # Kontrak: join lexical model_root/rel_path TIDAK berarti di dalam root
    # (traversal ../, symlink, absolut). Canonicalize keduanya; hasil di
    # luar root atau tak-resolve → error. Absolut rel_path aman-by-construction
    # (join menempelkannya di bawah root).
    var root_canon = c_realpath(model_root if model_root != "" else ".")
    if root_canon == "":
        raise Error(
            error_json(
                "FILE_NOT_FOUND",
                String("model directory not found: ", model_root),
                model_root,
                "",
            )
        )
    var joined = String(root_canon, "/", rel_path)
    var canon = c_realpath(joined)
    if canon == "":
        raise Error(
            error_json(
                "FILE_NOT_FOUND",
                String("shard not found: ", joined),
                joined,
                "",
            )
        )
    if not path_is_within(root_canon, canon):
        raise Error(
            error_json(
                "WEIGHT_LOAD_FAILED",
                String("shard escapes model root: ", rel_path),
                canon,
                "",
            )
        )
    return canon


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
        raise Error(error_json("INVALID_HEADER", "index too large", path, ""))
    var out = f.read_bytes(n)
    f.close()
    if len(out) < n:
        raise Error(error_json("INVALID_HEADER", "index truncated", path, ""))
    return out^
