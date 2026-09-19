# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Klien tokenizer REAL via pustaka BPE HF (M12, fix #3).

Kontrak non-negotiable (fix #3):
- ID token SELALU berasal dari pustaka BPE HF (`tools/tokenizer_cli.py`)
  atas tokenizer.json yang sama dengan model. TIDAK ADA hash h*31,
  TIDAK ADA fallback sintetis, TIDAK ADA tebakan diam-diam.
- tokenizer.json hilang / helper gagal / output tak-terparse ->
  fail-closed TOKENIZER_* (raise, bukan fallback).

NOTA DESAIN: transport adalah one-shot `system()` per operasi (pola yang
sudah terbukti di codebase: oracle fallback, tokenizer satu-tembak lama).
Percobaan fork-pipe-exec persistent-daemon dari dalam Mojo DIBATALKAN:
di runtime Mojo 1.0 ini anak hasil fork teramati exit(127) tanpa pernah
menerbitkan execve (terbukti via strace), sementara replika terisolasi
lolos — interaksi tak-terlihat dengan runtime multithread/JIT. Satu-tembak
lebih lambat per token tetapi BENAR dan dapat diukur (liDW42ablasan
0,3 tok/s justru bahan eval). Bila suatu hari tersedia primitif spawn
yang andal, ganti badan fungsi ini tanpa mengubah kontrak caller.
"""

from format.file_io import read_small_file
from format.scanner import Scanner
from std.collections import List
from std.ffi import external_call
from std.time import perf_counter_ns


def _file_size_or_neg(path: String) -> Int:
    """Ukuran berkas atau -1 bila tak-terbuka (tanpa modul cli)."""
    try:
        var f = open(path, "r")
        var sz = Int(f.seek(0, 2))
        f.close()
        return sz
    except:
        return -1


def _unlink_quiet(path: String):
    """Hapus berkas sementara; abaikan kegagalan."""
    var b = path.as_bytes()
    var z = List[UInt8]()
    for i in range(len(b)):
        z.append(b[i])
    z.append(0)
    _ = external_call["unlink", Int32](z.unsafe_ptr())


def _path_is_executable(path: String) -> Bool:
    """Uji X_OK via access(2) tanpa modul lain (hindari siklus import)."""
    var b = path.as_bytes()
    var z = List[UInt8]()
    for i in range(len(b)):
        z.append(b[i])
    z.append(0)
    return external_call["access", Int32](z.unsafe_ptr(), Int32(1)) == 0


def _resolve_python_exe() -> String:
    """Preferensi interpreter repo: .venv dulu (ada lib tokenizers)."""
    if _path_is_executable(".venv/bin/python"):
        return ".venv/bin/python"
    if _path_is_executable(".venv/bin/python3"):
        return ".venv/bin/python3"
    if _path_is_executable("/usr/bin/python3"):
        return "/usr/bin/python3"
    return "python3"


def _shell_quote_single(s: String) raises -> String:
    """Kutip-tunggal untuk PATH: tolak karakter shell-aktif fail-closed."""
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c == 39 or c == 34 or c == 36 or c == 96 or c == 92 or c < 32:
            raise Error(
                String(
                    (
                        '{"error_type":"TOKENIZER_DAEMON","detail":"model dir'
                        " contains shell-active characters: "
                    ),
                    s,
                    '"}',
                )
            )
    return String("'", s, "'")


def _shell_squote_text(raw: String) raises -> String:
    """Kutip-tunggal aman untuk TEKS ARBITRER (input pengguna).

    Setiap `'` menjadi `'\''` pada level byte (tutup-kutip + escape +
    buka-kutip); dalam kutip-tunggal shell tidak ada karakter aktif lain,
    dan byte non-quote disalin verbatim (UTF-8 multi-byte utuh).
    Berbeda dari path (yang DITOLAK bila mengandung karakter aktif),
    teks tidak boleh ditolak — harus lolos utuh apa pun isinya.
    """
    var out = List[UInt8]()
    out.append(39)
    var b = raw.as_bytes()
    for i in range(len(b)):
        if Int(b[i]) == 39:
            out.append(39)
            out.append(92)
            out.append(39)
            out.append(39)
        else:
            out.append(b[i])
    out.append(39)
    return String(from_utf8_lossy=Span(out))


def _run_helper(cmd: String, out_path: String) raises:
    """Menjalankan helper tokenizer satu-tembak; gagal -> raise keras."""
    var cmd_b = cmd.as_bytes()
    var cmd_z = List[UInt8]()
    for i in range(len(cmd_b)):
        cmd_z.append(cmd_b[i])
    cmd_z.append(0)
    var ret = external_call["system", Int32](cmd_z.unsafe_ptr())
    if ret != 0:
        _unlink_quiet(out_path)
        raise Error(
            String(
                (
                    '{"error_type":"TOKENIZER_HELPER","detail":"tokenizer'
                    " helper failed with exit "
                ),
                Int(ret),
                " (record:",
                out_path,
                ')"}',
            )
        )
    if _file_size_or_neg(out_path) <= 0:
        raise Error(
            String(
                (
                    '{"error_type":"TOKENIZER_HELPER","detail":"tokenizer'
                    " helper produced no output (record:"
                ),
                out_path,
                ')"}',
            )
        )


def _parse_ids_json(raw: List[UInt8], source: String) raises -> List[Int]:
    """Parsing ketat array JSON [id, ...] (tolak float/string/negatif)."""
    var buf = raw.copy()
    var sc = Scanner(buf^, source, True)
    sc.skip_ws()
    var out = sc.parse_int_array()
    sc.skip_ws()
    if not sc.eof():
        raise Error(
            '{"error_type":"TOKENIZER_PROTOCOL","detail":"trailing bytes in'
            ' ids output"}'
        )
    return out^


def encode_via_hf(text: String, model_dir: String) raises -> List[Int]:
    """Encode REAL: teks -> IDs BPE via helper HF (fail-closed)."""
    if _file_size_or_neg(String(model_dir, "/tokenizer.json")) <= 0:
        raise Error(
            String(
                (
                    '{"error_type":"TOKENIZER_NOT_FOUND","detail":"no'
                    " tokenizer.json in model dir (hash fallback dilarang per"
                    " fix #3): "
                ),
                model_dir,
                '"}',
            )
        )
    var stamp = String(perf_counter_ns())
    var out_path = String("/tmp/dismoen_tok_enc_", stamp, ".json")
    var q_dir = _shell_quote_single(model_dir)
    var q_text = _shell_squote_text(text)
    var cmd = String(
        _resolve_python_exe(),
        " tools/tokenizer_cli.py --model-dir ",
        q_dir,
        " --encode ",
        q_text,
        ' --output "',
        out_path,
        '" 2>/dev/null',
    )
    _run_helper(cmd, out_path)
    var raw = read_small_file(out_path)
    _unlink_quiet(out_path)
    return _parse_ids_json(raw, out_path)


def decode_via_hf(ids: List[Int], model_dir: String) raises -> String:
    """Decode REAL: IDs -> teks via helper HF (fail-closed)."""
    if _file_size_or_neg(String(model_dir, "/tokenizer.json")) <= 0:
        raise Error(
            String(
                (
                    '{"error_type":"TOKENIZER_NOT_FOUND","detail":"no'
                    " tokenizer.json in model dir (hash fallback dilarang per"
                    " fix #3): "
                ),
                model_dir,
                '"}',
            )
        )
    var items = String("")
    for i in range(len(ids)):
        if i > 0:
            items += ","
        items += String(ids[i])
    var stamp = String(perf_counter_ns())
    var out_path = String("/tmp/dismoen_tok_dec_", stamp, ".txt")
    var q_dir = _shell_quote_single(model_dir)
    var cmd = String(
        _resolve_python_exe(),
        " tools/tokenizer_cli.py --model-dir ",
        q_dir,
        ' --decode "[',
        items,
        ']" > "',
        out_path,
        '" 2>/dev/null',
    )
    _run_helper(cmd, out_path)
    var raw = read_small_file(out_path)
    _unlink_quiet(out_path)
    return String(from_utf8_lossy=Span(raw))
