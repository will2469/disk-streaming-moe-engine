# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Harness pengujian eksekusi GDN chunked scan (M8-W2).

Membaca tokens JSON dan weights safetensors, menjalankan GDN chunked scan,
dan menuliskan GDNS v1 framed binary output untuk verifikasi compare gate G-M8-1.
"""

from core.config import LoadMemoryTelemetry
from core.tensor_loader import load_tensor_f32_chunked
from format.file_io import read_small_file
from format.gdns import write_gdns_v1
from format.reader import read_header
from format.scanner import Scanner
from format.types import TensorMeta
from layers.gdn import (
    GDNConfig,
    GDNState,
    chunked_gdn_scan,
    project_tokens_to_kv_beta,
)
from std.collections import Dict, List
from std.sys.arg import argv
from std.sys.terminate import exit


def parse_tokens(path: String) raises -> List[Int]:
    """Membaca array token IDs dari format tokens JSON."""
    var raw = read_small_file(path)
    var sc = Scanner(raw^, path)
    var tokens = List[Int]()

    # Cari kata kunci "tokens"
    var found_tokens = False
    while not sc.eof():
        sc.skip_ws()
        if sc.peek() == 34:  # '"'
            var s = sc.parse_string()
            if s == "tokens":
                found_tokens = True
                break
        else:
            sc.pos += 1

    if not found_tokens:
        raise Error(
            '{"error_type":"TOKEN_INVALID","detail":"tokens array not found in'
            ' JSON"}'
        )

    sc.skip_ws()
    sc.expect(58)  # ':'
    sc.skip_ws()
    sc.expect(91)  # '['

    while not sc.eof():
        sc.skip_ws()
        if sc.peek() == 93:  # ']'
            sc.pos += 1
            break
        var tok_id = sc.parse_uint()
        tokens.append(tok_id)
        sc.skip_ws()
        if sc.peek() == 44:  # ','
            sc.pos += 1
        elif sc.peek() == 93:  # ']'
            sc.pos += 1
            break
        else:
            raise Error(
                '{"error_type":"TOKEN_INVALID","detail":"expected comma or'
                ' closing bracket in tokens"}'
            )

    return tokens^


def load_tensor_by_key(
    weights_path: String,
    data_base: Int,
    entries: List[TensorMeta],
    key: String,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    """Mencari dan memuat tensor berdasarkan nama dari safetensors."""
    for i in range(len(entries)):
        if entries[i].name == key:
            return load_tensor_f32_chunked(
                weights_path, data_base, entries[i], telemetry
            )
    raise Error(
        '{"error_type":"WEIGHT_LOAD_FAILED","detail":"tensor not found: '
        + key
        + '"}'
    )


def main() raises:
    var args = argv()
    var tokens_path = String("")
    var weights_path = String("")
    var output_path = String("")
    var layers = 2
    var dk = 32
    var dv = 32
    var chunk_size = 8

    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--tokens" and i + 1 < len(args):
            tokens_path = String(args[i + 1])
            i += 2
        elif arg == "--weights" and i + 1 < len(args):
            weights_path = String(args[i + 1])
            i += 2
        elif arg == "--output" and i + 1 < len(args):
            output_path = String(args[i + 1])
            i += 2
        elif arg == "--layers" and i + 1 < len(args):
            layers = Int(String(args[i + 1]))
            i += 2
        elif arg == "--dk" and i + 1 < len(args):
            dk = Int(String(args[i + 1]))
            i += 2
        elif arg == "--dv" and i + 1 < len(args):
            dv = Int(String(args[i + 1]))
            i += 2
        elif arg == "--chunk-size" and i + 1 < len(args):
            chunk_size = Int(String(args[i + 1]))
            i += 2
        else:
            i += 1

    if tokens_path == "" or weights_path == "" or output_path == "":
        print(
            "Usage: run_gdn_scan_test --tokens <path> --weights <path> --output"
            " <path> [--layers N] [--dk N] [--dv N] [--chunk-size N]"
        )
        exit(1)

    var cfg = GDNConfig(layers, dk, dv, chunk_size)
    cfg.validate()

    # 1. Baca tokens
    var tokens = parse_tokens(tokens_path)
    var seq_len = len(tokens)

    # 2. Baca weights header
    var header = read_header(weights_path)
    var telemetry = LoadMemoryTelemetry()

    # Muat embed_tokens.weight: shape [vocab, dk]
    var embed_table = load_tensor_by_key(
        weights_path,
        header.data_base,
        header.entries,
        "embed_tokens.weight",
        telemetry,
    )

    # Susun matriks input activations X: [seq_len, dk]
    var x_act = List[Float32]()
    x_act.resize(seq_len * dk, Float32(0.0))
    var p_x = x_act.unsafe_ptr()
    var p_embed = embed_table.unsafe_ptr()

    for t in range(seq_len):
        var tid = tokens[t]
        var emb_off = tid * dk
        var x_off = t * dk
        for c in range(dk):
            p_x[unsafe_offset=x_off + c] = p_embed[unsafe_offset=emb_off + c]

    # Inisialisasi GDNState
    var state = GDNState(layers, dv, dk)
    state.zero()

    # 3. Jalankan forward GDN chunked scan per layer
    for lyr in range(layers):
        var wk_name = String("layers.", String(lyr), ".k_proj.weight")
        var wv_name = String("layers.", String(lyr), ".v_proj.weight")
        var wbeta_name = String("layers.", String(lyr), ".beta_proj.weight")

        var w_k = load_tensor_by_key(
            weights_path,
            header.data_base,
            header.entries,
            wk_name,
            telemetry,
        )
        var w_v = load_tensor_by_key(
            weights_path,
            header.data_base,
            header.entries,
            wv_name,
            telemetry,
        )
        var w_beta = load_tensor_by_key(
            weights_path,
            header.data_base,
            header.entries,
            wbeta_name,
            telemetry,
        )

        var proj = project_tokens_to_kv_beta(
            x_act, w_k, w_v, w_beta, seq_len, dk, dk, dv
        )

        chunked_gdn_scan(
            state,
            lyr,
            proj.k_mat,
            proj.v_mat,
            proj.beta,
            seq_len,
            dk,
            dv,
            chunk_size,
        )

    # 4. Tulis state ke berkas biner GDNS v1
    write_gdns_v1(output_path, state)
    print(
        "Mojo GDN scan completed successfully -> "
        + output_path
        + " (layers="
        + String(layers)
        + ", dv="
        + String(dv)
        + ", dk="
        + String(dk)
        + ", seq_len="
        + String(seq_len)
        + ", chunk_size="
        + String(chunk_size)
        + ")"
    )
