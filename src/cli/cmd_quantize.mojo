# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Handler subcommand quantize (M6)."""

from cli.errors import basename, dirname
from cli.m6_errors import fail_m6
from cli.sys_utils import c_mkdir, c_realpath, c_rename, c_unlink
from core.config import LoadMemoryTelemetry
from core.tensor_loader import load_tensor_f32_chunked
from format import (
    STHeader,
    TensorMeta,
    read_header,
)
from format.file_io import resolve_within_root
from format.index import parse_index
from format.half_float import (
    QUANT_DEFAULT_GROUP_SIZE,
    float16_to_u16,
    is_allowed_group_size,
    u16_to_float16,
)
from format.quant_reader import (
    BlockHeader,
    BlockTensorMeta,
    validate_block_header,
    validate_block_tensor_meta,
)
from format.types import json_escape
from quant.quant_algo import (
    QuantizedTensor,
    compute_quant_metrics,
    dequantize_tensor_q32,
    quantize_tensor_f11a,
    verify_qdomain_property,
)
from std.collections import Dict, List
from std.math import isinf, isnan, max
from std.os import SEEK_END, SEEK_SET
from std.sys.terminate import exit
from std.time import perf_counter_ns


@fieldwise_init
struct DiscoveredTensor(Copyable, Movable):
    var name: String
    var shard_file: String
    var shape: List[Int]
    var dtype: String
    var begin: Int
    var end: Int
    var data_base: Int


def do_check(check_file: String, workdir: String) raises:
    """Mode validasi read-only file kuantisasi (--check)."""
    var f: FileHandle
    try:
        f = open(check_file, "r")
    except:
        fail_m6(
            "M6_ERR_VALIDATION",
            "validation",
            "cannot open check file: " + check_file,
        )
        return

    var file_size = Int(f.seek(0, SEEK_END))
    if file_size < 256:
        f.close()
        fail_m6(
            "M6_ERR_VALIDATION",
            "validation",
            "file size too short (< 256 bytes): " + String(file_size),
        )
        return

    _ = f.seek(0, SEEK_SET)
    var raw_hdr = f.read_bytes(256)
    var hdr: BlockHeader
    try:
        hdr = BlockHeader.from_bytes(raw_hdr)
    except e:
        f.close()
        fail_m6(
            "M6_ERR_VALIDATION",
            "validation",
            "header parse failed: " + String(e),
        )
        return

    try:
        validate_block_header(hdr, file_size)
    except e:
        f.close()
        fail_m6(
            "M6_ERR_VALIDATION",
            "validation",
            "header validation failed: " + String(e),
        )
        return

    var cur_off = 256
    for t_idx in range(hdr.num_tensors):
        if cur_off + 4 > file_size:
            f.close()
            fail_m6(
                "M6_ERR_VALIDATION",
                "validation",
                "truncated record framing before tensor index " + String(t_idx),
            )
            return

        _ = f.seek(cur_off, SEEK_SET)
        var len_bytes = f.read_bytes(4)
        var meta_len = (
            Int(len_bytes[0])
            | (Int(len_bytes[1]) << 8)
            | (Int(len_bytes[2]) << 16)
            | (Int(len_bytes[3]) << 24)
        )
        if meta_len <= 0 or cur_off + 4 + meta_len > file_size:
            f.close()
            fail_m6(
                "M6_ERR_VALIDATION",
                "validation",
                "metadata JSON framing truncated",
            )
            return

        var json_bytes = f.read_bytes(meta_len)
        var meta: BlockTensorMeta
        try:
            meta = BlockTensorMeta.from_json_bytes(json_bytes)
        except e:
            f.close()
            fail_m6(
                "M6_ERR_VALIDATION",
                "validation",
                "metadata JSON parsing failed: " + String(e),
            )
            return

        try:
            validate_block_tensor_meta(meta)
        except e:
            f.close()
            fail_m6(
                "M6_ERR_VALIDATION",
                "validation",
                "tensor metadata validation failed: " + String(e),
            )
            return

        var s_bytes = meta.scales_bytes()
        var w_bytes = meta.weights_bytes()
        var payload_bytes = s_bytes + w_bytes
        if cur_off + 4 + meta_len + payload_bytes > file_size:
            f.close()
            fail_m6(
                "M6_ERR_VALIDATION",
                "validation",
                "tensor payload truncated: " + meta.name,
            )
            return

        # Validasi scales
        var scales_raw = f.read_bytes(s_bytes)
        for g in range(meta.num_groups):
            var u = UInt16(scales_raw[g * 2]) | (
                UInt16(scales_raw[g * 2 + 1]) << 8
            )
            var exp_bits = (u >> 10) & 0x1F
            if exp_bits == 0x1F:
                f.close()
                fail_m6(
                    "M6_ERR_DEQUANT",
                    "dequant",
                    "scale is NaN or Inf in tensor: " + meta.name,
                )
                return
            var s = u16_to_float16(u)
            if s <= Float16(0.0):
                f.close()
                fail_m6(
                    "M6_ERR_DEQUANT",
                    "dequant",
                    "scale is non-positive in tensor: " + meta.name,
                )
                return

        # Validasi weights
        var weights_raw = f.read_bytes(w_bytes)
        for k in range(w_bytes):
            var b = weights_raw[k]
            var lo = Int(b & 0x0F)
            var hi = Int((b >> 4) & 0x0F)
            if lo == 8 or hi == 8:
                f.close()
                fail_m6(
                    "M6_ERR_DEQUANT",
                    "dequant",
                    "reserved nibble 0x8 (-8) in tensor: " + meta.name,
                )
                return

        cur_off += 4 + meta_len + payload_bytes

    if cur_off != file_size:
        f.close()
        fail_m6(
            "M6_ERR_VALIDATION",
            "validation",
            "file size mismatch: trailing bytes after records ("
            + String(cur_off)
            + " != "
            + String(file_size)
            + ")",
        )
        return

    f.close()
    print(
        '{"status":"success","mode":"check","file":"'
        + json_escape(check_file)
        + '","num_tensors":'
        + String(hdr.num_tensors)
        + ',"total_bytes":'
        + String(hdr.total_bytes)
        + "}"
    )
    exit(0)


def cmd_quantize(args: List[String]) raises:
    var t0 = perf_counter_ns()
    var input_dir = String("")
    var output_dir = String("")
    var group_size = QUANT_DEFAULT_GROUP_SIZE
    var workdir = String("./work")
    var check_file = String("")
    var error_report_path = String("")

    # Parse arguments
    var i = 2
    while i < len(args):
        var a = args[i]
        if a == "--input-dir":
            if i + 1 >= len(args):
                fail_m6(
                    "M6_ERR_INPUT", "input", "--input-dir requires an argument"
                )
            input_dir = args[i + 1]
            i += 2
        elif a == "--output-dir":
            if i + 1 >= len(args):
                fail_m6(
                    "M6_ERR_INPUT", "input", "--output-dir requires an argument"
                )
            output_dir = args[i + 1]
            i += 2
        elif a == "--group-size":
            if i + 1 >= len(args):
                fail_m6(
                    "M6_ERR_INPUT", "input", "--group-size requires an argument"
                )
            var val_str = args[i + 1]
            var num = 0
            var bs = val_str.as_bytes()
            for b_idx in range(len(bs)):
                if bs[b_idx] >= 48 and bs[b_idx] <= 57:
                    num = num * 10 + (Int(bs[b_idx]) - 48)
                else:
                    fail_m6(
                        "M6_ERR_INPUT",
                        "input",
                        "invalid group-size: " + val_str,
                    )
            group_size = num
            i += 2
        elif a == "--workdir":
            if i + 1 >= len(args):
                fail_m6(
                    "M6_ERR_INPUT", "input", "--workdir requires an argument"
                )
            workdir = args[i + 1]
            i += 2
        elif a == "--check":
            if i + 1 >= len(args):
                fail_m6("M6_ERR_INPUT", "input", "--check requires an argument")
            check_file = args[i + 1]
            i += 2
        elif a == "--error-report":
            if i + 1 >= len(args):
                fail_m6(
                    "M6_ERR_INPUT",
                    "input",
                    "--error-report requires an argument",
                )
            error_report_path = args[i + 1]
            i += 2
        else:
            fail_m6("M6_ERR_INPUT", "input", "unknown argument: " + a)

    # Bila dalam mode --check, jalankan validasi read-only
    if check_file.byte_length() > 0:
        do_check(check_file, workdir)
        return

    # Validasi input required
    if input_dir.byte_length() == 0:
        fail_m6(
            "M6_ERR_INPUT",
            "input",
            "--input-dir is required (unless --check is specified)",
        )
    if output_dir.byte_length() == 0:
        fail_m6("M6_ERR_INPUT", "input", "--output-dir is required")
    if not is_allowed_group_size(group_size):
        fail_m6(
            "M6_ERR_INPUT",
            "input",
            "group-size must be in {32, 64, 128, 256}, got "
            + String(group_size),
        )

    # Validasi keberadaan input directory
    var model_root = c_realpath(input_dir)
    if model_root == "":
        fail_m6(
            "M6_ERR_INPUT",
            "input",
            "input directory does not exist: " + input_dir,
        )

    # Temukan index.json atau single shard safetensors
    var index_file = String(model_root, "/model.safetensors.index.json")
    var has_index: Bool
    try:
        var fi = open(index_file, "r")
        fi.close()
        has_index = True
    except:
        has_index = False

    var discovered = List[DiscoveredTensor]()
    if has_index:
        var packed: List[String]
        try:
            packed = parse_index(index_file)
        except e:
            fail_m6(
                "M6_ERR_INPUT",
                "input",
                "cannot parse safetensors index: " + String(e),
            )
            return

        var nn = 0
        var cs = packed[0].as_bytes()
        for c_i in range(len(cs)):
            nn = nn * 10 + (Int(cs[c_i]) - 48)

        # Cache headers per shard
        var shard_names = List[String]()
        var shard_headers = List[STHeader]()

        for idx in range(nn):
            var t_name = packed[1 + 2 * idx]
            var s_file = packed[1 + 2 * idx + 1]

            # Cari atau baca shard header
            var s_idx = -1
            for k in range(len(shard_names)):
                if shard_names[k] == s_file:
                    s_idx = k
                    break
            if s_idx < 0:
                var spath = String(model_root, "/", s_file)
                try:
                    var st = read_header(spath)
                    shard_names.append(s_file)
                    shard_headers.append(st^)
                    s_idx = len(shard_names) - 1
                except e:
                    fail_m6(
                        "M6_ERR_INPUT",
                        "input",
                        "cannot read shard: " + s_file + " (" + String(e) + ")",
                    )
                    return

            ref st = shard_headers[s_idx]
            var entry_idx = -1
            for e_i in range(len(st.entries)):
                if st.entries[e_i].name == t_name:
                    entry_idx = e_i
                    break
            if entry_idx < 0:
                fail_m6(
                    "M6_ERR_INPUT",
                    "input",
                    "tensor " + t_name + " not found in shard " + s_file,
                )
                return

            ref meta = st.entries[entry_idx]
            discovered.append(
                DiscoveredTensor(
                    name=t_name,
                    shard_file=s_file,
                    shape=meta.shape.copy(),
                    dtype=meta.dtype,
                    begin=meta.begin,
                    end=meta.end,
                    data_base=st.data_base,
                )
            )
    else:
        var single_shard = String(model_root, "/model.safetensors")
        try:
            var st = read_header(single_shard)
            for e_i in range(len(st.entries)):
                ref meta = st.entries[e_i]
                discovered.append(
                    DiscoveredTensor(
                        name=meta.name,
                        shard_file="model.safetensors",
                        shape=meta.shape.copy(),
                        dtype=meta.dtype,
                        begin=meta.begin,
                        end=meta.end,
                        data_base=st.data_base,
                    )
                )
        except e:
            fail_m6(
                "M6_ERR_INPUT",
                "input",
                "no valid safetensors index or shard found in: " + input_dir,
            )
            return

    if len(discovered) == 0:
        fail_m6(
            "M6_ERR_INPUT",
            "input",
            "no tensors discovered in input directory",
        )

    # ------------------------------------------------------------------
    # Verifikasi Tail Contract (N % G == 0) SEBELUM mulai kuantisasi
    # ------------------------------------------------------------------
    for d_i in range(len(discovered)):
        ref dt = discovered[d_i]
        var n_elem = 1
        for sh_i in range(len(dt.shape)):
            n_elem *= dt.shape[sh_i]
        if n_elem % group_size != 0:
            fail_m6(
                "M6_ERR_INPUT",
                "input",
                "tensor violates tail contract (N % G != 0): "
                + dt.name
                + " (N="
                + String(n_elem)
                + ", G="
                + String(group_size)
                + ")",
            )
            return

    # ------------------------------------------------------------------
    # Setup Direktori & Atomic Staging File
    # ------------------------------------------------------------------
    _ = c_mkdir(workdir)
    _ = c_mkdir(output_dir)

    var tmp_filename = String(
        workdir, "/quant_model.tmp.", String(perf_counter_ns())
    )
    var cleanup_files = List[String]()
    cleanup_files.append(tmp_filename)

    var f_out: FileHandle
    try:
        f_out = open(tmp_filename, "w")
    except e:
        fail_m6(
            "M6_ERR_OUTPUT",
            "output",
            "cannot create temporary quant file in workdir: " + String(e),
            cleanup_files=cleanup_files,
        )
        return

    # Tulis 256 dummy bytes untuk reservasi header
    var dummy_hdr = List[UInt8]()
    for _ in range(256):
        dummy_hdr.append(32)
    f_out.write_bytes(Span(dummy_hdr))

    var total_input_bytes = 0
    var total_records_bytes = 0
    var max_epsilon_rel = Float32(0.0)
    var min_epsilon_rel = Float32(1e9)
    var sum_epsilon_rel = Float32(0.0)
    var count_with_var = 0
    var num_high_error = 0
    var tensor_reports = List[String]()
    var telemetry = LoadMemoryTelemetry()

    # ------------------------------------------------------------------
    # Loop Kuantisasi Setiap Tensor
    # ------------------------------------------------------------------
    for d_i in range(len(discovered)):
        ref dt = discovered[d_i]
        var shard_path = String(model_root, "/", dt.shard_file)
        var tmeta = TensorMeta(
            dt.name,
            dt.dtype,
            dt.shape.copy(),
            dt.begin,
            dt.end,
        )

        var weights: List[Float32]
        try:
            weights = load_tensor_f32_chunked(
                shard_path, dt.data_base, tmeta, telemetry
            )
        except e:
            f_out.close()
            fail_m6(
                "M6_ERR_QUANT",
                "quantization",
                "failed to load tensor: " + dt.name + " (" + String(e) + ")",
                cleanup_files=cleanup_files,
            )
            return

        # Cek nilai NaN/Inf di dalam tensor
        for w_i in range(len(weights)):
            var w = weights[w_i]
            if isnan(w) or isinf(w):
                f_out.close()
                fail_m6(
                    "M6_ERR_QUANT",
                    "quantization",
                    "NaN or Inf encountered in tensor: " + dt.name,
                    cleanup_files=cleanup_files,
                )
                return

        # Kuantisasi F11a
        var q_res: QuantizedTensor
        try:
            q_res = quantize_tensor_f11a(weights, group_size)
        except e:
            f_out.close()
            fail_m6(
                "M6_ERR_QUANT",
                "quantization",
                "quantization failed for tensor: "
                + dt.name
                + " ("
                + String(e)
                + ")",
                cleanup_files=cleanup_files,
            )
            return

        # Buat metadata tensor
        var qmeta: BlockTensorMeta
        try:
            qmeta = BlockTensorMeta(
                name=dt.name,
                shape=dt.shape.copy(),
                dtype="BF16",
                group_size=group_size,
            )
        except e:
            f_out.close()
            fail_m6(
                "M6_ERR_QUANT",
                "quantization",
                "cannot construct tensor metadata: " + String(e),
                cleanup_files=cleanup_files,
            )
            return

        var rec_bytes = qmeta.to_record_bytes()
        var scales_raw = List[UInt8]()
        for sc_i in range(len(q_res.scales)):
            var u = float16_to_u16(q_res.scales[sc_i])
            scales_raw.append(UInt8(u & 0xFF))
            scales_raw.append(UInt8((u >> 8) & 0xFF))

        f_out.write_bytes(Span(rec_bytes))
        f_out.write_bytes(Span(scales_raw))
        f_out.write_bytes(Span(q_res.packed_bytes))

        total_input_bytes += len(weights) * 2
        total_records_bytes += (
            len(rec_bytes) + len(scales_raw) + len(q_res.packed_bytes)
        )

        # Hitung metrik relative error
        var dequant_32 = dequantize_tensor_q32(
            q_res.scales, q_res.q_weights, group_size
        )
        var m = compute_quant_metrics(weights, dequant_32)
        if not m.zero_variance:
            if m.epsilon_rel > max_epsilon_rel:
                max_epsilon_rel = m.epsilon_rel
            if m.epsilon_rel < min_epsilon_rel:
                min_epsilon_rel = m.epsilon_rel
            sum_epsilon_rel += m.epsilon_rel
            count_with_var += 1
            if m.epsilon_rel > Float32(0.01):
                num_high_error += 1

        if error_report_path.byte_length() > 0:
            var prop_ok: Bool
            var max_abs_err: Float32
            try:
                var prop_res = verify_qdomain_property(
                    weights, dequant_32, q_res.scales, group_size
                )
                prop_ok = prop_res[0]
                max_abs_err = prop_res[1]
            except:
                prop_ok = False
                max_abs_err = Float32(0.0)

            var rec_json = (
                String('{"name":"') + json_escape(dt.name) + '","shape":['
            )
            for s_i in range(len(dt.shape)):
                if s_i > 0:
                    rec_json += ","
                rec_json += String(dt.shape[s_i])
            rec_json += '],"epsilon_rel":'
            if m.zero_variance:
                rec_json += 'null,"zero_variance":true'
            else:
                rec_json += String(m.epsilon_rel) + ',"zero_variance":false'
            rec_json += ',"mse":' + String(m.mse)
            rec_json += ',"property_ok":' + (
                String("true") if prop_ok else String("false")
            )
            rec_json += ',"max_abs_error":' + String(max_abs_err)
            rec_json += ',"num_groups":' + String(len(q_res.scales)) + "}"
            tensor_reports.append(rec_json)

    # ------------------------------------------------------------------
    # Tulis Header Final 256 Byte
    # ------------------------------------------------------------------
    var total_output_bytes = 256 + total_records_bytes
    var model_name = "qwen1.5-moe-a2.7b-chat"
    var final_hdr = BlockHeader(
        model=model_name,
        num_tensors=len(discovered),
        total_bytes=total_output_bytes,
        group_size=group_size,
    )

    var hdr_bytes = final_hdr.to_header_bytes()
    try:
        _ = f_out.seek(0, SEEK_SET)
        f_out.write_bytes(Span(hdr_bytes))
    except e:
        f_out.close()
        fail_m6(
            "M6_ERR_OUTPUT",
            "output",
            "failed writing final header: " + String(e),
            cleanup_files=cleanup_files,
        )
        return
    f_out.close()

    # ------------------------------------------------------------------
    # Validasi Internal Output File (F11b Validation)
    # ------------------------------------------------------------------
    var val_f: FileHandle
    try:
        val_f = open(tmp_filename, "r")
    except:
        fail_m6(
            "M6_ERR_VALIDATION",
            "validation",
            "cannot reopen temp file for internal validation",
            cleanup_files=cleanup_files,
        )
        return

    var verified_sz = Int(val_f.seek(0, SEEK_END))
    val_f.close()
    if verified_sz != total_output_bytes:
        fail_m6(
            "M6_ERR_VALIDATION",
            "validation",
            "internal output size mismatch: expected "
            + String(total_output_bytes)
            + ", got "
            + String(verified_sz),
            cleanup_files=cleanup_files,
        )
        return

    # ------------------------------------------------------------------
    # Atomic Rename ke Output Destination
    # ------------------------------------------------------------------
    var final_dest = String(output_dir, "/model_quant.bin")
    var ren_res = c_rename(tmp_filename, final_dest)
    if ren_res != 0:
        fail_m6(
            "M6_ERR_OUTPUT",
            "output",
            "atomic rename failed from " + tmp_filename + " to " + final_dest,
            cleanup_files=cleanup_files,
        )
        return

    # ------------------------------------------------------------------
    # Output JSON Sukses ke STDOUT
    # ------------------------------------------------------------------
    var walltime_sec = Float64(perf_counter_ns() - t0) / 1e9
    var comp_ratio = (
        Float64(total_input_bytes)
        / Float64(total_output_bytes) if total_output_bytes
        > 0 else 0.0
    )
    var avg_epsilon_rel = (
        Float64(sum_epsilon_rel) / Float64(count_with_var) if count_with_var
        > 0 else 0.0
    )

    if min_epsilon_rel == Float32(1e9):
        min_epsilon_rel = Float32(0.0)

    if error_report_path.byte_length() > 0:
        try:
            var frep = open(error_report_path, "w")
            frep.write("{\n")
            frep.write('  "run_id": "M6-QUANT-CLI",\n')
            frep.write('  "model": "' + json_escape(model_name) + '",\n')
            frep.write('  "group_size": ' + String(group_size) + ",\n")
            frep.write('  "summary": {\n')
            frep.write('    "num_tensors": ' + String(len(discovered)) + ",\n")
            frep.write(
                '    "max_epsilon_rel": ' + String(max_epsilon_rel) + ",\n"
            )
            frep.write(
                '    "avg_epsilon_rel": ' + String(avg_epsilon_rel) + ",\n"
            )
            frep.write(
                '    "min_epsilon_rel": ' + String(min_epsilon_rel) + ",\n"
            )
            frep.write('    "num_high_error": ' + String(num_high_error) + "\n")
            frep.write("  },\n")
            frep.write('  "tensors": [\n')
            for r_i in range(len(tensor_reports)):
                frep.write("    " + tensor_reports[r_i])
                if r_i + 1 < len(tensor_reports):
                    frep.write(",\n")
                else:
                    frep.write("\n")
            frep.write("  ]\n")
            frep.write("}\n")
            frep.close()
        except:
            pass

    print(
        '{"status":"success","run_id":"M6-QUANT-CLI","model":"'
        + json_escape(model_name)
        + '","input_format":"BF16","output_format":"4-bit'
        ' per-group","group_size":'
        + String(group_size)
        + ',"num_tensors":'
        + String(len(discovered))
        + ',"metrics":{"walltime_sec":'
        + String(walltime_sec)
        + ',"input_bytes":'
        + String(total_input_bytes)
        + ',"output_bytes":'
        + String(total_output_bytes)
        + ',"compression_ratio":'
        + String(comp_ratio)
        + ',"avg_epsilon_rel":'
        + String(avg_epsilon_rel)
        + ',"max_epsilon_rel":'
        + String(max_epsilon_rel)
        + "}}"
    )
    exit(0)
