# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah forward CLI kimo (M4 full forward streaming)."""

from cli.config_parser import parse_model_config
from cli.errors import dirname
from cli.m4_errors import fail_m4
from cli.sys_utils import (
    allocate_run_id_and_dir,
    c_access_w,
    c_close_fd,
    c_mkdir,
    c_open_tmp_excl,
    c_rename,
    c_rmdir,
    c_unlink,
    c_write_f32_fd_n,
    cleanup_run_resources,
    get_cgroup_oom_kills,
    get_cgroup_peak_bytes,
    get_file_size,
    get_proc_io_read_bytes,
    get_vmhwm_bytes,
)
from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import (
    ShardHeaderCache,
    _load_one_tensor_by_name,
)
from format.file_io import (
    c_realpath,
    path_is_within,
    read_small_file,
    resolve_within_root,
)
from format.index import parse_index
from format.types import json_escape
from layers.forward_layer import (
    LayerTiming,
    forward_single_layer,
)
from layers.head import (
    embedding_lookup,
    matmul_activation_head,
    validate_logits,
)
from layers.qkv_bias import validate_attention_bias_in_index
from layers.rmsnorm import rmsnorm
from std.collections import Dict, List
from std.time import perf_counter_ns


def parse_flat_u32_tokens(
    raw: List[UInt8], tokens_path: String
) raises -> List[Int]:
    var n = len(raw)
    var pos = 0

    # Lewati whitespace di awal
    while pos < n and (
        raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
    ):
        pos += 1

    if pos >= n or raw[pos] != 91:  # '['
        raise Error("tokens JSON must start with '['")
    pos += 1

    var out = List[Int]()

    # Cek array kosong ']'
    while pos < n and (
        raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
    ):
        pos += 1

    if pos < n and raw[pos] == 93:  # ']'
        pos += 1
        while pos < n and (
            raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
        ):
            pos += 1
        if pos < n:
            raise Error("trailing characters after closing ']'")
        return out^

    while True:
        while pos < n and (
            raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
        ):
            pos += 1

        if pos >= n:
            raise Error("unterminated tokens array (missing ']')")

        var b = raw[pos]
        if b == 91:
            raise Error("nested array not allowed, expected flat u32 array")
        if b == 45:
            raise Error("negative integer not allowed, expected unsigned u32")
        if b == 34:
            raise Error("string element not allowed, expected unsigned u32")
        if b < 48 or b > 57:
            raise Error("expected unsigned integer digit")

        # Cek leading zero
        if (
            b == 48
            and pos + 1 < n
            and (raw[pos + 1] >= 48 and raw[pos + 1] <= 57)
        ):
            raise Error("leading zero in integer is not valid JSON")

        var val = 0
        while pos < n and (raw[pos] >= 48 and raw[pos] <= 57):
            var d = Int(raw[pos] - 48)
            if val > 429496729 or (val == 429496729 and d > 5):
                raise Error("token integer exceeds u32 range")
            val = val * 10 + d
            pos += 1

        # Cek float
        if pos < n and (raw[pos] == 46 or raw[pos] == 101 or raw[pos] == 69):
            raise Error("float not allowed, expected integer")

        out.append(val)

        while pos < n and (
            raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
        ):
            pos += 1

        if pos >= n:
            raise Error("unterminated tokens array (missing ']')")

        var next_c = raw[pos]
        pos += 1
        if next_c == 93:  # ']'
            break
        elif next_c == 44:  # ','
            var p_peek = pos
            while p_peek < n and (
                raw[p_peek] == 32
                or raw[p_peek] == 9
                or raw[p_peek] == 10
                or raw[p_peek] == 13
            ):
                p_peek += 1
            if p_peek < n and raw[p_peek] == 93:
                raise Error("trailing comma not allowed in JSON array")
        else:
            raise Error("expected ',' or ']' after token integer")

    while pos < n and (
        raw[pos] == 32 or raw[pos] == 9 or raw[pos] == 10 or raw[pos] == 13
    ):
        pos += 1

    if pos < n:
        raise Error("trailing characters after closing ']'")

    return out^


def cmd_forward(args: List[String]) raises:
    var t_start = perf_counter_ns()

    var model_dir = String("")
    var tokens_path = String("")
    var output_file = String("")
    var workdir = String("./work")
    var dump_routing = String("")
    var threads = 1
    var layer_timing = String("")
    var custom_run_id = String("")
    var mock_error = String("")
    var mock_forward = False

    # Parse command-line flags
    var i = 2
    while i < len(args):
        var a = String(args[i])
        if a == "--model-dir":
            if i + 1 >= len(args):
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "missing argument for --model-dir",
                )
            model_dir = String(args[i + 1])
            i += 2
        elif a == "--tokens":
            if i + 1 >= len(args):
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "missing argument for --tokens",
                )
            tokens_path = String(args[i + 1])
            i += 2
        elif a == "--output":
            if i + 1 >= len(args):
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "missing argument for --output",
                )
            output_file = String(args[i + 1])
            i += 2
        elif a == "--workdir":
            if i + 1 >= len(args):
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "missing argument for --workdir",
                )
            workdir = String(args[i + 1])
            i += 2
        elif a == "--dump-routing":
            if i + 1 >= len(args):
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "missing argument for --dump-routing",
                )
            dump_routing = String(args[i + 1])
            i += 2
        elif a == "--threads":
            if i + 1 >= len(args):
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "missing argument for --threads",
                )
            try:
                threads = Int(String(args[i + 1]))
            except:
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "invalid integer for --threads: " + String(args[i + 1]),
                )
            i += 2
        elif a == "--layer-timing":
            if i + 1 >= len(args):
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "missing argument for --layer-timing",
                )
            layer_timing = String(args[i + 1])
            i += 2
        elif a == "--run-id":
            if i + 1 >= len(args):
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "missing argument for --run-id",
                )
            custom_run_id = String(args[i + 1])
            i += 2
        elif a == "--mock-error":
            if i + 1 >= len(args):
                fail_m4(
                    "M4_ERR_INPUT",
                    "input",
                    "missing argument for --mock-error",
                )
            mock_error = String(args[i + 1])
            i += 2
        elif a == "--mock-forward":
            mock_forward = True
            i += 1
        elif a.startswith("-"):
            fail_m4("M4_ERR_INPUT", "input", "unknown option: " + a)
        else:
            fail_m4(
                "M4_ERR_INPUT",
                "input",
                "unexpected positional argument: " + a,
            )

    # Validasi keberadaan argumen wajib
    if model_dir.byte_length() == 0:
        fail_m4("M4_ERR_INPUT", "input", "missing required option: --model-dir")
    if tokens_path.byte_length() == 0:
        fail_m4("M4_ERR_INPUT", "input", "missing required option: --tokens")
    if output_file.byte_length() == 0:
        fail_m4("M4_ERR_INPUT", "input", "missing required option: --output")
    if threads < 1:
        fail_m4("M4_ERR_INPUT", "input", "--threads must be positive integer")

    # VALIDASI BERURUTAN SEBELUM ALOKASI BESAR
    # (1) stat ukuran file <= MAX_TOKENS_FILE_BYTES (1 MiB = 1048576 B)
    var tok_sz = get_file_size(tokens_path)
    if tok_sz < 0:
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "tokens file not found or cannot be opened: " + tokens_path,
        )
    if tok_sz > 1048576:
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            String(
                "tokens file size ",
                tok_sz,
                " bytes exceeds MAX_TOKENS_FILE_BYTES (1048576 bytes)",
            ),
        )

    # (2) parse sebagai array-datar-u32
    var raw_tok = List[UInt8]()
    try:
        raw_tok = read_small_file(tokens_path)
    except e:
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "failed reading tokens file: " + String(e),
        )

    var tokens = List[Int]()
    try:
        tokens = parse_flat_u32_tokens(raw_tok, tokens_path)
    except e:
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "invalid tokens JSON: " + String(e),
        )

    # (3) count 1..MAX_TOKENS (1024)
    var num_tokens = len(tokens)
    if num_tokens < 1:
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "tokens array is empty (count must be 1..1024)",
        )
    if num_tokens > 1024:
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            String(
                "tokens count ",
                num_tokens,
                " exceeds MAX_TOKENS (1024)",
            ),
        )

    # (4) tiap ID < 151936 (vocab size)
    for ti in range(num_tokens):
        var tid = tokens[ti]
        if tid >= 151936:
            var det = String(
                '{"token_id":',
                tid,
                ',"vocab_size":151936,"tokens_path":"',
                json_escape(tokens_path),
                '"}',
            )
            fail_m4(
                "M4_ERR_INPUT",
                "input",
                String(
                    "Token ID ",
                    tid,
                    " out of vocabulary range (max: 151935)",
                ),
                det,
            )

    # (5) model dir ada
    var model_canon = c_realpath(model_dir)
    if model_canon.byte_length() == 0:
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "model directory not found: " + model_dir,
        )

    # (6) workdir writable
    _ = c_mkdir(workdir)
    var workdir_canon = c_realpath(workdir)
    if workdir_canon.byte_length() == 0:
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "workdir does not exist and cannot be created: " + workdir,
        )
    if not c_access_w(workdir_canon):
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "workdir not writable: " + workdir_canon,
        )

    # (7) Aturan output path (opsi A)
    var target_output = output_file
    if not output_file.startswith("/"):
        target_output = String(workdir_canon, "/", output_file)

    var out_parent = dirname(target_output)
    if out_parent.byte_length() == 0:
        out_parent = workdir_canon
    _ = c_mkdir(out_parent)
    var parent_canon = c_realpath(out_parent)
    if parent_canon.byte_length() == 0 or not path_is_within(
        workdir_canon, parent_canon
    ):
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "output path escapes workdir: " + target_output,
        )
    var target_canon = c_realpath(target_output)
    if target_canon.byte_length() > 0 and not path_is_within(
        workdir_canon, target_canon
    ):
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "output path escapes workdir: " + target_output,
        )
    if not c_access_w(parent_canon):
        fail_m4(
            "M4_ERR_INPUT",
            "input",
            "output parent directory not writable: " + parent_canon,
        )

    # (8) layer-timing path check (jika ada)
    var target_timing = String("")
    if layer_timing.byte_length() > 0:
        target_timing = layer_timing
        if not layer_timing.startswith("/"):
            target_timing = String(workdir_canon, "/", layer_timing)
        var timing_parent = dirname(target_timing)
        if timing_parent.byte_length() == 0:
            timing_parent = workdir_canon
        _ = c_mkdir(timing_parent)
        var timing_parent_canon = c_realpath(timing_parent)
        if timing_parent_canon.byte_length() == 0 or not path_is_within(
            workdir_canon, timing_parent_canon
        ):
            fail_m4(
                "M4_ERR_INPUT",
                "input",
                "layer-timing path escapes workdir: " + target_timing,
            )

    # (9) dump-routing check (jika ada)
    if dump_routing.byte_length() > 0:
        _ = c_mkdir(dump_routing)
        var dump_canon = c_realpath(dump_routing)
        if dump_canon.byte_length() == 0:
            fail_m4(
                "M4_ERR_INPUT",
                "input",
                "cannot access or create dump-routing directory: "
                + dump_routing,
            )

    # Alokasi run-id dan direktori temp runs/<run-id>/
    var run_pair = allocate_run_id_and_dir(workdir_canon, custom_run_id)
    var run_id = run_pair[0]
    var run_dir = run_pair[1]

    var tmp_files = List[String]()
    var tmp_output = String(target_output, ".tmp.", run_id)
    tmp_files.append(tmp_output)
    if target_timing.byte_length() > 0:
        tmp_files.append(String(target_timing, ".tmp.", run_id))

    # Mock error testing support (Wave 1)
    if mock_error.byte_length() > 0:
        if mock_error == "M4_ERR_INDEX":
            fail_m4(
                "M4_ERR_INDEX",
                "index_load",
                "F15 validation failed: shard header checksum mismatch",
                '{"shard":"model-00001-of-00008.safetensors"}',
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        elif mock_error == "M4_ERR_MEMORY":
            fail_m4(
                "M4_ERR_MEMORY",
                "layer_forward",
                "Memory allocation failed: layer weight buffer",
                '{"layer":12,"requested_bytes":1141121024}',
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        elif mock_error == "M4_ERR_SHARD_IO":
            fail_m4(
                "M4_ERR_SHARD_IO",
                "layer_forward",
                "Failed to read shard: I/O error",
                '{"layer":12,"shard":"model-00002-of-00008.safetensors"}',
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        elif mock_error == "M4_ERR_LAYER_FORWARD":
            fail_m4(
                "M4_ERR_LAYER_FORWARD",
                "layer_forward",
                "NaN detected in layer forward",
                '{"layer":12}',
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        elif mock_error == "M4_ERR_OUTPUT":
            fail_m4(
                "M4_ERR_OUTPUT",
                "output",
                "Failed atomic write logits",
                "{}",
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        elif mock_error == "M4_ERR_COMPARE":
            fail_m4(
                "M4_ERR_COMPARE",
                "compare",
                "Rust compare failed",
                "{}",
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        else:
            fail_m4(
                "M4_ERR_INPUT",
                "input",
                "unknown mock error: " + mock_error,
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

    if mock_forward:
        var t_write_start = perf_counter_ns()
        var fd = c_open_tmp_excl(tmp_output)
        if fd < 0:
            fail_m4(
                "M4_ERR_OUTPUT",
                "output",
                "cannot create tmp file: " + tmp_output,
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

        var total_floats = num_tokens * 151936
        var chunk_floats = 1024
        var zero_chunk = List[Float32]()
        for _ in range(chunk_floats):
            zero_chunk.append(Float32(0.0))

        var written_floats = 0
        var write_ok = True
        while written_floats < total_floats:
            var cur = chunk_floats
            if total_floats - written_floats < cur:
                cur = total_floats - written_floats
            if not c_write_f32_fd_n(fd, zero_chunk, cur):
                write_ok = False
                break
            written_floats += cur

        _ = c_close_fd(fd)
        if not write_ok:
            fail_m4(
                "M4_ERR_OUTPUT",
                "output",
                "write failed to tmp file: " + tmp_output,
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

        var ren_ret = c_rename(tmp_output, target_output)
        if ren_ret != 0:
            fail_m4(
                "M4_ERR_OUTPUT",
                "output",
                "atomic rename failed from "
                + tmp_output
                + " to "
                + target_output,
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

        if target_timing.byte_length() > 0:
            var tmp_timing = String(target_timing, ".tmp.", run_id)
            try:
                var ft = open(tmp_timing, "w")
                ft.write(
                    String(
                        '{"run_id":"',
                        run_id,
                        '","layer_timing":[],"total_layer_forward_sec":0.0}\n',
                    )
                )
                ft.close()
                _ = c_rename(tmp_timing, target_timing)
            except:
                fail_m4(
                    "M4_ERR_OUTPUT",
                    "output",
                    "failed writing layer timing",
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

        var t_write_end = perf_counter_ns()
        var write_sec = Float64(t_write_end - t_write_start) / 1e9

        cleanup_run_resources(run_dir, tmp_files)

        var t_end = perf_counter_ns()
        var walltime_sec = Float64(t_end - t_start) / 1e9
        var vmhwm_bytes = get_vmhwm_bytes()
        var phys_read = get_proc_io_read_bytes()
        var cgroup_peak = get_cgroup_peak_bytes()
        var cgroup_oom = get_cgroup_oom_kills()

        var out_json = String(
            '{\n  "status": "success",\n  "run_id": "',
            run_id,
            '",\n  "model": "qwen1.5-moe-a2.7b-chat",\n  "num_tokens": ',
            String(num_tokens),
            ',\n  "num_layers": 24,\n  "logits_path": "',
            json_escape(target_output),
            '",\n  "metrics": {\n    "walltime_sec": ',
            String(walltime_sec),
            ',\n    "vmhwm_bytes": ',
            String(vmhwm_bytes),
            ',\n    "logical_bytes_read": 0,\n    "physical_read_bytes": ',
            String(phys_read),
            ',\n    "cgroup_peak_bytes": ',
            String(cgroup_peak),
            ',\n    "cgroup_oom_kills": ',
            String(cgroup_oom),
            (
                ',\n    "phases": {\n      "index_load_sec": 0.0,\n     '
                ' "embedding_sec": 0.0,\n      "layer_forward_sec": 0.0,\n     '
                ' "final_norm_sec": 0.0,\n      "lm_head_sec": 0.0,\n     '
                ' "write_sec": '
            ),
            String(write_sec),
            "\n    }\n  }\n}",
        )
        print(out_json)
        return

    # -------------------------------------------------------------
    # FASE 1: INDEX LOAD & VALIDASI TENSOR/BIAS (HEADER-ONLY)
    # -------------------------------------------------------------
    var t_idx0 = perf_counter_ns()
    var index_path = String(model_canon, "/model.safetensors.index.json")
    if c_realpath(index_path) == "":
        fail_m4(
            "M4_ERR_INDEX",
            "index_load",
            "index file not found: " + index_path,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    var config_path = String(model_canon, "/model_config.json")
    if c_realpath(config_path) == "":
        var alt_cfg = String(model_canon, "/config.json")
        if c_realpath(alt_cfg) != "":
            config_path = alt_cfg
    if c_realpath(config_path) == "":
        fail_m4(
            "M4_ERR_INDEX",
            "index_load",
            "model config file not found: " + config_path,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    var cfg_tuple = parse_model_config(config_path)
    var cfg = cfg_tuple[0].copy()
    var eps = cfg_tuple[1]

    # Parse index packed dan bangun weight_map
    var packed = parse_index(index_path)
    var num_tensors = 0
    var cs = packed[0].as_bytes()
    for ci in range(len(cs)):
        num_tensors = num_tensors * 10 + (Int(cs[ci]) - 48)

    var weight_map = Dict[String, String]()
    var total_biases = 0
    for w in range(num_tensors):
        var tname = packed[1 + 2 * w]
        var sf = packed[1 + 2 * w + 1]
        weight_map[tname] = sf

        if tname.endswith(".bias"):
            total_biases += 1
            if tname.find("o_proj.bias") >= 0:
                fail_m4(
                    "M4_ERR_INDEX",
                    "index_load",
                    "checkpoint must NOT have o_proj.bias: " + tname,
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )
            if tname.find("mlp.") >= 0:
                fail_m4(
                    "M4_ERR_INDEX",
                    "index_load",
                    "checkpoint must NOT have mlp bias: " + tname,
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

    # Validasi 72 bias attention (q/k/v * 24 layer)
    try:
        validate_attention_bias_in_index(weight_map, cfg.num_hidden_layers)
    except e:
        fail_m4(
            "M4_ERR_INDEX",
            "index_load",
            "attention bias validation failed: " + String(e),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    if total_biases != 3 * cfg.num_hidden_layers:
        fail_m4(
            "M4_ERR_INDEX",
            "index_load",
            String(
                "total bias tensors mismatch: expected ",
                3 * cfg.num_hidden_layers,
                " got ",
                total_biases,
            ),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # Validasi keberadaan resident tensors
    var req_embed = "model.embed_tokens.weight"
    var req_norm = "model.norm.weight"
    var req_head = "lm_head.weight"
    if (
        req_embed not in weight_map
        or req_norm not in weight_map
        or req_head not in weight_map
    ):
        fail_m4(
            "M4_ERR_INDEX",
            "index_load",
            (
                "missing resident tensors in index (embed_tokens, norm, or"
                " lm_head)"
            ),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # Header-only validation across all unique shards (0 byte tensor body read)
    var cache = ShardHeaderCache()
    var telemetry = LoadMemoryTelemetry()

    var unique_shards = List[String]()
    for w in range(num_tensors):
        var sf = packed[1 + 2 * w + 1]
        var found = False
        for u in range(len(unique_shards)):
            if unique_shards[u] == sf:
                found = True
                break
        if not found:
            unique_shards.append(sf)

    for u in range(len(unique_shards)):
        var sf = unique_shards[u]
        var sp = resolve_within_root(model_canon, sf)
        if c_realpath(sp) == "":
            fail_m4(
                "M4_ERR_SHARD_IO",
                "index_load",
                "shard file not found on disk: " + sp,
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        try:
            _ = cache.get_or_read(sp)
        except e:
            fail_m4(
                "M4_ERR_INDEX",
                "index_load",
                "failed reading shard header F15 for " + sp + ": " + String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

    var t_idx1 = perf_counter_ns()
    var index_load_sec = Float64(t_idx1 - t_idx0) / 1e9

    # -------------------------------------------------------------
    # FASE 2: RESIDENT WEIGHTS & EMBEDDING LOOKUP
    # -------------------------------------------------------------
    var t_emb0 = perf_counter_ns()
    var embed_tokens = List[Float32]()
    var norm_weight = List[Float32]()
    var lm_head = List[Float32]()

    var shard_embed = weight_map[req_embed]
    var shard_norm = weight_map[req_norm]
    var shard_head = weight_map[req_head]

    try:
        embed_tokens = _load_one_tensor_by_name(
            cache,
            model_canon,
            shard_embed,
            req_embed,
            cfg.vocab_size,
            cfg.hidden_size,
            True,
            telemetry,
        )
        norm_weight = _load_one_tensor_by_name(
            cache,
            model_canon,
            shard_norm,
            req_norm,
            cfg.hidden_size,
            1,
            False,
            telemetry,
        )
        lm_head = _load_one_tensor_by_name(
            cache,
            model_canon,
            shard_head,
            req_head,
            cfg.vocab_size,
            cfg.hidden_size,
            True,
            telemetry,
        )
    except e:
        fail_m4(
            "M4_ERR_SHARD_IO",
            "embedding",
            "failed loading resident weights: " + String(e),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # Embedding lookup: token IDs -> hidden state [num_tokens, H]
    var hidden = List[Float32]()
    try:
        hidden = embedding_lookup(
            tokens, embed_tokens, cfg.vocab_size, cfg.hidden_size
        )
    except e:
        fail_m4(
            "M4_ERR_INPUT",
            "embedding",
            "failed embedding lookup: " + String(e),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # Free embed_tokens buffer segera setelah lookup (menghemat ~1.24 GB RAM)
    _ = embed_tokens^

    var t_emb1 = perf_counter_ns()
    var embedding_sec = Float64(t_emb1 - t_emb0) / 1e9

    # -------------------------------------------------------------
    # FASE 3: STREAMING LAYER LOOP (0..num_hidden_layers - 1)
    # -------------------------------------------------------------
    var t_layers0 = perf_counter_ns()
    var layer_timings = List[LayerTiming]()

    for l in range(cfg.num_hidden_layers):
        try:
            var timing = forward_single_layer(
                hidden,
                l,
                num_tokens,
                model_canon,
                weight_map,
                cfg,
                eps,
                cache,
                telemetry,
                dump_routing,
            )
            layer_timings.append(timing^)
        except e:
            var err_msg = String(e)
            if (
                err_msg.find("non-finite") >= 0
                or err_msg.find("overflow") >= 0
                or err_msg.find("NaN") >= 0
            ):
                fail_m4(
                    "M4_ERR_LAYER_FORWARD",
                    "layer_forward",
                    String(
                        "NaN/INF or overflow detected in layer ",
                        l,
                        ": ",
                        err_msg,
                    ),
                    String('{"layer":', l, "}"),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )
            elif (
                err_msg.find("memory") >= 0
                or err_msg.find("allocate") >= 0
                or err_msg.find("MEMORY") >= 0
            ):
                fail_m4(
                    "M4_ERR_MEMORY",
                    "layer_forward",
                    String(
                        "Memory allocation failed in layer ", l, ": ", err_msg
                    ),
                    String('{"layer":', l, "}"),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )
            else:
                fail_m4(
                    "M4_ERR_SHARD_IO",
                    "layer_forward",
                    String("failed forward layer ", l, ": ", err_msg),
                    String('{"layer":', l, "}"),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

    var t_layers1 = perf_counter_ns()
    var layer_forward_sec = Float64(t_layers1 - t_layers0) / 1e9

    # -------------------------------------------------------------
    # FASE 4: FINAL RMSNORM & LM HEAD
    # -------------------------------------------------------------
    var t_fn0 = perf_counter_ns()
    var normed_hidden = List[Float32]()
    normed_hidden.reserve(num_tokens * cfg.hidden_size)
    var tok_v = List[Float32]()
    tok_v.resize(cfg.hidden_size, Float32(0.0))
    var p_tok_v = tok_v.unsafe_ptr()
    var p_hid = hidden.unsafe_ptr()
    for t in range(num_tokens):
        var row = t * cfg.hidden_size
        for h in range(cfg.hidden_size):
            p_tok_v[unsafe_offset=h] = p_hid[unsafe_offset=row + h]
        try:
            var normed = rmsnorm(tok_v, norm_weight, eps)
            var p_normed = normed.unsafe_ptr()
            for h in range(cfg.hidden_size):
                normed_hidden.append(p_normed[unsafe_offset=h])
        except e:
            fail_m4(
                "M4_ERR_LAYER_FORWARD",
                "final_norm",
                "failed final rmsnorm: " + String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

    var t_fn1 = perf_counter_ns()
    var final_norm_sec = Float64(t_fn1 - t_fn0) / 1e9

    # LM Head Projection: [num_tokens, H] x [H, V] -> [num_tokens, V]
    var t_lm0 = perf_counter_ns()
    var logits = matmul_activation_head(
        normed_hidden, lm_head, num_tokens, cfg.vocab_size, cfg.hidden_size
    )

    try:
        validate_logits(logits, 1, num_tokens, cfg.vocab_size)
    except e:
        fail_m4(
            "M4_ERR_LAYER_FORWARD",
            "lm_head",
            "logits validation failed: " + String(e),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    var t_lm1 = perf_counter_ns()
    var lm_head_sec = Float64(t_lm1 - t_lm0) / 1e9

    # -------------------------------------------------------------
    # FASE 5: ATOMIC WRITE & TIMING COMMITS
    # -------------------------------------------------------------
    var t_write_start = perf_counter_ns()
    var fd = c_open_tmp_excl(tmp_output)
    if fd < 0:
        fail_m4(
            "M4_ERR_OUTPUT",
            "output",
            "cannot create tmp file: " + tmp_output,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    var total_floats = len(logits)
    var chunk_floats = 8192
    var written_floats = 0
    var write_ok = True
    var chunk_buf = List[Float32]()
    chunk_buf.resize(chunk_floats, Float32(0.0))
    var p_chunk = chunk_buf.unsafe_ptr()
    var p_logits = logits.unsafe_ptr()

    while written_floats < total_floats:
        var cur = chunk_floats
        if total_floats - written_floats < cur:
            cur = total_floats - written_floats
        for ci in range(cur):
            p_chunk[unsafe_offset=ci] = p_logits[
                unsafe_offset=written_floats + ci
            ]
        if not c_write_f32_fd_n(fd, chunk_buf, cur):
            write_ok = False
            break
        written_floats += cur

    _ = c_close_fd(fd)
    if not write_ok:
        fail_m4(
            "M4_ERR_OUTPUT",
            "output",
            "write failed to tmp file: " + tmp_output,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # Atomic rename logits output
    var ren_ret = c_rename(tmp_output, target_output)
    if ren_ret != 0:
        fail_m4(
            "M4_ERR_OUTPUT",
            "output",
            "atomic rename failed from " + tmp_output + " to " + target_output,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # Write layer timing bila diminta (--layer-timing)
    if target_timing.byte_length() > 0:
        var tmp_timing = String(target_timing, ".tmp.", run_id)
        try:
            var ft = open(tmp_timing, "w")
            var timing_json = String(
                '{\n  "run_id": "', run_id, '",\n  "layer_timing": [\n'
            )
            for li in range(len(layer_timings)):
                ref lt = layer_timings[li]
                if li > 0:
                    timing_json += ",\n"
                timing_json += String(
                    '    {\n      "layer": ',
                    lt.layer,
                    ',\n      "pread_sec": ',
                    lt.pread_sec,
                    ',\n      "attention_sec": ',
                    lt.attention_sec,
                    ',\n      "moe_sec": ',
                    lt.moe_sec,
                    ',\n      "total_sec": ',
                    lt.total_sec,
                    "\n    }",
                )
            timing_json += String(
                '\n  ],\n  "total_layer_forward_sec": ',
                layer_forward_sec,
                "\n}\n",
            )
            ft.write(timing_json)
            ft.close()
            _ = c_rename(tmp_timing, target_timing)
        except:
            fail_m4(
                "M4_ERR_OUTPUT",
                "output",
                "failed writing layer timing",
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

    var t_write_end = perf_counter_ns()
    var write_sec = Float64(t_write_end - t_write_start) / 1e9

    # Cleanup temp resources
    cleanup_run_resources(run_dir, tmp_files)

    # Metrics
    var t_end = perf_counter_ns()
    var walltime_sec = Float64(t_end - t_start) / 1e9
    var vmhwm_bytes = get_vmhwm_bytes()
    if vmhwm_bytes == 0:
        vmhwm_bytes = (
            telemetry.resident_target_bytes + telemetry.conversion_buffer_bytes
        )
    var phys_read = get_proc_io_read_bytes()
    var cgroup_peak = get_cgroup_peak_bytes()
    var cgroup_oom = get_cgroup_oom_kills()
    var total_logical_bytes = (
        cache.total_header_bytes + telemetry.logical_bytes_read
    )

    # Invariant: Leak (VmHWM > 5 GiB atau oom_kill > 0) = FAIL
    if (vmhwm_bytes > 5368709120) or (cgroup_oom > 0):
        fail_m4(
            "M4_ERR_MEMORY",
            "layer_forward",
            "Memory limit exceeded: VmHWM > 5 GiB or oom_kill > 0",
            String(
                '{"vmhwm_bytes":',
                vmhwm_bytes,
                ',"cgroup_oom_kills":',
                cgroup_oom,
                "}",
            ),
        )

    # Success JSON ke stdout
    var out_json = String(
        '{\n  "status": "success",\n  "run_id": "',
        run_id,
        '",\n  "model": "qwen1.5-moe-a2.7b-chat",\n  "num_tokens": ',
        String(num_tokens),
        ',\n  "num_layers": ',
        String(cfg.num_hidden_layers),
        ',\n  "logits_path": "',
        json_escape(target_output),
        '",\n  "metrics": {\n    "walltime_sec": ',
        String(walltime_sec),
        ',\n    "vmhwm_bytes": ',
        String(vmhwm_bytes),
        ',\n    "logical_bytes_read": ',
        String(total_logical_bytes),
        ',\n    "physical_read_bytes": ',
        String(phys_read),
        ',\n    "cgroup_peak_bytes": ',
        String(cgroup_peak),
        ',\n    "cgroup_oom_kills": ',
        String(cgroup_oom),
        ',\n    "phases": {\n      "index_load_sec": ',
        String(index_load_sec),
        ',\n      "embedding_sec": ',
        String(embedding_sec),
        ',\n      "layer_forward_sec": ',
        String(layer_forward_sec),
        ',\n      "final_norm_sec": ',
        String(final_norm_sec),
        ',\n      "lm_head_sec": ',
        String(lm_head_sec),
        ',\n      "write_sec": ',
        String(write_sec),
        "\n    }\n  }\n}",
    )
    print(out_json)
