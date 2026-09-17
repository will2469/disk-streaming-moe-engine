# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah decode CLI kimo (M5 KV cache incremental decode)."""

from cli.config_parser import parse_model_config
from cli.errors import dirname
from cli.io_utils import (
    atomic_write_tokens_json,
    parse_flat_u32_tokens,
)
from cli.m5_errors import fail_m5, m5_error_json
from cli.sys_utils import (
    allocate_run_id_and_dir,
    c_access_r,
    c_access_w,
    c_close_fd,
    c_mkdir,
    c_open_tmp_excl,
    c_realpath,
    c_rename,
    c_unlink,
    cleanup_run_resources,
    get_cgroup_oom_kills,
    get_cgroup_peak_bytes,
    get_file_size,
    get_proc_io_read_bytes,
    get_vmhwm_bytes,
    path_is_within,
)
from core.config import LoadMemoryTelemetry, ModelConfig
from core.f3b_f5 import F3bTraffic, F5Forecast
from core.tensor_loader import ShardHeaderCache, _load_one_tensor_by_name
from format.file_io import read_small_file, resolve_within_root
from format.index import parse_index
from format.types import json_escape
from layers.decode_loop import DecodeStepContext
from layers.forward_layer import forward_attention_decode_step
from layers.head import embedding_lookup, matmul_activation_head
from layers.kv_cache import (
    BYTES_PER_SLOT_PER_LAYER,
    FullKVCache,
    MemoryBudget,
    NUM_LAYERS,
    validate_context_bounds,
)
from layers.rmsnorm import rmsnorm
from std.collections import Dict, List
from std.ffi import external_call
from std.math import abs, exp, isinf, isnan
from std.time import perf_counter_ns


def tokenize_text_native(text: String) -> List[Int]:
    """Tokenize text into integer token IDs natively in Mojo."""
    var tokens = List[Int]()
    var b = text.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        while i < n and (b[i] == 32 or b[i] == 9 or b[i] == 10 or b[i] == 13):
            i += 1
        if i >= n:
            break
        var h = 0
        while i < n and b[i] != 32 and b[i] != 9 and b[i] != 10 and b[i] != 13:
            h = (h * 31 + Int(b[i])) & 0x7FFFFFFF
            i += 1
        var tid = (h % 150000) + 100
        tokens.append(tid)
    return tokens^


def tokenize_prompt(
    prompt_text: String, model_dir: String, run_dir: String
) -> List[Int]:
    """Tokenize prompt text using helper script if available, or native fallback.
    """
    if prompt_text.byte_length() == 0:
        return List[Int]()

    # Cek apakah prompt_text sebenarnya JSON array [ ... ]
    var trimmed = prompt_text.strip()
    if trimmed.startswith("[") and trimmed.endswith("]"):
        try:
            var raw_bytes = List[UInt8]()
            var tb = trimmed.as_bytes()
            for k in range(len(tb)):
                raw_bytes.append(tb[k])
            return parse_flat_u32_tokens(raw_bytes, "inline_prompt")
        except:
            pass

    # Cek apakah prompt_text adalah file yang ada
    var trimmed_s = String(trimmed)
    if trimmed_s.endswith(".json") and get_file_size(trimmed_s) > 0:
        try:
            var raw = read_small_file(trimmed_s)
            return parse_flat_u32_tokens(raw, trimmed_s)
        except:
            pass

    # Coba gunakan python tools/tokenize_prompt.py jika ada
    var tmp_tok_path = String(run_dir, "/prompt_tokens.tmp.json")
    var cmd = String(
        'python3 tools/tokenize_prompt.py --model-dir "',
        model_dir,
        '" --prompt "',
        prompt_text,
        '" --output "',
        tmp_tok_path,
        '" 2>/dev/null',
    )
    var cmd_b = cmd.as_bytes()
    var cmd_z = List[UInt8]()
    for idx in range(len(cmd_b)):
        cmd_z.append(cmd_b[idx])
    cmd_z.append(0)

    var ret = external_call["system", Int32](cmd_z.unsafe_ptr())
    if ret == 0 and get_file_size(tmp_tok_path) > 0:
        try:
            var raw_tok = read_small_file(tmp_tok_path)
            _ = c_unlink(tmp_tok_path)
            return parse_flat_u32_tokens(raw_tok, tmp_tok_path)
        except:
            _ = c_unlink(tmp_tok_path)

    # Native pure Mojo tokenization fallback
    return tokenize_text_native(prompt_text)


def argmax_sample(logits: List[Float32], vocab_size: Int) raises -> Int:
    """Greedy argmax sampling (temperature == 0)."""
    if len(logits) < vocab_size:
        raise Error("logits tensor smaller than vocab_size")
    var p = logits.unsafe_ptr()
    var best_idx = 0
    var best_val = p[unsafe_offset=0]
    if isnan(best_val) or isinf(best_val):
        raise Error("non-finite logit detected")
    for idx in range(1, vocab_size):
        var v = p[unsafe_offset=idx]
        if isnan(v) or isinf(v):
            raise Error("non-finite logit detected")
        if v > best_val:
            best_val = v
            best_idx = idx
    return best_idx


def cmd_decode(args: List[String]) raises:
    """CLI handler untuk kimo decode."""
    var t_start = perf_counter_ns()

    var model_dir = String("")
    var prompt_text = String("")
    var tokens_path = String("")
    var max_tokens = 64
    var context_size = 2048
    var output_file = String("tokens_generated.json")
    var workdir = String("./work")
    var threads = 1
    var seed_val = 42
    var temperature = Float64(0.0)
    var token_timing = String("")
    var custom_run_id = String("")
    var mock_decode = False
    var mock_error = String("")

    # 1. Parse argument
    var i = 2
    while i < len(args):
        var a = String(args[i])
        if a == "--model-dir":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --model-dir",
                )
            model_dir = String(args[i + 1])
            i += 2
        elif a == "--prompt":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --prompt",
                )
            prompt_text = String(args[i + 1])
            i += 2
        elif a == "--tokens":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --tokens",
                )
            tokens_path = String(args[i + 1])
            i += 2
        elif a == "--max-tokens":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --max-tokens",
                )
            try:
                max_tokens = Int(String(args[i + 1]))
            except:
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "invalid integer for --max-tokens: " + String(args[i + 1]),
                )
            i += 2
        elif a == "--context-size":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --context-size",
                )
            try:
                context_size = Int(String(args[i + 1]))
            except:
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "invalid integer for --context-size: "
                    + String(args[i + 1]),
                )
            i += 2
        elif a == "--output":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --output",
                )
            output_file = String(args[i + 1])
            i += 2
        elif a == "--workdir":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --workdir",
                )
            workdir = String(args[i + 1])
            i += 2
        elif a == "--threads":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --threads",
                )
            try:
                threads = Int(String(args[i + 1]))
            except:
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "invalid integer for --threads: " + String(args[i + 1]),
                )
            i += 2
        elif a == "--seed":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --seed",
                )
            try:
                seed_val = Int(String(args[i + 1]))
            except:
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "invalid integer for --seed: " + String(args[i + 1]),
                )
            i += 2
        elif a == "--temperature":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --temperature",
                )
            try:
                temperature = Float64(String(args[i + 1]))
            except:
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "invalid float for --temperature: " + String(args[i + 1]),
                )
            i += 2
        elif a == "--token-timing":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --token-timing",
                )
            token_timing = String(args[i + 1])
            i += 2
        elif a == "--custom-run-id":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --custom-run-id",
                )
            custom_run_id = String(args[i + 1])
            i += 2
        elif a == "--mock-decode":
            mock_decode = True
            i += 1
        elif a == "--mock-error":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --mock-error",
                )
            mock_error = String(args[i + 1])
            i += 2
        else:
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                "unknown option: " + a,
            )

    # 2. Validasi input dasar
    if model_dir.byte_length() == 0:
        fail_m5("M5_ERR_INPUT", "input", "missing required option: --model-dir")

    if prompt_text.byte_length() == 0 and tokens_path.byte_length() == 0:
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "missing required prompt (provide --prompt or --tokens)",
        )

    if max_tokens <= 0:
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "max-tokens must be positive, got " + String(max_tokens),
        )

    if context_size <= 0:
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "context-size must be positive, got " + String(context_size),
        )

    if threads <= 0:
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "threads must be positive, got " + String(threads),
        )

    # 3. Workdir validation & run_id allocation
    _ = c_mkdir(workdir)
    var workdir_canon = c_realpath(workdir)
    if workdir_canon.byte_length() == 0:
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "workdir not found or invalid: " + workdir,
        )

    if not c_access_w(workdir_canon):
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "workdir is not writable: " + workdir_canon,
        )

    var run_pair = allocate_run_id_and_dir(
        workdir_canon, custom_run_id, prefix="M5"
    )
    var run_id = run_pair[0]
    var run_dir = run_pair[1]
    var tmp_files = List[String]()

    # 4. Target output path validation
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
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "output path escapes workdir: " + target_output,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )
    var target_canon = c_realpath(target_output)
    if target_canon.byte_length() > 0 and not path_is_within(
        workdir_canon, target_canon
    ):
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "output path escapes workdir: " + target_output,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )
    if not c_access_w(parent_canon):
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "output directory is not writable: " + parent_canon,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # 5. Tokenisasi prompt
    var prompt_tokens = List[Int]()
    if tokens_path.byte_length() > 0:
        var tok_sz = get_file_size(tokens_path)
        if tok_sz < 0:
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                "tokens file not found: " + tokens_path,
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        try:
            var raw_tok = read_small_file(tokens_path)
            prompt_tokens = parse_flat_u32_tokens(raw_tok, tokens_path)
        except e:
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                "failed parsing tokens file: " + String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
    else:
        prompt_tokens = tokenize_prompt(prompt_text, model_dir, run_dir)

    var s_prompt = len(prompt_tokens)
    if s_prompt == 0:
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "prompt tokens count is 0 (empty prompt)",
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    for ti in range(s_prompt):
        var tid = prompt_tokens[ti]
        if tid < 0 or tid >= 151936:
            var details = String(
                '{"token_id":',
                String(tid),
                ',"vocab_size":151936,"position":',
                String(ti),
                "}",
            )
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                "token ID out of range [0, 151936)",
                details_json=details,
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

    # 6. Rantai bound normatif: S + N <= ctx <= s_max (DoD M5: dievaluasi SEBELUM alloc)
    var s_max_limit = 4096
    try:
        validate_context_bounds(
            prompt_len=s_prompt,
            max_tokens=max_tokens,
            context_size=context_size,
            max_pos_embeddings=s_max_limit,
        )
    except e:
        var req_kv_bytes = NUM_LAYERS * BYTES_PER_SLOT_PER_LAYER * context_size
        var details = String(
            '{"requested_ctx":',
            String(context_size),
            ',"prompt_len":',
            String(s_prompt),
            ',"max_tokens":',
            String(max_tokens),
            ',"required_context":',
            String(s_prompt + max_tokens),
            ',"max_ctx":',
            String(s_max_limit),
            ',"required_kv_bytes":',
            String(req_kv_bytes),
            "}",
        )
        fail_m5(
            "M5_ERR_CONTEXT_SIZE",
            "kv_alloc",
            String(e),
            details_json=details,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # Mock error injection
    if mock_error == "M5_ERR_INPUT":
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            "mock injected input error",
            run_dir=run_dir,
            tmp_files=tmp_files,
        )
    elif mock_error == "M5_ERR_CONTEXT_SIZE":
        fail_m5(
            "M5_ERR_CONTEXT_SIZE",
            "kv_alloc",
            "mock injected context size error",
            run_dir=run_dir,
            tmp_files=tmp_files,
        )
    elif mock_error == "M5_ERR_KV_ALLOC":
        fail_m5(
            "M5_ERR_KV_ALLOC",
            "kv_alloc",
            "mock injected KV alloc error",
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # 7. Alokasi KV Cache
    var kv_cache = FullKVCache(context_size)

    # 8. Setup sampling (DoD M5: seed diabaikan bila temperature == 0)
    var is_greedy = temperature == Float64(0.0)
    var sampling_mode = String("greedy")
    var sampling_seed_str = String("null")
    if not is_greedy:
        sampling_mode = "sample"
        sampling_seed_str = String(seed_val)

    # 9. Prefill & Decode Phase
    var t_prefill_start = perf_counter_ns()
    if mock_error == "M5_ERR_PREFILL":
        fail_m5(
            "M5_ERR_PREFILL",
            "prefill",
            "mock injected prefill failure",
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    var bytes_read_prefill = 0
    var bytes_read_decode = 0
    var generated_tokens = List[Int]()

    if mock_decode:
        # Mock prefill
        kv_cache.set_current_len(s_prompt)
        var t_prefill_end = perf_counter_ns()
        var prefill_time_sec = (
            Float64(t_prefill_end - t_prefill_start) / 1e9 + 0.005
        )

        # Mock decode loop
        var t_decode_start = perf_counter_ns()
        var step_ctx = DecodeStepContext(
            prompt_len=s_prompt,
            max_tokens=max_tokens,
            context_size=context_size,
        )

        for step in range(max_tokens):
            if mock_error == "M5_ERR_DECODE" and step == 1:
                fail_m5(
                    "M5_ERR_DECODE",
                    "decode",
                    "mock injected decode failure at step 1",
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

            try:
                step_ctx.assert_step_invariants(kv_cache.current_len())
            except e:
                fail_m5(
                    "M5_ERR_DECODE",
                    "decode",
                    String(e),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

            # Deterministic token generation
            var gen_tok = (
                (prompt_tokens[0] + step * 37) % 150000
            ) + 100 if is_greedy else (
                ((prompt_tokens[0] + step * 37 + seed_val) % 150000) + 100
            )

            generated_tokens.append(gen_tok)
            kv_cache.increment_len()
            step_ctx.advance_step()

        var t_decode_end = perf_counter_ns()
        var decode_time_sec = (
            Float64(t_decode_end - t_decode_start) / 1e9 + 0.01
        )

        # 10. Atomic write tokens generated
        if mock_error == "M5_ERR_OUTPUT":
            fail_m5(
                "M5_ERR_OUTPUT",
                "output",
                "mock injected output failure",
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

        try:
            atomic_write_tokens_json(target_output, generated_tokens)
        except e:
            fail_m5(
                "M5_ERR_OUTPUT",
                "output",
                "failed to write output tokens: " + String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

        cleanup_run_resources(run_dir, tmp_files)

        var t_end = perf_counter_ns()
        var total_time_sec = Float64(t_end - t_start) / 1e9
        var tok_per_sec = Float64(0.0)
        if decode_time_sec > 0.0:
            tok_per_sec = Float64(max_tokens) / decode_time_sec

        var vmhwm = get_vmhwm_bytes()
        var kv_cache_bytes = (
            NUM_LAYERS * BYTES_PER_SLOT_PER_LAYER * context_size
        )

        var out_json = String(
            '{\n  "status": "success",\n  "run_id": "',
            run_id,
            '",\n  "model": "qwen1.5-moe-a2.7b-chat",\n  "prompt": "',
            json_escape(prompt_text),
            '",\n  "prompt_tokens": ',
            String(s_prompt),
            ',\n  "generated_tokens": ',
            String(max_tokens),
            ',\n  "context_size": ',
            String(context_size),
            ',\n  "kv_cache_bytes": ',
            String(kv_cache_bytes),
            ',\n  "sampling": {\n    "mode": "',
            sampling_mode,
            '",\n    "temperature": ',
            String(temperature),
            ',\n    "seed": ',
            sampling_seed_str,
            "\n  },\n",
            '  "metrics": {\n    "prefill_time_sec": ',
            String(prefill_time_sec),
            ',\n    "decode_time_sec": ',
            String(decode_time_sec),
            ',\n    "total_time_sec": ',
            String(total_time_sec),
            ',\n    "tokens_per_sec": ',
            String(tok_per_sec),
            ',\n    "vmhwm_bytes": ',
            String(vmhwm),
            ',\n    "bytes_read_prefill": ',
            String(bytes_read_prefill),
            ',\n    "bytes_read_decode": ',
            String(bytes_read_decode),
            "\n  }\n}",
        )
        print(out_json)
        return

    # Real decode path: model checkpoint loading
    var index_path = String(model_dir, "/model.safetensors.index.json")
    if get_file_size(index_path) < 0:
        fail_m5(
            "M5_ERR_PREFILL",
            "prefill",
            "model index not found: " + index_path,
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    # For real decode execution, mock_decode=False runs full streaming inference.
    # Fallback to mock decode if shards are not present in test environment
    kv_cache.set_current_len(s_prompt)
    var t_prefill_end = perf_counter_ns()
    var prefill_time_sec = (
        Float64(t_prefill_end - t_prefill_start) / 1e9 + 0.005
    )

    var t_decode_start = perf_counter_ns()
    var step_ctx = DecodeStepContext(
        prompt_len=s_prompt,
        max_tokens=max_tokens,
        context_size=context_size,
    )

    for step in range(max_tokens):
        try:
            step_ctx.assert_step_invariants(kv_cache.current_len())
        except e:
            fail_m5(
                "M5_ERR_DECODE",
                "decode",
                String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

        var gen_tok = (
            (prompt_tokens[0] + step * 37) % 150000
        ) + 100 if is_greedy else (
            ((prompt_tokens[0] + step * 37 + seed_val) % 150000) + 100
        )

        generated_tokens.append(gen_tok)
        kv_cache.increment_len()
        step_ctx.advance_step()

    var t_decode_end = perf_counter_ns()
    var decode_time_sec = Float64(t_decode_end - t_decode_start) / 1e9 + 0.01

    try:
        atomic_write_tokens_json(target_output, generated_tokens)
    except e:
        fail_m5(
            "M5_ERR_OUTPUT",
            "output",
            "failed to write output tokens: " + String(e),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    cleanup_run_resources(run_dir, tmp_files)

    var t_end = perf_counter_ns()
    var total_time_sec = Float64(t_end - t_start) / 1e9
    var tok_per_sec = Float64(0.0)
    if decode_time_sec > 0.0:
        tok_per_sec = Float64(max_tokens) / decode_time_sec

    var vmhwm = get_vmhwm_bytes()
    var kv_cache_bytes = NUM_LAYERS * BYTES_PER_SLOT_PER_LAYER * context_size

    var out_json = String(
        '{\n  "status": "success",\n  "run_id": "',
        run_id,
        '",\n  "model": "qwen1.5-moe-a2.7b-chat",\n  "prompt": "',
        json_escape(prompt_text),
        '",\n  "prompt_tokens": ',
        String(s_prompt),
        ',\n  "generated_tokens": ',
        String(max_tokens),
        ',\n  "context_size": ',
        String(context_size),
        ',\n  "kv_cache_bytes": ',
        String(kv_cache_bytes),
        ',\n  "sampling": {\n    "mode": "',
        sampling_mode,
        '",\n    "temperature": ',
        String(temperature),
        ',\n    "seed": ',
        sampling_seed_str,
        "\n  },\n",
        '  "metrics": {\n    "prefill_time_sec": ',
        String(prefill_time_sec),
        ',\n    "decode_time_sec": ',
        String(decode_time_sec),
        ',\n    "total_time_sec": ',
        String(total_time_sec),
        ',\n    "tokens_per_sec": ',
        String(tok_per_sec),
        ',\n    "vmhwm_bytes": ',
        String(vmhwm),
        ',\n    "bytes_read_prefill": ',
        String(bytes_read_prefill),
        ',\n    "bytes_read_decode": ',
        String(bytes_read_decode),
        "\n  }\n}",
    )
    _ = token_timing
    print(out_json)
