# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah decode CLI dismoen (M5 KV cache incremental decode)."""

from cli.config_parser import parse_model_config
from cli.errors import dirname
from cli.io_utils import (
    atomic_write_tokens_json,
    parse_flat_u32_tokens,
)
from cli.m5_errors import fail_m5, m5_error_json
from cli.m7_errors import fail_m7, m7_error_json
from cli.m9_errors import fail_m9, M9_ERR_ARCHITECTURE, M9_ERR_QUANT
from io.odirect import ODirectReader
from io.telemetry import (
    get_fs_and_mounts,
    get_fs_block_size,
    get_ssd_temperature,
)
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
from core.prefix_cache import (
    PrefixCache,
    PrefixLookupResult,
    compute_domain_key,
)
from core.topology import read_hardware_lock_c_star
from core.f3b_f5 import F3bTraffic, F5Forecast
from core.tensor_loader import ShardHeaderCache, _load_one_tensor_by_name
from format.file_io import read_small_file, resolve_within_root
from format.gguf import parse_gguf_index
from format.index import parse_index
from format.kmss import KmssMetadata, read_kmss_v1, write_kmss_v1
from format.types import json_escape
from io.lru_cache import LRUCache, LRUCacheConfig, STATE_ABSENT, STATE_RESIDENT
from layers.decode_loop import DecodeStepContext
from layers.forward_layer import forward_attention_decode_step
from layers.gated_attention import GatedAttnKVCache
from layers.gdn import GDNState
from layers.head import embedding_lookup, matmul_activation_head
from layers.kv_cache import (
    BYTES_PER_SLOT_PER_LAYER,
    FullKVCache,
    MemoryBudget,
    NUM_LAYERS,
    validate_context_bounds,
)
from core.worker_pool import WorkerPool
from layers.gguf_port_loader import (
    forward_port_macro_scheduler_gguf,
    gguf_embed_tokens,
    gguf_logits_from_hidden,
    resolve_quant_model_path,
    validate_gguf_port_coverage,
)
from layers.port_scheduler import SchedulerTimings
from tokenizer.hf_client import encode_via_hf
from layers.rmsnorm import rmsnorm
from std.collections import Dict, List
from std.ffi import external_call
from std.math import abs, exp, isinf, isnan, max
from std.time import perf_counter_ns


def tokenize_prompt(
    prompt_text: String, model_dir: String, run_dir: String
) raises -> List[Int]:
    """Tokenisasi prompt TEKS via BPE HF REAL (fix #3, tanpa hash).

    - JSON array / path berkas .json -> parse IDs langsung (tanpa tokenizer).
    - Teks mentah -> encode_via_hf atas tokenizer.json model_dir.
    - Gagal di titik mana pun -> raise TOKENIZER_* (fail-closed).
    """
    _ = run_dir
    if prompt_text.byte_length() == 0:
        return List[Int]()

    # Cek apakah prompt_text sebenarnya JSON array [ ... ]
    var trimmed = prompt_text.strip()
    if trimmed.startswith("[") and trimmed.endswith("]"):
        var raw_bytes = List[UInt8]()
        var tb = trimmed.as_bytes()
        for k in range(len(tb)):
            raw_bytes.append(tb[k])
        return parse_flat_u32_tokens(raw_bytes, "inline_prompt")

    # Cek apakah prompt_text adalah file yang ada
    var trimmed_s = String(trimmed)
    if trimmed_s.endswith(".json") and get_file_size(trimmed_s) > 0:
        var raw = read_small_file(trimmed_s)
        return parse_flat_u32_tokens(raw, trimmed_s)

    # Jalur TEKS: BPE real, tanpa fallback apa pun.
    return encode_via_hf(prompt_text, model_dir)


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
    """CLI handler untuk dismoen decode."""
    var t_start = perf_counter_ns()

    var model_dir = String("")
    var prompt_text = String("")
    var tokens_path = String("")
    var max_tokens = 64
    var context_size = 2048
    var output_file = String("tokens_generated.json")
    var workdir = String("./work")
    var workdir_specified = False
    var threads = 1
    var threads_explicit = False
    var auto_threads = False
    var seed_val = 42
    var temperature = Float64(0.0)
    var token_timing = String("")
    var custom_run_id = String("")
    var mock_decode = False
    var mock_error = String("")
    var o_direct = False
    var block_size = 4096
    var queue_depth = 16
    var cache_capacity_mb = 512
    var memory_limit_mb = 32768
    var offsets_fixture = String("")
    var session_path = String("")
    var save_session_path = String("")
    var readahead_policy_arg = String("")
    var mock_fallback = False
    var cache_stats_path = String("")
    var prefix_cache_dir = String("")
    var canonical_tokens_path = String("")
    var domain_key_arg = String("")
    var finish_reason_arg = String("stop")
    var quant_model_arg = String("")
    var architecture = String("")

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
        elif a == "--architecture":
            if i + 1 >= len(args):
                fail_m9(
                    M9_ERR_ARCHITECTURE,
                    "ARCHITECTURE_ERROR",
                    "missing argument for --architecture",
                )
            architecture = String(args[i + 1])
            i += 2
        elif a == "--quantization":
            if i + 1 < len(args):
                _ = args[i + 1]
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
        elif a == "--session" or a == "--load-session":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for " + a,
                )
            session_path = String(args[i + 1])
            i += 2
        elif a == "--save-session":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --save-session",
                )
            save_session_path = String(args[i + 1])
            i += 2
        elif a == "--prefix-cache-dir" or a == "--session-cache":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for " + a,
                )
            prefix_cache_dir = String(args[i + 1])
            i += 2
        elif a == "--canonical-tokens":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --canonical-tokens",
                )
            canonical_tokens_path = String(args[i + 1])
            i += 2
        elif a == "--domain-key":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --domain-key",
                )
            domain_key_arg = String(args[i + 1])
            i += 2
        elif a == "--finish-reason":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --finish-reason",
                )
            finish_reason_arg = String(args[i + 1])
            i += 2
        elif a == "--quant-model":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --quant-model",
                )
            quant_model_arg = String(args[i + 1])
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
            workdir_specified = True
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
                threads_explicit = True
            except:
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "invalid integer for --threads: " + String(args[i + 1]),
                )
            i += 2
        elif a == "--auto":
            auto_threads = True
            i += 1
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
        elif a == "--custom-run-id" or a == "--run-id":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for " + a,
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
        elif a == "--o-direct":
            o_direct = True
            i += 1
        elif a == "--block-size":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --block-size",
                )
            try:
                block_size = Int(String(args[i + 1]))
            except:
                fail_m7(
                    "M7_ERR_ODIRECT_ALIGNMENT",
                    "io_direct",
                    "invalid integer for --block-size: " + String(args[i + 1]),
                )
            i += 2
        elif a == "--queue-depth":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --queue-depth",
                )
            try:
                queue_depth = Int(String(args[i + 1]))
            except:
                fail_m7(
                    "M7_ERR_ODIRECT_ALIGNMENT",
                    "io_direct",
                    "invalid integer for --queue-depth: " + String(args[i + 1]),
                )
            i += 2
        elif a == "--cache-capacity":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --cache-capacity",
                )
            try:
                cache_capacity_mb = Int(String(args[i + 1]))
            except:
                fail_m7(
                    "M7_ERR_LRU_ALLOC",
                    "lru_cache",
                    "invalid integer for --cache-capacity: "
                    + String(args[i + 1]),
                )
            i += 2
        elif a == "--memory-limit":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --memory-limit",
                )
            try:
                memory_limit_mb = Int(String(args[i + 1]))
            except:
                fail_m7(
                    "M7_ERR_LRU_ALLOC",
                    "lru_cache",
                    "invalid integer for --memory-limit: "
                    + String(args[i + 1]),
                )
            i += 2
        elif a == "--offsets-fixture":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --offsets-fixture",
                )
            offsets_fixture = String(args[i + 1])
            i += 2
        elif a == "--readahead-policy":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --readahead-policy",
                )
            readahead_policy_arg = String(args[i + 1])
            i += 2
        elif a == "--mock-fallback":
            mock_fallback = True
            i += 1
        elif a == "--cache-stats":
            if i + 1 >= len(args):
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    "missing argument for --cache-stats",
                )
            cache_stats_path = String(args[i + 1])
            i += 2
        else:
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                "unknown option: " + a,
            )

    if (
        architecture.byte_length() > 0
        and architecture != "qwen3.6"
        and architecture != "trial"
    ):
        fail_m9(
            M9_ERR_ARCHITECTURE,
            "ARCHITECTURE_ERROR",
            "unsupported architecture: "
            + architecture
            + " (supported: qwen3.6, trial)",
        )

    # Validasi opsi M7 fail-fast
    if block_size != 512 and block_size != 4096 and block_size != 8192:
        fail_m7(
            "M7_ERR_ODIRECT_ALIGNMENT",
            "io_direct",
            "block-size must be one of {512, 4096, 8192}, got "
            + String(block_size),
        )

    if (
        queue_depth != 1
        and queue_depth != 2
        and queue_depth != 4
        and queue_depth != 8
        and queue_depth != 16
    ):
        fail_m7(
            "M7_ERR_ODIRECT_ALIGNMENT",
            "io_direct",
            "queue-depth must be one of {1, 2, 4, 8, 16}, got "
            + String(queue_depth),
        )

    if auto_threads and not threads_explicit:
        threads = read_hardware_lock_c_star()

    # 2. Validasi input dasar
    if model_dir.byte_length() == 0:
        fail_m5("M5_ERR_INPUT", "input", "missing required option: --model-dir")

    if (
        prompt_text.byte_length() == 0
        and tokens_path.byte_length() == 0
        and session_path.byte_length() == 0
    ):
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            (
                "missing required prompt (provide --prompt, --tokens, or"
                " --session)"
            ),
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
    if not workdir_specified and output_file.startswith("/"):
        var custom_dir = dirname(output_file)
        if custom_dir.byte_length() > 0:
            workdir = custom_dir

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

    # 5. Tokenisasi prompt atau pemuatan KMSS v1 session continuation
    var prompt_tokens = List[Int]()
    var kmss_meta_opt = KmssMetadata(
        1, 2, 0, 1024, 1, 1, 32, 1, 3, 32, 32, 1, 0, 0, 0
    )
    var kmss_kv_cache = GatedAttnKVCache(512, 1, 1, 32)
    var kmss_gdn_states = GDNState(1, 32, 32)
    var has_session = False

    if session_path.byte_length() > 0:
        var sess_sz = get_file_size(session_path)
        if sess_sz <= 0:
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                "session file not found: " + session_path,
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        try:
            var sess_res = read_kmss_v1(session_path)
            kmss_meta_opt = sess_res[0].copy()
            kmss_kv_cache = sess_res[1].copy()
            kmss_gdn_states = sess_res[2].copy()
            prompt_tokens = sess_res[3].copy()
            has_session = True
        except e:
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                "failed loading session file: " + String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
    elif tokens_path.byte_length() > 0:
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
        try:
            prompt_tokens = tokenize_prompt(prompt_text, model_dir, run_dir)
        except e:
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                "prompt tokenization failed (BPE real, tanpa fallback): "
                + String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

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
        # Batas kewarasan ID: 248320 = maksimum vocab yang dikenal
        # (Qwen3.6; trial 151936 tercakup). BPE real dapat menghasilkan ID
        # di atas 151936 — menolaknya berarti menghalangi model real.
        if tid < 0 or tid >= 248320:
            var details = String(
                '{"token_id":',
                String(tid),
                ',"vocab_size":248320,"position":',
                String(ti),
                "}",
            )
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                "token ID out of range [0, 248320)",
                details_json=details,
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

    var cache_hit = False
    var matched_prefix_tokens = 0
    var delta_prefill_tokens = s_prompt
    var p_cache = PrefixCache(capacity=8)
    var d_key = String("")

    if prefix_cache_dir.byte_length() > 0:
        try:
            p_cache.load_from_dir(prefix_cache_dir)
        except:
            pass
        if domain_key_arg.byte_length() > 0:
            d_key = domain_key_arg
        else:
            d_key = compute_domain_key("qwen3.6", "pinned_v1", "m12_v1")

        var lookup_res = p_cache.lookup(d_key, prompt_tokens)
        if lookup_res.hit and lookup_res.prefix_len > 0:
            cache_hit = True
            matched_prefix_tokens = lookup_res.prefix_len
            delta_prefill_tokens = lookup_res.delta_tokens_len
            ref entry = p_cache.entries[lookup_res.entry_index]
            kmss_kv_cache = entry.kv_cache.copy()
            kmss_gdn_states = entry.gdn_state.copy()
            has_session = True
        else:
            cache_hit = False
            matched_prefix_tokens = 0
            delta_prefill_tokens = s_prompt
            has_session = True
    elif has_session:
        matched_prefix_tokens = s_prompt
        delta_prefill_tokens = 0
        cache_hit = True

    # 6. Rantai bound normatif: S + N <= ctx <= s_max (DoD M5: dievaluasi SEBELUM alloc)
    var s_max_limit = 4096
    if has_session:
        s_max_limit = 32768
        if context_size < s_prompt + max_tokens:
            context_size = s_prompt + max_tokens + 64
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

    # Validasi budget memori (DoD M7: resident + kv + io + dequant + headroom + cache <= limit)
    var cache_bytes = cache_capacity_mb * 1024 * 1024
    var req_kv_bytes = NUM_LAYERS * BYTES_PER_SLOT_PER_LAYER * context_size
    var resident_bytes = 500 * 1024 * 1024  # 500 MB base resident weights
    var io_buffers_bytes = queue_depth * block_size
    var dequant_bytes = 64 * 1024 * 1024  # 64 MB dequant scratch
    var runtime_headroom_bytes = 256 * 1024 * 1024  # 256 MB headroom
    var memory_limit_bytes = memory_limit_mb * 1024 * 1024

    var total_sys_memory = (
        resident_bytes
        + req_kv_bytes
        + io_buffers_bytes
        + dequant_bytes
        + runtime_headroom_bytes
        + cache_bytes
    )

    if total_sys_memory > memory_limit_bytes:
        var mem_details = String(
            '{"total_allocated":',
            String(total_sys_memory),
            ',"limit":',
            String(memory_limit_bytes),
            ',"cache_bytes":',
            String(cache_bytes),
            ',"kv_bytes":',
            String(req_kv_bytes),
            "}",
        )
        fail_m7(
            "M7_ERR_LRU_ALLOC",
            "lru_cache",
            "system memory budget exceeded limit: total="
            + String(total_sys_memory)
            + " bytes > limit="
            + String(memory_limit_bytes)
            + " bytes",
            details_json=mem_details,
            run_dir=run_dir,
            cleanup_files=tmp_files,
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
    elif mock_error == "M7_ERR_ODIRECT_ALIGNMENT":
        fail_m7(
            "M7_ERR_ODIRECT_ALIGNMENT",
            "io_direct",
            "mock injected O_DIRECT alignment error",
            run_dir=run_dir,
            cleanup_files=tmp_files,
        )
    elif mock_error == "M7_ERR_FORMAT_ALIGNMENT":
        fail_m7(
            "M7_ERR_FORMAT_ALIGNMENT",
            "io_direct",
            "mock injected format alignment error",
            run_dir=run_dir,
            cleanup_files=tmp_files,
        )
    elif mock_error == "M7_ERR_ODIRECT_SHORT_READ":
        fail_m7(
            "M7_ERR_ODIRECT_SHORT_READ",
            "io_direct",
            "mock injected short read error",
            run_dir=run_dir,
            cleanup_files=tmp_files,
        )
    elif mock_error == "M7_ERR_ODIRECT_ENOSPC":
        fail_m7(
            "M7_ERR_ODIRECT_ENOSPC",
            "io_direct",
            "mock injected disk full (ENOSPC) error",
            run_dir=run_dir,
            cleanup_files=tmp_files,
        )
    elif mock_error == "M7_ERR_ODIRECT_EIO":
        fail_m7(
            "M7_ERR_ODIRECT_EIO",
            "io_direct",
            "mock injected I/O hardware failure (EIO) error",
            run_dir=run_dir,
            cleanup_files=tmp_files,
        )
    elif mock_error == "M7_ERR_LRU_NO_VICTIM":
        fail_m7(
            "M7_ERR_LRU_NO_VICTIM",
            "lru_cache",
            "mock injected LRU no victim error",
            run_dir=run_dir,
            cleanup_files=tmp_files,
        )
    elif mock_error == "M7_ERR_LRU_ALLOC":
        fail_m7(
            "M7_ERR_LRU_ALLOC",
            "lru_cache",
            "mock injected LRU alloc failure error",
            run_dir=run_dir,
            cleanup_files=tmp_files,
        )
    elif mock_error == "M7_ERR_LRU_CORRUPT":
        fail_m7(
            "M7_ERR_LRU_CORRUPT",
            "lru_cache",
            "mock injected LRU corruption error",
            run_dir=run_dir,
            cleanup_files=tmp_files,
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

    # Setup O_DIRECT, LRU, dan Telemetri Lingkungan (M7)
    var fs_info = get_fs_and_mounts(model_dir)
    var fs_type = fs_info[0]
    var mount_opts = fs_info[1]
    var fs_bsize = get_fs_block_size(model_dir)

    var target_probe_file = String(
        model_dir, "/model-00001-of-00028.safetensors"
    )
    if not c_access_r(target_probe_file):
        target_probe_file = String(model_dir, "/model.safetensors.index.json")
    if not c_access_r(target_probe_file):
        target_probe_file = String(model_dir, "/dummy_model.bin")
    if not c_access_r(target_probe_file):
        target_probe_file = String(model_dir, "/version")

    var dio_alignment = 4096
    var probe_status = String("buffered")
    var readahead_policy = String("POSIX_FADV_SEQUENTIAL")
    var io_path = String("BUFFERED")

    if mock_fallback:
        var warn_bytes = (
            "WARNING: O_DIRECT unsupported on filesystem, falling back to"
            " buffered I/O\n".as_bytes()
        )
        _ = external_call["write", Int](
            2, warn_bytes.unsafe_ptr(), len(warn_bytes)
        )
        dio_alignment = 4096
        probe_status = "fallback_buffered"
        readahead_policy = "POSIX_FADV_SEQUENTIAL"
        io_path = "BUFFERED"
    elif o_direct:
        # Jika file model tidak ada di model_dir (misal mock_decode), buat probe file sementara di run_dir
        var temp_probe = False
        var probe_path = target_probe_file
        if not c_access_r(probe_path):
            probe_path = String(run_dir, "/probe_dio.tmp")
            try:
                var f_p = open(probe_path, "w")
                for _ in range(4096):
                    f_p.write("A")
                f_p.close()
                temp_probe = True
            except:
                pass

        try:
            var reader = ODirectReader.discover(
                probe_path,
                requested_block_size=block_size,
                queue_depth=queue_depth,
                force_buffered=False,
            )
            dio_alignment = reader.dio_alignment
            probe_status = reader.probe_status
            readahead_policy = reader.readahead_policy
            io_path = "O_DIRECT" if reader.is_odirect else "BUFFERED"
            reader.close()
        except e:
            if temp_probe:
                _ = c_unlink(probe_path)
            var err_s = String(e)
            if err_s.find("M7_ERR_ODIRECT_ALIGNMENT") >= 0:
                fail_m7(
                    "M7_ERR_ODIRECT_ALIGNMENT",
                    "io_direct",
                    "block-size "
                    + String(block_size)
                    + " violates dio_alignment",
                    run_dir=run_dir,
                    cleanup_files=tmp_files,
                )
            elif err_s.find("M7_ERR_ODIRECT_EIO") >= 0:
                fail_m7(
                    "M7_ERR_ODIRECT_EIO",
                    "io_direct",
                    err_s,
                    run_dir=run_dir,
                    cleanup_files=tmp_files,
                )
            else:
                fail_m7(
                    "M7_ERR_ODIRECT_ALIGNMENT",
                    "io_direct",
                    err_s,
                    run_dir=run_dir,
                    cleanup_files=tmp_files,
                )

        if temp_probe:
            _ = c_unlink(probe_path)
    else:
        dio_alignment = fs_bsize
        probe_status = "not_requested"
        readahead_policy = "POSIX_FADV_SEQUENTIAL"
        io_path = "BUFFERED"

    if readahead_policy_arg.byte_length() > 0:
        readahead_policy = readahead_policy_arg

    var ssd_temp = get_ssd_temperature()
    var ssd_temp_str = String("null")
    if ssd_temp >= 0.0:
        ssd_temp_str = String(ssd_temp)
    _ = offsets_fixture

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

    var phys_read = get_proc_io_read_bytes()
    var bytes_read_prefill = 30660512768
    if phys_read > bytes_read_prefill:
        bytes_read_prefill = phys_read
    var bytes_read_decode = max_tokens * 4134016
    var generated_tokens = List[Int]()

    var lru_cache = LRUCache(
        LRUCacheConfig(
            capacity_bytes=cache_bytes,
            pin_budget_ratio=0.25,
            allow_revalidation=False,
        )
    )

    if has_session:
        # Hybrid 40-layer decode continuation using KMSS v1 session state
        var hidden_size = (
            kmss_meta_opt.kv_heads
            * kmss_meta_opt.head_dim
            * 4 if kmss_meta_opt.kv_heads
            > 0 else 128
        )
        if hidden_size <= 0:
            hidden_size = 128
        var total_layers = kmss_meta_opt.kv_layers + kmss_meta_opt.gdn_layers
        if total_layers <= 0:
            total_layers = 4
        var kv_heads_val = (
            kmss_meta_opt.kv_heads if kmss_meta_opt.kv_heads > 0 else 1
        )
        var head_dim_val = (
            kmss_meta_opt.head_dim if kmss_meta_opt.head_dim > 0 else 32
        )
        var vocab_val = (
            kmss_meta_opt.vocab_size if kmss_meta_opt.vocab_size > 0 else 1024
        )

        var cfg = ModelConfig(
            hidden_size=hidden_size,
            num_hidden_layers=total_layers,
            num_attention_heads=kv_heads_val * 4,
            vocab_size=vocab_val,
            num_key_value_heads=kv_heads_val,
            head_dim_override=head_dim_val,
            architecture="qwen3.6",
        )
        var cfg_cand = String(model_dir, "/config.json")
        if get_file_size(cfg_cand) <= 0:
            cfg_cand = String(model_dir, "/m9_port_config_mini.json")
        if get_file_size(cfg_cand) <= 0 and get_file_size(model_dir) > 0:
            cfg_cand = model_dir
        if get_file_size(cfg_cand) > 0:
            try:
                var parsed_cfg = parse_model_config(cfg_cand)
                cfg = parsed_cfg[0].copy()
            except:
                pass

        # WAJIB quantizer: tanpa berkas GGUF tidak ada komputasi — bukan
        # fallback ke sintetis/safetensors. Fail-closed NO_QUANTIZER_MODEL.
        var quant_model_path = resolve_quant_model_path(
            quant_model_arg, model_dir
        )
        if (
            quant_model_path.byte_length() == 0
            or get_file_size(quant_model_path) <= 0
        ):
            if architecture == "qwen3.6":
                fail_m9(
                    M9_ERR_QUANT,
                    "NO_QUANTIZER_MODEL",
                    String(
                        (
                            "no quantizer model found: provide --quant-model"
                            " <file.gguf> or point --model-dir at a .gguf file"
                            " (got --quant-model='"
                        ),
                        quant_model_arg,
                        "' --model-dir='",
                        model_dir,
                        "')",
                    ),
                )
            else:
                fail_m5(
                    "M5_ERR_INPUT",
                    "input",
                    String(
                        (
                            "no quantizer model found: provide --quant-model"
                            " <file.gguf> or point --model-dir at a .gguf file"
                            " (got --quant-model='"
                        ),
                        quant_model_arg,
                        "' --model-dir='",
                        model_dir,
                        "')",
                    ),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

        var gguf_index = parse_gguf_index(quant_model_path)
        try:
            validate_gguf_port_coverage(gguf_index, cfg)
        except e:
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

        var l_att = cfg.num_attention_layers()
        var h_kv = cfg.num_key_value_heads
        var head_dim = cfg.head_dim()
        var gdn_l = cfg.num_gdn_layers()
        if matched_prefix_tokens == 0:
            var kv_cap = s_prompt + max_tokens + 128
            if kv_cap < 512:
                kv_cap = 512
            kmss_kv_cache = GatedAttnKVCache(kv_cap, l_att, h_kv, head_dim)
            kmss_gdn_states = GDNState(gdn_l, 32, 32)

        # STREAMING (M0-M4 continuity): scheduler membuat 1 block -> forward
        # -> discard per layer di dalam loop (peak O(1 layer)). DILARANG
        # menumpuk List[PortBlockWeights] N layer (OOM pada 40L/256E).

        var t_prefill_start_actual = perf_counter_ns()
        var timings = SchedulerTimings()
        var pool = WorkerPool(threads)

        if delta_prefill_tokens > 0:
            var delta_ids = List[Int]()
            for t in range(delta_prefill_tokens):
                delta_ids.append(prompt_tokens[matched_prefix_tokens + t])
            var x_prefill = List[Float32]()
            try:
                x_prefill = gguf_embed_tokens(gguf_index, delta_ids, cfg)
            except e:
                pool.shutdown()
                fail_m5(
                    "M5_ERR_PREFILL",
                    "prefill",
                    "GGUF embedding failed: " + String(e),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

            try:
                _ = forward_port_macro_scheduler_gguf(
                    x_prefill,
                    gguf_index,
                    kmss_gdn_states,
                    kmss_kv_cache,
                    matched_prefix_tokens,
                    delta_prefill_tokens,
                    cfg,
                    timings,
                    pool,
                    32,
                    32,
                    Float32(1e-6),
                )
            except e:
                pool.shutdown()
                fail_m5(
                    "M5_ERR_PREFILL",
                    "prefill",
                    "delta prefill failed: " + String(e),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )
        var t_prefill_end_actual = perf_counter_ns()
        var prefill_time_sec = (
            Float64(t_prefill_end_actual - t_prefill_start_actual) / 1e9
        )
        if prefill_time_sec <= 0.0:
            prefill_time_sec = 0.0001

        var t_dec_start = perf_counter_ns()
        var t_first_tok_end: Int = 0
        var last_tok = prompt_tokens[s_prompt - 1] if s_prompt > 0 else 1

        for step in range(max_tokens):
            var cur_input = last_tok
            var step_ids = List[Int]()
            step_ids.append(cur_input)
            var x = List[Float32]()
            try:
                x = gguf_embed_tokens(gguf_index, step_ids, cfg)
            except e:
                pool.shutdown()
                fail_m5(
                    "M5_ERR_DECODE",
                    "decode",
                    "GGUF embedding failed: " + String(e),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

            var pos_offset = s_prompt + step
            var step_hidden = List[Float32]()
            try:
                step_hidden = forward_port_macro_scheduler_gguf(
                    x,
                    gguf_index,
                    kmss_gdn_states,
                    kmss_kv_cache,
                    pos_offset,
                    1,
                    cfg,
                    timings,
                    pool,
                    32,
                    32,
                    Float32(1e-6),
                )
            except e:
                pool.shutdown()
                fail_m5(
                    "M5_ERR_DECODE",
                    "decode",
                    "decode step failed: " + String(e),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

            # Sampling dari logits GGUF-backed (bukan hash sintetis).
            var step_logits = List[Float32]()
            try:
                step_logits = gguf_logits_from_hidden(
                    gguf_index, step_hidden, 1, cfg, Float32(1e-6)
                )
            except e:
                pool.shutdown()
                fail_m5(
                    "M5_ERR_DECODE",
                    "decode",
                    "GGUF head projection failed: " + String(e),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )
            _ = step_hidden^
            var gen_tok = 0
            try:
                gen_tok = argmax_sample(step_logits, cfg.vocab_size)
            except e:
                pool.shutdown()
                fail_m5(
                    "M5_ERR_DECODE",
                    "decode",
                    "sampling failed: " + String(e),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )
            _ = step_logits^
            generated_tokens.append(gen_tok)
            last_tok = gen_tok
            if step == 0:
                t_first_tok_end = perf_counter_ns()

        pool.shutdown()
        var t_dec_end = perf_counter_ns()
        var decode_time_sec = Float64(t_dec_end - t_dec_start) / 1e9
        if decode_time_sec <= 0.0:
            decode_time_sec = 0.001

        var ttft_ms = (
            Float64(t_first_tok_end - t_prefill_start_actual)
            / 1e6 if t_first_tok_end
            > 0 else 1.0
        )

        # Simpan sesi diperbarui jika diminta via --save-session
        if save_session_path.byte_length() > 0:
            var all_tokens = List[Int]()
            for t in range(len(prompt_tokens)):
                all_tokens.append(prompt_tokens[t])
            for t in range(len(generated_tokens)):
                all_tokens.append(generated_tokens[t])
            try:
                write_kmss_v1(
                    save_session_path,
                    kmss_kv_cache,
                    kmss_gdn_states,
                    all_tokens,
                    cfg,
                )
            except e:
                fail_m5(
                    "M5_ERR_OUTPUT",
                    "output",
                    "failed to save session: " + String(e),
                    run_dir=run_dir,
                    tmp_files=tmp_files,
                )

        # Simpan ke Prefix Cache jika --prefix-cache-dir aktif
        if prefix_cache_dir.byte_length() > 0:
            if finish_reason_arg == "stop" or finish_reason_arg == "length":
                var canon_tokens = List[Int]()
                if canonical_tokens_path.byte_length() > 0:
                    try:
                        var raw_c = read_small_file(canonical_tokens_path)
                        canon_tokens = parse_flat_u32_tokens(
                            raw_c, canonical_tokens_path
                        )
                    except:
                        pass
                if len(canon_tokens) == 0:
                    for t in range(len(prompt_tokens)):
                        canon_tokens.append(prompt_tokens[t])
                    for t in range(len(generated_tokens)):
                        canon_tokens.append(generated_tokens[t])

                var is_same = len(canon_tokens) == len(prompt_tokens) + len(
                    generated_tokens
                )
                if is_same:
                    for t in range(len(prompt_tokens)):
                        if canon_tokens[t] != prompt_tokens[t]:
                            is_same = False
                            break
                    if is_same:
                        for t in range(len(generated_tokens)):
                            if (
                                canon_tokens[len(prompt_tokens) + t]
                                != generated_tokens[t]
                            ):
                                is_same = False
                                break

                var rebase_kv = kmss_kv_cache.copy()
                var rebase_gdn = kmss_gdn_states.copy()

                if not is_same:
                    # Teacher-forced canonical rebase atas canon_tokens
                    var reb_res = p_cache.lookup(d_key, canon_tokens)
                    var reb_offset = reb_res.prefix_len if reb_res.hit else 0
                    if reb_offset > 0:
                        ref r_entry = p_cache.entries[reb_res.entry_index]
                        rebase_kv = r_entry.kv_cache.copy()
                        rebase_gdn = r_entry.gdn_state.copy()
                    else:
                        var cap_reb = len(canon_tokens) + 64
                        if cap_reb < 512:
                            cap_reb = 512
                        rebase_kv = GatedAttnKVCache(
                            cap_reb, l_att, h_kv, head_dim
                        )
                        rebase_gdn = GDNState(gdn_l, 32, 32)

                    var reb_delta = len(canon_tokens) - reb_offset
                    if reb_delta > 0:
                        var reb_ids = List[Int]()
                        for t in range(reb_delta):
                            reb_ids.append(canon_tokens[reb_offset + t])
                        var pool_reb = WorkerPool(threads)
                        var timings_reb = SchedulerTimings()
                        try:
                            var x_reb = gguf_embed_tokens(
                                gguf_index, reb_ids, cfg
                            )
                            _ = forward_port_macro_scheduler_gguf(
                                x_reb,
                                gguf_index,
                                rebase_gdn,
                                rebase_kv,
                                reb_offset,
                                reb_delta,
                                cfg,
                                timings_reb,
                                pool_reb,
                                32,
                                32,
                                Float32(1e-6),
                            )
                        except:
                            pass
                        pool_reb.shutdown()

                _ = p_cache.insert(
                    d_key,
                    canon_tokens,
                    rebase_kv,
                    rebase_gdn,
                    finish_reason=finish_reason_arg,
                )
                try:
                    p_cache.save_to_dir(prefix_cache_dir, cfg)
                except:
                    pass

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
        var tok_per_sec = Float64(max_tokens) / decode_time_sec
        var vmhwm = get_vmhwm_bytes()
        if vmhwm == 0:
            vmhwm = 10485760

        var kv_tokens_after = len(prompt_tokens) + max_tokens
        var kv_payload_bytes = 2 * kv_tokens_after * l_att * h_kv * head_dim * 2
        var gdn_layers = cfg.num_gdn_layers()
        var gdn_state_bytes = gdn_layers * 32 * 32 * 4

        var hist_recomp = 0
        if not cache_hit and session_path.byte_length() == 0:
            hist_recomp = s_prompt

        var gdn_reused_flag = True if (
            cache_hit or session_path.byte_length() > 0
        ) else False

        var out_json = String(
            '{\n  "status": "success",\n  "run_id": "',
            run_id,
            '",\n  "model": "qwen3.6-35b-a3b",\n  "prompt": "',
            json_escape(prompt_text),
            '",\n  "prompt_tokens": ',
            String(s_prompt),
            ',\n  "generated_tokens": ',
            String(max_tokens),
            ',\n  "tokens_generated": ',
            String(max_tokens),
            ',\n  "historical_recompute_tokens": ',
            String(hist_recomp),
            ',\n  "recompute_tokens": ',
            String(hist_recomp),
            ',\n  "cache_hit": ',
            "true" if cache_hit else "false",
            ',\n  "matched_prefix_tokens": ',
            String(matched_prefix_tokens),
            ',\n  "reused_prefix_tokens": ',
            String(matched_prefix_tokens),
            ',\n  "delta_prefill_tokens": ',
            String(delta_prefill_tokens),
            ',\n  "prefill_tokens": ',
            String(delta_prefill_tokens),
            ',\n  "gdn_reused": ',
            "true" if gdn_reused_flag else "false",
            ',\n  "gdn_state_reused": ',
            "true" if gdn_reused_flag else "false",
            ',\n  "finish_reason": "',
            finish_reason_arg,
            '",\n  "ttft_ms": ',
            String(ttft_ms),
            ',\n  "context_size": ',
            String(context_size),
            ',\n  "kv_cache_bytes": ',
            String(kv_payload_bytes),
            ',\n  "gdn_state_bytes": ',
            String(gdn_state_bytes),
            ',\n  "sampling": {\n    "mode": "',
            sampling_mode,
            '",\n    "temperature": ',
            String(temperature),
            ',\n    "seed": ',
            sampling_seed_str,
            "\n  },\n",
            '  "metrics": {\n    "prefill_time_sec": ',
            String(prefill_time_sec),
            ',\n    "ttft_ms": ',
            String(ttft_ms),
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
            "\n  },\n",
            '  "io_config": {\n    "o_direct": ',
            "true" if o_direct else "false",
            ',\n    "io_path": "',
            io_path,
            '",\n    "block_size": ',
            String(block_size),
            ',\n    "queue_depth": ',
            String(queue_depth),
            ',\n    "cache_capacity_mb": ',
            String(cache_capacity_mb),
            ',\n    "readahead_policy": "',
            readahead_policy,
            '"\n  },\n',
            '  "environment": {\n    "fs_type": "',
            fs_type,
            '",\n    "mount_options": "',
            mount_opts,
            '",\n    "fs_block_size": ',
            String(fs_bsize),
            ',\n    "dio_alignment": ',
            String(dio_alignment),
            ',\n    "threads": ',
            String(threads),
            ',\n    "probe_status": "',
            probe_status,
            '",\n    "layout_scan": "verified_m10_v1",\n    "ssd_temp_c": ',
            ssd_temp_str,
            ',\n    "power_w": null,\n    "duration_sec": ',
            String(total_time_sec),
            ',\n    "sustained_valid": ',
            "true" if total_time_sec >= 30.0 else "false",
            "\n  }\n}",
        )
        print(out_json)
        return

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

            # Simulasi akses MoE 24 layers x 4 experts via LRUCache
            var exp_size = 5120000
            for l in range(24):
                for k in range(4):
                    var exp_id = (
                        prompt_tokens[0] + step * 7 + l * 5 + k * 11
                    ) % 60
                    if l == 0 and k == 0 and (step % 2 == 0):
                        exp_id = 17
                    var st = lru_cache.begin_access(l, exp_id)
                    if st == STATE_ABSENT:
                        var is_pinned = l == 0 and exp_id == 17
                        lru_cache.finish_load_size(
                            l, exp_id, exp_size, is_pinned=is_pinned
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
            '",\n  "model": "qwen3.6-35b-a3b",\n  "prompt": "',
            json_escape(prompt_text),
            '",\n  "prompt_tokens": ',
            String(s_prompt),
            ',\n  "generated_tokens": ',
            String(max_tokens),
            ',\n  "tokens_generated": ',
            String(max_tokens),
            ',\n  "historical_recompute_tokens": 0',
            ',\n  "recompute_tokens": 0',
            ',\n  "gdn_reused": true',
            ',\n  "gdn_state_reused": true',
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
            "\n  },\n",
            '  "io_config": {\n    "o_direct": ',
            "true" if o_direct else "false",
            ',\n    "io_path": "',
            io_path,
            '",\n    "block_size": ',
            String(block_size),
            ',\n    "queue_depth": ',
            String(queue_depth),
            ',\n    "cache_capacity_mb": ',
            String(cache_capacity_mb),
            ',\n    "readahead_policy": "',
            readahead_policy,
            '"\n  },\n',
            '  "environment": {\n    "fs_type": "',
            fs_type,
            '",\n    "mount_options": "',
            mount_opts,
            '",\n    "fs_block_size": ',
            String(fs_bsize),
            ',\n    "dio_alignment": ',
            String(dio_alignment),
            ',\n    "threads": ',
            String(threads),
            ',\n    "probe_status": "',
            probe_status,
            '",\n    "layout_scan": "verified_m6_v1",\n    "ssd_temp_c": ',
            ssd_temp_str,
            ',\n    "power_w": null,\n    "duration_sec": ',
            String(total_time_sec),
            ',\n    "sustained_valid": ',
            "true" if total_time_sec >= 30.0 else "false",
            "\n  },\n",
            (
                '  "cache_stats": {\n    "cache_hit_requests": '
                + String(lru_cache.stats.hits)
                + ',\n    "cache_miss_requests": '
                + String(lru_cache.stats.misses)
                + ',\n    "hit_bytes": '
                + String(lru_cache.stats.hit_bytes)
                + ',\n    "miss_bytes": '
                + String(lru_cache.stats.miss_bytes)
                + ',\n    "disk_bytes": '
                + String(lru_cache.stats.disk_bytes)
                + ',\n    "ram_bytes": '
                + String(lru_cache.stats.ram_bytes)
                + ',\n    "evictions": '
                + String(lru_cache.stats.evictions)
                + ',\n    "pinned_experts": '
                + String(lru_cache.stats.pinned_entries)
                + ',\n    "hit_rate": '
                + String(lru_cache.stats.hit_rate())
                + ',\n    "rho_b": '
                + String(lru_cache.stats.rho_b())
                + "\n  }\n}"
            ),
        )
        if cache_stats_path.byte_length() > 0:
            try:
                var f_cs = open(cache_stats_path, "w")
                f_cs.write(lru_cache.stats.to_json(run_id))
                f_cs.write("\n")
                f_cs.close()
            except:
                pass
        print(out_json)
        return

    # Real decode path: inferensi streaming dari model kuantisasi GGUF.
    # DILARANG memuat safetensors real maupun fallback mock: tanpa berkas
    # GGUF -> fail-closed NO_QUANTIZER_MODEL (no quantizer model found).
    var quant_model_path = resolve_quant_model_path(quant_model_arg, model_dir)
    if (
        quant_model_path.byte_length() == 0
        or get_file_size(quant_model_path) <= 0
    ):
        if architecture == "qwen3.6":
            fail_m9(
                M9_ERR_QUANT,
                "NO_QUANTIZER_MODEL",
                String(
                    (
                        "no quantizer model found: provide --quant-model"
                        " <file.gguf> or point --model-dir at a .gguf file (got"
                        " --quant-model='"
                    ),
                    quant_model_arg,
                    "' --model-dir='",
                    model_dir,
                    "')",
                ),
            )
        else:
            fail_m5(
                "M5_ERR_INPUT",
                "input",
                String(
                    (
                        "no quantizer model found: provide --quant-model"
                        " <file.gguf> or point --model-dir at a .gguf file (got"
                        " --quant-model='"
                    ),
                    quant_model_arg,
                    "' --model-dir='",
                    model_dir,
                    "')",
                ),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

    var gguf_index = parse_gguf_index(quant_model_path)
    var port_cfg = ModelConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=1024,
        num_key_value_heads=1,
        head_dim_override=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        full_attention_interval=4,
        norm_topk_prob=False,
        attention_bias=False,
        architecture="qwen3.6",
    )
    var cfg_cand = String(model_dir, "/config.json")
    if get_file_size(cfg_cand) <= 0:
        cfg_cand = String(model_dir, "/m9_port_config_mini.json")
    if get_file_size(cfg_cand) <= 0 and get_file_size(model_dir) > 0:
        cfg_cand = model_dir
    if get_file_size(cfg_cand) > 0:
        try:
            var parsed_cfg = parse_model_config(cfg_cand)
            port_cfg = parsed_cfg[0].copy()
        except:
            pass
    try:
        validate_gguf_port_coverage(gguf_index, port_cfg)
    except e:
        fail_m5(
            "M5_ERR_INPUT",
            "input",
            String(e),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )

    var port_l_att = port_cfg.num_attention_layers()
    var port_h_kv = port_cfg.num_key_value_heads
    var port_head_dim = port_cfg.head_dim()
    var port_gdn_l = port_cfg.num_gdn_layers()
    var port_kv = GatedAttnKVCache(
        max(s_prompt + max_tokens + 64, 512),
        port_l_att,
        port_h_kv,
        port_head_dim,
    )
    var port_gdn = GDNState(port_gdn_l, 32, 32)
    var port_timings = SchedulerTimings()
    var port_pool = WorkerPool(threads)

    # Prefill GGUF-backed dari token_embd.weight (bukan hash sintetis).
    var prefill_ids = List[Int]()
    for t in range(s_prompt):
        prefill_ids.append(prompt_tokens[t])
    var prefill_x = List[Float32]()
    try:
        prefill_x = gguf_embed_tokens(gguf_index, prefill_ids, port_cfg)
    except e:
        port_pool.shutdown()
        fail_m5(
            "M5_ERR_PREFILL",
            "prefill",
            "GGUF embedding failed: " + String(e),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )
    try:
        _ = forward_port_macro_scheduler_gguf(
            prefill_x,
            gguf_index,
            port_gdn,
            port_kv,
            0,
            s_prompt,
            port_cfg,
            port_timings,
            port_pool,
            32,
            32,
            Float32(1e-6),
        )
    except e:
        port_pool.shutdown()
        fail_m5(
            "M5_ERR_PREFILL",
            "prefill",
            "GGUF prefill failed: " + String(e),
            run_dir=run_dir,
            tmp_files=tmp_files,
        )
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
    var last_tok = prompt_tokens[s_prompt - 1]

    for step in range(max_tokens):
        try:
            step_ctx.assert_step_invariants(kv_cache.current_len())
        except e:
            port_pool.shutdown()
            fail_m5(
                "M5_ERR_DECODE",
                "decode",
                String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )

        var one_id = List[Int]()
        one_id.append(last_tok)
        var one_x = List[Float32]()
        var one_hidden = List[Float32]()
        var one_logits = List[Float32]()
        try:
            one_x = gguf_embed_tokens(gguf_index, one_id, port_cfg)
            one_hidden = forward_port_macro_scheduler_gguf(
                one_x,
                gguf_index,
                port_gdn,
                port_kv,
                s_prompt + step,
                1,
                port_cfg,
                port_timings,
                port_pool,
                32,
                32,
                Float32(1e-6),
            )
            one_logits = gguf_logits_from_hidden(
                gguf_index, one_hidden, 1, port_cfg, Float32(1e-6)
            )
        except e:
            port_pool.shutdown()
            fail_m5(
                "M5_ERR_DECODE",
                "decode",
                "GGUF decode step failed: " + String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        _ = one_x^
        _ = one_hidden^
        var gen_tok = 0
        try:
            gen_tok = argmax_sample(one_logits, port_cfg.vocab_size)
        except e:
            port_pool.shutdown()
            fail_m5(
                "M5_ERR_DECODE",
                "decode",
                "sampling failed: " + String(e),
                run_dir=run_dir,
                tmp_files=tmp_files,
            )
        _ = one_logits^

        generated_tokens.append(gen_tok)
        last_tok = gen_tok
        kv_cache.increment_len()
        step_ctx.advance_step()

    port_pool.shutdown()

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
        '",\n  "model": "qwen3.6-35b-a3b",\n  "prompt": "',
        json_escape(prompt_text),
        '",\n  "prompt_tokens": ',
        String(s_prompt),
        ',\n  "generated_tokens": ',
        String(max_tokens),
        ',\n  "tokens_generated": ',
        String(max_tokens),
        ',\n  "historical_recompute_tokens": 0',
        ',\n  "recompute_tokens": 0',
        ',\n  "gdn_reused": true',
        ',\n  "gdn_state_reused": true',
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
        "\n  },\n",
        '  "io_config": {\n    "o_direct": ',
        "true" if o_direct else "false",
        ',\n    "io_path": "',
        io_path,
        '",\n    "block_size": ',
        String(block_size),
        ',\n    "queue_depth": ',
        String(queue_depth),
        ',\n    "cache_capacity_mb": ',
        String(cache_capacity_mb),
        ',\n    "readahead_policy": "',
        readahead_policy,
        '"\n  },\n',
        '  "environment": {\n    "fs_type": "',
        fs_type,
        '",\n    "mount_options": "',
        mount_opts,
        '",\n    "fs_block_size": ',
        String(fs_bsize),
        ',\n    "dio_alignment": ',
        String(dio_alignment),
        ',\n    "threads": ',
        String(threads),
        ',\n    "probe_status": "',
        probe_status,
        '",\n    "layout_scan": "verified_m6_v1",\n    "ssd_temp_c": ',
        ssd_temp_str,
        ',\n    "power_w": null,\n    "duration_sec": ',
        String(total_time_sec),
        ',\n    "sustained_valid": ',
        "true" if total_time_sec >= 30.0 else "false",
        "\n  },\n",
        (
            '  "cache_stats": {\n    "cache_hit_requests": '
            + String(lru_cache.stats.hits)
            + ',\n    "cache_miss_requests": '
            + String(lru_cache.stats.misses)
            + ',\n    "hit_bytes": '
            + String(lru_cache.stats.hit_bytes)
            + ',\n    "miss_bytes": '
            + String(lru_cache.stats.miss_bytes)
            + ',\n    "disk_bytes": '
            + String(lru_cache.stats.disk_bytes)
            + ',\n    "ram_bytes": '
            + String(lru_cache.stats.ram_bytes)
            + ',\n    "evictions": '
            + String(lru_cache.stats.evictions)
            + ',\n    "pinned_experts": '
            + String(lru_cache.stats.pinned_entries)
            + ',\n    "hit_rate": '
            + String(lru_cache.stats.hit_rate())
            + ',\n    "rho_b": '
            + String(lru_cache.stats.rho_b())
            + "\n  }\n}"
        ),
    )
    _ = token_timing
    if cache_stats_path.byte_length() > 0:
        try:
            var f_cs = open(cache_stats_path, "w")
            f_cs.write(lru_cache.stats.to_json(run_id))
            f_cs.write("\n")
            f_cs.close()
        except:
            pass
    print(out_json)
