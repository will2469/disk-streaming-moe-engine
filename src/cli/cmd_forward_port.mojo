# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Handler subcommand forward-port (M9 Port Inference & Config Preflight)."""

from cli.config_parser import parse_model_config_adapter
from cli.errors import eprint_json
from cli.m9_errors import (
    M9_ERR_ARCHITECTURE,
    M9_ERR_CONFIG,
    M9_ERR_INPUT,
    fail_m9,
    m9_error_json,
)
from cli.sys_utils import c_access_r, c_realpath, get_file_size
from core.config import ModelConfig
from format.file_io import read_small_file
from format.index import parse_index
from format.kmss import read_kmss_v1, write_kmss_v1
from format.types import json_escape
from layers.gated_attention import GatedAttnKVCache
from layers.gdn import GDNState
from layers.port_scheduler import (
    PortBlockWeights,
    create_synthetic_block_weights,
    forward_port_macro_scheduler,
)
from std.collections import Dict, List
from std.math import max, min
from std.sys.terminate import exit


def _parse_tokens_from_file(path: String) raises -> List[Int]:
    var raw = read_small_file(path)
    var n = len(raw)
    var i = 0
    while i < n and (
        raw[i] == 32 or raw[i] == 9 or raw[i] == 10 or raw[i] == 13
    ):
        i += 1
    if i < n and raw[i] == 123:  # '{'
        while i < n and raw[i] != 91:  # '['
            i += 1
    if i >= n or raw[i] != 91:
        raise Error("Expected '[' in tokens file")
    i += 1
    var tokens = List[Int]()
    while i < n:
        while i < n and (
            raw[i] == 32 or raw[i] == 9 or raw[i] == 10 or raw[i] == 13
        ):
            i += 1
        if i < n and raw[i] == 93:  # ']'
            break
        var val = 0
        var is_digit = False
        while i < n and (raw[i] >= 48 and raw[i] <= 57):
            val = val * 10 + (Int(raw[i]) - 48)
            is_digit = True
            i += 1
        if is_digit:
            tokens.append(val)
        while i < n and (
            raw[i] == 32 or raw[i] == 9 or raw[i] == 10 or raw[i] == 13
        ):
            i += 1
        if i < n and raw[i] == 44:  # ','
            i += 1
        elif i < n and raw[i] == 93:  # ']'
            break
    return tokens^


def _cross_validate_checkpoint_index(
    model_dir: String, cfg: ModelConfig
) raises:
    """Validasi silang config terhadap model.safetensors.index.json (R7).

    Memastikan jadwal tensor linear_attn vs self_attn dan jumlah layer konsisten.
    Mismatch memicu exit code 3 (M9_ERR_CONFIG).
    """
    var index_path = String(model_dir, "/model.safetensors.index.json")
    if get_file_size(index_path) <= 0:
        # Checkpoint tidak memiliki index sharded (misal fixture single-file)
        return

    var packed = parse_index(index_path)
    if len(packed) < 1:
        raise Error("Index file is empty or corrupted")

    var cs = packed[0].as_bytes()
    var num_entries = 0
    for i in range(len(cs)):
        num_entries = num_entries * 10 + (Int(cs[i]) - 48)

    var max_layer_idx = -1
    var has_gdn_key = False
    var has_self_attn_key = False

    for i in range(num_entries):
        var tname = packed[1 + 2 * i]
        if cfg.architecture == "qwen3.6":
            var pfx = "model.language_model.layers."
            if tname.startswith(pfx):
                var rem = String(
                    tname[byte = pfx.byte_length() : tname.byte_length()]
                )
                var dot_pos = rem.find(".")
                if dot_pos > 0:
                    var lyr_str = String(rem[byte=0:dot_pos])
                    var lyr_b = lyr_str.as_bytes()
                    var lyr_val = 0
                    var is_num = True
                    for b in range(len(lyr_b)):
                        if lyr_b[b] >= 48 and lyr_b[b] <= 57:
                            lyr_val = lyr_val * 10 + (Int(lyr_b[b]) - 48)
                        else:
                            is_num = False
                            break
                    if is_num and lyr_val > max_layer_idx:
                        max_layer_idx = lyr_val
                if rem.find(".linear_attn.") >= 0:
                    has_gdn_key = True
                if rem.find(".self_attn.") >= 0:
                    has_self_attn_key = True
        elif cfg.architecture == "trial":
            var pfx = "model.layers."
            var pfx2 = "layers."
            var rem = String("")
            if tname.startswith(pfx):
                rem = String(
                    tname[byte = pfx.byte_length() : tname.byte_length()]
                )
            elif tname.startswith(pfx2):
                rem = String(
                    tname[byte = pfx2.byte_length() : tname.byte_length()]
                )
            if rem.byte_length() > 0:
                var dot_pos = rem.find(".")
                if dot_pos > 0:
                    var lyr_str = String(rem[byte=0:dot_pos])
                    var lyr_b = lyr_str.as_bytes()
                    var lyr_val = 0
                    var is_num = True
                    for b in range(len(lyr_b)):
                        if lyr_b[b] >= 48 and lyr_b[b] <= 57:
                            lyr_val = lyr_val * 10 + (Int(lyr_b[b]) - 48)
                        else:
                            is_num = False
                            break
                    if is_num and lyr_val > max_layer_idx:
                        max_layer_idx = lyr_val

    var detected_layers = max_layer_idx + 1
    if detected_layers > 0 and detected_layers != cfg.num_hidden_layers:
        raise Error(
            "Index weight_map layer count ("
            + String(detected_layers)
            + ") does not match config num_hidden_layers ("
            + String(cfg.num_hidden_layers)
            + ")"
        )

    if cfg.architecture == "qwen3.6" and detected_layers >= 4:
        if not has_gdn_key:
            raise Error("Qwen3.6 index missing linear_attn keys for GDN layers")
        if not has_self_attn_key:
            raise Error(
                "Qwen3.6 index missing self_attn keys for Gated Attention"
                " layers"
            )


def cmd_forward_port(args: List[String]) raises:
    """CLI handler untuk kimo forward-port."""
    var model_dir = String("")
    var architecture = String("")
    var tokens_path = String("")
    var save_session_path = String("")
    var load_session_path = String("")
    var output_file = String("")
    var check_config_only = False
    var threads = 1
    var quantization = String("none")

    # 1. Parse Arguments
    var i = 2
    while i < len(args):
        var a = String(args[i])
        if a == "--model-dir":
            if i + 1 >= len(args):
                fail_m9(
                    M9_ERR_INPUT,
                    "INPUT_ERROR",
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
        elif a == "--tokens":
            if i + 1 >= len(args):
                fail_m9(
                    M9_ERR_INPUT,
                    "INPUT_ERROR",
                    "missing argument for --tokens",
                )
            tokens_path = String(args[i + 1])
            i += 2
        elif a == "--save-session":
            if i + 1 >= len(args):
                fail_m9(
                    M9_ERR_INPUT,
                    "INPUT_ERROR",
                    "missing argument for --save-session",
                )
            save_session_path = String(args[i + 1])
            i += 2
        elif a == "--load-session" or a == "--session":
            if i + 1 >= len(args):
                fail_m9(
                    M9_ERR_INPUT,
                    "INPUT_ERROR",
                    "missing argument for --load-session",
                )
            load_session_path = String(args[i + 1])
            i += 2
        elif a == "--output":
            if i + 1 >= len(args):
                fail_m9(
                    M9_ERR_INPUT,
                    "INPUT_ERROR",
                    "missing argument for --output",
                )
            output_file = String(args[i + 1])
            i += 2
        elif a == "--check-config-only" or a == "--check-config":
            check_config_only = True
            i += 1
        elif a == "--threads":
            if i + 1 < len(args):
                try:
                    threads = Int(args[i + 1])
                except:
                    threads = 1
            i += 2
        elif a == "--quantization":
            if i + 1 < len(args):
                quantization = String(args[i + 1])
            i += 2
        else:
            i += 1

    _ = tokens_path
    _ = output_file
    _ = check_config_only
    _ = threads
    _ = quantization

    # 2. Validasi Argument Wajib: --architecture
    if architecture.byte_length() == 0:
        fail_m9(
            M9_ERR_ARCHITECTURE,
            "MISSING_ARCHITECTURE",
            "flag --architecture trial|qwen3.6 is required (no default)",
        )

    if architecture != "trial" and architecture != "qwen3.6":
        fail_m9(
            M9_ERR_ARCHITECTURE,
            "UNSUPPORTED_ARCHITECTURE",
            "unsupported architecture: '"
            + architecture
            + "'. Must be 'trial' or 'qwen3.6'",
        )

    # 3. Validasi Argument Wajib: --model-dir
    if model_dir.byte_length() == 0:
        fail_m9(
            M9_ERR_INPUT,
            "MISSING_MODEL_DIR",
            "flag --model-dir is required",
        )

    var model_dir_canon = c_realpath(model_dir)
    if model_dir_canon.byte_length() == 0:
        fail_m9(
            M9_ERR_INPUT,
            "MODEL_DIR_NOT_FOUND",
            "model directory not found: " + model_dir,
        )

    # 4. Temukan config.json
    var config_path = String(model_dir_canon, "/config.json")
    if get_file_size(config_path) <= 0:
        config_path = String(model_dir_canon, "/m9_port_config_mini.json")
    if get_file_size(config_path) <= 0:
        config_path = model_dir_canon  # user passing direct json path

    if get_file_size(config_path) <= 0:
        fail_m9(
            M9_ERR_INPUT,
            "CONFIG_FILE_NOT_FOUND",
            "config.json not found in: " + model_dir_canon,
        )

    # 5. Eksekusi Config Adapter & Mismatch Detector
    var cfg = ModelConfig(2048, 24, 16, 151936)
    var eps = Float32(1e-6)
    try:
        var res = parse_model_config_adapter(config_path, architecture)
        cfg = res[0].copy()
        eps = res[1]
    except e:
        var err_s = String(e)
        if err_s.find('"error_code":2') >= 0:
            eprint_json(err_s)
            exit(2)
        elif err_s.find('"error_code":3') >= 0:
            eprint_json(err_s)
            exit(3)
        elif err_s.find('"error_code":1') >= 0:
            eprint_json(err_s)
            exit(1)
        else:
            fail_m9(
                M9_ERR_CONFIG,
                "CONFIG_ERROR",
                "failed parsing model config: " + err_s,
            )

    # 6. Validasi Silang Checkpoint Index (R7)
    try:
        _cross_validate_checkpoint_index(model_dir_canon, cfg)
    except e:
        fail_m9(
            M9_ERR_CONFIG,
            "INDEX_CROSS_VALIDATION_MISMATCH",
            "checkpoint index mismatch: " + String(e),
        )

    # 7. Eksekusi Hybrid Scheduler bila --tokens diberikan
    var exec_json = String("")
    if tokens_path.byte_length() > 0:
        var tokens = _parse_tokens_from_file(tokens_path)
        var seq_len = len(tokens)
        if seq_len == 0:
            fail_m9(M9_ERR_INPUT, "INPUT_ERROR", "tokens array is empty")

        var kv_layers = cfg.num_attention_layers()
        var kv_heads = cfg.num_key_value_heads
        var head_dim = cfg.head_dim()
        var gdn_layers = cfg.num_gdn_layers()
        var dk = 32
        var dv = 32

        var kv_cache = GatedAttnKVCache(
            max(seq_len + 64, 512), kv_layers, kv_heads, head_dim
        )
        var gdn_states = GDNState(gdn_layers, dv, dk)
        var prev_tokens = List[Int]()
        var pos_offset = 0
        var recompute_tokens = seq_len
        var gdn_state_reused = False

        if load_session_path.byte_length() > 0:
            try:
                var session_res = read_kmss_v1(load_session_path)
                kv_cache = session_res[1].copy()
                gdn_states = session_res[2].copy()
                prev_tokens = session_res[3].copy()
                pos_offset = len(prev_tokens)
                recompute_tokens = 0
                gdn_state_reused = True
            except e:
                fail_m9(
                    M9_ERR_INPUT,
                    "SESSION_LOAD_FAILED",
                    "failed loading session: " + String(e),
                )

        var kv_tokens_after = pos_offset + seq_len

        # Buat bobot sintetis untuk seluruh layer
        var blocks = List[PortBlockWeights]()
        for l in range(cfg.num_hidden_layers):
            blocks.append(create_synthetic_block_weights(cfg, l, dv, dk))

        # Inisialisasi token embedding aktivasi
        var x = List[Float32]()
        x.resize(seq_len * cfg.hidden_size, Float32(0.0))
        for t in range(seq_len):
            var tid = tokens[t]
            for d in range(cfg.hidden_size):
                x[t * cfg.hidden_size + d] = Float32(
                    (tid * 17 + d * 3) % 100
                ) * Float32(0.001)

        # Jalankan macro scheduler transformer penuh
        try:
            _ = forward_port_macro_scheduler(
                x,
                blocks,
                gdn_states,
                kv_cache,
                pos_offset,
                seq_len,
                cfg,
                dk,
                dv,
                eps,
            )
        except e:
            fail_m9(
                M9_ERR_CONFIG,
                "SCHEDULER_EXEC_FAILED",
                "forward scheduler failed: " + String(e),
            )

        # Simpan session jika diminta
        if save_session_path.byte_length() > 0:
            var all_tokens = List[Int]()
            for t in range(len(prev_tokens)):
                all_tokens.append(prev_tokens[t])
            for t in range(seq_len):
                all_tokens.append(tokens[t])
            try:
                write_kmss_v1(
                    save_session_path, kv_cache, gdn_states, all_tokens, cfg
                )
            except e:
                fail_m9(
                    M9_ERR_INPUT,
                    "SESSION_SAVE_FAILED",
                    "failed saving session: " + String(e),
                )

        var kv_cache_bytes = (
            2 * kv_tokens_after * kv_layers * kv_heads * head_dim * 4
        )
        var gdn_state_bytes = gdn_layers * dv * dk * 4

        exec_json = String(
            ',\n  "execution": {\n    "status": "COMPLETED",\n    "seq_len": ',
            String(seq_len),
            ',\n    "recompute_tokens": ',
            String(recompute_tokens),
            ',\n    "kv_tokens_after": ',
            String(kv_tokens_after),
            ',\n    "gdn_state_reused": ',
            "true" if gdn_state_reused else "false",
            ',\n    "kv_cache_bytes": ',
            String(kv_cache_bytes),
            ',\n    "gdn_state_bytes": ',
            String(gdn_state_bytes),
            "\n  }",
        )

    # 8. Output Laporan JSON Preflight & Eksekusi
    var out_json = String(
        (
            '{\n  "status": "success",\n  "command": "forward-port",\n '
            ' "architecture": "'
        ),
        architecture,
        '",\n  "model_dir": "',
        json_escape(model_dir_canon),
        '",\n  "config": {\n    "hidden_size": ',
        String(cfg.hidden_size),
        ',\n    "num_hidden_layers": ',
        String(cfg.num_hidden_layers),
        ',\n    "num_attention_heads": ',
        String(cfg.num_attention_heads),
        ',\n    "num_key_value_heads": ',
        String(cfg.num_key_value_heads),
        ',\n    "head_dim": ',
        String(cfg.head_dim()),
        ',\n    "vocab_size": ',
        String(cfg.vocab_size),
        ',\n    "num_experts": ',
        String(cfg.num_experts),
        ',\n    "num_experts_per_tok": ',
        String(cfg.num_experts_per_tok),
        ',\n    "moe_intermediate_size": ',
        String(cfg.moe_intermediate_size),
        ',\n    "shared_expert_intermediate_size": ',
        String(cfg.shared_expert_intermediate_size),
        ',\n    "num_gdn_layers": ',
        String(cfg.num_gdn_layers()),
        ',\n    "num_attention_layers": ',
        String(cfg.num_attention_layers()),
        ',\n    "full_attention_interval": ',
        String(cfg.full_attention_interval),
        ',\n    "attention_bias": ',
        "true" if cfg.attention_bias else "false",
        ',\n    "rms_norm_eps": ',
        String(eps),
        "\n  },\n",
        (
            '  "mismatch_detector": {\n    "status": "VERIFIED",\n   '
            ' "verdict": "PASS"\n  }'
        ),
        exec_json,
        "\n}",
    )
    print(out_json)
