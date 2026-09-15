# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah layer CLI kimo."""

from cli.config_parser import parse_model_config
from cli.errors import dirname, fail_layer
from cli.io_utils import (
    atomic_write_attn_output,
    check_layer_output_sanity,
    load_and_validate_activation,
)
from cli.sys_utils import (
    c_realpath,
    resolve_layer_model_root,
    resolve_target_output,
    validate_shards_coverage,
)
from core.config import LoadMemoryTelemetry, ModelConfig
from format.index import parse_index_to_dict
from format.types import json_escape
from layers.attention import (
    forward_attention_block,
    load_layer_attention_weights,
)
from layers.qkv_bias import validate_attention_bias_in_index
from std.collections import List
from std.sys.terminate import exit
from std.time import perf_counter_ns


def _is_m2_probe_layer(layer_val: Int) -> Bool:
    # Kebijakan milestone M2: hanya layer terverifikasi oracle (awal/tengah/
    # akhir) yang boleh di-probe. BUKAN invariant engine universal — perluas
    # di SATU tempat ini saat milestone berubah. Rentang vs
    # config.num_hidden_layers dicek terpisah pasca-load config di bawah.
    return layer_val == 0 or layer_val == 12 or layer_val == 23


def _parse_layer_index(layer_str: String) raises -> Int:
    if layer_str == "":
        fail_layer(
            "LAYER_INVALID",
            "missing required --layer argument",
            "attention",
            -1,
        )
    var is_neg = False
    var sb = layer_str.as_bytes()
    var start_k = 0
    if len(sb) > 0 and Int(sb[0]) == 45:
        is_neg = True
        start_k = 1
    if len(sb) == 0 or (is_neg and len(sb) == 1):
        fail_layer(
            "LAYER_INVALID",
            "invalid layer number format: " + layer_str,
            "attention",
            -1,
        )
    var layer_val = 0
    for k in range(start_k, len(sb)):
        var c = Int(sb[k])
        if c < 48 or c > 57:
            fail_layer(
                "LAYER_INVALID",
                "invalid layer number format: " + layer_str,
                "attention",
                -1,
            )
        layer_val = layer_val * 10 + (c - 48)
    if is_neg:
        layer_val = -layer_val
    if not _is_m2_probe_layer(layer_val):
        fail_layer(
            "LAYER_INVALID",
            "layer number invalid (M2 probe set: 0, 12, or 23): "
            + String(layer_val),
            "attention",
            layer_val,
        )
    return layer_val


def cmd_layer(args: List[String]) raises:
    var layer_str = String("")
    var model_dir = String("")
    var output_file = String("attn_output.bin")
    var workdir = String("")
    var positionals = List[String]()

    var i = 2
    while i < len(args):
        var a = String(args[i])
        if a == "--layer":
            if i + 1 >= len(args):
                fail_layer(
                    "LAYER_INVALID",
                    "missing argument for --layer",
                    "attention",
                    -1,
                )
            layer_str = String(args[i + 1])
            i += 2
        elif a == "--model-dir":
            if i + 1 >= len(args):
                fail_layer(
                    "WEIGHT_LOAD_FAILED",
                    "missing argument for --model-dir",
                    "attention",
                    -1,
                )
            model_dir = String(args[i + 1])
            i += 2
        elif a == "--output":
            if i + 1 >= len(args):
                fail_layer(
                    "OUTPUT_WRITE_FAILED",
                    "missing argument for --output",
                    "output",
                    -1,
                )
            output_file = String(args[i + 1])
            i += 2
        elif a == "--workdir":
            if i + 1 >= len(args):
                fail_layer(
                    "OUTPUT_WRITE_FAILED",
                    "missing argument for --workdir",
                    "output",
                    -1,
                )
            workdir = String(args[i + 1])
            i += 2
        elif a.startswith("-"):
            fail_layer("LAYER_INVALID", "unknown option: " + a, "attention", -1)
        else:
            positionals.append(a)
            i += 1

    var layer_val = _parse_layer_index(layer_str)
    if len(positionals) < 1:
        fail_layer(
            "ACT_LOAD_FAILED",
            "missing activation input file argument",
            "attention",
            layer_val,
        )

    var activation_path = positionals[0]
    var supplied_shards = List[String]()
    for si in range(1, len(positionals)):
        supplied_shards.append(positionals[si])

    # CLI strict: --model-dir melarang shard positional (sebelumnya diabaikan
    # diam-diam oleh cabang is_model_dir_mode).
    if model_dir != "" and len(supplied_shards) > 0:
        fail_layer(
            "USAGE",
            "ambiguous invocation: --model-dir forbids positional shards",
            "attention",
            layer_val,
        )

    var target_pair = resolve_target_output(output_file, workdir, layer_val)
    var target_output = target_pair[0]

    var root_info = resolve_layer_model_root(
        model_dir, supplied_shards, layer_val
    )
    var model_root = root_info[0]
    var is_model_dir_mode = root_info[1]

    var t_parse0 = perf_counter_ns()
    var index_path = String(model_root, "/model.safetensors.index.json")
    if c_realpath(index_path) == "":
        fail_layer(
            "FILE_NOT_FOUND",
            "index file not found: " + index_path,
            "attention",
            layer_val,
        )
    var weight_map = parse_index_to_dict(index_path)

    var config_path = String(model_root, "/model_config.json")
    if c_realpath(config_path) == "":
        var alt_cfg = String(model_root, "/config.json")
        if c_realpath(alt_cfg) != "":
            config_path = alt_cfg
    if c_realpath(config_path) == "":
        fail_layer(
            "FILE_NOT_FOUND",
            "config file not found in: " + model_root,
            "attention",
            layer_val,
        )

    var cfg_tuple = parse_model_config(config_path)
    var cfg = cfg_tuple[0].copy()
    var eps = cfg_tuple[1]

    if layer_val >= cfg.num_hidden_layers:
        fail_layer(
            "LAYER_INVALID",
            String(
                "layer index ",
                layer_val,
                " exceeds num_hidden_layers ",
                cfg.num_hidden_layers,
            ),
            "attention",
            layer_val,
        )

    try:
        validate_attention_bias_in_index(weight_map, cfg.num_hidden_layers)
    except:
        fail_layer(
            "WEIGHT_LOAD_FAILED",
            "attention bias count mismatch or missing in index",
            "attention",
            layer_val,
        )

    var prefix = "model.layers." + String(layer_val) + "."
    var req_list = List[String]()
    req_list.append(prefix + "input_layernorm.weight")
    req_list.append(prefix + "self_attn.q_proj.weight")
    req_list.append(prefix + "self_attn.q_proj.bias")
    req_list.append(prefix + "self_attn.k_proj.weight")
    req_list.append(prefix + "self_attn.k_proj.bias")
    req_list.append(prefix + "self_attn.v_proj.weight")
    req_list.append(prefix + "self_attn.v_proj.bias")
    req_list.append(prefix + "self_attn.o_proj.weight")

    for ri in range(len(req_list)):
        var rn = req_list[ri]
        if rn not in weight_map:
            fail_layer(
                "WEIGHT_LOAD_FAILED",
                "required tensor not in weight_map: " + rn,
                "attention",
                layer_val,
            )

    if not is_model_dir_mode:
        validate_shards_coverage(
            supplied_shards, req_list, weight_map, "attention", layer_val
        )

    # Kontrak probe M2 (bukan invariant engine): 16 token aktivasi,
    # RoPE base Qwen 1e6, offset posisi 0. Pindah ke config saat model
    # berikutnya butuh nilai lain.
    var act = load_and_validate_activation(
        activation_path, layer_val, 16, cfg.hidden_size
    )

    var telemetry = LoadMemoryTelemetry()
    var weights = load_layer_attention_weights(
        layer_val, model_root, weight_map, cfg, telemetry
    )
    var parse_time_ms = Float64(perf_counter_ns() - t_parse0) / 1000000.0

    var t_comp0 = perf_counter_ns()
    # Lihat kontrak probe M2 di atas: 16 token, RoPE base 1e6, offset 0.
    var out_act = forward_attention_block(
        act,
        weights,
        16,
        cfg,
        eps,
        pos_offset=0,
        base=Float32(1000000.0),
        layer_idx=layer_val,
    )
    var compute_time_ms = Float64(perf_counter_ns() - t_comp0) / 1000000.0

    check_layer_output_sanity(out_act, act, layer_val, cfg.hidden_size)
    atomic_write_attn_output(target_output, out_act, layer_val)

    print(
        String(
            '{"status":"success","layer":',
            String(layer_val),
            ',"num_tokens":16,"output_file":"',
            json_escape(output_file),
            '","parse_time_ms":',
            String(parse_time_ms),
            ',"compute_time_ms":',
            String(compute_time_ms),
            "}",
        )
    )
    exit(0)
