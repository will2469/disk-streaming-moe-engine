# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Pelaksana subperintah layer part attention."""

from cli.config_parser import parse_model_config
from cli.errors import fail_layer
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
from core.tensor_loader import ShardHeaderCache
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


def run_layer_attn(
    layer_val: Int,
    model_dir: String,
    output_file: String,
    workdir: String,
    activation_path: String,
    supplied_shards: List[String],
) raises:
    # CLI strict: --model-dir melarang shard positional
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

    if cfg.attention_bias:
        try:
            validate_attention_bias_in_index(weight_map, cfg.num_hidden_layers)
        except:
            fail_layer(
                "WEIGHT_LOAD_FAILED",
                "attention bias count mismatch or missing in index",
                "attention",
                layer_val,
            )

    var prefix = "model.language_model.layers." + String(layer_val) + "."
    if (prefix + "input_layernorm.weight") not in weight_map:
        prefix = "model.layers." + String(layer_val) + "."

    var is_qwen36 = (
        prefix + "self_attn.q_norm.weight"
    ) in weight_map or not cfg.attention_bias

    var req_list = List[String]()
    req_list.append(prefix + "input_layernorm.weight")
    req_list.append(prefix + "self_attn.q_proj.weight")
    if cfg.attention_bias:
        req_list.append(prefix + "self_attn.q_proj.bias")
    req_list.append(prefix + "self_attn.k_proj.weight")
    if cfg.attention_bias:
        req_list.append(prefix + "self_attn.k_proj.bias")
    req_list.append(prefix + "self_attn.v_proj.weight")
    if cfg.attention_bias:
        req_list.append(prefix + "self_attn.v_proj.bias")
    if is_qwen36 and (prefix + "self_attn.q_norm.weight") in weight_map:
        req_list.append(prefix + "self_attn.q_norm.weight")
        req_list.append(prefix + "self_attn.k_norm.weight")
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

    # Kontrak probe M2: 16 token aktivasi
    var act = load_and_validate_activation(
        activation_path, layer_val, 16, cfg.hidden_size, "attention"
    )

    var cache = ShardHeaderCache()
    var telemetry = LoadMemoryTelemetry()
    var weights = load_layer_attention_weights(
        layer_val, model_root, weight_map, cfg, cache, telemetry
    )
    var parse_time_ms = Float64(perf_counter_ns() - t_parse0) / 1000000.0

    var t_comp0 = perf_counter_ns()
    var out_act = forward_attention_block(
        act,
        weights,
        16,
        cfg,
        eps,
        pos_offset=0,
        base=cfg.rope_theta,
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
