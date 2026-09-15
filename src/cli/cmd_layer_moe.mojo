# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Pelaksana subperintah layer part MoE."""

from cli.config_parser import parse_model_config
from cli.errors import fail_layer, fail_routing_violation
from cli.io_utils import (
    atomic_write_moe_output,
    check_moe_output_sanity,
    load_and_validate_activation,
)
from cli.oracle_parser import OracleRoutingData, parse_oracle_routing_json
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
from layers.moe_block import forward_moe_block
from layers.router_types import RoutingInfo
from std.collections import List
from std.sys.terminate import exit
from std.time import perf_counter_ns


def _format_1d_ints(data: List[Int]) -> String:
    var s = String("[")
    for j in range(len(data)):
        if j > 0:
            s += ","
        s += String(data[j])
    s += "]"
    return s


def _format_1d_floats(data: List[Float32]) -> String:
    var s = String("[")
    for j in range(len(data)):
        if j > 0:
            s += ","
        s += String(data[j])
    s += "]"
    return s


def _format_2d_ints(data: List[List[Int]]) -> String:
    var s = String("[")
    for i in range(len(data)):
        if i > 0:
            s += ","
        s += _format_1d_ints(data[i])
    s += "]"
    return s


def _format_2d_floats(data: List[List[Float32]]) -> String:
    var s = String("[")
    for i in range(len(data)):
        if i > 0:
            s += ","
        s += _format_1d_floats(data[i])
    s += "]"
    return s


def _expert_sets_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        var val = a[i]
        var found = False
        for j in range(len(b)):
            if b[j] == val:
                found = True
                break
        if not found:
            return False
    return True


def run_layer_moe(
    layer_val: Int,
    model_dir: String,
    output_file: String,
    workdir: String,
    activation_path: String,
    supplied_shards: List[String],
    oracle_routing_path: String,
) raises:
    if model_dir != "" and len(supplied_shards) > 0:
        fail_layer(
            "USAGE",
            "ambiguous invocation: --model-dir forbids positional shards",
            "router",
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
            "router",
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
            "router",
            layer_val,
        )

    var cfg_tuple = parse_model_config(config_path)
    var cfg = cfg_tuple[0].copy()

    if layer_val >= cfg.num_hidden_layers:
        fail_layer(
            "LAYER_INVALID",
            String(
                "layer index ",
                layer_val,
                " exceeds num_hidden_layers ",
                cfg.num_hidden_layers,
            ),
            "router",
            layer_val,
        )

    var r_gate = "model.layers." + String(layer_val) + ".mlp.gate.weight"
    if r_gate not in weight_map:
        fail_layer(
            "WEIGHT_LOAD_FAILED",
            "router gate tensor not in weight_map: " + r_gate,
            "router",
            layer_val,
        )

    var sh_pfx = "model.layers." + String(layer_val) + ".mlp.shared_expert."
    var sh_gate_proj = sh_pfx + "gate_proj.weight"
    var sh_up_proj = sh_pfx + "up_proj.weight"
    var sh_down_proj = sh_pfx + "down_proj.weight"
    var sh_gate = (
        "model.layers." + String(layer_val) + ".mlp.shared_expert_gate.weight"
    )

    if (
        sh_gate_proj not in weight_map
        or sh_up_proj not in weight_map
        or sh_down_proj not in weight_map
        or sh_gate not in weight_map
    ):
        fail_layer(
            "WEIGHT_LOAD_FAILED",
            "shared expert tensor not in weight_map",
            "shared",
            layer_val,
        )

    if not is_model_dir_mode:
        var req_list = List[String]()
        req_list.append(r_gate)
        req_list.append(sh_gate_proj)
        req_list.append(sh_up_proj)
        req_list.append(sh_down_proj)
        req_list.append(sh_gate)
        for e in range(cfg.num_experts):
            var exp_pfx = (
                "model.layers."
                + String(layer_val)
                + ".mlp.experts."
                + String(e)
                + "."
            )
            var eg = exp_pfx + "gate_proj.weight"
            var eu = exp_pfx + "up_proj.weight"
            var ed = exp_pfx + "down_proj.weight"
            if eg in weight_map:
                req_list.append(eg)
            if eu in weight_map:
                req_list.append(eu)
            if ed in weight_map:
                req_list.append(ed)
        validate_shards_coverage(
            supplied_shards, req_list, weight_map, "experts", layer_val
        )

    var act = load_and_validate_activation(
        activation_path, layer_val, 16, cfg.hidden_size, "router"
    )
    var parse_time_ms = Float64(perf_counter_ns() - t_parse0) / 1000000.0

    var t_comp0 = perf_counter_ns()
    var cache = ShardHeaderCache()
    var telemetry = LoadMemoryTelemetry()
    var block_res = forward_moe_block(
        act, layer_val, 16, model_root, weight_map, cfg, cache, telemetry
    )
    var out_act = block_res[0].copy()
    var routing = block_res[1].copy()
    var compute_time_ms = Float64(perf_counter_ns() - t_comp0) / 1000000.0

    check_moe_output_sanity(out_act, act, layer_val, cfg.hidden_size)
    atomic_write_moe_output(target_output, out_act, layer_val)

    if oracle_routing_path != "":
        var oracle_data = parse_oracle_routing_json(
            oracle_routing_path, layer_val
        )
        for t in range(16):
            if t < len(oracle_data.selected_experts):
                ref eng_exp = routing.selected_experts[t]
                ref ora_exp = oracle_data.selected_experts[t]
                if not _expert_sets_equal(eng_exp, ora_exp):
                    var ora_p = List[Float32]()
                    if t < len(oracle_data.router_probs):
                        ora_p = oracle_data.router_probs[t].copy()
                    var violation_json = String(
                        '{"status":"mismatch","layer":',
                        String(layer_val),
                        ',"part":"moe","num_tokens":16,"output_file":"',
                        json_escape(output_file),
                        '","parse_time_ms":',
                        String(parse_time_ms),
                        ',"compute_time_ms":',
                        String(compute_time_ms),
                        ',"routing_violation":{"token_index":',
                        String(t),
                        ',"oracle_experts":',
                        _format_1d_ints(ora_exp),
                        ',"engine_experts":',
                        _format_1d_ints(eng_exp),
                        ',"oracle_probs":',
                        _format_1d_floats(ora_p),
                        ',"engine_probs":',
                        _format_1d_floats(routing.router_probs[t]),
                        "}}",
                    )
                    print(violation_json)
                    fail_routing_violation(
                        "SET expert terpilih beda dengan oracle pada token "
                        + String(t),
                        "router",
                        layer_val,
                        t,
                    )

    var success_json = String(
        '{"status":"success","layer":',
        String(layer_val),
        ',"part":"moe","num_tokens":16,"output_file":"',
        json_escape(output_file),
        '","routing_info":{"selected_experts":',
        _format_2d_ints(routing.selected_experts),
        ',"router_probs":',
        _format_2d_floats(routing.router_probs),
        '},"parse_time_ms":',
        String(parse_time_ms),
        ',"compute_time_ms":',
        String(compute_time_ms),
        "}",
    )
    print(success_json)
    exit(0)
