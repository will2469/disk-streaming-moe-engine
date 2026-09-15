# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah layer CLI kimo (dispatcher part attention dan MoE)."""

from cli.cmd_layer_attn import run_layer_attn
from cli.cmd_layer_moe import run_layer_moe
from cli.errors import fail_layer
from std.collections import List


def _is_m2_probe_layer(layer_val: Int) -> Bool:
    # Kebijakan milestone: hanya layer terverifikasi oracle (0, 12, 23)
    # yang di-probe di level single-layer CLI.
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
    var part = String("attn")
    var model_dir = String("")
    var output_file = String("")
    var workdir = String("")
    var oracle_routing = String("")
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
        elif a == "--part":
            if i + 1 >= len(args):
                fail_layer(
                    "PART_INVALID",
                    "missing argument for --part",
                    "cli",
                    -1,
                )
            part = String(args[i + 1])
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
        elif a == "--oracle-routing":
            if i + 1 >= len(args):
                fail_layer(
                    "ROUTER_ERROR",
                    "missing argument for --oracle-routing",
                    "router",
                    -1,
                )
            oracle_routing = String(args[i + 1])
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
            "attention" if part != "moe" else "router",
            layer_val,
        )

    var activation_path = positionals[0]
    var supplied_shards = List[String]()
    for si in range(1, len(positionals)):
        supplied_shards.append(positionals[si])

    if part == "attn":
        if output_file == "":
            output_file = "attn_output.bin"
        run_layer_attn(
            layer_val,
            model_dir,
            output_file,
            workdir,
            activation_path,
            supplied_shards,
        )
    elif part == "moe":
        if output_file == "":
            output_file = "moe_output.bin"
        run_layer_moe(
            layer_val,
            model_dir,
            output_file,
            workdir,
            activation_path,
            supplied_shards,
            oracle_routing,
        )
    else:
        fail_layer(
            "PART_INVALID",
            "unsupported layer part: " + part + " (expected 'attn' or 'moe')",
            "cli",
            layer_val,
        )
