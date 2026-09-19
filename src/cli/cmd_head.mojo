# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah head CLI dismoen."""

from cli.config_parser import parse_model_config, parse_tokens_json
from cli.errors import dirname, fail
from cli.io_utils import atomic_write_logits
from cli.sys_utils import (
    c_realpath,
    get_proc_io_read_bytes,
    get_vmhwm_bytes,
    resolve_target_output,
    validate_shards_coverage,
)
from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import load_tensor_f32_chunked
from format.index import parse_index_to_dict
from format.reader import read_header
from format.types import STHeader, TensorMeta, json_escape
from layers.head import (
    HeadWeights,
    forward_head,
    validate_logits,
)
from std.collections import List
from std.sys.terminate import exit
from std.time import perf_counter_ns


def _find_meta(
    h: STHeader, tensor_name: String, shard_path: String
) raises -> TensorMeta:
    for i in range(len(h.entries)):
        ref e = h.entries[i]
        if e.name == tensor_name:
            return e.copy()
    fail(
        "WEIGHT_LOAD_FAILED",
        "tensor missing from shard header",
        shard_path,
        tensor_name,
    )
    return TensorMeta("", "", List[Int](), 0, 0)


def cmd_head(args: List[String]) raises:
    if len(args) < 3:
        fail(
            "USAGE",
            (
                "pakai: dismoen head tokens.json (--model-dir <dir> |"
                " <shard...>) [--output <path>] [--workdir <dir>]"
            ),
            "",
            "",
        )
    var tokens_path = String(args[2])
    var model_dir = String("")
    var output_file = String("logits_mojo.bin")
    var workdir = String("")
    var supplied_shards = List[String]()

    var i = 3
    while i < len(args):
        var a = String(args[i])
        if a == "--model-dir":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --model-dir", "", "")
            model_dir = String(args[i + 1])
            i += 2
        elif a == "--output":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --output", "", "")
            output_file = String(args[i + 1])
            i += 2
        elif a == "--workdir":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --workdir", "", "")
            workdir = String(args[i + 1])
            i += 2
        elif a.startswith("-"):
            fail("USAGE", "unknown option: " + a, "", "")
        else:
            supplied_shards.append(a)
            i += 1

    # CLI strict: --model-dir melarang shard positional (sebelumnya diabaikan
    # diam-diam) dan sebaliknya sudah ditolak di bawah bila keduanya kosong.
    if model_dir != "" and len(supplied_shards) > 0:
        fail(
            "USAGE",
            "ambiguous invocation: --model-dir forbids positional shards",
            "",
            "",
        )

    var target_pair = resolve_target_output(output_file, workdir)
    var target_output = target_pair[0]
    var workdir_canon = target_pair[1]

    var model_root: String
    var is_model_dir_mode = False
    if model_dir != "":
        model_root = model_dir
        is_model_dir_mode = True
    else:
        if len(supplied_shards) < 1:
            fail(
                "USAGE",
                "butuh --model-dir <dir> atau N>=1 shard safetensors",
                "",
                "",
            )
        var r0 = dirname(supplied_shards[0])
        if r0 == "":
            r0 = "."
        # Bandingkan canonical path, bukan string lexical: "a/../a" vs "a"
        # sama; symlink yang tampak sama tapi target beda dibedakan.
        # Dua-duanya "" (tak-resolve) lolos ke FILE_NOT_FOUND downstream.
        var r0c = c_realpath(r0)
        for k in range(len(supplied_shards)):
            var rk = dirname(supplied_shards[k])
            if rk == "":
                rk = "."
            if c_realpath(rk) != r0c:
                fail(
                    "FILE_NOT_FOUND",
                    "all supplied shards must reside in the same directory",
                    supplied_shards[k],
                    "",
                )
        model_root = r0c if r0c != "" else r0

    var t_parse0 = perf_counter_ns()

    var index_path = String(model_root, "/model.safetensors.index.json")
    var weight_map = parse_index_to_dict(index_path)

    var config_path = String(model_root, "/model_config.json")
    if c_realpath(config_path) == "":
        var alt_cfg = String(model_root, "/config.json")
        if c_realpath(alt_cfg) != "":
            config_path = alt_cfg
    var cfg_tuple = parse_model_config(config_path)
    var cfg = cfg_tuple[0].copy()
    var eps = cfg_tuple[1]

    var req_embed = String(
        "model.language_model.embed_tokens.weight"
    ) if "model.language_model.embed_tokens.weight" in weight_map else String(
        "model.embed_tokens.weight"
    )
    var req_norm = String(
        "model.language_model.norm.weight"
    ) if "model.language_model.norm.weight" in weight_map else String(
        "model.norm.weight"
    )
    var req_head = String("lm_head.weight")

    if (
        req_embed not in weight_map
        or req_norm not in weight_map
        or req_head not in weight_map
    ):
        fail(
            "WEIGHT_LOAD_FAILED",
            "Required tensors not present in index weight_map",
            "",
            "",
        )

    var shard_embed_name = weight_map[req_embed]
    var shard_norm_name = weight_map[req_norm]
    var shard_head_name = weight_map[req_head]

    var shard_embed_path = String(model_root, "/", shard_embed_name)
    var shard_norm_path = String(model_root, "/", shard_norm_name)
    var shard_head_path = String(model_root, "/", shard_head_name)

    if not is_model_dir_mode:
        var req_list = List[String]()
        req_list.append(req_embed)
        req_list.append(req_norm)
        req_list.append(req_head)
        validate_shards_coverage(
            supplied_shards, req_list, weight_map, "embedding", -1
        )
        for si in range(len(supplied_shards)):
            var sp = supplied_shards[si]
            if c_realpath(sp) == "":
                fail("FILE_NOT_FOUND", "shard file not found: " + sp, sp, "")
    else:
        if c_realpath(shard_embed_path) == "":
            fail(
                "FILE_NOT_FOUND",
                "shard file not found: " + shard_embed_path,
                shard_embed_path,
                "",
            )
        if c_realpath(shard_norm_path) == "":
            fail(
                "FILE_NOT_FOUND",
                "shard file not found: " + shard_norm_path,
                shard_norm_path,
                "",
            )
        if c_realpath(shard_head_path) == "":
            fail(
                "FILE_NOT_FOUND",
                "shard file not found: " + shard_head_path,
                shard_head_path,
                "",
            )

    var h_embed = read_header(shard_embed_path)
    var h_norm = read_header(shard_norm_path)
    var h_head = read_header(shard_head_path)

    var meta_embed = _find_meta(h_embed, req_embed, shard_embed_path)
    var meta_norm = _find_meta(h_norm, req_norm, shard_norm_path)
    var meta_head = _find_meta(h_head, req_head, shard_head_path)

    if (
        len(meta_embed.shape) != 2
        or meta_embed.shape[0] != cfg.vocab_size
        or meta_embed.shape[1] != cfg.hidden_size
    ):
        fail(
            "WEIGHT_LOAD_FAILED",
            "embed shape mismatch vs config",
            shard_embed_path,
            req_embed,
        )
    if len(meta_norm.shape) != 1 or meta_norm.shape[0] != cfg.hidden_size:
        fail(
            "WEIGHT_LOAD_FAILED",
            "norm shape mismatch vs config",
            shard_norm_path,
            req_norm,
        )
    if (
        len(meta_head.shape) != 2
        or meta_head.shape[0] != cfg.vocab_size
        or meta_head.shape[1] != cfg.hidden_size
    ):
        fail(
            "WEIGHT_LOAD_FAILED",
            "head shape mismatch vs config",
            shard_head_path,
            req_head,
        )

    var tokens = parse_tokens_json(tokens_path, cfg.vocab_size)

    var telemetry = LoadMemoryTelemetry()
    var embed_t = load_tensor_f32_chunked(
        shard_embed_path, h_embed.data_base, meta_embed, telemetry
    )
    var norm_t = load_tensor_f32_chunked(
        shard_norm_path, h_norm.data_base, meta_norm, telemetry
    )
    var head_t = load_tensor_f32_chunked(
        shard_head_path, h_head.data_base, meta_head, telemetry
    )
    var weights = HeadWeights(embed_t^, norm_t^, head_t^)

    var parse_time_ms = Float64(perf_counter_ns() - t_parse0) / 1000000.0

    var t_comp0 = perf_counter_ns()
    var logits = forward_head(tokens, weights, cfg, eps)
    # Kontrak probe M2: 3 prompt × 16 token (sinkron dengan parse_tokens_json).
    validate_logits(logits, 3, 16, cfg.vocab_size)
    var compute_time_ms = Float64(perf_counter_ns() - t_comp0) / 1000000.0

    atomic_write_logits(target_output, logits)

    var vmhwm = get_vmhwm_bytes()
    if vmhwm == 0:
        vmhwm = (
            telemetry.resident_target_bytes + telemetry.conversion_buffer_bytes
        )

    print(
        String(
            '{"status":"success","num_prompts":3,"tokens_per_prompt":16,"num_tokens_total":48,"vocab_size":',
            cfg.vocab_size,
            ',"output_file":"',
            json_escape(output_file),
            '","workdir":"',
            json_escape(workdir_canon),
            '","parse_time_ms":',
            parse_time_ms,
            ',"compute_time_ms":',
            compute_time_ms,
            ',"memory":{"resident_target_bytes":',
            telemetry.resident_target_bytes,
            ',"conversion_buffer_bytes":',
            telemetry.conversion_buffer_bytes,
            ',"source_buffer_bytes":',
            telemetry.source_buffer_bytes,
            ',"vmhwm_bytes":',
            vmhwm,
            ',"io_read_bytes":',
            get_proc_io_read_bytes(),
            "}}",
        )
    )
    exit(0)
