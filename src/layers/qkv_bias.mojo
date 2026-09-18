# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Validasi tensor bias attention untuk arsitektur Qwen MoE (M2-W1, Invariant P-2)."""

from format.types import json_escape
from std.collections import Dict, List


def validate_bias_count(
    bias_names: List[String],
    num_layers: Int,
    shard: String,
) raises:
    """Validasi P-2: jumlah tensor bias attention == 3 * num_layers.

    @spec ref-ground-truth P-2
    """
    var expected = 3 * num_layers
    var actual = len(bias_names)
    if actual != expected:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"bias count mismatch:'
            " expected "
            + String(expected)
            + " got "
            + String(actual)
            + '","stage":"attention","layer":0,"shard":"'
            + json_escape(shard)
            + '","tensor_name":""}'
        )


def validate_attention_bias_in_index(
    weight_map: Dict[String, String],
    num_layers: Int = 24,
) raises:
    """Validasi P-2: index safetensors wajib memiliki 3 tensor bias per layer (q, k, v)
    sehingga total bias == 3 * num_layers (72 untuk 24 layer).
    Jika kurang/lebih atau ada layer yang tidak lengkap -> WEIGHT_LOAD_FAILED.
    """
    for l in range(num_layers):
        var prefix = "model.layers." + String(l) + ".self_attn."
        var q_bias = prefix + "q_proj.bias"
        var k_bias = prefix + "k_proj.bias"
        var v_bias = prefix + "v_proj.bias"
        if q_bias not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"missing '
                + q_bias
                + ' in index weight_map","shard":"","tensor_name":"'
                + q_bias
                + '"}'
            )
        if k_bias not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"missing '
                + k_bias
                + ' in index weight_map","shard":"","tensor_name":"'
                + k_bias
                + '"}'
            )
        if v_bias not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"missing '
                + v_bias
                + ' in index weight_map","shard":"","tensor_name":"'
                + v_bias
                + '"}'
            )


def collect_attention_bias_names(
    tensor_names: List[String],
    num_layers: Int,
) -> List[String]:
    """Kumpulkan nama tensor bias attention dari daftar semua tensor di index.
    """
    var result = List[String]()
    for i in range(len(tensor_names)):
        var name = tensor_names[i]
        var nb = name.as_bytes()
        if len(nb) >= 5:
            var is_bias = (
                Int(nb[len(nb) - 5]) == 46
                and Int(nb[len(nb) - 4]) == 98
                and Int(nb[len(nb) - 3]) == 105
                and Int(nb[len(nb) - 2]) == 97
                and Int(nb[len(nb) - 1]) == 115
            )
            if is_bias:
                var has_attn = name.find("self_attn") >= 0
                var has_proj = (
                    name.find("q_proj") >= 0
                    or name.find("k_proj") >= 0
                    or name.find("v_proj") >= 0
                )
                if has_attn and has_proj:
                    result.append(name)
    return result^
