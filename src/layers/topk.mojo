# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Seleksi Top-k probabilitas router tanpa renormalisasi (F8b)."""

from layers.router_types import RoutingInfo
from std.collections import List
from std.math import abs


def select_topk(
    probs: List[Float32],
    seq_len: Int,
    num_experts: Int,
    top_k: Int,
    norm_topk_prob: Bool,
    layer_idx: Int = 0,
) raises -> RoutingInfo:
    """Seleksi top-k probabilitas router tanpa renormalisasi (F8b).

    @spec scratch/wave/m3/m3-w1-router.md (F8b)
    """
    if norm_topk_prob:
        raise Error(
            '{"error_type":"ROUTER_ERROR","detail":"norm_topk_prob=true'
            " forbidden in trial configuration; renormalization"
            ' leak","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )
    if seq_len <= 0 or num_experts <= 0 or top_k <= 0:
        raise Error(
            '{"error_type":"ROUTER_ERROR","detail":"dimensions must be'
            ' positive","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )
    if top_k > num_experts:
        raise Error(
            '{"error_type":"ROUTER_ERROR","detail":"top_k cannot exceed'
            ' num_experts","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(probs) != seq_len * num_experts:
        raise Error(
            '{"error_type":"ROUTER_ERROR","detail":"probabilities length'
            " mismatch: expected "
            + String(seq_len * num_experts)
            + " got "
            + String(len(probs))
            + '","stage":"router","layer":'
            + String(layer_idx)
            + "}"
        )

    var all_selected = List[List[Int]]()
    var all_probs = List[List[Float32]]()
    all_selected.reserve(seq_len)
    all_probs.reserve(seq_len)

    var p_probs = probs.unsafe_ptr()

    for t in range(seq_len):
        var base = t * num_experts

        var ids = List[Int]()
        var vals = List[Float32]()
        ids.reserve(num_experts)
        vals.reserve(num_experts)
        var sum_all = Float32(0.0)

        for e in range(num_experts):
            var pv = p_probs[unsafe_offset=base + e]
            ids.append(e)
            vals.append(pv)
            sum_all += pv

        # Selection sort descending untuk top_k elemen
        for i in range(top_k):
            var max_idx = i
            var max_val = vals[i]
            for j in range(i + 1, num_experts):
                if vals[j] > max_val:
                    max_val = vals[j]
                    max_idx = j
                elif vals[j] == max_val and ids[j] < ids[max_idx]:
                    max_val = vals[j]
                    max_idx = j
            if max_idx != i:
                var tmp_v = vals[i]
                vals[i] = vals[max_idx]
                vals[max_idx] = tmp_v
                var tmp_id = ids[i]
                ids[i] = ids[max_idx]
                ids[max_idx] = tmp_id

        var top_ids = List[Int]()
        var top_vals = List[Float32]()
        top_ids.reserve(top_k)
        top_vals.reserve(top_k)
        var sum_topk = Float32(0.0)

        for i in range(top_k):
            top_ids.append(ids[i])
            top_vals.append(vals[i])
            sum_topk += vals[i]

        if sum_topk > Float32(1.00001):
            raise Error(
                '{"error_type":"ROUTER_ERROR","detail":"sum of top-k'
                ' probabilities exceeds 1.0","stage":"router","layer":'
                + String(layer_idx)
                + "}"
            )

        if top_k < num_experts:
            var sum_remaining = sum_all - sum_topk
            if sum_remaining > Float32(1e-5) and abs(
                sum_topk - Float32(1.0)
            ) < Float32(1e-6):
                raise Error(
                    '{"error_type":"ROUTER_ERROR","detail":"renormalization'
                    " leak detected: sum of top-k probabilities is"
                    ' 1.0","stage":"router","layer":'
                    + String(layer_idx)
                    + "}"
                )

        all_selected.append(top_ids^)
        all_probs.append(top_vals^)

    return RoutingInfo(all_selected^, all_probs^, seq_len, top_k)
