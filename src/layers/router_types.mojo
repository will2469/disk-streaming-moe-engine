# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Tipe data dan konfigurasi untuk MoE Router (F8a, F8b)."""

from std.collections import List


@fieldwise_init
struct RouterConfig(Copyable, Movable):
    """Konfigurasi MoE Router (F8a, F8b).

    Default parameter diturunkan dari Qwen1.5-MoE-A2.7B:
    - num_experts: 60
    - num_experts_per_tok: 4 (top-4)
    - norm_topk_prob: False (tanpa renormalisasi)
    """

    var num_experts: Int
    var num_experts_per_tok: Int
    var norm_topk_prob: Bool

    def validate(self) raises:
        if self.num_experts <= 0:
            raise Error(
                '{"error_type":"CONFIG_ERROR","detail":"num_experts must be'
                ' positive","stage":"router"}'
            )
        if self.num_experts_per_tok <= 0:
            raise Error(
                '{"error_type":"CONFIG_ERROR","detail":"num_experts_per_tok'
                ' must be positive","stage":"router"}'
            )
        if self.num_experts_per_tok > self.num_experts:
            raise Error(
                '{"error_type":"CONFIG_ERROR","detail":"num_experts_per_tok'
                ' cannot exceed num_experts","stage":"router"}'
            )
        if self.norm_topk_prob:
            raise Error(
                '{"error_type":"ROUTER_ERROR","detail":"norm_topk_prob=true'
                " forbidden; unrenormalized routing invariant"
                ' violated","stage":"router"}'
            )


@fieldwise_init
struct RoutingInfo(Copyable, Movable):
    """Hasil seleksi routing untuk batch/sequence token (F8b)."""

    var selected_experts: List[List[Int]]
    var router_probs: List[List[Float32]]
    var num_tokens: Int
    var top_k: Int

    def to_json(self) -> String:
        """Serialisasi routing_info ke format JSON sesuai spesifikasi M3."""
        var s = String('{"selected_experts":[')
        for i in range(self.num_tokens):
            if i > 0:
                s += ","
            s += "["
            ref exp_row = self.selected_experts[i]
            for j in range(len(exp_row)):
                if j > 0:
                    s += ","
                s += String(exp_row[j])
            s += "]"
        s += '],"router_probs":['
        for i in range(self.num_tokens):
            if i > 0:
                s += ","
            s += "["
            ref prob_row = self.router_probs[i]
            for j in range(len(prob_row)):
                if j > 0:
                    s += ","
                s += String(prob_row[j])
            s += "]"
        s += "]}"
        return s
