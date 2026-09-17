# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Konfigurasi model dan telemetri pemuatan memori (C1, G-M1-2)."""

from format.types import error_json

comptime CHUNK_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB


struct ModelConfig(Copyable, Movable):
    """Dimensi arsitektur Qwen2 MoE dari config.json."""

    var hidden_size: Int
    var num_hidden_layers: Int
    var num_attention_heads: Int
    var vocab_size: Int
    var num_experts: Int
    var num_experts_per_tok: Int
    var moe_intermediate_size: Int
    var shared_expert_intermediate_size: Int
    var norm_topk_prob: Bool
    var architecture: String
    var num_key_value_heads: Int
    var head_dim_override: Int
    var full_attention_interval: Int
    var attention_bias: Bool

    def __init__(
        out self,
        hidden_size: Int,
        num_hidden_layers: Int,
        num_attention_heads: Int,
        vocab_size: Int,
        num_experts: Int = 60,
        num_experts_per_tok: Int = 4,
        moe_intermediate_size: Int = 1408,
        shared_expert_intermediate_size: Int = 5632,
        norm_topk_prob: Bool = False,
        architecture: String = "trial",
        num_key_value_heads: Int = -1,
        head_dim_override: Int = 0,
        full_attention_interval: Int = 4,
        attention_bias: Bool = False,
    ) raises:
        # Invariant arsitektur: konstruktor menolak konfigurasi absurd
        # (fail hard, bukan default diam-diam). Default parameter hanya
        # untuk field ABSENT di parser (field present-but-malformed sudah
        # ditolak parser) — dua lapis tak pernah berkata "nggak apa-apa"
        # untuk input yang sama.
        if (
            hidden_size <= 0
            or num_hidden_layers <= 0
            or num_attention_heads <= 0
            or vocab_size <= 0
        ):
            raise Error(
                error_json(
                    "CONFIG_ERROR",
                    "invalid model dimensions: all sizes must be > 0",
                    "",
                    "",
                )
            )
        if hidden_size % num_attention_heads != 0 and head_dim_override <= 0:
            raise Error(
                error_json(
                    "CONFIG_ERROR",
                    "hidden_size must be divisible by num_attention_heads",
                    "",
                    "",
                )
            )
        if (
            num_experts <= 0
            or num_experts_per_tok <= 0
            or num_experts_per_tok > num_experts
        ):
            raise Error(
                error_json(
                    "CONFIG_ERROR",
                    (
                        "invalid MoE topology: 0 < num_experts_per_tok <="
                        " num_experts"
                    ),
                    "",
                    "",
                )
            )
        if moe_intermediate_size <= 0 or shared_expert_intermediate_size <= 0:
            raise Error(
                error_json(
                    "CONFIG_ERROR",
                    "invalid MoE intermediate sizes: must be > 0",
                    "",
                    "",
                )
            )

        var kv_heads = num_key_value_heads
        if kv_heads <= 0:
            kv_heads = num_attention_heads
        if num_attention_heads % kv_heads != 0:
            raise Error(
                error_json(
                    "CONFIG_ERROR",
                    (
                        "num_attention_heads must be divisible by"
                        " num_key_value_heads"
                    ),
                    "",
                    "",
                )
            )

        if architecture != "trial" and architecture != "qwen3.6":
            raise Error(
                error_json(
                    "ARCHITECTURE_ERROR",
                    "unsupported architecture: " + architecture,
                    "",
                    "",
                )
            )

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.architecture = architecture
        self.num_key_value_heads = kv_heads
        self.head_dim_override = head_dim_override
        self.full_attention_interval = (
            full_attention_interval if full_attention_interval > 0 else 4
        )
        self.attention_bias = attention_bias

    def head_dim(self) -> Int:
        if self.head_dim_override > 0:
            return self.head_dim_override
        return self.hidden_size // self.num_attention_heads

    def is_gdn_layer(self, layer_idx: Int) -> Bool:
        if self.architecture == "trial":
            return False
        return (layer_idx % self.full_attention_interval) != (
            self.full_attention_interval - 1
        )

    def is_attention_layer(self, layer_idx: Int) -> Bool:
        if self.architecture == "trial":
            return True
        return (layer_idx % self.full_attention_interval) == (
            self.full_attention_interval - 1
        )

    def num_attention_layers(self) -> Int:
        if self.architecture == "trial":
            return self.num_hidden_layers
        return self.num_hidden_layers // self.full_attention_interval

    def num_gdn_layers(self) -> Int:
        return self.num_hidden_layers - self.num_attention_layers()

    def gqa_group_size(self) -> Int:
        return self.num_attention_heads // self.num_key_value_heads


@fieldwise_init
struct LoadMemoryTelemetry(Copyable, Movable):
    """Telemetri memori fase load untuk memastikan strategi chunked (G-M1-2)."""

    var resident_target_bytes: Int
    var conversion_buffer_bytes: Int
    var source_buffer_bytes: Int
    var vmhwm_bytes: Int
    var logical_bytes_read: Int

    def __init__(out self):
        self.resident_target_bytes = 0
        self.conversion_buffer_bytes = 0
        self.source_buffer_bytes = 0
        self.vmhwm_bytes = 0
        self.logical_bytes_read = 0
