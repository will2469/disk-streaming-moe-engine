# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Konfigurasi model dan telemetri pemuatan memori (C1, G-M1-2)."""

comptime CHUNK_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB


@fieldwise_init
struct ModelConfig(Copyable, Movable):
    """Dimensi arsitektur Qwen2 MoE dari config.json."""

    var hidden_size: Int
    var num_hidden_layers: Int
    var num_attention_heads: Int
    var vocab_size: Int

    def head_dim(self) -> Int:
        return self.hidden_size // self.num_attention_heads


@fieldwise_init
struct LoadMemoryTelemetry(Copyable, Movable):
    """Telemetri memori fase load untuk memastikan strategi chunked (G-M1-2)."""

    var resident_target_bytes: Int
    var conversion_buffer_bytes: Int
    var source_buffer_bytes: Int
    var vmhwm_bytes: Int

    def __init__(out self):
        self.resident_target_bytes = 0
        self.conversion_buffer_bytes = 0
        self.source_buffer_bytes = 0
        self.vmhwm_bytes = 0


def _contains(s: String, sub: String) -> Bool:
    """Cek apakah string s mengandung substring sub."""
    var sb = s.as_bytes()
    var ub = sub.as_bytes()
    var slen = len(sb)
    var ulen = len(ub)
    if ulen > slen:
        return False
    for i in range(slen - ulen + 1):
        var found: Bool = True
        for j in range(ulen):
            if sb[i + j] != ub[j]:
                found = False
                break
        if found:
            return True
    return False
