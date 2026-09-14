# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""RoPE (Rotary Position Embedding) F7 — rotasi berpasangan non-interleaved."""

from core.config import ModelConfig
from std.collections import List
from std.math import abs, cos, isinf, isnan, max, pow, sin, sqrt


def verify_rope_isometry(
    orig: List[Float32],
    rotated: List[Float32],
    seq_len: Int,
    cfg: ModelConfig,
    layer_idx: Int = 0,
    tol: Float32 = Float32(1e-4),
) raises:
    """Verifikasi invariant isometri F7: ||q'||_2 == ||q||_2 per head dan per token.

    Pelanggaran isometri atau non-finite langsung raise ROPE_ERROR (P-3).
    @spec scratch/wave/m2/m2-w2-rope.md
    @spec docs/milestones/M2-attention.md (§ Rumus F7)
    """
    if len(orig) != len(rotated):
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"orig and rotated length'
            ' mismatch","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    var hidden = cfg.hidden_size
    var num_heads = cfg.num_attention_heads
    var head_dim = cfg.head_dim()
    if len(orig) != seq_len * hidden:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"length mismatch with'
            ' seq_len*hidden","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    var p_orig = orig.unsafe_ptr()
    var p_rot = rotated.unsafe_ptr()
    for t in range(seq_len):
        for h in range(num_heads):
            var head_start = t * hidden + h * head_dim
            var orig_norm_sq = Float32(0.0)
            var rot_norm_sq = Float32(0.0)
            for k in range(head_dim):
                var val_orig = p_orig[unsafe_offset=head_start + k]
                var val_rot = p_rot[unsafe_offset=head_start + k]
                if (
                    isnan(val_orig)
                    or isinf(val_orig)
                    or isnan(val_rot)
                    or isinf(val_rot)
                ):
                    raise Error(
                        '{"error_type":"ROPE_ERROR","detail":"non-finite value'
                        ' detected in isometry check","stage":"rope","layer":'
                        + String(layer_idx)
                        + "}"
                    )
                orig_norm_sq += val_orig * val_orig
                rot_norm_sq += val_rot * val_rot
            var norm_orig = sqrt(orig_norm_sq)
            var norm_rot = sqrt(rot_norm_sq)
            var diff = abs(norm_orig - norm_rot)
            var max_norm = max(Float32(1.0), norm_orig)
            if diff > tol * max_norm:
                raise Error(
                    '{"error_type":"ROPE_ERROR","detail":"RoPE invariant'
                    " violation: ||q'|| != ||q|| (diff="
                    + String(diff)
                    + ')","stage":"rope","layer":'
                    + String(layer_idx)
                    + "}"
                )


def rope_rotate_half(
    x: List[Float32],
    seq_len: Int,
    cfg: ModelConfig,
    pos_offset: Int = 0,
    base: Float32 = Float32(1000000.0),
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """RoPE rotate_half kernel F7 dengan verifikasi invariant isometri.

    Rumus F7: theta_i = m * omega_i, omega_i = base^(-2i / d_h)
    Rotasi 2x2 per pasangan (q_i, q_{i + d_h / 2}).
    @spec docs/milestones/M2-attention.md (§ Rumus F7)
    @spec scratch/wave/m2/m2-w2-rope.md
    """
    if seq_len <= 0:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"seq_len must be'
            ' positive","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    if cfg.hidden_size <= 0 or cfg.num_attention_heads <= 0:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"hidden_size and'
            ' num_attention_heads must be positive","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    var hidden = cfg.hidden_size
    var head_dim = cfg.head_dim()
    if cfg.num_attention_heads * head_dim != hidden:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"num_attention_heads *'
            ' head_dim != hidden_size","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    if head_dim % 2 != 0:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"head_dim must be'
            ' even","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"input length mismatch:'
            " expected "
            + String(seq_len * hidden)
            + " got "
            + String(len(x))
            + '","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    if base <= Float32(0.0):
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"base must be'
            ' positive","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )

    for i in range(len(x)):
        if isnan(x[i]) or isinf(x[i]):
            raise Error(
                '{"error_type":"ROPE_ERROR","detail":"non-finite value in'
                ' input","stage":"rope","layer":'
                + String(layer_idx)
                + "}"
            )

    var half_dim = head_dim // 2
    var inv_freq = List[Float32]()
    inv_freq.reserve(half_dim)
    for i in range(half_dim):
        var exp_val = Float32(2 * i) / Float32(head_dim)
        inv_freq.append(Float32(1.0) / pow(base, exp_val))

    var out = List[Float32]()
    out.resize(seq_len * hidden, Float32(0.0))

    var p_x = x.unsafe_ptr()
    var p_out = out.unsafe_ptr()
    var num_heads = cfg.num_attention_heads

    for t in range(seq_len):
        var m = Float32(pos_offset + t)
        var cos_m = List[Float32]()
        var sin_m = List[Float32]()
        cos_m.reserve(half_dim)
        sin_m.reserve(half_dim)
        for i in range(half_dim):
            var theta = m * inv_freq[i]
            cos_m.append(cos(theta))
            sin_m.append(sin(theta))
        var p_cos = cos_m.unsafe_ptr()
        var p_sin = sin_m.unsafe_ptr()

        for h in range(num_heads):
            var head_start = t * hidden + h * head_dim
            for i in range(half_dim):
                var x1 = p_x[unsafe_offset=head_start + i]
                var x2 = p_x[unsafe_offset=head_start + half_dim + i]
                var c = p_cos[unsafe_offset=i]
                var s = p_sin[unsafe_offset=i]
                var rot1 = x1 * c - x2 * s
                var rot2 = x1 * s + x2 * c
                if isnan(rot1) or isinf(rot1) or isnan(rot2) or isinf(rot2):
                    raise Error(
                        '{"error_type":"ROPE_ERROR","detail":"non-finite value'
                        ' in output","stage":"rope","layer":'
                        + String(layer_idx)
                        + "}"
                    )
                p_out[unsafe_offset=head_start + i] = rot1
                p_out[unsafe_offset=head_start + half_dim + i] = rot2

    verify_rope_isometry(x, out, seq_len, cfg, layer_idx)
    return out^


def apply_rope(
    q: List[Float32],
    k: List[Float32],
    seq_len: Int,
    cfg: ModelConfig,
    pos_offset: Int = 0,
    base: Float32 = Float32(1000000.0),
    layer_idx: Int = 0,
) raises -> Tuple[List[Float32], List[Float32]]:
    """Terapkan RoPE rotate_half F7 pada Q dan K."""
    var q_rot = rope_rotate_half(q, seq_len, cfg, pos_offset, base, layer_idx)
    var k_rot = rope_rotate_half(k, seq_len, cfg, pos_offset, base, layer_idx)
    return (q_rot^, k_rot^)
