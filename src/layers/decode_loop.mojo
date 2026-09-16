# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Manajemen konteks posisi decode, baseline recompute, dan lokalisasi verifikasi per-layer (M5-W2)."""

from cli.m5_errors import m5_error_json
from core.config import ModelConfig
from layers.attention import AttentionWeights, o_project
from layers.kv_cache import LayerKVCache
from layers.mha import mha_decode_step, mha_forward
from layers.qkv import qkv_forward
from layers.rmsnorm import rmsnorm
from layers.rope import apply_rope
from std.collections import List
from std.math import abs, isinf, isnan, sqrt


@fieldwise_init
struct DecodeStepContext(Copyable, Movable):
    """Kontrak posisi eksplisit (DoD M5: label t-1 DILARANG).

    Notasi terkunci:
    - S: panjang prompt
    - N: jumlah token generate maksimal
    - ctx: batas context size teralokasi
    - i: indeks iterasi decode (0..N-1)
    - g: jumlah token ter-generate sejauh ini (0..N)
    - p: posisi sekuens absolut (p = S + g)
    """

    var prompt_len: Int
    var max_tokens: Int
    var context_size: Int
    var step_idx: Int
    var tokens_generated: Int
    var current_pos: Int

    def __init__(out self, prompt_len: Int, max_tokens: Int, context_size: Int):
        self.prompt_len = prompt_len
        self.max_tokens = max_tokens
        self.context_size = context_size
        self.step_idx = 0
        self.tokens_generated = 0
        self.current_pos = prompt_len

    def assert_step_invariants(self, cache_len_before_step: Int) raises:
        """Menegakkan invarian debug assertions wajib DoD M5.

        cache_len_before_step = S + g
        input_position        = cache_len_before_step  (= p, bukan t-1)
        cache_len_after_step  = cache_len_before_step + 1 <= ctx
        """
        var expected_len = self.prompt_len + self.tokens_generated
        if cache_len_before_step != expected_len:
            var details = String(
                '{"expected_len":',
                String(expected_len),
                ',"actual_cache_len":',
                String(cache_len_before_step),
                ',"step":',
                String(self.step_idx),
                "}",
            )
            raise Error(
                m5_error_json(
                    "M5_ERR_DECODE",
                    "decode",
                    "Invariant violation: cache_len_before_step != S + g",
                    details,
                )
            )

        if self.current_pos != cache_len_before_step:
            var details = String(
                '{"current_pos":',
                String(self.current_pos),
                ',"cache_len_before":',
                String(cache_len_before_step),
                "}",
            )
            raise Error(
                m5_error_json(
                    "M5_ERR_DECODE",
                    "decode",
                    (
                        "Invariant violation: input_position !="
                        " cache_len_before_step"
                    ),
                    details,
                )
            )

        var after_step = cache_len_before_step + 1
        if after_step > self.context_size:
            var details = String(
                '{"after_step":',
                String(after_step),
                ',"context_size":',
                String(self.context_size),
                "}",
            )
            raise Error(
                m5_error_json(
                    "M5_ERR_CONTEXT_SIZE",
                    "kv_alloc",
                    "Invariant violation: cache_len_after_step > ctx",
                    details,
                )
            )

    def advance_step(mut self):
        """Memajukan iterasi decode 1 langkah: i+1, g+1, p+1."""
        self.step_idx += 1
        self.tokens_generated += 1
        self.current_pos += 1


def compare_tensors_loose(
    tensor_a: List[Float32],
    tensor_b: List[Float32],
    tensor_class: String,
    layer_idx: Int,
    delta_max_tol: Float32 = Float32(0.01),
    eps_rel_tol: Float32 = Float32(0.005),
) raises:
    """Membandingkan dua tensor fp32 dengan toleransi loose F10.

    Format kegagalan wajib: `layer L, <K|V|Q|attn-out|moe-out> mismatch`.
    """
    if len(tensor_a) != len(tensor_b):
        raise Error(
            String(
                "layer ",
                String(layer_idx),
                ", ",
                tensor_class,
                " mismatch: length mismatch (",
                String(len(tensor_a)),
                " vs ",
                String(len(tensor_b)),
                ")",
            )
        )

    var max_diff = Float32(0.0)
    var sum_sq_diff = Float32(0.0)
    var sum_sq_ref = Float32(0.0)

    var p_a = tensor_a.unsafe_ptr()
    var p_b = tensor_b.unsafe_ptr()
    for idx in range(len(tensor_a)):
        var val_a = p_a[unsafe_offset=idx]
        var val_b = p_b[unsafe_offset=idx]
        if isnan(val_a) or isinf(val_a) or isnan(val_b) or isinf(val_b):
            raise Error(
                String(
                    "layer ",
                    String(layer_idx),
                    ", ",
                    tensor_class,
                    " mismatch: non-finite value detected",
                )
            )
        var diff = abs(val_a - val_b)
        if diff > max_diff:
            max_diff = diff
        sum_sq_diff += diff * diff
        sum_sq_ref += val_b * val_b

    var eps_rel = Float32(0.0)
    if sum_sq_ref > Float32(0.0):
        eps_rel = sqrt(sum_sq_diff / sum_sq_ref)

    if max_diff > delta_max_tol or eps_rel > eps_rel_tol:
        raise Error(
            String(
                "layer ",
                String(layer_idx),
                ", ",
                tensor_class,
                " mismatch: delta_max=",
                String(max_diff),
                " > ",
                String(delta_max_tol),
                ", eps_rel=",
                String(eps_rel),
                " > ",
                String(eps_rel_tol),
            )
        )


def recompute_attention_at_position(
    full_sequence_x: List[Float32],
    weights: AttentionWeights,
    pos: Int,
    cfg: ModelConfig,
    eps: Float32,
    base: Float32 = Float32(1000000.0),
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Baseline full recompute attention: menghitung ulang semua K/V [0..pos] dari embedding.

    Invarian inti M5:
    recompute attention at position p == incremental attention using cached K/V[0:p) + Q[p].
    """
    var hidden = cfg.hidden_size
    var seq_len = pos + 1
    if len(full_sequence_x) != seq_len * hidden:
        raise Error("full_sequence_x length must equal (pos + 1) * hidden")

    # 1. RMSNorm untuk semua posisi 0..pos
    var x_norm = List[Float32]()
    x_norm.reserve(seq_len * hidden)
    var tok_vec = List[Float32]()
    tok_vec.resize(hidden, Float32(0.0))
    var p_tok = tok_vec.unsafe_ptr()
    var p_raw = full_sequence_x.unsafe_ptr()

    for t in range(seq_len):
        var row = t * hidden
        for k in range(hidden):
            p_tok[unsafe_offset=k] = p_raw[unsafe_offset=row + k]
        var normed = rmsnorm(tok_vec, weights.norm_gamma, eps)
        var p_normed = normed.unsafe_ptr()
        for k in range(hidden):
            x_norm.append(p_normed[unsafe_offset=k])

    # 2. QKV projection untuk semua posisi 0..pos
    var qkv_res = qkv_forward(x_norm, weights.qkv, seq_len, cfg)
    ref q = qkv_res[0]
    ref k = qkv_res[1]
    ref v = qkv_res[2]

    # 3. RoPE untuk semua posisi 0..pos
    var rope_res = apply_rope(q, k, seq_len, cfg, 0, base, layer_idx)
    ref q_rot = rope_res[0]
    ref k_rot = rope_res[1]

    # 4. MHA causal untuk semua posisi
    var full_attn_out = mha_forward(q_rot, k_rot, v, seq_len, cfg, layer_idx)

    # 5. Ekstraksi output token pada posisi spesifik p
    var out_pos = List[Float32]()
    out_pos.resize(hidden, Float32(0.0))
    var p_full = full_attn_out.unsafe_ptr()
    var p_out = out_pos.unsafe_ptr()
    var base_row = pos * hidden
    for k in range(hidden):
        p_out[unsafe_offset=k] = p_full[unsafe_offset=base_row + k]

    # 6. Proyeksi o_proj untuk 1 token pada posisi p
    var y = o_project(out_pos, weights.w_o, weights.b_o, 1, hidden, layer_idx)
    return y^
