# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Hybrid Transformer Macro Scheduler untuk Port Qwen3.6 (M9-W2).

Mengimplementasikan arsitektur 40 block hybrid:
10 Siklus Makro * [3 * (GDN + MoE) + 1 * (GatedAttn + MoE)]
- 30 State GDN independen per-layer S[0..29] berukuran [30, dv, dk] (bukan shared state)
- KV cache hanya di GatedAttn (10 layer, GDN no-op KV)
- MoE di setiap 40 block (top-8 routed + 1 shared expert)
"""

from core.config import ModelConfig
from layers.gated_attention import (
    GatedAttentionWeights,
    GatedAttnKVCache,
    gated_attention_forward,
    linear_projection,
)
from layers.gdn import (
    GDNState,
    ProjectedKVBeta,
    chunked_gdn_scan,
    project_tokens_to_kv_beta,
)
from layers.moe import moe_aggregate_forward, shared_gate_forward
from layers.rmsnorm import rmsnorm
from layers.router import router_forward
from layers.router_types import RouterConfig, RoutingInfo
from layers.swiglu import SwigluWeights, swiglu_forward
from std.collections import List
from std.math import isinf, isnan
from std.time import perf_counter_ns


struct SchedulerTimings(Copyable, Movable):
    """Pengukur waktu eksekusi sublayer mikro/nanosekon untuk profil performa.
    """

    var gdn_ns: Int
    var gated_attn_ns: Int
    var moe_ns: Int

    def __init__(out self):
        self.gdn_ns = 0
        self.gated_attn_ns = 0
        self.moe_ns = 0

    def __init__(out self, gdn_ns: Int, gated_attn_ns: Int, moe_ns: Int):
        self.gdn_ns = gdn_ns
        self.gated_attn_ns = gated_attn_ns
        self.moe_ns = moe_ns


@fieldwise_init
struct PortBlockWeights(Movable):
    """Bobot parameter lengkap untuk 1 block transformer hybrid."""

    var input_layernorm_gamma: List[Float32]
    var post_attention_layernorm_gamma: List[Float32]
    var is_linear_attn: Bool

    # GDN parameter (jika is_linear_attn)
    var gdn_w_k: List[Float32]
    var gdn_w_v: List[Float32]
    var gdn_w_beta: List[Float32]
    var gdn_w_out: List[Float32]

    # Gated Attention parameter (jika bukan is_linear_attn)
    var gated_attn: GatedAttentionWeights

    # MoE parameter (hadir di SETIAP block)
    var w_router: List[Float32]
    var routed_experts: List[SwigluWeights]
    var shared_expert: SwigluWeights
    var w_shared_gate: List[Float32]


def create_synthetic_block_weights(
    cfg: ModelConfig, layer_idx: Int, dv: Int = 32, dk: Int = 32
) raises -> PortBlockWeights:
    """Membuat bobot sintetis terinisialisasi deterministik untuk satu block."""
    var hidden = cfg.hidden_size
    var is_linear = cfg.is_linear_attn_layer(layer_idx)

    var in_norm = List[Float32]()
    in_norm.resize(hidden, Float32(1.0))
    var post_norm = List[Float32]()
    post_norm.resize(hidden, Float32(1.0))

    var gdn_wk = List[Float32]()
    var gdn_wv = List[Float32]()
    var gdn_wbeta = List[Float32]()
    var gdn_wout = List[Float32]()

    if is_linear:
        gdn_wk.resize(dk * hidden, Float32(0.01))
        gdn_wv.resize(dv * hidden, Float32(0.01))
        gdn_wbeta.resize(hidden, Float32(0.01))
        gdn_wout.resize(hidden * dv, Float32(0.01))
        for i in range(min(dk, hidden)):
            gdn_wk[i * hidden + i] = Float32(0.1)
        for i in range(min(dv, hidden)):
            gdn_wv[i * hidden + i] = Float32(0.1)
            gdn_wout[i * dv + i] = Float32(0.1)

    var h_q = cfg.num_attention_heads
    var h_kv = cfg.num_key_value_heads
    var head_dim = cfg.head_dim()
    var q_dim = h_q * head_dim
    var kv_dim = h_kv * head_dim

    var w_q = List[Float32]()
    var b_q = List[Float32]()
    var w_k = List[Float32]()
    var b_k = List[Float32]()
    var w_v = List[Float32]()
    var b_v = List[Float32]()
    var w_gate = List[Float32]()
    var b_gate = List[Float32]()
    var w_o = List[Float32]()
    var b_o = List[Float32]()

    if not is_linear:
        w_q.resize(q_dim * hidden, Float32(0.01))
        w_k.resize(kv_dim * hidden, Float32(0.01))
        w_v.resize(kv_dim * hidden, Float32(0.01))
        w_gate.resize(q_dim * hidden, Float32(0.01))
        w_o.resize(hidden * q_dim, Float32(0.01))
        for i in range(min(q_dim, hidden)):
            w_q[i * hidden + i] = Float32(0.1)
            w_gate[i * hidden + i] = Float32(0.05)
            w_o[i * q_dim + i] = Float32(0.1)
        for i in range(min(kv_dim, hidden)):
            w_k[i * hidden + i] = Float32(0.1)
            w_v[i * hidden + i] = Float32(0.1)

    var gated_attn = GatedAttentionWeights(
        w_q^, b_q^, w_k^, b_k^, w_v^, b_v^, w_gate^, b_gate^, w_o^, b_o^
    )

    # MoE: router, experts, shared
    var num_exp = cfg.num_experts
    var inter_dim = cfg.moe_intermediate_size
    var sh_inter_dim = cfg.shared_expert_intermediate_size

    var w_router = List[Float32]()
    w_router.resize(num_exp * hidden, Float32(0.01))
    for e in range(num_exp):
        w_router[e * hidden + (e % hidden)] = Float32(0.2)

    var routed_experts = List[SwigluWeights]()
    for _ in range(num_exp):
        var wg = List[Float32]()
        wg.resize(inter_dim * hidden, Float32(0.01))
        var wu = List[Float32]()
        wu.resize(inter_dim * hidden, Float32(0.01))
        var wd = List[Float32]()
        wd.resize(hidden * inter_dim, Float32(0.01))
        for i in range(min(inter_dim, hidden)):
            wg[i * hidden + i] = Float32(0.05)
            wu[i * hidden + i] = Float32(0.05)
            wd[i * inter_dim + i] = Float32(0.05)
        routed_experts.append(SwigluWeights(wg^, wu^, wd^, hidden, inter_dim))

    var sh_wg = List[Float32]()
    sh_wg.resize(sh_inter_dim * hidden, Float32(0.01))
    var sh_wu = List[Float32]()
    sh_wu.resize(sh_inter_dim * hidden, Float32(0.01))
    var sh_wd = List[Float32]()
    sh_wd.resize(hidden * sh_inter_dim, Float32(0.01))
    for i in range(min(sh_inter_dim, hidden)):
        sh_wg[i * hidden + i] = Float32(0.05)
        sh_wu[i * hidden + i] = Float32(0.05)
        sh_wd[i * sh_inter_dim + i] = Float32(0.05)
    var shared_expert = SwigluWeights(
        sh_wg^, sh_wu^, sh_wd^, hidden, sh_inter_dim
    )

    var w_shared_gate = List[Float32]()
    w_shared_gate.resize(hidden, Float32(0.01))
    w_shared_gate[0] = Float32(0.5)

    return PortBlockWeights(
        in_norm^,
        post_norm^,
        is_linear,
        gdn_wk^,
        gdn_wv^,
        gdn_wbeta^,
        gdn_wout^,
        gated_attn^,
        w_router^,
        routed_experts^,
        shared_expert^,
        w_shared_gate^,
    )


def compute_gdn_block_output(
    mut gdn_states: GDNState,
    gdn_idx: Int,
    k_all: List[Float32],
    w_out: List[Float32],
    seq_len: Int,
    dk: Int,
    dv: Int,
    hidden: Int,
) raises -> List[Float32]:
    """Menghitung aktivasi token output dari state GDN terisolasi: y = (S_t @ k_t) @ W_out^T.
    """
    var out = List[Float32]()
    out.resize(seq_len * hidden, Float32(0.0))
    var p_out = out.unsafe_ptr()
    var p_k = k_all.unsafe_ptr()
    var p_wout = w_out.unsafe_ptr()

    var ot = List[Float32]()
    ot.resize(dv, Float32(0.0))
    var p_ot = ot.unsafe_ptr()

    for t in range(seq_len):
        var k_off = t * dk
        # S_t @ k_t -> [dv]
        for r in range(dv):
            var acc = Float32(0.0)
            for c in range(dk):
                acc += (
                    gdn_states.get(gdn_idx, r, c) * p_k[unsafe_offset=k_off + c]
                )
            p_ot[unsafe_offset=r] = acc

        # ot @ W_out^T -> [hidden]
        var out_off = t * hidden
        for h in range(hidden):
            var w_off = h * dv
            var acc_w = Float32(0.0)
            for r in range(dv):
                acc_w += p_ot[unsafe_offset=r] * p_wout[unsafe_offset=w_off + r]
            p_out[unsafe_offset=out_off + h] = acc_w

    return out^


def apply_rmsnorm_sequence(
    x: List[Float32],
    gamma: List[Float32],
    seq_len: Int,
    hidden: Int,
    eps: Float32,
) raises -> List[Float32]:
    """Menerapkan RMSNorm per token pada sekuens berukuran [seq_len, hidden]."""
    var out = List[Float32]()
    out.reserve(seq_len * hidden)
    var tok_vec = List[Float32]()
    tok_vec.resize(hidden, Float32(0.0))
    var p_tok = tok_vec.unsafe_ptr()
    var p_x = x.unsafe_ptr()

    for t in range(seq_len):
        var row = t * hidden
        for k in range(hidden):
            p_tok[unsafe_offset=k] = p_x[unsafe_offset=row + k]
        var normed = rmsnorm(tok_vec, gamma, eps)
        var p_normed = normed.unsafe_ptr()
        for k in range(hidden):
            out.append(p_normed[unsafe_offset=k])
    return out^


def forward_port_block(
    x: List[Float32],
    block: PortBlockWeights,
    mut gdn_states: GDNState,
    mut kv_cache: GatedAttnKVCache,
    layer_idx: Int,
    pos_offset: Int,
    seq_len: Int,
    cfg: ModelConfig,
    mut timings: SchedulerTimings,
    dk: Int = 32,
    dv: Int = 32,
    eps: Float32 = Float32(1e-6),
) raises -> List[Float32]:
    """Forward pass 1 block transformer hybrid (Token Mixer + Channel Mixer)."""
    var hidden = cfg.hidden_size

    # --- Sublayer 1: Token Mixer ---
    var norm_x = apply_rmsnorm_sequence(
        x, block.input_layernorm_gamma, seq_len, hidden, eps
    )
    var mixer_out: List[Float32]

    if block.is_linear_attn:
        var t_gdn_start = perf_counter_ns()
        # Indeks isolasi GDN kanonis: 3 * (l // 4) + (l % 4)
        var gdn_idx = 3 * (layer_idx // 4) + (layer_idx % 4)

        # 1. Proyeksi K, V, Beta
        var proj = project_tokens_to_kv_beta(
            norm_x,
            block.gdn_w_k,
            block.gdn_w_v,
            block.gdn_w_beta,
            seq_len,
            hidden,
            dk,
            dv,
        )

        # 2. Chunked WY recurrence scan pada S[gdn_idx]
        chunked_gdn_scan(
            gdn_states,
            gdn_idx,
            proj.k_mat,
            proj.v_mat,
            proj.beta,
            seq_len,
            dk,
            dv,
        )

        # 3. Hitung token output mixer
        mixer_out = compute_gdn_block_output(
            gdn_states,
            gdn_idx,
            proj.k_mat,
            block.gdn_w_out,
            seq_len,
            dk,
            dv,
            hidden,
        )
        var t_gdn_end = perf_counter_ns()
        timings.gdn_ns += Int(t_gdn_end - t_gdn_start)
    else:
        var t_attn_start = perf_counter_ns()
        # Indeks perhatian kanonis: l // 4
        var att_idx = layer_idx // 4

        # Gated Attention dengan GQA 16Q/2KV dan 5D KV Cache
        mixer_out = gated_attention_forward(
            norm_x,
            block.gated_attn,
            kv_cache,
            att_idx,
            pos_offset,
            seq_len,
            cfg,
            layer_idx,
        )
        var t_attn_end = perf_counter_ns()
        timings.gated_attn_ns += Int(t_attn_end - t_attn_start)

    # Residual 1: x = x + mixer_out
    var x_mid = List[Float32]()
    x_mid.resize(seq_len * hidden, Float32(0.0))
    var p_x = x.unsafe_ptr()
    var p_m = mixer_out.unsafe_ptr()
    var p_mid = x_mid.unsafe_ptr()
    for i in range(seq_len * hidden):
        p_mid[unsafe_offset=i] = p_x[unsafe_offset=i] + p_m[unsafe_offset=i]

    # --- Sublayer 2: Channel Mixer (MoE MLP pada SETIAP block) ---
    var t_moe_start = perf_counter_ns()
    var norm_x2 = apply_rmsnorm_sequence(
        x_mid, block.post_attention_layernorm_gamma, seq_len, hidden, eps
    )

    var router_cfg = RouterConfig(
        cfg.num_experts, cfg.num_experts_per_tok, cfg.norm_topk_prob
    )
    var routing = router_forward(
        norm_x2, block.w_router, seq_len, hidden, router_cfg, layer_idx
    )

    # Evaluasi routed experts yang terpilih
    var routed_outputs = List[List[Float32]]()
    routed_outputs.reserve(seq_len)

    for t in range(seq_len):
        var t_off = t * hidden
        var x_tok = List[Float32]()
        x_tok.resize(hidden, Float32(0.0))
        for d in range(hidden):
            x_tok[d] = norm_x2[t_off + d]

        var token_routed = List[Float32]()
        ref exp_row = routing.selected_experts[t]
        var top_k = len(exp_row)
        token_routed.reserve(top_k * hidden)

        for k in range(top_k):
            var exp_id = exp_row[k]
            ref exp_w = block.routed_experts[exp_id]
            var out_exp = swiglu_forward(
                x_tok, exp_w, 1, hidden, exp_w.inter_dim
            )
            for d in range(hidden):
                token_routed.append(out_exp[d])
        routed_outputs.append(token_routed^)

    # Shared Expert SwiGLU + Sigmoid Gate
    var shared_out = swiglu_forward(
        norm_x2,
        block.shared_expert,
        seq_len,
        hidden,
        block.shared_expert.inter_dim,
    )
    var sh_gate_scores = shared_gate_forward(
        norm_x2, block.w_shared_gate, seq_len, hidden, layer_idx, "sigmoid"
    )

    # Agregasi MoE (top-k routed + 1 shared + x_mid)
    var res = moe_aggregate_forward(
        x_mid,
        routed_outputs,
        routing.router_probs,
        shared_out,
        sh_gate_scores,
        seq_len,
        hidden,
        layer_idx,
    )
    var t_moe_end = perf_counter_ns()
    timings.moe_ns += Int(t_moe_end - t_moe_start)
    return res^


def forward_port_block(
    x: List[Float32],
    block: PortBlockWeights,
    mut gdn_states: GDNState,
    mut kv_cache: GatedAttnKVCache,
    layer_idx: Int,
    pos_offset: Int,
    seq_len: Int,
    cfg: ModelConfig,
    dk: Int = 32,
    dv: Int = 32,
    eps: Float32 = Float32(1e-6),
) raises -> List[Float32]:
    """Forward pass 1 block transformer hybrid tanpa akumulator timing."""
    var dummy_timings = SchedulerTimings()
    return forward_port_block(
        x,
        block,
        gdn_states,
        kv_cache,
        layer_idx,
        pos_offset,
        seq_len,
        cfg,
        dummy_timings,
        dk,
        dv,
        eps,
    )


def forward_port_macro_scheduler(
    x: List[Float32],
    blocks: List[PortBlockWeights],
    mut gdn_states: GDNState,
    mut kv_cache: GatedAttnKVCache,
    pos_offset: Int,
    seq_len: Int,
    cfg: ModelConfig,
    mut timings: SchedulerTimings,
    dk: Int = 32,
    dv: Int = 32,
    eps: Float32 = Float32(1e-6),
) raises -> List[Float32]:
    """Mengeksekusi macro scheduler transformer penuh dengan tracking profil waktu.
    """
    var cur_x = List[Float32]()
    cur_x.resize(len(x), Float32(0.0))
    for i in range(len(x)):
        cur_x[i] = x[i]

    var num_layers = cfg.num_hidden_layers
    for layer_idx in range(num_layers):
        cur_x = forward_port_block(
            cur_x,
            blocks[layer_idx],
            gdn_states,
            kv_cache,
            layer_idx,
            pos_offset,
            seq_len,
            cfg,
            timings,
            dk,
            dv,
            eps,
        )

    return cur_x^


def forward_port_macro_scheduler(
    x: List[Float32],
    blocks: List[PortBlockWeights],
    mut gdn_states: GDNState,
    mut kv_cache: GatedAttnKVCache,
    pos_offset: Int,
    seq_len: Int,
    cfg: ModelConfig,
    dk: Int = 32,
    dv: Int = 32,
    eps: Float32 = Float32(1e-6),
) raises -> List[Float32]:
    """Mengeksekusi macro scheduler transformer penuh (40 block atau mini)."""
    var dummy_timings = SchedulerTimings()
    return forward_port_macro_scheduler(
        x,
        blocks,
        gdn_states,
        kv_cache,
        pos_offset,
        seq_len,
        cfg,
        dummy_timings,
        dk,
        dv,
        eps,
    )
