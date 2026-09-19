# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Pipeline layer forward streaming: Attention -> Residual 1 -> MoE -> Residual 2 (M4-W2)."""

from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import ShardHeaderCache, _load_one_tensor_by_name
from layers.attention import (
    AttentionWeights,
    forward_attention_block,
    load_layer_attention_weights,
    o_project,
)
from layers.gdn import (
    GDNWeights,
    forward_gdn_block,
    load_layer_gdn_weights,
)
from layers.kv_cache import LayerKVCache
from layers.mha import mha_decode_step, mha_forward
from layers.moe import shared_gate_forward
from layers.moe_loader import (
    load_layer_routed_expert_weights,
    load_layer_shared_expert_weights,
)
from layers.qkv import qkv_forward
from layers.residual import add_residual
from layers.rmsnorm import rmsnorm
from layers.rope import apply_rope
from layers.router import load_layer_router_weights, router_forward
from layers.router_types import RouterConfig, RoutingInfo
from layers.swiglu import SwigluWeights, swiglu_forward
from std.collections import Dict, List
from std.math import isinf, isnan
from std.time import perf_counter_ns


@fieldwise_init
struct LayerTiming(Copyable, Movable):
    """Metrik waktu satu layer streaming untuk --layer-timing."""

    var layer: Int
    var pread_sec: Float64
    var attention_sec: Float64
    var moe_sec: Float64
    var total_sec: Float64


def _format_1d_ints(data: List[Int]) -> String:
    var s = String("[")
    for j in range(len(data)):
        if j > 0:
            s += ","
        s += String(data[j])
    s += "]"
    return s


def _format_2d_ints(data: List[List[Int]]) -> String:
    var s = String("[")
    for i in range(len(data)):
        if i > 0:
            s += ","
        s += _format_1d_ints(data[i])
    s += "]"
    return s


def forward_attention_step(
    x: List[Float32],
    weights: AttentionWeights,
    seq_len: Int,
    cfg: ModelConfig,
    eps: Float32,
    pos_offset: Int = 0,
    base: Float32 = Float32(1000000.0),
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Pipeline blok attention TANPA penambahan residual.

    RMSNorm (F6) -> QKV (+bias) -> RoPE rotate_half (F7) -> MHA Causal -> o_proj.
    Residual connection dieksekusi terpisah di caller (Residual 1).
    """
    var hidden = cfg.hidden_size
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"activation length'
            ' mismatch","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    # 1. RMSNorm input per token (F6)
    var x_norm = List[Float32]()
    x_norm.reserve(seq_len * hidden)
    var tok_vec = List[Float32]()
    tok_vec.resize(hidden, Float32(0.0))
    var p_tok = tok_vec.unsafe_ptr()
    var p_raw_x = x.unsafe_ptr()
    for t in range(seq_len):
        var row = t * hidden
        for k in range(hidden):
            p_tok[unsafe_offset=k] = p_raw_x[unsafe_offset=row + k]
        var normed = rmsnorm(tok_vec, weights.norm_gamma, eps)
        var p_normed = normed.unsafe_ptr()
        for k in range(hidden):
            x_norm.append(p_normed[unsafe_offset=k])

    # 2. QKV Projection dengan bias q/k/v
    var qkv_res = qkv_forward(x_norm, weights.qkv, seq_len, cfg)
    ref q = qkv_res[0]
    ref k = qkv_res[1]
    ref v = qkv_res[2]

    # 3. RoPE rotate_half (F7)
    var rope_res = apply_rope(q, k, seq_len, cfg, pos_offset, base, layer_idx)
    ref q_rot = rope_res[0]
    ref k_rot = rope_res[1]

    # 4. MHA dengan Causal Mask & Softmax Stabil
    var attn_out = mha_forward(q_rot, k_rot, v, seq_len, cfg, layer_idx)

    # 5. Output projection o_proj (tanpa bias)
    var y = o_project(
        attn_out, weights.w_o, weights.b_o, seq_len, hidden, layer_idx
    )

    return y^


def forward_attention_decode_step(
    x: List[Float32],
    weights: AttentionWeights,
    mut layer_kv: LayerKVCache,
    pos: Int,
    cfg: ModelConfig,
    eps: Float32,
    base: Float32 = Float32(1000000.0),
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Pipeline blok attention incremental decode untuk 1 token pada posisi sekuens p.

    RMSNorm -> QKV -> RoPE(pos) -> append K/V(pos) -> mha_decode_step -> o_proj.
    Residual connection dieksekusi terpisah di caller (Residual 1).
    """
    var hidden = cfg.hidden_size
    if len(x) != hidden:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"activation length'
            ' mismatch against hidden_size","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    # 1. RMSNorm input 1 token (F6)
    var x_norm = rmsnorm(x, weights.norm_gamma, eps)

    # 2. QKV Projection dengan bias q/k/v (seq_len = 1)
    var qkv_res = qkv_forward(x_norm, weights.qkv, 1, cfg)
    ref q = qkv_res[0]
    ref k = qkv_res[1]
    ref v = qkv_res[2]

    # 3. RoPE rotate_half pada posisi sekuens absolut p (F7)
    var rope_res = apply_rope(q, k, 1, cfg, pos, base, layer_idx)
    ref q_rot = rope_res[0]
    ref k_rot = rope_res[1]

    # 4. Append K_rot dan V baru ke KV cache pada posisi p
    # Invarian: layer_kv.current_len sebelum append adalah pos; setelah append adalah pos + 1
    layer_kv.append_decode_token(k_rot, v, pos)

    # 5. Incremental MHA terhadap histori [0, pos + 1) yang tersimpan di layer_kv
    var attn_out = mha_decode_step(q_rot, layer_kv, pos + 1, cfg, layer_idx)

    # 6. Output projection o_proj (tanpa bias)
    var y = o_project(attn_out, weights.w_o, weights.b_o, 1, hidden, layer_idx)
    return y^


def moe_combine_no_residual(
    routed_outputs: List[List[Float32]],
    router_probs: List[List[Float32]],
    shared_output: List[Float32],
    shared_gate_scores: List[Float32],
    seq_len: Int,
    hidden_dim: Int,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Agregasi MoE TANPA penambahan residual.

    y = sum_{k in top-4} p_k E_{i_k}(x) + sigma(g_sh) E_sh(x).
    Residual connection dieksekusi terpisah di caller (Residual 2).
    """
    if seq_len <= 0 or hidden_dim <= 0:
        raise Error(
            '{"error_type":"EXPERT_ERROR","detail":"dimensions must be'
            ' positive","stage":"aggregation","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(shared_output) != seq_len * hidden_dim:
        raise Error(
            '{"error_type":"EXPERT_ERROR","detail":"length mismatch in'
            ' aggregation","stage":"aggregation","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(shared_gate_scores) != seq_len or len(routed_outputs) != seq_len:
        raise Error(
            '{"error_type":"EXPERT_ERROR","detail":"sequence length'
            ' mismatch","stage":"aggregation","layer":'
            + String(layer_idx)
            + "}"
        )

    var out = List[Float32]()
    out.reserve(seq_len * hidden_dim)
    var p_sh_out = shared_output.unsafe_ptr()

    for t in range(seq_len):
        var x_base = t * hidden_dim
        var gate_sh = shared_gate_scores[t]
        ref r_outs = routed_outputs[t]
        ref probs = router_probs[t]
        var top_k = len(probs)

        for d in range(hidden_dim):
            var routed_sum = Float32(0.0)
            for k in range(top_k):
                var p = probs[k]
                var val = r_outs[k * hidden_dim + d]
                routed_sum += p * val

            var sh_val = gate_sh * p_sh_out[unsafe_offset=x_base + d]
            var total = routed_sum + sh_val

            if isnan(total) or isinf(total):
                raise Error(
                    '{"error_type":"EXPERT_ERROR","detail":"non-finite value in'
                    ' aggregation","stage":"aggregation","layer":'
                    + String(layer_idx)
                    + "}"
                )
            out.append(total)

    return out^


def forward_single_layer(
    mut hidden: List[Float32],
    layer_idx: Int,
    seq_len: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    eps: Float32,
    mut cache: ShardHeaderCache,
    mut telemetry: LoadMemoryTelemetry,
    dump_routing_dir: String = "",
) raises -> LayerTiming:
    """Menjalankan full streaming satu layer transformer (0..39).

    1. Attention pread -> RMSNorm input -> Attention step -> Residual 1
    2. MoE pread norm -> RMSNorm post_attention -> Router -> Experts SwiGLU -> Shared SwiGLU -> MoE combine -> Residual 2
    3. Discard bobot layer buffer (zero accumulation)
    4. Catat waktu per tahap untuk telemetri --layer-timing.
    """
    var t_layer_start = perf_counter_ns()
    var pread_ns: Int = 0
    var attn_ns: Int = 0
    var moe_ns: Int = 0

    # -------------------------------------------------------------
    # TAHAP 1: TOKEN MIXER (GDN atau Gated Attention)
    # -------------------------------------------------------------
    var is_linear = cfg.is_linear_attn_layer(layer_idx)
    if is_linear:
        var t_gdn_pread0 = perf_counter_ns()
        var gdn_weights = load_layer_gdn_weights(
            layer_idx, model_root, weight_map, cfg, cache, telemetry
        )
        pread_ns += perf_counter_ns() - t_gdn_pread0

        var t_gdn_comp0 = perf_counter_ns()
        hidden = forward_gdn_block(
            hidden, gdn_weights, seq_len, cfg, eps, layer_idx
        )
        attn_ns += perf_counter_ns() - t_gdn_comp0
    else:
        var t_att_pread0 = perf_counter_ns()
        var attn_weights = load_layer_attention_weights(
            layer_idx, model_root, weight_map, cfg, cache, telemetry
        )
        pread_ns += perf_counter_ns() - t_att_pread0

        var t_att_comp0 = perf_counter_ns()
        hidden = forward_attention_block(
            hidden,
            attn_weights,
            seq_len,
            cfg,
            eps,
            pos_offset=0,
            base=cfg.rope_theta,
            layer_idx=layer_idx,
        )
        attn_ns += perf_counter_ns() - t_att_comp0

    # -------------------------------------------------------------
    # TAHAP 2: MoE (M3)
    # -------------------------------------------------------------
    # 2a. Pread post_attention_layernorm.weight
    var t_moe_pread0 = perf_counter_ns()
    var prefix = "model.language_model.layers." + String(layer_idx) + "."
    if (prefix + "post_attention_layernorm.weight") not in weight_map:
        prefix = "model.layers." + String(layer_idx) + "."
    var norm2_name = prefix + "post_attention_layernorm.weight"
    if norm2_name not in weight_map:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"post_attention_layernorm'
            ' not in weight_map","stage":"layer_forward","layer":'
            + String(layer_idx)
            + "}"
        )
    var norm2_shard = weight_map[norm2_name]
    var norm2_gamma = _load_one_tensor_by_name(
        cache,
        model_root,
        norm2_shard,
        norm2_name,
        cfg.hidden_size,
        1,
        False,
        telemetry,
    )
    pread_ns += perf_counter_ns() - t_moe_pread0

    # 2b. RMSNorm post-attention
    var t_moe_comp0 = perf_counter_ns()
    var x_norm_moe = List[Float32]()
    x_norm_moe.reserve(seq_len * cfg.hidden_size)
    var tok_vec = List[Float32]()
    tok_vec.resize(cfg.hidden_size, Float32(0.0))
    var p_tok = tok_vec.unsafe_ptr()
    var p_h = hidden.unsafe_ptr()
    for t in range(seq_len):
        var row = t * cfg.hidden_size
        for k in range(cfg.hidden_size):
            p_tok[unsafe_offset=k] = p_h[unsafe_offset=row + k]
        var normed = rmsnorm(tok_vec, norm2_gamma, eps)
        var p_normed = normed.unsafe_ptr()
        for k in range(cfg.hidden_size):
            x_norm_moe.append(p_normed[unsafe_offset=k])
    moe_ns += perf_counter_ns() - t_moe_comp0

    # 2c. Router pread & forward
    var t_r_pread0 = perf_counter_ns()
    var router_cfg = RouterConfig(
        cfg.num_experts, cfg.num_experts_per_tok, cfg.norm_topk_prob
    )
    var w_router = load_layer_router_weights(
        layer_idx, model_root, weight_map, cfg, router_cfg, cache, telemetry
    )
    pread_ns += perf_counter_ns() - t_r_pread0

    var t_r_comp0 = perf_counter_ns()
    var routing = router_forward(
        x_norm_moe, w_router, seq_len, cfg.hidden_size, router_cfg, layer_idx
    )

    # Dump routing jika diminta (--dump-routing)
    if dump_routing_dir.byte_length() > 0:
        var dump_file = String(
            dump_routing_dir, "/routing_L", String(layer_idx), ".json"
        )
        try:
            var f = open(dump_file, "w")
            f.write(
                String(
                    '{"selected_experts":',
                    _format_2d_ints(routing.selected_experts),
                    "}\n",
                )
            )
            f.close()
        except:
            raise Error(
                '{"error_type":"OUTPUT_WRITE_FAILED","stage":"layer_forward","detail":"failed'
                " writing routing dump to "
                + dump_file
                + '"}'
            )

    # Identifikasi unique experts yang terpilih
    var unique_experts = List[Int]()
    for t in range(seq_len):
        ref row = routing.selected_experts[t]
        for k in range(len(row)):
            var e_id = row[k]
            var found = False
            for u in range(len(unique_experts)):
                if unique_experts[u] == e_id:
                    found = True
                    break
            if not found:
                unique_experts.append(e_id)
    moe_ns += perf_counter_ns() - t_r_comp0

    # 2d. Muat bobot routed experts yang terpilih
    var t_exp_pread0 = perf_counter_ns()
    var loaded_experts = List[SwigluWeights]()
    for u in range(len(unique_experts)):
        var e_id = unique_experts[u]
        var w = load_layer_routed_expert_weights(
            layer_idx,
            e_id,
            model_root,
            weight_map,
            cfg,
            cfg.moe_intermediate_size,
            cache,
            telemetry,
        )
        loaded_experts.append(w^)

    # 2e. Muat bobot shared expert & shared gate
    var shared_weights = load_layer_shared_expert_weights(
        layer_idx,
        model_root,
        weight_map,
        cfg,
        cfg.shared_expert_intermediate_size,
        cache,
        telemetry,
    )
    pread_ns += perf_counter_ns() - t_exp_pread0

    # 2f. Hitung SwiGLU routed experts per token
    var t_exp_comp0 = perf_counter_ns()
    var all_routed_outputs = List[List[Float32]]()
    for t in range(seq_len):
        ref row = routing.selected_experts[t]
        var tok_out = List[Float32]()
        tok_out.reserve(cfg.num_experts_per_tok * cfg.hidden_size)

        var x_t = List[Float32]()
        x_t.reserve(cfg.hidden_size)
        var offset = t * cfg.hidden_size
        for d in range(cfg.hidden_size):
            x_t.append(x_norm_moe[offset + d])

        for k in range(cfg.num_experts_per_tok):
            var e_id = row[k]
            var exp_idx = -1
            for u in range(len(unique_experts)):
                if unique_experts[u] == e_id:
                    exp_idx = u
                    break
            var e_out = swiglu_forward(
                x_t, loaded_experts[exp_idx], 1, layer_idx, e_id
            )
            for d in range(cfg.hidden_size):
                tok_out.append(e_out[d])
        all_routed_outputs.append(tok_out^)

    # 2g. Hitung shared expert SwiGLU + sigmoid gate (INVARIANT KERAS sigmoid)
    var shared_out = swiglu_forward(
        x_norm_moe, shared_weights.swiglu, seq_len, layer_idx, -1
    )
    var shared_gate_scores = shared_gate_forward(
        x_norm_moe,
        shared_weights.w_gate_sh,
        seq_len,
        cfg.hidden_size,
        layer_idx,
        "sigmoid",
    )

    # 2h. Agregasi gabungan routed + shared (tanpa residual)
    var moe_out = moe_combine_no_residual(
        all_routed_outputs,
        routing.router_probs,
        shared_out,
        shared_gate_scores,
        seq_len,
        cfg.hidden_size,
        layer_idx,
    )

    # RESIDUAL 2: hidden = hidden + moe_out
    hidden = add_residual(moe_out, hidden, layer_idx)
    moe_ns += perf_counter_ns() - t_exp_comp0

    var t_layer_end = perf_counter_ns()
    var total_ns = t_layer_end - t_layer_start

    return LayerTiming(
        layer_idx,
        Float64(pread_ns) / 1e9,
        Float64(attn_ns) / 1e9,
        Float64(moe_ns) / 1e9,
        Float64(total_ns) / 1e9,
    )
