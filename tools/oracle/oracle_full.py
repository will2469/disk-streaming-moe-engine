#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle Full Forward Reference (PyTorch FP32 CPU Deterministic).

Menjalankan full forward pass referensi 40-layer hybrid untuk Qwen3.6-35B-A3B:
  Token IDs -> Embedding Lookup -> 40 Layer Transformer Hybrid
  (30 GDN Linear Attention + 10 Gated Attention + 40 MoE Channel Mixers)
  -> Final RMSNorm -> LM Head -> Logits FP32 [s, 248320].

Bobot di-stream layer-per-layer dari safetensors shards dengan pelepasan buffer seketika
agar memori proses tetap terikat aman (<= 2-3 GiB).
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

# Determinism fail-closed CPU-only per kontrak M4
torch.manual_seed(42)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cpu")


def fail(
    error_type: str,
    detail: str,
    stage: str = "oracle",
    layer: int = -1,
    exit_code: int = 2,
) -> None:
    payload = {
        "status": "error",
        "error": {
            "code": error_type,
            "stage": stage,
            "message": detail,
            "details": {"layer": layer},
        },
    }
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.exit(exit_code)


def parse_model_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        fail("M4_ERR_CONFIG", f"config not found: {config_path}", stage="config")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        fail("M4_ERR_CONFIG", f"failed parsing config JSON: {e}", stage="config")

    cfg = raw.get("text_config", raw)
    cfg["hidden_size"] = int(cfg.get("hidden_size", 2048))
    cfg["num_hidden_layers"] = int(cfg.get("num_hidden_layers", 40))
    cfg["num_attention_heads"] = int(cfg.get("num_attention_heads", 16))
    cfg["num_key_value_heads"] = int(cfg.get("num_key_value_heads", 2))
    cfg["head_dim"] = int(
        cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    )
    cfg["vocab_size"] = int(cfg.get("vocab_size", 248320))
    cfg["rms_norm_eps"] = float(cfg.get("rms_norm_eps", 1e-6))
    cfg["rope_theta"] = float(
        cfg.get("rope_parameters", {}).get(
            "rope_theta", cfg.get("rope_theta", 10000000.0)
        )
    )
    cfg["partial_rotary_factor"] = float(
        cfg.get("rope_parameters", {}).get(
            "partial_rotary_factor", cfg.get("partial_rotary_factor", 0.25)
        )
    )
    cfg["num_experts"] = int(cfg.get("num_experts", 256))
    cfg["num_experts_per_tok"] = int(cfg.get("num_experts_per_tok", 8))
    cfg["moe_intermediate_size"] = int(cfg.get("moe_intermediate_size", 512))
    cfg["shared_expert_intermediate_size"] = int(
        cfg.get("shared_expert_intermediate_size", 512)
    )
    cfg["full_attention_interval"] = int(cfg.get("full_attention_interval", 4))
    return cfg


def load_tokens(tokens_path: str, vocab_size: int) -> list[int]:
    if not os.path.exists(tokens_path):
        fail(
            "M4_ERR_INPUT",
            f"tokens file not found: {tokens_path}",
            stage="input",
            exit_code=1,
        )
    try:
        with open(tokens_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        fail(
            "M4_ERR_INPUT",
            f"failed to parse tokens JSON: {e}",
            stage="input",
            exit_code=1,
        )

    token_list = []
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], list):
            token_list = raw[0]
        else:
            token_list = raw
    elif isinstance(raw, dict):
        if "tokens" in raw and isinstance(raw["tokens"], list):
            token_list = raw["tokens"]
        elif "prompts" in raw and isinstance(raw["prompts"], list):
            token_list = raw["prompts"][0]["tokens"]
        else:
            fail(
                "M4_ERR_INPUT",
                "tokens JSON missing 'tokens' or 'prompts'",
                stage="input",
                exit_code=1,
            )
    else:
        fail(
            "M4_ERR_INPUT",
            "tokens JSON must be an array or object",
            stage="input",
            exit_code=1,
        )

    if len(token_list) == 0:
        fail(
            "M4_ERR_INPUT",
            "tokens sequence is empty",
            stage="input",
            exit_code=1,
        )

    for idx, tid in enumerate(token_list):
        if not isinstance(tid, int):
            fail(
                "M4_ERR_INPUT",
                f"token at index {idx} not an integer",
                stage="input",
                exit_code=1,
            )
        if tid < 0 or tid >= vocab_size:
            fail(
                "M4_ERR_INPUT",
                f"token {tid} at index {idx} out of range [0, {vocab_size})",
                stage="input",
                exit_code=1,
            )
    return token_list


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def load_shard_tensor(
    model_dir: str,
    weight_map: dict[str, str],
    name: str,
    shards: dict,
    layer: int = -1,
) -> torch.Tensor:
    if name not in weight_map:
        fail("M4_ERR_SHARD_IO", f"missing tensor: {name}", layer=layer, exit_code=4)
    sf_name = weight_map[name]
    if sf_name not in shards:
        path = os.path.join(model_dir, sf_name)
        if not os.path.exists(path):
            fail(
                "M4_ERR_SHARD_IO",
                f"shard file not found: {path}",
                layer=layer,
                exit_code=4,
            )
        try:
            shards[sf_name] = safe_open(path, framework="pt", device="cpu")
        except Exception as e:
            fail(
                "M4_ERR_SHARD_IO",
                f"failed to open shard {sf_name}: {e}",
                layer=layer,
                exit_code=4,
            )
    try:
        return shards[sf_name].get_tensor(name).float()
    except Exception as e:
        fail(
            "M4_ERR_SHARD_IO",
            f"failed reading tensor {name}: {e}",
            layer=layer,
            exit_code=4,
        )


def forward_gdn_step(
    x_norm: torch.Tensor,
    weights: dict[str, torch.Tensor],
    s_state: torch.Tensor,
    seq_len: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eksekusi token mixer Gated DeltaNet (F14 recurrence baseline)."""
    # 1. QKV Proyeksi + Depthwise 1D Causal Convolution
    w_qkv = weights["linear_attn.in_proj_qkv.weight"]
    qkv = F.linear(x_norm, w_qkv)

    w_conv = weights["linear_attn.conv1d.weight"]
    qkv_t = qkv.t().unsqueeze(0)
    qkv_padded = F.pad(qkv_t, (3, 0))
    qkv_conv = F.conv1d(qkv_padded, w_conv, groups=8192)
    qkv_act = F.silu(qkv_conv.squeeze(0).t())

    q = qkv_act[:, :2048].view(seq_len, 16, 128).repeat_interleave(2, dim=1)
    k = qkv_act[:, 2048:4096].view(seq_len, 16, 128).repeat_interleave(2, dim=1)
    v = qkv_act[:, 4096:].view(seq_len, 32, 128)

    w_z = weights["linear_attn.in_proj_z.weight"]
    z_act = F.silu(F.linear(x_norm, w_z))

    w_a = weights["linear_attn.in_proj_a.weight"]
    w_b = weights["linear_attn.in_proj_b.weight"]
    a = F.linear(x_norm, w_a)
    b = F.linear(x_norm, w_b)
    dt_bias = weights["linear_attn.dt_bias"]
    a_log = weights["linear_attn.A_log"]

    decay = torch.exp(-torch.exp(a_log) * F.softplus(a + dt_bias))
    beta = torch.sigmoid(b)

    # 2. Recurrence DeltaNet scan per head
    outputs = []
    eye = torch.eye(128, device=x_norm.device, dtype=torch.float32)
    new_s = s_state.clone()

    for t in range(seq_len):
        out_heads = []
        for h in range(32):
            qt = q[t, h]
            kt = k[t, h]
            kt = kt / (torch.norm(kt) + eps)
            vt = v[t, h]
            g = decay[t, h]
            b_val = beta[t, h]

            trans = eye - b_val * torch.outer(kt, kt)
            bias_m = b_val * torch.outer(vt, kt)
            new_s[h] = g * torch.matmul(new_s[h], trans) + bias_m

            ot = torch.matmul(new_s[h], qt)
            out_heads.append(ot)
        outputs.append(torch.cat(out_heads, dim=-1))

    out_tensor = torch.stack(outputs, dim=0).view(seq_len, 32, 128)

    # 3. Head RMSNorm + Gate Z + Out Proj
    norm_w = weights["linear_attn.norm.weight"]
    var_h = torch.mean(out_tensor**2, dim=-1, keepdim=True)
    out_normed = (out_tensor * torch.rsqrt(var_h + eps) * norm_w).view(seq_len, 4096)
    out_gated = out_normed * z_act
    w_out = weights["linear_attn.out_proj.weight"]
    y = F.linear(out_gated, w_out)
    return y, new_s


def forward_attn_step(
    x_norm: torch.Tensor,
    weights: dict[str, torch.Tensor],
    seq_len: int,
    cfg: dict,
) -> torch.Tensor:
    """Eksekusi token mixer Gated Attention
    (GQA, QK-Norm, Partial RoPE, Sigmoid Gate)."""
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    base = cfg["rope_theta"]
    rotary_factor = cfg["partial_rotary_factor"]

    wq = weights["self_attn.q_proj.weight"]
    wk = weights["self_attn.k_proj.weight"]
    wv = weights["self_attn.v_proj.weight"]
    q_norm = weights["self_attn.q_norm.weight"]
    k_norm = weights["self_attn.k_norm.weight"]
    wo = weights["self_attn.o_proj.weight"]

    q_proj_out = F.linear(x_norm, wq)
    q, gate = torch.chunk(q_proj_out, 2, dim=-1)
    k = F.linear(x_norm, wk)
    v = F.linear(x_norm, wv)

    qh = q.view(seq_len, num_heads, head_dim)
    kh = k.view(seq_len, num_kv_heads, head_dim)
    vh = v.view(seq_len, num_kv_heads, head_dim)

    # QK-Norm
    qh = qh * torch.rsqrt(torch.mean(qh**2, dim=-1, keepdim=True) + eps) * q_norm
    kh = kh * torch.rsqrt(torch.mean(kh**2, dim=-1, keepdim=True) + eps) * k_norm

    # Partial RoPE 0.25
    rot_dim = int(head_dim * rotary_factor)
    qh_rot = qh[..., :rot_dim]
    qh_pass = qh[..., rot_dim:]
    kh_rot = kh[..., :rot_dim]
    kh_pass = kh[..., rot_dim:]

    inv_freq = 1.0 / (
        base ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim)
    )
    t_pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t_pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
    cos_emb, sin_emb = emb.cos(), emb.sin()

    qh_rot = (qh_rot * cos_emb) + (rotate_half(qh_rot) * sin_emb)
    kh_rot = (kh_rot * cos_emb) + (rotate_half(kh_rot) * sin_emb)

    qh = torch.cat((qh_rot, qh_pass), dim=-1).permute(1, 0, 2)
    kh = torch.cat((kh_rot, kh_pass), dim=-1).permute(1, 0, 2)
    vh = vh.permute(1, 0, 2)

    # GQA repeat
    group = num_heads // num_kv_heads
    if group > 1:
        kh = kh.repeat_interleave(group, dim=0)
        vh = vh.repeat_interleave(group, dim=0)

    # Causal MHA
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )
    scores = scores + mask
    attn_w = F.softmax(scores, dim=-1)
    attn_out = (
        torch.matmul(attn_w, vh)
        .permute(1, 0, 2)
        .contiguous()
        .view(seq_len, num_heads * head_dim)
    )
    attn_gated = attn_out * torch.sigmoid(gate)
    return F.linear(attn_gated, wo)


def forward_moe_step(
    x_norm_moe: torch.Tensor,
    model_dir: str,
    weight_map: dict[str, str],
    pfx: str,
    layer_idx: int,
    cfg: dict,
    dump_routing_dir: str | None = None,
) -> torch.Tensor:
    """Eksekusi channel mixer MoE (top-8 routed experts 3D slice + shared expert)."""
    seq_len, hidden_size = x_norm_moe.shape
    top_k = cfg["num_experts_per_tok"]
    inter_moe = cfg["moe_intermediate_size"]
    sh_shards: dict[str, safe_open] = {}

    # 1. Router logits & unrenormalized top-k
    w_router = load_shard_tensor(
        model_dir, weight_map, f"{pfx}mlp.gate.weight", sh_shards, layer_idx
    )
    router_logits = F.linear(x_norm_moe, w_router)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_probs, topk_indices = torch.topk(router_probs, k=top_k, dim=-1, sorted=True)

    selected_experts = []
    for t in range(seq_len):
        selected_experts.append([int(topk_indices[t, k].item()) for k in range(top_k)])

    if dump_routing_dir:
        rout_path = os.path.join(dump_routing_dir, f"routing_L{layer_idx}.json")
        with open(rout_path, "w", encoding="utf-8") as rf:
            json.dump({"selected_experts": selected_experts}, rf)
            rf.write("\n")

    unique_exp = sorted(list({e for row in selected_experts for e in row}))

    # 2. 3D Slice streaming untuk routed experts
    gu_name = f"{pfx}mlp.experts.gate_up_proj"
    d_name = f"{pfx}mlp.experts.down_proj"
    gu_shard = os.path.join(model_dir, weight_map[gu_name])
    d_shard = os.path.join(model_dir, weight_map[d_name])

    routed_gates = {}
    routed_ups = {}
    routed_downs = {}
    with (
        safe_open(gu_shard, framework="pt") as sf_gu,
        safe_open(d_shard, framework="pt") as sf_d,
    ):
        sl_gu = sf_gu.get_slice(gu_name)
        sl_d = sf_d.get_slice(d_name)
        for exp_id in unique_exp:
            w_gu = sl_gu[exp_id, :, :].float()
            routed_gates[exp_id] = w_gu[:inter_moe, :]
            routed_ups[exp_id] = w_gu[inter_moe:, :]
            routed_downs[exp_id] = sl_d[exp_id, :, :].float()

    y_routed = torch.zeros_like(x_norm_moe)
    for exp_id in unique_exp:
        wg = routed_gates[exp_id]
        wu = routed_ups[exp_id]
        wd = routed_downs[exp_id]
        for t in range(seq_len):
            for k in range(top_k):
                if selected_experts[t][k] == exp_id:
                    p = float(topk_probs[t, k].item())
                    xt = x_norm_moe[t : t + 1]
                    g = F.silu(F.linear(xt, wg))
                    u = F.linear(xt, wu)
                    e_out = F.linear(g * u, wd)[0]
                    y_routed[t] += p * e_out

    # 3. Shared expert SwiGLU + Sigmoid gate
    sh_pfx = f"{pfx}mlp.shared_expert."
    sh_wg = load_shard_tensor(
        model_dir, weight_map, f"{sh_pfx}gate_proj.weight", sh_shards, layer_idx
    )
    sh_wu = load_shard_tensor(
        model_dir, weight_map, f"{sh_pfx}up_proj.weight", sh_shards, layer_idx
    )
    sh_wd = load_shard_tensor(
        model_dir, weight_map, f"{sh_pfx}down_proj.weight", sh_shards, layer_idx
    )
    sh_gate = load_shard_tensor(
        model_dir,
        weight_map,
        f"{pfx}mlp.shared_expert_gate.weight",
        sh_shards,
        layer_idx,
    )

    g_sh = torch.sigmoid(F.linear(x_norm_moe, sh_gate.view(1, hidden_size)))
    e_sh = F.linear(
        F.silu(F.linear(x_norm_moe, sh_wg)) * F.linear(x_norm_moe, sh_wu), sh_wd
    )
    y_shared = g_sh * e_sh
    return y_routed + y_shared


def forward_single_hybrid_layer(
    x: torch.Tensor,
    layer_idx: int,
    model_dir: str,
    weight_map: dict[str, str],
    cfg: dict,
    s_gdn_layer: torch.Tensor,
    dump_routing_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eksekusi 1 layer transformer hybrid: Mixer -> Res 1 -> MoE -> Res 2."""
    pfx = f"model.language_model.layers.{layer_idx}."
    if f"{pfx}input_layernorm.weight" not in weight_map:
        pfx = f"model.layers.{layer_idx}."

    sh_shards: dict[str, safe_open] = {}
    seq_len, hidden_size = x.shape
    eps = cfg["rms_norm_eps"]

    # 1. Input RMSNorm
    w_norm1 = load_shard_tensor(
        model_dir, weight_map, f"{pfx}input_layernorm.weight", sh_shards, layer_idx
    )
    var1 = torch.mean(x**2, dim=-1, keepdim=True)
    x_norm1 = x * torch.rsqrt(var1 + eps) * w_norm1

    # 2. Token Mixer (GDN vs Gated Attention)
    is_linear = (layer_idx % cfg["full_attention_interval"]) != (
        cfg["full_attention_interval"] - 1
    )
    new_s = s_gdn_layer

    if is_linear:
        gdn_names = [
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.conv1d.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.dt_bias",
            "linear_attn.A_log",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
        ]
        gdn_w = {}
        for n in gdn_names:
            gdn_w[n] = load_shard_tensor(
                model_dir, weight_map, f"{pfx}{n}", sh_shards, layer_idx
            )
        mixer_out, new_s = forward_gdn_step(x_norm1, gdn_w, s_gdn_layer, seq_len, eps)
    else:
        attn_names = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.o_proj.weight",
        ]
        attn_w = {}
        for n in attn_names:
            attn_w[n] = load_shard_tensor(
                model_dir, weight_map, f"{pfx}{n}", sh_shards, layer_idx
            )
        mixer_out = forward_attn_step(x_norm1, attn_w, seq_len, cfg)

    # Residual 1
    x_mid = x + mixer_out

    # 3. Post-Attention RMSNorm
    w_norm2 = load_shard_tensor(
        model_dir,
        weight_map,
        f"{pfx}post_attention_layernorm.weight",
        sh_shards,
        layer_idx,
    )
    var2 = torch.mean(x_mid**2, dim=-1, keepdim=True)
    x_norm2 = x_mid * torch.rsqrt(var2 + eps) * w_norm2

    # 4. MoE Channel Mixer
    moe_out = forward_moe_step(
        x_norm2, model_dir, weight_map, pfx, layer_idx, cfg, dump_routing_dir
    )

    # Residual 2
    x_final = x_mid + moe_out
    if not torch.isfinite(x_final).all():
        fail(
            "M4_ERR_LAYER_FORWARD",
            f"non-finite values in layer {layer_idx}",
            stage="layer_forward",
            layer=layer_idx,
            exit_code=5,
        )
    return x_final, new_s


def write_logits_and_hash(
    logits: torch.Tensor,
    out_file: str,
    sha256_dest: str | None,
    seq_len: int,
    vocab_size: int,
) -> str:
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_bytes = logits.detach().cpu().to(torch.float32).numpy().tobytes()
    expected = seq_len * vocab_size * 4
    if len(out_bytes) != expected:
        fail(
            "M4_ERR_OUTPUT",
            f"output bytes {len(out_bytes)} != expected {expected}",
            stage="final_norm",
            exit_code=6,
        )

    tmp_out = f"{out_file}.tmp.{os.getpid()}"
    try:
        with open(tmp_out, "wb") as f:
            f.write(out_bytes)
        os.replace(tmp_out, out_file)
    except Exception as e:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass
        fail(
            "M4_ERR_OUTPUT",
            f"failed writing logits to {out_file}: {e}",
            stage="final_norm",
            exit_code=6,
        )

    sha256_hash = hashlib.sha256(out_bytes).hexdigest()
    sha_file = sha256_dest if sha256_dest else f"{out_file}.sha256"
    sha_dir = os.path.dirname(os.path.abspath(sha_file))
    if sha_dir:
        os.makedirs(sha_dir, exist_ok=True)

    base_name = os.path.basename(out_file)
    with open(sha_file, "w", encoding="utf-8") as f:
        f.write(f"{sha256_hash}  {base_name}\n")
    return sha256_hash


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Full Forward (PyTorch FP32 CPU Deterministic Reference)"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing model config, index, and safetensors shards",
    )
    parser.add_argument(
        "--tokens",
        required=True,
        help="Path to tokens JSON file",
    )
    parser.add_argument(
        "--output",
        default="logits_oracle.bin",
        help="Path to output FP32 logits binary",
    )
    parser.add_argument(
        "--dump-routing",
        default=None,
        help="Directory to dump per-layer routing JSON files",
    )
    parser.add_argument(
        "--sha256",
        default=None,
        help="Path to write SHA-256 checksum file",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, "model_config.json")
    if not os.path.exists(config_path):
        alt = os.path.join(args.model_dir, "config.json")
        if os.path.exists(alt):
            config_path = alt
    cfg = parse_model_config(config_path)

    vocab_size = cfg["vocab_size"]
    num_layers = cfg["num_hidden_layers"]
    eps = cfg["rms_norm_eps"]

    tokens = load_tokens(args.tokens, vocab_size)
    seq_len = len(tokens)

    index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        fail("M4_ERR_INDEX", f"index file not found: {index_path}", stage="index_load")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
    except Exception as e:
        fail("M4_ERR_INDEX", f"failed parsing index: {e}", stage="index_load")

    # Prefix deteksi
    pfx_embed = "model.language_model.embed_tokens.weight"
    if pfx_embed not in weight_map:
        pfx_embed = "model.embed_tokens.weight"

    # Embedding lookup
    init_shards: dict[str, safe_open] = {}
    w_embed = load_shard_tensor(
        args.model_dir, weight_map, pfx_embed, init_shards, layer=-1
    )
    token_t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
    x = torch.embedding(w_embed, token_t).clone()
    del w_embed
    init_shards.clear()

    if not torch.isfinite(x).all():
        fail(
            "M4_ERR_LAYER_FORWARD",
            "non-finite values in embedding output",
            stage="embedding",
            exit_code=5,
        )

    if args.dump_routing:
        os.makedirs(args.dump_routing, exist_ok=True)

    # Inisialisasi GDN states untuk 30 layer GDN
    gdn_states: dict[int, torch.Tensor] = {}
    for layer_i in range(num_layers):
        if (layer_i % cfg["full_attention_interval"]) != (
            cfg["full_attention_interval"] - 1
        ):
            gdn_states[layer_i] = torch.zeros((32, 128, 128), dtype=torch.float32)

    # 40-Layer Streaming Loop
    for layer_idx in range(num_layers):
        s_prev = gdn_states.get(layer_idx, torch.zeros(1))
        x, s_next = forward_single_hybrid_layer(
            x,
            layer_idx,
            args.model_dir,
            weight_map,
            cfg,
            s_prev,
            args.dump_routing,
        )
        if layer_idx in gdn_states:
            gdn_states[layer_idx] = s_next

    # Final RMSNorm
    norm_pfx = "model.language_model.norm.weight"
    if norm_pfx not in weight_map:
        norm_pfx = "model.norm.weight"

    final_shards: dict[str, safe_open] = {}
    w_final_norm = load_shard_tensor(
        args.model_dir, weight_map, norm_pfx, final_shards, layer=-1
    )
    var_final = torch.mean(x**2, dim=-1, keepdim=True)
    x_final_norm = x * torch.rsqrt(var_final + eps) * w_final_norm

    # LM Head Projection
    w_lm_head = load_shard_tensor(
        args.model_dir, weight_map, "lm_head.weight", final_shards, layer=-1
    )
    logits = F.linear(x_final_norm, w_lm_head)
    del w_lm_head, w_final_norm
    final_shards.clear()

    if not torch.isfinite(logits).all():
        fail(
            "M4_ERR_LAYER_FORWARD",
            "non-finite values in final logits",
            stage="final_norm",
            exit_code=5,
        )

    out_file = os.path.abspath(args.output)
    sha256_hash = write_logits_and_hash(
        logits, out_file, args.sha256, seq_len, vocab_size
    )

    summary = {
        "status": "success",
        "num_tokens": seq_len,
        "vocab_size": vocab_size,
        "output_file": out_file,
        "output_bytes": seq_len * vocab_size * 4,
        "sha256": sha256_hash,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
