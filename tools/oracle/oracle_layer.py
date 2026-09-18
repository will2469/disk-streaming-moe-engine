#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle Layer Path — PyTorch FP32 Reference.
Covers Part Attention (M2-W5) and Part MoE (M3-W4).

10-step reference attention pipeline:
  1. Load activation input: shape [16, 2048] (or [L, hidden_dim]), fp32 LE
  2. RMSNorm (F6): y = x / RMS(x) * gamma
  3. Load QKV weights & 72 biases (16 heads x 128)
  4. QKV projection: [Q, K, V] = [W_q, W_k, W_v] * x + [b_q, b_k, b_v]
  5. RoPE rotate_half (F7): positional rotation on Q and K
  6. Causal mask: triangular autoregressive mask
  7. MHA attention: softmax(QK^T / sqrt(d) + mask) * V (stable softmax)
  8. o_proj + bias: y = W_o * attn_out + b_o
  9. Residual: y_final = y + x
  10. Output: write binary fp32 LE to attn_ref.bin + compute SHA-256

11-step reference MoE pipeline (F8):
  1. Load activation input: shape [16, 2048], fp32 LE
  2. Router softmax fp32: p = softmax(W_r x) in R^60
  3. Top-4 selection TANPA renormalisasi: A = Top-4(p)
  4. Load routed experts weights (gate, up, down) untuk selected experts
  5. Load shared expert weights (gate, up, down) + shared expert gate
  6. Routed experts SwiGLU: E_i(x) = W_down(SiLU(W_gate x) * W_up x)
  7. Weighted sum routed: y_routed = sum_{i in A} p_i E_i(x)
  8. Shared expert sigmoid gate: g_sh = W_{gate_sh} x, sigma(g_sh) (sigmoid)
  9. Shared expert computation: y_shared = sigma(g_sh) E_sh(x)
  10. Residual: y = y_routed + y_shared + x
  11. Output: write binary fp32 LE to moe_ref.bin + routing_info

CLI:
  oracle_layer.py --part attn|moe --layer 0|12|23 --activation <path>
                  --model-dir <dir> [--output <path>] [--routing-output <path>]
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch
from safetensors import safe_open


def fail(error_type: str, detail: str, stage: str = "oracle", layer: int = -1) -> None:
    payload = {
        "error_type": error_type,
        "detail": detail,
        "stage": stage,
        "layer": layer,
    }
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.exit(2)


def parse_model_config(config_path: str, layer: int) -> dict:
    if not os.path.exists(config_path):
        fail(
            "FILE_NOT_FOUND",
            f"config file not found: {config_path}",
            stage="config",
            layer=layer,
        )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        fail(
            "CONFIG_ERROR",
            f"failed to parse config JSON: {e}",
            stage="config",
            layer=layer,
        )

    # Unwrap text_config if present (Qwen3.6-35B-A3B)
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        text_cfg = cfg["text_config"]
        for k, v in text_cfg.items():
            if k not in cfg or k in [
                "hidden_size",
                "num_attention_heads",
                "num_hidden_layers",
                "rms_norm_eps",
                "rope_parameters",
                "head_dim",
                "num_key_value_heads",
                "attn_output_gate",
                "attention_bias",
            ]:
                cfg[k] = v

    for req in ["hidden_size", "num_attention_heads"]:
        if req not in cfg:
            fail(
                "CONFIG_ERROR",
                f"missing required field: {req}",
                stage="config",
                layer=layer,
            )
        if not isinstance(cfg[req], int) or cfg[req] <= 0:
            fail(
                "CONFIG_ERROR",
                f"field {req} must be a positive integer",
                stage="config",
                layer=layer,
            )

    is_qwen36 = (
        cfg.get("num_hidden_layers", 24) == 40
        or cfg.get("model_type") in ["qwen3_5_moe", "qwen3_5_moe_text"]
        or "full_attention_interval" in cfg
    )
    cfg["is_qwen36"] = is_qwen36

    num_hidden_layers = cfg.get("num_hidden_layers", 40 if is_qwen36 else 24)
    cfg["num_hidden_layers"] = num_hidden_layers

    eps = float(cfg.get("rms_norm_eps", 1e-6))
    if eps <= 0:
        fail(
            "CONFIG_ERROR",
            "rms_norm_eps must be positive",
            stage="config",
            layer=layer,
        )
    cfg["rms_norm_eps"] = eps

    cfg["head_dim"] = int(
        cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    )
    cfg["num_key_value_heads"] = int(
        cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    )
    cfg["attn_output_gate"] = bool(cfg.get("attn_output_gate", is_qwen36))
    cfg["attention_bias"] = bool(cfg.get("attention_bias", False))

    default_theta = 10000000.0 if is_qwen36 else 1000000.0
    rope_theta = float(
        cfg.get("rope_parameters", {}).get(
            "rope_theta", cfg.get("rope_theta", default_theta)
        )
    )
    cfg["rope_theta"] = rope_theta
    default_rotary_factor = 0.25 if is_qwen36 else 1.0
    cfg["partial_rotary_factor"] = float(
        cfg.get("rope_parameters", {}).get(
            "partial_rotary_factor",
            cfg.get("partial_rotary_factor", default_rotary_factor),
        )
    )

    # MoE architecture fields (Qwen1.5-MoE defaults if absent)
    cfg["num_experts"] = int(cfg.get("num_experts", 256 if is_qwen36 else 60))
    cfg["num_experts_per_tok"] = int(
        cfg.get("num_experts_per_tok", 8 if is_qwen36 else 4)
    )
    cfg["moe_intermediate_size"] = int(
        cfg.get("moe_intermediate_size", 512 if is_qwen36 else 1408)
    )
    cfg["shared_expert_intermediate_size"] = int(
        cfg.get("shared_expert_intermediate_size", 512 if is_qwen36 else 5632)
    )
    cfg["norm_topk_prob"] = bool(cfg.get("norm_topk_prob", False))

    return cfg


def load_activation(path: str, hidden_dim: int, layer: int) -> tuple[torch.Tensor, int]:
    if not os.path.exists(path):
        fail(
            "FILE_NOT_FOUND",
            f"activation file not found: {path}",
            stage="activation",
            layer=layer,
        )
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        fail(
            "ACT_LOAD_FAILED",
            f"cannot read activation file: {e}",
            stage="activation",
            layer=layer,
        )

    if len(raw_bytes) == 0 or len(raw_bytes) % (hidden_dim * 4) != 0:
        fail(
            "ACT_LOAD_FAILED",
            f"activation byte size {len(raw_bytes)} not divisible "
            f"by row size {hidden_dim * 4}",
            stage="activation",
            layer=layer,
        )

    num_tokens = len(raw_bytes) // (hidden_dim * 4)
    floats = (
        torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32)
        .reshape(num_tokens, hidden_dim)
        .clone()
    )

    if not torch.isfinite(floats).all():
        fail(
            "ACT_LOAD_FAILED",
            "activation contains non-finite values (NaN or Inf)",
            stage="activation",
            layer=layer,
        )

    if torch.max(torch.abs(floats)).item() > 1e6:
        fail(
            "ACT_LOAD_FAILED",
            "activation contains values out of reasonable range (>1e6)",
            stage="activation",
            layer=layer,
        )

    return floats, num_tokens


def _load_tensor_from_shard(
    model_dir: str,
    weight_map: dict[str, str],
    name: str,
    shards_cache: dict,
    layer: int,
    stage: str = "attention",
) -> torch.Tensor:
    if name not in weight_map:
        fail(
            "WEIGHT_LOAD_FAILED",
            f"missing required tensor in index: {name}",
            stage=stage,
            layer=layer,
        )
    shard_file = weight_map[name]
    shard_path = os.path.join(model_dir, shard_file)
    if not os.path.exists(shard_path):
        fail(
            "FILE_NOT_FOUND",
            f"shard file not found on disk: {shard_path}",
            stage=stage,
            layer=layer,
        )
    if shard_file not in shards_cache:
        try:
            shards_cache[shard_file] = safe_open(
                shard_path, framework="pt", device="cpu"
            )
        except Exception as e:
            fail(
                "WEIGHT_LOAD_FAILED",
                f"failed to open shard {shard_file}: {e}",
                stage=stage,
                layer=layer,
            )
    shard = shards_cache[shard_file]
    try:
        t = shard.get_tensor(name)
    except Exception as e:
        fail(
            "WEIGHT_LOAD_FAILED",
            f"tensor {name} missing from shard {shard_file}: {e}",
            stage=stage,
            layer=layer,
        )
    return t.float()


def _validate_weight_shapes(
    weights: dict[str, torch.Tensor],
    pfx: str,
    hidden_size: int,
    layer: int,
    cfg: dict,
) -> None:
    if cfg.get("is_qwen36", False):
        head_dim = cfg.get("head_dim", 256)
        num_heads = cfg.get("num_attention_heads", 16)
        num_kv_heads = cfg.get("num_key_value_heads", 2)
        q_out_dim = (
            num_heads * head_dim * 2
            if cfg.get("attn_output_gate", True)
            else num_heads * head_dim
        )
        kv_out_dim = num_kv_heads * head_dim
        expected_shapes = [
            (f"{pfx}input_layernorm.weight", torch.Size([hidden_size]), "norm"),
            (
                f"{pfx}self_attn.q_proj.weight",
                torch.Size([q_out_dim, hidden_size]),
                "q_proj weight",
            ),
            (
                f"{pfx}self_attn.k_proj.weight",
                torch.Size([kv_out_dim, hidden_size]),
                "k_proj weight",
            ),
            (
                f"{pfx}self_attn.v_proj.weight",
                torch.Size([kv_out_dim, hidden_size]),
                "v_proj weight",
            ),
            (f"{pfx}self_attn.q_norm.weight", torch.Size([head_dim]), "q_norm weight"),
            (f"{pfx}self_attn.k_norm.weight", torch.Size([head_dim]), "k_norm weight"),
            (
                f"{pfx}self_attn.o_proj.weight",
                torch.Size([hidden_size, num_heads * head_dim]),
                "o_proj weight",
            ),
        ]
    else:
        expected_shapes = [
            (f"{pfx}input_layernorm.weight", torch.Size([hidden_size]), "norm"),
            (
                f"{pfx}self_attn.q_proj.weight",
                torch.Size([hidden_size, hidden_size]),
                "q_proj weight",
            ),
            (f"{pfx}self_attn.q_proj.bias", torch.Size([hidden_size]), "q_proj bias"),
            (
                f"{pfx}self_attn.k_proj.weight",
                torch.Size([hidden_size, hidden_size]),
                "k_proj weight",
            ),
            (f"{pfx}self_attn.k_proj.bias", torch.Size([hidden_size]), "k_proj bias"),
            (
                f"{pfx}self_attn.v_proj.weight",
                torch.Size([hidden_size, hidden_size]),
                "v_proj weight",
            ),
            (f"{pfx}self_attn.v_proj.bias", torch.Size([hidden_size]), "v_proj bias"),
            (
                f"{pfx}self_attn.o_proj.weight",
                torch.Size([hidden_size, hidden_size]),
                "o_proj weight",
            ),
        ]
    for tensor_name, expected, desc in expected_shapes:
        actual = weights[tensor_name].shape
        if actual != expected:
            fail(
                "WEIGHT_LOAD_FAILED",
                f"{desc} shape mismatch: {actual} vs expected {expected}",
                stage="attention",
                layer=layer,
            )


def load_layer_weights(
    model_dir: str, layer: int, cfg: dict
) -> dict[str, torch.Tensor]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        fail(
            "FILE_NOT_FOUND",
            f"index file not found: {index_path}",
            stage="attention",
            layer=layer,
        )
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    except Exception as e:
        fail(
            "WEIGHT_LOAD_FAILED",
            f"failed to parse index: {e}",
            stage="attention",
            layer=layer,
        )

    # Prefix resolution (Qwen3.6-35B-A3B vs legacy)
    pfx = f"model.language_model.layers.{layer}."
    if f"{pfx}input_layernorm.weight" not in weight_map:
        pfx = f"model.layers.{layer}."

    is_qwen36 = (
        cfg.get("is_qwen36", False) or f"{pfx}self_attn.q_norm.weight" in weight_map
    )
    cfg["is_qwen36"] = is_qwen36
    cfg["pfx"] = pfx

    if is_qwen36:
        req_tensors = [
            f"{pfx}input_layernorm.weight",
            f"{pfx}self_attn.q_proj.weight",
            f"{pfx}self_attn.k_proj.weight",
            f"{pfx}self_attn.v_proj.weight",
            f"{pfx}self_attn.q_norm.weight",
            f"{pfx}self_attn.k_norm.weight",
            f"{pfx}self_attn.o_proj.weight",
        ]
    else:
        req_tensors = [
            f"{pfx}input_layernorm.weight",
            f"{pfx}self_attn.q_proj.weight",
            f"{pfx}self_attn.q_proj.bias",
            f"{pfx}self_attn.k_proj.weight",
            f"{pfx}self_attn.k_proj.bias",
            f"{pfx}self_attn.v_proj.weight",
            f"{pfx}self_attn.v_proj.bias",
            f"{pfx}self_attn.o_proj.weight",
        ]

    for req in req_tensors:
        if req not in weight_map:
            fail(
                "WEIGHT_LOAD_FAILED",
                f"missing required tensor in index: {req}",
                stage="attention",
                layer=layer,
            )

    shards_cache = {}
    weights = {}
    for req in req_tensors:
        weights[req] = _load_tensor_from_shard(
            model_dir, weight_map, req, shards_cache, layer, "attention"
        )

    o_bias_name = f"{pfx}self_attn.o_proj.bias"
    if o_bias_name in weight_map:
        weights[o_bias_name] = _load_tensor_from_shard(
            model_dir, weight_map, o_bias_name, shards_cache, layer, "attention"
        )

    _validate_weight_shapes(weights, pfx, cfg["hidden_size"], layer, cfg)
    return weights


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Formula F7 rotate_half: [-x_{half..}, x_{..half}]."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def forward_layer_attention_oracle(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    cfg: dict,
) -> torch.Tensor:
    """Full reference attention block execution."""
    pfx = cfg.get("pfx", f"model.layers.{layer}.")
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim", hidden_size // num_heads)
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    seq_len = x.shape[0]
    eps = cfg["rms_norm_eps"]
    base = cfg["rope_theta"]
    is_qwen36 = cfg.get("is_qwen36", False)

    # 1. RMSNorm F6
    var = torch.mean(x**2, dim=-1, keepdim=True)
    norm_w = weights[f"{pfx}input_layernorm.weight"]
    x_norm = x * torch.rsqrt(var + eps) * norm_w

    if is_qwen36:
        wq = weights[f"{pfx}self_attn.q_proj.weight"]
        wk = weights[f"{pfx}self_attn.k_proj.weight"]
        wv = weights[f"{pfx}self_attn.v_proj.weight"]
        q_norm = weights[f"{pfx}self_attn.q_norm.weight"]
        k_norm = weights[f"{pfx}self_attn.k_norm.weight"]
        wo = weights[f"{pfx}self_attn.o_proj.weight"]

        q_proj_out = torch.matmul(x_norm, wq.t())
        if cfg.get("attn_output_gate", True):
            q, gate = torch.chunk(q_proj_out, 2, dim=-1)
        else:
            q = q_proj_out
            gate = None

        k = torch.matmul(x_norm, wk.t())
        v = torch.matmul(x_norm, wv.t())

        qh = q.view(seq_len, num_heads, head_dim)
        kh = k.view(seq_len, num_kv_heads, head_dim)
        vh = v.view(seq_len, num_kv_heads, head_dim)

        # QK-Norm
        qh = qh * torch.rsqrt(torch.mean(qh**2, dim=-1, keepdim=True) + eps) * q_norm
        kh = kh * torch.rsqrt(torch.mean(kh**2, dim=-1, keepdim=True) + eps) * k_norm

        # Partial RoPE (factor 0.25)
        rotary_factor = cfg.get("partial_rotary_factor", 0.25)
        rotary_dim = int(head_dim * rotary_factor)
        qh_rot = qh[..., :rotary_dim]
        qh_pass = qh[..., rotary_dim:]
        kh_rot = kh[..., :rotary_dim]
        kh_pass = kh[..., rotary_dim:]

        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        t_pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t_pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()

        qh_rot = (qh_rot * cos_emb) + (rotate_half(qh_rot) * sin_emb)
        kh_rot = (kh_rot * cos_emb) + (rotate_half(kh_rot) * sin_emb)
        qh = torch.cat((qh_rot, qh_pass), dim=-1)
        kh = torch.cat((kh_rot, kh_pass), dim=-1)

        # GQA
        qh = qh.permute(1, 0, 2)
        kh = kh.permute(1, 0, 2)
        vh = vh.permute(1, 0, 2)
        group_size = num_heads // num_kv_heads
        if group_size > 1:
            kh = kh.repeat_interleave(group_size, dim=0)
            vh = vh.repeat_interleave(group_size, dim=0)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32),
            diagonal=1,
        )
        scores = scores + mask
        max_s = torch.max(scores, dim=-1, keepdim=True)[0]
        exp_s = torch.exp(scores - max_s)
        attn_weights = exp_s / torch.sum(exp_s, dim=-1, keepdim=True)

        attn_out = torch.matmul(attn_weights, vh)
        attn_out = attn_out.permute(1, 0, 2).contiguous()
        attn_out = attn_out.view(seq_len, num_heads * head_dim)

        if gate is not None:
            attn_out = attn_out * torch.sigmoid(gate)

        y = torch.matmul(attn_out, wo.t())
    else:
        # Legacy trial path
        wq = weights[f"{pfx}self_attn.q_proj.weight"]
        bq = weights[f"{pfx}self_attn.q_proj.bias"]
        wk = weights[f"{pfx}self_attn.k_proj.weight"]
        bk = weights[f"{pfx}self_attn.k_proj.bias"]
        wv = weights[f"{pfx}self_attn.v_proj.weight"]
        bv = weights[f"{pfx}self_attn.v_proj.bias"]

        q = torch.matmul(x_norm, wq.t()) + bq
        k = torch.matmul(x_norm, wk.t()) + bk
        v = torch.matmul(x_norm, wv.t()) + bv

        qh = q.view(seq_len, num_heads, head_dim)
        kh = k.view(seq_len, num_heads, head_dim)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t_pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t_pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()

        q_rot = (qh * cos_emb) + (rotate_half(qh) * sin_emb)
        k_rot = (kh * cos_emb) + (rotate_half(kh) * sin_emb)

        qh = q_rot.permute(1, 0, 2)
        kh = k_rot.permute(1, 0, 2)
        vh = v.view(seq_len, num_heads, head_dim).permute(1, 0, 2)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32),
            diagonal=1,
        )
        scores = scores + mask
        max_s = torch.max(scores, dim=-1, keepdim=True)[0]
        exp_s = torch.exp(scores - max_s)
        attn_weights = exp_s / torch.sum(exp_s, dim=-1, keepdim=True)

        attn_out = torch.matmul(attn_weights, vh)
        attn_out = attn_out.permute(1, 0, 2).contiguous().view(seq_len, hidden_size)

        wo = weights[f"{pfx}self_attn.o_proj.weight"]
        y = torch.matmul(attn_out, wo.t())
        o_bias_name = f"{pfx}self_attn.o_proj.bias"
        if o_bias_name in weights:
            y = y + weights[o_bias_name]

    y_final = y + x
    if not torch.isfinite(y_final).all():
        fail(
            "ATTENTION_ERROR",
            "non-finite value in attention oracle output",
            stage="attention",
            layer=layer,
        )
    return y_final


def compute_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Hitung SwiGLU(x) = W_down (SiLU(W_gate x) * W_up x)."""
    g = torch.nn.functional.linear(x, w_gate)
    u = torch.nn.functional.linear(x, w_up)
    h = torch.nn.functional.silu(g) * u
    y = torch.nn.functional.linear(h, w_down)
    return y


def _load_fused_3d_expert_weights(
    model_dir: str,
    weight_map: dict[str, str],
    pfx: str,
    unique_experts: list[int],
    inter_routed: int,
    hidden_dim: int,
    layer: int,
) -> tuple[dict, dict, dict]:
    fused_gu_name = f"{pfx}mlp.experts.gate_up_proj"
    fused_d_name = f"{pfx}mlp.experts.down_proj"
    gu_shard = weight_map[fused_gu_name]
    d_shard = weight_map[fused_d_name]
    gu_path = os.path.join(model_dir, gu_shard)
    d_path = os.path.join(model_dir, d_shard)
    if not os.path.exists(gu_path):
        fail(
            "FILE_NOT_FOUND",
            f"shard file not found: {gu_path}",
            stage="experts",
            layer=layer,
        )
    if not os.path.exists(d_path):
        fail(
            "FILE_NOT_FOUND",
            f"shard file not found: {d_path}",
            stage="experts",
            layer=layer,
        )
    routed_gates = {}
    routed_ups = {}
    routed_downs = {}
    with (
        safe_open(gu_path, framework="pt") as sf_gu,
        safe_open(d_path, framework="pt") as sf_d,
    ):
        slice_gu = sf_gu.get_slice(fused_gu_name)
        slice_d = sf_d.get_slice(fused_d_name)
        for exp_id in unique_experts:
            w_gu = slice_gu[exp_id, :, :].float()
            if w_gu.shape != torch.Size([2 * inter_routed, hidden_dim]):
                fail(
                    "WEIGHT_LOAD_FAILED",
                    f"expert {exp_id} gate_up slice shape mismatch: {w_gu.shape}",
                    stage="experts",
                    layer=layer,
                )
            routed_gates[exp_id] = w_gu[:inter_routed, :]
            routed_ups[exp_id] = w_gu[inter_routed:, :]
            w_d = slice_d[exp_id, :, :].float()
            if w_d.shape != torch.Size([hidden_dim, inter_routed]):
                fail(
                    "WEIGHT_LOAD_FAILED",
                    f"expert {exp_id} down slice shape mismatch: {w_d.shape}",
                    stage="experts",
                    layer=layer,
                )
            routed_downs[exp_id] = w_d
    return routed_gates, routed_ups, routed_downs


def _load_legacy_2d_expert_weights(
    model_dir: str,
    weight_map: dict[str, str],
    pfx: str,
    unique_experts: list[int],
    inter_routed: int,
    hidden_dim: int,
    layer: int,
    shards_cache: dict,
) -> tuple[dict, dict, dict]:
    routed_gates = {}
    routed_ups = {}
    routed_downs = {}
    for exp_id in unique_experts:
        pfx_exp = f"{pfx}mlp.experts.{exp_id}."
        w_g = _load_tensor_from_shard(
            model_dir,
            weight_map,
            pfx_exp + "gate_proj.weight",
            shards_cache,
            layer,
            "experts",
        )
        w_u = _load_tensor_from_shard(
            model_dir,
            weight_map,
            pfx_exp + "up_proj.weight",
            shards_cache,
            layer,
            "experts",
        )
        w_d = _load_tensor_from_shard(
            model_dir,
            weight_map,
            pfx_exp + "down_proj.weight",
            shards_cache,
            layer,
            "experts",
        )
        if w_g.shape != torch.Size([inter_routed, hidden_dim]):
            fail(
                "WEIGHT_LOAD_FAILED",
                f"expert {exp_id} gate_proj shape mismatch: {w_g.shape}",
                stage="experts",
                layer=layer,
            )
        if w_u.shape != torch.Size([inter_routed, hidden_dim]):
            fail(
                "WEIGHT_LOAD_FAILED",
                f"expert {exp_id} up_proj shape mismatch: {w_u.shape}",
                stage="experts",
                layer=layer,
            )
        if w_d.shape != torch.Size([hidden_dim, inter_routed]):
            fail(
                "WEIGHT_LOAD_FAILED",
                f"expert {exp_id} down_proj shape mismatch: {w_d.shape}",
                stage="experts",
                layer=layer,
            )
        routed_gates[exp_id] = w_g
        routed_ups[exp_id] = w_u
        routed_downs[exp_id] = w_d
    return routed_gates, routed_ups, routed_downs


def _load_routed_expert_weights(
    model_dir: str,
    weight_map: dict[str, str],
    pfx: str,
    unique_experts: list[int],
    inter_routed: int,
    hidden_dim: int,
    layer: int,
    shards_cache: dict,
) -> tuple[dict, dict, dict]:
    fused_gu_name = f"{pfx}mlp.experts.gate_up_proj"
    fused_d_name = f"{pfx}mlp.experts.down_proj"
    if fused_gu_name in weight_map and fused_d_name in weight_map:
        return _load_fused_3d_expert_weights(
            model_dir,
            weight_map,
            pfx,
            unique_experts,
            inter_routed,
            hidden_dim,
            layer,
        )
    return _load_legacy_2d_expert_weights(
        model_dir,
        weight_map,
        pfx,
        unique_experts,
        inter_routed,
        hidden_dim,
        layer,
        shards_cache,
    )


def _load_shared_expert_weights(
    model_dir: str,
    weight_map: dict[str, str],
    pfx: str,
    inter_shared: int,
    hidden_dim: int,
    layer: int,
    shards_cache: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pfx_sh = f"{pfx}mlp.shared_expert."
    w_sh_gate_proj = _load_tensor_from_shard(
        model_dir,
        weight_map,
        pfx_sh + "gate_proj.weight",
        shards_cache,
        layer,
        "shared",
    )
    w_sh_up_proj = _load_tensor_from_shard(
        model_dir,
        weight_map,
        pfx_sh + "up_proj.weight",
        shards_cache,
        layer,
        "shared",
    )
    w_sh_down_proj = _load_tensor_from_shard(
        model_dir,
        weight_map,
        pfx_sh + "down_proj.weight",
        shards_cache,
        layer,
        "shared",
    )
    w_sh_gate = _load_tensor_from_shard(
        model_dir,
        weight_map,
        f"{pfx}mlp.shared_expert_gate.weight",
        shards_cache,
        layer,
        "shared",
    )

    if w_sh_gate_proj.shape != torch.Size([inter_shared, hidden_dim]):
        fail(
            "WEIGHT_LOAD_FAILED",
            f"shared gate_proj shape mismatch: {w_sh_gate_proj.shape}",
            stage="shared",
            layer=layer,
        )
    if w_sh_up_proj.shape != torch.Size([inter_shared, hidden_dim]):
        fail(
            "WEIGHT_LOAD_FAILED",
            f"shared up_proj shape mismatch: {w_sh_up_proj.shape}",
            stage="shared",
            layer=layer,
        )
    if w_sh_down_proj.shape != torch.Size([hidden_dim, inter_shared]):
        fail(
            "WEIGHT_LOAD_FAILED",
            f"shared down_proj shape mismatch: {w_sh_down_proj.shape}",
            stage="shared",
            layer=layer,
        )
    if w_sh_gate.numel() != hidden_dim:
        fail(
            "WEIGHT_LOAD_FAILED",
            f"shared_expert_gate size {w_sh_gate.numel()} != {hidden_dim}",
            stage="shared",
            layer=layer,
        )
    return (
        w_sh_gate_proj,
        w_sh_up_proj,
        w_sh_down_proj,
        w_sh_gate.reshape(1, hidden_dim),
    )


def forward_layer_moe_oracle(
    act: torch.Tensor,
    model_dir: str,
    weight_map: dict[str, str],
    layer: int,
    cfg: dict,
) -> tuple[torch.Tensor, dict]:
    """11-step reference MoE pipeline (F8)."""
    seq_len, hidden_dim = act.shape
    num_experts = cfg["num_experts"]
    top_k = cfg["num_experts_per_tok"]
    inter_routed = cfg["moe_intermediate_size"]
    inter_shared = cfg["shared_expert_intermediate_size"]
    shards_cache = {}

    pfx_lm = f"model.language_model.layers.{layer}."
    pfx_legacy = f"model.layers.{layer}."
    pfx = pfx_lm if f"{pfx_lm}mlp.gate.weight" in weight_map else pfx_legacy

    # 2. Router softmax fp32
    router_name = f"{pfx}mlp.gate.weight"
    w_router = _load_tensor_from_shard(
        model_dir, weight_map, router_name, shards_cache, layer, "router"
    )
    if w_router.shape != torch.Size([num_experts, hidden_dim]):
        fail(
            "WEIGHT_LOAD_FAILED",
            f"router shape {w_router.shape} != [{num_experts}, {hidden_dim}]",
            stage="router",
            layer=layer,
        )

    logits = torch.nn.functional.linear(act, w_router)
    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)

    # 3. Top-k selection TANPA renormalisasi (sorted descending, tie-break by index)
    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1, sorted=True)

    selected_experts = []
    router_probs = []
    for t in range(seq_len):
        row_exp = [int(topk_indices[t, k].item()) for k in range(top_k)]
        row_prob = [float(topk_probs[t, k].item()) for k in range(top_k)]
        selected_experts.append(row_exp)
        router_probs.append(row_prob)

    unique_experts = sorted(list({e for row in selected_experts for e in row}))

    # 4. Load routed experts weights (3D slice streaming / 2D legacy)
    routed_gates, routed_ups, routed_downs = _load_routed_expert_weights(
        model_dir,
        weight_map,
        pfx,
        unique_experts,
        inter_routed,
        hidden_dim,
        layer,
        shards_cache,
    )

    # 5. Load shared expert weights
    w_sh_gate_proj, w_sh_up_proj, w_sh_down_proj, w_sh_gate = (
        _load_shared_expert_weights(
            model_dir,
            weight_map,
            pfx,
            inter_shared,
            hidden_dim,
            layer,
            shards_cache,
        )
    )

    # 6 & 7. Routed experts SwiGLU + weighted sum
    y_routed = torch.zeros_like(act)
    for t in range(seq_len):
        x_t = act[t : t + 1]
        accum = torch.zeros(1, hidden_dim, dtype=torch.float32)
        for k in range(top_k):
            exp_id = selected_experts[t][k]
            prob = router_probs[t][k]
            e_out = compute_swiglu(
                x_t,
                routed_gates[exp_id],
                routed_ups[exp_id],
                routed_downs[exp_id],
            )
            accum += prob * e_out
        y_routed[t] = accum[0]

    # 8. Shared expert sigmoid gate: INVARIANT KERAS sigmoid, NOT softmax
    shared_logits = torch.nn.functional.linear(act, w_sh_gate)
    g_sh = torch.sigmoid(shared_logits)

    # 9. Shared expert computation
    e_sh = compute_swiglu(act, w_sh_gate_proj, w_sh_up_proj, w_sh_down_proj)
    y_shared = g_sh * e_sh

    # 10. Residual connection: y = y_routed + y_shared + act
    y_final = y_routed + y_shared + act

    if not torch.isfinite(y_final).all():
        fail(
            "EXPERT_ERROR",
            "non-finite value in MoE oracle output",
            stage="moe",
            layer=layer,
        )

    routing_info = {
        "selected_experts": selected_experts,
        "router_probs": router_probs,
    }
    return y_final, routing_info


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Layer (PyTorch fp32) — Attention or MoE"
    )
    parser.add_argument(
        "--part",
        required=True,
        choices=["attn", "moe"],
        help="Part name (attn|moe)",
    )
    parser.add_argument(
        "--layer",
        required=True,
        type=int,
        help="Layer index (0, 3, 12, 23, or 39)",
    )
    parser.add_argument(
        "--activation", required=True, help="Path to input activation.bin"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing model weights and index",
    )
    parser.add_argument("--output", default=None, help="Path to write output binary")
    parser.add_argument(
        "--routing-output", default=None, help="Path to write routing info JSON"
    )
    args = parser.parse_args()

    if args.part not in ["attn", "moe"]:
        fail(
            "PART_INVALID",
            f"unsupported part: {args.part}",
            stage="cli",
            layer=args.layer,
        )

    if args.layer not in [0, 3, 12, 23, 39]:
        fail(
            "LAYER_INVALID",
            f"layer must be 0, 3, 12, 23, or 39, got {args.layer}",
            stage="cli",
            layer=args.layer,
        )

    config_path = os.path.join(args.model_dir, "model_config.json")
    if not os.path.exists(config_path):
        alt = os.path.join(args.model_dir, "config.json")
        if os.path.exists(alt):
            config_path = alt

    cfg = parse_model_config(config_path, args.layer)

    if args.layer >= cfg["num_hidden_layers"]:
        fail(
            "LAYER_INVALID",
            f"layer {args.layer} exceeds num_hidden_layers {cfg['num_hidden_layers']}",
            stage="cli",
            layer=args.layer,
        )

    act, num_tokens = load_activation(args.activation, cfg["hidden_size"], args.layer)

    out_file = args.output
    if not out_file:
        out_file = "attn_ref.bin" if args.part == "attn" else "moe_ref.bin"

    routing_info = None
    if args.part == "attn":
        weights = load_layer_weights(args.model_dir, args.layer, cfg)
        ref_out = forward_layer_attention_oracle(act, weights, args.layer, cfg)
    else:
        index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            fail(
                "FILE_NOT_FOUND",
                f"index file not found: {index_path}",
                stage="moe",
                layer=args.layer,
            )
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        ref_out, routing_info = forward_layer_moe_oracle(
            act, args.model_dir, weight_map, args.layer, cfg
        )

    out_dir = os.path.dirname(os.path.abspath(out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_bytes = ref_out.detach().cpu().to(torch.float32).numpy().tobytes()
    with open(out_file, "wb") as f:
        f.write(out_bytes)

    sha256_hash = hashlib.sha256(out_bytes).hexdigest()

    summary = {
        "status": "success",
        "part": args.part,
        "layer": args.layer,
        "num_tokens": num_tokens,
        "hidden_dim": cfg["hidden_size"],
        "output_file": out_file,
        "output_bytes": len(out_bytes),
        "sha256": sha256_hash,
    }
    if routing_info is not None:
        summary["routing_info"] = routing_info
        if args.routing_output:
            rout_dir = os.path.dirname(os.path.abspath(args.routing_output))
            if rout_dir:
                os.makedirs(rout_dir, exist_ok=True)
            with open(args.routing_output, "w", encoding="utf-8") as rf:
                json.dump(routing_info, rf, indent=2)

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
