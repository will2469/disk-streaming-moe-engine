#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle Layer Path — PyTorch FP32 Reference for Part Attention (M2-W5).

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

CLI:
  oracle_layer.py --part attn --layer 0|12|23 --activation <path>
                  --model-dir <dir> [--output <path>]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import torch
from safetensors.torch import load_file


def fail(
    error_type: str, detail: str, stage: str = "attention", layer: int = -1
) -> None:
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

    num_hidden_layers = cfg.get("num_hidden_layers", 24)
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

    rope_theta = float(cfg.get("rope_theta", 1000000.0))
    cfg["rope_theta"] = rope_theta

    return cfg


def load_activation(path: str, hidden_dim: int, layer: int) -> tuple[torch.Tensor, int]:
    if not os.path.exists(path):
        fail(
            "FILE_NOT_FOUND",
            f"activation file not found: {path}",
            stage="attention",
            layer=layer,
        )
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        fail(
            "ACT_LOAD_FAILED",
            f"cannot read activation file: {e}",
            stage="attention",
            layer=layer,
        )

    if len(raw_bytes) == 0 or len(raw_bytes) % (hidden_dim * 4) != 0:
        fail(
            "ACT_LOAD_FAILED",
            f"activation byte size {len(raw_bytes)} not divisible "
            f"by row size {hidden_dim * 4}",
            stage="attention",
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
            stage="attention",
            layer=layer,
        )

    if torch.max(torch.abs(floats)).item() > 1e6:
        fail(
            "ACT_LOAD_FAILED",
            "activation contains values out of reasonable range (>1e6)",
            stage="attention",
            layer=layer,
        )

    return floats, num_tokens


def _load_tensor_from_shard(
    model_dir: str,
    weight_map: dict[str, str],
    name: str,
    shards_cache: dict[str, dict[str, torch.Tensor]],
    layer: int,
) -> torch.Tensor:
    shard_file = weight_map[name]
    shard_path = os.path.join(model_dir, shard_file)
    if not os.path.exists(shard_path):
        fail(
            "FILE_NOT_FOUND",
            f"shard file not found on disk: {shard_path}",
            stage="attention",
            layer=layer,
        )
    if shard_file not in shards_cache:
        try:
            shards_cache[shard_file] = load_file(shard_path, device="cpu")
        except Exception as e:
            fail(
                "WEIGHT_LOAD_FAILED",
                f"failed to load shard {shard_file}: {e}",
                stage="attention",
                layer=layer,
            )
    shard_dict = shards_cache[shard_file]
    if name not in shard_dict:
        fail(
            "WEIGHT_LOAD_FAILED",
            f"tensor {name} missing from shard {shard_file}",
            stage="attention",
            layer=layer,
        )
    return shard_dict[name].float()


def _validate_weight_shapes(
    weights: dict[str, torch.Tensor],
    pfx: str,
    hidden_size: int,
    layer: int,
) -> None:
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
                f"{desc} shape mismatch: {actual}",
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

    pfx = f"model.layers.{layer}."
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
            model_dir, weight_map, req, shards_cache, layer
        )

    o_bias_name = f"{pfx}self_attn.o_proj.bias"
    if o_bias_name in weight_map:
        weights[o_bias_name] = _load_tensor_from_shard(
            model_dir, weight_map, o_bias_name, shards_cache, layer
        )

    _validate_weight_shapes(weights, pfx, cfg["hidden_size"], layer)
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
    """Full 10-step reference attention block execution."""
    pfx = f"model.layers.{layer}."
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    head_dim = hidden_size // num_heads
    seq_len = x.shape[0]
    eps = cfg["rms_norm_eps"]
    base = cfg["rope_theta"]

    # 1 & 2. RMSNorm F6
    var = torch.mean(x**2, dim=-1, keepdim=True)
    norm_w = weights[f"{pfx}input_layernorm.weight"]
    x_norm = x * torch.rsqrt(var + eps) * norm_w

    # 3 & 4. QKV Projection with bias
    wq = weights[f"{pfx}self_attn.q_proj.weight"]
    bq = weights[f"{pfx}self_attn.q_proj.bias"]
    wk = weights[f"{pfx}self_attn.k_proj.weight"]
    bk = weights[f"{pfx}self_attn.k_proj.bias"]
    wv = weights[f"{pfx}self_attn.v_proj.weight"]
    bv = weights[f"{pfx}self_attn.v_proj.bias"]

    q = torch.matmul(x_norm, wq.t()) + bq
    k = torch.matmul(x_norm, wk.t()) + bk
    v = torch.matmul(x_norm, wv.t()) + bv

    # 5. RoPE rotate_half F7
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

    # 6 & 7. Causal MHA with stable softmax
    qh = q_rot.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
    kh = k_rot.permute(1, 0, 2)
    vh = v.view(seq_len, num_heads, head_dim).permute(1, 0, 2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale

    # Triangular causal mask: 0 if i >= j, -inf if i < j
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )
    scores = scores + mask

    # Softmax stabil: max shift
    max_s = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_s = torch.exp(scores - max_s)
    attn_weights = exp_s / torch.sum(exp_s, dim=-1, keepdim=True)

    attn_out = torch.matmul(attn_weights, vh)
    attn_out = attn_out.permute(1, 0, 2).contiguous().view(seq_len, hidden_size)

    # 8. o_proj + bias
    wo = weights[f"{pfx}self_attn.o_proj.weight"]
    y = torch.matmul(attn_out, wo.t())
    o_bias_name = f"{pfx}self_attn.o_proj.bias"
    if o_bias_name in weights:
        y = y + weights[o_bias_name]

    # 9. Residual connection
    y_final = y + x

    if not torch.isfinite(y_final).all():
        fail(
            "ATTENTION_ERROR",
            "non-finite value in attention oracle output",
            stage="attention",
            layer=layer,
        )

    return y_final


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Layer Attention (PyTorch fp32)"
    )
    parser.add_argument(
        "--part", required=True, choices=["attn"], help="Part name (attn)"
    )
    parser.add_argument(
        "--layer", required=True, type=int, help="Layer index (0, 12, or 23)"
    )
    parser.add_argument(
        "--activation", required=True, help="Path to input activation.bin"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing model weights and index",
    )
    parser.add_argument(
        "--output", default="attn_ref.bin", help="Path to write output binary"
    )
    args = parser.parse_args()

    if args.part != "attn":
        fail(
            "PART_INVALID",
            f"unsupported part: {args.part}",
            stage="cli",
            layer=args.layer,
        )

    if args.layer not in [0, 12, 23]:
        fail(
            "LAYER_INVALID",
            f"layer must be 0, 12, or 23, got {args.layer}",
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
    weights = load_layer_weights(args.model_dir, args.layer, cfg)

    ref_out = forward_layer_attention_oracle(act, weights, args.layer, cfg)

    # 10. Write binary output fp32 LE & compute SHA-256
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_bytes = ref_out.detach().cpu().to(torch.float32).numpy().tobytes()
    with open(args.output, "wb") as f:
        f.write(out_bytes)

    sha256_hash = hashlib.sha256(out_bytes).hexdigest()

    summary = {
        "status": "success",
        "part": "attn",
        "layer": args.layer,
        "num_tokens": num_tokens,
        "hidden_dim": cfg["hidden_size"],
        "output_file": args.output,
        "output_bytes": len(out_bytes),
        "sha256": sha256_hash,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
