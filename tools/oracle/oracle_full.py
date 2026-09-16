#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle Full Forward Reference (PyTorch FP32).

Menjalankan full forward pass referensi 24-layer deterministik (fail-closed CPU-only).
Pipeline:
  Token IDs -> Embedding Lookup -> 24 Layer Transformer
            -> Final RMSNorm -> LM Head -> Logits FP32 [s, V]
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch
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
        fail(
            "M4_ERR_CONFIG",
            f"config file not found: {config_path}",
            stage="config",
        )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        fail(
            "M4_ERR_CONFIG",
            f"failed to parse config JSON: {e}",
            stage="config",
        )

    for req in ["hidden_size", "num_attention_heads"]:
        if req not in cfg:
            fail(
                "M4_ERR_CONFIG",
                f"missing required field: {req}",
                stage="config",
            )
        if not isinstance(cfg[req], int) or cfg[req] <= 0:
            fail(
                "M4_ERR_CONFIG",
                f"field {req} must be a positive integer",
                stage="config",
            )

    cfg["num_hidden_layers"] = int(cfg.get("num_hidden_layers", 24))
    cfg["vocab_size"] = int(cfg.get("vocab_size", 151936))
    cfg["rms_norm_eps"] = float(cfg.get("rms_norm_eps", 1e-6))
    cfg["rope_theta"] = float(cfg.get("rope_theta", 1000000.0))
    cfg["num_experts"] = int(cfg.get("num_experts", 60))
    cfg["num_experts_per_tok"] = int(cfg.get("num_experts_per_tok", 4))
    cfg["moe_intermediate_size"] = int(cfg.get("moe_intermediate_size", 1408))
    cfg["shared_expert_intermediate_size"] = int(
        cfg.get("shared_expert_intermediate_size", 5632)
    )
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
                "tokens JSON dict missing 'tokens' or 'prompts' field",
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
            "token sequence cannot be empty",
            stage="input",
            exit_code=1,
        )

    for idx, tid in enumerate(token_list):
        if not isinstance(tid, int):
            fail(
                "M4_ERR_INPUT",
                f"token at index {idx} is not an integer",
                stage="input",
                exit_code=1,
            )
        if tid < 0 or tid >= vocab_size:
            fail(
                "M4_ERR_INPUT",
                f"token {tid} at index {idx} out of vocab range [0, {vocab_size})",
                stage="input",
                exit_code=1,
            )

    return token_list


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Formula F7 rotate_half: [-x_{half..}, x_{..half}]."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def forward_attention(
    x: torch.Tensor,
    layer_idx: int,
    cfg: dict,
    get_tensor,
) -> torch.Tensor:
    """Eksekusi attention block:
    RMSNorm -> QKV -> RoPE -> MHA -> o_proj -> Residual 1.
    """
    pfx = f"model.layers.{layer_idx}."
    seq_len = x.shape[0]
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    head_dim = hidden_size // num_heads
    eps = cfg["rms_norm_eps"]
    base_theta = cfg["rope_theta"]

    w_in_norm = get_tensor(f"{pfx}input_layernorm.weight", "attention", layer_idx)
    var_in = torch.mean(x**2, dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var_in + eps) * w_in_norm

    wq = get_tensor(f"{pfx}self_attn.q_proj.weight", "attention", layer_idx)
    bq = get_tensor(f"{pfx}self_attn.q_proj.bias", "attention", layer_idx)
    wk = get_tensor(f"{pfx}self_attn.k_proj.weight", "attention", layer_idx)
    bk = get_tensor(f"{pfx}self_attn.k_proj.bias", "attention", layer_idx)
    wv = get_tensor(f"{pfx}self_attn.v_proj.weight", "attention", layer_idx)
    bv = get_tensor(f"{pfx}self_attn.v_proj.bias", "attention", layer_idx)

    q = torch.matmul(x_norm, wq.t()) + bq
    k = torch.matmul(x_norm, wk.t()) + bk
    v = torch.matmul(x_norm, wv.t()) + bv

    qh = q.view(seq_len, num_heads, head_dim)
    kh = k.view(seq_len, num_heads, head_dim)

    inv_freq = 1.0 / (
        base_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
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

    wo = get_tensor(f"{pfx}self_attn.o_proj.weight", "attention", layer_idx)
    attn_proj = torch.matmul(attn_out, wo.t())

    x_out = x + attn_proj
    if not torch.isfinite(x_out).all():
        fail(
            "M4_ERR_LAYER_FORWARD",
            f"non-finite values after attention residual in layer {layer_idx}",
            stage="attention",
            layer=layer_idx,
            exit_code=5,
        )
    return x_out


def forward_moe(
    x: torch.Tensor,
    layer_idx: int,
    cfg: dict,
    get_tensor,
    dump_routing_dir: str | None = None,
) -> torch.Tensor:
    """Eksekusi MoE block: RMSNorm -> Router -> Experts -> Shared -> Residual 2."""
    pfx = f"model.layers.{layer_idx}."
    seq_len = x.shape[0]
    hidden_size = cfg["hidden_size"]
    eps = cfg["rms_norm_eps"]
    top_k = cfg["num_experts_per_tok"]

    w_post_norm = get_tensor(f"{pfx}post_attention_layernorm.weight", "moe", layer_idx)
    var_post = torch.mean(x**2, dim=-1, keepdim=True)
    x_norm_moe = x * torch.rsqrt(var_post + eps) * w_post_norm

    w_router = get_tensor(f"{pfx}mlp.gate.weight", "router", layer_idx)
    router_logits = torch.matmul(x_norm_moe, w_router.t())
    router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)

    topk_probs, topk_indices = torch.topk(router_probs, k=top_k, dim=-1, sorted=True)

    selected_experts = []
    for t in range(seq_len):
        row_exp = [int(topk_indices[t, k].item()) for k in range(top_k)]
        selected_experts.append(row_exp)

    if dump_routing_dir:
        rout_file = os.path.join(dump_routing_dir, f"routing_L{layer_idx}.json")
        with open(rout_file, "w", encoding="utf-8") as rf:
            json.dump({"selected_experts": selected_experts}, rf)
            rf.write("\n")

    unique_experts = sorted(list({e for row in selected_experts for e in row}))
    y_routed = torch.zeros_like(x)

    for exp_id in unique_experts:
        pfx_exp = f"{pfx}mlp.experts.{exp_id}."
        wg = get_tensor(pfx_exp + "gate_proj.weight", "experts", layer_idx)
        wu = get_tensor(pfx_exp + "up_proj.weight", "experts", layer_idx)
        wd = get_tensor(pfx_exp + "down_proj.weight", "experts", layer_idx)

        for t in range(seq_len):
            for k in range(top_k):
                if selected_experts[t][k] == exp_id:
                    p = float(topk_probs[t, k].item())
                    xt = x_norm_moe[t : t + 1]
                    g = torch.nn.functional.silu(torch.matmul(xt, wg.t()))
                    u = torch.matmul(xt, wu.t())
                    e_out = torch.matmul(g * u, wd.t())[0]
                    y_routed[t] += p * e_out

    pfx_sh = f"{pfx}mlp.shared_expert."
    w_sh_gate_proj = get_tensor(pfx_sh + "gate_proj.weight", "shared", layer_idx)
    w_sh_up_proj = get_tensor(pfx_sh + "up_proj.weight", "shared", layer_idx)
    w_sh_down_proj = get_tensor(pfx_sh + "down_proj.weight", "shared", layer_idx)
    w_sh_gate = get_tensor(
        f"{pfx}mlp.shared_expert_gate.weight", "shared", layer_idx
    ).view(1, hidden_size)

    shared_logits = torch.matmul(x_norm_moe, w_sh_gate.t())
    g_sh = torch.sigmoid(shared_logits)

    sh_g = torch.nn.functional.silu(torch.matmul(x_norm_moe, w_sh_gate_proj.t()))
    sh_u = torch.matmul(x_norm_moe, w_sh_up_proj.t())
    e_sh = torch.matmul(sh_g * sh_u, w_sh_down_proj.t())
    y_shared = g_sh * e_sh

    moe_out = y_routed + y_shared
    x_out = x + moe_out

    if not torch.isfinite(x_out).all():
        fail(
            "M4_ERR_LAYER_FORWARD",
            f"non-finite values after MoE residual in layer {layer_idx}",
            stage="moe",
            layer=layer_idx,
            exit_code=5,
        )
    return x_out


def write_logits_and_hash(
    logits: torch.Tensor,
    out_file: str,
    sha256_dest: str | None,
    seq_len: int,
    vocab_size: int,
) -> str:
    """Serialisasi output logits ke file biner dan tulis file sha256."""
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_bytes = logits.detach().cpu().to(torch.float32).numpy().tobytes()
    expected_bytes = seq_len * vocab_size * 4
    if len(out_bytes) != expected_bytes:
        fail(
            "M4_ERR_OUTPUT",
            f"output bytes {len(out_bytes)} != expected {expected_bytes}",
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
        help="Directory to dump per-layer routing JSON files (routing_L0..23.json)",
    )
    parser.add_argument(
        "--sha256",
        default=None,
        help="Path to write SHA-256 checksum file (defaults to <output>.sha256)",
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
        fail("M4_ERR_INDEX", f"failed to parse index: {e}", stage="index_load")

    shards_cache = {}

    def get_tensor(tensor_name: str, stage: str, layer_i: int = -1) -> torch.Tensor:
        if tensor_name not in weight_map:
            fail(
                "M4_ERR_SHARD_IO",
                f"tensor {tensor_name} missing from index weight_map",
                stage=stage,
                layer=layer_i,
                exit_code=4,
            )
        shard_file = weight_map[tensor_name]
        shard_path = os.path.join(args.model_dir, shard_file)
        if not os.path.exists(shard_path):
            fail(
                "M4_ERR_SHARD_IO",
                f"shard file {shard_file} not found on disk",
                stage=stage,
                layer=layer_i,
                exit_code=4,
            )
        if shard_file not in shards_cache:
            try:
                shards_cache[shard_file] = safe_open(
                    shard_path, framework="pt", device="cpu"
                )
            except Exception as e:
                fail(
                    "M4_ERR_SHARD_IO",
                    f"failed to open shard {shard_file}: {e}",
                    stage=stage,
                    layer=layer_i,
                    exit_code=4,
                )
        shard = shards_cache[shard_file]
        try:
            return shard.get_tensor(tensor_name).float()
        except Exception as e:
            fail(
                "M4_ERR_SHARD_IO",
                f"failed to read tensor {tensor_name} from shard {shard_file}: {e}",
                stage=stage,
                layer=layer_i,
                exit_code=4,
            )

    w_embed = get_tensor("model.embed_tokens.weight", stage="embedding")
    token_t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
    x = torch.embedding(w_embed, token_t).clone()
    del w_embed

    if not torch.isfinite(x).all():
        fail(
            "M4_ERR_LAYER_FORWARD",
            "non-finite values in embedding output",
            stage="embedding",
            exit_code=5,
        )

    if args.dump_routing:
        os.makedirs(args.dump_routing, exist_ok=True)

    for layer_idx in range(num_layers):
        x = forward_attention(x, layer_idx, cfg, get_tensor)
        x = forward_moe(x, layer_idx, cfg, get_tensor, args.dump_routing)

    w_final_norm = get_tensor("model.norm.weight", stage="final_norm")
    var_final = torch.mean(x**2, dim=-1, keepdim=True)
    x_final_norm = x * torch.rsqrt(var_final + eps) * w_final_norm

    w_lm_head = get_tensor("lm_head.weight", stage="lm_head")
    logits = torch.matmul(x_final_norm, w_lm_head.t())

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
