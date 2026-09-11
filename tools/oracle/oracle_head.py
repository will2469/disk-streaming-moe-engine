#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle Head Path — PyTorch FP32 Reference (M1-W4).

Binding eksplisit:
  embedding = model.embed_tokens.weight   # shape [V, d], fp32
  gamma     = model.norm.weight           # shape [d], fp32
  head      = lm_head.weight              # shape [V, d], untied, fp32

Pipeline:
  Token IDs -> Embedding Lookup -> Final RMSNorm (F6) -> Untied LM Head
            -> logits_ref.bin [3, 16, V]
"""

import argparse
import json
import os
import sys
import torch
from safetensors.torch import load_file


def fail(
    error_type: str, detail: str, stage: str = "oracle", extra: dict = None
) -> None:
    payload = {
        "error_type": error_type,
        "detail": detail,
        "stage": stage,
    }
    if extra:
        payload.update(extra)
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.exit(2)


def parse_model_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        fail("CONFIG_ERROR", f"config file not found: {config_path}", stage="config")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        fail("CONFIG_ERROR", f"failed to parse config JSON: {e}", stage="config")

    for req in ["hidden_size", "vocab_size"]:
        if req not in cfg:
            fail("CONFIG_ERROR", f"missing required field: {req}", stage="config")
        if not isinstance(cfg[req], int) or cfg[req] <= 0:
            fail(
                "CONFIG_ERROR",
                f"field {req} must be a positive integer",
                stage="config",
            )

    if "rms_norm_eps" not in cfg:
        fail("CONFIG_ERROR", "missing required field: rms_norm_eps", stage="config")
    try:
        eps = float(cfg["rms_norm_eps"])
    except Exception:
        fail("CONFIG_ERROR", "rms_norm_eps must be numeric", stage="config")

    if eps <= 0:
        fail("CONFIG_ERROR", "rms_norm_eps must be positive", stage="config")

    cfg["rms_norm_eps"] = eps
    return cfg


def parse_tokens(tokens_path: str, vocab_size: int) -> torch.Tensor:
    if not os.path.exists(tokens_path):
        fail(
            "FILE_NOT_FOUND",
            f"cannot open tokens file: {tokens_path}",
            stage="embedding",
        )
    try:
        with open(tokens_path, "r", encoding="utf-8") as f:
            tokens_data = json.load(f)
    except Exception as e:
        fail("JSON_PARSE_ERROR", f"failed to parse tokens JSON: {e}", stage="embedding")

    if not isinstance(tokens_data, list) or len(tokens_data) != 3:
        n_p = len(tokens_data) if isinstance(tokens_data, list) else "non-list"
        fail("TOKEN_INVALID", f"expected 3 prompts, got {n_p}", stage="embedding")

    for p_idx, prompt in enumerate(tokens_data):
        if not isinstance(prompt, list) or len(prompt) != 16:
            p_len = len(prompt) if isinstance(prompt, list) else "non-list"
            fail(
                "TOKEN_INVALID",
                f"prompt length must be 16, got {p_len}",
                stage="embedding",
                extra={"prompt_idx": p_idx},
            )
        for t_idx, tid in enumerate(prompt):
            if not isinstance(tid, int) or tid < 0 or tid >= vocab_size:
                fail(
                    "TOKEN_INVALID",
                    f"Token ID {tid} out of bounds [0, {vocab_size})",
                    stage="embedding",
                    extra={"prompt_idx": p_idx, "token_pos": t_idx, "token_id": tid},
                )

    return torch.tensor(tokens_data, dtype=torch.long)


def load_weights(
    model_root: str, cfg: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index_path = os.path.join(model_root, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        fail("FILE_NOT_FOUND", f"index file not found: {index_path}", stage="embedding")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    except Exception as e:
        fail("WEIGHT_LOAD_FAILED", f"failed to parse index: {e}", stage="embedding")

    req_embed = "model.embed_tokens.weight"
    req_norm = "model.norm.weight"
    req_head = "lm_head.weight"

    for req in [req_embed, req_norm, req_head]:
        if req not in weight_map:
            fail(
                "WEIGHT_LOAD_FAILED",
                f"missing {req} in weight_map",
                stage="embedding",
                extra={"missing_tensors": [req]},
            )

    shard_embed_path = os.path.join(model_root, weight_map[req_embed])
    shard_norm_path = os.path.join(model_root, weight_map[req_norm])
    shard_head_path = os.path.join(model_root, weight_map[req_head])

    for sp in set([shard_embed_path, shard_norm_path, shard_head_path]):
        if not os.path.exists(sp):
            fail("FILE_NOT_FOUND", f"shard not found on disk: {sp}", stage="embedding")

    try:
        tensors_embed = load_file(shard_embed_path, device="cpu")
        embed = tensors_embed[req_embed].float()
    except Exception as e:
        fail(
            "WEIGHT_LOAD_FAILED",
            f"failed to load {req_embed}: {e}",
            stage="embedding",
            extra={"tensor_name": req_embed},
        )

    try:
        tensors_norm = load_file(shard_norm_path, device="cpu")
        norm = tensors_norm[req_norm].float()
    except Exception as e:
        fail(
            "WEIGHT_LOAD_FAILED",
            f"failed to load {req_norm}: {e}",
            stage="rmsnorm",
            extra={"tensor_name": req_norm},
        )

    try:
        tensors_head = load_file(shard_head_path, device="cpu")
        head = tensors_head[req_head].float()
    except Exception as e:
        fail(
            "WEIGHT_LOAD_FAILED",
            f"failed to load {req_head}: {e}",
            stage="lm_head",
            extra={"tensor_name": req_head},
        )

    hidden_size = cfg["hidden_size"]
    vocab_size = cfg["vocab_size"]

    if embed.shape != torch.Size([vocab_size, hidden_size]):
        fail(
            "WEIGHT_LOAD_FAILED",
            f"embed shape mismatch: expected [{vocab_size}, {hidden_size}], "
            f"got {list(embed.shape)}",
            stage="embedding",
            extra={"tensor_name": req_embed},
        )

    if norm.shape != torch.Size([hidden_size]):
        fail(
            "WEIGHT_LOAD_FAILED",
            f"norm shape mismatch: expected [{hidden_size}], got {list(norm.shape)}",
            stage="rmsnorm",
            extra={"tensor_name": req_norm},
        )

    if head.shape != torch.Size([vocab_size, hidden_size]):
        fail(
            "WEIGHT_LOAD_FAILED",
            f"head shape mismatch: expected [{vocab_size}, {hidden_size}], "
            f"got {list(head.shape)}",
            stage="lm_head",
            extra={"tensor_name": req_head},
        )

    return embed, norm, head


def rmsnorm_f6(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm F6: y = x / RMS(x) * gamma."""
    # Compute in fp32
    d = x.shape[-1]
    sum_sq = torch.sum(x * x, dim=-1, keepdim=True)
    rms = torch.sqrt(sum_sq / float(d) + eps)
    return (x / rms) * gamma


def forward_head_oracle(
    tokens: torch.Tensor,
    embed: torch.Tensor,
    gamma: torch.Tensor,
    head: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Execute token lookup -> RMSNorm -> LM head projection."""
    # 1. Lookup: [3, 16, d]
    x = embed[tokens]

    # 2. RMSNorm: [3, 16, d]
    y = rmsnorm_f6(x, gamma, eps)

    # 3. LM head matmul: [3, 16, V] (head is [V, d], untied)
    logits = torch.matmul(y, head.t())

    # Check finite
    if not torch.isfinite(logits).all():
        fail("NORM_ERROR", "non-finite value in oracle logits", stage="output")

    return logits


def main():
    parser = argparse.ArgumentParser(description="Oracle Head Path (PyTorch fp32)")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing canonical model_config.json and shards",
    )
    parser.add_argument("--tokens", required=True, help="Path to tokens.json")
    parser.add_argument("--output", required=True, help="Path to output logits_ref.bin")
    args = parser.parse_args()

    config_path = os.path.join(args.model_dir, "model_config.json")
    cfg = parse_model_config(config_path)

    tokens = parse_tokens(args.tokens, cfg["vocab_size"])
    embed, gamma, head = load_weights(args.model_dir, cfg)

    logits = forward_head_oracle(tokens, embed, gamma, head, cfg["rms_norm_eps"])

    # Write binary: float32, little-endian, contiguous
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    logits_np = logits.detach().cpu().to(torch.float32).numpy()
    with open(args.output, "wb") as f:
        f.write(logits_np.tobytes())

    total_bytes = os.path.getsize(args.output)
    expected_bytes = 3 * 16 * cfg["vocab_size"] * 4

    out_summary = {
        "status": "success",
        "num_prompts": 3,
        "tokens_per_prompt": 16,
        "num_tokens_total": 48,
        "vocab_size": cfg["vocab_size"],
        "output_file": args.output,
        "output_bytes": total_bytes,
        "expected_bytes": expected_bytes,
        "rms_norm_eps": cfg["rms_norm_eps"],
    }
    print(json.dumps(out_summary))


if __name__ == "__main__":
    main()
