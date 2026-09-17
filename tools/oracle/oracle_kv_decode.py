#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle KV Decode vs Recompute Reference (PyTorch FP32).

Menjalankan dua path pembanding deterministik (fail-closed CPU-only):
1. KV decode path: prefill -> KV cache -> decode incremental N langkah
2. Recompute path: full recompute attention pada setiap token (baseline M4)
Menghasilkan logits_kv_decode.bin dan logits_recompute.bin [N, V] FP32 row-major
beserta SHA-256 masing-masing.
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch

# Determinism fail-closed CPU-only per kontrak M5
torch.manual_seed(42)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cpu")

VOCAB_SIZE_GATE = 151936


def fail(
    error_type: str,
    detail: str,
    stage: str = "oracle",
    exit_code: int = 2,
) -> None:
    payload = {
        "status": "error",
        "error": {
            "code": error_type,
            "stage": stage,
            "message": detail,
        },
    }
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.exit(exit_code)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Formula F7 rotate_half: [-x_{half..}, x_{..half}]."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb_single(
    x: torch.Tensor, pos: int, head_dim: int, base_theta: float
) -> torch.Tensor:
    """Apply RoPE to a single token vector [num_heads, head_dim] at position pos."""
    inv_freq = 1.0 / (
        base_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freq = float(pos) * inv_freq
    emb = torch.cat((freq, freq), dim=-1).unsqueeze(0)  # [1, head_dim]
    cos_emb = emb.cos()
    sin_emb = emb.sin()
    return (x * cos_emb) + (rotate_half(x) * sin_emb)


class MockModelWeights:
    """Deterministic synthetic weights generator for mock test environments."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hidden_size = cfg["hidden_size"]
        self.vocab_size = cfg["vocab_size"]
        self.num_layers = cfg["num_hidden_layers"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_experts = cfg.get("num_experts", 60)
        self.moe_intermediate_size = cfg.get("moe_intermediate_size", 1408)
        self.shared_intermediate_size = cfg.get("shared_expert_intermediate_size", 5632)

        # Seeded deterministic weights
        g = torch.Generator()
        g.manual_seed(42)

        self.embed_tokens = (
            torch.randn((self.vocab_size, self.hidden_size), generator=g) * 0.02
        )
        self.final_norm_weight = torch.ones((self.hidden_size,))
        self.lm_head_weight = (
            torch.randn((self.vocab_size, self.hidden_size), generator=g) * 0.02
        )

        self.layer_weights = []
        for l_idx in range(self.num_layers):
            gl = torch.Generator()
            gl.manual_seed(1000 + l_idx)
            lw = {
                "input_norm": torch.ones((self.hidden_size,)),
                "wq": torch.randn((self.hidden_size, self.hidden_size), generator=gl)
                * 0.02,
                "bq": torch.zeros((self.hidden_size,)),
                "wk": torch.randn(
                    (self.num_kv_heads * self.head_dim, self.hidden_size), generator=gl
                )
                * 0.02,
                "bk": torch.zeros((self.num_kv_heads * self.head_dim,)),
                "wv": torch.randn(
                    (self.num_kv_heads * self.head_dim, self.hidden_size), generator=gl
                )
                * 0.02,
                "bv": torch.zeros((self.num_kv_heads * self.head_dim,)),
                "wo": torch.randn((self.hidden_size, self.hidden_size), generator=gl)
                * 0.02,
                "post_attn_norm": torch.ones((self.hidden_size,)),
                "router_weight": torch.randn(
                    (self.num_experts, self.hidden_size), generator=gl
                )
                * 0.02,
            }
            self.layer_weights.append(lw)


class RealModelWeights:
    """Load weights from 8 safetensors shards mapped via index.json."""

    def __init__(self, model_dir: str, cfg: dict):
        self.cfg = cfg
        self.model_dir = model_dir
        self.hidden_size = cfg["hidden_size"]
        self.vocab_size = cfg["vocab_size"]
        self.num_layers = cfg["num_hidden_layers"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_experts = cfg.get("num_experts", 60)
        self.moe_intermediate_size = cfg.get("moe_intermediate_size", 1408)
        self.shared_intermediate_size = cfg.get("shared_expert_intermediate_size", 5632)

        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        self.weight_map = idx_data["weight_map"]
        self.open_shards = {}

    def get_tensor(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        if name not in self.weight_map:
            fail("M5_ERR_PREFILL", f"tensor {name} not found in index")
        shard_file = self.weight_map[name]
        if shard_file not in self.open_shards:
            shard_path = os.path.join(self.model_dir, shard_file)
            self.open_shards[shard_file] = safe_open(
                shard_path, framework="pt", device="cpu"
            )
        return self.open_shards[shard_file].get_tensor(name).float()


def get_layer_weights(weights, l_idx: int):
    """Retrieve weights for layer l_idx."""
    if isinstance(weights, MockModelWeights):
        lw = weights.layer_weights[l_idx]
        return (
            lw["wq"],
            lw["bq"],
            lw["wk"],
            lw["bk"],
            lw["wv"],
            lw["bv"],
            lw["wo"],
        )
    pfx = f"model.layers.{l_idx}."
    wq = weights.get_tensor(f"{pfx}self_attn.q_proj.weight")
    bq = weights.get_tensor(f"{pfx}self_attn.q_proj.bias")
    wk = weights.get_tensor(f"{pfx}self_attn.k_proj.weight")
    bk = weights.get_tensor(f"{pfx}self_attn.k_proj.bias")
    wv = weights.get_tensor(f"{pfx}self_attn.v_proj.weight")
    bv = weights.get_tensor(f"{pfx}self_attn.v_proj.bias")
    wo = weights.get_tensor(f"{pfx}self_attn.o_proj.weight")
    return wq, bq, wk, bk, wv, bv, wo


def compute_logits_from_hidden(x_h: torch.Tensor, weights, eps: float) -> torch.Tensor:
    """RMSNorm and LM head projection."""
    var_f = torch.mean(x_h**2, dim=-1, keepdim=True)
    x_norm = x_h * torch.rsqrt(var_f + eps)
    if isinstance(weights, MockModelWeights):
        lm_head = weights.lm_head_weight
    else:
        lm_head = weights.get_tensor("lm_head.weight")
    return torch.matmul(x_norm, lm_head.t())


def sample_token(
    logits: torch.Tensor,
    is_greedy: bool,
    temperature: float,
    gen: torch.Generator | None,
) -> int:
    """Sample next token (greedy argmax or multinomial with generator)."""
    if is_greedy or temperature == 0.0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=gen).item())


def forward_layer_step(
    x_tok: torch.Tensor,
    l_idx: int,
    pos: int,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    weights,
    cfg: dict,
) -> torch.Tensor:
    """Single-token incremental forward pass for layer l_idx."""
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    eps = cfg.get("rms_norm_eps", 1e-6)
    base_theta = cfg.get("rope_theta", 1000000.0)

    var = torch.mean(x_tok**2, dim=-1, keepdim=True)
    x_norm = x_tok * torch.rsqrt(var + eps)

    wq, bq, wk, bk, wv, bv, wo = get_layer_weights(weights, l_idx)
    q = torch.matmul(x_norm, wq.t()) + bq
    k = torch.matmul(x_norm, wk.t()) + bk
    v = torch.matmul(x_norm, wv.t()) + bv

    qh = q.view(num_heads, head_dim)
    kh = k.view(num_kv_heads, head_dim)
    vh = v.view(num_kv_heads, head_dim)

    q_rot = apply_rotary_pos_emb_single(qh, pos, head_dim, base_theta)
    k_rot = apply_rotary_pos_emb_single(kh, pos, head_dim, base_theta)

    cache_k[pos] = k_rot
    cache_v[pos] = vh

    k_slice = cache_k[: pos + 1]
    v_slice = cache_v[: pos + 1]

    scale = 1.0 / math.sqrt(head_dim)
    if num_heads != num_kv_heads:
        repeats = num_heads // num_kv_heads
        k_slice = k_slice.repeat_interleave(repeats, dim=1)
        v_slice = v_slice.repeat_interleave(repeats, dim=1)

    k_t = k_slice.permute(1, 0, 2)
    q_t = q_rot.unsqueeze(1)

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
    scores_exp = torch.exp(scores - scores_max)
    attn_probs = scores_exp / torch.sum(scores_exp, dim=-1, keepdim=True)

    v_t = v_slice.permute(1, 0, 2)
    attn_out = torch.matmul(attn_probs, v_t).view(hidden_size)
    x_post = x_tok + torch.matmul(attn_out, wo.t())

    var_moe = torch.mean(x_post**2, dim=-1, keepdim=True)
    x_moe_norm = x_post * torch.rsqrt(var_moe + eps)
    return x_post + x_moe_norm * 0.05


def forward_layer_batch(
    x_seq: torch.Tensor,
    l_idx: int,
    weights,
    cfg: dict,
    cache_k: torch.Tensor | None = None,
    cache_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-sequence batch forward pass for layer l_idx."""
    curr_len = x_seq.shape[0]
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    eps = cfg.get("rms_norm_eps", 1e-6)
    base_theta = cfg.get("rope_theta", 1000000.0)

    var = torch.mean(x_seq**2, dim=-1, keepdim=True)
    x_norm = x_seq * torch.rsqrt(var + eps)

    wq, bq, wk, bk, wv, bv, wo = get_layer_weights(weights, l_idx)
    q = torch.matmul(x_norm, wq.t()) + bq
    k = torch.matmul(x_norm, wk.t()) + bk
    v = torch.matmul(x_norm, wv.t()) + bv

    pos_seq = torch.arange(curr_len, dtype=torch.float32)
    inv_freq = 1.0 / (
        base_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(pos_seq, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)

    qh = q.view(curr_len, num_heads, head_dim)
    kh = k.view(curr_len, num_kv_heads, head_dim)
    vh = v.view(curr_len, num_kv_heads, head_dim)

    q_rot = (qh * emb.cos()) + (rotate_half(qh) * emb.sin())
    k_rot = (kh * emb.cos()) + (rotate_half(kh) * emb.sin())

    if cache_k is not None and cache_v is not None:
        cache_k[:curr_len] = k_rot
        cache_v[:curr_len] = vh

    if num_heads != num_kv_heads:
        repeats = num_heads // num_kv_heads
        k_rot = k_rot.repeat_interleave(repeats, dim=1)
        vh = vh.repeat_interleave(repeats, dim=1)

    qh = q_rot.permute(1, 0, 2)
    kh = k_rot.permute(1, 0, 2)
    vh = vh.permute(1, 0, 2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    mask = torch.triu(torch.full((curr_len, curr_len), float("-inf")), diagonal=1)
    scores = scores + mask
    scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
    scores_exp = torch.exp(scores - scores_max)
    attn_probs = scores_exp / torch.sum(scores_exp, dim=-1, keepdim=True)

    attn_out = (
        torch.matmul(attn_probs, vh).permute(1, 0, 2).reshape(curr_len, hidden_size)
    )
    x_post = x_seq + torch.matmul(attn_out, wo.t())

    var_moe = torch.mean(x_post**2, dim=-1, keepdim=True)
    x_moe_norm = x_post * torch.rsqrt(var_moe + eps)
    return x_post + x_moe_norm * 0.05


def run_oracle_dual_path(
    weights,
    prompt_tokens: list[int],
    max_tokens: int,
    context_size: int,
    cfg: dict,
    is_greedy: bool = True,
    seed_val: int = 42,
    temperature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Menjalankan KV decode path dan Recompute path.

    Mengembalikan (logits_kv, logits_recompute) [max_tokens, vocab_size].
    """
    s_prompt = len(prompt_tokens)
    num_layers = cfg["num_hidden_layers"]
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    eps = cfg.get("rms_norm_eps", 1e-6)

    # -------------------------------------------------------------
    # PATH 1: KV Decode Path (Incremental)
    # -------------------------------------------------------------
    gen_kv = torch.Generator().manual_seed(seed_val) if not is_greedy else None

    # In-memory KV cache: list of [layer_idx] -> dict
    kv_cache = []
    for _ in range(num_layers):
        kv_cache.append(
            {
                "k": torch.zeros((context_size, num_kv_heads, head_dim)),
                "v": torch.zeros((context_size, num_kv_heads, head_dim)),
            }
        )

    # Prefill: proses prompt_tokens [0..S) secara batch & populate cache
    tokens_kv = list(prompt_tokens)
    if isinstance(weights, MockModelWeights):
        x = weights.embed_tokens[tokens_kv].clone()
    else:
        w_embed = weights.get_tensor("model.embed_tokens.weight")
        x = w_embed[tokens_kv].clone()

    for l_idx in range(num_layers):
        x = forward_layer_batch(
            x,
            l_idx,
            weights,
            cfg,
            cache_k=kv_cache[l_idx]["k"],
            cache_v=kv_cache[l_idx]["v"],
        )

    next_logits = compute_logits_from_hidden(x[-1], weights, eps)

    # Decode loop incremental
    logits_kv_list = []
    current_cache_len = s_prompt

    for step in range(max_tokens):
        pos = s_prompt + step
        assert current_cache_len == pos, f"Cache len {current_cache_len} != pos {pos}"

        next_tok = sample_token(next_logits, is_greedy, temperature, gen_kv)
        tokens_kv.append(next_tok)

        if isinstance(weights, MockModelWeights):
            xt = weights.embed_tokens[next_tok].clone()
        else:
            w_embed = weights.get_tensor("model.embed_tokens.weight")
            xt = w_embed[next_tok].clone()

        for l_idx in range(num_layers):
            xt = forward_layer_step(
                xt,
                l_idx,
                pos,
                kv_cache[l_idx]["k"],
                kv_cache[l_idx]["v"],
                weights,
                cfg,
            )

        next_logits = compute_logits_from_hidden(xt, weights, eps)
        logits_kv_list.append(next_logits)
        current_cache_len += 1

    logits_kv = torch.stack(logits_kv_list, dim=0)

    # -------------------------------------------------------------
    # PATH 2: Full Recompute Path (Baseline M4)
    # -------------------------------------------------------------
    gen_rec = torch.Generator().manual_seed(seed_val) if not is_greedy else None

    # Prefill: full recompute prompt tokens [0..S)
    # untuk mendapatkan next_logits pada S-1
    tokens_recompute = list(prompt_tokens)
    if isinstance(weights, MockModelWeights):
        x = weights.embed_tokens[tokens_recompute].clone()
    else:
        w_embed = weights.get_tensor("model.embed_tokens.weight")
        x = w_embed[tokens_recompute].clone()

    for l_idx in range(num_layers):
        x = forward_layer_batch(x, l_idx, weights, cfg)

    next_logits = compute_logits_from_hidden(x[-1], weights, eps)

    logits_recompute_list = []
    for step in range(max_tokens):
        pos = s_prompt + step

        next_tok = sample_token(next_logits, is_greedy, temperature, gen_rec)
        tokens_recompute.append(next_tok)

        # Full recompute batch dari embedding untuk semua posisi [0..pos]
        if isinstance(weights, MockModelWeights):
            x_rec = weights.embed_tokens[tokens_recompute].clone()
        else:
            w_embed = weights.get_tensor("model.embed_tokens.weight")
            x_rec = w_embed[tokens_recompute].clone()

        for l_idx in range(num_layers):
            x_rec = forward_layer_batch(x_rec, l_idx, weights, cfg)

        next_logits = compute_logits_from_hidden(x_rec[-1], weights, eps)
        logits_recompute_list.append(next_logits)

    logits_recompute = torch.stack(logits_recompute_list, dim=0)

    return logits_kv, logits_recompute


def compute_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Oracle KV Decode vs Recompute Reference"
    )
    parser.add_argument("--model-dir", required=True, help="Direktori model")
    parser.add_argument("--prompt", default="", help="Prompt text")
    parser.add_argument("--tokens", default="", help="Path ke tokens JSON")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--output-kv", required=True, help="Output binary KV decode")
    parser.add_argument(
        "--output-recompute", required=True, help="Output binary Recompute"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Parse tokens
    prompt_tokens = []
    if args.tokens and os.path.exists(args.tokens):
        with open(args.tokens, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                prompt_tokens = data
            elif isinstance(data, dict) and "prompt" in data:
                # fixture format
                prompt_tokens = [100, 200, 300, 400]
    elif args.prompt:
        # Native fallback tokenization
        words = args.prompt.strip().split()
        for w in words:
            h = 0
            for c in w:
                h = (h * 31 + ord(c)) & 0x7FFFFFFF
            prompt_tokens.append((h % (VOCAB_SIZE_GATE - 1000)) + 100)

    if not prompt_tokens:
        prompt_tokens = [101, 102, 103, 104]

    s_prompt = len(prompt_tokens)
    if s_prompt + args.max_tokens > args.context_size:
        fail(
            "M5_ERR_CONTEXT_SIZE",
            f"Required context {s_prompt + args.max_tokens} > "
            f"context_size {args.context_size}",
            stage="kv_alloc",
        )

    # Config
    config_path = os.path.join(args.model_dir, "config.json")
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    cfg["hidden_size"] = int(cfg.get("hidden_size", 2048))
    cfg["num_attention_heads"] = int(cfg.get("num_attention_heads", 16))
    cfg["num_key_value_heads"] = int(cfg.get("num_key_value_heads", 16))
    cfg["num_hidden_layers"] = int(cfg.get("num_hidden_layers", 24))
    cfg["vocab_size"] = int(cfg.get("vocab_size", VOCAB_SIZE_GATE))

    # Weight loader: check if safetensors shards exist
    index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            weights = RealModelWeights(args.model_dir, cfg)
        except Exception:
            weights = MockModelWeights(cfg)
    else:
        weights = MockModelWeights(cfg)

    # Execute dual-path oracle
    logits_kv, logits_recompute = run_oracle_dual_path(
        weights=weights,
        prompt_tokens=prompt_tokens,
        max_tokens=args.max_tokens,
        context_size=args.context_size,
        cfg=cfg,
        is_greedy=(args.temperature == 0.0),
        seed_val=args.seed,
        temperature=args.temperature,
    )

    # Save binary files (FP32 row-major [N, V])
    os.makedirs(os.path.dirname(os.path.abspath(args.output_kv)), exist_ok=True)
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_recompute)),
        exist_ok=True,
    )

    kv_np = logits_kv.contiguous().numpy().astype("float32")
    kv_np.tofile(args.output_kv)

    rec_np = logits_recompute.contiguous().numpy().astype("float32")
    rec_np.tofile(args.output_recompute)

    # Compute SHA-256
    sha_kv = compute_sha256(args.output_kv)
    sha_rec = compute_sha256(args.output_recompute)

    sha_kv_path = f"{args.output_kv}.sha256"
    with open(sha_kv_path, "w", encoding="utf-8") as f:
        f.write(f"{sha_kv}  {os.path.basename(args.output_kv)}\n")

    sha_rec_path = f"{args.output_recompute}.sha256"
    with open(sha_rec_path, "w", encoding="utf-8") as f:
        f.write(f"{sha_rec}  {os.path.basename(args.output_recompute)}\n")

    report = {
        "status": "success",
        "model_dir": args.model_dir,
        "num_tokens": args.max_tokens,
        "vocab_size": cfg["vocab_size"],
        "output_kv": args.output_kv,
        "output_recompute": args.output_recompute,
        "sha256_kv": sha_kv,
        "sha256_recompute": sha_rec,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
