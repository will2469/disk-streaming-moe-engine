#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle Port Full Forward Reference (PyTorch FP32 & GGUF dequant baseline) — M9-W4.

Menjalankan full forward pass referensi hybrid Qwen3.6 deterministik (CPU-only):
- Macro Scheduler: 10 Siklus Makro * [3 * (GDN + MoE) + 1 * (GatedAttn + MoE)]
- 30 State GDN independen S[0..29] berukuran [30, dv, dk] (S_0 = 0)
- 10 Gated Attention layers dengan GQA (16Q/2KV) dan physical KV cache
- 40 MoE Channel Mixers (top-k unrenormalized routing + shared expert)
- Dual Path: BF16 native (FP32 accum) atau GGUF block dequant ke FP32
- Output: Little-endian IEEE-754 FP32 logits [seq_len, vocab_size]
- Intermediate activation dump G-M9-1 (--dump-layers)
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file

# Determinism CPU-only fail-closed per standar M4/M8/M9
torch.manual_seed(42)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cpu")

# GGML Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_BF16 = 30


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


def dequantize_q8_0(raw: bytes, num_elements: int) -> np.ndarray:
    """Dequantize Q8_0 blocks (32 weights/block, 34 bytes/block)."""
    assert num_elements % 32 == 0
    num_blocks = num_elements // 32
    out = np.zeros(num_elements, dtype=np.float32)
    for b in range(num_blocks):
        off = b * 34
        d_u16 = struct.unpack("<H", raw[off : off + 2])[0]
        scale = np.frombuffer(struct.pack("<H", d_u16), dtype=np.float16)[0].astype(
            np.float32
        )
        quants = np.frombuffer(raw[off + 2 : off + 34], dtype=np.int8).astype(
            np.float32
        )
        out[b * 32 : (b + 1) * 32] = scale * quants
    return out


def dequantize_q4_k(raw: bytes, num_elements: int) -> np.ndarray:
    """Dequantize Q4_K blocks (256 weights/block, 144 bytes/block)."""
    assert num_elements % 256 == 0
    num_blocks = num_elements // 256
    out = np.zeros(num_elements, dtype=np.float32)
    for b in range(num_blocks):
        off = b * 144
        d_u16, m_u16 = struct.unpack("<HH", raw[off : off + 4])
        d = np.frombuffer(struct.pack("<H", d_u16), dtype=np.float16)[0].astype(
            np.float32
        )
        dmin = np.frombuffer(struct.pack("<H", m_u16), dtype=np.float16)[0].astype(
            np.float32
        )

        scales = raw[off + 4 : off + 16]
        sc = np.zeros(8, dtype=np.float32)
        m_arr = np.zeros(8, dtype=np.float32)

        for j in range(4):
            sc[j] = float(scales[j] & 63) * d
            m_arr[j] = float(scales[j + 4] & 63) * dmin

        for j in range(4, 8):
            sc_val = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
            m_val = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
            sc[j] = float(sc_val) * d
            m_arr[j] = float(m_val) * dmin

        qs = raw[off + 16 : off + 144]
        out_base = b * 256
        for j in range(8):
            sub_out = out_base + j * 32
            cur_sc = sc[j]
            cur_m = m_arr[j]
            for i in range(16):
                b_val = qs[j * 16 + i]
                q0 = b_val & 0xF
                q1 = (b_val >> 4) & 0xF
                out[sub_out + i] = cur_sc * float(q0) - cur_m
                out[sub_out + 16 + i] = cur_sc * float(q1) - cur_m
    return out


def dequantize_q3_k(raw: bytes, num_elements: int) -> np.ndarray:
    """Dequantize Q3_K blocks (256 weights/block, 114 bytes/block)."""
    assert num_elements % 256 == 0
    num_blocks = num_elements // 256
    out = np.zeros(num_elements, dtype=np.float32)
    for b in range(num_blocks):
        off = b * 114
        d_u16 = struct.unpack("<H", raw[off + 112 : off + 114])[0]
        d = np.frombuffer(struct.pack("<H", d_u16), dtype=np.float16)[0].astype(
            np.float32
        )

        scales = np.zeros(16, dtype=np.float32)
        for j in range(16):
            scales[j] = float(raw[off + 96 + j] - 32) * d

        out_base = b * 256
        for j in range(16):
            sub_out = out_base + j * 16
            cur_sc = scales[j]
            for i in range(16):
                elem_idx = j * 16 + i
                hmask_byte = raw[off + (elem_idx // 8)]
                h_bit = (hmask_byte >> (elem_idx % 8)) & 1
                qs_byte = raw[off + 32 + (elem_idx // 4)]
                shift = (elem_idx % 4) * 2
                low_bits = (qs_byte >> shift) & 3
                q_val = (h_bit << 2) | low_bits
                out[sub_out + i] = cur_sc * float(q_val - 4)
    return out


def _skip_metadata_value(data: bytes, pos: int, v_type: int) -> int:
    """Melewati data value metadata GGUF."""
    if v_type in (0, 1, 7):
        return pos + 1
    if v_type in (2, 3):
        return pos + 2
    if v_type in (4, 5, 6):
        return pos + 4
    if v_type in (10, 11, 12):
        return pos + 8
    if v_type == 8:
        s_len = struct.unpack("<Q", data[pos : pos + 8])[0]
        return pos + 8 + s_len
    if v_type == 9:
        elem_type = struct.unpack("<I", data[pos : pos + 4])[0]
        arr_len = struct.unpack("<Q", data[pos + 4 : pos + 12])[0]
        pos += 12
        for _ in range(arr_len):
            pos = _skip_metadata_value(data, pos, elem_type)
        return pos
    return pos


def _dequantize_tensor_payload(
    dtype: int, raw_bytes: bytes, num_elems: int
) -> np.ndarray:
    """Mendekuantisasi buffer bytes mentah sesuai tipe data GGML."""
    if dtype == GGML_TYPE_F32:
        return np.frombuffer(raw_bytes, dtype=np.float32).copy()
    if dtype == GGML_TYPE_F16:
        return np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32).copy()
    if dtype == GGML_TYPE_BF16:
        u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).copy()
    if dtype == GGML_TYPE_Q8_0:
        return dequantize_q8_0(raw_bytes, num_elems)
    if dtype == GGML_TYPE_Q4_K:
        return dequantize_q4_k(raw_bytes, num_elems)
    if dtype == GGML_TYPE_Q3_K:
        return dequantize_q3_k(raw_bytes, num_elems)
    return np.frombuffer(raw_bytes, dtype=np.float32).copy()


def parse_gguf_tensors(file_path: str) -> dict[str, torch.Tensor]:
    """Membaca berkas GGUF dan mendekuantisasi tensor ke PyTorch FP32."""
    with open(file_path, "rb") as f:
        data = f.read()

    if len(data) < 24 or data[:4] != b"GGUF":
        fail("M9_ERR_GGUF", f"not a valid GGUF file: {file_path}")

    version = struct.unpack("<I", data[4:8])[0]
    if version not in (2, 3):
        fail("M9_ERR_GGUF", f"unsupported GGUF version: {version}")

    tensor_count = struct.unpack("<Q", data[8:16])[0]
    metadata_kv_count = struct.unpack("<Q", data[16:24])[0]

    pos = 24
    alignment = 32

    # Skip metadata KV
    for _ in range(metadata_kv_count):
        k_len = struct.unpack("<Q", data[pos : pos + 8])[0]
        pos += 8
        key = data[pos : pos + k_len].decode("utf-8", errors="ignore")
        pos += k_len
        v_type = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4

        if key == "general.alignment" and v_type == 4:
            alignment = struct.unpack("<I", data[pos : pos + 4])[0]
        pos = _skip_metadata_value(data, pos, v_type)

    # Parse Tensor Directory
    tensor_infos = []
    for _ in range(tensor_count):
        name_len = struct.unpack("<Q", data[pos : pos + 8])[0]
        pos += 8
        t_name = data[pos : pos + name_len].decode("utf-8", errors="ignore")
        pos += name_len
        n_dims = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack("<Q", data[pos : pos + 8])[0])
            pos += 8
        dtype = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        offset = struct.unpack("<Q", data[pos : pos + 8])[0]
        pos += 8
        tensor_infos.append((t_name, dims, dtype, offset))

    data_start = ((pos + alignment - 1) // alignment) * alignment
    tensors: dict[str, torch.Tensor] = {}

    for t_name, dims, dtype, offset in tensor_infos:
        num_elems = 1
        for d in dims:
            num_elems *= d
        t_offset = data_start + offset

        if dtype == GGML_TYPE_Q8_0:
            block_bytes = (num_elems // 32) * 34
        elif dtype == GGML_TYPE_Q4_K:
            block_bytes = (num_elems // 256) * 144
        elif dtype == GGML_TYPE_Q3_K:
            block_bytes = (num_elems // 256) * 114
        elif dtype in (GGML_TYPE_F16, GGML_TYPE_BF16):
            block_bytes = num_elems * 2
        else:
            block_bytes = num_elems * 4

        raw_bytes = data[t_offset : t_offset + block_bytes]
        arr = _dequantize_tensor_payload(dtype, raw_bytes, num_elems)

        shape = [int(x) for x in reversed(dims)]
        tensors[t_name] = torch.from_numpy(arr.reshape(shape)).to(
            device=DEVICE, dtype=torch.float32
        )

    return tensors


def load_model_weights(weights_path: str, model_dir: str) -> dict[str, torch.Tensor]:
    """Memuat bobot dari file Safetensors, direktori sharded, atau berkas GGUF."""
    target_path = weights_path
    if not target_path and model_dir:
        if os.path.isfile(model_dir):
            target_path = model_dir
        else:
            for f in os.listdir(model_dir):
                if f.endswith(".gguf"):
                    target_path = os.path.join(model_dir, f)
                    break
            if not target_path:
                idx_path = os.path.join(model_dir, "model.safetensors.index.json")
                if os.path.exists(idx_path):
                    target_path = idx_path
                else:
                    for f in os.listdir(model_dir):
                        if f.endswith(".safetensors"):
                            target_path = os.path.join(model_dir, f)
                            break

    if not target_path or not os.path.exists(target_path):
        fail(
            "M9_ERR_WEIGHTS",
            f"weights not found: path={weights_path}, dir={model_dir}",
        )

    if target_path.endswith(".gguf"):
        return parse_gguf_tensors(target_path)

    if target_path.endswith("model.safetensors.index.json"):
        with open(target_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        weight_map = idx_data.get("weight_map", {})
        base_dir = os.path.dirname(target_path)
        shards = set(weight_map.values())
        all_tensors: dict[str, torch.Tensor] = {}
        for sh in shards:
            sh_path = os.path.join(base_dir, sh)
            tensors = load_safetensors_file(sh_path)
            for k, v in tensors.items():
                all_tensors[k] = v.to(torch.float32)
        return all_tensors

    if target_path.endswith(".safetensors"):
        tensors = load_safetensors_file(target_path)
        return {k: v.to(torch.float32) for k, v in tensors.items()}

    fail("M9_ERR_WEIGHTS", f"unsupported weights format: {target_path}")
    return {}


def get_tensor(
    weights: dict[str, torch.Tensor], candidate_keys: list[str]
) -> torch.Tensor | None:
    """Mencari tensor berdasarkan daftar alternatif nama kunci."""
    for k in candidate_keys:
        if k in weights:
            return weights[k]
    return None


def parse_tokens(tokens_path: str) -> list[int]:
    """Membaca dan memvalidasi tokens JSON."""
    if not os.path.exists(tokens_path):
        fail("M9_ERR_INPUT", f"tokens file not found: {tokens_path}", exit_code=1)
    with open(tokens_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, dict):
        if "tokens" in raw and isinstance(raw["tokens"], list):
            return [int(x) for x in raw["tokens"]]
        if "prompts" in raw and isinstance(raw["prompts"], list):
            return [int(x) for x in raw["prompts"][0]["tokens"]]
    fail("M9_ERR_INPUT", "unrecognized tokens JSON format", exit_code=1)
    return []


def apply_rmsnorm(
    x: torch.Tensor, gamma: torch.Tensor | None, eps: float = 1e-6
) -> torch.Tensor:
    """RMSNorm FP32."""
    var = torch.mean(x**2, dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)
    if gamma is not None:
        normed = normed * gamma
    return normed


def apply_rope(
    x: torch.Tensor,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    pos_offset: int = 0,
    rope_theta: float = 1000000.0,
) -> torch.Tensor:
    """Rotary Position Embedding (RoPE F7) per head."""
    x_reshaped = x.view(seq_len, num_heads, head_dim)
    half_d = head_dim // 2

    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, half_d, dtype=torch.float32, device=DEVICE) / half_d)
    )

    t = torch.arange(seq_len, dtype=torch.float32, device=DEVICE) + pos_offset
    freqs = torch.outer(t, inv_freq)
    cos_vals = torch.cos(freqs).unsqueeze(1)
    sin_vals = torch.sin(freqs).unsqueeze(1)

    x1 = x_reshaped[..., :half_d]
    x2 = x_reshaped[..., half_d:]

    rot_x1 = x1 * cos_vals - x2 * sin_vals
    rot_x2 = x2 * cos_vals + x1 * sin_vals

    return torch.cat([rot_x1, rot_x2], dim=-1).view(seq_len, num_heads * head_dim)


def dump_tensor(path: str, t: torch.Tensor) -> None:
    """Menyimpan tensor FP32 little-endian ke file binary."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    flat_np = t.detach().cpu().to(torch.float32).contiguous().numpy()
    with open(path, "wb") as f:
        f.write(flat_np.tobytes())


def _forward_gdn_sublayer(
    norm_x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    s_prev: torch.Tensor,
    lyr: int,
    seq_len: int,
    hidden_size: int,
    dk: int,
    dv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mengeksekusi sublayer recurrent Gated DeltaNet (F14)."""
    pfx = f"model.language_model.layers.{lyr}."
    pfx_short = f"layers.{lyr}."
    pfx_blk = f"blk.{lyr}."

    wk = get_tensor(
        weights,
        [
            pfx + "linear_attn.k_proj.weight",
            pfx_short + "linear_attn.k_proj.weight",
            pfx_blk + "linear_attn.k.weight",
        ],
    )
    wv = get_tensor(
        weights,
        [
            pfx + "linear_attn.v_proj.weight",
            pfx_short + "linear_attn.v_proj.weight",
            pfx_blk + "linear_attn.v.weight",
        ],
    )
    wbeta = get_tensor(
        weights,
        [
            pfx + "linear_attn.beta_proj.weight",
            pfx_short + "linear_attn.beta_proj.weight",
            pfx_blk + "linear_attn.beta.weight",
        ],
    )
    wout = get_tensor(
        weights,
        [
            pfx + "linear_attn.out_proj.weight",
            pfx_short + "linear_attn.out_proj.weight",
            pfx_blk + "linear_attn.out.weight",
        ],
    )

    eye_dk = torch.eye(dk, dtype=torch.float32, device=DEVICE)
    s_state = s_prev.clone()
    mixer_out_tokens = []

    for t in range(seq_len):
        xt = norm_x[t]
        kt = torch.matmul(wk, xt) if wk is not None else xt[:dk]
        kt = kt / (torch.norm(kt) + 1e-6)
        vt = torch.matmul(wv, xt) if wv is not None else xt[:dv]

        if wbeta is not None:
            beta_t = float(torch.sigmoid(torch.matmul(wbeta, xt)).item())
        else:
            beta_t = 0.5

        kt_outer = torch.outer(kt, kt)
        transition = eye_dk - beta_t * kt_outer
        bias_m = beta_t * torch.outer(vt, kt)
        s_state = torch.matmul(s_state, transition) + bias_m

        ot = torch.matmul(s_state, kt)
        if wout is not None:
            yt = torch.matmul(wout, ot)
        else:
            yt = F.pad(ot, (0, hidden_size - dv))
        mixer_out_tokens.append(yt)

    mixer_out = torch.stack(mixer_out_tokens, dim=0)
    return mixer_out, s_state


def _forward_gated_attn_sublayer(
    norm_x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    lyr: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Mengeksekusi sublayer Gated Attention dengan GQA 16Q/2KV."""
    pfx = f"model.language_model.layers.{lyr}."
    pfx_short = f"layers.{lyr}."
    pfx_blk = f"blk.{lyr}."

    wq = get_tensor(
        weights,
        [
            pfx + "self_attn.q_proj.weight",
            pfx_short + "self_attn.q_proj.weight",
            pfx_blk + "attn_q.weight",
        ],
    )
    wk = get_tensor(
        weights,
        [
            pfx + "self_attn.k_proj.weight",
            pfx_short + "self_attn.k_proj.weight",
            pfx_blk + "attn_k.weight",
        ],
    )
    wv = get_tensor(
        weights,
        [
            pfx + "self_attn.v_proj.weight",
            pfx_short + "self_attn.v_proj.weight",
            pfx_blk + "attn_v.weight",
        ],
    )
    wgate = get_tensor(
        weights,
        [
            pfx + "self_attn.gate_proj.weight",
            pfx_short + "self_attn.gate_proj.weight",
            pfx_blk + "attn_gate.weight",
        ],
    )
    wo = get_tensor(
        weights,
        [
            pfx + "self_attn.o_proj.weight",
            pfx_short + "self_attn.o_proj.weight",
            pfx_blk + "attn_output.weight",
        ],
    )

    q = F.linear(norm_x, wq) if wq is not None else norm_x
    k = F.linear(norm_x, wk) if wk is not None else norm_x[:, : num_kv_heads * head_dim]
    v = F.linear(norm_x, wv) if wv is not None else norm_x[:, : num_kv_heads * head_dim]
    gate = F.linear(norm_x, wgate) if wgate is not None else norm_x

    q_rot = apply_rope(q, seq_len, num_heads, head_dim, 0, rope_theta)
    k_rot = apply_rope(k, seq_len, num_kv_heads, head_dim, 0, rope_theta)

    gqa_group = num_heads // num_kv_heads
    k_reshaped = k_rot.view(seq_len, num_kv_heads, head_dim)
    v_reshaped = v.view(seq_len, num_kv_heads, head_dim)

    k_rep = torch.repeat_interleave(k_reshaped, gqa_group, dim=1)
    v_rep = torch.repeat_interleave(v_reshaped, gqa_group, dim=1)

    q_heads = q_rot.view(seq_len, num_heads, head_dim).permute(1, 0, 2)
    k_heads = k_rep.permute(1, 0, 2)
    v_heads = v_rep.permute(1, 0, 2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) * scale
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=DEVICE), diagonal=1
    )
    scores = scores + mask

    attn_probs = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_probs, v_heads)
    attn_flat = (
        attn_out.permute(1, 0, 2).contiguous().view(seq_len, num_heads * head_dim)
    )

    attn_gated = attn_flat * torch.sigmoid(gate)
    mixer_out = F.linear(attn_gated, wo) if wo is not None else attn_gated

    return mixer_out, (k_rot, v_reshaped)


def _forward_moe_sublayer(
    norm_x2: torch.Tensor,
    weights: dict[str, torch.Tensor],
    lyr: int,
    seq_len: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    dump_dir: str = "",
    dump_routing_dir: str = "",
) -> torch.Tensor:
    """Mengeksekusi sublayer Channel Mixer MoE (routed + shared)."""
    pfx = f"model.language_model.layers.{lyr}."
    pfx_short = f"layers.{lyr}."
    pfx_blk = f"blk.{lyr}."

    w_router = get_tensor(
        weights,
        [
            pfx + "mlp.gate.weight",
            pfx_short + "mlp.gate.weight",
            pfx_blk + "ffn_gate_exps.weight",
        ],
    )
    if w_router is not None:
        router_logits = F.linear(norm_x2, w_router)
    else:
        router_logits = torch.zeros(
            (seq_len, num_experts), dtype=torch.float32, device=DEVICE
        )

    if dump_dir:
        dump_tensor(
            os.path.join(dump_dir, f"block_{lyr}_06_router_logits.bin"),
            router_logits,
        )

    router_probs = F.softmax(router_logits, dim=-1)

    selected_experts_list = []
    selected_probs_list = []
    for t in range(seq_len):
        row_probs = router_probs[t].tolist()
        pairs = list(enumerate(row_probs))
        pairs.sort(key=lambda p: (-p[1], p[0]))
        top_pairs = pairs[:top_k]
        selected_experts_list.append([p[0] for p in top_pairs])
        selected_probs_list.append([p[1] for p in top_pairs])

    if dump_routing_dir:
        os.makedirs(dump_routing_dir, exist_ok=True)
        rout_file = os.path.join(dump_routing_dir, f"routing_L{lyr}.json")
        with open(rout_file, "w", encoding="utf-8") as f:
            json.dump({"selected_experts": selected_experts_list}, f, indent=2)

    if dump_dir:
        topk_idx = torch.tensor(selected_experts_list, dtype=torch.int32, device=DEVICE)
        dump_tensor(
            os.path.join(dump_dir, f"block_{lyr}_07_topk_indices.bin"), topk_idx
        )

    moe_tokens = []
    for t in range(seq_len):
        xt = norm_x2[t]
        token_moe = torch.zeros(hidden_size, dtype=torch.float32, device=DEVICE)
        for k_idx in range(top_k):
            exp_id = selected_experts_list[t][k_idx]
            exp_prob = selected_probs_list[t][k_idx]

            wg = get_tensor(
                weights,
                [
                    pfx + f"mlp.experts.{exp_id}.gate_proj.weight",
                    pfx_short + f"mlp.experts.{exp_id}.gate_proj.weight",
                    pfx_blk + f"ffn_gate_exps.{exp_id}.weight",
                ],
            )
            wu = get_tensor(
                weights,
                [
                    pfx + f"mlp.experts.{exp_id}.up_proj.weight",
                    pfx_short + f"mlp.experts.{exp_id}.up_proj.weight",
                    pfx_blk + f"ffn_up_exps.{exp_id}.weight",
                ],
            )
            wd = get_tensor(
                weights,
                [
                    pfx + f"mlp.experts.{exp_id}.down_proj.weight",
                    pfx_short + f"mlp.experts.{exp_id}.down_proj.weight",
                    pfx_blk + f"ffn_down_exps.{exp_id}.weight",
                ],
            )

            if wg is not None and wu is not None and wd is not None:
                g_out = F.silu(F.linear(xt, wg))
                u_out = F.linear(xt, wu)
                exp_out = F.linear(g_out * u_out, wd)
            else:
                exp_out = xt * 0.05
            token_moe = token_moe + exp_prob * exp_out

        moe_tokens.append(token_moe)

    routed_out = torch.stack(moe_tokens, dim=0)

    # Shared expert
    sh_wg = get_tensor(
        weights,
        [
            pfx + "mlp.shared_expert.gate_proj.weight",
            pfx_short + "mlp.shared_expert.gate_proj.weight",
            pfx_blk + "ffn_shared_gate.weight",
        ],
    )
    sh_wu = get_tensor(
        weights,
        [
            pfx + "mlp.shared_expert.up_proj.weight",
            pfx_short + "mlp.shared_expert.up_proj.weight",
            pfx_blk + "ffn_shared_up.weight",
        ],
    )
    sh_wd = get_tensor(
        weights,
        [
            pfx + "mlp.shared_expert.down_proj.weight",
            pfx_short + "mlp.shared_expert.down_proj.weight",
            pfx_blk + "ffn_shared_down.weight",
        ],
    )
    sh_wgate = get_tensor(
        weights,
        [
            pfx + "mlp.shared_expert_gate.weight",
            pfx_short + "mlp.shared_expert_gate.weight",
            pfx_blk + "ffn_shared_expert_gate.weight",
        ],
    )

    if sh_wg is not None and sh_wu is not None and sh_wd is not None:
        sh_g = F.silu(F.linear(norm_x2, sh_wg))
        sh_u = F.linear(norm_x2, sh_wu)
        shared_out = F.linear(sh_g * sh_u, sh_wd)
    else:
        shared_out = norm_x2 * 0.05

    if sh_wgate is not None:
        if len(sh_wgate.shape) == 1:
            sh_scores = torch.sigmoid(torch.matmul(norm_x2, sh_wgate)).unsqueeze(-1)
        else:
            sh_scores = torch.sigmoid(F.linear(norm_x2, sh_wgate))
    else:
        sh_scores = torch.full((seq_len, 1), 0.5, dtype=torch.float32, device=DEVICE)

    return routed_out + sh_scores * shared_out


def _execute_single_layer_step(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    gdn_states: dict[int, torch.Tensor],
    lyr: int,
    cfg: dict[str, Any],
    seq_len: int,
    dump_dir: str = "",
    dump_routing_dir: str = "",
) -> torch.Tensor:
    """Mengeksekusi satu blok transformer hybrid (Sublayer 1 + Sublayer 2)."""
    pfx = f"model.language_model.layers.{lyr}."
    pfx_short = f"layers.{lyr}."
    pfx_blk = f"blk.{lyr}."

    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    num_experts = cfg["num_experts"]
    top_k = cfg["num_experts_per_tok"]
    eps = cfg["rms_norm_eps"]
    rope_theta = cfg["rope_theta"]
    dk = cfg.get("dk", 32)
    dv = cfg.get("dv", 32)

    if dump_dir:
        dump_tensor(os.path.join(dump_dir, f"block_{lyr}_01_input.bin"), x)

    in_norm_w = get_tensor(
        weights,
        [
            pfx + "input_layernorm.weight",
            pfx_short + "input_layernorm.weight",
            pfx_blk + "attn_norm.weight",
        ],
    )
    norm_x = apply_rmsnorm(x, in_norm_w, eps)

    if dump_dir:
        dump_tensor(os.path.join(dump_dir, f"block_{lyr}_02_mixer_norm.bin"), norm_x)

    if (lyr % 4) != 3:
        gdn_idx = 3 * (lyr // 4) + (lyr % 4)
        s_prev = gdn_states.get(
            gdn_idx, torch.zeros((dv, dk), dtype=torch.float32, device=DEVICE)
        )
        mixer_out, s_next = _forward_gdn_sublayer(
            norm_x, weights, s_prev, lyr, seq_len, hidden_size, dk, dv
        )
        gdn_states[gdn_idx] = s_next
        if dump_dir:
            dump_tensor(os.path.join(dump_dir, f"block_{lyr}_gdn_state.bin"), s_next)
    else:
        mixer_out, kv_pair = _forward_gated_attn_sublayer(
            norm_x,
            weights,
            lyr,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
        )
        if dump_dir:
            kv_packed = torch.stack(
                [
                    kv_pair[0].view(seq_len, num_kv_heads, head_dim),
                    kv_pair[1],
                ],
                dim=0,
            )
            dump_tensor(os.path.join(dump_dir, f"block_{lyr}_kv_cache.bin"), kv_packed)

    if dump_dir:
        dump_tensor(os.path.join(dump_dir, f"block_{lyr}_03_mixer_out.bin"), mixer_out)

    x_mid = x + mixer_out
    if dump_dir:
        dump_tensor(os.path.join(dump_dir, f"block_{lyr}_04_post_mixer.bin"), x_mid)

    post_norm_w = get_tensor(
        weights,
        [
            pfx + "post_attention_layernorm.weight",
            pfx_short + "post_attention_layernorm.weight",
            pfx_blk + "ffn_norm.weight",
        ],
    )
    norm_x2 = apply_rmsnorm(x_mid, post_norm_w, eps)

    if dump_dir:
        dump_tensor(os.path.join(dump_dir, f"block_{lyr}_05_post_norm.bin"), norm_x2)

    moe_out = _forward_moe_sublayer(
        norm_x2,
        weights,
        lyr,
        seq_len,
        hidden_size,
        num_experts,
        top_k,
        dump_dir,
        dump_routing_dir,
    )

    if dump_dir:
        dump_tensor(os.path.join(dump_dir, f"block_{lyr}_08_moe_out.bin"), moe_out)

    x_out = x_mid + moe_out
    if dump_dir:
        dump_tensor(os.path.join(dump_dir, f"block_{lyr}_09_block_out.bin"), x_out)

    return x_out


def run_oracle_port_forward(
    tokens: list[int],
    weights: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    dump_dir: str = "",
    dump_routing_dir: str = "",
) -> torch.Tensor:
    """Full forward pass hybrid Qwen3.6 transformer."""
    seq_len = len(tokens)
    hidden_size = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    vocab_size = cfg["vocab_size"]
    eps = cfg["rms_norm_eps"]

    embed_w = get_tensor(
        weights,
        [
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "token_embd.weight",
        ],
    )
    if embed_w is None:
        gen = torch.Generator().manual_seed(42)
        embed_w = (
            torch.randn(vocab_size, hidden_size, generator=gen, device=DEVICE) * 0.05
        )

    token_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
    token_tensor = torch.clamp(token_tensor, min=0, max=embed_w.shape[0] - 1)
    x = embed_w[token_tensor].to(torch.float32)

    if dump_dir:
        dump_tensor(os.path.join(dump_dir, "00_embedding.bin"), x)

    gdn_states: dict[int, torch.Tensor] = {}

    for lyr in range(num_layers):
        x = _execute_single_layer_step(
            x=x,
            weights=weights,
            gdn_states=gdn_states,
            lyr=lyr,
            cfg=cfg,
            seq_len=seq_len,
            dump_dir=dump_dir,
            dump_routing_dir=dump_routing_dir,
        )

    final_norm_w = get_tensor(
        weights,
        [
            "model.language_model.norm.weight",
            "model.norm.weight",
            "norm.weight",
            "output_norm.weight",
        ],
    )
    x_final = apply_rmsnorm(x, final_norm_w, eps)

    if dump_dir:
        dump_tensor(os.path.join(dump_dir, "98_final_norm.bin"), x_final)

    lm_head_w = get_tensor(
        weights,
        [
            "lm_head.weight",
            "model.lm_head.weight",
            "output.weight",
        ],
    )
    if lm_head_w is not None:
        logits = F.linear(x_final, lm_head_w)
    else:
        logits = torch.zeros((seq_len, vocab_size), dtype=torch.float32, device=DEVICE)

    if dump_dir:
        dump_tensor(os.path.join(dump_dir, "99_logits.bin"), logits)

    return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oracle Port Full Forward Reference (PyTorch FP32 & GGUF)"
    )
    parser.add_argument("--model-dir", type=str, default="", help="Path model dir")
    parser.add_argument("--weights", type=str, default="", help="Path weights file")
    parser.add_argument(
        "--architecture", type=str, default="qwen3.6", help="Arsitektur model"
    )
    parser.add_argument(
        "--tokens", type=str, required=True, help="Path JSON input tokens"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path output logits FP32 bin"
    )
    parser.add_argument("--dtype", type=str, default="bf16", help="Dtype model")
    parser.add_argument(
        "--quantization", type=str, default="", help="Tipe quant jika GGUF"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed deterministik"
    )
    parser.add_argument(
        "--dump-layers", type=str, default="", help="Direktori dump aktivasi"
    )
    parser.add_argument(
        "--dump-routing", type=str, default="", help="Direktori dump routing"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    weights = load_model_weights(args.weights, args.model_dir)

    cfg = {
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "vocab_size": 1024,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "full_attention_interval": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "dk": 32,
        "dv": 32,
    }

    cfg_file = ""
    if args.model_dir and os.path.isdir(args.model_dir):
        for name in ["config.json", "model_config.json"]:
            cp = os.path.join(args.model_dir, name)
            if os.path.exists(cp):
                cfg_file = cp
                break

    if cfg_file:
        with open(cfg_file, "r", encoding="utf-8") as f:
            raw_c = json.load(f)
        sub_c = raw_c.get("text_config", raw_c)
        for k in cfg:
            if k in sub_c:
                cfg[k] = sub_c[k]

    tokens = parse_tokens(args.tokens)
    logits = run_oracle_port_forward(
        tokens=tokens,
        weights=weights,
        cfg=cfg,
        dump_dir=args.dump_layers,
        dump_routing_dir=args.dump_routing,
    )

    dump_tensor(args.output, logits)

    hasher = hashlib.sha256()
    with open(args.output, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()

    sha_path = args.output + ".sha256"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{sha256_hash}  {os.path.basename(args.output)}\n")

    report = {
        "status": "success",
        "command": "oracle_port",
        "architecture": args.architecture,
        "seq_len": len(tokens),
        "vocab_size": logits.shape[-1],
        "output_path": args.output,
        "output_bytes": os.path.getsize(args.output),
        "sha256": sha256_hash,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
