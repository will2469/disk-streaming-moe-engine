#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
r"""Oracle Naive Loop Reference untuk Gated DeltaNet (F14) — M8-W1.

Mengimplementasikan loop rekuren serial token-by-token murni dalam FP32:
    S_t = \gamma_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
dengan S_0 = 0, layout kanonis [layers, dv, dk] (row = dv, col = dk).
Menyimpan state akhir dalam format biner framed kanonis GDNS v1.
"""

import argparse
import hashlib
import json
import os
import struct
import sys

import torch
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oracle naive recurrence loop for Gated DeltaNet (F14)"
    )
    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path ke file JSON dengan input tokens",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Jumlah layer GDN (default: 2)",
    )
    parser.add_argument(
        "--dk",
        type=int,
        default=32,
        help="Dimensi key dk (default: 32)",
    )
    parser.add_argument(
        "--dv",
        type=int,
        default=32,
        help="Dimensi value dv (default: 32)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path ke file safetensors weights (opsional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path output binary state final GDNS v1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed jika weights dibuat on-the-fly (default: 42)",
    )
    return parser.parse_args()


def load_tokens(tokens_path: str) -> list[int]:
    """Membaca dan memvalidasi tokens dari file JSON."""
    if not os.path.exists(tokens_path):
        sys.stderr.write(f"ERROR: file tokens tidak ditemukan: {tokens_path}\n")
        sys.exit(1)
    try:
        with open(tokens_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"ERROR: gagal membaca JSON tokens: {e}\n")
        sys.exit(1)

    if isinstance(data, list):
        tokens = data
    elif isinstance(data, dict) and "tokens" in data:
        tokens = data["tokens"]
    else:
        sys.stderr.write("ERROR: format tokens JSON tidak valid\n")
        sys.exit(1)

    if not tokens:
        sys.stderr.write("ERROR: tokens kosong\n")
        sys.exit(1)

    return [int(tok) for tok in tokens]


def load_or_synthesize_weights(
    weights_path: str,
    layers: int,
    dk: int,
    dv: int,
    vocab: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Memuat bobot dari safetensors atau membangkitkan secara deterministik."""
    if weights_path and os.path.exists(weights_path):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k).to(torch.float32)
        return tensors

    # Bangkitkan bobot on-the-fly bila weights tidak diberikan
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    scale = 1.0 / (dk**0.5)
    tensors["embed_tokens.weight"] = (
        torch.randn((vocab, dk), generator=gen, dtype=torch.float32) * scale
    )
    for lyr in range(layers):
        tensors[f"layers.{lyr}.k_proj.weight"] = (
            torch.randn((dk, dk), generator=gen, dtype=torch.float32) * scale
        )
        tensors[f"layers.{lyr}.v_proj.weight"] = (
            torch.randn((dv, dk), generator=gen, dtype=torch.float32) * scale
        )
        tensors[f"layers.{lyr}.beta_proj.weight"] = (
            torch.randn((1, dk), generator=gen, dtype=torch.float32) * scale
        )
    return tensors


def write_gdns_v1(
    path: str,
    state_tensor: torch.Tensor,
    layers: int,
    dv: int,
    dk: int,
    manifest_hash: bytes | None = None,
) -> None:
    """Menuliskan state tensor ke format kanonis GDNS v1 framed binary."""
    if manifest_hash is None:
        manifest_hash = b"\x00" * 32

    state_bytes = layers * dv * dk * 4
    header = bytearray(128)

    # 1. Magic bytes: 'G', 'D', 'N', 'S' (0x47, 0x44, 0x4E, 0x53)
    header[0:4] = b"GDNS"
    # 2. Version = 1 (uint32 LE)
    struct.pack_into("<I", header, 4, 1)
    # 3. Architecture ID = 1 (ARCH_QWEN_GDN)
    struct.pack_into("<I", header, 8, 1)
    # 4. dtype = 1 (FP32)
    struct.pack_into("<I", header, 12, 1)
    # 5. Dims: layers, dv, dk
    struct.pack_into("<I", header, 16, layers)
    struct.pack_into("<I", header, 20, dv)
    struct.pack_into("<I", header, 24, dk)
    # 6. Reserved1 (uint32 LE) = 0
    struct.pack_into("<I", header, 28, 0)
    # 7. Total state_bytes (uint64 LE)
    struct.pack_into("<Q", header, 32, state_bytes)
    # 8. Model manifest hash (32 bytes)
    header[40:72] = manifest_hash[:32]
    # 9. Reserved2 (56 bytes, zeroed)

    # Payload state tensor contiguous row-major FP32
    flat_np = state_tensor.detach().cpu().to(torch.float32).numpy()
    payload = flat_np.tobytes()

    if len(payload) != state_bytes:
        sys.stderr.write(
            f"ERROR: ukuran payload ({len(payload)}) != state_bytes ({state_bytes})\n"
        )
        sys.exit(6)

    # Trailing SHA-256 checksum: SHA-256(header[128] || payload[state_bytes])
    hasher = hashlib.sha256()
    hasher.update(header)
    hasher.update(payload)
    digest = hasher.digest()

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    temp_path = path + ".tmp"
    try:
        with open(temp_path, "wb") as f:
            f.write(header)
            f.write(payload)
            f.write(digest)
        os.replace(temp_path, path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        sys.stderr.write(f"ERROR: gagal menulis state atomic: {e}\n")
        sys.exit(6)


def run_oracle_naive(
    tokens: list[int],
    layers: int,
    dk: int,
    dv: int,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Menjalankan loop rekuren naive FP32 untuk seluruh layer GDN."""
    seq_len = len(tokens)
    vocab = 512
    if "embed_tokens.weight" in weights:
        vocab = weights["embed_tokens.weight"].shape[0]

    # Ambil embedding tokens: [seq_len, embed_dim]
    embed_table = weights["embed_tokens.weight"]
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    # Clamp token IDs jika melebihi vocab untuk robustness
    token_tensor = torch.clamp(token_tensor, min=0, max=vocab - 1)
    x = embed_table[token_tensor]  # [seq_len, embed_dim]

    all_layer_states = []
    eye_dk = torch.eye(dk, dtype=torch.float32)

    for lyr in range(layers):
        # State initial S_0 = 0, shape [dv, dk] (kanonis F14: row=dv, col=dk)
        s_state = torch.zeros((dv, dk), dtype=torch.float32)

        # Proyeksi bobot layer
        w_k_key = f"layers.{lyr}.k_proj.weight"
        w_v_key = f"layers.{lyr}.v_proj.weight"
        w_beta_key = f"layers.{lyr}.beta_proj.weight"

        w_k = weights.get(w_k_key)
        w_v = weights.get(w_v_key)
        w_beta = weights.get(w_beta_key)

        for t in range(seq_len):
            x_t = x[t]  # [embed_dim]

            # 1. Compute kt [dk]
            if w_k is not None:
                kt = torch.matmul(w_k, x_t)
            else:
                kt = x_t[:dk] if x_t.shape[0] >= dk else x_t

            # Normalize kt (L2-norm) untuk stabilitas DeltaNet
            norm_k = torch.norm(kt) + 1e-6
            kt = kt / norm_k

            # 2. Compute vt [dv]
            if w_v is not None:
                vt = torch.matmul(w_v, x_t)
            else:
                vt = x_t[:dv] if x_t.shape[0] >= dv else x_t

            # 3. Compute decay gamma_t dan update rate beta_t
            # M8 baseline: gamma_t = 1.0 (un-gated)
            gamma_t = 1.0

            if w_beta is not None:
                beta_raw = torch.matmul(w_beta, x_t)
                beta_t = float(torch.sigmoid(beta_raw).item())
            else:
                beta_t = 0.5

            # 4. F14: Delta rule
            # outer(kt, kt): [dk, dk], I: [dk, dk]
            # S @ (I - beta_t * outer(kt, kt)): [dv, dk] @ [dk, dk] -> [dv, dk]
            # outer(vt, kt): [dv, dk]
            kt_outer = torch.outer(kt, kt)
            transition_mat = eye_dk - beta_t * kt_outer
            bias_mat = beta_t * torch.outer(vt, kt)

            s_state = gamma_t * torch.matmul(s_state, transition_mat) + bias_mat

            # Cek NaN/INF
            if torch.isnan(s_state).any() or torch.isinf(s_state).any():
                sys.stderr.write(
                    f"ERROR: NaN/INF terdeteksi pada layer {lyr}, token {t}\n"
                )
                sys.exit(5)

        all_layer_states.append(s_state)

    # Tensor gabungan: shape [layers, dv, dk]
    final_states = torch.stack(all_layer_states, dim=0)
    return final_states


def main():
    args = parse_args()

    if args.layers <= 0 or args.dk <= 0 or args.dv <= 0:
        sys.stderr.write("ERROR: layers, dk, dan dv harus bernilai positif\n")
        sys.exit(2)

    tokens = load_tokens(args.tokens)
    weights = load_or_synthesize_weights(
        weights_path=args.weights,
        layers=args.layers,
        dk=args.dk,
        dv=args.dv,
        vocab=512,
        seed=args.seed,
    )

    state_tensor = run_oracle_naive(
        tokens=tokens,
        layers=args.layers,
        dk=args.dk,
        dv=args.dv,
        weights=weights,
    )

    write_gdns_v1(
        path=args.output,
        state_tensor=state_tensor,
        layers=args.layers,
        dv=args.dv,
        dk=args.dk,
    )

    total_bytes = os.path.getsize(args.output)
    print(
        f"Oracle state generated successfully: {args.output} "
        f"(shape: [{args.layers}, {args.dv}, {args.dk}], size: {total_bytes} B)"
    )


if __name__ == "__main__":
    main()
