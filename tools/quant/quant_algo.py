# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Algoritma Kuantisasi F11a & Evaluasi Properti (M6-W2).

Mengimplementasikan:
1. Pembulatan eksplisit RNE (Round-Half-to-Even) tie-breaking fp32.
2. Penegakan tail group (N % G == 0).
3. Perhitungan skala FP16 ceil (s_g = ceil_FP16(max|w| / 7.0)).
4. Kuantisasi 4-bit bertanda [-7, 7] (0x8 reserved).
5. Dekuantisasi Q-domain (FP32) dan Kernel-domain (BF16 pembulatan tunggal).
6. Verifikasi properti dua-domain:
   - Q-domain: |w - w_hat^(32)| <= s_g / 2
   - Kernel-domain: |w - w_hat^(bf16)| <= s_g / 2 + |w_hat^(32)| / 256
7. Perhitungan epsilon_rel & penanganan tensor variansi nol via jalur absolut.
"""

import math
from typing import Any

import numpy as np
import torch

from tools.quant.quant_format import (
    QUANT_ALLOWED_GROUP_SIZES,
    QUANT_DEFAULT_GROUP_SIZE,
    compute_fp16_scale_ceil,
    pack_4bit_pair,
)


def rne_fp32(x: float) -> int:
    """Pembulatan eksplisit round-half-to-even (RNE) untuk nilai skalar FP32.

    Pecahan tepat 0.5 selalu dibulatkan ke integer genap terdekat.
    """
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"rne_fp32: NaN or Inf encountered: {x}")
    k = math.floor(x)
    diff = x - k
    if diff < 0.5:
        return int(k)
    elif diff > 0.5:
        return int(k + 1)
    else:
        # diff == 0.5 tepat: tie-break ke integer genap
        return int(k if (int(k) & 1 == 0) else k + 1)


def clip_q4(m: int) -> int:
    """Membatasi nilai integer ke rentang kuantisasi 4-bit [-7, 7].

    Nilai -8 (nibble 0x8) tidak pernah dipancarkan (reserved).
    """
    if m > 7:
        return 7
    elif m < -7:
        return -7
    return m


def f32_to_bf16_f32(val: float) -> float:
    """Konversi pembulatan tunggal Float32 -> BFloat16 -> Float32."""
    t = torch.tensor(val, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
    return float(t.item())


def quantize_group_f11a(weights: list[float]) -> tuple[float, list[int]]:
    """Menguantisasi satu grup bobot berukuran G sesuai formula F11a.

    Mengembalikan (scale_fp16, q_weights) di mana q_weights ∈ [-7, 7].
    """
    max_abs = 0.0
    for w in weights:
        if math.isnan(w) or math.isinf(w):
            raise ValueError(f"quantize_group_f11a: NaN or Inf in weight: {w}")
        aw = abs(w)
        if aw > max_abs:
            max_abs = aw

    s_g = compute_fp16_scale_ceil(max_abs)
    s_f32 = float(np.float32(s_g))

    q_list: list[int] = []
    if max_abs == 0.0:
        # Grup nol: s_g = 1.0, q = 0
        q_list = [0] * len(weights)
    else:
        for w in weights:
            # Komputasi dalam domain FP32
            x = float(np.float32(w) / np.float32(s_f32))
            m = rne_fp32(x)
            q = clip_q4(m)
            q_list.append(q)

    return s_g, q_list


def quantize_tensor_f11a(
    weights: list[float], group_size: int = QUANT_DEFAULT_GROUP_SIZE
) -> tuple[list[float], list[int], bytes]:
    """Menguantisasi satu tensor penuh ke representasi 4-bit F11a.

    Penegakan kontrak tail Opsi A: N % G == 0.
    Mengembalikan (scales, q_weights, packed_bytes).
    """
    if group_size not in QUANT_ALLOWED_GROUP_SIZES:
        allowed = sorted(QUANT_ALLOWED_GROUP_SIZES)
        raise ValueError(f"group_size must be in {allowed}, got {group_size}")

    n_elem = len(weights)
    if n_elem == 0:
        raise ValueError("quantize_tensor_f11a: tensor cannot be empty")
    if n_elem % group_size != 0:
        raise ValueError(
            f"N % G != 0 (tail group not supported): N={n_elem}, G={group_size}"
        )

    num_groups = n_elem // group_size
    scales: list[float] = []
    q_weights: list[int] = []
    packed_raw = bytearray()

    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = start_idx + group_size
        g_weights = weights[start_idx:end_idx]
        s_g, q_g = quantize_group_f11a(g_weights)
        scales.append(s_g)
        q_weights.extend(q_g)

    # Pack weights 2-per-byte LE
    for i in range(0, len(q_weights), 2):
        w0 = q_weights[i]
        w1 = q_weights[i + 1] if i + 1 < len(q_weights) else 0
        packed_raw.append(pack_4bit_pair(w0, w1))

    return scales, q_weights, bytes(packed_raw)


def dequantize_tensor_q32(
    scales: list[float],
    q_weights: list[int],
    group_size: int = QUANT_DEFAULT_GROUP_SIZE,
) -> list[float]:
    """Dekuantisasi ke Q-domain (FP32): w_hat^(32) = fp32(s_g) * q."""
    n_elem = len(q_weights)
    out: list[float] = [0.0] * n_elem
    num_groups = len(scales)
    if n_elem != num_groups * group_size:
        raise ValueError("dequantize_tensor_q32: dimension mismatch")

    for g in range(num_groups):
        s_f32 = float(np.float32(scales[g]))
        base = g * group_size
        for j in range(group_size):
            out[base + j] = float(np.float32(s_f32 * float(q_weights[base + j])))

    return out


def dequantize_tensor_bf16(
    scales: list[float],
    q_weights: list[int],
    group_size: int = QUANT_DEFAULT_GROUP_SIZE,
) -> list[float]:
    """Dekuantisasi ke Kernel-domain (BF16): w_hat^(bf16) = bf16(w_hat^(32))."""
    q32 = dequantize_tensor_q32(scales, q_weights, group_size)
    t = torch.tensor(q32, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
    return t.tolist()


def verify_qdomain_property(
    weights: list[float],
    q32_weights: list[float],
    scales: list[float],
    group_size: int = QUANT_DEFAULT_GROUP_SIZE,
) -> tuple[bool, float]:
    """Verifikasi properti Q-domain: |fp32(w) - w_hat^(32)| <= s_g / 2.

    Mengembalikan (all_passed, max_error_observed).
    """
    n_elem = len(weights)
    if len(q32_weights) != n_elem:
        raise ValueError("verify_qdomain_property: length mismatch")

    max_err = 0.0
    for i in range(n_elem):
        g = i // group_size
        s_g = scales[g]
        bound = float(np.float32(s_g) / np.float32(2.0))
        err = abs(float(np.float32(weights[i])) - float(np.float32(q32_weights[i])))
        if err > max_err:
            max_err = err
        # Mengizinkan toleransi 1 ULP fp32 (1e-6)
        if err > bound + 1e-6:
            return False, max_err

    return True, max_err


def verify_kerneldomain_property(
    weights: list[float],
    bf16_weights: list[float],
    q32_weights: list[float],
    scales: list[float],
    group_size: int = QUANT_DEFAULT_GROUP_SIZE,
) -> tuple[bool, float]:
    """Verifikasi properti Kernel-domain:
    |w - w_hat^(bf16)| <= s_g / 2 + |w_hat^(32)| / 256.

    Mengembalikan (all_passed, max_error_observed).
    """
    n_elem = len(weights)
    if len(bf16_weights) != n_elem or len(q32_weights) != n_elem:
        raise ValueError("verify_kerneldomain_property: length mismatch")

    max_err = 0.0
    for i in range(n_elem):
        g = i // group_size
        s_g = scales[g]
        w32 = q32_weights[i]
        bound = float(np.float32(s_g) / np.float32(2.0)) + abs(w32) / 256.0
        err = abs(float(np.float32(weights[i])) - float(np.float32(bf16_weights[i])))
        if err > max_err:
            max_err = err
        if err > bound + 1e-6:
            return False, max_err

    return True, max_err


def compute_quant_metrics(
    weights: list[float], dequant_weights: list[float]
) -> dict[str, Any]:
    """Menghitung MSE, variansi, dan epsilon_rel = sqrt(MSE) / sqrt(var(w)).

    Bila var(w) == 0 (tensor konstan / nol):
    laporkan epsilon_rel: None, zero_variance: True.
    """
    n = len(weights)
    if n == 0 or len(dequant_weights) != n:
        raise ValueError("compute_quant_metrics: invalid inputs")

    w_arr = np.array(weights, dtype=np.float32)
    dq_arr = np.array(dequant_weights, dtype=np.float32)

    diff = w_arr - dq_arr
    mse = float(np.mean(diff**2))
    var_w = float(np.var(w_arr))

    if var_w <= 1e-12:
        return {
            "mse": mse,
            "variance": 0.0,
            "zero_variance": True,
            "epsilon_rel": None,
        }

    eps_rel = float(math.sqrt(mse) / math.sqrt(var_w))
    return {
        "mse": mse,
        "variance": var_w,
        "zero_variance": False,
        "epsilon_rel": eps_rel,
    }
