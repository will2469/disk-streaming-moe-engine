# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Algoritma Kuantisasi F11a & Evaluasi Properti (M6-W2).

Komponen C5 & C2 dari disk-streaming-moe-engine:
- Pembulatan eksplit RNE (round-half-to-even) dalam fp32.
- Penegakan tail group (N % G == 0).
- Skala FP16 ceil (s_g = ceil_FP16(max|w| / 7.0)).
- Kuantisasi 4-bit bertanda [-7, 7] (0x8 reserved).
- Dekuantisasi Q-domain (FP32) dan Kernel-domain (BF16 pembulatan tunggal).
- Verifikasi properti dua-domain:
  * Q-domain: |w - w_hat^(32)| <= s_g / 2
  * Kernel-domain: |w - w_hat^(bf16)| <= s_g / 2 + |w_hat^(32)| / 256
- Perhitungan epsilon_rel & jalur absolut variansi nol.
"""

from format.quant_format import (
    QUANT_DEFAULT_GROUP_SIZE,
    compute_fp16_scale_ceil,
    is_allowed_group_size,
    pack_4bit_pair,
    unpack_4bit_pair,
)
from std.collections import List
from std.math import abs, floor, isinf, isnan, sqrt


def rne_fp32(x: Float32) raises -> Int:
    """Pembulatan eksplisit round-half-to-even (RNE) untuk nilai skalar FP32.

    Pecahan tepat 0.5 selalu dibulatkan ke integer genap terdekat.
    Dilarang mengandalkan tie-break bawaan round bahasa/hardware.
    """
    if isnan(x) or isinf(x):
        raise Error("rne_fp32: NaN or Inf encountered")
    var k_f = floor(x)
    var k = Int(k_f)
    var diff = x - k_f
    if diff < 0.5:
        return k
    elif diff > 0.5:
        return k + 1
    else:
        # diff == 0.5 tepat: tie-break ke integer genap
        if (k & 1) == 0:
            return k
        else:
            return k + 1


def clip_q4(m: Int) -> Int8:
    """Membatasi nilai integer ke rentang kuantisasi 4-bit [-7, 7].

    Nilai -8 (nibble 0x8) dilarang dipancarkan oleh encoder.
    """
    if m > 7:
        return Int8(7)
    elif m < -7:
        return Int8(-7)
    return Int8(m)


def quantize_group_f11a(
    weights: List[Float32], start_idx: Int, count: Int
) raises -> Tuple[Float16, List[Int8]]:
    """Menguantisasi satu grup bobot berukuran count sesuai formula F11a.

    Mengembalikan Tuple (scale_fp16, q_weights) di mana q_weights bertanda [-7, 7].
    """
    var max_abs = Float32(0.0)
    for i in range(count):
        var w = weights[start_idx + i]
        if isnan(w) or isinf(w):
            raise Error("quantize_group_f11a: NaN or Inf in weight tensor")
        var aw = abs(w)
        if aw > max_abs:
            max_abs = aw

    var s_g = compute_fp16_scale_ceil(max_abs)
    var s_f32 = Float32(s_g)

    var q_list = List[Int8]()
    q_list.resize(count, Int8(0))

    if max_abs == Float32(0.0):
        # Grup nol: s_g = 1.0, q = 0
        return (s_g, q_list^)

    for i in range(count):
        var w = weights[start_idx + i]
        var x = w / s_f32
        var m = rne_fp32(x)
        var q = clip_q4(m)
        q_list[i] = q

    return (s_g, q_list^)


struct QuantizedTensor(Copyable, Movable):
    """Representasi tensor yang telah terkuantisasi secara in-memory."""

    var scales: List[Float16]
    var q_weights: List[Int8]
    var packed_bytes: List[UInt8]
    var num_elements: Int
    var group_size: Int
    var num_groups: Int

    def __init__(
        out self,
        scales: List[Float16],
        q_weights: List[Int8],
        packed_bytes: List[UInt8],
        num_elements: Int,
        group_size: Int,
        num_groups: Int,
    ):
        self.scales = scales.copy()
        self.q_weights = q_weights.copy()
        self.packed_bytes = packed_bytes.copy()
        self.num_elements = num_elements
        self.group_size = group_size
        self.num_groups = num_groups


def quantize_tensor_f11a(
    weights: List[Float32], group_size: Int = QUANT_DEFAULT_GROUP_SIZE
) raises -> QuantizedTensor:
    """Menguantisasi tensor penuh Float32 ke representasi 4-bit per-group F11a.

    Penegakan kontrak tail Opsi A: N % G == 0 eksak.
    """
    if not is_allowed_group_size(group_size):
        raise Error(
            "quantize_tensor_f11a: group_size not in {32, 64, 128, 256}"
        )

    var n_elem = len(weights)
    if n_elem == 0:
        raise Error("quantize_tensor_f11a: weights list cannot be empty")
    if n_elem % group_size != 0:
        raise Error(
            "quantize_tensor_f11a: N % G != 0 (tail group not supported)"
        )

    var num_groups = n_elem // group_size
    var scales = List[Float16]()
    scales.resize(num_groups, Float16(1.0))

    var q_weights = List[Int8]()
    q_weights.resize(n_elem, Int8(0))

    for g in range(num_groups):
        var start_idx = g * group_size
        var res = quantize_group_f11a(weights, start_idx, group_size)
        scales[g] = res[0]
        for j in range(group_size):
            q_weights[start_idx + j] = res[1][j]

    # Mengemas bobot 2-per-byte LE
    var packed_bytes = List[UInt8]()
    var num_packed = (n_elem + 1) // 2
    packed_bytes.resize(num_packed, 0)

    var byte_idx = 0
    var i = 0
    while i < n_elem:
        var w0 = q_weights[i]
        var w1 = Int8(0)
        if i + 1 < n_elem:
            w1 = q_weights[i + 1]
        packed_bytes[byte_idx] = pack_4bit_pair(w0, w1)
        byte_idx += 1
        i += 2

    return QuantizedTensor(
        scales=scales^,
        q_weights=q_weights^,
        packed_bytes=packed_bytes^,
        num_elements=n_elem,
        group_size=group_size,
        num_groups=num_groups,
    )


def dequantize_tensor_q32(
    scales: List[Float16],
    q_weights: List[Int8],
    group_size: Int = QUANT_DEFAULT_GROUP_SIZE,
) raises -> List[Float32]:
    """Dekuantisasi ke domain FP32 (Q-domain): w_hat^(32) = fp32(s_g) * q."""
    var n_elem = len(q_weights)
    var num_groups = len(scales)
    if n_elem != num_groups * group_size:
        raise Error("dequantize_tensor_q32: dimensions mismatch")

    var out = List[Float32]()
    out.resize(n_elem, Float32(0.0))

    for g in range(num_groups):
        var s_f32 = Float32(scales[g])
        var base = g * group_size
        for j in range(group_size):
            out[base + j] = s_f32 * Float32(q_weights[base + j])

    return out^


def dequantize_tensor_bf16(
    scales: List[Float16],
    q_weights: List[Int8],
    group_size: Int = QUANT_DEFAULT_GROUP_SIZE,
) raises -> List[BFloat16]:
    """Dekuantisasi ke Kernel-domain: w_hat^(bf16) = bf16(w_hat^(32))."""
    var q32 = dequantize_tensor_q32(scales, q_weights, group_size)
    var out = List[BFloat16]()
    out.resize(len(q32), BFloat16(0.0))
    for i in range(len(q32)):
        out[i] = BFloat16(q32[i])
    return out^


def verify_qdomain_property(
    weights: List[Float32],
    q32_weights: List[Float32],
    scales: List[Float16],
    group_size: Int = QUANT_DEFAULT_GROUP_SIZE,
) raises -> Tuple[Bool, Float32]:
    """Verifikasi properti Q-domain (FP32): |fp32(w) - w_hat^(32)| <= s_g / 2.

    Mengembalikan Tuple (all_passed, max_observed_error).
    """
    var n_elem = len(weights)
    if len(q32_weights) != n_elem:
        raise Error("verify_qdomain_property: length mismatch")

    var max_err = Float32(0.0)
    for i in range(n_elem):
        var g = i // group_size
        var s_g = scales[g]
        var bound = Float32(s_g) / Float32(2.0)
        var err = abs(weights[i] - q32_weights[i])
        if err > max_err:
            max_err = err
        # Toleransi numerik 1 ULP fp32
        if err > bound + Float32(1e-6):
            return (False, max_err)

    return (True, max_err)


def verify_kerneldomain_property(
    weights: List[Float32],
    bf16_weights: List[BFloat16],
    q32_weights: List[Float32],
    scales: List[Float16],
    group_size: Int = QUANT_DEFAULT_GROUP_SIZE,
) raises -> Tuple[Bool, Float32]:
    """Verifikasi properti Kernel-domain: |w - w_hat^(bf16)| <= s_g / 2 + |w_hat^(32)| / 256.

    Mengembalikan Tuple (all_passed, max_observed_error).
    """
    var n_elem = len(weights)
    if len(bf16_weights) != n_elem or len(q32_weights) != n_elem:
        raise Error("verify_kerneldomain_property: length mismatch")

    var max_err = Float32(0.0)
    for i in range(n_elem):
        var g = i // group_size
        var s_g = scales[g]
        var w32 = q32_weights[i]
        var bound = (Float32(s_g) / Float32(2.0)) + (abs(w32) / Float32(256.0))
        var err = abs(weights[i] - Float32(bf16_weights[i]))
        if err > max_err:
            max_err = err
        if err > bound + Float32(1e-6):
            return (False, max_err)

    return (True, max_err)


struct QuantMetrics(Copyable, Movable):
    """Metrik statistik kesalahan kuantisasi."""

    var mse: Float32
    var variance: Float32
    var zero_variance: Bool
    var epsilon_rel: Float32

    def __init__(
        out self,
        mse: Float32,
        variance: Float32,
        zero_variance: Bool,
        epsilon_rel: Float32,
    ):
        self.mse = mse
        self.variance = variance
        self.zero_variance = zero_variance
        self.epsilon_rel = epsilon_rel


def compute_quant_metrics(
    weights: List[Float32], dequant_f32: List[Float32]
) raises -> QuantMetrics:
    """Menghitung MSE, variansi, dan epsilon_rel = sqrt(MSE) / sqrt(var(w)).

    Bila var(w) == 0 (tensor konstan / nol):
    laporkan epsilon_rel = -1.0 (null) dan zero_variance = True.
    """
    var n = len(weights)
    if n == 0 or len(dequant_f32) != n:
        raise Error("compute_quant_metrics: length mismatch or empty")

    var sum_w = Float64(0.0)
    var sum_sq_diff = Float64(0.0)

    for i in range(n):
        sum_w += Float64(weights[i])
        var d = Float64(weights[i]) - Float64(dequant_f32[i])
        sum_sq_diff += d * d

    var mean_w = sum_w / Float64(n)
    var sum_var = Float64(0.0)
    for i in range(n):
        var vd = Float64(weights[i]) - mean_w
        sum_var += vd * vd

    var mse = Float32(sum_sq_diff / Float64(n))
    var variance = Float32(sum_var / Float64(n))

    if variance <= Float32(1e-12):
        return QuantMetrics(
            mse=mse,
            variance=Float32(0.0),
            zero_variance=True,
            epsilon_rel=Float32(-1.0),
        )

    var eps_rel = Float32(sqrt(Float64(mse)) / sqrt(Float64(variance)))
    return QuantMetrics(
        mse=mse,
        variance=variance,
        zero_variance=False,
        epsilon_rel=eps_rel,
    )
