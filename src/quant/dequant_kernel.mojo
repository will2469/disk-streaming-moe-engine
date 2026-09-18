# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Kernel Dequantisasi 4-bit ke BF16 & F32 berakselerasi SIMD (M6-W4).

Spesifikasi (docs/milestones/M6-quantizer.md § Dequant Kernel Specification):
- Input: scales FP16 [num_groups], packed 4-bit weights [num_elements / 2]
- Grup: G in {32, 64, 128, 256}, N % G == 0 (tanpa grup parsial)
- Unpack nibble bertanda [-7, 7] Little-Endian (w0: low nibble, w1: high nibble)
- Penolakan reserved nibble 0b1000 (-8) dan skala liar (NaN, Inf, <= 0)
- Konversi pembulatan tunggal ke BFloat16: bf16(fp32(s_g) * q_j)
- Vektor SIMD 8-lane (AVX2/NEON friendly), cache-friendly, overhead <= 10% decode.
"""

from core.worker_pool import WorkerPool
from format.half_float import float16_to_u16, is_allowed_group_size
from std.builtin.dtype import DType
from std.collections import List
from std.math import isinf, isnan


def dequant_kernel_simd(
    scales: List[Float16],
    packed_weights: List[UInt8],
    num_elements: Int,
    group_size: Int = 128,
) raises -> List[BFloat16]:
    """Mengonversi bobot 4-bit ter-pack + skala FP16 ke output BFloat16.

    Menegakkan penolakan reserved nibble 0b1000 (-8) dan skala non-finite.
    Hasil bit-identical 100% dengan evaluasi tunggal bf16(fp32(s_g) * q).
    """
    if not is_allowed_group_size(group_size):
        raise Error(
            "dequant_kernel: group_size must be in {32, 64, 128, 256}, got "
            + String(group_size)
        )

    if num_elements <= 0:
        raise Error("dequant_kernel: num_elements must be positive")

    if num_elements % group_size != 0:
        raise Error(
            "M6_ERR_INPUT: N % G != 0 (tail group not supported): N="
            + String(num_elements)
            + ", G="
            + String(group_size)
        )

    var num_groups = num_elements // group_size
    if len(scales) != num_groups:
        raise Error(
            "dequant_kernel: scales length mismatch: expected "
            + String(num_groups)
            + ", got "
            + String(len(scales))
        )

    var exp_packed = (num_elements + 1) // 2
    if len(packed_weights) != exp_packed:
        raise Error(
            "dequant_kernel: packed weights length mismatch: expected "
            + String(exp_packed)
            + ", got "
            + String(len(packed_weights))
        )

    # Validasi skala (harus finite dan > 0)
    for g in range(num_groups):
        var s = scales[g]
        var u = float16_to_u16(s)
        var exp_bits = (u >> 10) & 0x1F
        if exp_bits == 0x1F:
            raise Error(
                "dequant_kernel: scale is NaN or Inf at group " + String(g)
            )
        if s <= Float16(0.0):
            raise Error(
                "dequant_kernel: scale is non-positive at group " + String(g)
            )

    var out = List[BFloat16]()
    out.resize(num_elements, BFloat16(0.0))
    var p_out = out.unsafe_ptr()
    var p_packed = packed_weights.unsafe_ptr()

    for g in range(num_groups):
        var s_g = scales[g]
        var s_val = Float32(s_g)
        var s_simd = SIMD[DType.float32, 8](s_val)

        var g_elem_base = g * group_size
        var bytes_in_group = group_size // 2
        var g_byte_base = g * bytes_in_group

        var b = 0
        while b + 8 <= bytes_in_group:
            var raw = p_packed.unsafe_load[width=8](g_byte_base + b)
            # Low nibble: bits 0-3 digeser ke bits 4-7 lalu sign-extended
            var lo_s = (raw << 4).cast[DType.int8]() >> 4
            # High nibble: bits 4-7 langsung sign-extended
            var hi_s = raw.cast[DType.int8]() >> 4

            # Penolakan reserved nibble 0b1000 (-8)
            if lo_s.reduce_min() < -7 or hi_s.reduce_min() < -7:
                raise Error(
                    "dequant_kernel: reserved nibble 0b1000 (-8) encountered"
                )

            var lo_out = (s_simd * lo_s.cast[DType.float32]()).cast[
                DType.bfloat16
            ]()
            var hi_out = (s_simd * hi_s.cast[DType.float32]()).cast[
                DType.bfloat16
            ]()

            var out_base = g_elem_base + b * 2
            for j in range(8):
                p_out[unsafe_offset=out_base + 2 * j] = lo_out[j]
                p_out[unsafe_offset=out_base + 2 * j + 1] = hi_out[j]

            b += 8

    return out^


def dequant_kernel_simd_f32(
    scales: List[Float16],
    packed_weights: List[UInt8],
    num_elements: Int,
    group_size: Int = 128,
) raises -> List[Float32]:
    """Mengonversi bobot 4-bit ter-pack + skala FP16 langsung ke List[Float32].

    Menggunakan pembulatan tunggal BFloat16 internal, lalu diekspansi ke Float32
    (pelebaran eksak tanpa loss) untuk kompatibilitas matmul activation forward.
    """
    var bf16_list = dequant_kernel_simd(
        scales, packed_weights, num_elements, group_size
    )
    var out_f32 = List[Float32]()
    out_f32.resize(num_elements, Float32(0.0))
    var p_bf16 = bf16_list.unsafe_ptr()
    var p_f32 = out_f32.unsafe_ptr()

    var i = 0
    while i + 16 <= num_elements:
        var bf_v = p_bf16.unsafe_load[width=16](i)
        p_f32.unsafe_store[width=16](i, bf_v.cast[DType.float32]())
        i += 16
    while i < num_elements:
        p_f32[unsafe_offset=i] = Float32(p_bf16[unsafe_offset=i])
        i += 1

    return out_f32^


def dequant_kernel_simd_parallel(
    scales: List[Float16],
    packed_weights: List[UInt8],
    num_elements: Int,
    group_size: Int,
    mut pool: WorkerPool,
) raises -> List[BFloat16]:
    """Mengonversi bobot 4-bit ke BFloat16 secara paralel pada WorkerPool.

    Menjamin 100% bit-exact (Delta_max == 0) terhadap dequant_kernel_simd.
    """
    if pool.num_threads <= 1:
        return dequant_kernel_simd(
            scales, packed_weights, num_elements, group_size
        )

    if not is_allowed_group_size(group_size):
        raise Error(
            "dequant_kernel: group_size must be in {32, 64, 128, 256}, got "
            + String(group_size)
        )

    if num_elements <= 0:
        raise Error("dequant_kernel: num_elements must be positive")

    if num_elements % group_size != 0:
        raise Error(
            "M6_ERR_INPUT: N % G != 0 (tail group not supported): N="
            + String(num_elements)
            + ", G="
            + String(group_size)
        )

    var num_groups = num_elements // group_size
    if len(scales) != num_groups:
        raise Error(
            "dequant_kernel: scales length mismatch: expected "
            + String(num_groups)
            + ", got "
            + String(len(scales))
        )

    var exp_packed = (num_elements + 1) // 2
    if len(packed_weights) != exp_packed:
        raise Error(
            "dequant_kernel: packed weights length mismatch: expected "
            + String(exp_packed)
            + ", got "
            + String(len(packed_weights))
        )

    # Validasi skala (harus finite dan > 0)
    for g in range(num_groups):
        var s = scales[g]
        var u = float16_to_u16(s)
        var exp_bits = (u >> 10) & 0x1F
        if exp_bits == 0x1F:
            raise Error(
                "dequant_kernel: scale is NaN or Inf at group " + String(g)
            )
        if s <= Float16(0.0):
            raise Error(
                "dequant_kernel: scale is non-positive at group " + String(g)
            )

    var out = List[BFloat16]()
    out.resize(num_elements, BFloat16(0.0))

    pool.parallel_dequant_bf16(
        Int(scales.unsafe_ptr()),
        Int(packed_weights.unsafe_ptr()),
        Int(out.unsafe_ptr()),
        num_elements,
        group_size,
    )

    return out^


def dequant_kernel_simd_f32_parallel(
    scales: List[Float16],
    packed_weights: List[UInt8],
    num_elements: Int,
    group_size: Int,
    mut pool: WorkerPool,
) raises -> List[Float32]:
    """Mengonversi bobot 4-bit ke Float32 secara paralel pada WorkerPool."""
    var bf16_list = dequant_kernel_simd_parallel(
        scales, packed_weights, num_elements, group_size, pool
    )
    var out_f32 = List[Float32]()
    out_f32.resize(num_elements, Float32(0.0))
    var p_bf16 = bf16_list.unsafe_ptr()
    var p_f32 = out_f32.unsafe_ptr()

    var i = 0
    while i + 16 <= num_elements:
        var bf_v = p_bf16.unsafe_load[width=16](i)
        p_f32.unsafe_store[width=16](i, bf_v.cast[DType.float32]())
        i += 16
    while i < num_elements:
        p_f32[unsafe_offset=i] = Float32(p_bf16[unsafe_offset=i])
        i += 1

    return out_f32^


def audit_kernel_domain_bound(
    orig_w: List[Float32],
    bf16_w: List[BFloat16],
    scales: List[Float16],
    group_size: Int = 128,
) raises -> Tuple[Bool, Float32]:
    """Memvalidasi batas audit Kernel-domain:
    |fp32(w) - fp32(w_hat^(bf16))| <= s_g / 2 + |w_hat^(32)| / 256.

    Mengembalikan Tuple(lulus_100_persen, max_error).
    """
    var n = len(orig_w)
    if len(bf16_w) != n:
        raise Error("audit_kernel_domain_bound: length mismatch")

    var max_err: Float32 = 0.0
    for i in range(n):
        var g = i // group_size
        var s_g = Float32(scales[g])
        var w_orig = orig_w[i]
        var w_bf = Float32(bf16_w[i])

        var err = abs(w_orig - w_bf)
        if err > max_err:
            max_err = err

        # Batas Kernel-domain: s_g/2 + |w_hat^(32)| / 256
        # Karena |w_hat^(32)| <= 7 * s_g, batas maksimum <= 0.5274 * s_g
        var bound = (s_g / 2.0) + (abs(w_bf) / 256.0) + Float32(1e-5)
        if err > bound:
            return (False, max_err)

    return (True, max_err)
