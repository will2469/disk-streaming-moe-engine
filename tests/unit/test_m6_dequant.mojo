# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit test suite untuk Dequant Kernel SIMD, SEC-4 Parser Hardening, dan G-M6-K (M6-W4)."""

from format.half_float import (
    float16_to_u16,
    pack_4bit_pair,
    safe_multiply_int,
    u16_to_float16,
)
from format.quant_reader import (
    CONFIGURED_MAX_NAME,
    CONFIGURED_MAX_NDIM,
    CONFIGURED_MAX_TENSORS,
    BlockHeader,
    BlockTensorMeta,
    QuantModelIndex,
    QuantTensorEntry,
    scan_quant_file,
    validate_block_header,
    validate_block_tensor_meta,
)
from quant.dequant_kernel import (
    audit_kernel_domain_bound,
    dequant_kernel_simd,
    dequant_kernel_simd_f32,
)
from quant.quant_algo import quantize_tensor_f11a
from std.collections import List
from std.os import SEEK_END, SEEK_SET
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def bfloat16_to_u16(val: BFloat16) -> UInt16:
    var buf = List[UInt8]()
    buf.resize(2, 0)
    buf.unsafe_ptr().unsafe_bitcast[BFloat16]()[unsafe_offset=0] = val
    return buf.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=0]


def test_dequant_simd_vs_scalar_all_225_valid_combinations() raises:
    """Verifikasi 225 kombinasi nibble valid [-7, 7] x [-7, 7] cocok eksak bit-for-bit.
    """
    # 225 pasangan = 225 byte. Kita gunakan 4 grup x 128 elemen = 512 elemen (256 byte)
    var scales = List[Float16]()
    scales.append(Float16(0.25))
    scales.append(Float16(0.5))
    scales.append(Float16(1.0))
    scales.append(Float16(2.0))

    var packed = List[UInt8]()
    var w0_list = List[Int8]()
    var w1_list = List[Int8]()

    for v1 in range(-7, 8):
        for v0 in range(-7, 8):
            packed.append(pack_4bit_pair(Int8(v0), Int8(v1)))
            w0_list.append(Int8(v0))
            w1_list.append(Int8(v1))

    # Pad hingga 256 byte (31 byte nol)
    while len(packed) < 256:
        packed.append(0)
        w0_list.append(0)
        w1_list.append(0)

    var res = dequant_kernel_simd(scales, packed, 512, group_size=128)
    assert_equal(len(res), 512)

    # Verifikasi seluruh 256 byte (512 bobot) cocok eksak
    for b in range(256):
        var g = b // 64
        var s_f32 = Float32(scales[g])
        var expected_0 = BFloat16(s_f32 * Float32(w0_list[b]))
        var expected_1 = BFloat16(s_f32 * Float32(w1_list[b]))
        assert_equal(bfloat16_to_u16(res[b * 2]), bfloat16_to_u16(expected_0))
        assert_equal(
            bfloat16_to_u16(res[b * 2 + 1]), bfloat16_to_u16(expected_1)
        )


def test_dequant_reserved_nibble_rejection() raises:
    """Memverifikasi bahwa nibble reserved 0b1000 (-8) ditolak oleh kernel SIMD.
    """
    var scales = List[Float16]()
    scales.append(Float16(1.0))

    # Low nibble = 8
    var packed_bad_lo = List[UInt8]()
    packed_bad_lo.resize(64, 0)
    packed_bad_lo[10] = 0x08  # lo = 8

    var rejected_lo = False
    try:
        _ = dequant_kernel_simd(scales, packed_bad_lo, 128, 128)
    except:
        rejected_lo = True
    assert_true(
        rejected_lo, "Kernel harus menolak nibble reserved di low nibble"
    )

    # High nibble = 8
    var packed_bad_hi = List[UInt8]()
    packed_bad_hi.resize(64, 0)
    packed_bad_hi[10] = 0x80  # hi = 8

    var rejected_hi = False
    try:
        _ = dequant_kernel_simd(scales, packed_bad_hi, 128, 128)
    except:
        rejected_hi = True
    assert_true(
        rejected_hi, "Kernel harus menolak nibble reserved di high nibble"
    )


def test_dequant_invalid_scales_rejection() raises:
    """Memverifikasi penolakan skala NaN, Inf, dan non-positif."""
    var packed = List[UInt8]()
    packed.resize(64, 0)

    # Skala NaN
    var scales_nan = List[Float16]()
    scales_nan.append(u16_to_float16(0x7E00))  # NaN FP16
    var rej_nan = False
    try:
        _ = dequant_kernel_simd(scales_nan, packed, 128, 128)
    except:
        rej_nan = True
    assert_true(rej_nan, "Kernel harus menolak skala NaN")

    # Skala Inf
    var scales_inf = List[Float16]()
    scales_inf.append(u16_to_float16(0x7C00))  # +Inf FP16
    var rej_inf = False
    try:
        _ = dequant_kernel_simd(scales_inf, packed, 128, 128)
    except:
        rej_inf = True
    assert_true(rej_inf, "Kernel harus menolak skala Inf")

    # Skala <= 0
    var scales_zero = List[Float16]()
    scales_zero.append(Float16(0.0))
    var rej_zero = False
    try:
        _ = dequant_kernel_simd(scales_zero, packed, 128, 128)
    except:
        rej_zero = True
    assert_true(rej_zero, "Kernel harus menolak skala <= 0")


def test_dequant_kernel_domain_property_bound() raises:
    """Memverifikasi bahwa output kernel SIMD memenuhi audit batas Kernel-domain 100%.
    """
    # Bangkitkan tensor bobot sintetis [128, 128] = 16.384 elemen
    var n = 16384
    var orig = List[Float32]()
    orig.resize(n, 0.0)
    for i in range(n):
        orig[i] = Float32((i % 1000) - 500) * Float32(0.005)

    var q_res = quantize_tensor_f11a(orig, group_size=128)
    var bf16_out = dequant_kernel_simd(
        q_res.scales, q_res.packed_bytes, n, group_size=128
    )

    var audit = audit_kernel_domain_bound(
        orig, bf16_out, q_res.scales, group_size=128
    )
    assert_true(audit[0], "Batas Kernel-domain gagal dipenuhi 100% elemen")


def test_sec4_parser_hardening_caps_and_overflow() raises:
    """Memverifikasi penegakan caps dan pencegahan overflow aritmetika SEC-4."""
    # 1. Header num_tensors > 100.000 ditolak
    var hdr_overflow = BlockHeader(
        model="test",
        num_tensors=100001,
        total_bytes=1000,
        group_size=128,
        version=1,
    )
    var rej_tensors = False
    try:
        validate_block_header(hdr_overflow, file_size=1000)
    except:
        rej_tensors = True
    assert_true(rej_tensors, "Header num_tensors > 100000 harus ditolak")

    # 2. Metadata name_len > 512 ditolak
    var long_name = String("")
    for _ in range(513):
        long_name += "a"
    var shape = List[Int]()
    shape.append(128)
    var meta_long_name = BlockTensorMeta(
        name=long_name, shape=shape, dtype="BF16", group_size=128
    )
    var rej_name = False
    try:
        validate_block_tensor_meta(meta_long_name)
    except:
        rej_name = True
    assert_true(rej_name, "Tensor name > 512 byte harus ditolak")

    # 3. Metadata ndim > 8 ditolak
    var shape_9d = List[Int]()
    for _ in range(9):
        shape_9d.append(2)
    var meta_9d = BlockTensorMeta(
        name="test", shape=shape_9d, dtype="BF16", group_size=128
    )
    var rej_ndim = False
    try:
        validate_block_tensor_meta(meta_9d)
    except:
        rej_ndim = True
    assert_true(rej_ndim, "Tensor ndim > 8 harus ditolak")

    # 4. Safe multiply overflow ditolak
    var rej_mult = False
    try:
        _ = safe_multiply_int(0x7FFFFFFFFFFFFFFF, 2)
    except:
        rej_mult = True
    assert_true(rej_mult, "Multiplikasi overflow harus ditolak")


def test_g_m6_k_conformance_seed42() raises:
    """Memverifikasi bit-identical 100% vs golden fixture oracle seed-42 jika ada.
    """
    var fix_dir = "/tmp/test_dequant_fixtures"
    var path_scales = fix_dir + "/fixture_seed42_scales.bin"
    var path_packed = fix_dir + "/fixture_seed42_packed.bin"
    var path_golden = fix_dir + "/fixture_seed42_golden_bf16.bin"
    var path_corrupt = fix_dir + "/fixture_reserved_packed.bin"

    # Jika fixture belum dibangkitkan, lewati langkah ini di unit test lokal
    var f_test: FileHandle
    try:
        f_test = open(path_scales, "r")
        f_test.close()
    except:
        return

    # Baca scales
    var f_s = open(path_scales, "r")
    var sz_s = Int(f_s.seek(0, SEEK_END))
    _ = f_s.seek(0, SEEK_SET)
    var s_bytes = f_s.read_bytes(sz_s)
    f_s.close()

    var num_groups = sz_s // 2
    var scales = List[Float16]()
    scales.resize(num_groups, Float16(0.0))
    for g in range(num_groups):
        var u = UInt16(s_bytes[g * 2]) | (UInt16(s_bytes[g * 2 + 1]) << 8)
        scales[g] = u16_to_float16(u)

    # Baca packed weights
    var f_p = open(path_packed, "r")
    var sz_p = Int(f_p.seek(0, SEEK_END))
    _ = f_p.seek(0, SEEK_SET)
    var p_bytes = f_p.read_bytes(sz_p)
    f_p.close()

    var packed = List[UInt8]()
    packed.resize(sz_p, 0)
    for i in range(sz_p):
        packed[i] = p_bytes[i]

    var total_elements = num_groups * 128
    var res_simd = dequant_kernel_simd(
        scales, packed, total_elements, group_size=128
    )

    # Baca golden BF16 bytes dari oracle
    var f_g = open(path_golden, "r")
    var sz_g = Int(f_g.seek(0, SEEK_END))
    _ = f_g.seek(0, SEEK_SET)
    var g_bytes = f_g.read_bytes(sz_g)
    f_g.close()

    assert_equal(len(res_simd), total_elements)
    assert_equal(sz_g, total_elements * 2)

    # Asersi bit-identical 100% untuk seluruh 131.072 elemen!
    for i in range(total_elements):
        var expected_u16 = UInt16(g_bytes[i * 2]) | (
            UInt16(g_bytes[i * 2 + 1]) << 8
        )
        var actual_u16 = bfloat16_to_u16(res_simd[i])
        if actual_u16 != expected_u16:
            print(
                "Mismatch at element",
                i,
                "actual:",
                hex(Int(actual_u16)),
                "expected:",
                hex(Int(expected_u16)),
            )
            assert_equal(actual_u16, expected_u16)

    # Uji penolakan fixture corrupt dengan reserved nibble 0x8
    var f_c = open(path_corrupt, "r")
    var sz_c = Int(f_c.seek(0, SEEK_END))
    _ = f_c.seek(0, SEEK_SET)
    var c_bytes = f_c.read_bytes(sz_c)
    f_c.close()

    var c_packed = List[UInt8]()
    c_packed.resize(sz_c, 0)
    for i in range(sz_c):
        c_packed[i] = c_bytes[i]

    var rej_corrupt = False
    try:
        _ = dequant_kernel_simd(
            scales, c_packed, total_elements, group_size=128
        )
    except:
        rej_corrupt = True
    assert_true(rej_corrupt, "Kernel harus menolak fixture reserved nibble")


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_dequant_simd_vs_scalar_all_225_valid_combinations]()
    suite.test[test_dequant_reserved_nibble_rejection]()
    suite.test[test_dequant_invalid_scales_rejection]()
    suite.test[test_dequant_kernel_domain_property_bound]()
    suite.test[test_sec4_parser_hardening_caps_and_overflow]()
    suite.test[test_g_m6_k_conformance_seed42]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
