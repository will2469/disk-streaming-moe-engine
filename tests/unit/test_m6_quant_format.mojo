# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M6-W1: Format Kuantisasi 4-bit, Packing, Skala, dan Validasi."""

from format.quant_format import (
    QUANT_DEFAULT_GROUP_SIZE,
    QUANT_FORMAT_NAME,
    QUANT_HEADER_SIZE,
    QUANT_SCALE_DTYPE,
    QUANTIZED_DTYPE_NAME,
    QuantHeader,
    QuantTensorMetadata,
    calculate_tensor_quant_size,
    compute_fp16_scale_ceil,
    pack_4bit_pair,
    unpack_4bit_pair,
    validate_quant_header,
    validate_quant_payload,
    validate_tensor_meta,
)
from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def test_quant_header_roundtrip_and_padding() raises:
    """Verifikasi header tepat 256 byte, padding spasi, dan parse JSON roundtrip."""
    var hdr = QuantHeader(
        model="qwen1.5-moe-a2.7b-chat",
        num_tensors=4659,
        total_bytes=7934542592,
        group_size=128,
        version=1,
    )

    var bytes = hdr.to_header_bytes()
    assert_equal(len(bytes), QUANT_HEADER_SIZE)

    # Verifikasi padding di ujung berupa spasi (0x20)
    assert_equal(bytes[QUANT_HEADER_SIZE - 1], 32)
    assert_equal(bytes[QUANT_HEADER_SIZE - 2], 32)

    # Parse kembali dari bytes
    var parsed = QuantHeader.from_bytes(bytes)
    assert_equal(parsed.version, 1)
    assert_equal(parsed.model, "qwen1.5-moe-a2.7b-chat")
    assert_equal(parsed.format, QUANT_FORMAT_NAME)
    assert_equal(parsed.group_size, 128)
    assert_equal(parsed.scale_dtype, QUANT_SCALE_DTYPE)
    assert_equal(parsed.num_tensors, 4659)
    assert_equal(parsed.total_bytes, 7934542592)

    # Validasi header lolos
    validate_quant_header(parsed, 7934542592)


def test_quant_tensor_metadata_serialization() raises:
    """Verifikasi metadata tensor, perhitungan grup, dan panjang payload."""
    var shape: List[Int] = [2048, 2048]
    var meta = QuantTensorMetadata(
        name="model.layers.0.self_attn.q_proj.weight",
        shape=shape,
        dtype="BF16",
        group_size=128,
    )

    assert_equal(meta.name, "model.layers.0.self_attn.q_proj.weight")
    assert_equal(meta.num_elements(), 4194304)
    assert_equal(meta.num_groups, 32768)
    assert_equal(meta.scale_offset, 0)
    assert_equal(meta.data_offset, 65536)  # 32768 * 2
    assert_equal(meta.scales_bytes(), 65536)
    assert_equal(meta.weights_bytes(), 2097152)
    assert_equal(meta.payload_bytes(), 2162688)

    # Validasi lolos
    validate_tensor_meta(meta)

    # Record bytes: 4 byte LE length + JSON
    var rec = meta.to_record_bytes()
    assert_true(len(rec) > 4)
    var json_len = Int(rec[0]) | (Int(rec[1]) << 8) | (Int(rec[2]) << 16) | (Int(rec[3]) << 24)
    assert_equal(len(rec), 4 + json_len)

    # Parse metadata dari buffer JSON
    var json_bytes = List[UInt8]()
    for i in range(4, len(rec)):
        json_bytes.append(rec[i])
    var parsed_meta = QuantTensorMetadata.from_json_bytes(json_bytes)
    assert_equal(parsed_meta.name, meta.name)
    assert_equal(len(parsed_meta.shape), 2)
    assert_equal(parsed_meta.shape[0], 2048)
    assert_equal(parsed_meta.shape[1], 2048)
    assert_equal(parsed_meta.num_groups, 32768)
    assert_equal(parsed_meta.scale_offset, 0)
    assert_equal(parsed_meta.data_offset, 65536)


def test_4bit_packing_all_256_byte_combinations() raises:
    """Uji roundtrip lengkap bitwise 4-bit packing/unpacking untuk 256 nilai byte."""
    for b_int in range(256):
        var b = UInt8(b_int)
        var pair = unpack_4bit_pair(b)
        var w0 = pair[0]
        var w1 = pair[1]

        # Rentang harus strictly [-8, 7]
        assert_true(w0 >= -8 and w0 <= 7)
        assert_true(w1 >= -8 and w1 <= 7)

        var b_repack = pack_4bit_pair(w0, w1)
        assert_equal(b, b_repack)

    # Uji titik-titik krusial representasi two's complement 4-bit
    assert_equal(pack_4bit_pair(0, 0), 0x00)
    assert_equal(pack_4bit_pair(7, 7), 0x77)
    assert_equal(pack_4bit_pair(-1, -1), 0xFF)
    assert_equal(pack_4bit_pair(-8, -8), 0x88)
    assert_equal(pack_4bit_pair(1, -1), 0xF1)
    assert_equal(pack_4bit_pair(-1, 1), 0x1F)


def test_scale_computation_f11a_property() raises:
    """Verifikasi s_g = ceil_FP16(max_abs / 7.0) menjamin Float32(s_g) * 7.0 >= max_abs."""
    # Kasus nol
    var s_zero = compute_fp16_scale_ceil(0.0)
    assert_equal(s_zero, Float16(1.0))

    # Kasus tepat 7.0
    var s_seven = compute_fp16_scale_ceil(7.0)
    assert_equal(s_seven, Float16(1.0))

    # Kasus pecahan dan non-kelipatan FP16
    var test_vals: List[Float32] = [0.001, 0.05, 0.12345, 0.7, 1.0, 3.14159, 14.0, 100.0, 500.0]
    for i in range(len(test_vals)):
        var max_v = test_vals[i]
        var s = compute_fp16_scale_ceil(max_v)
        assert_true(Float32(s) * 7.0 >= max_v)


def test_hand_calculation_tensor_2048x2048() raises:
    """Verifikasi hitungan tangan untuk tensor [2048, 2048] tepat 2.162.688 B (rasio ~3.88x)."""
    var shape: List[Int] = [2048, 2048]
    var res = calculate_tensor_quant_size(shape, 128)
    var num_groups = res[0]
    var scales_bytes = res[1]
    var weights_bytes = res[2]
    var total_bytes = res[3]

    assert_equal(num_groups, 32768)
    assert_equal(scales_bytes, 65536)
    assert_equal(weights_bytes, 2097152)
    assert_equal(total_bytes, 2162688)

    var original_bf16 = 4194304 * 2
    assert_equal(original_bf16, 8388608)

    var ratio = Float64(original_bf16) / Float64(total_bytes)
    # Rasio kompresi 8388608 / 2162688 = 3.878787... =~ 3.88x
    assert_true(ratio >= 3.87 and ratio <= 3.89)


def test_negative_validation_cases() raises:
    """Verifikasi penolakan terhadap header dan metadata yang rusak/tidak valid."""
    # Header salah version
    var bad_hdr = QuantHeader("model", 1, 100, version=2)
    var threw = False
    try:
        validate_quant_header(bad_hdr, 100)
    except:
        threw = True
    assert_true(threw)

    # Header salah group_size (bukan pangkat 2)
    var bad_grp_hdr = QuantHeader("model", 1, 100, group_size=100)
    threw = False
    try:
        validate_quant_header(bad_grp_hdr, 100)
    except:
        threw = True
    assert_true(threw)

    # Metadata nama kosong
    var empty_shape: List[Int] = [128]
    var bad_meta = QuantTensorMetadata("", empty_shape)
    threw = False
    try:
        validate_tensor_meta(bad_meta)
    except:
        threw = True
    assert_true(threw)

    # Payload scales tidak boleh NaN/Inf/Zero
    var invalid_scales: List[Float16] = [Float16(1.0), Float16(0.0)]
    var dummy_weights: List[UInt8] = [0x00]
    assert_false(validate_quant_payload(invalid_scales, dummy_weights, 2))


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_quant_header_roundtrip_and_padding]()
    suite.test[test_quant_tensor_metadata_serialization]()
    suite.test[test_4bit_packing_all_256_byte_combinations]()
    suite.test[test_scale_computation_f11a_property]()
    suite.test[test_hand_calculation_tensor_2048x2048]()
    suite.test[test_negative_validation_cases]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
