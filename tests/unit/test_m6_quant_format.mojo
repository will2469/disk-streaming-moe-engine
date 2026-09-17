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
    """Verifikasi header tepat 256 byte, padding spasi, dan parse JSON roundtrip.
    """
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
    var json_len = (
        Int(rec[0])
        | (Int(rec[1]) << 8)
        | (Int(rec[2]) << 16)
        | (Int(rec[3]) << 24)
    )
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


def test_4bit_packing_valid_225_byte_combinations() raises:
    """Roundtrip bitwise untuk 225 byte valid; nibble 0x8 wajib ditolak."""
    for b_int in range(256):
        var b = UInt8(b_int)
        var lo = b_int & 0x0F
        var hi = (b_int >> 4) & 0x0F
        if lo == 8 or hi == 8:
            var threw_unpack = False
            try:
                var rejected = unpack_4bit_pair(b)
            except:
                threw_unpack = True
            assert_true(threw_unpack)
        else:
            var pair = unpack_4bit_pair(b)
            var w0 = pair[0]
            var w1 = pair[1]

            # Rentang kontrak [-7, 7]; 0b1000 reserved
            assert_true(w0 >= -7 and w0 <= 7)
            assert_true(w1 >= -7 and w1 <= 7)

            var b_repack = pack_4bit_pair(w0, w1)
            assert_equal(b, b_repack)

    # Uji titik-titik krusial representasi two's complement 4-bit (tanpa -8)
    assert_equal(pack_4bit_pair(0, 0), 0x00)
    assert_equal(pack_4bit_pair(7, 7), 0x77)
    assert_equal(pack_4bit_pair(-1, -1), 0xFF)
    assert_equal(pack_4bit_pair(-7, -7), 0x99)
    assert_equal(pack_4bit_pair(1, -1), 0xF1)
    assert_equal(pack_4bit_pair(-1, 1), 0x1F)

    # Encoder menolak -8 dan nilai di luar [-7, 7]
    var threw_neg8 = False
    try:
        var bad_pack = pack_4bit_pair(Int8(-8), Int8(0))
    except:
        threw_neg8 = True
    assert_true(threw_neg8)

    var threw_pos8 = False
    try:
        var bad_pack2 = pack_4bit_pair(Int8(8), Int8(0))
    except:
        threw_pos8 = True
    assert_true(threw_pos8)


def test_scale_computation_f11a_property() raises:
    """Verifikasi s_g = ceil_FP16(max_abs / 7.0) menjamin Float32(s_g) * 7.0 >= max_abs.
    """
    # Kasus nol
    var s_zero = compute_fp16_scale_ceil(0.0)
    assert_equal(s_zero, Float16(1.0))

    # Kasus tepat 7.0
    var s_seven = compute_fp16_scale_ceil(7.0)
    assert_equal(s_seven, Float16(1.0))

    # Kasus pecahan dan non-kelipatan FP16
    var test_vals: List[Float32] = [
        0.001,
        0.05,
        0.12345,
        0.7,
        1.0,
        3.14159,
        14.0,
        100.0,
        500.0,
    ]
    for i in range(len(test_vals)):
        var max_v = test_vals[i]
        var s = compute_fp16_scale_ceil(max_v)
        assert_true(Float32(s) * 7.0 >= max_v)


def test_hand_calculation_tensor_2048x2048() raises:
    """Verifikasi hitungan tangan untuk tensor [2048, 2048] tepat 2.162.688 B (rasio ~3.88x).
    """
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
    """Verifikasi penolakan terhadap header dan metadata yang rusak/tidak valid.
    """
    # Header salah version
    var bad_hdr = QuantHeader("model", 1, 100, version=2)
    var threw = False
    try:
        validate_quant_header(bad_hdr, 100)
    except:
        threw = True
    assert_true(threw)

    # Header salah group_size (di luar himpunan izin, mis. 100)
    var bad_grp_hdr = QuantHeader("model", 1, 100, group_size=100)
    threw = False
    try:
        validate_quant_header(bad_grp_hdr, 100)
    except:
        threw = True
    assert_true(threw)

    # Header group_size pangkat-2 tapi di luar himpunan izin (512)
    var bad_512_hdr = QuantHeader("model", 1, 100, group_size=512)
    threw = False
    try:
        validate_quant_header(bad_512_hdr, 100)
    except:
        threw = True
    assert_true(threw)

    # Tail group N % G != 0 ditolak saat konstruksi metadata
    var tail_shape: List[Int] = [32]
    var threw_tail = False
    try:
        var tail_meta = QuantTensorMetadata(name="m.tail", shape=tail_shape)
    except:
        threw_tail = True
    assert_true(threw_tail)

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

    # Payload dengan nibble reserved 0x8 ditolak
    var ok_scales: List[Float16] = [Float16(1.0)]
    var reserved_lo: List[UInt8] = [0x08]
    assert_false(validate_quant_payload(ok_scales, reserved_lo, 2))
    var reserved_hi: List[UInt8] = [0x80]
    assert_false(validate_quant_payload(ok_scales, reserved_hi, 2))


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_quant_header_roundtrip_and_padding]()
    suite.test[test_quant_tensor_metadata_serialization]()
    suite.test[test_4bit_packing_valid_225_byte_combinations]()
    suite.test[test_scale_computation_f11a_property]()
    suite.test[test_hand_calculation_tensor_2048x2048]()
    suite.test[test_negative_validation_cases]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
