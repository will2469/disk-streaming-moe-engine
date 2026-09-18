# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit test suite untuk O_DIRECT Reader, Triple Alignment, dan Staging Buffer (M7-W1)."""

from cli.m7_errors import m7_error_json
from format.quant_format import QuantHeader, QuantTensorMetadata, u16_to_float16
from io.odirect import ODirectReader, ReadToken

from quant.dequant_kernel import dequant_kernel_simd
from std.collections import List
from std.os import SEEK_SET
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def get_target_model_path() -> String:
    """Mengembalikan path model kuantisasi nyata jika ada, atau fallback fixture.
    """
    return "fixtures/m10_quant_mini.bin"


def test_dio_discovery_probe() raises:
    """Menguji discovery dio_alignment [512, 4096] dan probe read nyata."""
    var path = get_target_model_path()
    var reader = ODirectReader.discover(
        path, requested_block_size=4096, queue_depth=16
    )

    # dio_alignment harus 512 atau 4096
    assert_true(
        reader.dio_alignment == 512 or reader.dio_alignment == 4096,
        "dio_alignment must be 512 or 4096",
    )
    assert_equal(reader.block_size, 4096)
    assert_equal(reader.queue_depth, 16)
    assert_equal(reader.outstanding_count, 0)
    reader.close()


def test_triple_alignment_and_block_size_validation() raises:
    """Menguji penolakan block-size yang melanggar relasi dio_alignment."""
    var path = get_target_model_path()
    var caught = False
    try:
        # block-size 100 tidak memenuhi relasi kelipatan dio_alignment (512 / 4096)
        var reader = ODirectReader.discover(
            path, requested_block_size=100, queue_depth=16
        )
        reader.close()
    except e:
        caught = True
        var err_s = String(e)
        assert_true(
            err_s.find("M7_ERR_ODIRECT_ALIGNMENT") >= 0,
            "must fail with M7_ERR_ODIRECT_ALIGNMENT",
        )
    assert_true(caught, "must reject invalid block-size")


def test_span_based_unaligned_m6_slice() raises:
    """Menguji pembacaan slice logis unaligned dari physical span selaras via O_DIRECT vs Buffered.
    """
    var path = get_target_model_path()
    var reader = ODirectReader.discover(
        path, requested_block_size=4096, queue_depth=16
    )

    # Offset 256 adalah akhir header M6 v1. Kita baca metadata tensor pertama (unaligned offset 256..600)
    var test_offset = 256
    var test_length = 256

    var odirect_bytes = reader.read_logical_payload(test_offset, test_length)
    assert_equal(len(odirect_bytes), test_length)

    # Baca data yang sama via buffered FileHandle standar untuk membuktikan kesetaraan bit-identical
    var f = open(path, "r")
    _ = f.seek(test_offset, SEEK_SET)
    var buffered_bytes = f.read_bytes(test_length)
    f.close()

    for i in range(test_length):
        assert_equal(
            odirect_bytes[i],
            buffered_bytes[i],
            "O_DIRECT byte must match buffered byte at offset "
            + String(test_offset + i),
        )

    reader.close()


def test_queue_depth_tracking() raises:
    """Menguji tracking outstanding_count dan max_outstanding_observed API dua-fase.
    """
    var path = get_target_model_path()
    var reader = ODirectReader.discover(
        path, requested_block_size=4096, queue_depth=4
    )

    # Submit 3 read requests secara konkuren
    var tok1 = reader.submit_read(256, 128)
    var tok2 = reader.submit_read(384, 128)
    var tok3 = reader.submit_read(512, 128)

    assert_equal(reader.outstanding_count, 3)
    assert_equal(reader.max_outstanding_observed, 3)

    # Complete read satu per satu
    var b1 = reader.complete_read(tok1^)
    assert_equal(reader.outstanding_count, 2)
    assert_equal(len(b1), 128)

    var b2 = reader.complete_read(tok2^)
    assert_equal(reader.outstanding_count, 1)
    assert_equal(len(b2), 128)

    var b3 = reader.complete_read(tok3^)
    assert_equal(reader.outstanding_count, 0)
    assert_equal(len(b3), 128)

    # Submit melebihi queue_depth=4 harus ditolak
    var tok_a = reader.submit_read(0, 64)
    var tok_b = reader.submit_read(64, 64)
    var tok_c = reader.submit_read(128, 64)
    var tok_d = reader.submit_read(192, 64)
    assert_equal(reader.outstanding_count, 4)
    assert_equal(reader.max_outstanding_observed, 4)

    var qd_exceeded = False
    try:
        var tok_overflow = reader.submit_read(256, 64)
    except:
        qd_exceeded = True
    assert_true(qd_exceeded, "must reject request when queue_depth is exceeded")

    _ = reader.complete_read(tok_a^)
    _ = reader.complete_read(tok_b^)
    _ = reader.complete_read(tok_c^)
    _ = reader.complete_read(tok_d^)
    assert_equal(reader.outstanding_count, 0)

    reader.close()


def test_buffered_fallback_flag() raises:
    """Menguji mode force_buffered fallback tetap membaca byte yang identik."""
    var path = get_target_model_path()
    var reader_buf = ODirectReader.discover(
        path, requested_block_size=4096, queue_depth=4, force_buffered=True
    )
    assert_false(
        reader_buf.is_odirect, "is_odirect must be False in force_buffered"
    )

    var data = reader_buf.read_logical_payload(256, 64)
    assert_equal(len(data), 64)

    var f = open(path, "r")
    _ = f.seek(256, SEEK_SET)
    var exp_data = f.read_bytes(64)
    f.close()

    for i in range(64):
        assert_equal(data[i], exp_data[i])
    reader_buf.close()


def test_streaming_dequant_single_layer_step() raises:
    """Menguji integrasi end-to-end: baca payload tensor via O_DIRECT -> dequant SIMD.
    """
    var path = get_target_model_path()
    var reader = ODirectReader.discover(
        path, requested_block_size=4096, queue_depth=8
    )

    # Baca 4 byte meta_len di offset 256
    var meta_len_raw = reader.read_logical_payload(256, 4)
    var meta_len = (
        Int(meta_len_raw[0])
        | (Int(meta_len_raw[1]) << 8)
        | (Int(meta_len_raw[2]) << 16)
        | (Int(meta_len_raw[3]) << 24)
    )
    assert_true(meta_len > 0 and meta_len < 65536)

    # Baca JSON metadata tensor 0
    var meta_json = reader.read_logical_payload(260, meta_len)
    var meta = QuantTensorMetadata.from_json_bytes(meta_json)

    var s_bytes = meta.scales_bytes()
    var w_bytes = meta.weights_bytes()
    var payload_offset = 260 + meta_len

    # Baca scales FP16 dan weights 4-bit via O_DIRECT
    var scales_raw = reader.read_logical_payload(payload_offset, s_bytes)
    var weights_raw = reader.read_logical_payload(
        payload_offset + s_bytes, w_bytes
    )

    assert_equal(len(scales_raw), s_bytes)
    assert_equal(len(weights_raw), w_bytes)

    # Parse scales FP16
    var scales = List[Float16]()
    for g in range(meta.num_groups):
        var u = UInt16(scales_raw[g * 2]) | (UInt16(scales_raw[g * 2 + 1]) << 8)
        scales.append(u16_to_float16(u))

    # Eksekusi dequant SIMD (M6)
    var out_bf16 = dequant_kernel_simd(
        scales, weights_raw, meta.num_elements(), meta.group_size
    )
    assert_equal(len(out_bf16), meta.num_elements())

    # Verifikasi bukan data kosong/sampah
    var non_zero = False
    for i in range(min(1024, len(out_bf16))):
        if out_bf16[i] != BFloat16(0.0):
            non_zero = True
            break
    assert_true(
        non_zero, "dequantized layer weights must contain non-zero values"
    )

    reader.close()


def test_m7_error_schema() raises:
    """Menguji pembentukan skema JSON RFC 8259 untuk kode error M7."""
    var err_str = m7_error_json(
        "M7_ERR_ODIRECT_EIO", "io_direct", "disk failure simulated"
    )
    assert_true(err_str.find('"status":"error"') >= 0)
    assert_true(err_str.find('"code":"M7_ERR_ODIRECT_EIO"') >= 0)
    assert_true(err_str.find('"stage":"io_direct"') >= 0)


def main() raises:
    var suite = TestSuite()
    suite.test[test_dio_discovery_probe]()
    suite.test[test_triple_alignment_and_block_size_validation]()
    suite.test[test_span_based_unaligned_m6_slice]()
    suite.test[test_queue_depth_tracking]()
    suite.test[test_buffered_fallback_flag]()
    suite.test[test_streaming_dequant_single_layer_step]()
    suite.test[test_m7_error_schema]()
    suite^.run()
