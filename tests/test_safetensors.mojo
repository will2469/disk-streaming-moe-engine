# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests parser safetensors + F15 (M0-W1). Jalankan: mojo run tests/test_safetensors.mojo."""

from std.testing import assert_equal, assert_raises, TestSuite
from safetensors import read_header


def write_shard(path: String, header: String, payload_len: Int) raises:
    var f = open(path, "w")
    var n = header.byte_length()
    var prefix = List[UInt8]()
    var mult = 1
    for _ in range(8):
        prefix.append(UInt8((n // mult) % 256))
        mult = mult * 256
    f.write_all(Span(prefix))
    f.write_all(header.as_bytes())
    var zeros = List[UInt8]()
    for _ in range(payload_len):
        zeros.append(0)
    f.write_all(Span(zeros))
    f.close()


def test_valid_two_tensors() raises:
    var h = '{"t1":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"t2":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}'
    write_shard("/tmp/kimo_t1.st", h, 16)
    var st = read_header("/tmp/kimo_t1.st")
    assert_equal(len(st.entries), 2)
    assert_equal(st.entries[0].name, "t1")
    assert_equal(st.entries[0].dtype, "BF16")
    assert_equal(st.entries[0].begin, 0)
    assert_equal(st.entries[0].end, 8)
    assert_equal(st.entries[1].name, "t2")
    assert_equal(st.data_base, 8 + h.byte_length())
    assert_equal(st.bytes_header_read, 8 + h.byte_length())


def test_begin_gt_end() raises:
    var h = '{"t":{"dtype":"BF16","shape":[4],"data_offsets":[8,0]}}'
    write_shard("/tmp/kimo_t2.st", h, 16)
    with assert_raises(contains="OFFSET_OVERFLOW"):
        _ = read_header("/tmp/kimo_t2.st")


def test_hole() raises:
    var h = '{"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"b":{"dtype":"BF16","shape":[4],"data_offsets":[12,20]}}'
    write_shard("/tmp/kimo_t3.st", h, 20)
    with assert_raises(contains="OFFSET_OVERFLOW"):
        _ = read_header("/tmp/kimo_t3.st")


def test_unknown_dtype() raises:
    var h = '{"t":{"dtype":"Q8","shape":[4],"data_offsets":[0,8]}}'
    write_shard("/tmp/kimo_t4.st", h, 8)
    with assert_raises(contains="UNKNOWN_DTYPE"):
        _ = read_header("/tmp/kimo_t4.st")


def test_layout_mismatch() raises:
    var h = '{"t":{"dtype":"BF16","shape":[4],"data_offsets":[0,4]}}'
    write_shard("/tmp/kimo_t5.st", h, 8)
    with assert_raises(contains="LAYOUT_MISMATCH"):
        _ = read_header("/tmp/kimo_t5.st")


def test_duplicate_json_key() raises:
    var h = '{"t":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"t":{"dtype":"BF16","shape":[4],"data_offsets":[8,16]}}'
    write_shard("/tmp/kimo_t6.st", h, 16)
    with assert_raises(contains="DUPLICATE_JSON_KEY"):
        _ = read_header("/tmp/kimo_t6.st")


def test_empty_accepted() raises:
    # tensor kosong (BEGIN==END, shape [0]) WAJIB diterima — regresi bug `<` ketat
    var h = '{"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"e":{"dtype":"F32","shape":[0],"data_offsets":[8,8]}}'
    write_shard("/tmp/kimo_t7.st", h, 8)
    var st = read_header("/tmp/kimo_t7.st")
    assert_equal(len(st.entries), 2)
    assert_equal(st.entries[1].begin, 8)
    assert_equal(st.entries[1].end, 8)


def test_header_too_big() raises:
    # klaim 200 MB tanpa body: tolak tanpa alokasi besar
    var f = open("/tmp/kimo_t8.st", "w")
    var prefix = List[UInt8]()
    var n = 200000000
    var mult = 1
    for _ in range(8):
        prefix.append(UInt8((n // mult) % 256))
        mult = mult * 256
    f.write_all(Span(prefix))
    f.close()
    with assert_raises(contains="INVALID_HEADER"):
        _ = read_header("/tmp/kimo_t8.st")


def test_truncated_header() raises:
    var f = open("/tmp/kimo_t9.st", "w")
    var prefix = List[UInt8]()
    var n = 100
    var mult = 1
    for _ in range(8):
        prefix.append(UInt8((n // mult) % 256))
        mult = mult * 256
    f.write_all(Span(prefix))
    var tiny = List[UInt8]()
    tiny.append(123)
    tiny.append(125)
    f.write_all(Span(tiny))
    f.close()
    with assert_raises(contains="INVALID_HEADER"):
        _ = read_header("/tmp/kimo_t9.st")


def test_overflow_beyond_filesize() raises:
    var h = '{"t":{"dtype":"BF16","shape":[4000],"data_offsets":[0,8000]}}'
    write_shard("/tmp/kimo_t10.st", h, 8)
    with assert_raises(contains="OFFSET_OVERFLOW"):
        _ = read_header("/tmp/kimo_t10.st")


def test_file_not_found() raises:
    with assert_raises(contains="FILE_NOT_FOUND"):
        _ = read_header("/tmp/kimo_tidak_ada.st")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
