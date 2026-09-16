# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M4 CLI parser, error schema, dan utilitas sistem."""

from cli.cmd_forward import parse_flat_u32_tokens
from cli.m4_errors import m4_error_code_to_exit_code, m4_error_json
from cli.sys_utils import (
    get_cgroup_oom_kills,
    get_cgroup_peak_bytes,
    get_current_yyyymmdd,
    get_file_size,
)
from std.collections import List
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)


def _to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        out.append(b[i])
    return out^


def test_parse_flat_u32_tokens_valid() raises:
    """Parser membaca array integer datar valid."""
    var raw = _to_bytes("[0, 15, 200, 151935]")
    var tok = parse_flat_u32_tokens(raw, "test.json")
    assert_equal(len(tok), 4)
    assert_equal(tok[0], 0)
    assert_equal(tok[1], 15)
    assert_equal(tok[2], 200)
    assert_equal(tok[3], 151935)


def test_parse_flat_u32_tokens_empty() raises:
    """Parser menerima array kosong [] (validasi count terpisah di caller)."""
    var raw = _to_bytes("   [  ] \n")
    var tok = parse_flat_u32_tokens(raw, "test.json")
    assert_equal(len(tok), 0)


def test_parse_flat_u32_tokens_reject_nested() raises:
    """Nested array [[1, 2]] wajib ditolak."""
    var raw = _to_bytes("[[1, 2]]")
    var raised = False
    try:
        var _out = parse_flat_u32_tokens(raw, "test.json")
    except:
        raised = True
    assert_true(raised)


def test_parse_flat_u32_tokens_reject_negative() raises:
    """Negative integer [-1] wajib ditolak."""
    var raw = _to_bytes("[-1, 2]")
    var raised = False
    try:
        var _out = parse_flat_u32_tokens(raw, "test.json")
    except:
        raised = True
    assert_true(raised)


def test_parse_flat_u32_tokens_reject_float() raises:
    """Float [1.5] wajib ditolak."""
    var raw = _to_bytes("[1.5, 2]")
    var raised = False
    try:
        var _out = parse_flat_u32_tokens(raw, "test.json")
    except:
        raised = True
    assert_true(raised)


def test_parse_flat_u32_tokens_reject_string() raises:
    """String ['123'] wajib ditolak."""
    var raw = _to_bytes('["123", 2]')
    var raised = False
    try:
        var _out = parse_flat_u32_tokens(raw, "test.json")
    except:
        raised = True
    assert_true(raised)


def test_parse_flat_u32_tokens_reject_leading_zero() raises:
    """Leading zero [01] wajib ditolak (invalid JSON integer)."""
    var raw = _to_bytes("[01, 2]")
    var raised = False
    try:
        var _out = parse_flat_u32_tokens(raw, "test.json")
    except:
        raised = True
    assert_true(raised)


def test_parse_flat_u32_tokens_reject_trailing_garbage() raises:
    """Trailing characters setelah ] wajib ditolak."""
    var raw = _to_bytes("[1, 2]GARBAGE")
    var raised = False
    try:
        var _out = parse_flat_u32_tokens(raw, "test.json")
    except:
        raised = True
    assert_true(raised)


def test_m4_error_code_exit_mapping() raises:
    """Exit code mapping 1..6 sesuai tabel spesifikasi M4."""
    assert_equal(m4_error_code_to_exit_code("M4_ERR_INPUT"), 1)
    assert_equal(m4_error_code_to_exit_code("M4_ERR_INDEX"), 2)
    assert_equal(m4_error_code_to_exit_code("M4_ERR_INDEX_VALIDATION"), 2)
    assert_equal(m4_error_code_to_exit_code("M4_ERR_MEMORY"), 3)
    assert_equal(m4_error_code_to_exit_code("M4_ERR_SHARD_IO"), 4)
    assert_equal(m4_error_code_to_exit_code("M4_ERR_LAYER_FORWARD"), 5)
    assert_equal(m4_error_code_to_exit_code("M4_ERR_OUTPUT"), 6)
    assert_equal(m4_error_code_to_exit_code("M4_ERR_COMPARE"), 7)


def test_m4_error_json_format() raises:
    """Fungsi m4_error_json membentuk struktur JSON error yang tepat."""
    var j = m4_error_json(
        "M4_ERR_INPUT", "input", "test msg", '{"token_id":100}'
    )
    assert_true(j.startswith('{"status":"error"'))
    assert_true('"code":"M4_ERR_INPUT"' in j)
    assert_true('"stage":"input"' in j)
    assert_true('"token_id":100' in j)


def test_current_yyyymmdd_length() raises:
    """Tanggal saat ini berformat YYYYMMDD sepanjang 8 karakter."""
    var d = get_current_yyyymmdd()
    assert_equal(d.byte_length(), 8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
