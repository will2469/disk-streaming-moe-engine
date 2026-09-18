# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests parser safetensors + F15 (M0-W1). Jalankan: mojo run tests/test_safetensors.mojo."""

from std.testing import assert_equal, assert_raises, TestSuite
from safetensors import error_json, json_escape, read_header


def write_shard_bytes(
    path: String, header: List[UInt8], payload_len: Int
) raises:
    var f = open(path, "w")
    var n = len(header)
    var prefix = List[UInt8]()
    var mult = 1
    for _ in range(8):
        prefix.append(UInt8((n // mult) % 256))
        mult = mult * 256
    f.write_all(Span(prefix))
    f.write_all(Span(header))
    var zeros = List[UInt8]()
    for _ in range(payload_len):
        zeros.append(0)
    f.write_all(Span(zeros))
    f.close()


def write_shard(path: String, header: String, payload_len: Int) raises:
    var hb = header.as_bytes()
    var buf = List[UInt8]()
    for i in range(len(hb)):
        buf.append(hb[i])
    write_shard_bytes(path, buf, payload_len)


def test_valid_two_tensors() raises:
    var h = '{"t1":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"t2":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}'
    write_shard("/tmp/dismoen_t1.st", h, 16)
    var st = read_header("/tmp/dismoen_t1.st")
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
    write_shard("/tmp/dismoen_t2.st", h, 16)
    with assert_raises(contains="OFFSET_OVERFLOW"):
        _ = read_header("/tmp/dismoen_t2.st")


def test_hole() raises:
    var h = '{"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"b":{"dtype":"BF16","shape":[4],"data_offsets":[12,20]}}'
    write_shard("/tmp/dismoen_t3.st", h, 20)
    with assert_raises(contains="OFFSET_OVERFLOW"):
        _ = read_header("/tmp/dismoen_t3.st")


def test_unknown_dtype() raises:
    var h = '{"t":{"dtype":"Q8","shape":[4],"data_offsets":[0,8]}}'
    write_shard("/tmp/dismoen_t4.st", h, 8)
    with assert_raises(contains="UNKNOWN_DTYPE"):
        _ = read_header("/tmp/dismoen_t4.st")


def test_layout_mismatch() raises:
    var h = '{"t":{"dtype":"BF16","shape":[4],"data_offsets":[0,4]}}'
    write_shard("/tmp/dismoen_t5.st", h, 8)
    with assert_raises(contains="LAYOUT_MISMATCH"):
        _ = read_header("/tmp/dismoen_t5.st")


def test_duplicate_json_key() raises:
    var h = '{"t":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"t":{"dtype":"BF16","shape":[4],"data_offsets":[8,16]}}'
    write_shard("/tmp/dismoen_t6.st", h, 16)
    with assert_raises(contains="DUPLICATE_JSON_KEY"):
        _ = read_header("/tmp/dismoen_t6.st")


def test_empty_accepted() raises:
    # tensor kosong (BEGIN==END, shape [0]) WAJIB diterima — regresi bug `<` ketat
    var h = '{"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"e":{"dtype":"F32","shape":[0],"data_offsets":[8,8]}}'
    write_shard("/tmp/dismoen_t7.st", h, 8)
    var st = read_header("/tmp/dismoen_t7.st")
    assert_equal(len(st.entries), 2)
    assert_equal(st.entries[1].begin, 8)
    assert_equal(st.entries[1].end, 8)


def test_header_too_big() raises:
    # klaim 200 MB tanpa body: tolak tanpa alokasi besar
    var f = open("/tmp/dismoen_t8.st", "w")
    var prefix = List[UInt8]()
    var n = 200000000
    var mult = 1
    for _ in range(8):
        prefix.append(UInt8((n // mult) % 256))
        mult = mult * 256
    f.write_all(Span(prefix))
    f.close()
    with assert_raises(contains="INVALID_HEADER"):
        _ = read_header("/tmp/dismoen_t8.st")


def test_truncated_header() raises:
    var f = open("/tmp/dismoen_t9.st", "w")
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
        _ = read_header("/tmp/dismoen_t9.st")


def test_overflow_beyond_filesize() raises:
    var h = '{"t":{"dtype":"BF16","shape":[4000],"data_offsets":[0,8000]}}'
    write_shard("/tmp/dismoen_t10.st", h, 8)
    with assert_raises(contains="OFFSET_OVERFLOW"):
        _ = read_header("/tmp/dismoen_t10.st")


def test_file_not_found() raises:
    with assert_raises(contains="FILE_NOT_FOUND"):
        _ = read_header("/tmp/dismoen_tidak_ada.st")


struct LCG(Movable):
    """RNG deterministik untuk property test (pengganti hypothesis: offline, seed tetap).
    """

    var s: Int

    def __init__(out self, seed: Int):
        self.s = seed

    def below(mut self, n: Int) -> Int:
        self.s = (self.s * 1103515245 + 12345) % 2147483648
        return self.s % n


def build_case(mut rng: LCG, corrupt: Int, mut payload: List[Int]) -> String:
    var dtypes = List[String]()
    dtypes.append("BF16")
    dtypes.append("F32")
    dtypes.append("F16")
    dtypes.append("F64")
    var nt = 1 + rng.below(4)
    var js = String("{")
    var off = 0
    for i in range(nt):
        if i > 0:
            js += ","
        var dt = dtypes[rng.below(4)]
        var dim = 1 + rng.below(8)
        var sz = 2
        if dt == "F32":
            sz = 4
        elif dt == "F64":
            sz = 8
        var ln = dim * sz
        var b = off
        var e = off + ln
        if corrupt == 1 and i == nt - 1 and nt > 1:
            b += 1  # lubang 1 byte di tensor terakhir
            e += 1
        elif corrupt == 2 and i == nt - 1 and nt > 1:
            b -= 1  # overlap 1 byte, panjang dijaga (e ikut geser)
            e -= 1
        elif corrupt == 3 and i == 0:
            b = e + 1  # BEGIN > END
        js += String(
            '"t',
            i,
            '":{"dtype":"',
            dt,
            '","shape":[',
            dim,
            '],"data_offsets":[',
            b,
            ",",
            e,
            "]}",
        )
        off = e
    js += "}"
    payload.append(off)
    payload.append(nt)
    return js


def test_missing_field() raises:
    var h = String('{"t":{"dtype":"BF16","shape":[4]}}')
    write_shard("/tmp/dismoen_t11.st", h, 8)
    with assert_raises(contains="INVALID_HEADER"):
        _ = read_header("/tmp/dismoen_t11.st")


def test_not_json() raises:
    write_shard("/tmp/dismoen_t12.st", String("not json!!"), 0)
    with assert_raises(contains="JSON_PARSE_ERROR"):
        _ = read_header("/tmp/dismoen_t12.st")


def test_bad_arity() raises:
    var h = String('{"t":{"dtype":"BF16","shape":[4],"data_offsets":[0]}}')
    write_shard("/tmp/dismoen_t13.st", h, 8)
    with assert_raises(contains="JSON_PARSE_ERROR"):
        _ = read_header("/tmp/dismoen_t13.st")


def test_control_rejected() raises:
    # kontrol mentah 0x01 dalam nama -> tolak (JSON valid melarang < 0x20)
    var js = String('{"t":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}')
    var hb = js.as_bytes()
    var buf = List[UInt8]()
    for i in range(len(hb)):
        buf.append(hb[i])
    buf[2] = 1
    write_shard_bytes("/tmp/dismoen_t14.st", buf, 8)
    with assert_raises(contains="JSON_PARSE_ERROR"):
        _ = read_header("/tmp/dismoen_t14.st")


def test_lone_surrogate() raises:
    var js = String(
        '{"\\ud800":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
    )
    write_shard("/tmp/dismoen_t15.st", js, 2)
    with assert_raises(contains="surrogate"):
        _ = read_header("/tmp/dismoen_t15.st")


def test_surrogate_pair() raises:
    # U+1F600 via pasangan -> nama 4 byte, diterima
    var js = String(
        '{"\\ud83d\\ude00":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
    )
    write_shard("/tmp/dismoen_t16.st", js, 2)
    var st = read_header("/tmp/dismoen_t16.st")
    assert_equal(len(st.entries), 1)
    assert_equal(st.entries[0].name.byte_length(), 4)


def test_cross_nesting() raises:
    # {"a": [1}} silang di field tak dikenal -> tolak (stack penutup)
    var js = String(
        '{"t":{"dtype":"BF16","shape":[1],"data_offsets":[0,2],"x":{"a":[1}}}}'
    )
    write_shard("/tmp/dismoen_t17.st", js, 2)
    with assert_raises(contains="JSON_PARSE_ERROR"):
        _ = read_header("/tmp/dismoen_t17.st")


def test_escape_roundtrip() raises:
    var raw = List[UInt8]()
    raw.append(97)
    raw.append(34)
    raw.append(98)
    raw.append(92)
    raw.append(99)
    var got = json_escape(String(from_utf8_lossy=Span(raw)))
    assert_equal(got, String('a\\"b\\\\c'))


def test_escape_controls() raises:
    # Invariant global: kontrol < 0x20 (newline/tab di nama file Linux)
    # wajib di-escape agar output JSON tetap valid.
    var raw = List[UInt8]()
    raw.append(97)
    raw.append(10)
    raw.append(9)
    raw.append(13)
    raw.append(1)
    raw.append(98)
    var got = json_escape(String(from_utf8_lossy=Span(raw)))
    assert_equal(got, String("a\\n\\t\\r\\u0001b"))


def test_error_json_escapes() raises:
    # Helper kanonik: keempat field dinamis selalu di-escape (nama tensor
    # 'foo"bar' dari header eksternal tak boleh merusak JSON).
    var got = error_json("E", String("a", '"', "b"), String("s", '"', "h"), "t")
    assert_equal(
        got,
        String(
            '{"error_type":"E","detail":"a\\"b","shard":"s\\"h",'
            '"tensor_name":"t"}'
        ),
    )


def test_prop_random_valid() raises:
    var rng = LCG(42)
    for trial in range(20):
        var payload = List[Int]()
        var js = build_case(rng, 0, payload)
        var path = String("/tmp/dismoen_pv", trial, ".st")
        write_shard(path, js, payload[0])
        var st = read_header(path)
        assert_equal(len(st.entries), payload[1])


def test_prop_random_invalid() raises:
    var rng = LCG(1337)
    for trial in range(20):
        var payload = List[Int]()
        var js = build_case(rng, 1 + trial % 3, payload)
        var path = String("/tmp/dismoen_pi", trial, ".st")
        write_shard(path, js, payload[0])
        with assert_raises(contains="OFFSET_OVERFLOW"):
            _ = read_header(path)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
