# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi deterministik murni SHA-256 (FIPS 180-4) dalam Mojo 1.0.0."""

from std.collections import List


def _rotr(x: UInt32, n: Int) -> UInt32:
    var u = UInt32(n)
    return (x >> u) | (x << (UInt32(32) - u))


def sha256(data: List[UInt8]) -> List[UInt8]:
    """Menghitung SHA-256 digest dari array byte."""
    var h0: UInt32 = 0x6A09E667
    var h1: UInt32 = 0xBB67AE85
    var h2: UInt32 = 0x3C6EF372
    var h3: UInt32 = 0xA54FF53A
    var h4: UInt32 = 0x510E527F
    var h5: UInt32 = 0x9B05688C
    var h6: UInt32 = 0x1F83D9AB
    var h7: UInt32 = 0x5BE0CD19

    var k_const: List[UInt32] = [
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    ]

    var orig_len = len(data)
    var bit_len: UInt64 = UInt64(orig_len) * 8

    var msg = List[UInt8]()
    msg.reserve(orig_len + 64)
    for i in range(orig_len):
        msg.append(data[i])
    msg.append(0x80)

    while (len(msg) % 64) != 56:
        msg.append(0x00)

    # Append bit_len big-endian 8 bytes
    for i in range(8):
        var shift = (7 - i) * 8
        msg.append(UInt8(Int((bit_len >> UInt64(shift)) & 0xFF)))

    var num_blocks = len(msg) // 64
    for b in range(num_blocks):
        var w = List[UInt32]()
        w.resize(64, UInt32(0))
        var b_off = b * 64
        for i in range(16):
            var off = b_off + i * 4
            var val: UInt32 = (
                (UInt32(msg[off]) << 24)
                | (UInt32(msg[off + 1]) << 16)
                | (UInt32(msg[off + 2]) << 8)
                | UInt32(msg[off + 3])
            )
            w[i] = val

        for i in range(16, 64):
            var s0 = (
                _rotr(w[i - 15], 7)
                ^ _rotr(w[i - 15], 18)
                ^ (w[i - 15] >> UInt32(3))
            )
            var s1 = (
                _rotr(w[i - 2], 17)
                ^ _rotr(w[i - 2], 19)
                ^ (w[i - 2] >> UInt32(10))
            )
            w[i] = w[i - 16] + s0 + w[i - 7] + s1

        var a = h0
        var b_val = h1
        var c = h2
        var d = h3
        var e = h4
        var f_val = h5
        var g = h6
        var h = h7

        for i in range(64):
            var s_one = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            var ch = (e & f_val) ^ ((~e) & g)
            var temp1 = h + s_one + ch + k_const[i] + w[i]
            var s_zero = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
            var maj = (a & b_val) ^ (a & c) ^ (b_val & c)
            var temp2 = s_zero + maj

            h = g
            g = f_val
            f_val = e
            e = d + temp1
            d = c
            c = b_val
            b_val = a
            a = temp1 + temp2

        h0 += a
        h1 += b_val
        h2 += c
        h3 += d
        h4 += e
        h5 += f_val
        h6 += g
        h7 += h

    var digest = List[UInt8]()
    digest.reserve(32)
    var states = [h0, h1, h2, h3, h4, h5, h6, h7]
    for s_idx in range(8):
        var s = states[s_idx]
        for i in range(4):
            var shift = (3 - i) * 8
            digest.append(UInt8(Int((s >> UInt32(shift)) & 0xFF)))

    return digest^
