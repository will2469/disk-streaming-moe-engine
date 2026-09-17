# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit Test Suite untuk M8-W2: Hukum Komposisi Operator GDN, Remainder, dan Representasi WY.

Menguji:
1. Koefisien representasi WY (backward substitution solve)
2. Invarian komposisi operator: (M_{A U B}, B_{A U B}) == (M_A M_B, B_A M_B + B_B)
3. Ekuivalensi update state chunked vs komposisi operator
4. Penanganan native partial remainder (seq_len % chunk != 0) tanpa fallback
5. Kontrak layout kanonis asimetris (dk != dv, mis. dk=32, dv=48)
6. Deteksi dan mitigasi fail-fast NaN/INF
7. State continuation melintasi batas pemotongan arbitrer (boundary stress)
8. Konsistensi numerik sweep ukuran chunk (C in {4, 8, 16})
"""

from format.gdns import read_gdns_v1, write_gdns_v1
from layers.gdn import (
    ChunkOperator,
    GDNConfig,
    GDNState,
    apply_wy_chunk_update,
    chunked_gdn_scan,
    compose_chunk_operators,
    compute_wy_coefficients,
    extract_chunk_operator,
)
from std.collections import List
from std.math import abs, max, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def pseudo_random_vector(size: Int, seed: Int) -> List[Float32]:
    """Menghasilkan vektor deterministik menggunakan LCG sederhana."""
    var out = List[Float32]()
    out.reserve(size)
    var state: UInt64 = UInt64(seed) * 6364136223846793005 + 1442695040888963407
    for _ in range(size):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int((state >> 33) & 0x7FFFFFFF)) / Float32(2147483648.0)
        # Distribusi [-1.0, 1.0]
        out.append(u * Float32(2.0) - Float32(1.0))
    return out^


def normalize_k_matrix(mut k_mat: List[Float32], m: Int, dk: Int):
    """Menormalkan setiap baris key matrix k_mat dengan L2-norm."""
    var p_k = k_mat.unsafe_ptr()
    for i in range(m):
        var row_off = i * dk
        var sum_sq = Float32(0.0)
        for c in range(dk):
            var val = p_k[unsafe_offset=row_off + c]
            sum_sq += val * val
        var norm = sqrt(sum_sq) + Float32(1e-6)
        var inv_norm = Float32(1.0) / norm
        for c in range(dk):
            p_k[unsafe_offset=row_off + c] *= inv_norm


def test_wy_coefficients_backward_solve() raises:
    """Verifikasi bahwa backward substitution memenuhi relasi deterministik."""
    var m = 6
    var dk = 8
    var k_mat = pseudo_random_vector(m * dk, 101)
    normalize_k_matrix(k_mat, m, dk)

    var beta = List[Float32]()
    for i in range(m):
        beta.append(Float32(0.3 + 0.1 * Float32(i)))

    var w_mat = compute_wy_coefficients(k_mat, beta, m, dk)

    var p_k = k_mat.unsafe_ptr()
    var p_w = w_mat.unsafe_ptr()

    for i in range(m):
        var b_i = beta[i]
        var row_i_k = i * dk
        var row_i_w = i * dk

        for c in range(dk):
            var expected_rhs = b_i * p_k[unsafe_offset=row_i_k + c]
            var actual = p_w[unsafe_offset=row_i_w + c]

            # Tambahkan b_i * sum_{j > i} (K_i . K_j) * W_j[c]
            var sum_term = Float32(0.0)
            for j in range(i + 1, m):
                var row_j_k = j * dk
                var row_j_w = j * dk
                var dot_ij = Float32(0.0)
                for k_idx in range(dk):
                    dot_ij += (
                        p_k[unsafe_offset=row_i_k + k_idx]
                        * p_k[unsafe_offset=row_j_k + k_idx]
                    )
                sum_term += dot_ij * p_w[unsafe_offset=row_j_w + c]

            actual += b_i * sum_term
            assert_almost_equal(actual, expected_rhs, atol=1e-5)


def test_operator_composition_invariant() raises:
    """Menguji invarian aljabar hukum komposisi operator chunk berdampingan."""
    var m_a = 5
    var m_b = 7
    var m_total = m_a + m_b
    var dk = 16
    var dv = 20

    var k_total = pseudo_random_vector(m_total * dk, 202)
    normalize_k_matrix(k_total, m_total, dk)
    var v_total = pseudo_random_vector(m_total * dv, 303)

    var beta_total = List[Float32]()
    for i in range(m_total):
        beta_total.append(Float32(0.2 + 0.05 * Float32(i % 10)))

    # Pisah segmen A dan B
    var k_a = List[Float32]()
    k_a.resize(m_a * dk, Float32(0.0))
    for i in range(m_a * dk):
        k_a[i] = k_total[i]

    var v_a = List[Float32]()
    v_a.resize(m_a * dv, Float32(0.0))
    for i in range(m_a * dv):
        v_a[i] = v_total[i]

    var beta_a = List[Float32]()
    for i in range(m_a):
        beta_a.append(beta_total[i])

    var k_b = List[Float32]()
    k_b.resize(m_b * dk, Float32(0.0))
    var b_k_off = m_a * dk
    for i in range(m_b * dk):
        k_b[i] = k_total[b_k_off + i]

    var v_b = List[Float32]()
    v_b.resize(m_b * dv, Float32(0.0))
    var b_v_off = m_a * dv
    for i in range(m_b * dv):
        v_b[i] = v_total[b_v_off + i]

    var beta_b = List[Float32]()
    for i in range(m_b):
        beta_b.append(beta_total[m_a + i])

    # 1. Ekstrak operator A
    var w_a = compute_wy_coefficients(k_a, beta_a, m_a, dk)
    var ops_a = extract_chunk_operator(k_a, v_a, w_a, m_a, dk, dv)

    # 2. Ekstrak operator B
    var w_b = compute_wy_coefficients(k_b, beta_b, m_b, dk)
    var ops_b = extract_chunk_operator(k_b, v_b, w_b, m_b, dk, dv)

    # 3. Komposisikan A dan B via hukum monoid
    var ops_comp = compose_chunk_operators(
        ops_a.m_mat, ops_a.b_mat, ops_b.m_mat, ops_b.b_mat, dk, dv
    )

    # 4. Ekstrak operator total langsung dari A U B
    var w_total = compute_wy_coefficients(k_total, beta_total, m_total, dk)
    var ops_total = extract_chunk_operator(
        k_total, v_total, w_total, m_total, dk, dv
    )

    # 5. Assert selisih maksimum <= 1e-5
    var max_diff_m = Float32(0.0)
    for i in range(dk * dk):
        var diff = abs(ops_total.m_mat[i] - ops_comp.m_mat[i])
        if diff > max_diff_m:
            max_diff_m = diff
    assert_true(
        max_diff_m <= Float32(1e-5),
        "Operator composition invariant for M violated",
    )

    var max_diff_b = Float32(0.0)
    for i in range(dv * dk):
        var diff = abs(ops_total.b_mat[i] - ops_comp.b_mat[i])
        if diff > max_diff_b:
            max_diff_b = diff
    assert_true(
        max_diff_b <= Float32(1e-5),
        "Operator composition invariant for B violated",
    )


def test_chunked_state_update_equivalence() raises:
    """Menguji ekuivalensi numerik antara pembaruan state gabungan vs berantai.
    """
    var m_a = 4
    var m_b = 6
    var m_total = m_a + m_b
    var dk = 16
    var dv = 16

    var k_total = pseudo_random_vector(m_total * dk, 404)
    normalize_k_matrix(k_total, m_total, dk)
    var v_total = pseudo_random_vector(m_total * dv, 505)

    var beta_total = List[Float32]()
    for _ in range(m_total):
        beta_total.append(Float32(0.5))

    var s_init = pseudo_random_vector(dv * dk, 606)

    # Jalur 1: Gabungan langsung dalam 1 chunk
    var s_single = s_init.copy()
    var w_total = compute_wy_coefficients(k_total, beta_total, m_total, dk)
    apply_wy_chunk_update(s_single, k_total, v_total, w_total, m_total, dk, dv)

    # Jalur 2: Dieksekusi berurutan chunk A lalu chunk B
    var s_chain = s_init.copy()

    var k_a = List[Float32]()
    k_a.resize(m_a * dk, Float32(0.0))
    for i in range(m_a * dk):
        k_a[i] = k_total[i]
    var v_a = List[Float32]()
    v_a.resize(m_a * dv, Float32(0.0))
    for i in range(m_a * dv):
        v_a[i] = v_total[i]
    var beta_a = List[Float32]()
    for i in range(m_a):
        beta_a.append(beta_total[i])

    var w_a = compute_wy_coefficients(k_a, beta_a, m_a, dk)
    apply_wy_chunk_update(s_chain, k_a, v_a, w_a, m_a, dk, dv)

    var k_b = List[Float32]()
    k_b.resize(m_b * dk, Float32(0.0))
    var b_k_off = m_a * dk
    for i in range(m_b * dk):
        k_b[i] = k_total[b_k_off + i]
    var v_b = List[Float32]()
    v_b.resize(m_b * dv, Float32(0.0))
    var b_v_off = m_a * dv
    for i in range(m_b * dv):
        v_b[i] = v_total[b_v_off + i]
    var beta_b = List[Float32]()
    for i in range(m_b):
        beta_b.append(beta_total[m_a + i])

    var w_b = compute_wy_coefficients(k_b, beta_b, m_b, dk)
    apply_wy_chunk_update(s_chain, k_b, v_b, w_b, m_b, dk, dv)

    # Bandingkan s_single vs s_chain
    var max_diff = Float32(0.0)
    for i in range(dv * dk):
        var diff = abs(s_single[i] - s_chain[i])
        if diff > max_diff:
            max_diff = diff

    assert_true(
        max_diff <= Float32(1e-5),
        "Chunked state update equivalence failed",
    )


def test_native_partial_remainder() raises:
    """Menguji penanganan remainder secara native (seq_len % chunk != 0).

    Contoh: seq_len = 13, chunk_size = 8 -> chunk 0: m=8, chunk 1: m=5.
    Membandingkan chunked scan dengan token-by-token sequential recurrence.
    """
    var seq_len = 13
    var chunk_size = 8
    var dk = 16
    var dv = 16

    var k_all = pseudo_random_vector(seq_len * dk, 707)
    normalize_k_matrix(k_all, seq_len, dk)
    var v_all = pseudo_random_vector(seq_len * dv, 808)

    var beta_all = List[Float32]()
    for i in range(seq_len):
        beta_all.append(Float32(0.4 + 0.02 * Float32(i)))

    # 1. Jalankan via chunked_gdn_scan
    var state_chunked = GDNState(1, dv, dk)
    state_chunked.zero()
    chunked_gdn_scan(
        state_chunked, 0, k_all, v_all, beta_all, seq_len, dk, dv, chunk_size
    )

    # 2. Jalankan via serial token-by-token recurrence (naive reference)
    var s_serial = List[Float32]()
    s_serial.resize(dv * dk, Float32(0.0))
    var p_s_serial = s_serial.unsafe_ptr()
    var p_k_all = k_all.unsafe_ptr()
    var p_v_all = v_all.unsafe_ptr()

    for t in range(seq_len):
        var k_off = t * dk
        var v_off = t * dv
        var b_t = beta_all[t]

        # S @ (I - b_t * k_t k_t^T) + b_t * v_t k_t^T
        # 1. sk = S @ k_t in R^{dv}
        var sk = List[Float32]()
        sk.resize(dv, Float32(0.0))
        for r in range(dv):
            var dot = Float32(0.0)
            for c in range(dk):
                dot += (
                    p_s_serial[unsafe_offset=r * dk + c]
                    * p_k_all[unsafe_offset=k_off + c]
                )
            sk[r] = dot

        # 2. Update S[r, c] -= b_t * sk[r] * k_t[c] + b_t * v_t[r] * k_t[c]
        for r in range(dv):
            var sk_r = sk[r]
            var v_r = p_v_all[unsafe_offset=v_off + r]
            for c in range(dk):
                var k_c = p_k_all[unsafe_offset=k_off + c]
                p_s_serial[unsafe_offset=r * dk + c] += b_t * (v_r - sk_r) * k_c

    # 3. Hitung delta maksimum
    var max_diff = Float32(0.0)
    for r in range(dv):
        for c in range(dk):
            var val_chunked = state_chunked.get(0, r, c)
            var val_serial = s_serial[r * dk + c]
            var diff = abs(val_chunked - val_serial)
            if diff > max_diff:
                max_diff = diff

    assert_true(
        max_diff <= Float32(1e-5),
        "Native partial remainder diff exceeded 1e-5",
    )


def test_asymmetric_dimensions() raises:
    """Menguji kepatuhan kontrak dimensi asimetris: dk=32, dv=48.

    Memastikan tidak ada bug transposisi orientasi antara dv dan dk.
    """
    var seq_len = 16
    var chunk_size = 8
    var dk = 32
    var dv = 48

    var k_all = pseudo_random_vector(seq_len * dk, 909)
    normalize_k_matrix(k_all, seq_len, dk)
    var v_all = pseudo_random_vector(seq_len * dv, 1010)

    var beta_all = List[Float32]()
    for _ in range(seq_len):
        beta_all.append(Float32(0.5))

    var state = GDNState(2, dv, dk)
    state.zero()
    assert_equal(len(state.data), 2 * 48 * 32)

    # Jalankan layer 0 dan 1
    chunked_gdn_scan(
        state, 0, k_all, v_all, beta_all, seq_len, dk, dv, chunk_size
    )
    chunked_gdn_scan(
        state, 1, k_all, v_all, beta_all, seq_len, dk, dv, chunk_size
    )

    state.check_finite(0)
    state.check_finite(1)

    # Verifikasi state non-zero
    var sum_l0 = Float32(0.0)
    for r in range(dv):
        for c in range(dk):
            sum_l0 += abs(state.get(0, r, c))
    assert_true(sum_l0 > Float32(0.1), "State layer 0 should be non-zero")


def test_nan_inf_guard() raises:
    """Menguji fail-fast deteksi NaN dan INF pada GDN state."""
    var state = GDNState(1, 4, 4)
    state.zero()

    # Inisialisasi normal harus lulus
    state.check_finite()

    # Masukkan NaN
    state.set(0, 1, 2, Float32(0.0) / Float32(0.0))
    var caught_nan = False
    try:
        state.check_finite()
    except:
        caught_nan = True
    assert_true(caught_nan, "State check_finite must raise on NaN")

    # Masukkan INF
    state.set(0, 1, 2, Float32(1.0) / Float32(0.0))
    var caught_inf = False
    try:
        state.check_finite()
    except:
        caught_inf = True
    assert_true(caught_inf, "State check_finite must raise on INF")


def test_boundary_split_continuation() raises:
    """Menguji invarian kontinuitas state melintasi batas pemotongan token."""
    var total_seq = 16
    var chunk_size = 8
    var dk = 16
    var dv = 16

    var k_all = pseudo_random_vector(total_seq * dk, 1111)
    normalize_k_matrix(k_all, total_seq, dk)
    var v_all = pseudo_random_vector(total_seq * dv, 1212)

    var beta_all = List[Float32]()
    for i in range(total_seq):
        beta_all.append(Float32(0.35 + 0.01 * Float32(i)))

    # Single pass baseline
    var state_single = GDNState(1, dv, dk)
    state_single.zero()
    chunked_gdn_scan(
        state_single,
        0,
        k_all,
        v_all,
        beta_all,
        total_seq,
        dk,
        dv,
        chunk_size,
    )

    # Uji boundary splits
    var split_points = List[Int]()
    split_points.append(8)  # Clean boundary (50/50)
    split_points.append(7)  # Under-cut
    split_points.append(9)  # Over-cut
    split_points.append(5)  # Prime split

    for s_idx in range(len(split_points)):
        var s1 = split_points[s_idx]
        var s2 = total_seq - s1

        # 1. Prefill seq1
        var state_cont = GDNState(1, dv, dk)
        state_cont.zero()

        var k1 = List[Float32]()
        k1.resize(s1 * dk, Float32(0.0))
        for i in range(s1 * dk):
            k1[i] = k_all[i]
        var v1 = List[Float32]()
        v1.resize(s1 * dv, Float32(0.0))
        for i in range(s1 * dv):
            v1[i] = v_all[i]
        var beta1 = List[Float32]()
        for i in range(s1):
            beta1.append(beta_all[i])

        chunked_gdn_scan(state_cont, 0, k1, v1, beta1, s1, dk, dv, chunk_size)

        # 2. Continuation seq2 dari state_cont
        var k2 = List[Float32]()
        k2.resize(s2 * dk, Float32(0.0))
        var k2_off = s1 * dk
        for i in range(s2 * dk):
            k2[i] = k_all[k2_off + i]
        var v2 = List[Float32]()
        v2.resize(s2 * dv, Float32(0.0))
        var v2_off = s1 * dv
        for i in range(s2 * dv):
            v2[i] = v_all[v2_off + i]
        var beta2 = List[Float32]()
        for i in range(s2):
            beta2.append(beta_all[s1 + i])

        chunked_gdn_scan(state_cont, 0, k2, v2, beta2, s2, dk, dv, chunk_size)

        # 3. Verifikasi ekuivalensi vs single-pass baseline
        var max_diff = Float32(0.0)
        for r in range(dv):
            for c in range(dk):
                var d = abs(state_single.get(0, r, c) - state_cont.get(0, r, c))
                if d > max_diff:
                    max_diff = d
        assert_true(
            max_diff <= Float32(1e-5),
            "Continuation failed for split s1=" + String(s1),
        )


def test_chunk_size_sweep_consistency() raises:
    """Memverifikasi bahwa variasi chunk_size {4, 8, 16} menghasilkan state yang sama.
    """
    var seq_len = 32
    var dk = 16
    var dv = 16

    var k_all = pseudo_random_vector(seq_len * dk, 1313)
    normalize_k_matrix(k_all, seq_len, dk)
    var v_all = pseudo_random_vector(seq_len * dv, 1414)

    var beta_all = List[Float32]()
    for _ in range(seq_len):
        beta_all.append(Float32(0.45))

    var state_c8 = GDNState(1, dv, dk)
    state_c8.zero()
    chunked_gdn_scan(
        state_c8, 0, k_all, v_all, beta_all, seq_len, dk, dv, chunk_size=8
    )

    var state_c16 = GDNState(1, dv, dk)
    state_c16.zero()
    chunked_gdn_scan(
        state_c16, 0, k_all, v_all, beta_all, seq_len, dk, dv, chunk_size=16
    )

    var state_c32 = GDNState(1, dv, dk)
    state_c32.zero()
    chunked_gdn_scan(
        state_c32, 0, k_all, v_all, beta_all, seq_len, dk, dv, chunk_size=32
    )

    # Periksa selisih C=8 vs C=16
    var max_diff_8_16 = Float32(0.0)
    for r in range(dv):
        for c in range(dk):
            var d = abs(state_c8.get(0, r, c) - state_c16.get(0, r, c))
            if d > max_diff_8_16:
                max_diff_8_16 = d
    assert_true(
        max_diff_8_16 <= Float32(1e-5),
        "Diff between C=8 and C=16 exceeded 1e-5",
    )

    # Periksa selisih C=16 vs C=32
    var max_diff_16_32 = Float32(0.0)
    for r in range(dv):
        for c in range(dk):
            var d = abs(state_c16.get(0, r, c) - state_c32.get(0, r, c))
            if d > max_diff_16_32:
                max_diff_16_32 = d
    assert_true(
        max_diff_16_32 <= Float32(1e-5),
        "Diff between C=16 and C=32 exceeded 1e-5",
    )


def test_gdns_v1_roundtrip_and_checksum() raises:
    """Menguji serialisasi GDNS v1 roundtrip, atomic write, dan verifikasi checksum SHA-256.
    """
    var path = "/tmp/test_m8_gdns_roundtrip.bin"
    var state_orig = GDNState(2, 4, 8)
    var test_vals = pseudo_random_vector(2 * 4 * 8, 1515)
    for i in range(2 * 4 * 8):
        state_orig.data[i] = test_vals[i]

    # 1. Tulis ke file
    write_gdns_v1(path, state_orig)

    # 2. Baca kembali dan verifikasi
    var state_read = read_gdns_v1(path)
    assert_equal(state_read.layers, 2)
    assert_equal(state_read.dv, 4)
    assert_equal(state_read.dk, 8)
    for i in range(2 * 4 * 8):
        assert_almost_equal(state_read.data[i], state_orig.data[i], atol=1e-6)

    # 3. Uji tamper deteksi: ubah 1 byte di payload -> harus raise CORRUPT_STATE_CHECKSUM
    var f = open(path, "r")
    var raw_bytes = f.read_bytes()
    f.close()

    # Ubah 1 byte di payload (offset 130)
    raw_bytes[130] = raw_bytes[130] ^ UInt8(0xFF)
    var f_tamper = open(path, "w")
    f_tamper.write_bytes(Span(raw_bytes))
    f_tamper.close()

    var caught_tamper = False
    try:
        _ = read_gdns_v1(path)
    except:
        caught_tamper = True
    assert_true(
        caught_tamper, "Tampered GDNS file must be rejected by SHA-256 checksum"
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_wy_coefficients_backward_solve]()
    suite.test[test_operator_composition_invariant]()
    suite.test[test_chunked_state_update_equivalence]()
    suite.test[test_native_partial_remainder]()
    suite.test[test_asymmetric_dimensions]()
    suite.test[test_nan_inf_guard]()
    suite.test[test_boundary_split_continuation]()
    suite.test[test_chunk_size_sweep_consistency]()
    suite.test[test_gdns_v1_roundtrip_and_checksum]()
    suite^.run()
