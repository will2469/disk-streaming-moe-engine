# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M11-W1: Static Worker Pool dan Kontrak Determinisme Multithreading CPU (§3.2).
"""

from core.worker_pool import WorkerPool, partition_range
from layers.swiglu import SwigluWeights, swiglu_forward
from quant.dequant_kernel import (
    dequant_kernel_simd,
    dequant_kernel_simd_parallel,
)
from std.collections import List
from std.math import abs, max, min
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)


def test_partition_range_invariants() raises:
    """Verifikasi 4 invarian matematis fungsi partition_range(N, thread_id, c).
    """
    var thread_counts = List[Int]()
    thread_counts.append(1)
    thread_counts.append(2)
    thread_counts.append(4)
    thread_counts.append(8)

    var item_counts = List[Int]()
    item_counts.append(1)
    item_counts.append(7)
    item_counts.append(8)
    item_counts.append(32)
    item_counts.append(100)
    item_counts.append(101)

    for tc_idx in range(len(thread_counts)):
        var c = thread_counts[tc_idx]
        for ic_idx in range(len(item_counts)):
            var total = item_counts[ic_idx]

            var min_chunk = total + 1
            var max_chunk = -1

            # Invarian 1: start(0) == 0
            var p0 = partition_range(total, 0, c)
            assert_equal(p0[0], 0)

            # Invarian 2: end(c - 1) == total
            var p_last = partition_range(total, c - 1, c)
            assert_equal(p_last[1], total)

            # Invarian 3 & 4: kontinuitas dan selisih beban <= 1
            for tid in range(c):
                var part = partition_range(total, tid, c)
                var count = part[1] - part[0]
                assert_true(count >= 0)
                if count < min_chunk:
                    min_chunk = count
                if count > max_chunk:
                    max_chunk = count

                if tid + 1 < c:
                    var part_next = partition_range(total, tid + 1, c)
                    assert_equal(part[1], part_next[0])

            assert_true(max_chunk - min_chunk <= 1)


def test_worker_pool_lifecycle() raises:
    """Verifikasi inisialisasi, status aktif, dan shutdown WorkerPool."""
    # Single-thread: tidak membuat thread OS tambahan
    var pool1 = WorkerPool(1)
    assert_equal(pool1.num_threads, 1)
    assert_false(pool1.is_active)
    pool1.shutdown()
    assert_false(pool1.is_active)

    # Multi-thread: membuat c-1 worker thread aktif
    var pool4 = WorkerPool(4)
    assert_equal(pool4.num_threads, 4)
    assert_true(pool4.is_active)
    pool4.shutdown()
    assert_false(pool4.is_active)


def test_parallel_dequant_bit_exact() raises:
    """Verifikasi paritas bit-exact dekuantisasi 4-bit ke BF16 (Delta_max == 0.0).
    """
    var num_elements = 2048
    var group_size = 32
    var num_groups = num_elements // group_size

    var scales = List[Float16]()
    for g in range(num_groups):
        scales.append(Float16(0.25 * Float32((g % 7) + 1)))

    var packed = List[UInt8]()
    var num_bytes = num_elements // 2
    for b in range(num_bytes):
        var lo_nib = (b * 3 + 1) & 0x07
        var hi_nib = (b * 5 + 3) & 0x07
        packed.append(UInt8((hi_nib << 4) | lo_nib))

    # Eksekusi sekuensial (single-core)
    var seq_out = dequant_kernel_simd(scales, packed, num_elements, group_size)

    # Eksekusi paralel 4 worker
    var pool = WorkerPool(4)
    var par_out = dequant_kernel_simd_parallel(
        scales, packed, num_elements, group_size, pool
    )
    pool.shutdown()

    assert_equal(len(seq_out), len(par_out))
    var max_diff = Float32(0.0)
    for i in range(num_elements):
        var d = abs(Float32(seq_out[i]) - Float32(par_out[i]))
        if d > max_diff:
            max_diff = d
        assert_equal(seq_out[i], par_out[i])

    assert_equal(max_diff, Float32(0.0))


def test_parallel_moe_experts_bit_exact() raises:
    """Verifikasi paritas bit-exact MoE 8-expert GEMM (Delta_max == 0.0)."""
    var hidden_dim = 64
    var inter_dim = 128
    var num_experts = 8
    var top_k = 8

    var x_tok = List[Float32]()
    for d in range(hidden_dim):
        x_tok.append(Float32(0.01 * Float32((d % 11) + 1)))

    var experts = List[SwigluWeights]()
    var w_gate_ptrs = List[Int]()
    var w_up_ptrs = List[Int]()
    var w_down_ptrs = List[Int]()

    for exp_id in range(num_experts):
        var wg = List[Float32]()
        wg.resize(inter_dim * hidden_dim, Float32(0.0))
        var wu = List[Float32]()
        wu.resize(inter_dim * hidden_dim, Float32(0.0))
        var wd = List[Float32]()
        wd.resize(hidden_dim * inter_dim, Float32(0.0))

        for j in range(inter_dim):
            for d in range(hidden_dim):
                wg[j * hidden_dim + d] = Float32(
                    0.001 * Float32((exp_id + j + d) % 17)
                )
                wu[j * hidden_dim + d] = Float32(
                    0.001 * Float32((exp_id * 2 + j + d) % 19)
                )
                wd[d * inter_dim + j] = Float32(
                    0.001 * Float32((exp_id * 3 + j + d) % 23)
                )

        experts.append(SwigluWeights(wg^, wu^, wd^, hidden_dim, inter_dim))

    for exp_id in range(num_experts):
        w_gate_ptrs.append(Int(experts[exp_id].w_gate.unsafe_ptr()))
        w_up_ptrs.append(Int(experts[exp_id].w_up.unsafe_ptr()))
        w_down_ptrs.append(Int(experts[exp_id].w_down.unsafe_ptr()))

    var selected_experts = List[Int]()
    for k in range(top_k):
        selected_experts.append(k)

    # 1. Evaluasi sekuensial
    var seq_out = List[Float32]()
    seq_out.reserve(top_k * hidden_dim)
    for k in range(top_k):
        var exp_id = selected_experts[k]
        ref exp_w = experts[exp_id]
        var out_exp = swiglu_forward(
            x_tok, exp_w, 1, hidden_dim, exp_w.inter_dim
        )
        for d in range(hidden_dim):
            seq_out.append(out_exp[d])

    # 2. Evaluasi paralel dengan WorkerPool(4)
    var pool = WorkerPool(4)
    var par_out = List[Float32]()
    par_out.resize(top_k * hidden_dim, Float32(0.0))

    pool.parallel_moe_experts(
        top_k,
        hidden_dim,
        inter_dim,
        Int(x_tok.unsafe_ptr()),
        Int(par_out.unsafe_ptr()),
        Int(selected_experts.unsafe_ptr()),
        Int(w_gate_ptrs.unsafe_ptr()),
        Int(w_up_ptrs.unsafe_ptr()),
        Int(w_down_ptrs.unsafe_ptr()),
    )
    _ = x_tok
    _ = experts
    _ = selected_experts
    _ = w_gate_ptrs
    _ = w_up_ptrs
    _ = w_down_ptrs

    pool.shutdown()

    assert_equal(len(seq_out), len(par_out))
    var max_diff = Float32(0.0)
    for i in range(len(seq_out)):
        var d = abs(seq_out[i] - par_out[i])
        if d > max_diff:
            max_diff = d
        assert_equal(seq_out[i], par_out[i])

    assert_equal(max_diff, Float32(0.0))


def test_multithreading_hash_determinism_5_runs() raises:
    """Verifikasi bahwa 5 eksekusi berturut-turut pada thread pool menghasilkan output bit-identik.
    """
    var hidden_dim = 32
    var inter_dim = 64
    var num_experts = 8
    var top_k = 8

    var x_tok = List[Float32]()
    for d in range(hidden_dim):
        x_tok.append(Float32(0.05 * Float32((d % 7) + 1)))

    var experts = List[SwigluWeights]()
    var w_gate_ptrs = List[Int]()
    var w_up_ptrs = List[Int]()
    var w_down_ptrs = List[Int]()

    for _ in range(num_experts):
        var wg = List[Float32]()
        wg.resize(inter_dim * hidden_dim, Float32(0.01))
        var wu = List[Float32]()
        wu.resize(inter_dim * hidden_dim, Float32(0.01))
        var wd = List[Float32]()
        wd.resize(hidden_dim * inter_dim, Float32(0.01))
        experts.append(SwigluWeights(wg^, wu^, wd^, hidden_dim, inter_dim))

    for exp_id in range(num_experts):
        w_gate_ptrs.append(Int(experts[exp_id].w_gate.unsafe_ptr()))
        w_up_ptrs.append(Int(experts[exp_id].w_up.unsafe_ptr()))
        w_down_ptrs.append(Int(experts[exp_id].w_down.unsafe_ptr()))

    var selected_experts = List[Int]()
    for k in range(top_k):
        selected_experts.append(k)

    var pool = WorkerPool(4)

    # Run 1: baseline
    var baseline = List[Float32]()
    baseline.resize(top_k * hidden_dim, Float32(0.0))
    pool.parallel_moe_experts(
        top_k,
        hidden_dim,
        inter_dim,
        Int(x_tok.unsafe_ptr()),
        Int(baseline.unsafe_ptr()),
        Int(selected_experts.unsafe_ptr()),
        Int(w_gate_ptrs.unsafe_ptr()),
        Int(w_up_ptrs.unsafe_ptr()),
        Int(w_down_ptrs.unsafe_ptr()),
    )

    # Runs 2 s/d 5: harus bit-identik dengan baseline
    for _ in range(4):
        var current = List[Float32]()
        current.resize(top_k * hidden_dim, Float32(0.0))
        pool.parallel_moe_experts(
            top_k,
            hidden_dim,
            inter_dim,
            Int(x_tok.unsafe_ptr()),
            Int(current.unsafe_ptr()),
            Int(selected_experts.unsafe_ptr()),
            Int(w_gate_ptrs.unsafe_ptr()),
            Int(w_up_ptrs.unsafe_ptr()),
            Int(w_down_ptrs.unsafe_ptr()),
        )
        for i in range(len(baseline)):
            assert_equal(baseline[i], current[i])

    _ = x_tok
    _ = experts
    _ = selected_experts
    _ = w_gate_ptrs
    _ = w_up_ptrs
    _ = w_down_ptrs
    pool.shutdown()


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_partition_range_invariants]()
    suite.test[test_worker_pool_lifecycle]()
    suite.test[test_parallel_dequant_bit_exact]()
    suite.test[test_parallel_moe_experts_bit_exact]()
    suite.test[test_multithreading_hash_determinism_5_runs]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
