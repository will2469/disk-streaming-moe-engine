# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle cross-check runner for M3-W1 MoE Router (F8a, F8b).

Membandingkan eksekusi engine router Mojo terhadap ground truth PyTorch oracle
pada 256 input acak:
- Invariant SET expert top-4 identik 100% (0 flip seleksi, F8b)
- Probabilitas unrenormalized match dalam atol 1e-5
- Invariant norm_topk_prob=false: sum(top4) < 1.0
"""

from model import RouterConfig, RoutingInfo, router_forward
from safetensors import read_small_file
from std.builtin.dtype import DType
from std.collections import List
from std.os import getenv
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)


def read_f32_bin(path: String, count: Int) raises -> List[Float32]:
    var raw = read_small_file(path)
    if len(raw) != count * 4:
        raise Error("file size mismatch: " + path)
    var out = List[Float32]()
    out.reserve(count)
    var p_f32 = raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
    for i in range(count):
        out.append(p_f32[unsafe_offset=i])
    return out^


def read_i32_bin(path: String, count: Int) raises -> List[Int]:
    var raw = read_small_file(path)
    if len(raw) != count * 4:
        raise Error("file size mismatch: " + path)
    var out = List[Int]()
    out.reserve(count)
    var p_i32 = raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.int32]]()
    for i in range(count):
        out.append(Int(p_i32[unsafe_offset=i]))
    return out^


def main() raises:
    var fixture_dir = getenv("ROUTER_FIXTURE_DIR")
    if fixture_dir == "":
        fixture_dir = "/tmp/test_m3_w1"

    var seq_len = 256
    var hidden = 2048
    var num_experts = 60
    var top_k = 4

    var x_path = String(fixture_dir, "/x.bin")
    var w_path = String(fixture_dir, "/w_router.bin")
    var ref_idx_path = String(fixture_dir, "/topk_indices_ref.bin")
    var ref_prob_path = String(fixture_dir, "/topk_probs_ref.bin")

    var x = read_f32_bin(x_path, seq_len * hidden)
    var w = read_f32_bin(w_path, num_experts * hidden)
    var ref_indices = read_i32_bin(ref_idx_path, seq_len * top_k)
    var ref_probs = read_f32_bin(ref_prob_path, seq_len * top_k)

    var cfg = RouterConfig(num_experts, top_k, False)
    var info = router_forward(x, w, seq_len, hidden, cfg)

    assert_equal(info.num_tokens, seq_len)
    assert_equal(info.top_k, top_k)

    var flip_count = 0
    for t in range(seq_len):
        ref exp_row = info.selected_experts[t]
        ref prob_row = info.router_probs[t]
        var sum_topk = Float32(0.0)

        for k in range(top_k):
            var expected_expert = ref_indices[t * top_k + k]
            var expected_prob = ref_probs[t * top_k + k]
            if exp_row[k] != expected_expert:
                flip_count += 1
            assert_almost_equal(prob_row[k], expected_prob, atol=1e-5)
            sum_topk += prob_row[k]

        # Property norm_topk_prob=false: sum <= 1.0 (dan < 1.0 jika sisa > 0)
        assert_true(sum_topk <= Float32(1.0))
        assert_true(sum_topk < Float32(1.0))

    if flip_count > 0:
        raise Error(
            '{"error_type":"ROUTING_VIOLATION","detail":"Router expert flip'
            ' detected vs oracle PyTorch","flips":'
            + String(flip_count)
            + "}"
        )

    print("M3-W1 Oracle Cross-Check PASSED:")
    print("  Tokens verified:", seq_len)
    print("  Total expert comparisons:", seq_len * top_k)
    print("  Flips detected:", flip_count)
    print("  Property SET routing invariant: 100% MATCH")
    print("  Property norm_topk_prob=false (sum < 1.0): VERIFIED")
