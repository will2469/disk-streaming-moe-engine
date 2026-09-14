# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle cross-check runner for M3-W2 MoE SwiGLU + Shared Gate (F8c, F8d).

Membandingkan eksekusi engine MoE SwiGLU Mojo terhadap ground truth PyTorch oracle:
- SwiGLU routed experts (60 experts, inter 1408): W_down (SiLU(W_gate x) * W_up x)
- Shared expert SwiGLU (inter 5632) + Sigmoid Gate (Invariant #2: WAJIB Sigmoid)
- MoE aggregation: sum(p_i E_i(x)) + sigma(g_sh) E_sh(x) + x
- Kriteria F10: delta_max <= 1e-3, epsilon_rel <= 1e-4
- Invariant Keras #2: deteksi error jika non-sigmoid gate dicoba.
"""

from model import (
    RouterConfig,
    RoutingInfo,
    SwigluWeights,
    moe_aggregate_forward,
    router_forward,
    shared_gate_forward,
    swiglu_forward,
)
from safetensors import read_small_file
from std.builtin.dtype import DType
from std.collections import List
from std.math import abs, max
from std.os import getenv
from std.testing import assert_almost_equal, assert_equal, assert_true


def read_f32_bin(path: String, count: Int) raises -> List[Float32]:
    """Membaca array float32 dari file biner."""
    var raw = read_small_file(path)
    if len(raw) != count * 4:
        raise Error(
            "file size mismatch: "
            + path
            + " expected "
            + String(count * 4)
            + " got "
            + String(len(raw))
        )
    var out = List[Float32]()
    out.reserve(count)
    var p_f32 = raw.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
    for i in range(count):
        out.append(p_f32[unsafe_offset=i])
    return out^


def read_i32_bin(path: String, count: Int) raises -> List[Int]:
    """Membaca array int32 dari file biner."""
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
    var fixture_dir = getenv("MOE_FIXTURE_DIR")
    if fixture_dir == "":
        fixture_dir = "/tmp/test_m3_w2"

    var seq_len = 8
    var hidden = 2048
    var inter_routed = 1408
    var inter_shared = 5632
    var num_experts = 60
    var top_k = 4

    print("Loading activations and router weights...")
    var x = read_f32_bin(String(fixture_dir, "/x.bin"), seq_len * hidden)
    var w_router = read_f32_bin(
        String(fixture_dir, "/w_router.bin"), num_experts * hidden
    )
    var ref_indices = read_i32_bin(
        String(fixture_dir, "/topk_indices_ref.bin"), seq_len * top_k
    )
    var ref_probs = read_f32_bin(
        String(fixture_dir, "/topk_probs_ref.bin"), seq_len * top_k
    )

    print("1. Running router forward...")
    var cfg = RouterConfig(num_experts, top_k, False)
    var routing = router_forward(x, w_router, seq_len, hidden, cfg)

    var flip_count = 0
    for t in range(seq_len):
        ref exp_row = routing.selected_experts[t]
        ref prob_row = routing.router_probs[t]
        for k in range(top_k):
            var exp_ref = ref_indices[t * top_k + k]
            var prob_ref = ref_probs[t * top_k + k]
            if exp_row[k] != exp_ref:
                flip_count += 1
            assert_almost_equal(prob_row[k], prob_ref, atol=1e-5)
    assert_equal(flip_count, 0)
    print("   Router selection 100% MATCH (0 flip)")

    print("2. Running routed experts SwiGLU...")
    var all_routed_outputs = List[List[Float32]]()
    for t in range(seq_len):
        ref exp_row = routing.selected_experts[t]
        var tok_out = List[Float32]()
        tok_out.reserve(top_k * hidden)

        var x_t = List[Float32]()
        x_t.reserve(hidden)
        for d in range(hidden):
            x_t.append(x[t * hidden + d])

        for k in range(top_k):
            var e_idx = exp_row[k]
            var w_g = read_f32_bin(
                String(fixture_dir, "/expert_", String(e_idx), "_gate.bin"),
                inter_routed * hidden,
            )
            var w_u = read_f32_bin(
                String(fixture_dir, "/expert_", String(e_idx), "_up.bin"),
                inter_routed * hidden,
            )
            var w_d = read_f32_bin(
                String(fixture_dir, "/expert_", String(e_idx), "_down.bin"),
                hidden * inter_routed,
            )
            var weights = SwigluWeights(w_g^, w_u^, w_d^, hidden, inter_routed)
            var e_val = swiglu_forward(x_t, weights, 1, 0, e_idx)
            for d in range(hidden):
                tok_out.append(e_val[d])

        all_routed_outputs.append(tok_out^)
    print("   Routed experts SwiGLU complete.")

    print("3. Running shared expert SwiGLU & Sigmoid Gate...")
    var w_sh_gate_proj = read_f32_bin(
        String(fixture_dir, "/w_shared_gate_proj.bin"), inter_shared * hidden
    )
    var w_sh_up_proj = read_f32_bin(
        String(fixture_dir, "/w_shared_up_proj.bin"), inter_shared * hidden
    )
    var w_sh_down_proj = read_f32_bin(
        String(fixture_dir, "/w_shared_down_proj.bin"), hidden * inter_shared
    )
    var weights_sh = SwigluWeights(
        w_sh_gate_proj^, w_sh_up_proj^, w_sh_down_proj^, hidden, inter_shared
    )
    var shared_out = swiglu_forward(x, weights_sh, seq_len, 0, -1)

    var w_sh_gate = read_f32_bin(
        String(fixture_dir, "/w_shared_gate.bin"), hidden
    )
    var shared_gate_scores = shared_gate_forward(
        x, w_sh_gate, seq_len, hidden, 0, "sigmoid"
    )
    print("   Shared expert SwiGLU and Sigmoid Gate complete.")

    print("4. Running MoE aggregation forward...")
    var y_final = moe_aggregate_forward(
        x,
        all_routed_outputs,
        routing.router_probs,
        shared_out,
        shared_gate_scores,
        seq_len,
        hidden,
        0,
    )
    print("   MoE aggregation complete.")

    print("5. Comparing vs PyTorch oracle reference...")
    var moe_ref = read_f32_bin(
        String(fixture_dir, "/moe_ref.bin"), seq_len * hidden
    )
    var max_delta = Float32(0.0)
    var max_ref = Float32(0.0)
    for i in range(seq_len * hidden):
        var d = abs(y_final[i] - moe_ref[i])
        if d > max_delta:
            max_delta = d
        var r = abs(moe_ref[i])
        if r > max_ref:
            max_ref = r

    var eps_rel = max_delta / max(max_ref, Float32(1e-6))
    print("   Delta max:  ", max_delta)
    print("   Epsilon rel:", eps_rel)

    # Invariant criteria F10: delta_max <= 1e-3, eps_rel <= 1e-4
    assert_true(max_delta <= Float32(0.001))
    assert_true(eps_rel <= Float32(0.0001))

    print("6. Invariant Keras #2 negative test (rejection of non-sigmoid)...")
    var raised_linear = False
    try:
        var _s_bad = shared_gate_forward(
            x, w_sh_gate, seq_len, hidden, 0, "linear"
        )
    except e:
        raised_linear = String(e).find("GATE_ERROR") != -1
    assert_true(raised_linear)
    print("   Non-sigmoid gate rejection (GATE_ERROR): VERIFIED")

    print("M3-W2 Oracle Cross-Check PASSED:")
    print("  Tokens verified:", seq_len)
    print("  Experts routed per token:", top_k)
    print("  Shared expert inter dim:", inter_shared)
    print("  Flips detected: 0")
    print("  Max Delta vs PyTorch:", max_delta)
    print("  Relative Epsilon:", eps_rel)
    print("  Invariant #2 (Sigmoid gate): VERIFIED")
