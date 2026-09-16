# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk forward_layer: residual-free attention, moe_combine, dan timing."""

from core.config import ModelConfig
from layers.attention import AttentionWeights
from layers.forward_layer import (
    LayerTiming,
    _format_1d_ints,
    _format_2d_ints,
    forward_attention_step,
    moe_combine_no_residual,
)
from layers.qkv import QKVWeights
from std.collections import List
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)


def test_format_ints() raises:
    """Format integer array 1D dan 2D untuk routing dumps."""
    var v1 = List[Int]()
    v1.append(48)
    v1.append(33)
    v1.append(4)
    v1.append(36)
    assert_equal(_format_1d_ints(v1), "[48,33,4,36]")

    var v2 = List[List[Int]]()
    v2.append(v1^)
    var v1_b = List[Int]()
    v1_b.append(1)
    v1_b.append(2)
    v1_b.append(3)
    v1_b.append(4)
    v2.append(v1_b^)
    assert_equal(_format_2d_ints(v2), "[[48,33,4,36],[1,2,3,4]]")


def test_layer_timing_init() raises:
    """Inisialisasi dan akses nilai struct LayerTiming."""
    var lt = LayerTiming(3, 5.2, 0.07, 0.25, 5.52)
    assert_equal(lt.layer, 3)
    assert_almost_equal(lt.pread_sec, 5.2, rtol=1e-5)
    assert_almost_equal(lt.attention_sec, 0.07, rtol=1e-5)
    assert_almost_equal(lt.moe_sec, 0.25, rtol=1e-5)
    assert_almost_equal(lt.total_sec, 5.52, rtol=1e-5)


def test_moe_combine_no_residual_math() raises:
    """Agregasi MoE membuktikan y = sum(p_i E_i) + sigma(g_sh) E_sh (TANPA input x).
    """
    var seq_len = 2
    var hidden_dim = 4

    # 2 token, masing-masing top-2 experts
    var routed_outputs = List[List[Float32]]()
    # token 0: expert 0 = [1, 2, 3, 4], expert 1 = [5, 6, 7, 8]
    var tok0_exp = List[Float32]()
    tok0_exp.append(Float32(1.0))
    tok0_exp.append(Float32(2.0))
    tok0_exp.append(Float32(3.0))
    tok0_exp.append(Float32(4.0))
    tok0_exp.append(Float32(5.0))
    tok0_exp.append(Float32(6.0))
    tok0_exp.append(Float32(7.0))
    tok0_exp.append(Float32(8.0))
    routed_outputs.append(tok0_exp^)

    # token 1: expert 0 = [2, 2, 2, 2], expert 1 = [4, 4, 4, 4]
    var tok1_exp = List[Float32]()
    for _ in range(4):
        tok1_exp.append(Float32(2.0))
    for _ in range(4):
        tok1_exp.append(Float32(4.0))
    routed_outputs.append(tok1_exp^)

    # probs: token 0 = [0.6, 0.4], token 1 = [0.5, 0.5]
    var router_probs = List[List[Float32]]()
    var p0 = List[Float32]()
    p0.append(Float32(0.6))
    p0.append(Float32(0.4))
    router_probs.append(p0^)
    var p1 = List[Float32]()
    p1.append(Float32(0.5))
    p1.append(Float32(0.5))
    router_probs.append(p1^)

    # shared expert output: token 0 = [10, 10, 10, 10], token 1 = [20, 20, 20, 20]
    var shared_out = List[Float32]()
    for _ in range(4):
        shared_out.append(Float32(10.0))
    for _ in range(4):
        shared_out.append(Float32(20.0))

    # shared gate scores: token 0 = 0.5, token 1 = 0.25
    var shared_gate = List[Float32]()
    shared_gate.append(Float32(0.5))
    shared_gate.append(Float32(0.25))

    var out = moe_combine_no_residual(
        routed_outputs,
        router_probs,
        shared_out,
        shared_gate,
        seq_len,
        hidden_dim,
        0,
    )
    assert_equal(len(out), seq_len * hidden_dim)

    # Token 0 index 0:
    # routed = 0.6 * 1.0 + 0.4 * 5.0 = 0.6 + 2.0 = 2.6
    # shared = 0.5 * 10.0 = 5.0
    # total = 2.6 + 5.0 = 7.6 (TANPA input x!)
    assert_almost_equal(out[0], Float32(7.6), rtol=1e-5)

    # Token 0 index 1:
    # routed = 0.6 * 2.0 + 0.4 * 6.0 = 1.2 + 2.4 = 3.6
    # shared = 5.0
    # total = 8.6
    assert_almost_equal(out[1], Float32(8.6), rtol=1e-5)

    # Token 1 index 0:
    # routed = 0.5 * 2.0 + 0.5 * 4.0 = 1.0 + 2.0 = 3.0
    # shared = 0.25 * 20.0 = 5.0
    # total = 8.0
    assert_almost_equal(out[4], Float32(8.0), rtol=1e-5)


def test_moe_combine_no_residual_error_checks() raises:
    """Error handling pada dimensi mismatch atau non-finite di moe_combine."""
    var empty_routed = List[List[Float32]]()
    var empty_probs = List[List[Float32]]()
    var empty_sh = List[Float32]()
    var empty_gate = List[Float32]()

    with assert_raises(contains="dimensions must be positive"):
        _ = moe_combine_no_residual(
            empty_routed, empty_probs, empty_sh, empty_gate, 0, 4, 0
        )

    with assert_raises(contains="length mismatch in aggregation"):
        var fake_sh = List[Float32]()
        fake_sh.append(Float32(1.0))
        _ = moe_combine_no_residual(
            empty_routed, empty_probs, fake_sh, empty_gate, 2, 4, 0
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
