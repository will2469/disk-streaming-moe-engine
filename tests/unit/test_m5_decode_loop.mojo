# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M5-W2: Decode Loop (Posisi, Recompute Equivalence, Lokalisasi Tensor, F3b/F5)."""

from core.config import ModelConfig
from core.f3b_f5 import (
    B_TOK_DISK_BYTES,
    F3bTraffic,
    F5Forecast,
    KV_LAYER_TOTAL_SLOT_BYTES,
    KV_SLOT_BYTES_PER_LAYER,
)
from layers.attention import AttentionWeights, o_project
from layers.decode_loop import (
    DecodeStepContext,
    compare_tensors_loose,
    recompute_attention_at_position,
)
from layers.forward_layer import forward_attention_decode_step
from layers.kv_cache import (
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_LAYERS,
    SLOT_DIM,
    LayerKVCache,
)
from layers.mha import mha_decode_step, mha_forward
from layers.qkv import QKVWeights, qkv_forward
from layers.rmsnorm import rmsnorm
from layers.rope import apply_rope
from std.collections import List
from std.math import abs, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)


def _make_dummy_cfg() raises -> ModelConfig:
    """Konfigurasi ringkas dengan dimensi standar Qwen1.5-MoE (hidden=2048, heads=16, dim=128).
    """
    return ModelConfig(2048, 16, 16, 24)


def test_position_contract_invariants() raises:
    """Verifikasi kontrak posisi eksplisit tanpa-ambiguitas dan penolakan label t-1.
    """
    var s = 4
    var n = 4
    var ctx = 10
    var ctx_mgr = DecodeStepContext(s, n, ctx)

    # Inisial: g = 0, p = S = 4
    assert_equal(ctx_mgr.step_idx, 0)
    assert_equal(ctx_mgr.tokens_generated, 0)
    assert_equal(ctx_mgr.current_pos, 4)

    # Invarian lolos saat cache_len_before_step == 4 (S + g)
    ctx_mgr.assert_step_invariants(4)

    # Pelanggaran: cache_len_before_step != S + g wajib gagal
    var failed_len = False
    try:
        ctx_mgr.assert_step_invariants(3)
    except e:
        failed_len = True
        assert_true("cache_len_before_step != S + g" in String(e))
    assert_true(failed_len)

    # Step maju: i=1, g=1, p=5
    ctx_mgr.advance_step()
    assert_equal(ctx_mgr.step_idx, 1)
    assert_equal(ctx_mgr.tokens_generated, 1)
    assert_equal(ctx_mgr.current_pos, 5)
    ctx_mgr.assert_step_invariants(5)

    # Simulasi overflow context: maju hingga melebihi ctx
    var small_ctx = DecodeStepContext(2, 2, 3)
    small_ctx.assert_step_invariants(2)
    small_ctx.advance_step()  # g = 1, p = 3
    # cache_len_before = 3; after = 4 > ctx (3)
    var failed_overflow = False
    try:
        small_ctx.assert_step_invariants(3)
    except e:
        failed_overflow = True
        assert_true("cache_len_after_step > ctx" in String(e))
    assert_true(failed_overflow)


def test_mha_decode_step_vs_recompute_equivalence() raises:
    """Invarian inti: recompute attention at p == incremental attention using cached K/V[0:p) + Q[p].
    """
    var cfg = _make_dummy_cfg()
    var hidden = cfg.hidden_size
    var s_prompt = 3

    # Buat K dan V prompt untuk posisi 0..2
    var k_prompt = List[Float32]()
    var v_prompt = List[Float32]()
    for t in range(s_prompt):
        for k in range(hidden):
            var val_k = Float32(0.1) * Float32((t + 1) * (k % 17 + 1))
            var val_v = Float32(0.05) * Float32((t + 2) * (k % 13 + 1))
            k_prompt.append(val_k)
            v_prompt.append(val_v)

    # Inisialisasi KV cache dan simpan prefill
    var layer_kv = LayerKVCache(16, 0)
    layer_kv.store_prefill(k_prompt, v_prompt, s_prompt)
    assert_equal(layer_kv.current_len, s_prompt)

    # Decode step di posisi p = 3
    var p = 3
    var q_new = List[Float32]()
    var k_new = List[Float32]()
    var v_new = List[Float32]()
    for k in range(hidden):
        q_new.append(Float32(0.2) * Float32(k % 19 + 1))
        k_new.append(Float32(0.1) * Float32((p + 1) * (k % 17 + 1)))
        v_new.append(Float32(0.05) * Float32((p + 2) * (k % 13 + 1)))

    # Path 1: Incremental MHA
    # Append K dan V baru di posisi p
    layer_kv.append_decode_token(k_new, v_new, p)
    assert_equal(layer_kv.current_len, p + 1)

    var out_incremental = mha_decode_step(q_new, layer_kv, p + 1, cfg, 0)
    assert_equal(len(out_incremental), hidden)

    # Path 2: Recompute baseline MHA
    # Susun matriks Q, K, V penuh untuk seluruh sekuens [0..p] (seq_len = 4)
    var seq_len = p + 1
    var q_full = List[Float32]()
    var k_full = List[Float32]()
    var v_full = List[Float32]()

    # Posisi 0..p-1 dummy Q, dan K/V dari prompt yang ter-retrieve dari cache
    # (menguji ekuivalensi numerik persis data dari cache BF16)
    var k_retrieved = layer_kv.get_k_slice(seq_len)
    var v_retrieved = layer_kv.get_v_slice(seq_len)

    for t in range(seq_len):
        if t == p:
            for k in range(hidden):
                q_full.append(q_new[k])
        else:
            for _ in range(hidden):
                q_full.append(Float32(0.0))

        for k in range(hidden):
            k_full.append(k_retrieved[t * hidden + k])
            v_full.append(v_retrieved[t * hidden + k])

    var out_full = mha_forward(q_full, k_full, v_full, seq_len, cfg, 0)

    # Ekstrak token output pada posisi p
    var out_recompute_p = List[Float32]()
    for k in range(hidden):
        out_recompute_p.append(out_full[p * hidden + k])

    # Bandingkan output incremental vs recompute pada posisi p
    # Matematika dot-product dan softmax identik -> toleransi ketat FP32 (atol <= 1e-4)
    for k in range(hidden):
        assert_almost_equal(out_incremental[k], out_recompute_p[k], atol=1e-4)


def test_layer_localization_comparison_harness() raises:
    """Verifikasi harness lokalisasi per-layer dan format pesan failure."""
    var tensor_a = List[Float32]()
    var tensor_b = List[Float32]()
    for i in range(100):
        tensor_a.append(Float32(1.0 + Float32(i) * 0.01))
        tensor_b.append(Float32(1.0 + Float32(i) * 0.01))

    # 1. Tensor cocok -> pass
    compare_tensors_loose(tensor_a, tensor_b, "attn-out", 3)

    # 2. Tensor mismatch -> format error harus persis 'layer L, <kelas> mismatch'
    tensor_a[50] += Float32(0.5)  # induksi selisih
    var mismatch_caught = False
    try:
        compare_tensors_loose(tensor_a, tensor_b, "attn-out", 3)
    except e:
        mismatch_caught = True
        var err_msg = String(e)
        assert_true("layer 3, attn-out mismatch" in err_msg)
    assert_true(mismatch_caught)


def test_f3b_and_f5_formulas() raises:
    """Verifikasi rumus F3b (traffic terdekomposisi) dan F5 (forecast latensi).
    """
    # 1. F3b traffic disk terkunci
    assert_equal(B_TOK_DISK_BYTES, 4133600000)
    assert_equal(KV_LAYER_TOTAL_SLOT_BYTES, 196608)

    # KV read @2048 = 2048 * 192 KiB = 384 MiB
    var b_read_2k = F3bTraffic.compute_b_tok_kv_read(2048)
    assert_equal(b_read_2k, 402653184)

    # KV write konstan = 192 KiB
    var b_write = F3bTraffic.compute_b_tok_kv_write()
    assert_equal(b_write, 196608)

    # Total KV traffic @2048 = 384 MiB + 192 KiB
    var b_kv_tot = F3bTraffic.compute_b_tok_kv_total(2048)
    assert_equal(b_kv_tot, 402653184 + 196608)

    # Total traffic (union disk + KV)
    var b_tot = F3bTraffic.compute_b_tok_total(2048)
    assert_equal(b_tot, 4133600000 + b_kv_tot)

    # 2. F5 forecast model serial
    var bw_ram = Float64(15000000000.0)  # 15 GB/s
    var bw_ssd = Float64(3000000000.0)  # 3 GB/s
    var rho_b = Float64(0.1154)

    var f5 = F5Forecast(2048, rho_b, bw_ram, bw_ssd, 0.05, 0.005)
    # T_data ≈ 1.25 s
    assert_true(f5.t_data > 1.20 and f5.t_data < 1.30)
    # T_kv ≈ 0.0268 s
    assert_true(f5.t_kv > 0.020 and f5.t_kv < 0.035)
    # T_tok ≈ 1.33 s
    assert_true(f5.t_tok > 1.28 and f5.t_tok < 1.38)

    # Gate G-M5-4 acceptance e_T <= 30%
    assert_true(f5.is_within_time_gate(1.35))
    assert_false(f5.is_within_time_gate(2.50))  # Meleset > 30%


def test_forward_attention_decode_step_vs_recompute() raises:
    """Verifikasi pipeline utuh forward attention decode step vs full recompute pada bobot terdefinisi.
    """
    var hidden = 64
    var num_heads = 2
    var cfg = ModelConfig(hidden, num_heads, num_heads, 2)
    var eps = Float32(1e-5)

    # Bobot dummy terdefinisi
    var norm_gamma = List[Float32]()
    for _ in range(hidden):
        norm_gamma.append(Float32(1.0))

    var w_q = List[Float32]()
    var w_k = List[Float32]()
    var w_v = List[Float32]()
    for i in range(hidden * hidden):
        var row = i // hidden
        var col = i % hidden
        if row == col:
            w_q.append(Float32(0.5))
            w_k.append(Float32(0.5))
            w_v.append(Float32(0.5))
        else:
            w_q.append(Float32(0.0))
            w_k.append(Float32(0.0))
            w_v.append(Float32(0.0))

    var b_zero = List[Float32]()
    for _ in range(hidden):
        b_zero.append(Float32(0.0))

    var qkv = QKVWeights(
        w_q^,
        w_k^,
        w_v^,
        b_zero.copy(),
        b_zero.copy(),
        b_zero^,
    )

    var w_o = List[Float32]()
    for i in range(hidden * hidden):
        var row = i // hidden
        var col = i % hidden
        if row == col:
            w_o.append(Float32(0.8))
        else:
            w_o.append(Float32(0.0))
    var b_o = List[Float32]()

    var attn_weights = AttentionWeights(norm_gamma^, qkv^, w_o^, b_o^)

    # Input prompt: S = 2
    var s_prompt = 2
    var p_pos = 2  # Decode di posisi p = 2
    var full_seq = List[Float32]()
    for t in range(s_prompt + 1):  # 3 token
        for k in range(hidden):
            full_seq.append(Float32(0.2 * Float32(t + 1) + 0.01 * Float32(k)))

    # 1. Prefill KV Cache untuk posisi 0..1
    var layer_kv = LayerKVCache(16, 0)
    var prompt_x = List[Float32]()
    for i in range(s_prompt * hidden):
        prompt_x.append(full_seq[i])

    # Jalankan RMSNorm + QKV + RoPE untuk prompt
    var prompt_norm = List[Float32]()
    for t in range(s_prompt):
        var row_vec = List[Float32]()
        for k in range(hidden):
            row_vec.append(prompt_x[t * hidden + k])
        var n_tok = rmsnorm(row_vec, attn_weights.norm_gamma, eps)
        for k in range(hidden):
            prompt_norm.append(n_tok[k])

    var qkv_prompt = qkv_forward(prompt_norm, attn_weights.qkv, s_prompt, cfg)
    var rope_prompt = apply_rope(
        qkv_prompt[0], qkv_prompt[1], s_prompt, cfg, 0, Float32(1000000.0), 0
    )

    # Pad ke SLOT_DIM jika diperlukan oleh LayerKVCache
    var k_prefill_padded = List[Float32]()
    var v_prefill_padded = List[Float32]()
    for t in range(s_prompt):
        for k in range(SLOT_DIM):
            if k < hidden:
                k_prefill_padded.append(rope_prompt[1][t * hidden + k])
                v_prefill_padded.append(qkv_prompt[2][t * hidden + k])
            else:
                k_prefill_padded.append(Float32(0.0))
                v_prefill_padded.append(Float32(0.0))

    layer_kv.store_prefill(k_prefill_padded, v_prefill_padded, s_prompt)
    assert_equal(layer_kv.current_len, s_prompt)

    # 2. Decode step pada posisi p = 2
    var x_tok = List[Float32]()
    for k in range(hidden):
        x_tok.append(full_seq[p_pos * hidden + k])

    # RMSNorm 1 token
    var x_tok_norm = rmsnorm(x_tok, attn_weights.norm_gamma, eps)
    var qkv_tok = qkv_forward(x_tok_norm, attn_weights.qkv, 1, cfg)
    var rope_tok = apply_rope(
        qkv_tok[0], qkv_tok[1], 1, cfg, p_pos, Float32(1000000.0), 0
    )

    # Append ke layer_kv
    var k_tok_pad = List[Float32]()
    var v_tok_pad = List[Float32]()
    for k in range(SLOT_DIM):
        if k < hidden:
            k_tok_pad.append(rope_tok[1][k])
            v_tok_pad.append(qkv_tok[2][k])
        else:
            k_tok_pad.append(Float32(0.0))
            v_tok_pad.append(Float32(0.0))

    layer_kv.append_decode_token(k_tok_pad, v_tok_pad, p_pos)
    assert_equal(layer_kv.current_len, p_pos + 1)

    # Incremental attention
    var attn_inc = mha_decode_step(rope_tok[0], layer_kv, p_pos + 1, cfg, 0)
    var y_inc = o_project(
        attn_inc, attn_weights.w_o, attn_weights.b_o, 1, hidden, 0
    )

    # 3. Full recompute
    var y_recompute = recompute_attention_at_position(
        full_seq, attn_weights, p_pos, cfg, eps, Float32(1000000.0), 0
    )

    # 4. Bandingkan via compare_tensors_loose
    compare_tensors_loose(y_inc, y_recompute, "attn-out", 0)


def test_hard_fail_category_rejection() raises:
    """Verifikasi penolakan kategori hard fail: threshold ketat menolak penyimpangan.
    """
    var a = List[Float32]()
    var b = List[Float32]()
    for i in range(50):
        a.append(Float32(1.0 + Float32(i) * 0.1))
        # Simulasi penyimpangan RoPE atau bias (perbedaan signifikan)
        b.append(Float32(1.0 + Float32(i) * 0.1) + Float32(0.15))

    var failed = False
    try:
        compare_tensors_loose(a, b, "Q", 1)
    except e:
        failed = True
        var err_msg = String(e)
        assert_true("layer 1, Q mismatch" in err_msg)
    assert_true(failed)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_position_contract_invariants]()
    suite.test[test_mha_decode_step_vs_recompute_equivalence]()
    suite.test[test_layer_localization_comparison_harness]()
    suite.test[test_f3b_and_f5_formulas]()
    suite.test[test_forward_attention_decode_step_vs_recompute]()
    suite.test[test_hard_fail_category_rejection]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
