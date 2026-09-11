# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Model kernels — config, rmsnorm, rope, attn, moe (menyusul M1-M3).

@see scratch/wave/m1/m1-w1-rmsnorm.md
@see scratch/wave/m2/m2-w1-qkv-bias.md
@see docs/milestones/M2-attention.md
"""

from std.collections import List
from std.math import sqrt
from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_almost_equal,
    TestSuite,
)


@fieldwise_init
struct ModelConfig(Copyable, Movable):
    """Dimensi arsitektur Qwen2 MoE dari config.json."""

    var hidden_size: Int
    var num_hidden_layers: Int
    var num_attention_heads: Int
    var vocab_size: Int

    def head_dim(self) -> Int:
        return self.hidden_size // self.num_attention_heads


def rmsnorm(
    x: List[Float32],
    gamma: List[Float32],
    eps: Float32,
) raises -> List[Float32]:
    """RMSNorm F6: y = x / sqrt(mean(x^2) + eps) ⊙ gamma, fp32.

    @spec scratch/wave/m1/m1-w1-rmsnorm.md (F6)
    Kontrak eps: parameter wajib tanpa default diam-diam. Sumber kanonis =
    field `rms_norm_eps` di `fixtures/m1/model_config.json` (artefak W4);
    kernel ini menerima nilainya sebagai argumen. eps <= 0 → CONFIG_ERROR;
    input kosong / panjang mismatch → NORM_ERROR.
    """
    var n = len(x)
    if n == 0:
        raise Error(
            '{"error_type":"NORM_ERROR","detail":"empty'
            ' input","stage":"rmsnorm"}'
        )
    if len(gamma) != n:
        raise Error(
            '{"error_type":"NORM_ERROR","detail":"length'
            ' mismatch","stage":"rmsnorm"}'
        )
    if eps <= Float32(0.0):
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"rms_norm_eps must be > 0",'
            '"stage":"config"}'
        )
    var acc = Float32(0.0)
    for i in range(n):
        acc += x[i] * x[i]
    var denom = sqrt(acc / Float32(n) + eps)
    var out = List[Float32]()
    for i in range(n):
        out.append(x[i] / denom * gamma[i])
    return out^


@fieldwise_init
struct QKVWeights(Copyable, Movable):
    """Bobot QKV satu layer: W_q, W_k, W_v [hidden, hidden] + bias [hidden]."""

    var w_q: List[Float32]
    var w_k: List[Float32]
    var w_v: List[Float32]
    var b_q: List[Float32]
    var b_k: List[Float32]
    var b_v: List[Float32]


def qkv_project(
    x: List[Float32],
    w: List[Float32],
    b: List[Float32],
    seq_len: Int,
    hidden: Int,
) raises -> List[Float32]:
    """Proyeksi linear y = xW^T + b untuk satu matriks Q/K/V.

    @spec m2-w1-qkv-bias.md
    """
    var out = List[Float32]()
    for _ in range(seq_len * hidden):
        out.append(Float32(0.0))

    for t in range(seq_len):
        for j in range(hidden):
            var acc = Float32(0.0)
            for k in range(hidden):
                acc += x[t * hidden + k] * w[j * hidden + k]
            out[t * hidden + j] = acc + b[j]
    return out^


def qkv_forward(
    x: List[Float32],
    weights: QKVWeights,
    seq_len: Int,
    cfg: ModelConfig,
) raises -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """Hitung Q, K, V dari input x menggunakan bobot QKV satu layer."""
    var hidden = cfg.hidden_size
    var q = qkv_project(x, weights.w_q, weights.b_q, seq_len, hidden)
    var k = qkv_project(x, weights.w_k, weights.b_k, seq_len, hidden)
    var v = qkv_project(x, weights.w_v, weights.b_v, seq_len, hidden)
    return (q^, k^, v^)


def validate_bias_count(
    bias_names: List[String],
    num_layers: Int,
    shard: String,
) raises:
    """Validasi P-2: jumlah tensor bias attention == 3 * num_layers.

    @spec ref-ground-truth P-2
    """
    var expected = 3 * num_layers
    var actual = len(bias_names)
    if actual != expected:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"bias count mismatch:'
            " expected "
            + String(expected)
            + " got "
            + String(actual)
            + '","shard":"'
            + shard
            + '","tensor_name":""}'
        )


def collect_attention_bias_names(
    tensor_names: List[String],
    num_layers: Int,
) -> List[String]:
    """Kumpulkan nama tensor bias attention dari daftar semua tensor di index.
    """
    var result = List[String]()
    for i in range(len(tensor_names)):
        var name = tensor_names[i]
        var nb = name.as_bytes()
        if len(nb) >= 5:
            var is_bias = (
                Int(nb[len(nb) - 5]) == 46
                and Int(nb[len(nb) - 4]) == 98
                and Int(nb[len(nb) - 3]) == 105
                and Int(nb[len(nb) - 2]) == 97
                and Int(nb[len(nb) - 1]) == 115
            )
            if is_bias:
                var has_attn = _contains(name, "self_attn")
                var has_proj = (
                    _contains(name, "q_proj")
                    or _contains(name, "k_proj")
                    or _contains(name, "v_proj")
                )
                if has_attn and has_proj:
                    result.append(name)
    return result^


def _contains(s: String, sub: String) -> Bool:
    """Cek apakah string s mengandung substring sub."""
    var sb = s.as_bytes()
    var ub = sub.as_bytes()
    var slen = len(sb)
    var ulen = len(ub)
    if ulen > slen:
        return False
    for i in range(slen - ulen + 1):
        var found: Bool = True
        for j in range(ulen):
            if sb[i + j] != ub[j]:
                found = False
                break
        if found:
            return True
    return False


# === Tests ===


def test_qkv_project_identity() raises:
    """Proyeksi dengan weight=identity, bias=0 -> output == input."""
    var hidden = 4
    var seq_len = 2
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var w: List[Float32] = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    var b: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    var out = qkv_project(x, w, b, seq_len, hidden)
    for i in range(seq_len * hidden):
        assert_almost_equal(out[i], x[i], atol=1e-6)


def test_qkv_project_with_bias() raises:
    """Proyeksi identity + bias konstan -> output = input + bias."""
    var hidden = 4
    var seq_len = 1
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var w: List[Float32] = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    var b: List[Float32] = [0.5, 0.5, 0.5, 0.5]
    var out = qkv_project(x, w, b, seq_len, hidden)
    assert_almost_equal(out[0], 1.5, atol=1e-6)
    assert_almost_equal(out[1], 2.5, atol=1e-6)
    assert_almost_equal(out[2], 3.5, atol=1e-6)
    assert_almost_equal(out[3], 4.5, atol=1e-6)


def test_qkv_project_matmul() raises:
    """Proyeksi dengan weight non-trivial, verifikasi manual."""
    var hidden = 2
    var seq_len = 1
    var x: List[Float32] = [1.0, 2.0]
    var w: List[Float32] = [3.0, 4.0, 5.0, 6.0]
    var b: List[Float32] = [0.1, 0.2]
    var out = qkv_project(x, w, b, seq_len, hidden)
    assert_almost_equal(out[0], 11.1, atol=1e-5)
    assert_almost_equal(out[1], 17.2, atol=1e-5)


def test_validate_bias_count_ok() raises:
    """72 bias names untuk 24 layers -> tidak raise."""
    var names: List[String] = []
    for i in range(24):
        names.append("model.layers." + String(i) + ".self_attn.q_proj.bias")
        names.append("model.layers." + String(i) + ".self_attn.k_proj.bias")
        names.append("model.layers." + String(i) + ".self_attn.v_proj.bias")
    validate_bias_count(names, 24, "test.safetensors")


def test_validate_bias_count_fail() raises:
    """Jumlah bias salah -> raise WEIGHT_LOAD_FAILED."""
    var names: List[String] = ["a", "b"]
    var raised: Bool = False
    try:
        validate_bias_count(names, 24, "test.safetensors")
    except e:
        raised = True
    assert_true(raised)


def test_collect_attention_bias_names() raises:
    """Filter nama tensor bias attention dari daftar campuran."""
    var names: List[String] = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.bias",
        "model.embed_tokens.weight",
    ]
    var result = collect_attention_bias_names(names, 2)
    assert_equal(len(result), 4)


def test_contains() raises:
    assert_true(_contains("hello world", "world"))
    assert_true(_contains("abc", "abc"))
    assert_false(_contains("abc", "xyz"))
    assert_false(_contains("ab", "abc"))


def test_model_config_head_dim() raises:
    var cfg = ModelConfig(2048, 24, 16, 151936)
    assert_equal(cfg.head_dim(), 128)


def test_rmsnorm_known_vector() raises:
    """Oracle sebaris fp32: x=[1,2,3,4], gamma=1, eps=1e-6.

    Nilai harapan dari komputasi fp32 independen (emulasi float32 murni,
    skrip di log kerja W1; oracle PyTorch normatif menyusul W4).
    Provenance eps: default `Qwen2MoeConfig.rms_norm_eps` = 1e-6 (docs
    transformers Qwen2MoE, mencakup Qwen1.5-MoE-A2.7B); TBM → measured:
    verifikasi silang ke config checkpoint + kode modeling dicatat W4
    via `fixtures/m1/model_config.json` kanonis.
    """
    var eps = Float32(1e-6)
    print("rms_norm_eps =", eps)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var g: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var y = rmsnorm(x, g, eps)
    assert_almost_equal(y[0], Float32(0.365148365), atol=1e-6)
    assert_almost_equal(y[1], Float32(0.730296731), atol=1e-6)
    assert_almost_equal(y[2], Float32(1.095445037), atol=1e-6)
    assert_almost_equal(y[3], Float32(1.460593462), atol=1e-6)


def test_rmsnorm_gamma_scales() raises:
    """Perkalian gamma: y = normalized ⊙ gamma."""
    var eps = Float32(1e-6)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var g: List[Float32] = [2.0, 0.5, 1.0, 3.0]
    var y = rmsnorm(x, g, eps)
    assert_almost_equal(y[0], Float32(0.730296731), atol=1e-6)
    assert_almost_equal(y[1], Float32(0.365148365), atol=1e-6)
    assert_almost_equal(y[2], Float32(1.095445037), atol=1e-6)
    assert_almost_equal(y[3], Float32(4.381780148), atol=1e-6)


def test_rmsnorm_scale_invariant() raises:
    """Invariant F6: rmsnorm(c*x) == rmsnorm(x) untuk c > 0, gamma = 1."""
    var eps = Float32(1e-6)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var xs: List[Float32] = [2.0, 4.0, 6.0, 8.0]
    var g: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var y1 = rmsnorm(x, g, eps)
    var y2 = rmsnorm(xs, g, eps)
    for i in range(len(y1)):
        assert_almost_equal(y1[i], y2[i], atol=1e-6)


def test_rmsnorm_gamma_one_normalized() raises:
    """Invariant F6: gamma = 1 → RMS(output) ≈ 1."""
    var eps = Float32(1e-6)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var g: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var y = rmsnorm(x, g, eps)
    var acc = Float32(0.0)
    for i in range(len(y)):
        acc += y[i] * y[i]
    var rms = sqrt(acc / Float32(len(y)))
    assert_almost_equal(rms, Float32(1.0), atol=1e-6)


def test_rmsnorm_eps_rejected() raises:
    """Kontrak eps: 0 / negatif → CONFIG_ERROR, tanpa default diam-diam."""
    var x: List[Float32] = [1.0, 2.0]
    var g: List[Float32] = [1.0, 1.0]
    var bad: List[Float32] = [Float32(0.0), Float32(-1e-6)]
    for i in range(len(bad)):
        var raised = False
        try:
            var _y = rmsnorm(x, g, bad[i])
        except e:
            raised = True
        assert_true(raised)


def test_rmsnorm_shape_errors() raises:
    """Input kosong / panjang gamma mismatch → NORM_ERROR."""
    var x: List[Float32] = [1.0, 2.0]
    var g_short: List[Float32] = [1.0]
    var empty: List[Float32] = []
    var raised_mismatch = False
    try:
        var _y = rmsnorm(x, g_short, Float32(1e-6))
    except e:
        raised_mismatch = True
    assert_true(raised_mismatch)
    var raised_empty = False
    try:
        var _z = rmsnorm(empty, empty, Float32(1e-6))
    except e:
        raised_empty = True
    assert_true(raised_empty)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
