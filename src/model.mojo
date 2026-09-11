# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Model kernels — config, rmsnorm, head path, rope, attn, moe (M1-M3).

@see scratch/wave/m1/m1-w1-rmsnorm.md
@see scratch/wave/m1/m1-w2-embed-lmhead.md
@see scratch/wave/m2/m2-w1-qkv-bias.md
@see docs/milestones/M2-attention.md
"""

from safetensors import read_header, STHeader, TensorMeta
from std.builtin.dtype import DType
from std.collections import List
from std.math import min, sqrt, isnan, isinf, isfinite
from std.memory import Pointer
from std.os import SEEK_SET
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
struct LoadMemoryTelemetry(Copyable, Movable):
    """Telemetri memori fase load untuk memastikan strategi chunked (G-M1-2)."""

    var resident_target_bytes: Int
    var conversion_buffer_bytes: Int
    var source_buffer_bytes: Int
    var vmhwm_bytes: Int

    def __init__(out self):
        self.resident_target_bytes = 0
        self.conversion_buffer_bytes = 0
        self.source_buffer_bytes = 0
        self.vmhwm_bytes = 0


comptime CHUNK_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB (<= 64 MiB batas keras normatif)


def load_tensor_f32_chunked(
    shard_path: String,
    data_base: Int,
    meta: TensorMeta,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    """Pemuatan tensor chunked BF16/F32 -> F32 resident tanpa double-residency.

    @spec m1-w2-embed-lmhead.md (§ Anggaran memori M1)
    """
    var total_bytes = meta.end - meta.begin
    var num_elements: Int
    var element_size: Int
    if meta.dtype == "BF16":
        element_size = 2
        num_elements = total_bytes // 2
    elif meta.dtype == "F32":
        element_size = 4
        num_elements = total_bytes // 4
    else:
        raise Error(
            '{"error_type":"UNKNOWN_DTYPE","detail":"unsupported dtype: '
            + meta.dtype
            + '","shard":"'
            + shard_path
            + '","tensor_name":"'
            + meta.name
            + '"}'
        )

    var out = List[Float32]()
    out.reserve(num_elements)
    telemetry.resident_target_bytes += num_elements * 4

    var f = open(shard_path, "r")
    _ = f.seek(data_base + meta.begin, SEEK_SET)

    var bytes_remaining = total_bytes
    while bytes_remaining > 0:
        var to_read = min(bytes_remaining, CHUNK_MAX_BYTES)
        to_read = (to_read // element_size) * element_size
        if to_read == 0:
            to_read = bytes_remaining

        var chunk_bytes = f.read_bytes(to_read)
        if len(chunk_bytes) < to_read:
            f.close()
            raise Error(
                '{"error_type":"INVALID_HEADER","detail":"truncated tensor'
                ' data","shard":"'
                + shard_path
                + '","tensor_name":"'
                + meta.name
                + '"}'
            )

        if len(chunk_bytes) > telemetry.source_buffer_bytes:
            telemetry.source_buffer_bytes = len(chunk_bytes)

        var chunk_elements = len(chunk_bytes) // element_size
        var conv_bytes = chunk_elements * 4
        if conv_bytes > telemetry.conversion_buffer_bytes:
            telemetry.conversion_buffer_bytes = conv_bytes

        if meta.dtype == "BF16":
            var p_u8 = chunk_bytes.unsafe_ptr()
            var p_bf = p_u8.unsafe_bitcast[Scalar[DType.bfloat16]]()
            for i in range(chunk_elements):
                out.append(p_bf[unsafe_offset=i].cast[DType.float32]())
        else:
            var p_u8 = chunk_bytes.unsafe_ptr()
            var p_f32 = p_u8.unsafe_bitcast[Scalar[DType.float32]]()
            for i in range(chunk_elements):
                out.append(p_f32[unsafe_offset=i])

        bytes_remaining -= len(chunk_bytes)

    f.close()
    return out^


@fieldwise_init
struct HeadWeights(Copyable, Movable):
    """Bobot head path model (embedding, final norm, lm_head untied)."""

    var embed_tokens: List[Float32]
    var norm_weight: List[Float32]
    var lm_head: List[Float32]

    def is_untied(self) -> Bool:
        """Verifikasi bahwa lm_head bukan alias pointer dari embed_tokens."""
        return self.embed_tokens.unsafe_ptr() != self.lm_head.unsafe_ptr()


def embedding_lookup(
    token_ids: List[Int],
    embed_table: List[Float32],
    vocab_size: Int,
    hidden_size: Int,
) raises -> List[Float32]:
    """Lookup embedding token ID -> vektor baris F32.

    @spec m1-w2-embed-lmhead.md
    """
    var num_tokens = len(token_ids)
    var out = List[Float32]()
    out.reserve(num_tokens * hidden_size)
    for t in range(num_tokens):
        var tid = token_ids[t]
        if tid < 0 or tid >= vocab_size:
            raise Error(
                '{"error_type":"TOKEN_INVALID","detail":"Token ID '
                + String(tid)
                + " exceeds vocab size "
                + String(vocab_size)
                + '","stage":"embedding","token_id":'
                + String(tid)
                + "}"
            )
        var offset = tid * hidden_size
        for h in range(hidden_size):
            out.append(embed_table[offset + h])
    return out^


def matmul_activation_head(
    activation: List[Float32],
    head: List[Float32],
    num_tokens: Int,
    vocab_size: Int,
    hidden_size: Int,
) -> List[Float32]:
    """Perkalian matriks aktivasi [num_tokens, d] x head^T [d, V] -> logits [num_tokens, V].

    @spec m1-w2-embed-lmhead.md
    """
    var logits = List[Float32]()
    logits.reserve(num_tokens * vocab_size)
    var p_act = activation.unsafe_ptr()
    var p_head = head.unsafe_ptr()
    for t in range(num_tokens):
        var act_row = t * hidden_size
        for v in range(vocab_size):
            var head_row = v * hidden_size
            var acc = Float32(0.0)
            for k in range(hidden_size):
                acc += (
                    p_act[unsafe_offset=act_row + k]
                    * p_head[unsafe_offset=head_row + k]
                )
            logits.append(acc)
    return logits^


def forward_head(
    token_ids: List[Int],
    weights: HeadWeights,
    cfg: ModelConfig,
    eps: Float32,
) raises -> List[Float32]:
    """Alur lengkap M1: embedding lookup -> final RMSNorm F6 -> lm_head matmul.
    """
    var num_tokens = len(token_ids)
    var hidden = cfg.hidden_size
    var vocab = cfg.vocab_size

    # 1. Embedding lookup
    var embed_act = embedding_lookup(
        token_ids, weights.embed_tokens, vocab, hidden
    )

    # 2. Final RMSNorm per token
    var normed_act = List[Float32]()
    normed_act.reserve(num_tokens * hidden)
    for t in range(num_tokens):
        var tok_vec = List[Float32]()
        tok_vec.reserve(hidden)
        var row = t * hidden
        for h in range(hidden):
            tok_vec.append(embed_act[row + h])
        var normed = rmsnorm(tok_vec, weights.norm_weight, eps)
        for h in range(hidden):
            normed_act.append(normed[h])

    # 3. LM Head Matmul
    var logits = matmul_activation_head(
        normed_act, weights.lm_head, num_tokens, vocab, hidden
    )
    return logits^


def validate_logits(
    logits: List[Float32],
    num_prompts: Int,
    tokens_per_prompt: Int,
    vocab_size: Int,
) raises:
    """Validasi format biner logits: shape [num_prompts, tokens_per_prompt, V], finite semua.

    @spec m1-w2-embed-lmhead.md (§ Logits File Format)
    """
    var total_tokens = num_prompts * tokens_per_prompt
    var expected_len = total_tokens * vocab_size
    if len(logits) != expected_len:
        raise Error(
            '{"error_type":"OUTPUT_WRITE_FAILED","detail":"logits length'
            " mismatch: expected "
            + String(expected_len)
            + " got "
            + String(len(logits))
            + '","stage":"output"}'
        )
    for i in range(len(logits)):
        var val = logits[i]
        if isnan(val) or isinf(val):
            raise Error(
                '{"error_type":"NORM_ERROR","detail":"non-finite value in'
                " logits at index "
                + String(i)
                + '","stage":"output"}'
            )


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


def test_embedding_lookup_valid() raises:
    """Lookup baris embedding menghasilkan vektor yang benar."""
    var vocab = 3
    var hidden = 2
    var table: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var ids: List[Int] = [2, 0, 1]
    var out = embedding_lookup(ids, table, vocab, hidden)
    assert_equal(len(out), 6)
    assert_almost_equal(out[0], Float32(5.0))
    assert_almost_equal(out[1], Float32(6.0))
    assert_almost_equal(out[2], Float32(1.0))
    assert_almost_equal(out[3], Float32(2.0))
    assert_almost_equal(out[4], Float32(3.0))
    assert_almost_equal(out[5], Float32(4.0))


def test_embedding_lookup_out_of_bounds() raises:
    """Token id di luar rentang [0, vocab_size-1] -> TOKEN_INVALID."""
    var table: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var bad_ids: List[Int] = [2]
    var raised = False
    try:
        var _out = embedding_lookup(bad_ids, table, 2, 2)
    except e:
        raised = True
    assert_true(raised)

    var neg_ids: List[Int] = [-1]
    var raised_neg = False
    try:
        var _out2 = embedding_lookup(neg_ids, table, 2, 2)
    except e:
        raised_neg = True
    assert_true(raised_neg)


def test_matmul_activation_head_known() raises:
    """Perkalian aktivasi [2, 2] x head^T [3, 2]^T -> logits [2, 3]."""
    var act: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var head: List[Float32] = [
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    var logits = matmul_activation_head(act, head, 2, 3, 2)
    assert_equal(len(logits), 6)
    assert_almost_equal(logits[0], Float32(1.0), atol=1e-5)
    assert_almost_equal(logits[1], Float32(2.0), atol=1e-5)
    assert_almost_equal(logits[2], Float32(3.0), atol=1e-5)
    assert_almost_equal(logits[3], Float32(3.0), atol=1e-5)
    assert_almost_equal(logits[4], Float32(4.0), atol=1e-5)
    assert_almost_equal(logits[5], Float32(7.0), atol=1e-5)


def test_head_weights_untied() raises:
    """Verifikasi struktur HeadWeights memisahkan alokasi embed dan lm_head."""
    var embed: List[Float32] = [1.0, 2.0]
    var norm: List[Float32] = [1.0, 1.0]
    var head: List[Float32] = [1.0, 2.0]
    var hw = HeadWeights(embed^, norm^, head^)
    assert_true(hw.is_untied())


def test_load_tensor_chunked() raises:
    """Pemuatan tensor chunked dari fixture m0 & verifikasi telemetri memori."""
    var path = "fixtures/m0/fixture-00001-of-00003.safetensors"
    var st = read_header(path)
    var telem = LoadMemoryTelemetry()
    var found = False
    for i in range(len(st.entries)):
        ref e = st.entries[i]
        if e.name == "model.embed_tokens.weight":
            found = True
            var t = load_tensor_f32_chunked(st.shard, st.data_base, e, telem)
            assert_equal(len(t), 512 * 64)
            assert_almost_equal(t[0], Float32(1.0189883e35), rtol=1e-5)
            assert_true(telem.source_buffer_bytes <= 64 * 1024 * 1024)
            assert_true(telem.conversion_buffer_bytes <= 64 * 1024 * 1024)
            assert_equal(telem.resident_target_bytes, 512 * 64 * 4)
    assert_true(found)


def test_forward_head_pipeline() raises:
    """Pipeline forward head lengkap: token -> lookup -> norm -> matmul."""
    var cfg = ModelConfig(64, 2, 2, 512)
    var eps = Float32(1e-6)

    # Inisialisasi bobot numerik valid terdefinisi (skala kecil agar stabil)
    var embed = List[Float32]()
    embed.reserve(cfg.vocab_size * cfg.hidden_size)
    for i in range(cfg.vocab_size * cfg.hidden_size):
        embed.append(Float32(0.01) * Float32((i % 17) - 8))

    var norm = List[Float32]()
    norm.reserve(cfg.hidden_size)
    for _ in range(cfg.hidden_size):
        norm.append(Float32(1.0))

    var head = List[Float32]()
    head.reserve(cfg.vocab_size * cfg.hidden_size)
    for k in range(cfg.vocab_size * cfg.hidden_size):
        head.append(Float32(0.01) * Float32((k % 13) - 6))

    var weights = HeadWeights(embed^, norm^, head^)
    assert_true(weights.is_untied())

    # Uji 48 token (3 prompt x 16 token)
    var tokens = List[Int]()
    for p in range(3):
        for t in range(16):
            tokens.append((p * 16 + t) % cfg.vocab_size)

    var logits = forward_head(tokens, weights, cfg, eps)
    validate_logits(logits, 3, 16, cfg.vocab_size)
    assert_equal(len(logits), 3 * 16 * cfg.vocab_size)


def test_validate_logits_ok_and_fails() raises:
    """Validasi format logits (dimensi dan deteksi finite)."""
    var valid = List[Float32]()
    for _ in range(48 * 4):
        valid.append(Float32(0.5))
    validate_logits(valid, 3, 16, 4)

    var invalid_len = List[Float32]()
    invalid_len.append(Float32(1.0))
    var raised_len = False
    try:
        validate_logits(invalid_len, 3, 16, 4)
    except e:
        raised_len = True
    assert_true(raised_len)

    var nan_logits = List[Float32]()
    for _ in range(48 * 4):
        nan_logits.append(Float32(0.5))
    nan_logits[10] = Float32(0.0) / Float32(0.0)
    var raised_nan = False
    try:
        validate_logits(nan_logits, 3, 16, 4)
    except e:
        raised_nan = True
    assert_true(raised_nan)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
