# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Model kernels — config, rmsnorm, head path, rope, attn, moe (M1-M3).

@see scratch/wave/m1/m1-w1-rmsnorm.md
@see scratch/wave/m1/m1-w2-embed-lmhead.md
@see scratch/wave/m2/m2-w1-qkv-bias.md
@see docs/milestones/M2-attention.md
"""

from safetensors import (
    STHeader,
    TensorMeta,
    parse_index,
    parse_index_to_dict,
    read_header,
    read_small_file,
)
from std.builtin.dtype import DType
from std.collections import Dict, List
from std.math import (
    abs,
    cos,
    exp,
    isfinite,
    isinf,
    isnan,
    max,
    min,
    pow,
    sin,
    sqrt,
)
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


@fieldwise_init
struct AttentionWeights(Copyable, Movable):
    """Bobot attention satu layer: norm_gamma, QKV (w+b), w_o, b_o."""

    var norm_gamma: List[Float32]
    var qkv: QKVWeights
    var w_o: List[Float32]
    var b_o: List[Float32]


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
    if seq_len <= 0 or hidden <= 0:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"seq_len and hidden must'
            ' be positive","stage":"qkv"}'
        )
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"activation length'
            " mismatch: expected "
            + String(seq_len * hidden)
            + " got "
            + String(len(x))
            + '","stage":"qkv"}'
        )
    if len(w) != hidden * hidden:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"weight length'
            " mismatch: expected "
            + String(hidden * hidden)
            + " got "
            + String(len(w))
            + '","stage":"qkv"}'
        )
    if len(b) != hidden:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"bias length'
            " mismatch: expected "
            + String(hidden)
            + " got "
            + String(len(b))
            + '","stage":"qkv"}'
        )

    for i in range(len(x)):
        if isnan(x[i]) or isinf(x[i]):
            raise Error(
                '{"error_type":"ACT_LOAD_FAILED","detail":"non-finite value in'
                ' activation","stage":"qkv"}'
            )
    for i in range(len(w)):
        if isnan(w[i]) or isinf(w[i]):
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"non-finite value'
                ' in weight","stage":"qkv"}'
            )
    for i in range(len(b)):
        if isnan(b[i]) or isinf(b[i]):
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"non-finite value'
                ' in bias","stage":"qkv"}'
            )

    var out = List[Float32]()
    out.reserve(seq_len * hidden)

    var p_x = x.unsafe_ptr()
    var p_w = w.unsafe_ptr()
    var p_b = b.unsafe_ptr()

    for t in range(seq_len):
        var x_row = t * hidden
        for j in range(hidden):
            var w_row = j * hidden
            var acc = Float32(0.0)
            for k in range(hidden):
                acc += (
                    p_x[unsafe_offset=x_row + k] * p_w[unsafe_offset=w_row + k]
                )
            var val = acc + p_b[unsafe_offset=j]
            if isnan(val) or isinf(val):
                raise Error(
                    '{"error_type":"ATTENTION_ERROR","detail":"non-finite'
                    ' value in qkv projection","stage":"qkv"}'
                )
            out.append(val)
    return out^


def qkv_forward(
    x: List[Float32],
    weights: QKVWeights,
    seq_len: Int,
    cfg: ModelConfig,
) raises -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """Hitung Q, K, V dari input x menggunakan bobot QKV satu layer."""
    var hidden = cfg.hidden_size
    if cfg.num_attention_heads * cfg.head_dim() != hidden:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"num_attention_heads *'
            ' head_dim != hidden_size","stage":"qkv"}'
        )
    var q = qkv_project(x, weights.w_q, weights.b_q, seq_len, hidden)
    var k = qkv_project(x, weights.w_k, weights.b_k, seq_len, hidden)
    var v = qkv_project(x, weights.w_v, weights.b_v, seq_len, hidden)
    return (q^, k^, v^)


def verify_rope_isometry(
    orig: List[Float32],
    rotated: List[Float32],
    seq_len: Int,
    cfg: ModelConfig,
    layer_idx: Int = 0,
    tol: Float32 = Float32(1e-4),
) raises:
    """Verifikasi invariant isometri F7: ||q'||_2 == ||q||_2 per head dan per token.

    Pelanggaran isometri atau non-finite langsung raise ROPE_ERROR (P-3).
    @spec scratch/wave/m2/m2-w2-rope.md
    @spec docs/milestones/M2-attention.md (§ Rumus F7)
    """
    if len(orig) != len(rotated):
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"orig and rotated length'
            ' mismatch","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    var hidden = cfg.hidden_size
    var num_heads = cfg.num_attention_heads
    var head_dim = cfg.head_dim()
    if len(orig) != seq_len * hidden:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"length mismatch with'
            ' seq_len*hidden","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    var p_orig = orig.unsafe_ptr()
    var p_rot = rotated.unsafe_ptr()
    for t in range(seq_len):
        for h in range(num_heads):
            var head_start = t * hidden + h * head_dim
            var orig_norm_sq = Float32(0.0)
            var rot_norm_sq = Float32(0.0)
            for k in range(head_dim):
                var val_orig = p_orig[unsafe_offset=head_start + k]
                var val_rot = p_rot[unsafe_offset=head_start + k]
                if (
                    isnan(val_orig)
                    or isinf(val_orig)
                    or isnan(val_rot)
                    or isinf(val_rot)
                ):
                    raise Error(
                        '{"error_type":"ROPE_ERROR","detail":"non-finite value'
                        ' detected in isometry check","stage":"rope","layer":'
                        + String(layer_idx)
                        + "}"
                    )
                orig_norm_sq += val_orig * val_orig
                rot_norm_sq += val_rot * val_rot
            var norm_orig = sqrt(orig_norm_sq)
            var norm_rot = sqrt(rot_norm_sq)
            var diff = abs(norm_orig - norm_rot)
            var max_norm = max(Float32(1.0), norm_orig)
            if diff > tol * max_norm:
                raise Error(
                    '{"error_type":"ROPE_ERROR","detail":"RoPE invariant'
                    " violation: ||q'|| != ||q|| (diff="
                    + String(diff)
                    + ')","stage":"rope","layer":'
                    + String(layer_idx)
                    + "}"
                )


def rope_rotate_half(
    x: List[Float32],
    seq_len: Int,
    cfg: ModelConfig,
    pos_offset: Int = 0,
    base: Float32 = Float32(1000000.0),
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """RoPE rotate_half kernel F7 dengan verifikasi invariant isometri.

    Rumus F7: theta_i = m * omega_i, omega_i = base^(-2i / d_h)
    Rotasi 2x2 per pasangan (q_i, q_{i + d_h / 2}).
    @spec docs/milestones/M2-attention.md (§ Rumus F7)
    @spec scratch/wave/m2/m2-w2-rope.md
    """
    if seq_len <= 0:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"seq_len must be'
            ' positive","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    if cfg.hidden_size <= 0 or cfg.num_attention_heads <= 0:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"hidden_size and'
            ' num_attention_heads must be positive","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    var hidden = cfg.hidden_size
    var head_dim = cfg.head_dim()
    if cfg.num_attention_heads * head_dim != hidden:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"num_attention_heads *'
            ' head_dim != hidden_size","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    if head_dim % 2 != 0:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"head_dim must be'
            ' even","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"input length mismatch:'
            " expected "
            + String(seq_len * hidden)
            + " got "
            + String(len(x))
            + '","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )
    if base <= Float32(0.0):
        raise Error(
            '{"error_type":"ROPE_ERROR","detail":"base must be'
            ' positive","stage":"rope","layer":'
            + String(layer_idx)
            + "}"
        )

    for i in range(len(x)):
        if isnan(x[i]) or isinf(x[i]):
            raise Error(
                '{"error_type":"ROPE_ERROR","detail":"non-finite value in'
                ' input","stage":"rope","layer":'
                + String(layer_idx)
                + "}"
            )

    var half_dim = head_dim // 2
    var inv_freq = List[Float32]()
    inv_freq.reserve(half_dim)
    for i in range(half_dim):
        var exp_val = Float32(2 * i) / Float32(head_dim)
        inv_freq.append(Float32(1.0) / pow(base, exp_val))

    var out = List[Float32]()
    out.resize(seq_len * hidden, Float32(0.0))

    var p_x = x.unsafe_ptr()
    var p_out = out.unsafe_ptr()
    var num_heads = cfg.num_attention_heads

    for t in range(seq_len):
        var m = Float32(pos_offset + t)
        var cos_m = List[Float32]()
        var sin_m = List[Float32]()
        cos_m.reserve(half_dim)
        sin_m.reserve(half_dim)
        for i in range(half_dim):
            var theta = m * inv_freq[i]
            cos_m.append(cos(theta))
            sin_m.append(sin(theta))
        var p_cos = cos_m.unsafe_ptr()
        var p_sin = sin_m.unsafe_ptr()

        for h in range(num_heads):
            var head_start = t * hidden + h * head_dim
            for i in range(half_dim):
                var x1 = p_x[unsafe_offset=head_start + i]
                var x2 = p_x[unsafe_offset=head_start + half_dim + i]
                var c = p_cos[unsafe_offset=i]
                var s = p_sin[unsafe_offset=i]
                var rot1 = x1 * c - x2 * s
                var rot2 = x1 * s + x2 * c
                if isnan(rot1) or isinf(rot1) or isnan(rot2) or isinf(rot2):
                    raise Error(
                        '{"error_type":"ROPE_ERROR","detail":"non-finite value'
                        ' in output","stage":"rope","layer":'
                        + String(layer_idx)
                        + "}"
                    )
                p_out[unsafe_offset=head_start + i] = rot1
                p_out[unsafe_offset=head_start + half_dim + i] = rot2

    verify_rope_isometry(x, out, seq_len, cfg, layer_idx)
    return out^


def apply_rope(
    q: List[Float32],
    k: List[Float32],
    seq_len: Int,
    cfg: ModelConfig,
    pos_offset: Int = 0,
    base: Float32 = Float32(1000000.0),
    layer_idx: Int = 0,
) raises -> Tuple[List[Float32], List[Float32]]:
    """Terapkan RoPE rotate_half F7 pada Q dan K."""
    var q_rot = rope_rotate_half(q, seq_len, cfg, pos_offset, base, layer_idx)
    var k_rot = rope_rotate_half(k, seq_len, cfg, pos_offset, base, layer_idx)
    return (q_rot^, k_rot^)


def build_causal_mask(seq_len: Int) raises -> List[Float32]:
    """Bangun matriks causal mask triangular [seq_len, seq_len].

    M[i, j] = 0.0 jika i >= j, else -inf.
    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if seq_len <= 0:
        raise Error(
            '{"error_type":"MASK_ERROR","detail":"seq_len must be'
            ' positive","stage":"attention","layer":0}'
        )
    var mask = List[Float32]()
    mask.resize(seq_len * seq_len, Float32(0.0))
    var p_mask = mask.unsafe_ptr()
    var neg_inf = -Float32(1.0) / Float32(0.0)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                p_mask[unsafe_offset=i * seq_len + j] = neg_inf
            else:
                p_mask[unsafe_offset=i * seq_len + j] = Float32(0.0)
    return mask^


def verify_causal_mask_property(
    probs: List[Float32],
    seq_len: Int,
    num_heads: Int,
    layer_idx: Int = 0,
) raises:
    """Verifikasi bahwa token i HANYA mengattend ke token j <= i (prob[i, j] == 0 untuk j > i).

    @spec scratch/wave/m2/m2-w3-attention.md
    """
    var p_probs = probs.unsafe_ptr()
    for h in range(num_heads):
        var head_offset = h * seq_len * seq_len
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                var val = p_probs[unsafe_offset=head_offset + i * seq_len + j]
                if val != Float32(0.0):
                    raise Error(
                        '{"error_type":"MASK_ERROR","detail":"causal mask'
                        " violation: future attention detected at i="
                        + String(i)
                        + ", j="
                        + String(j)
                        + '","stage":"attention","layer":'
                        + String(layer_idx)
                        + "}"
                    )


def softmax_row_stable(
    scores: List[Float32],
    offset: Int,
    length: Int,
    valid_len: Int,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Softmax stabil numerik dengan pergeseran nilai maksimum: exp(z - max z) / sum exp(z - max z).

    Elemen j >= valid_len diatur ke probabilitas 0.0 secara eksak.
    Deteksi NaN atau unmasked Inf menghasilkan ATTENTION_ERROR.
    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if valid_len <= 0 or valid_len > length:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"invalid'
            ' valid_len","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )
    var p_scores = scores.unsafe_ptr()

    # Cari max dari elemen valid
    var max_val = p_scores[unsafe_offset=offset]
    for j in range(valid_len):
        var s = p_scores[unsafe_offset=offset + j]
        if isnan(s) or isinf(s):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite score in'
                ' attention logits","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )
        if s > max_val:
            max_val = s

    # Hitung exp(z - max_val)
    var sum_exp = Float32(0.0)
    var probs = List[Float32]()
    probs.resize(length, Float32(0.0))
    var p_probs = probs.unsafe_ptr()

    for j in range(valid_len):
        var e = exp(p_scores[unsafe_offset=offset + j] - max_val)
        if isnan(e) or isinf(e):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"overflow in'
                ' softmax exponential","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )
        p_probs[unsafe_offset=j] = e
        sum_exp += e

    if sum_exp <= Float32(0.0) or isnan(sum_exp) or isinf(sum_exp):
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"invalid softmax'
            ' denominator","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    # Normalisasi
    for j in range(valid_len):
        p_probs[unsafe_offset=j] = p_probs[unsafe_offset=j] / sum_exp

    return probs^


def mha_forward(
    q: List[Float32],
    k: List[Float32],
    v: List[Float32],
    seq_len: Int,
    cfg: ModelConfig,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Multi-Head Attention dengan causal mask dan softmax stabil.

    Rumus: softmax(Q K^T / sqrt(d_h) + M) V
    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if seq_len <= 0:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"seq_len must be'
            ' positive","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )
    var hidden = cfg.hidden_size
    var num_heads = cfg.num_attention_heads
    var head_dim = cfg.head_dim()

    if num_heads * head_dim != hidden:
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"num_attention_heads *'
            ' head_dim != hidden_size","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )
    if (
        len(q) != seq_len * hidden
        or len(k) != seq_len * hidden
        or len(v) != seq_len * hidden
    ):
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"Q, K, V length'
            ' mismatch","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    for i in range(len(q)):
        if isnan(q[i]) or isinf(q[i]):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' Q","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )
    for i in range(len(k)):
        if isnan(k[i]) or isinf(k[i]):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' K","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )
    for i in range(len(v)):
        if isnan(v[i]) or isinf(v[i]):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' V","stage":"attention","layer":'
                + String(layer_idx)
                + "}"
            )

    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var out = List[Float32]()
    out.resize(seq_len * hidden, Float32(0.0))

    var p_q = q.unsafe_ptr()
    var p_k = k.unsafe_ptr()
    var p_v = v.unsafe_ptr()
    var p_out = out.unsafe_ptr()

    var scores_row = List[Float32]()
    scores_row.resize(seq_len, Float32(0.0))
    var p_scores = scores_row.unsafe_ptr()

    # Track probabilities untuk verifikasi causal mask
    var all_probs = List[Float32]()
    all_probs.resize(num_heads * seq_len * seq_len, Float32(0.0))
    var p_all_probs = all_probs.unsafe_ptr()

    for h in range(num_heads):
        for i in range(seq_len):
            # 1. Hitung Q K^T / sqrt(d_h) untuk j <= i
            for j in range(i + 1):
                var acc = Float32(0.0)
                var q_offset = i * hidden + h * head_dim
                var k_offset = j * hidden + h * head_dim
                for d in range(head_dim):
                    acc += (
                        p_q[unsafe_offset=q_offset + d]
                        * p_k[unsafe_offset=k_offset + d]
                    )
                p_scores[unsafe_offset=j] = acc * scale

            # 2. Softmax stabil numerik untuk baris i
            var probs_row = softmax_row_stable(
                scores_row, 0, seq_len, i + 1, layer_idx
            )
            var p_prow = probs_row.unsafe_ptr()

            # Catat probabilitas ke all_probs
            var prob_base = (h * seq_len + i) * seq_len
            for j in range(seq_len):
                p_all_probs[unsafe_offset=prob_base + j] = p_prow[
                    unsafe_offset=j
                ]

            # 3. Akumulasi bobot perhatian * V
            var out_offset = i * hidden + h * head_dim
            for d in range(head_dim):
                var val = Float32(0.0)
                for j in range(i + 1):
                    var v_offset = j * hidden + h * head_dim
                    val += (
                        p_prow[unsafe_offset=j]
                        * p_v[unsafe_offset=v_offset + d]
                    )
                if isnan(val) or isinf(val):
                    raise Error(
                        '{"error_type":"ATTENTION_ERROR","detail":"non-finite'
                        " value in MHA context"
                        ' output","stage":"attention","layer":'
                        + String(layer_idx)
                        + "}"
                    )
                p_out[unsafe_offset=out_offset + d] = val

    # Verifikasi causal mask property
    verify_causal_mask_property(all_probs, seq_len, num_heads, layer_idx)

    return out^


def o_project(
    x: List[Float32],
    w_o: List[Float32],
    b_o: List[Float32],
    seq_len: Int,
    hidden: Int,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Proyeksi linear output attention: y = x W_o^T (+ b_o jika ada).

    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if seq_len <= 0 or hidden <= 0:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"seq_len and hidden must'
            ' be positive","stage":"oproj","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"input length'
            ' mismatch","stage":"oproj","layer":'
            + String(layer_idx)
            + "}"
        )
    if len(w_o) != hidden * hidden:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"w_o length'
            ' mismatch","stage":"oproj","layer":'
            + String(layer_idx)
            + "}"
        )
    var has_bias = len(b_o) > 0
    if has_bias and len(b_o) != hidden:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"b_o length'
            ' mismatch","stage":"oproj","layer":'
            + String(layer_idx)
            + "}"
        )

    for i in range(len(x)):
        if isnan(x[i]) or isinf(x[i]):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' o_proj input","stage":"oproj","layer":'
                + String(layer_idx)
                + "}"
            )
    for i in range(len(w_o)):
        if isnan(w_o[i]) or isinf(w_o[i]):
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"non-finite value'
                ' in w_o","stage":"oproj","layer":'
                + String(layer_idx)
                + "}"
            )
    if has_bias:
        for i in range(len(b_o)):
            if isnan(b_o[i]) or isinf(b_o[i]):
                raise Error(
                    '{"error_type":"WEIGHT_LOAD_FAILED","detail":"non-finite'
                    ' value in b_o","stage":"oproj","layer":'
                    + String(layer_idx)
                    + "}"
                )

    var out = List[Float32]()
    out.resize(seq_len * hidden, Float32(0.0))

    var p_x = x.unsafe_ptr()
    var p_w = w_o.unsafe_ptr()
    var p_b = b_o.unsafe_ptr()
    var p_out = out.unsafe_ptr()

    for t in range(seq_len):
        var x_row = t * hidden
        for j in range(hidden):
            var w_row = j * hidden
            var acc = Float32(0.0)
            for k in range(hidden):
                acc += (
                    p_x[unsafe_offset=x_row + k] * p_w[unsafe_offset=w_row + k]
                )
            if has_bias:
                acc += p_b[unsafe_offset=j]
            if isnan(acc) or isinf(acc):
                raise Error(
                    '{"error_type":"ATTENTION_ERROR","detail":"non-finite'
                    ' value in o_proj output","stage":"oproj","layer":'
                    + String(layer_idx)
                    + "}"
                )
            p_out[unsafe_offset=x_row + j] = acc
    return out^


def add_residual(
    y: List[Float32],
    x: List[Float32],
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Residual connection: out = y + x.

    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if len(y) != len(x):
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"residual length'
            ' mismatch","stage":"residual","layer":'
            + String(layer_idx)
            + "}"
        )
    var n = len(y)
    var out = List[Float32]()
    out.resize(n, Float32(0.0))
    var p_y = y.unsafe_ptr()
    var p_x = x.unsafe_ptr()
    var p_out = out.unsafe_ptr()
    for i in range(n):
        var vy = p_y[unsafe_offset=i]
        var vx = p_x[unsafe_offset=i]
        if isnan(vy) or isinf(vy) or isnan(vx) or isinf(vx):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' residual","stage":"residual","layer":'
                + String(layer_idx)
                + "}"
            )
        var val = vy + vx
        if isnan(val) or isinf(val):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"overflow in'
                ' residual addition","stage":"residual","layer":'
                + String(layer_idx)
                + "}"
            )
        p_out[unsafe_offset=i] = val
    return out^


def forward_attention_block(
    x: List[Float32],
    weights: AttentionWeights,
    seq_len: Int,
    cfg: ModelConfig,
    eps: Float32,
    pos_offset: Int = 0,
    base: Float32 = Float32(1000000.0),
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Pipeline blok attention utuh: RMSNorm F6 -> QKV (+bias) -> RoPE rotate_half F7 -> MHA Causal -> o_proj (+bias) -> Residual.

    @spec docs/milestones/M2-attention.md
    @spec scratch/wave/m2/m2-w3-attention.md
    """
    var hidden = cfg.hidden_size
    if len(x) != seq_len * hidden:
        raise Error(
            '{"error_type":"ACT_LOAD_FAILED","detail":"activation length'
            ' mismatch","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    # 1. RMSNorm per-token (reuse F6 dari M1 tanpa duplikasi)
    var x_norm = List[Float32]()
    x_norm.reserve(seq_len * hidden)
    for t in range(seq_len):
        var tok_vec = List[Float32]()
        tok_vec.reserve(hidden)
        var row = t * hidden
        for k in range(hidden):
            tok_vec.append(x[row + k])
        var normed = rmsnorm(tok_vec, weights.norm_gamma, eps)
        for k in range(hidden):
            x_norm.append(normed[k])

    # 2. QKV Projection (W1)
    var qkv_res = qkv_forward(x_norm, weights.qkv, seq_len, cfg)
    ref q = qkv_res[0]
    ref k = qkv_res[1]
    ref v = qkv_res[2]

    # 3. RoPE rotate_half F7 (W2)
    var rope_res = apply_rope(q, k, seq_len, cfg, pos_offset, base, layer_idx)
    ref q_rot = rope_res[0]
    ref k_rot = rope_res[1]

    # 4. MHA dengan Causal Mask & Softmax Stabil (W3)
    var attn_out = mha_forward(q_rot, k_rot, v, seq_len, cfg, layer_idx)

    # 5. o_proj (W3)
    var y = o_project(
        attn_out, weights.w_o, weights.b_o, seq_len, hidden, layer_idx
    )

    # 6. Residual connection y_final = y + x (W3)
    var y_final = add_residual(y, x, layer_idx)
    return y_final^


def load_layer_attention_weights(
    layer_idx: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    mut telemetry: LoadMemoryTelemetry,
) raises -> AttentionWeights:
    """Memuat seluruh bobot blok attention satu layer dari shard safetensors.

    - input_layernorm.weight
    - QKV weights (W_q, W_k, W_v, b_q, b_k, b_v)
    - o_proj.weight (+ b_o jika ada di checkpoint)
    """
    if layer_idx < 0 or layer_idx >= cfg.num_hidden_layers:
        raise Error(
            '{"error_type":"LAYER_INVALID","detail":"invalid layer index: '
            + String(layer_idx)
            + " (expected 0.."
            + String(cfg.num_hidden_layers - 1)
            + ')","stage":"attention","layer":'
            + String(layer_idx)
            + "}"
        )

    var hidden = cfg.hidden_size
    var prefix = "model.layers." + String(layer_idx) + "."
    var norm_name = prefix + "input_layernorm.weight"
    var o_proj_name = prefix + "self_attn.o_proj.weight"
    var o_bias_name = prefix + "self_attn.o_proj.bias"

    if norm_name not in weight_map:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"norm weight not in'
            " weight_map: "
            + norm_name
            + '","shard":"","tensor_name":"'
            + norm_name
            + '"}'
        )
    if o_proj_name not in weight_map:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"o_proj weight not in'
            " weight_map: "
            + o_proj_name
            + '","shard":"","tensor_name":"'
            + o_proj_name
            + '"}'
        )

    var norm_gamma = _load_one_tensor_by_name(
        model_root,
        weight_map[norm_name],
        norm_name,
        hidden,
        1,
        False,
        telemetry,
    )
    var qkv = load_layer_qkv_weights(
        layer_idx, model_root, weight_map, cfg, telemetry
    )
    var w_o = _load_one_tensor_by_name(
        model_root,
        weight_map[o_proj_name],
        o_proj_name,
        hidden,
        hidden,
        True,
        telemetry,
    )
    var b_o = List[Float32]()
    if o_bias_name in weight_map:
        b_o = _load_one_tensor_by_name(
            model_root,
            weight_map[o_bias_name],
            o_bias_name,
            hidden,
            1,
            False,
            telemetry,
        )

    return AttentionWeights(norm_gamma^, qkv^, w_o^, b_o^)


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


def validate_attention_bias_in_index(
    weight_map: Dict[String, String],
    num_layers: Int = 24,
) raises:
    """Validasi P-2: index safetensors wajib memiliki 3 tensor bias per layer (q, k, v)
    sehingga total bias == 3 * num_layers (72 untuk 24 layer).
    Jika kurang/lebih atau ada layer yang tidak lengkap -> WEIGHT_LOAD_FAILED.
    """
    for l in range(num_layers):
        var prefix = "model.layers." + String(l) + ".self_attn."
        var q_bias = prefix + "q_proj.bias"
        var k_bias = prefix + "k_proj.bias"
        var v_bias = prefix + "v_proj.bias"
        if q_bias not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"missing '
                + q_bias
                + ' in index weight_map","shard":"","tensor_name":"'
                + q_bias
                + '"}'
            )
        if k_bias not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"missing '
                + k_bias
                + ' in index weight_map","shard":"","tensor_name":"'
                + k_bias
                + '"}'
            )
        if v_bias not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"missing '
                + v_bias
                + ' in index weight_map","shard":"","tensor_name":"'
                + v_bias
                + '"}'
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


def _load_one_tensor_by_name(
    model_root: String,
    shard_file: String,
    tensor_name: String,
    dim0: Int,
    dim1: Int,
    is_2d: Bool,
    mut telemetry: LoadMemoryTelemetry,
) raises -> List[Float32]:
    var shard_path = (
        String(model_root, "/", shard_file) if model_root != "" else shard_file
    )
    var header = read_header(shard_path)
    var found = False
    var meta = TensorMeta("", "", List[Int](), 0, 0)
    for i in range(len(header.entries)):
        ref e = header.entries[i]
        if e.name == tensor_name:
            meta = e.copy()
            found = True
            break
    if not found:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"tensor '
            + tensor_name
            + ' not found in shard header","shard":"'
            + shard_path
            + '","tensor_name":"'
            + tensor_name
            + '"}'
        )
    if is_2d:
        if (
            len(meta.shape) != 2
            or meta.shape[0] != dim0
            or meta.shape[1] != dim1
        ):
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"weight shape'
                " mismatch: expected ["
                + String(dim0)
                + ", "
                + String(dim1)
                + "] got length "
                + String(len(meta.shape))
                + '","shard":"'
                + shard_path
                + '","tensor_name":"'
                + tensor_name
                + '"}'
            )
    else:
        if len(meta.shape) != 1 or meta.shape[0] != dim0:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"bias shape'
                " mismatch: expected ["
                + String(dim0)
                + "] got length "
                + String(len(meta.shape))
                + '","shard":"'
                + shard_path
                + '","tensor_name":"'
                + tensor_name
                + '"}'
            )
    return load_tensor_f32_chunked(
        shard_path, header.data_base, meta, telemetry
    )


def load_layer_qkv_weights(
    layer_idx: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    mut telemetry: LoadMemoryTelemetry,
) raises -> QKVWeights:
    """Memuat bobot QKV (w_q, w_k, w_v + b_q, b_k, b_v) untuk satu layer dari shard safetensors.

    @spec m2-w1-qkv-bias.md
    Validasi:
    - layer_idx dalam [0, num_hidden_layers) (else LAYER_INVALID)
    - bobot shape [hidden_size, hidden_size]
    - bias shape [hidden_size]
    """
    if layer_idx < 0 or layer_idx >= cfg.num_hidden_layers:
        raise Error(
            '{"error_type":"LAYER_INVALID","detail":"invalid layer index: '
            + String(layer_idx)
            + " (expected 0.."
            + String(cfg.num_hidden_layers - 1)
            + ')","stage":"qkv","layer":'
            + String(layer_idx)
            + "}"
        )

    var prefix = "model.layers." + String(layer_idx) + ".self_attn."
    var req_wq = prefix + "q_proj.weight"
    var req_bq = prefix + "q_proj.bias"
    var req_wk = prefix + "k_proj.weight"
    var req_bk = prefix + "k_proj.bias"
    var req_wv = prefix + "v_proj.weight"
    var req_bv = prefix + "v_proj.bias"

    var required = List[String]()
    required.append(req_wq)
    required.append(req_bq)
    required.append(req_wk)
    required.append(req_bk)
    required.append(req_wv)
    required.append(req_bv)

    for i in range(len(required)):
        var r = required[i]
        if r not in weight_map:
            raise Error(
                '{"error_type":"WEIGHT_LOAD_FAILED","detail":"required QKV'
                " tensor not in weight_map: "
                + r
                + '","shard":"","tensor_name":"'
                + r
                + '"}'
            )

    var hidden = cfg.hidden_size
    var w_q = _load_one_tensor_by_name(
        model_root, weight_map[req_wq], req_wq, hidden, hidden, True, telemetry
    )
    var b_q = _load_one_tensor_by_name(
        model_root, weight_map[req_bq], req_bq, hidden, 1, False, telemetry
    )
    var w_k = _load_one_tensor_by_name(
        model_root, weight_map[req_wk], req_wk, hidden, hidden, True, telemetry
    )
    var b_k = _load_one_tensor_by_name(
        model_root, weight_map[req_bk], req_bk, hidden, 1, False, telemetry
    )
    var w_v = _load_one_tensor_by_name(
        model_root, weight_map[req_wv], req_wv, hidden, hidden, True, telemetry
    )
    var b_v = _load_one_tensor_by_name(
        model_root, weight_map[req_bv], req_bv, hidden, 1, False, telemetry
    )

    return QKVWeights(w_q^, w_k^, w_v^, b_q^, b_k^, b_v^)


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


def test_qkv_project_shape_errors() raises:
    """Uji deteksi kesalahan shape pada x, w, dan b -> ACT_LOAD_FAILED / WEIGHT_LOAD_FAILED.
    """
    var valid_x: List[Float32] = [1.0, 2.0]
    var short_x: List[Float32] = [1.0]
    var valid_w: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var short_w: List[Float32] = [1.0, 0.0]
    var valid_b: List[Float32] = [0.0, 0.0]
    var short_b: List[Float32] = [0.0]

    # x mismatch
    var raised_x = False
    try:
        var _out = qkv_project(short_x, valid_w, valid_b, 1, 2)
    except:
        raised_x = True
    assert_true(raised_x)

    # w mismatch
    var raised_w = False
    try:
        var _out = qkv_project(valid_x, short_w, valid_b, 1, 2)
    except:
        raised_w = True
    assert_true(raised_w)

    # b mismatch
    var raised_b = False
    try:
        var _out = qkv_project(valid_x, valid_w, short_b, 1, 2)
    except:
        raised_b = True
    assert_true(raised_b)

    # seq_len <= 0
    var raised_seq = False
    try:
        var _out = qkv_project(valid_x, valid_w, valid_b, 0, 2)
    except:
        raised_seq = True
    assert_true(raised_seq)


def test_qkv_project_nan_inf() raises:
    """Deteksi nilai non-finite (NaN / Inf) pada input / bobot -> raise error.
    """
    var nan_val = Float32(0.0) / Float32(0.0)
    var inf_val = Float32(1.0) / Float32(0.0)

    var nan_x: List[Float32] = [nan_val, 1.0]
    var valid_x: List[Float32] = [1.0, 1.0]
    var valid_w: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var valid_b: List[Float32] = [0.0, 0.0]

    var raised_nan_x = False
    try:
        var _out = qkv_project(nan_x, valid_w, valid_b, 1, 2)
    except:
        raised_nan_x = True
    assert_true(raised_nan_x)

    var nan_w: List[Float32] = [1.0, 0.0, nan_val, 1.0]
    var raised_nan_w = False
    try:
        var _out = qkv_project(valid_x, nan_w, valid_b, 1, 2)
    except:
        raised_nan_w = True
    assert_true(raised_nan_w)

    var inf_b: List[Float32] = [0.0, inf_val]
    var raised_inf_b = False
    try:
        var _out = qkv_project(valid_x, valid_w, inf_b, 1, 2)
    except:
        raised_inf_b = True
    assert_true(raised_inf_b)


def test_qkv_forward_multi_token() raises:
    """Multi-token QKV forward: pastikan Q, K, V terpisah dan terhitung benar.
    """
    var cfg = ModelConfig(4, 1, 2, 64)
    assert_equal(cfg.head_dim(), 2)

    var x: List[Float32] = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]  # seq_len = 2, hidden = 4

    var w_q: List[Float32] = [
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
    var b_q: List[Float32] = [0.1, 0.1, 0.1, 0.1]

    var w_k: List[Float32] = [
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
    ]
    var b_k: List[Float32] = [0.2, 0.2, 0.2, 0.2]

    var w_v: List[Float32] = [
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
    ]
    var b_v: List[Float32] = [0.3, 0.3, 0.3, 0.3]

    var weights = QKVWeights(w_q^, w_k^, w_v^, b_q^, b_k^, b_v^)
    var res = qkv_forward(x, weights, 2, cfg)
    ref q = res[0]
    ref k = res[1]
    ref v = res[2]

    assert_equal(len(q), 8)
    assert_equal(len(k), 8)
    assert_equal(len(v), 8)

    # Token 0: x = [1, 0, 0, 0]
    assert_almost_equal(q[0], 1.1, atol=1e-5)
    assert_almost_equal(q[1], 0.1, atol=1e-5)
    assert_almost_equal(k[0], 2.2, atol=1e-5)
    assert_almost_equal(k[1], 0.2, atol=1e-5)
    assert_almost_equal(v[0], 3.3, atol=1e-5)
    assert_almost_equal(v[1], 0.3, atol=1e-5)

    # Token 1: x = [0, 1, 0, 0]
    assert_almost_equal(q[4], 0.1, atol=1e-5)
    assert_almost_equal(q[5], 1.1, atol=1e-5)
    assert_almost_equal(k[4], 0.2, atol=1e-5)
    assert_almost_equal(k[5], 2.2, atol=1e-5)
    assert_almost_equal(v[4], 0.3, atol=1e-5)
    assert_almost_equal(v[5], 3.3, atol=1e-5)


def test_qkv_forward_head_dim_config_error() raises:
    """Config dengan num_attention_heads * head_dim != hidden_size -> CONFIG_ERROR.
    """
    var cfg = ModelConfig(5, 1, 2, 64)  # 5 // 2 = 2, 2 * 2 = 4 != 5
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var w = List[Float32]()
    for _ in range(25):
        w.append(0.0)
    var b: List[Float32] = [0.0, 0.0, 0.0, 0.0, 0.0]
    var weights = QKVWeights(
        w.copy(), w.copy(), w.copy(), b.copy(), b.copy(), b.copy()
    )
    var raised = False
    try:
        var _res = qkv_forward(x, weights, 1, cfg)
    except:
        raised = True
    assert_true(raised)


def test_validate_attention_bias_in_index_72() raises:
    """24 layer x 3 bias = 72 bias lengkap -> lolos; 71 bias -> WEIGHT_LOAD_FAILED.
    """
    var full_map = Dict[String, String]()
    for l in range(24):
        var prefix = "model.layers." + String(l) + ".self_attn."
        full_map[prefix + "q_proj.bias"] = "shard-0.safetensors"
        full_map[prefix + "k_proj.bias"] = "shard-0.safetensors"
        full_map[prefix + "v_proj.bias"] = "shard-0.safetensors"
    validate_attention_bias_in_index(full_map, 24)

    # Hapus 1 bias (hanya 71 bias)
    var incomplete_map = Dict[String, String]()
    for l in range(24):
        var prefix = "model.layers." + String(l) + ".self_attn."
        incomplete_map[prefix + "q_proj.bias"] = "shard-0.safetensors"
        if l != 23:
            incomplete_map[prefix + "k_proj.bias"] = "shard-0.safetensors"
        incomplete_map[prefix + "v_proj.bias"] = "shard-0.safetensors"
    var raised = False
    try:
        validate_attention_bias_in_index(incomplete_map, 24)
    except:
        raised = True
    assert_true(raised)


def test_property_p2_config_vs_index() raises:
    """Property P-2 (§2.3): config tidak menulis attention_bias, namun index memiliki 72 bias.
    """
    # 1. Verifikasi model_config.json tidak memuat field attention_bias
    var cfg_bytes = read_small_file("fixtures/m0/model_config.json")
    var cfg_str = String(from_utf8_lossy=Span(cfg_bytes))
    assert_false(_contains(cfg_str, "attention_bias"))

    # 2. Verifikasi index asli fixtures/m0_qwen_index.json memiliki tepat 72 tensor bias attention
    var packed = parse_index("fixtures/m0_qwen_index.json")
    var nn = 0
    var cs = packed[0].as_bytes()
    for ci in range(len(cs)):
        nn = nn * 10 + (Int(cs[ci]) - 48)
    assert_equal(nn, 4659)

    var all_names = List[String]()
    var weight_map = Dict[String, String]()
    for i in range(nn):
        var nm = packed[1 + 2 * i]
        var sf = packed[1 + 2 * i + 1]
        all_names.append(nm)
        weight_map[nm] = sf

    var bias_names = collect_attention_bias_names(all_names, 24)
    assert_equal(len(bias_names), 72)
    validate_bias_count(bias_names, 24, "fixtures/m0_qwen_index.json")
    validate_attention_bias_in_index(weight_map, 24)


def test_load_layer_qkv_weights_validation() raises:
    """Validasi load_layer_qkv_weights: layer invalid dan tensor hilang."""
    var cfg = ModelConfig(64, 24, 2, 512)
    var empty_map = Dict[String, String]()
    var telem = LoadMemoryTelemetry()

    # Layer < 0
    var raised_neg = False
    try:
        var _w = load_layer_qkv_weights(-1, "", empty_map, cfg, telem)
    except:
        raised_neg = True
    assert_true(raised_neg)

    # Layer >= num_hidden_layers
    var raised_high = False
    try:
        var _w2 = load_layer_qkv_weights(24, "", empty_map, cfg, telem)
    except:
        raised_high = True
    assert_true(raised_high)

    # Missing tensor in weight map
    var raised_missing = False
    try:
        var _w3 = load_layer_qkv_weights(0, "", empty_map, cfg, telem)
    except:
        raised_missing = True
    assert_true(raised_missing)


def test_qkv_project_oracle_slice() raises:
    """Oracle slice test: memverifikasi proyeksi linear fp32 y = xW^T + b
    secara presisi terhadap ground truth independen (PyTorch fp32 formula).
    """
    var seq_len = 2
    var hidden = 4
    # x: [2, 4]
    var x: List[Float32] = [
        1.0,
        2.0,
        3.0,
        4.0,
        0.5,
        -1.0,
        2.5,
        -0.5,
    ]
    # W_q: [4, 4] (row-major: W[j, k] * x[k])
    var w_q: List[Float32] = [
        0.1,
        0.2,
        0.3,
        0.4,
        -0.1,
        0.5,
        0.0,
        0.2,
        0.3,
        -0.2,
        0.1,
        0.0,
        0.0,
        0.1,
        -0.3,
        0.2,
    ]
    # b_q: [4]
    var b_q: List[Float32] = [0.01, -0.02, 0.03, -0.04]

    var q = qkv_project(x, w_q, b_q, seq_len, hidden)
    assert_equal(len(q), 8)
    assert_almost_equal(q[0], Float32(3.01), atol=1e-5)
    assert_almost_equal(q[1], Float32(1.68), atol=1e-5)
    assert_almost_equal(q[2], Float32(0.23), atol=1e-5)
    assert_almost_equal(q[3], Float32(0.06), atol=1e-5)
    assert_almost_equal(q[4], Float32(0.41), atol=1e-5)
    assert_almost_equal(q[5], Float32(-0.67), atol=1e-5)
    assert_almost_equal(q[6], Float32(0.63), atol=1e-5)
    assert_almost_equal(q[7], Float32(-0.99), atol=1e-5)


def test_rope_rotate_half_position_zero() raises:
    """Invariant F7: pada posisi m=0, RoPE adalah identitas: rope(x, 0) == x."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var y = rope_rotate_half(x, 1, cfg, pos_offset=0)
    assert_equal(len(y), 4)
    for i in range(4):
        assert_almost_equal(y[i], x[i], atol=1e-6)


def test_rope_rotate_half_known_vector() raises:
    """Verifikasi analitis vektor d_h=4 pada posisi m=1."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var y = rope_rotate_half(x, 1, cfg, pos_offset=1)
    # y[0] = 1*cos(1) - 3*sin(1) ≈ -1.9841106
    # y[1] = 2*cos(0.001) - 4*sin(0.001) ≈ 1.9959991
    # y[2] = 1*sin(1) + 3*cos(1) ≈ 2.4623778
    # y[3] = 2*sin(0.001) + 4*cos(0.001) ≈ 4.001998
    assert_almost_equal(y[0], Float32(-1.9841106), atol=1e-5)
    assert_almost_equal(y[1], Float32(1.9959991), atol=1e-5)
    assert_almost_equal(y[2], Float32(2.4623778), atol=1e-5)
    assert_almost_equal(y[3], Float32(4.001998), atol=1e-5)


def test_rope_rotate_half_oracle_slice() raises:
    """Oracle slice test: cross-check 2-token batch terhadap PyTorch fp32 formula.
    """
    var cfg = ModelConfig(4, 1, 1, 10)
    var seq_len = 2
    var x: List[Float32] = [
        1.0,
        2.0,
        3.0,
        4.0,
        0.5,
        -1.0,
        2.5,
        -0.5,
    ]
    var y = rope_rotate_half(x, seq_len, cfg, pos_offset=1)
    assert_equal(len(y), 8)
    # Token 0 (pos = 1)
    assert_almost_equal(y[0], Float32(-1.9841106), atol=1e-5)
    assert_almost_equal(y[1], Float32(1.9959991), atol=1e-5)
    assert_almost_equal(y[2], Float32(2.4623780), atol=1e-5)
    assert_almost_equal(y[3], Float32(4.0019979), atol=1e-5)
    # Token 1 (pos = 2)
    assert_almost_equal(y[4], Float32(-2.4813168), atol=1e-5)
    assert_almost_equal(y[5], Float32(-0.9989980), atol=1e-5)
    assert_almost_equal(y[6], Float32(-0.5857184), atol=1e-5)
    assert_almost_equal(y[7], Float32(-0.5019990), atol=1e-5)


def test_rope_rotate_half_isometry_properties() raises:
    """Property test P-3: isometri ||q'|| == ||q|| di berbagai posisi (0, 1, 15, 100, 2048)
    dan layer (0, 12, 23) dengan dimensi kepala aktual d_h=128.
    """
    var cfg = ModelConfig(256, 24, 2, 1000)
    var seq_len = 1
    var hidden = cfg.hidden_size

    # Bentuk vektor aktivasi bervariasi
    var x = List[Float32]()
    x.reserve(hidden)
    for i in range(hidden):
        x.append(Float32((i % 29) - 14) * Float32(0.25))

    var positions: List[Int] = [0, 1, 15, 100, 2048]
    var layers: List[Int] = [0, 12, 23]

    for p in range(len(positions)):
        var pos = positions[p]
        for l in range(len(layers)):
            var layer = layers[l]
            var y = rope_rotate_half(
                x, seq_len, cfg, pos_offset=pos, layer_idx=layer
            )
            # verify_rope_isometry otomatis dipanggil dalam rope_rotate_half,
            # tetapi dipanggil eksplisit di sini untuk assert properti.
            verify_rope_isometry(x, y, seq_len, cfg, layer_idx=layer)


def test_rope_rotate_half_linearity() raises:
    """Sifat linear: RoPE(c * q) == c * RoPE(q)."""
    var cfg = ModelConfig(128, 1, 1, 100)
    var seq_len = 1
    var c = Float32(3.5)

    var x1 = List[Float32]()
    var x2 = List[Float32]()
    for i in range(128):
        var val = Float32((i % 17) - 8) * Float32(0.1)
        x1.append(val)
        x2.append(val * c)

    var y1 = rope_rotate_half(x1, seq_len, cfg, pos_offset=7)
    var y2 = rope_rotate_half(x2, seq_len, cfg, pos_offset=7)

    for i in range(128):
        assert_almost_equal(y2[i], y1[i] * c, atol=1e-4)


def test_rope_rotate_half_vs_interleaved_distinction() raises:
    """Verifikasi isolasi gaya rotasi: rotate_half vs interleaved (FAIL category rope-style).

    Perbedaan kedua metode rotasi harus signifikan (> 0.5) untuk mencegah
    salah gaya.
    """
    var cfg = ModelConfig(4, 1, 1, 10)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var y_half = rope_rotate_half(x, 1, cfg, pos_offset=1)

    # Interleaved manual: (x0, x1) diputar dg theta[0], (x2, x3) diputar dg theta[1]
    # theta[0] = 1.0 * (1.0 / pow(1e6, 0.0)) = 1.0
    # theta[1] = 1.0 * (1.0 / pow(1e6, 2/4)) = 0.001
    var th0 = Float32(1.0)
    var th1 = Float32(0.001)
    var y_inter0 = x[0] * cos(th0) - x[1] * sin(th0)
    var y_inter1 = x[0] * sin(th0) + x[1] * cos(th0)
    var y_inter2 = x[2] * cos(th1) - x[3] * sin(th1)
    var y_inter3 = x[2] * sin(th1) + x[3] * cos(th1)

    var max_diff = Float32(0.0)
    var diff0 = abs(y_half[0] - y_inter0)
    var diff1 = abs(y_half[1] - y_inter1)
    var diff2 = abs(y_half[2] - y_inter2)
    var diff3 = abs(y_half[3] - y_inter3)
    if diff0 > max_diff:
        max_diff = diff0
    if diff1 > max_diff:
        max_diff = diff1
    if diff2 > max_diff:
        max_diff = diff2
    if diff3 > max_diff:
        max_diff = diff3

    # Selisih harus > 0.5 (teori: ~0.8415), membuktikan kedua gaya berbeda tegas.
    assert_true(max_diff > Float32(0.5))


def test_rope_rotate_half_shape_errors() raises:
    """Error handling dimensi: seq_len <= 0, length mismatch, odd head_dim, base <= 0 -> ROPE_ERROR.
    """
    var valid_cfg = ModelConfig(4, 1, 1, 10)
    var odd_cfg = ModelConfig(3, 1, 1, 10)  # head_dim = 3 (ganjil)
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var short_x: List[Float32] = [1.0, 2.0, 3.0]

    # seq_len <= 0
    var r1 = False
    try:
        var _y = rope_rotate_half(x, 0, valid_cfg)
    except e:
        r1 = True
    assert_true(r1)

    # length mismatch
    var r2 = False
    try:
        var _y = rope_rotate_half(short_x, 1, valid_cfg)
    except e:
        r2 = True
    assert_true(r2)

    # odd head_dim
    var r3 = False
    try:
        var _y = rope_rotate_half(x, 1, odd_cfg)
    except e:
        r3 = True
    assert_true(r3)

    # base <= 0
    var r4 = False
    try:
        var _y = rope_rotate_half(x, 1, valid_cfg, base=Float32(-1.0))
    except e:
        r4 = True
    assert_true(r4)


def test_rope_rotate_half_nan_inf_detection() raises:
    """Non-finite value di input memicu ROPE_ERROR."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var nan_x: List[Float32] = [
        1.0,
        Float32(0.0) / Float32(0.0),
        3.0,
        4.0,
    ]
    var r1 = False
    try:
        var _y = rope_rotate_half(nan_x, 1, cfg)
    except e:
        r1 = True
    assert_true(r1)

    var inf_x: List[Float32] = [
        1.0,
        Float32(1.0) / Float32(0.0),
        3.0,
        4.0,
    ]
    var r2 = False
    try:
        var _y = rope_rotate_half(inf_x, 1, cfg)
    except e:
        r2 = True
    assert_true(r2)


def test_rope_isometry_violation_error() raises:
    """Pelanggaran invariant isometri memicu ROPE_ERROR eksplisit."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var orig: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    # Vektor yang sengaja diubah norm-nya (dikalikan 1.5)
    var violated: List[Float32] = [1.5, 3.0, 4.5, 6.0]
    var raised = False
    try:
        verify_rope_isometry(orig, violated, 1, cfg, layer_idx=5)
    except e:
        raised = True
    assert_true(raised)


def test_apply_rope_q_and_k() raises:
    """Terapkan RoPE memutar Q dan K secara bersamaan."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var q: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var k: List[Float32] = [0.5, -1.0, 2.5, -0.5]
    var res = apply_rope(q, k, 1, cfg, pos_offset=1)
    ref q_rot = res[0]
    ref k_rot = res[1]
    assert_equal(len(q_rot), 4)
    assert_equal(len(k_rot), 4)
    assert_almost_equal(q_rot[0], Float32(-1.9841106), atol=1e-5)
    assert_almost_equal(k_rot[0], Float32(-1.8335261), atol=1e-5)


def test_causal_mask_triangular_structure() raises:
    """Struktur matriks causal mask: 0 pada j <= i, -inf pada j > i."""
    var seq_len = 4
    var mask = build_causal_mask(seq_len)
    assert_equal(len(mask), 16)
    for i in range(seq_len):
        for j in range(seq_len):
            var val = mask[i * seq_len + j]
            if j <= i:
                assert_almost_equal(val, Float32(0.0), atol=1e-6)
            else:
                assert_true(isinf(val) and val < Float32(0.0))


def test_causal_mask_autoregressive_property() raises:
    """Property test causal mask: output token t hanya dari input <= t."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var seq_len = 3
    var hidden = 4

    # Sequence A: [t0, t1, t2]
    var q_a: List[Float32] = [
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    var k_a: List[Float32] = [
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    var v_a: List[Float32] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    ]

    # Sequence B: [t0, t1, t2_prime] (token 0 dan 1 sama persis, token 2 berbeda drastis)
    var q_b: List[Float32] = [
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        -5.0,
        8.0,
        -3.0,
        2.0,
    ]
    var k_b: List[Float32] = [
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        99.0,
        -42.0,
        13.0,
        7.0,
    ]
    var v_b: List[Float32] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        -100.0,
        200.0,
        -300.0,
        400.0,
    ]

    var out_a = mha_forward(q_a, k_a, v_a, seq_len, cfg)
    var out_b = mha_forward(q_b, k_b, v_b, seq_len, cfg)

    # Output pada token 0 dan token 1 wajib IDENTIK (tidak terpengaruh perubahan token 2)
    for t in range(2):
        for d in range(hidden):
            assert_almost_equal(
                out_a[t * hidden + d], out_b[t * hidden + d], atol=1e-6
            )

    # Output pada token 2 wajib BERBEDA
    var diff_t2 = Float32(0.0)
    for d in range(hidden):
        diff_t2 += abs(out_a[2 * hidden + d] - out_b[2 * hidden + d])
    assert_true(diff_t2 > Float32(1.0))


def test_softmax_stable_max_shift() raises:
    """Property test softmax stabil: invariansi max-shift softmax(z + c) == softmax(z).
    """
    var z: List[Float32] = [1.0, 5.0, 2.0, 4.0]
    var z_shifted: List[Float32] = [1001.0, 1005.0, 1002.0, 1004.0]
    var p1 = softmax_row_stable(z, 0, 4, 4)
    var p2 = softmax_row_stable(z_shifted, 0, 4, 4)
    for j in range(4):
        assert_almost_equal(p1[j], p2[j], atol=1e-6)


def test_softmax_stable_sum_to_one() raises:
    """Probabilitas softmax wajib berjumlah tepat 1.0 pada elemen valid."""
    var z: List[Float32] = [-3.0, 2.0, 0.5, 99.0]
    # Uji valid_len = 3 (elemen ke-4 dimask)
    var p = softmax_row_stable(z, 0, 4, 3)
    var sum_prob = Float32(0.0)
    for j in range(3):
        sum_prob += p[j]
    assert_almost_equal(sum_prob, Float32(1.0), atol=1e-6)
    assert_almost_equal(p[3], Float32(0.0), atol=1e-6)


def test_softmax_stable_nan_inf_rejected() raises:
    """Non-finite value di unmasked logits memicu ATTENTION_ERROR."""
    var nan_z: List[Float32] = [1.0, Float32(0.0) / Float32(0.0), 3.0]
    var r1 = False
    try:
        var _p = softmax_row_stable(nan_z, 0, 3, 3)
    except e:
        r1 = True
    assert_true(r1)


def test_mha_forward_known_oracle() raises:
    """Verifikasi analitis MHA dengan bobot deterministik kecil vs PyTorch oracle.
    """
    var cfg = ModelConfig(4, 1, 1, 10)
    var seq_len = 2
    var q: List[Float32] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    var k: List[Float32] = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    var v: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var out = mha_forward(q, k, v, seq_len, cfg)
    assert_equal(len(out), 8)
    # Token 0 (hanya attend ke token 0): [1, 2, 3, 4]
    assert_almost_equal(out[0], Float32(1.0), atol=1e-5)
    assert_almost_equal(out[1], Float32(2.0), atol=1e-5)
    assert_almost_equal(out[2], Float32(3.0), atol=1e-5)
    assert_almost_equal(out[3], Float32(4.0), atol=1e-5)
    # Token 1 (attend 50% ke 0 dan 50% ke 1): [3, 4, 5, 6]
    assert_almost_equal(out[4], Float32(3.0), atol=1e-5)
    assert_almost_equal(out[5], Float32(4.0), atol=1e-5)
    assert_almost_equal(out[6], Float32(5.0), atol=1e-5)
    assert_almost_equal(out[7], Float32(6.0), atol=1e-5)


def test_o_project_matmul_and_bias() raises:
    """Proyeksi output o_project: perkalian matriks + penambahan bias opsional.
    """
    var seq_len = 1
    var hidden = 2
    var x: List[Float32] = [2.0, 3.0]
    var w_o: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var b_o: List[Float32] = [0.5, -0.5]
    var empty_b = List[Float32]()

    # Tanpa bias
    var y1 = o_project(x, w_o, empty_b, seq_len, hidden)
    assert_almost_equal(y1[0], Float32(2.0), atol=1e-6)
    assert_almost_equal(y1[1], Float32(3.0), atol=1e-6)

    # Dengan bias
    var y2 = o_project(x, w_o, b_o, seq_len, hidden)
    assert_almost_equal(y2[0], Float32(2.5), atol=1e-6)
    assert_almost_equal(y2[1], Float32(2.5), atol=1e-6)


def test_add_residual() raises:
    """Residual connection y + x dan validasi dimensi."""
    var y: List[Float32] = [1.0, 2.0, 3.0]
    var x: List[Float32] = [0.5, 1.0, 1.5]
    var res = add_residual(y, x)
    assert_equal(len(res), 3)
    assert_almost_equal(res[0], Float32(1.5), atol=1e-6)
    assert_almost_equal(res[1], Float32(3.0), atol=1e-6)
    assert_almost_equal(res[2], Float32(4.5), atol=1e-6)


def test_forward_attention_block_oracle_slice() raises:
    """End-to-end forward attention block vs PyTorch fp32 oracle."""
    var cfg = ModelConfig(4, 1, 1, 10)
    var seq_len = 2
    var eps = Float32(1e-6)

    var x: List[Float32] = [
        1.0,
        2.0,
        3.0,
        4.0,
        0.5,
        -1.0,
        2.5,
        -0.5,
    ]
    var gamma: List[Float32] = [1.0, 1.0, 1.0, 1.0]

    # Matriks identitas 4x4 untuk W_q, W_k, W_v, W_o
    var eye4 = List[Float32]()
    eye4.resize(16, Float32(0.0))
    eye4[0] = Float32(1.0)
    eye4[5] = Float32(1.0)
    eye4[10] = Float32(1.0)
    eye4[15] = Float32(1.0)

    var zero_bias = List[Float32]()
    zero_bias.resize(4, Float32(0.0))

    var qkv = QKVWeights(
        eye4.copy(),
        eye4.copy(),
        eye4.copy(),
        zero_bias.copy(),
        zero_bias.copy(),
        zero_bias.copy(),
    )
    var empty_b = List[Float32]()
    var weights = AttentionWeights(gamma^, qkv^, eye4^, empty_b^)

    var y_final = forward_attention_block(
        x, weights, seq_len, cfg, eps, pos_offset=0
    )
    assert_equal(len(y_final), 8)

    # Cross-check nilai presisi terhadap oracle PyTorch fp32
    assert_almost_equal(y_final[0], Float32(1.3651483), atol=1e-5)
    assert_almost_equal(y_final[1], Float32(2.7302966), atol=1e-5)
    assert_almost_equal(y_final[2], Float32(4.0954452), atol=1e-5)
    assert_almost_equal(y_final[3], Float32(5.4605932), atol=1e-5)
    assert_almost_equal(y_final[4], Float32(0.8598768), atol=1e-5)
    assert_almost_equal(y_final[5], Float32(-1.5558764), atol=1e-5)
    assert_almost_equal(y_final[6], Float32(4.2174449), atol=1e-5)
    assert_almost_equal(y_final[7], Float32(-0.6550304), atol=1e-5)


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
