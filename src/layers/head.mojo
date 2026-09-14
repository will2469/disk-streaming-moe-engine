# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Head path model: embedding lookup, final RMSNorm, dan LM Head matmul (M1)."""

from core.config import ModelConfig
from layers.rmsnorm import rmsnorm
from std.collections import List
from std.math import isinf, isnan


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
