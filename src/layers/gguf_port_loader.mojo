# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Loader bobot port Qwen3.6 dari model kuantisasi GGUF (M9-W3/M6 wiring).

Kontrak non-negotiable (fix #2):
- Runtime port (forward/decode) TIDAK PERNAH me-load safetensors real
  (BF16) maupun memanggil oracle Python. Satu-satunya sumber bobot adalah
  berkas kuantisasi GGUF yang di-stream on-demand per tensor.
- Berkas GGUF hilang/rusak -> fail-closed (NO_QUANTIZER_MODEL /
  GGUF_FILE_CORRUPT). DILARANG fallback diam-diam ke bobot sintetis,
  safetensors, atau oracle.
- Peak RAM O(1 layer): 1 block di-load -> forward -> discard per layer,
  kontinuitas M0-M4 dan fix #1.

Skema nama tensor GGUF mini (generate_m9_gguf_fixture.py):
- Global: token_embd.weight, output.weight, output_norm.weight.
- Per layer blk.{l}.: attn_norm.weight, ffn_norm.weight,
  GDN (l%4!=3): linear_attn.{k,v,beta,out}.weight,
  GatedAttn (l%4==3): attn_{q,k,v,gate,o}.weight,
  MoE: ffn_gate_exps.weight (router), ffn_{gate,up,down}.{e}.weight (8),
  ffn_shared_{gate,up,down}.weight, shared_gate.weight.
"""

from core.config import ModelConfig
from core.worker_pool import WorkerPool
from format.gguf import GGUFIndex, stream_gguf_tensor_f32
from layers.gated_attention import GatedAttentionWeights, GatedAttnKVCache
from layers.gdn import GDNState
from layers.head import embedding_lookup, matmul_activation_head
from layers.port_scheduler import (
    PortBlockWeights,
    SchedulerTimings,
    forward_port_block,
)
from layers.rmsnorm import rmsnorm
from layers.swiglu import SwigluWeights
from std.collections import List


def resolve_quant_model_path(quant_arg: String, model_dir: String) -> String:
    """Resolusi path model kuantisasi GGUF tanpa I/O.

    Prioritas: --quant-model eksplisit > --model-dir langsung berkas .gguf.
    Mengembalikan String kosong bila tidak ada kandidat (caller wajib
    fail-closed NO_QUANTIZER_MODEL, bukan fallback).
    """
    if quant_arg.byte_length() > 0:
        return quant_arg
    if model_dir.endswith(".gguf"):
        return model_dir
    return ""


def _expect_len(vals: List[Float32], want: Int, tname: String) raises:
    """Gate dimensi tensor hasil stream (fail-closed bila mismatch)."""
    if len(vals) != want:
        raise Error(
            String(
                "GGUF_FILE_CORRUPT: dimension mismatch for tensor ",
                tname,
                ": expected ",
                want,
                " elements, got ",
                len(vals),
            )
        )


def validate_gguf_port_coverage(index: GGUFIndex, cfg: ModelConfig) raises:
    """Validasi full coverage tensor komputasi port pada index GGUF.

    Menolak fail-closed bila satu tensor komputasi hilang (bukan fallback).
    Tensor legacy probe (ffn_down_exps.0.weight) tidak wajib untuk komputasi.
    """
    var required = List[String]()
    required.append("token_embd.weight")
    required.append("output.weight")
    required.append("output_norm.weight")
    for l in range(cfg.num_hidden_layers):
        var pfx = String("blk.", l, ".")
        required.append(String(pfx, "attn_norm.weight"))
        required.append(String(pfx, "ffn_norm.weight"))
        if cfg.is_linear_attn_layer(l):
            required.append(String(pfx, "linear_attn.k.weight"))
            required.append(String(pfx, "linear_attn.v.weight"))
            required.append(String(pfx, "linear_attn.beta.weight"))
            required.append(String(pfx, "linear_attn.out.weight"))
        else:
            required.append(String(pfx, "attn_q.weight"))
            required.append(String(pfx, "attn_k.weight"))
            required.append(String(pfx, "attn_v.weight"))
            required.append(String(pfx, "attn_gate.weight"))
            required.append(String(pfx, "attn_o.weight"))
        required.append(String(pfx, "ffn_gate_exps.weight"))
        for e in range(cfg.num_experts):
            required.append(String(pfx, "ffn_gate.", e, ".weight"))
            required.append(String(pfx, "ffn_up.", e, ".weight"))
            required.append(String(pfx, "ffn_down.", e, ".weight"))
        required.append(String(pfx, "ffn_shared_gate.weight"))
        required.append(String(pfx, "ffn_shared_up.weight"))
        required.append(String(pfx, "ffn_shared_down.weight"))
        required.append(String(pfx, "shared_gate.weight"))

    for i in range(len(required)):
        if required[i] not in index.tensor_map:
            raise Error(
                String(
                    "GGUF_FILE_CORRUPT: missing required port tensor: ",
                    required[i],
                )
            )


def load_port_block_from_gguf(
    index: GGUFIndex,
    layer_idx: Int,
    cfg: ModelConfig,
    dv: Int = 32,
    dk: Int = 32,
) raises -> PortBlockWeights:
    """Stream SATU block transformer dari GGUF (peak O(1 layer)).

    Tiap tensor di-pread + dequant on-the-fly via stream_gguf_tensor_f32,
    lalu block di-discard caller setelah forward 1 layer. Tidak ada
    konstanta sintetis: semua nilai berasal dari berkas GGUF (kecuali
    vektor bias kosong yang memang tidak ada pada arsitektur mini,
    sama seperti path referensi).
    """
    var hidden = cfg.hidden_size
    var pfx = String("blk.", layer_idx, ".")
    var is_linear = cfg.is_linear_attn_layer(layer_idx)

    var in_norm = stream_gguf_tensor_f32(index, String(pfx, "attn_norm.weight"))
    _expect_len(in_norm, hidden, String(pfx, "attn_norm.weight"))
    var post_norm = stream_gguf_tensor_f32(
        index, String(pfx, "ffn_norm.weight")
    )
    _expect_len(post_norm, hidden, String(pfx, "ffn_norm.weight"))

    var gdn_wk = List[Float32]()
    var gdn_wv = List[Float32]()
    var gdn_wbeta = List[Float32]()
    var gdn_wout = List[Float32]()
    var gated_attn = GatedAttentionWeights(
        List[Float32](),
        List[Float32](),
        List[Float32](),
        List[Float32](),
        List[Float32](),
        List[Float32](),
        List[Float32](),
        List[Float32](),
        List[Float32](),
        List[Float32](),
    )

    if is_linear:
        gdn_wk = stream_gguf_tensor_f32(
            index, String(pfx, "linear_attn.k.weight")
        )
        _expect_len(gdn_wk, dk * hidden, String(pfx, "linear_attn.k.weight"))
        gdn_wv = stream_gguf_tensor_f32(
            index, String(pfx, "linear_attn.v.weight")
        )
        _expect_len(gdn_wv, dv * hidden, String(pfx, "linear_attn.v.weight"))
        gdn_wbeta = stream_gguf_tensor_f32(
            index, String(pfx, "linear_attn.beta.weight")
        )
        _expect_len(gdn_wbeta, hidden, String(pfx, "linear_attn.beta.weight"))
        gdn_wout = stream_gguf_tensor_f32(
            index, String(pfx, "linear_attn.out.weight")
        )
        _expect_len(
            gdn_wout, hidden * dv, String(pfx, "linear_attn.out.weight")
        )
    else:
        var h_q = cfg.num_attention_heads
        var h_kv = cfg.num_key_value_heads
        var head_dim = cfg.head_dim()
        var q_dim = h_q * head_dim
        var kv_dim = h_kv * head_dim
        var w_q = stream_gguf_tensor_f32(index, String(pfx, "attn_q.weight"))
        _expect_len(w_q, q_dim * hidden, String(pfx, "attn_q.weight"))
        var w_k = stream_gguf_tensor_f32(index, String(pfx, "attn_k.weight"))
        _expect_len(w_k, kv_dim * hidden, String(pfx, "attn_k.weight"))
        var w_v = stream_gguf_tensor_f32(index, String(pfx, "attn_v.weight"))
        _expect_len(w_v, kv_dim * hidden, String(pfx, "attn_v.weight"))
        var w_gate = stream_gguf_tensor_f32(
            index, String(pfx, "attn_gate.weight")
        )
        _expect_len(w_gate, q_dim * hidden, String(pfx, "attn_gate.weight"))
        var w_o = stream_gguf_tensor_f32(index, String(pfx, "attn_o.weight"))
        _expect_len(w_o, hidden * q_dim, String(pfx, "attn_o.weight"))
        gated_attn = GatedAttentionWeights(
            w_q^,
            List[Float32](),
            w_k^,
            List[Float32](),
            w_v^,
            List[Float32](),
            w_gate^,
            List[Float32](),
            w_o^,
            List[Float32](),
        )

    var w_router = stream_gguf_tensor_f32(
        index, String(pfx, "ffn_gate_exps.weight")
    )
    _expect_len(
        w_router,
        cfg.num_experts * hidden,
        String(pfx, "ffn_gate_exps.weight"),
    )

    var inter = cfg.moe_intermediate_size
    var routed_experts = List[SwigluWeights]()
    for e in range(cfg.num_experts):
        var w_eg = stream_gguf_tensor_f32(
            index, String(pfx, "ffn_gate.", e, ".weight")
        )
        _expect_len(
            w_eg, inter * hidden, String(pfx, "ffn_gate.", e, ".weight")
        )
        var w_eu = stream_gguf_tensor_f32(
            index, String(pfx, "ffn_up.", e, ".weight")
        )
        _expect_len(w_eu, inter * hidden, String(pfx, "ffn_up.", e, ".weight"))
        var w_ed = stream_gguf_tensor_f32(
            index, String(pfx, "ffn_down.", e, ".weight")
        )
        _expect_len(
            w_ed, hidden * inter, String(pfx, "ffn_down.", e, ".weight")
        )
        routed_experts.append(SwigluWeights(w_eg^, w_eu^, w_ed^, hidden, inter))

    var sh_inter = cfg.shared_expert_intermediate_size
    var sh_g = stream_gguf_tensor_f32(
        index, String(pfx, "ffn_shared_gate.weight")
    )
    _expect_len(sh_g, sh_inter * hidden, String(pfx, "ffn_shared_gate.weight"))
    var sh_u = stream_gguf_tensor_f32(
        index, String(pfx, "ffn_shared_up.weight")
    )
    _expect_len(sh_u, sh_inter * hidden, String(pfx, "ffn_shared_up.weight"))
    var sh_d = stream_gguf_tensor_f32(
        index, String(pfx, "ffn_shared_down.weight")
    )
    _expect_len(sh_d, hidden * sh_inter, String(pfx, "ffn_shared_down.weight"))
    var shared_expert = SwigluWeights(sh_g^, sh_u^, sh_d^, hidden, sh_inter)

    var w_shared_gate = stream_gguf_tensor_f32(
        index, String(pfx, "shared_gate.weight")
    )
    _expect_len(w_shared_gate, hidden, String(pfx, "shared_gate.weight"))

    return PortBlockWeights(
        in_norm^,
        post_norm^,
        is_linear,
        gdn_wk^,
        gdn_wv^,
        gdn_wbeta^,
        gdn_wout^,
        gated_attn^,
        w_router^,
        routed_experts^,
        shared_expert^,
        w_shared_gate^,
    )


def gguf_embed_tokens(
    index: GGUFIndex, token_ids: List[Int], cfg: ModelConfig
) raises -> List[Float32]:
    """Embedding lookup dari tabel GGUF (stream sekali, discard tabel).

    Pola M1: tabel embed di-stream transient, lookup, lalu dibuang —
    tidak pernah resident bersama bobot layer.
    """
    var table = stream_gguf_tensor_f32(index, "token_embd.weight")
    _expect_len(table, cfg.vocab_size * cfg.hidden_size, "token_embd.weight")
    var out = embedding_lookup(
        token_ids, table, cfg.vocab_size, cfg.hidden_size
    )
    _ = table^
    return out^


def gguf_logits_from_hidden(
    index: GGUFIndex,
    hidden: List[Float32],
    seq_len: Int,
    cfg: ModelConfig,
    eps: Float32,
) raises -> List[Float32]:
    """Final RMSNorm + proyeksi output.weight GGUF -> logits (stream, discard).

    Pola M1/M4: norm + lm_head di-stream transient di ekor forward.
    """
    var norm_w = stream_gguf_tensor_f32(index, "output_norm.weight")
    _expect_len(norm_w, cfg.hidden_size, "output_norm.weight")
    var normed = List[Float32]()
    normed.reserve(seq_len * cfg.hidden_size)
    var tok_vec = List[Float32]()
    tok_vec.resize(cfg.hidden_size, Float32(0.0))
    var p_tok = tok_vec.unsafe_ptr()
    var p_hid = hidden.unsafe_ptr()
    for t in range(seq_len):
        var row = t * cfg.hidden_size
        for h in range(cfg.hidden_size):
            p_tok[unsafe_offset=h] = p_hid[unsafe_offset=row + h]
        var normed_tok = rmsnorm(tok_vec, norm_w, eps)
        var p_normed = normed_tok.unsafe_ptr()
        for h in range(cfg.hidden_size):
            normed.append(p_normed[unsafe_offset=h])
    _ = norm_w^

    var head_w = stream_gguf_tensor_f32(index, "output.weight")
    _expect_len(head_w, cfg.vocab_size * cfg.hidden_size, "output.weight")
    var logits = matmul_activation_head(
        normed, head_w, seq_len, cfg.vocab_size, cfg.hidden_size
    )
    _ = head_w^
    return logits^


def forward_port_macro_scheduler_gguf(
    x: List[Float32],
    index: GGUFIndex,
    mut gdn_states: GDNState,
    mut kv_cache: GatedAttnKVCache,
    pos_offset: Int,
    seq_len: Int,
    cfg: ModelConfig,
    mut timings: SchedulerTimings,
    mut pool: WorkerPool,
    dk: Int = 32,
    dv: Int = 32,
    eps: Float32 = Float32(1e-6),
) raises -> List[Float32]:
    """Macro scheduler STREAMING dari GGUF: load 1 block -> forward -> discard.

    Tidak ada bobot sintetis, tidak ada safetensors, tidak ada oracle.
    Peak O(1 layer) per invarian fix #1.
    """
    var cur_x = List[Float32]()
    cur_x.resize(len(x), Float32(0.0))
    for i in range(len(x)):
        cur_x[i] = x[i]

    var num_layers = cfg.num_hidden_layers
    for layer_idx in range(num_layers):
        var block = load_port_block_from_gguf(index, layer_idx, cfg, dv, dk)
        cur_x = forward_port_block(
            cur_x,
            block^,
            gdn_states,
            kv_cache,
            layer_idx,
            pos_offset,
            seq_len,
            cfg,
            timings,
            pool,
            dk,
            dv,
            eps,
        )

    return cur_x^


def forward_port_macro_scheduler_gguf(
    x: List[Float32],
    index: GGUFIndex,
    mut gdn_states: GDNState,
    mut kv_cache: GatedAttnKVCache,
    pos_offset: Int,
    seq_len: Int,
    cfg: ModelConfig,
    mut timings: SchedulerTimings,
    dk: Int = 32,
    dv: Int = 32,
    eps: Float32 = Float32(1e-6),
) raises -> List[Float32]:
    """Macro scheduler GGUF streaming tanpa worker pool (single-threaded)."""
    var dummy_pool = WorkerPool(1)
    return forward_port_macro_scheduler_gguf(
        x,
        index,
        gdn_states,
        kv_cache,
        pos_offset,
        seq_len,
        cfg,
        timings,
        dummy_pool,
        dk,
        dv,
        eps,
    )


def forward_port_macro_scheduler_gguf(
    x: List[Float32],
    index: GGUFIndex,
    mut gdn_states: GDNState,
    mut kv_cache: GatedAttnKVCache,
    pos_offset: Int,
    seq_len: Int,
    cfg: ModelConfig,
    dk: Int = 32,
    dv: Int = 32,
    eps: Float32 = Float32(1e-6),
) raises -> List[Float32]:
    """Macro scheduler GGUF streaming tanpa akumulator timing."""
    var dummy_timings = SchedulerTimings()
    return forward_port_macro_scheduler_gguf(
        x,
        index,
        gdn_states,
        kv_cache,
        pos_offset,
        seq_len,
        cfg,
        dummy_timings,
        dk,
        dv,
        eps,
    )
