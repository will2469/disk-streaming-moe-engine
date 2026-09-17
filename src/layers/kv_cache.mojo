# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""KV Cache Management: Layout formal, rantai bound konteks, alokasi statis, dan anggaran memori (M5-W1)."""

from cli.m5_errors import m5_error_json
from core.config import ModelConfig
from format.types import json_escape
from std.builtin.dtype import DType
from std.collections import List
from std.math import isinf, isnan

# Asumsi Terkunci Arsitektur Qwen1.5-MoE-A2.7B (F2 / M5 DoD)
comptime NUM_KV_HEADS: Int = 16
comptime HEAD_DIM: Int = 128
comptime NUM_LAYERS: Int = 24
comptime BYTES_PER_ELEM: Int = 2  # BF16
comptime DEFAULT_MAX_POS: Int = 32768  # s_max model card

# Dimensi per slot per layer: H_kv * d_h
comptime SLOT_DIM: Int = NUM_KV_HEADS * HEAD_DIM  # 2048 elemen

# Ukuran per slot per layer: K + V dalam BF16 (8 KiB)
comptime BYTES_PER_SLOT_PER_LAYER: Int = 2 * SLOT_DIM * BYTES_PER_ELEM  # 8192 B


def compute_kv_cache_bytes(ctx: Int) -> Int:
    """Menghitung total kebutuhan memori KV cache (F2): L * 8192 * ctx bytes.

    @2048: 402,653,184 B (384 MiB)
    @4096: 805,306,368 B (768 MiB = 0.75 GiB)
    """
    return NUM_LAYERS * BYTES_PER_SLOT_PER_LAYER * ctx


def validate_context_bounds(
    prompt_len: Int,
    max_tokens: Int,
    context_size: Int,
    max_pos_embeddings: Int = DEFAULT_MAX_POS,
) raises:
    """Validasi normatif rantai bound: S + N <= ctx <= s_max.

    Divalidasi SETELAH tokenisasi dan SEBELUM alokasi KV cache.
    Pelanggaran sisi kiri maupun kanan memicu M5_ERR_CONTEXT_SIZE (exit 2).
    Input non-positif memicu M5_ERR_INPUT (exit 1).
    """
    if prompt_len <= 0:
        raise Error(
            m5_error_json(
                "M5_ERR_INPUT",
                "input",
                String(
                    "Prompt length must be positive, got: ", String(prompt_len)
                ),
            )
        )
    if max_tokens <= 0:
        raise Error(
            m5_error_json(
                "M5_ERR_INPUT",
                "input",
                String(
                    "max-tokens must be positive, got: ", String(max_tokens)
                ),
            )
        )
    if context_size <= 0:
        raise Error(
            m5_error_json(
                "M5_ERR_INPUT",
                "input",
                String(
                    "context-size must be positive, got: ", String(context_size)
                ),
            )
        )

    var req_ctx = prompt_len + max_tokens
    if req_ctx > context_size:
        var details = String(
            '{"prompt_tokens":',
            String(prompt_len),
            ',"max_tokens":',
            String(max_tokens),
            ',"required_context":',
            String(req_ctx),
            ',"context_size":',
            String(context_size),
            ',"max_pos_embeddings":',
            String(max_pos_embeddings),
            "}",
        )
        raise Error(
            m5_error_json(
                "M5_ERR_CONTEXT_SIZE",
                "kv_alloc",
                String(
                    "Required context S + N (",
                    String(req_ctx),
                    ") exceeds allocated context_size (",
                    String(context_size),
                    ")",
                ),
                details,
            )
        )

    if context_size > max_pos_embeddings:
        var details = String(
            '{"context_size":',
            String(context_size),
            ',"max_pos_embeddings":',
            String(max_pos_embeddings),
            "}",
        )
        raise Error(
            m5_error_json(
                "M5_ERR_CONTEXT_SIZE",
                "kv_alloc",
                String(
                    "Allocated context_size (",
                    String(context_size),
                    ") exceeds maximum position embeddings (",
                    String(max_pos_embeddings),
                    ")",
                ),
                details,
            )
        )


@fieldwise_init
struct MemoryBudget(Copyable, Movable):
    """Dekomposisi anggaran memori M_tensor dan bound proses M_peak_bound (M5 DoD).
    """

    var ctx: Int
    var m_res_f32: Int
    var m_norm_f32: Int
    var m_kv_cache: Int
    var m_layer_weights: Int
    var m_dequant_scratch: Int
    var m_hidden_f32: Int
    var m_attn_scratch: Int
    var m_moe_scratch: Int
    var m_io_buffers: Int
    var m_tensor: Int
    var m_runtime_max: Int
    var m_alloc_max: Int
    var m_peak_bound: Int
    var gate_limit: Int

    def __init__(out self, ctx: Int):
        self.ctx = ctx
        # Embedding + lm_head resident F32 (2 * 151936 * 2048 * 4)
        self.m_res_f32 = 2489319424
        # Final norm weight F32 (2048 * 4)
        self.m_norm_f32 = 8192
        # KV Cache: L * 8192 * ctx
        self.m_kv_cache = compute_kv_cache_bytes(ctx)
        # Per-layer weights streaming BF16 (1 layer saja)
        self.m_layer_weights = 1141600000
        # Dequant scratch chunked <= 64 MiB
        self.m_dequant_scratch = 67108864
        # Hidden state [1, H] F32 (2048 * 4)
        self.m_hidden_f32 = 8192
        # Attention scratch F32: 168 KiB @ 2K, 296 KiB @ 4K
        if ctx <= 2048:
            self.m_attn_scratch = 172032
        else:
            self.m_attn_scratch = 303104
        # MoE scratch F32: 115 KiB
        self.m_moe_scratch = 117760
        # I/O buffers: 1 MiB
        self.m_io_buffers = 1048576

        # M_tensor accounted peak
        self.m_tensor = (
            self.m_res_f32
            + self.m_norm_f32
            + self.m_kv_cache
            + self.m_layer_weights
            + self.m_dequant_scratch
            + self.m_hidden_f32
            + self.m_attn_scratch
            + self.m_moe_scratch
            + self.m_io_buffers
        )

        # Allowance runtime dan allocator (masing-masing <= 150 MiB)
        self.m_runtime_max = 157286400
        self.m_alloc_max = 157286400

        # M_peak_bound = M_tensor + M_runtime + M_alloc
        self.m_peak_bound = (
            self.m_tensor + self.m_runtime_max + self.m_alloc_max
        )
        # Gate limit cgroup 5 GiB
        self.gate_limit = 5368709120

    def is_within_gate(self) -> Bool:
        return self.m_peak_bound <= self.gate_limit

    def check_accounting(self, observed_vmhwm: Int) -> Bool:
        """Memeriksa integritas akuntansi memori.

        observed_vmhwm > m_peak_bound mengindikasikan lubang akuntansi.
        """
        return observed_vmhwm <= self.m_peak_bound


struct LayerKVCache(Movable):
    """KV Cache per layer l in [0, 24).

    K buffer: [ctx, 16, 128] BF16 row-major (capacity * 2048 elemen)
    V buffer: [ctx, 16, 128] BF16 row-major (capacity * 2048 elemen)
    Pelacakan posisi: strictly half-open [0, L).
    """

    var k: List[BFloat16]
    var v: List[BFloat16]
    var capacity: Int
    var current_len: Int
    var layer_idx: Int

    def __init__(out self, capacity: Int, layer_idx: Int = 0) raises:
        if capacity <= 0:
            raise Error(
                m5_error_json(
                    "M5_ERR_KV_ALLOC",
                    "kv_alloc",
                    "KV cache capacity must be positive",
                )
            )
        self.capacity = capacity
        self.current_len = 0
        self.layer_idx = layer_idx
        self.k = List[BFloat16]()
        self.v = List[BFloat16]()
        var total_elems = capacity * SLOT_DIM
        self.k.resize(total_elems, BFloat16(0.0))
        self.v.resize(total_elems, BFloat16(0.0))

    def store_prefill(
        mut self,
        k_prompt: List[Float32],
        v_prompt: List[Float32],
        seq_len: Int,
    ) raises:
        """Menyimpan Key dan Value hasil prefill untuk posisi [0, seq_len).

        Konversi eksak Float32 -> BFloat16.
        """
        if seq_len <= 0:
            raise Error(
                m5_error_json(
                    "M5_ERR_PREFILL",
                    "prefill",
                    "Prefill seq_len must be positive",
                )
            )
        if seq_len > self.capacity:
            raise Error(
                m5_error_json(
                    "M5_ERR_CONTEXT_SIZE",
                    "kv_alloc",
                    "Prefill seq_len exceeds KV cache capacity",
                )
            )
        var expected_elems = seq_len * SLOT_DIM
        if len(k_prompt) != expected_elems or len(v_prompt) != expected_elems:
            raise Error(
                m5_error_json(
                    "M5_ERR_PREFILL",
                    "prefill",
                    "Prompt KV length mismatch against seq_len * 2048",
                )
            )

        var p_k_in = k_prompt.unsafe_ptr()
        var p_v_in = v_prompt.unsafe_ptr()
        var p_k_cache = self.k.unsafe_ptr()
        var p_v_cache = self.v.unsafe_ptr()

        for idx in range(expected_elems):
            var k_val = p_k_in[unsafe_offset=idx]
            var v_val = p_v_in[unsafe_offset=idx]
            if isnan(k_val) or isinf(k_val) or isnan(v_val) or isinf(v_val):
                raise Error(
                    m5_error_json(
                        "M5_ERR_PREFILL",
                        "prefill",
                        "Non-finite value detected in prefill KV tensors",
                    )
                )
            p_k_cache[unsafe_offset=idx] = BFloat16(k_val)
            p_v_cache[unsafe_offset=idx] = BFloat16(v_val)

        self.current_len = seq_len

    def append_decode_token(
        mut self,
        k_token: List[Float32],
        v_token: List[Float32],
        pos: Int,
    ) raises:
        """Menambahkan Key dan Value 1 token baru pada posisi sequence p.

        Menegakkan invarian kontrak posisi:
        - pos == self.current_len
        - self.current_len + 1 <= self.capacity
        """
        if pos != self.current_len:
            var details = String(
                '{"expected_pos":',
                String(self.current_len),
                ',"actual_pos":',
                String(pos),
                "}",
            )
            raise Error(
                m5_error_json(
                    "M5_ERR_DECODE",
                    "decode",
                    "Position mismatch on KV append: expected current_len",
                    details,
                )
            )
        if self.current_len + 1 > self.capacity:
            var details = String(
                '{"current_len":',
                String(self.current_len),
                ',"capacity":',
                String(self.capacity),
                "}",
            )
            raise Error(
                m5_error_json(
                    "M5_ERR_CONTEXT_SIZE",
                    "kv_alloc",
                    "KV cache capacity exceeded on token append",
                    details,
                )
            )
        if len(k_token) != SLOT_DIM or len(v_token) != SLOT_DIM:
            raise Error(
                m5_error_json(
                    "M5_ERR_DECODE",
                    "decode",
                    "Decode token KV length mismatch against SLOT_DIM (2048)",
                )
            )

        var base_offset = pos * SLOT_DIM
        var p_k_in = k_token.unsafe_ptr()
        var p_v_in = v_token.unsafe_ptr()
        var p_k_cache = self.k.unsafe_ptr()
        var p_v_cache = self.v.unsafe_ptr()

        for idx in range(SLOT_DIM):
            var k_val = p_k_in[unsafe_offset=idx]
            var v_val = p_v_in[unsafe_offset=idx]
            if isnan(k_val) or isinf(k_val) or isnan(v_val) or isinf(v_val):
                raise Error(
                    m5_error_json(
                        "M5_ERR_DECODE",
                        "decode",
                        "Non-finite value detected in decode step KV token",
                    )
                )
            p_k_cache[unsafe_offset=base_offset + idx] = BFloat16(k_val)
            p_v_cache[unsafe_offset=base_offset + idx] = BFloat16(v_val)

        self.current_len += 1

    def get_k_slice(self, length: Int) raises -> List[Float32]:
        """Mengekstraksi slice Key rentang [0, length) sebagai Float32."""
        if length <= 0 or length > self.current_len:
            raise Error(
                m5_error_json(
                    "M5_ERR_DECODE",
                    "decode",
                    "Invalid slice length requested from LayerKVCache",
                )
            )
        var total_elems = length * SLOT_DIM
        var out = List[Float32]()
        out.resize(total_elems, Float32(0.0))

        var p_cache = self.k.unsafe_ptr()
        var p_out = out.unsafe_ptr()
        for idx in range(total_elems):
            p_out[unsafe_offset=idx] = Float32(p_cache[unsafe_offset=idx])
        return out^

    def get_v_slice(self, length: Int) raises -> List[Float32]:
        """Mengekstraksi slice Value rentang [0, length) sebagai Float32."""
        if length <= 0 or length > self.current_len:
            raise Error(
                m5_error_json(
                    "M5_ERR_DECODE",
                    "decode",
                    "Invalid slice length requested from LayerKVCache",
                )
            )
        var total_elems = length * SLOT_DIM
        var out = List[Float32]()
        out.resize(total_elems, Float32(0.0))

        var p_cache = self.v.unsafe_ptr()
        var p_out = out.unsafe_ptr()
        for idx in range(total_elems):
            p_out[unsafe_offset=idx] = Float32(p_cache[unsafe_offset=idx])
        return out^

    def clear(mut self):
        """Membersihkan buffer dan mereset status alokasi."""
        self.k.clear()
        self.v.clear()
        self.current_len = 0


struct FullKVCache(Movable):
    """KV Cache lengkap untuk seluruh 24 layer model Qwen1.5-MoE."""

    var layers: List[LayerKVCache]
    var capacity: Int

    def __init__(out self, capacity: Int) raises:
        self.capacity = capacity
        self.layers = List[LayerKVCache]()
        try:
            for layer_idx in range(NUM_LAYERS):
                self.layers.append(LayerKVCache(capacity, layer_idx))
        except e:
            raise Error(
                m5_error_json(
                    "M5_ERR_KV_ALLOC",
                    "kv_alloc",
                    String(
                        "Failed to allocate FullKVCache 24 layers: ", String(e)
                    ),
                )
            )

    def store_prefill_layer(
        mut self,
        layer_idx: Int,
        k_prompt: List[Float32],
        v_prompt: List[Float32],
        seq_len: Int,
    ) raises:
        if layer_idx < 0 or layer_idx >= NUM_LAYERS:
            raise Error(
                m5_error_json(
                    "M5_ERR_PREFILL",
                    "prefill",
                    String("Invalid layer_idx: ", String(layer_idx)),
                )
            )
        self.layers[layer_idx].store_prefill(k_prompt, v_prompt, seq_len)

    def append_decode_token_layer(
        mut self,
        layer_idx: Int,
        k_token: List[Float32],
        v_token: List[Float32],
        pos: Int,
    ) raises:
        if layer_idx < 0 or layer_idx >= NUM_LAYERS:
            raise Error(
                m5_error_json(
                    "M5_ERR_DECODE",
                    "decode",
                    String("Invalid layer_idx: ", String(layer_idx)),
                )
            )
        self.layers[layer_idx].append_decode_token(k_token, v_token, pos)

    def get_layer_k_slice(
        self, layer_idx: Int, length: Int
    ) raises -> List[Float32]:
        if layer_idx < 0 or layer_idx >= NUM_LAYERS:
            raise Error(
                m5_error_json(
                    "M5_ERR_DECODE",
                    "decode",
                    String("Invalid layer_idx: ", String(layer_idx)),
                )
            )
        return self.layers[layer_idx].get_k_slice(length)

    def get_layer_v_slice(
        self, layer_idx: Int, length: Int
    ) raises -> List[Float32]:
        if layer_idx < 0 or layer_idx >= NUM_LAYERS:
            raise Error(
                m5_error_json(
                    "M5_ERR_DECODE",
                    "decode",
                    String("Invalid layer_idx: ", String(layer_idx)),
                )
            )
        return self.layers[layer_idx].get_v_slice(length)

    def current_len(self) -> Int:
        if len(self.layers) > 0:
            return self.layers[0].current_len
        return 0

    def set_current_len(mut self, len_val: Int):
        for i in range(len(self.layers)):
            self.layers[i].current_len = len_val

    def increment_len(mut self):
        for i in range(len(self.layers)):
            self.layers[i].current_len += 1

    def clear(mut self):
        for i in range(len(self.layers)):
            self.layers[i].clear()
        self.layers.clear()
