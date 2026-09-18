# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Loader bobot terkuantisasi 4-bit on-the-fly untuk layer forward (M6-W4).

Alur kerja streaming:
1. pread skala FP16 dan bobot 4-bit ter-pack dari berkas model binary.
2. Dequantisasi on-the-fly via kernel SIMD ke buffer Float32/BF16.
3. Konstruksi struct bobot layer (AttentionWeights / SwigluWeights).
4. Discard buffer setelah komputasi forward selesai (lifetime otomatis Mojo).
"""

from core.config import ModelConfig
from format.quant_reader import (
    QuantModelIndex,
    QuantTensorEntry,
    pread_and_dequant_tensor,
    pread_and_dequant_tensor_f32,
)
from layers.attention import AttentionWeights
from layers.moe_loader import SharedExpertWeights
from layers.qkv import QKVWeights
from layers.swiglu import SwigluWeights
from std.collections import List


def load_one_quant_tensor_f32(
    index: QuantModelIndex, tensor_name: String
) raises -> List[Float32]:
    """Memuat dan mendekuantisasi satu tensor dari model quant ke List[Float32].
    """
    if tensor_name not in index.tensor_map:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"tensor not found in'
            ' quant model","tensor":"'
            + tensor_name
            + '"}'
        )
    var idx = index.tensor_map[tensor_name]
    ref entry = index.entries[idx]
    return pread_and_dequant_tensor_f32(index.file_path, entry)


def load_one_quant_tensor_bf16(
    index: QuantModelIndex, tensor_name: String
) raises -> List[BFloat16]:
    """Memuat dan mendekuantisasi satu tensor dari model quant ke List[BFloat16].
    """
    if tensor_name not in index.tensor_map:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"tensor not found in'
            ' quant model","tensor":"'
            + tensor_name
            + '"}'
        )
    var idx = index.tensor_map[tensor_name]
    ref entry = index.entries[idx]
    return pread_and_dequant_tensor(index.file_path, entry)


def load_quant_attention_weights(
    index: QuantModelIndex, layer_idx: Int, cfg: ModelConfig
) raises -> AttentionWeights:
    """Memuat bobot attention layer dari berkas quant on-the-fly."""
    var prefix = "model.layers." + String(layer_idx) + "."
    var norm_gamma = load_one_quant_tensor_f32(
        index, prefix + "input_layernorm.weight"
    )

    var attn_prefix = prefix + "self_attn."
    var w_q = load_one_quant_tensor_f32(index, attn_prefix + "q_proj.weight")
    var b_q = load_one_quant_tensor_f32(index, attn_prefix + "q_proj.bias")
    var w_k = load_one_quant_tensor_f32(index, attn_prefix + "k_proj.weight")
    var b_k = load_one_quant_tensor_f32(index, attn_prefix + "k_proj.bias")
    var w_v = load_one_quant_tensor_f32(index, attn_prefix + "v_proj.weight")
    var b_v = load_one_quant_tensor_f32(index, attn_prefix + "v_proj.bias")
    var qkv = QKVWeights(
        w_q=w_q^, b_q=b_q^, w_k=w_k^, b_k=b_k^, w_v=w_v^, b_v=b_v^
    )

    var w_o = load_one_quant_tensor_f32(index, attn_prefix + "o_proj.weight")
    var b_o = List[Float32]()  # o_proj tidak memiliki bias pada arsitektur Qwen

    return AttentionWeights(
        norm_gamma=norm_gamma^, qkv=qkv^, w_o=w_o^, b_o=b_o^
    )


def load_quant_routed_expert_weights(
    index: QuantModelIndex, layer_idx: Int, expert_id: Int, cfg: ModelConfig
) raises -> SwigluWeights:
    """Memuat bobot routed expert dari berkas quant on-the-fly."""
    var prefix = (
        "model.layers."
        + String(layer_idx)
        + ".mlp.experts."
        + String(expert_id)
        + "."
    )
    var w_gate = load_one_quant_tensor_f32(index, prefix + "gate_proj.weight")
    var w_up = load_one_quant_tensor_f32(index, prefix + "up_proj.weight")
    var w_down = load_one_quant_tensor_f32(index, prefix + "down_proj.weight")
    var hidden = cfg.hidden_size
    var inter = len(w_gate) // hidden
    return SwigluWeights(
        w_gate=w_gate^,
        w_up=w_up^,
        w_down=w_down^,
        hidden_dim=hidden,
        inter_dim=inter,
    )


def load_quant_shared_expert_weights(
    index: QuantModelIndex, layer_idx: Int, cfg: ModelConfig
) raises -> SharedExpertWeights:
    """Memuat bobot shared expert dari berkas quant on-the-fly."""
    var prefix = "model.layers." + String(layer_idx) + ".mlp.shared_expert."
    var w_gate = load_one_quant_tensor_f32(index, prefix + "gate_proj.weight")
    var w_up = load_one_quant_tensor_f32(index, prefix + "up_proj.weight")
    var w_down = load_one_quant_tensor_f32(index, prefix + "down_proj.weight")
    var hidden = cfg.hidden_size
    var inter = len(w_gate) // hidden
    var swiglu = SwigluWeights(
        w_gate=w_gate^,
        w_up=w_up^,
        w_down=w_down^,
        hidden_dim=hidden,
        inter_dim=inter,
    )

    var gate_prefix = (
        "model.layers." + String(layer_idx) + ".mlp.shared_expert_gate."
    )
    var w_gate_sh = load_one_quant_tensor_f32(index, gate_prefix + "weight")
    return SharedExpertWeights(swiglu=swiglu^, w_gate_sh=w_gate_sh^)
