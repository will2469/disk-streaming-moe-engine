# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Neural network layers and kernels."""

from layers.attention import (
    AttentionWeights,
    forward_attention_block,
    load_layer_attention_weights,
    o_project,
)
from layers.forward_layer import (
    LayerTiming,
    forward_attention_decode_step,
    forward_attention_step,
    forward_single_layer,
    moe_combine_no_residual,
)
from layers.head import (
    HeadWeights,
    embedding_lookup,
    matmul_activation_head,
    forward_head,
    validate_logits,
)
from layers.mha import (
    build_causal_mask,
    mha_decode_step,
    mha_forward,
    softmax_row_stable,
    verify_causal_mask_property,
)
from layers.moe import (
    moe_aggregate_forward,
    shared_gate_forward,
)
from layers.moe_block import forward_moe_block
from layers.moe_loader import (
    SharedExpertWeights,
    load_layer_routed_expert_weights,
    load_layer_shared_expert_weights,
)
from layers.qkv import (
    QKVWeights,
    load_layer_qkv_weights,
    qkv_forward,
    qkv_project,
)
from layers.qkv_bias import (
    collect_attention_bias_names,
    validate_attention_bias_in_index,
    validate_bias_count,
)
from layers.residual import add_residual
from layers.rmsnorm import rmsnorm
from layers.rope import apply_rope, rope_rotate_half, verify_rope_isometry
from layers.router import (
    load_layer_router_weights,
    router_forward,
    router_project,
    router_softmax,
)
from layers.router_types import RouterConfig, RoutingInfo
from layers.swiglu import SwigluWeights, sigmoid_f32, silu_f32, swiglu_forward
from layers.topk import select_topk
from layers.kv_cache import (
    BYTES_PER_SLOT_PER_LAYER,
    DEFAULT_MAX_POS,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_LAYERS,
    SLOT_DIM,
    FullKVCache,
    LayerKVCache,
    MemoryBudget,
    compute_kv_cache_bytes,
    validate_context_bounds,
)
from layers.decode_loop import (
    DecodeStepContext,
    compare_tensors_loose,
    recompute_attention_at_position,
)
from layers.gdn import (
    GDNConfig,
    GDNState,
    apply_wy_chunk_update,
    chunked_gdn_scan,
    compose_chunk_operators,
    compute_wy_coefficients,
    extract_chunk_operator,
    project_tokens_to_kv_beta,
)
