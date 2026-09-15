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
from layers.head import (
    HeadWeights,
    embedding_lookup,
    matmul_activation_head,
    forward_head,
    validate_logits,
)
from layers.mha import (
    build_causal_mask,
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
