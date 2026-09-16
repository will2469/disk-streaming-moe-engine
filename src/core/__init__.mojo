# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Core package: config, memory telemetry, and tensor chunk loader."""

from core.config import (
    CHUNK_MAX_BYTES,
    LoadMemoryTelemetry,
    ModelConfig,
)
from core.tensor_loader import (
    ShardHeaderCache,
    _load_one_tensor_by_name,
    load_tensor_f32_chunked,
    read_shard_header,
)
from core.f3b_f5 import (
    B_TOK_DISK_BYTES,
    F3bTraffic,
    F5Forecast,
    KV_LAYER_TOTAL_SLOT_BYTES,
    KV_SLOT_BYTES_PER_LAYER,
)
