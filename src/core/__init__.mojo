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
from core.worker_pool import (
    WorkerPool,
    partition_range,
)
from core.topology import (
    MODE_ASYNC_DOUBLE_BUFFER,
    MODE_SYNC_FALLBACK,
    OS_RAM_RESERVE_BYTES,
    CoreAllocation,
    CpuInfo,
    CpuTopology,
    build_core_allocation,
    parse_cpu_list,
    probe_cpu_topology,
    probe_ram_available,
    read_hardware_lock_c_star,
    read_sysfs_string,
    validate_runtime_feasibility,
)
from core.prefix_cache import (
    PrefixCache,
    PrefixCacheEntry,
    PrefixLookupResult,
    compute_domain_key,
)
