# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Modul I/O disk-streaming-moe-engine (O_DIRECT, Staging, Ring, LRU, Telemetri)."""

from io.dio_probe import (
    BASE_BUFFER_CAPACITY,
    BASE_CHUNK_SIZE,
    DEFAULT_TARGET_ALIGN,
    DioAlignment,
    calculate_buffer_capacity,
    calculate_chunk_size,
    is_triple_aligned,
    probe_dio_alignment,
    round_up_dio,
    validate_dio_constraints,
)
from io.lru_cache import DynamicLRUCache
from io.odirect import ODirectReader, ReadToken
from io.staging_ring import (
    RING_SLOT_COMPUTING,
    RING_SLOT_EMPTY,
    RING_SLOT_IO_IN_FLIGHT,
    RING_SLOT_READY,
    ChunkRingBuffer,
    RingSlot,
    StagingMemory,
    ring_slot_state_name,
)
from io.telemetry import (
    EnvironmentTelemetry,
    collect_environment_telemetry,
)
