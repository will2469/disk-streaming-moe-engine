# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Core package: config, memory telemetry, and tensor chunk loader."""

from core.config import (
    CHUNK_MAX_BYTES,
    LoadMemoryTelemetry,
    ModelConfig,
    _contains,
)
from core.tensor_loader import _load_one_tensor_by_name, load_tensor_f32_chunked
