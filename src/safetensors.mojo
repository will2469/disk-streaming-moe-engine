# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Safetensors multi-shard reader façade (backward-compatible API).

Re-exports core safetensors parsing and I/O structures from `format` package.
"""

from format import (
    HEADER_MAX,
    STError,
    STHeader,
    TENSOR_MAX,
    TensorMeta,
    _dtype_or_fail,
    _dtype_size,
    _fail,
    _numel_or_fail,
    json_escape,
    parse_index,
    parse_index_to_dict,
    read_header,
    read_small_file,
    Scanner,
)
from format.reader import _open_shard
