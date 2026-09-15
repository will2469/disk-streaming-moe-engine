# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Format safetensors package."""

from format.file_io import c_realpath, path_is_within, resolve_within_root
from format.index import parse_index, parse_index_to_dict
from format.reader import read_header, read_small_file
from format.scanner import Scanner
from format.types import (
    HEADER_MAX,
    STError,
    STHeader,
    TENSOR_MAX,
    TensorMeta,
    _dtype_or_fail,
    _dtype_size,
    _fail,
    _numel_or_fail,
    decode_bf16_le,
    decode_f32_le,
    error_json,
    json_escape,
)
