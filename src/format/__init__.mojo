# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Format safetensors package."""

from format.file_io import c_realpath, path_is_within, resolve_within_root
from format.index import parse_index, parse_index_to_dict
from format.quant_format import (
    CONFIGURED_MAX_NAME,
    CONFIGURED_MAX_NDIM,
    CONFIGURED_MAX_TENSORS,
    QUANT_DEFAULT_GROUP_SIZE,
    QUANT_FORMAT_NAME,
    QUANT_HEADER_SIZE,
    QUANT_SCALE_DTYPE,
    QUANTIZED_DTYPE_NAME,
    QuantHeader,
    QuantTensorMetadata,
    calculate_tensor_quant_size,
    compute_fp16_scale_ceil,
    float16_to_u16,
    is_allowed_group_size,
    pack_4bit_pair,
    safe_multiply_int,
    u16_to_float16,
    unpack_4bit_pair,
    validate_quant_header,
    validate_quant_payload,
    validate_tensor_meta,
)
from format.quant_reader import (
    QuantModelIndex,
    QuantTensorEntry,
    pread_and_dequant_tensor,
    pread_and_dequant_tensor_f32,
    pread_tensor_quant,
    scan_quant_file,
)
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
from format.gdns import read_gdns_v1, write_gdns_v1
from format.kmss import KmssMetadata, read_kmss_v1, write_kmss_v1
from format.sha256 import sha256
from format.format_detector import (
    FORMAT_GGUF,
    FORMAT_SAFETENSORS,
    FORMAT_UNKNOWN,
    detect_file_format,
    format_to_string,
)
from format.gguf import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q8_0,
    GGUFIndex,
    GGUFTensorInfo,
    dequantize_q3_k,
    dequantize_q4_k,
    dequantize_q8_0,
    ggml_type_to_string,
    parse_gguf_index,
    stream_gguf_tensor_f32,
)
