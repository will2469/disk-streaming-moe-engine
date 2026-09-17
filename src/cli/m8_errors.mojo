# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Skema error M8 dan penanganan failure fail-closed untuk Kimo GDN CLI."""

from cli.errors import eprint_json
from cli.sys_utils import c_unlink
from format.types import json_escape
from std.sys.terminate import exit


# Konstanta Error Codes M8
comptime M8_ERR_INPUT_INVALID = 1
comptime M8_ERR_CONFIG_INVALID = 2
comptime M8_ERR_MEMORY_ALLOC_FAILURE = 3
comptime M8_ERR_IO_ERROR = 4
comptime M8_ERR_GDN_FORWARD_ERROR = 5
comptime M8_ERR_OUTPUT_ERROR = 6
comptime M8_ERR_CHUNK_SIZE_ERROR = 7


def m8_error_json(
    error_code: Int, error_type: String, message: String
) -> String:
    """Membentuk string JSON strict RFC 8259 untuk status error M8."""
    return String(
        '{"status":"error","error_code":',
        String(error_code),
        ',"error_type":"',
        json_escape(error_type),
        '","message":"',
        json_escape(message),
        '"}',
    )


def fail_m8(
    error_code: Int,
    error_type: String,
    message: String,
    temp_file_to_clean: String = "",
) raises:
    """Menangani kegagalan fatal M8 dengan mencetak error JSON ke stderr dan exit code spesifik."""
    if temp_file_to_clean != "":
        _ = c_unlink(temp_file_to_clean)
    eprint_json(m8_error_json(error_code, error_type, message))
    exit(error_code)
