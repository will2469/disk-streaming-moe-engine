# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Skema error M9 dan penanganan failure fail-closed untuk Kimo Port CLI."""

from cli.errors import eprint_json
from cli.sys_utils import c_unlink
from format.types import json_escape
from std.sys.terminate import exit


# Konstanta Normatif Exit Codes M9 (docs/milestones/M9-port.md § Exit Codes)
comptime M9_ERR_INPUT = 1
comptime M9_ERR_ARCHITECTURE = 2
comptime M9_ERR_CONFIG = 3
comptime M9_ERR_MEMORY = 4
comptime M9_ERR_IO = 5
comptime M9_ERR_FORWARD = 6
comptime M9_ERR_QUANT = 7
comptime M9_ERR_OUTPUT = 8


def m9_error_json(
    error_code: Int,
    error_type: String,
    message: String,
    stage: String = "config",
    details_json: String = "",
) -> String:
    """Membentuk string JSON strict RFC 8259 untuk status error M9."""
    var out = String(
        '{"status":"error","error_code":',
        String(error_code),
        ',"error_type":"',
        json_escape(error_type),
        '","stage":"',
        json_escape(stage),
        '","message":"',
        json_escape(message),
        '"',
    )
    if details_json.byte_length() > 0:
        out = String(out, ',"details":', details_json)
    out = String(out, "}")
    return out


def fail_m9(
    error_code: Int,
    error_type: String,
    message: String,
    stage: String = "config",
    details_json: String = "",
    temp_file_to_clean: String = "",
) raises:
    """Menangani kegagalan fatal M9 dengan mencetak error JSON ke stderr dan exit code spesifik.
    """
    if temp_file_to_clean != "":
        _ = c_unlink(temp_file_to_clean)
    eprint_json(
        m9_error_json(
            error_code,
            error_type,
            message,
            stage=stage,
            details_json=details_json,
        )
    )
    exit(error_code)
