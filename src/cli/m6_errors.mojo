# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Skema error M6 dan penanganan failure fail-closed untuk Kimo quantize CLI."""

from cli.sys_utils import c_unlink
from format.types import json_escape
from std.collections import List
from std.sys.terminate import exit


def m6_error_code_to_exit_code(code: String) -> Int:
    """Pemetaan kode error normatif M6 ke kode exit proses (SSOT)."""
    if code == "M6_ERR_INPUT":
        return 1
    elif code == "M6_ERR_QUANT":
        return 2
    elif code == "M6_ERR_DEQUANT":
        return 2
    elif code == "M6_ERR_OUTPUT":
        return 3
    elif code == "M6_ERR_VALIDATION":
        return 4
    return 1


def m6_error_json(
    code: String, stage: String, message: String, details_json: String = "{}"
) -> String:
    """Membentuk string JSON strict RFC 8259 untuk error status M6."""
    var det = details_json
    if det.byte_length() == 0:
        det = "{}"
    return String(
        '{"status":"error","error":{"code":"',
        json_escape(code),
        '","stage":"',
        json_escape(stage),
        '","message":"',
        json_escape(message),
        '","details":',
        det,
        "}}",
    )


def fail_m6(
    code: String,
    stage: String,
    message: String,
    details_json: String = "{}",
    cleanup_files: List[String] = List[String](),
) raises:
    """Membersihkan file sementara, mencetak JSON error ke stdout, lalu exit."""
    for i in range(len(cleanup_files)):
        var p = cleanup_files[i]
        if p.byte_length() > 0:
            _ = c_unlink(p)

    var err_doc = m6_error_json(code, stage, message, details_json)
    print(err_doc)
    exit(m6_error_code_to_exit_code(code))
