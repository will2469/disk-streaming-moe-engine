# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Skema error M7 dan penanganan failure fail-closed untuk Dismoen O_DIRECT dan LRU Cache."""

from cli.sys_utils import c_unlink, cleanup_run_resources
from format.types import json_escape
from std.collections import List
from std.sys.terminate import exit


def m7_error_code_to_exit_code(code: String) -> Int:
    """Pemetaan kode error normatif M7 ke kode exit proses (SSOT)."""
    if code == "M7_ERR_ODIRECT_ALIGNMENT":
        return 1
    elif code == "M7_ERR_FORMAT_ALIGNMENT":
        return 2
    elif code == "M7_ERR_ODIRECT_SHORT_READ":
        return 2
    elif code == "M7_ERR_ODIRECT_ENOSPC":
        return 3
    elif code == "M7_ERR_ODIRECT_EIO":
        return 3
    elif code == "M7_ERR_LRU_NO_VICTIM":
        return 4
    elif code == "M7_ERR_LRU_ALLOC":
        return 4
    elif code == "M7_ERR_LRU_CORRUPT":
        return 4
    return 1


def m7_error_json(
    code: String, stage: String, message: String, details_json: String = "{}"
) -> String:
    """Membentuk string JSON strict RFC 8259 untuk error status M7."""
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


def fail_m7(
    code: String,
    stage: String,
    message: String,
    details_json: String = "{}",
    run_dir: String = "",
    cleanup_files: List[String] = List[String](),
) raises:
    """Membersihkan file sementara dan direktori run, mencetak JSON error ke stdout, lalu exit.
    """
    cleanup_run_resources(run_dir, cleanup_files)
    var err_doc = m7_error_json(code, stage, message, details_json)
    print(err_doc)
    exit(m7_error_code_to_exit_code(code))
