# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Skema error M5 dan penanganan failure fail-closed untuk Kimo decode CLI."""

from cli.errors import eprint_json
from cli.sys_utils import cleanup_run_resources
from format.types import json_escape
from std.collections import List
from std.sys.terminate import exit


def m5_error_code_to_exit_code(code: String) -> Int:
    if code == "M5_ERR_INPUT":
        return 1
    elif code == "M5_ERR_CONTEXT_SIZE":
        return 2
    elif code == "M5_ERR_KV_ALLOC":
        return 3
    elif code == "M5_ERR_PREFILL":
        return 4
    elif code == "M5_ERR_DECODE":
        return 5
    elif code == "M5_ERR_OUTPUT":
        return 6
    return 1


def m5_error_json(
    code: String, stage: String, message: String, details_json: String = "{}"
) -> String:
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


def fail_m5(
    code: String,
    stage: String,
    message: String,
    details_json: String = "{}",
    run_dir: String = "",
    tmp_files: List[String] = List[String](),
) raises:
    cleanup_run_resources(run_dir, tmp_files)
    eprint_json(m5_error_json(code, stage, message, details_json))
    exit(m5_error_code_to_exit_code(code))
