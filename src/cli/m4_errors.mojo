# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Skema error M4 dan penanganan failure fail-closed untuk Kimo CLI."""

from cli.errors import eprint_json
from cli.sys_utils import cleanup_run_resources
from format.types import json_escape
from std.collections import List
from std.sys.terminate import exit


def m4_error_code_to_exit_code(code: String) -> Int:
    if code == "M4_ERR_INPUT":
        return 1
    elif code == "M4_ERR_INDEX" or code == "M4_ERR_INDEX_VALIDATION":
        return 2
    elif code == "M4_ERR_MEMORY":
        return 3
    elif code == "M4_ERR_SHARD_IO":
        return 4
    elif code == "M4_ERR_LAYER_FORWARD":
        return 5
    elif code == "M4_ERR_OUTPUT":
        return 6
    elif code == "M4_ERR_COMPARE":
        return 7
    return 1


def m4_error_json(
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


def fail_m4(
    code: String,
    stage: String,
    message: String,
    details_json: String = "{}",
    run_dir: String = "",
    tmp_files: List[String] = List[String](),
) raises:
    cleanup_run_resources(run_dir, tmp_files)
    eprint_json(m4_error_json(code, stage, message, details_json))
    exit(m4_error_code_to_exit_code(code))
