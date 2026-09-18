# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah compare CLI kimo (F10 numerical equivalence)."""

from cli.errors import fail
from cli.sys_utils import c_access_r
from std.collections import List
from std.ffi import external_call
from std.sys.terminate import exit


def c_system(cmd: String) -> Int:
    var b = cmd.as_bytes()
    var z = List[UInt8]()
    for i in range(len(b)):
        z.append(b[i])
    z.append(0)
    var ret = external_call["system", Int32](z.unsafe_ptr())
    return Int(ret)


def find_compare_binary() -> String:
    var candidates = List[String]()
    candidates.append("target/release/dismoen-tools")
    candidates.append("target/debug/dismoen-tools")
    candidates.append("target/release/kimo-tools")
    candidates.append("target/debug/kimo-tools")
    candidates.append("tools/kimo-tools/target/release/dismoen-tools")
    candidates.append("tools/kimo-tools/target/debug/dismoen-tools")
    candidates.append("tools/kimo-tools/target/release/kimo-tools")
    candidates.append("tools/kimo-tools/target/debug/kimo-tools")
    for i in range(len(candidates)):
        if c_access_r(candidates[i]):
            return candidates[i]
    return ""


def cmd_compare(args: List[String]) raises:
    if len(args) < 3:
        fail(
            "USAGE",
            (
                "pakai: dismoen compare --reference <ref.bin> --candidate"
                " <cand.bin> [--tolerance <tol>]"
            ),
            "",
            "",
        )

    var bin_path = find_compare_binary()
    var full_cmd: String
    if bin_path != "":
        full_cmd = bin_path + " compare"
    else:
        full_cmd = "python3 tools/compare.py"

    for i in range(2, len(args)):
        var arg = args[i]
        full_cmd += " '" + arg + "'"

    var ret = c_system(full_cmd)
    var exit_code = (ret >> 8) & 255
    exit(exit_code)
