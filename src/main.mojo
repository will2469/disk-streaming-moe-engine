# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Kimo CLI entry point."""

from cli.cmd_check_index import cmd_check_index
from cli.cmd_decode import cmd_decode
from cli.cmd_forward import cmd_forward
from cli.cmd_head import cmd_head
from cli.cmd_layer import cmd_layer
from cli.errors import eprint_json, fail, fail_layer
from cli.m4_errors import m4_error_json
from cli.m5_errors import m5_error_json
from std.collections import List
from std.sys.arg import argv
from std.sys.terminate import exit


def main() raises:
    var args = argv()
    if len(args) < 2:
        fail(
            "USAGE",
            "pakai: kimo (check-index|head|layer|forward|decode) ...",
            "",
            "",
        )
    var cmd = String(args[1])
    if cmd == "check-index":
        var shards = List[String]()
        for i in range(2, len(args)):
            shards.append(String(args[i]))
        cmd_check_index(shards^)
    elif cmd == "head":
        var pass_args = List[String]()
        for i in range(len(args)):
            pass_args.append(String(args[i]))
        try:
            cmd_head(pass_args^)
        except e:
            var err_s = String(e)
            if err_s.startswith("{"):
                eprint_json(err_s)
            else:
                fail("INTERNAL_ERROR", err_s, "", "")
            exit(2)
    elif cmd == "layer":
        var pass_args = List[String]()
        for i in range(len(args)):
            pass_args.append(String(args[i]))
        try:
            cmd_layer(pass_args^)
        except e:
            var err_s = String(e)
            if err_s.startswith("{"):
                eprint_json(err_s)
            else:
                fail_layer("INTERNAL_ERROR", err_s, "attention", -1)
            exit(2)
    elif cmd == "forward":
        var pass_args = List[String]()
        for i in range(len(args)):
            pass_args.append(String(args[i]))
        try:
            cmd_forward(pass_args^)
        except e:
            var err_s = String(e)
            if err_s.startswith("{"):
                eprint_json(err_s)
            else:
                eprint_json(
                    m4_error_json(
                        "M4_ERR_LAYER_FORWARD",
                        "forward",
                        err_s,
                    )
                )
            exit(5)
    elif cmd == "decode":
        var pass_args = List[String]()
        for i in range(len(args)):
            pass_args.append(String(args[i]))
        try:
            cmd_decode(pass_args^)
        except e:
            var err_s = String(e)
            if err_s.startswith("{"):
                eprint_json(err_s)
            else:
                eprint_json(
                    m5_error_json(
                        "M5_ERR_DECODE",
                        "decode",
                        err_s,
                    )
                )
            exit(5)
    else:
        fail("USAGE", String("subcommand tak dikenal: ", cmd), "", "")
