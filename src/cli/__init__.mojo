# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Kimo CLI subcommands and utilities."""

from cli.cmd_check_index import cmd_check_index
from cli.cmd_decode import cmd_decode
from cli.cmd_gdn import cmd_gdn
from cli.cmd_head import cmd_head
from cli.cmd_layer import cmd_layer
from cli.config_parser import parse_model_config, parse_tokens_json
from cli.errors import (
    basename,
    dirname,
    eprint_json,
    err_json,
    err_layer_json,
    fail,
    fail_layer,
    fail_routing_violation,
)
from cli.m8_errors import fail_m8, m8_error_json
from cli.io_utils import (
    atomic_write_attn_output,
    atomic_write_logits,
    atomic_write_moe_output,
    load_and_validate_activation,
)
from cli.oracle_parser import OracleRoutingData, parse_oracle_routing_json
from cli.sys_utils import (
    c_realpath,
    c_rename,
    c_unlink,
    get_proc_io_read_bytes,
    get_vmhwm_bytes,
    resolve_target_output,
    str_to_float,
    validate_shards_coverage,
)
