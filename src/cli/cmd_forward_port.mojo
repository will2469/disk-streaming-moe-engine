# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Handler subcommand forward-port (alias transisi ke dismoen forward)."""

from cli.cmd_forward import cmd_forward
from std.collections import List


def cmd_forward_port(args: List[String]) raises:
    """Alias transisi kimo forward-port menuju dismoen forward."""
    cmd_forward(args)
