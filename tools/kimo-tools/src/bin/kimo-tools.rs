// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! Binary entry point for kimo-tools (backwards compatibility wrapper).

fn main() {
    let code = dismoen_tools::run_cli("kimo-tools");
    std::process::exit(code);
}
