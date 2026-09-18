// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! Binary entry point for dismoen-tools.

fn main() {
    let code = dismoen_tools::run_cli("dismoen-tools");
    std::process::exit(code);
}
