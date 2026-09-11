// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! kimo-tools: orchestration/verification (dispatcher subcommand).

mod compare;
mod verify;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(|s| s.as_str()) {
        Some("verify") => verify::run(&args[2..]),
        Some("compare") => compare::run(&args[2..]),
        _ => {
            eprintln!(
                r#"{{"error_type":"USAGE","detail":"pakai: kimo-tools verify ... | kimo-tools compare <ref.bin> <cand.bin> [--gate G-M1-1]"}}"#
            );
            2
        }
    };
    std::process::exit(code);
}
