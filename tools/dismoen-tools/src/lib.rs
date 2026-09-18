// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! dismoen-tools: orchestration and verification tool for DISMOEN engine.

pub mod compare;
pub mod verify;

pub fn run_cli(bin_name: &str) -> i32 {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("verify") => verify::run(&args[2..]),
        Some("compare") => compare::run(&args[2..]),
        _ => {
            eprintln!(
                r#"{{"error_type":"USAGE","detail":"pakai: {} verify ... | {} compare <ref.bin> <cand.bin> [--gate G-M1-1]"}}"#,
                bin_name, bin_name
            );
            2
        }
    }
}
