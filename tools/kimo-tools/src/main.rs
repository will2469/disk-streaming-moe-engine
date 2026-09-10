// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! kimo-tools: orchestration/verification (dispatcher subcommand).

mod verify;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(|s| s.as_str()) {
        Some("verify") => verify::run(&args[2..]),
        _ => {
            eprintln!(
                r#"{{"error_type":"USAGE","detail":"pakai: kimo-tools verify --lock <models.lock.json> --dir <model-dir>"}}"#
            );
            2
        }
    };
    std::process::exit(code);
}
