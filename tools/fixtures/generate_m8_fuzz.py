#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generator corpus fuzz M8 (DoD M8 § Fuzzing, Gate G-M8-4).

Membangkitkan 27 kasus mutasi token, konfigurasi dimensi ekstrim, rentang chunk,
dan file integrity untuk memverifikasi ketahanan fail-closed tanpa crash/hang/OOM.
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "m8-fuzz"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Bangkitkan file-file mutasi token JSON
    (OUT_DIR / "tokens_empty.json").write_text(json.dumps({"tokens": [], "seq_len": 0}))
    (OUT_DIR / "tokens_negative.json").write_text(
        json.dumps({"tokens": [-5, 12], "seq_len": 2})
    )
    (OUT_DIR / "tokens_oov.json").write_text(
        json.dumps({"tokens": [999999], "seq_len": 1})
    )
    (OUT_DIR / "tokens_corrupt.json").write_text('{"tokens": [1, 2, 3')
    (OUT_DIR / "tokens_no_array.json").write_text(json.dumps({"seq_len": 3}))
    (OUT_DIR / "tokens_string.json").write_text(
        json.dumps({"tokens": "invalid_string_array"})
    )
    (OUT_DIR / "tokens_valid.json").write_text(
        json.dumps({"tokens": [1, 23, 45, 67], "seq_len": 4})
    )

    valid_tokens = str(OUT_DIR / "tokens_valid.json")
    model_fixtures = "fixtures"

    # 2. Definisikan 27 kasus fuzzing
    cases = [
        # --- Kategori 1: Token Mutasi (Exit 1 INPUT_INVALID) ---
        {
            "id": "01_empty_tokens",
            "args": ["--tokens", str(OUT_DIR / "tokens_empty.json")],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
        {
            "id": "02_negative_token",
            "args": ["--tokens", str(OUT_DIR / "tokens_negative.json")],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
        {
            "id": "03_out_of_vocab_token",
            "args": ["--tokens", str(OUT_DIR / "tokens_oov.json")],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
        {
            "id": "04_corrupt_json_syntax",
            "args": ["--tokens", str(OUT_DIR / "tokens_corrupt.json")],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
        {
            "id": "05_missing_tokens_key",
            "args": ["--tokens", str(OUT_DIR / "tokens_no_array.json")],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
        {
            "id": "06_tokens_string_type",
            "args": ["--tokens", str(OUT_DIR / "tokens_string.json")],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
        {
            "id": "07_nonexistent_tokens_file",
            "args": ["--tokens", str(OUT_DIR / "missing_file_404.json")],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
        {
            "id": "08_unknown_cli_flag_seed",
            "args": ["--tokens", valid_tokens, "--seed", "42"],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
        {
            "id": "09_missing_model_dir",
            "args": [
                "--tokens",
                valid_tokens,
                "--model-dir",
                "/nonexistent/models/qwen",
            ],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
        # --- Kategori 2: Dimensi & Checked Arithmetic (Exit 2 CONFIG_INVALID) ---
        {
            "id": "10_zero_layers",
            "args": ["--tokens", valid_tokens, "--layers", "0"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "11_negative_layers",
            "args": ["--tokens", valid_tokens, "--layers", "-5"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "12_layers_above_100",
            "args": ["--tokens", valid_tokens, "--layers", "101"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "13_extreme_layers_overflow",
            "args": ["--tokens", valid_tokens, "--layers", "1000000"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "14_zero_dk",
            "args": ["--tokens", valid_tokens, "--dk", "0"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "15_negative_dk",
            "args": ["--tokens", valid_tokens, "--dk", "-32"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "16_dk_above_4096",
            "args": ["--tokens", valid_tokens, "--dk", "4097"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "17_extreme_dk_overflow",
            "args": ["--tokens", valid_tokens, "--dk", "1000000000"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "18_zero_dv",
            "args": ["--tokens", valid_tokens, "--dv", "0"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "19_negative_dv",
            "args": ["--tokens", valid_tokens, "--dv", "-32"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "20_dv_above_4096",
            "args": ["--tokens", valid_tokens, "--dv", "4097"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "21_zero_threads",
            "args": ["--tokens", valid_tokens, "--threads", "0"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        {
            "id": "22_negative_threads",
            "args": ["--tokens", valid_tokens, "--threads", "-4"],
            "want_exit": 2,
            "want_err": "CONFIG_INVALID",
        },
        # --- Kategori 3: Chunk Size Limits (Exit 7 CHUNK_SIZE_ERROR) ---
        {
            "id": "23_chunk_size_zero",
            "args": ["--tokens", valid_tokens, "--chunk-size", "0"],
            "want_exit": 7,
            "want_err": "CHUNK_SIZE_ERROR",
        },
        {
            "id": "24_chunk_size_negative",
            "args": ["--tokens", valid_tokens, "--chunk-size", "-8"],
            "want_exit": 7,
            "want_err": "CHUNK_SIZE_ERROR",
        },
        {
            "id": "25_chunk_size_below_8",
            "args": ["--tokens", valid_tokens, "--chunk-size", "7"],
            "want_exit": 7,
            "want_err": "CHUNK_SIZE_ERROR",
        },
        {
            "id": "26_chunk_size_above_4096",
            "args": ["--tokens", valid_tokens, "--chunk-size", "4097"],
            "want_exit": 7,
            "want_err": "CHUNK_SIZE_ERROR",
        },
        # --- Kategori 4: Output / State Input Integrity ---
        {
            "id": "27_nonexistent_state_input",
            "args": [
                "--tokens",
                valid_tokens,
                "--state-input",
                str(OUT_DIR / "missing_state.bin"),
            ],
            "want_exit": 1,
            "want_err": "INPUT_INVALID",
        },
    ]

    manifest = {
        "model_dir": model_fixtures,
        "default_layers": 2,
        "default_dk": 32,
        "default_dv": 32,
        "default_chunk_size": 8,
        "cases": cases,
    }

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Generated {len(cases)} fuzz cases -> {manifest_path}")


if __name__ == "__main__":
    main()
