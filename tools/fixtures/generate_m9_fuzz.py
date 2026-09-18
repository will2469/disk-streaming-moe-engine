#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generator untuk fuzz corpus M9-W3 (25+ mutasi adversarial).

Menghasilkan kasus uji adversarial pada fixtures/m9-fuzz/ dan manifest.json
untuk memverifikasi tidak adanya crash/hang/OOM (0 crash, clean exit code 1-8).
"""

import json
import os


def main():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    os.chdir(repo_root)
    fuzz_dir = os.path.join("fixtures", "m9-fuzz")
    os.makedirs(fuzz_dir, exist_ok=True)

    base_gguf = os.path.join("fixtures", "m9_port_mini.gguf")
    with open(base_gguf, "rb") as f:
        valid_gguf_data = bytearray(f.read())

    cases = []

    # 1. Corrupt Magic 'XXXX'
    p1 = os.path.join(fuzz_dir, "corrupt_magic.gguf")
    d1 = bytearray(valid_gguf_data)
    d1[0:4] = b"XXXX"
    with open(p1, "wb") as f:
        f.write(d1)
    cases.append(
        {
            "id": "FUZZ_01_CORRUPT_MAGIC",
            "description": "GGUF magic diganti 'XXXX' -> ditolak di format detector",
            "args": [
                "--model-dir",
                p1,
                "--architecture",
                "qwen3.6",
                "--tokens",
                "fixtures/m9_port_tokens.json",
            ],
            "want_exit": 5,
            "want_err": "FORMAT_ERROR",
        }
    )

    # 2. Unsupported Version 99
    p2 = os.path.join(fuzz_dir, "unsupported_version.gguf")
    d2 = bytearray(valid_gguf_data)
    d2[4:8] = (99).to_bytes(4, "little")
    with open(p2, "wb") as f:
        f.write(d2)
    cases.append(
        {
            "id": "FUZZ_02_UNSUPPORTED_VERSION",
            "description": "GGUF version 99 tidak didukung",
            "args": [
                "--model-dir",
                p2,
                "--architecture",
                "qwen3.6",
                "--tokens",
                "fixtures/m9_port_tokens.json",
            ],
            "want_exit": 5,
            "want_err": "unsupported GGUF version",
        }
    )

    # 3. Truncated Header (12 bytes)
    p3 = os.path.join(fuzz_dir, "truncated_header.gguf")
    with open(p3, "wb") as f:
        f.write(valid_gguf_data[:12])
    cases.append(
        {
            "id": "FUZZ_03_TRUNCATED_HEADER",
            "description": "GGUF file terpotong hanya 12 bytes",
            "args": [
                "--model-dir",
                p3,
                "--architecture",
                "qwen3.6",
                "--tokens",
                "fixtures/m9_port_tokens.json",
            ],
            "want_exit": 1,
            "want_err": "GGUF_FILE_CORRUPT",
        }
    )

    # 4. Zero Byte File
    p4 = os.path.join(fuzz_dir, "zero_byte.gguf")
    with open(p4, "wb") as f:
        pass
    cases.append(
        {
            "id": "FUZZ_04_ZERO_BYTE",
            "description": "File kosong 0 byte",
            "args": [
                "--model-dir",
                p4,
                "--architecture",
                "qwen3.6",
                "--tokens",
                "fixtures/m9_port_tokens.json",
            ],
            "want_exit": 5,
            "want_err": "too short",
        }
    )

    # 5. F11b Exact Size Mismatch (+1 trailing byte)
    p5 = os.path.join(fuzz_dir, "f11b_size_mismatch.gguf")
    d5 = bytearray(valid_gguf_data)
    d5.append(0xEE)
    with open(p5, "wb") as f:
        f.write(d5)
    cases.append(
        {
            "id": "FUZZ_05_F11B_SIZE_MISMATCH",
            "description": "GGUF file kelebihan 1 byte -> F11b delta != 0",
            "args": [
                "--model-dir",
                p5,
                "--architecture",
                "qwen3.6",
                "--tokens",
                "fixtures/m9_port_tokens.json",
            ],
            "want_exit": 1,
            "want_err": "GGUF_FILE_CORRUPT",
        }
    )

    # 6. Truncated Data Section (-500 bytes)
    p6 = os.path.join(fuzz_dir, "truncated_data.gguf")
    with open(p6, "wb") as f:
        f.write(valid_gguf_data[:-500])
    cases.append(
        {
            "id": "FUZZ_06_TRUNCATED_DATA",
            "description": "GGUF data terpotong 500 bytes dari ukuran analitik",
            "args": [
                "--model-dir",
                p6,
                "--architecture",
                "qwen3.6",
                "--tokens",
                "fixtures/m9_port_tokens.json",
            ],
            "want_exit": 1,
            "want_err": "GGUF_FILE_CORRUPT",
        }
    )

    # 7. Wild Config Vocab (10M vocab -> CONFIG_MISMATCH)
    p7 = os.path.join(fuzz_dir, "wild_vocab_config.json")
    cfg7 = {
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "vocab_size": 10000000,
        "rms_norm_eps": 1e-6,
        "architecture": "qwen3.6",
    }
    with open(p7, "w") as f:
        json.dump(cfg7, f)
        f.write("\n")
    cases.append(
        {
            "id": "FUZZ_07_WILD_VOCAB_CONFIG",
            "description": "Vocab liar 10M memicu CONFIG_MISMATCH",
            "args": [
                "--model-dir",
                p7,
                "--architecture",
                "qwen3.6",
                "--tokens",
                "fixtures/m9_port_tokens.json",
            ],
            "want_exit": 3,
            "want_err": "CONFIG_MISMATCH",
        }
    )

    # 8. Wild Config Hidden (65536 hidden -> CONFIG_MISMATCH)
    p8 = os.path.join(fuzz_dir, "wild_hidden_config.json")
    cfg8 = {
        "hidden_size": 65536,
        "moe_intermediate_size": 65536,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "vocab_size": 1024,
        "rms_norm_eps": 1e-6,
        "architecture": "qwen3.6",
    }
    with open(p8, "w") as f:
        json.dump(cfg8, f)
        f.write("\n")
    cases.append(
        {
            "id": "FUZZ_08_WILD_HIDDEN_CONFIG",
            "description": "Dimensi tidak cocok memicu CONFIG_MISMATCH",
            "args": [
                "--model-dir",
                p8,
                "--architecture",
                "qwen3.6",
                "--tokens",
                "fixtures/m9_port_tokens.json",
            ],
            "want_exit": 3,
            "want_err": "CONFIG_MISMATCH",
        }
    )

    # 9. Architecture Mismatch (trial config on qwen3.6)
    cases.append(
        {
            "id": "FUZZ_09_ARCH_MISMATCH_TRIAL_TO_PORT",
            "description": "Model trial dijalankan pada flag --architecture qwen3.6",
            "args": [
                "--model-dir",
                "fixtures/m9_mismatch_layers.json",
                "--architecture",
                "qwen3.6",
                "--check-config-only",
            ],
            "want_exit": 3,
            "want_err": "CONFIG_MISMATCH",
        }
    )

    # 10. Empty Tokens Array
    p10 = os.path.join(fuzz_dir, "empty_tokens.json")
    with open(p10, "w") as f:
        json.dump({"tokens": [], "seq_len": 0}, f)
        f.write("\n")
    cases.append(
        {
            "id": "FUZZ_10_EMPTY_TOKENS",
            "description": "Array tokens kosong",
            "args": [
                "--model-dir",
                "fixtures/m9_port_config_mini.json",
                "--architecture",
                "qwen3.6",
                "--tokens",
                p10,
            ],
            "want_exit": 1,
            "want_err": "INPUT_ERROR",
        }
    )

    # 11. Missing Architecture Flag
    cases.append(
        {
            "id": "FUZZ_11_MISSING_ARCH_FLAG",
            "description": "Flag --architecture wajib tidak disertakan",
            "args": [
                "--model-dir",
                "fixtures/m9_port_config_mini.json",
                "--check-config-only",
            ],
            "want_exit": 2,
            "want_err": "MISSING_ARCHITECTURE",
        }
    )

    # 12. Invalid Architecture Flag Value
    cases.append(
        {
            "id": "FUZZ_12_INVALID_ARCH_VALUE",
            "description": "Nilai arsitektur tidak valid (--architecture llama)",
            "args": [
                "--model-dir",
                "fixtures/m9_port_config_mini.json",
                "--architecture",
                "llama",
                "--check-config-only",
            ],
            "want_exit": 2,
            "want_err": "UNSUPPORTED_ARCHITECTURE",
        }
    )

    # 13..27: Tambahan variasi mutasi GGUF & konfigurasi
    for idx in range(13, 28):
        mut_path = os.path.join(fuzz_dir, f"mutation_{idx:02d}.gguf")
        d_mut = bytearray(valid_gguf_data)
        # Corrupt offset di berbagai posisi
        corrupt_pos = 16 + (idx * 17) % (len(d_mut) - 32)
        d_mut[corrupt_pos] ^= 0xFF
        with open(mut_path, "wb") as f:
            f.write(d_mut)
        cases.append(
            {
                "id": f"FUZZ_{idx:02d}_MUTATION_CORRUPT_OFFSET_{corrupt_pos}",
                "description": f"Mutasi byte acak pada offset {corrupt_pos}",
                "args": [
                    "--model-dir",
                    mut_path,
                    "--architecture",
                    "qwen3.6",
                    "--tokens",
                    "fixtures/m9_port_tokens.json",
                ],
                "want_exit": 1,
                "want_err": "GGUF",
            }
        )

    manifest = {
        "description": "M9-W3 Fuzzing Corpus (27 Adversarial Test Cases)",
        "cases_count": len(cases),
        "cases": cases,
    }

    manifest_path = os.path.join(fuzz_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Generated {len(cases)} fuzz test cases in {fuzz_dir} and {manifest_path}")


if __name__ == "__main__":
    main()
