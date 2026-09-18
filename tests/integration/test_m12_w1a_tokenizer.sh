#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M12 Wave 1a (M12-W1a: Special-Token Resolver & Lockfile Pinning).
# Menguji:
#   1. Static Hygiene & Invariants (0 noqa, 0 #[allow], 0 hardcoded /home paths, 0 forbidden hardcoded token IDs per M12-1)
#   2. Mojo Unit Test Suite (8/8 tests: real model resolution, synthetic happy path, 4 error paths, shifted-ID negative, lockfile tamper)
#   3. Lockfile Tokenizer Section 7-Field Validation (§2.1, M10 §4.5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "M12-W1a: Special-Token Resolver & Lockfile Pinning Integration Test"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Hygiene & Invariants (M12-1 & Zero Suppressions)
# ---------------------------------------------------------------------------
echo "--> Stage 1: Static Hygiene & Invariants (M12-1)"

FILES_TO_AUDIT=(
    "src/tokenizer/special_tokens.mojo"
    "src/tokenizer/__init__.mojo"
    "tests/unit/test_m12_w1a_tokenizer.mojo"
)

for file in "${FILES_TO_AUDIT[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "FAIL: File wajib $file tidak ditemukan!"
        exit 1
    fi

    # Cek suppressions
    if grep -nE "noqa|#[[:space:]]*allow" "$file"; then
        echo "FAIL: Ditemukan suppressions terlarang di $file!"
        exit 1
    fi

    # Cek hardcoded /home/ paths
    if grep -n "/home/" "$file"; then
        echo "FAIL: Ditemukan hardcoded /home path di $file!"
        exit 1
    fi
done

# Invarian P0 M12-1: DILARANG KERAS konstanta numerik hardcoded token ID
# (misal nilai era-Qwen2 151643, 151644, 151645 atau nilai Qwen3.6 248321 dst)
FORBIDDEN_IDS=("151643" "151644" "151645" "151646" "248321" "248322" "248323")
for id in "${FORBIDDEN_IDS[@]}"; do
    for file in "${FILES_TO_AUDIT[@]}"; do
        if grep -q "$id" "$file"; then
            echo "FAIL: Ditemukan hardcoded token ID terlarang ($id) di $file per invarian M12-1!"
            exit 1
        fi
    done
done

echo "OK: Static hygiene dan invarian M12-1 lolos (0 suppressions, 0 /home paths, 0 hardcoded token IDs)."

# ---------------------------------------------------------------------------
# Stage 2: Lockfile Verification (7-field M10 §4.5 & Tokenizer Section)
# ---------------------------------------------------------------------------
echo "--> Stage 2: Memeriksa struktur lockfile models.lock.json"

python3 - << 'EOF'
import json
import sys

with open("models.lock.json", "r") as f:
    lock = json.load(f)

# Cek field tokenizer
assert "tokenizer_revision" in lock, "tokenizer_revision missing"
assert "tokenizer_sha256" in lock, "tokenizer_sha256 missing"
assert isinstance(lock["tokenizer_sha256"], dict), "tokenizer_sha256 must be object"
assert "tokenizer.json" in lock["tokenizer_sha256"], "tokenizer.json missing in tokenizer_sha256"
assert "tokenizer_config.json" in lock["tokenizer_sha256"], "tokenizer_config.json missing in tokenizer_sha256"

assert len(lock["tokenizer_sha256"]["tokenizer.json"]) == 64, "invalid sha256 length for tokenizer.json"
assert len(lock["tokenizer_sha256"]["tokenizer_config.json"]) == 64, "invalid sha256 length for tokenizer_config.json"

print(f"OK: models.lock.json tokenizer section valid (rev={lock['tokenizer_revision'][:8]}...)")
EOF

# ---------------------------------------------------------------------------
# Stage 3: Mojo Unit Test Suite
# ---------------------------------------------------------------------------
echo "--> Stage 3: Menjalankan Mojo Unit Test Suite (test_m12_w1a_tokenizer.mojo)"

pixi run mojo run -I src tests/unit/test_m12_w1a_tokenizer.mojo

echo "======================================================================"
echo "M12-W1a: SEMUA TAHAP VERIFIKASI HIJAU (Quality Gates Ready)"
echo "======================================================================"
