#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M12 Wave 1b (M12-W1b: ChatML Renderer, Rejection Protocol, & Streaming Detokenizer).
# Menguji:
#   1. Static Hygiene & Invariants (0 noqa, 0 #[allow], 0 hardcoded /home paths, 0 hardcoded forbidden token IDs per M12-1)
#   2. Oracle Differential Parity (Jinja2 upstream template byte-exact 100% & Tokenizer fidelity @ pinned revision)
#   3. Mojo Unit Test Suite (13/13 tests: 5 normative branches, 9 negative rejections, streaming UTF-8 multi-byte boundary, tamper-ID)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "M12-W1b: ChatML Template Renderer & Streaming Detokenizer Test"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Hygiene & Invariants (M12-1 & Zero Suppressions)
# ---------------------------------------------------------------------------
echo "--> Stage 1: Static Hygiene & Invariants (M12-1)"

FILES_TO_AUDIT=(
    "src/tokenizer/chatml.mojo"
    "src/tokenizer/detokenizer.mojo"
    "src/tokenizer/__init__.mojo"
    "tests/unit/test_chatml_formatter.mojo"
    "tools/oracle/oracle_chatml.py"
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
# Stage 2: Oracle Differential Parity (Jinja2 Upstream Template & Tokenizer)
# ---------------------------------------------------------------------------
echo "--> Stage 2: Menjalankan Oracle Parity Check (tools/oracle/oracle_chatml.py)"

PYTHON_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" tools/oracle/oracle_chatml.py --check

# ---------------------------------------------------------------------------
# Stage 3: Mojo Unit Test Suite
# ---------------------------------------------------------------------------
echo "--> Stage 3: Menjalankan Mojo Unit Test Suite (test_chatml_formatter.mojo)"

pixi run mojo run -I src tests/unit/test_chatml_formatter.mojo

echo "======================================================================"
echo "M12-W1b: SEMUA TAHAP VERIFIKASI HIJAU (Quality Gate G-M12-1 Ready)"
echo "======================================================================"
