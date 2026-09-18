#!/bin/bash
# ==============================================================================
# test_m4_w4_verdict.sh — Integration Test Suite M4-W4 (Verdict A->N->S & FAIL Cats)
#
# Memverifikasi DoD M4-W4:
# 1. Rust integration_m4 tests:
#    - Short-circuit F10-A -> F10-N -> F10-S
#    - Sifat diskrit n=80 (100% PASS vs 98.75% FAIL)
#    - Kesetaraan SET Tier-1 routing (order-insensitive)
#    - Hierarki pita delta (dtype-layout > bias-placement > rope-style > numeric-order)
#    - Toleransi loose hanya untuk numeric-order
#    - Penolakan noise floor BF16 pada gate G-M4-1
# 2. Python BF16 normative decode & F10-S round-trip tolerances:
#    - <u2 -> <<16 -> view f32 bit-exact vs torch
#    - float16 decode DILARANG (1.0 -> 1.875 terdeteksi)
#    - F10-S bound: eps_rel <= 5e-3, delta_max <= 0.15
# 3. Validasi identity & routing check pada ke-5 golden prompt oracle bins.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COMPARE_BIN="${COMPARE_BIN:-target/debug/dismoen-tools}"
FIXTURE_DIR="tools/fixtures"

if [ -x ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "M4-W4: Verdict A->N->S, Hard-FAIL Categories, and BF16 Decode Tests"
echo "======================================================================"

# 1. Jalankan Rust integration test suite (integration_m4.rs)
echo ">> [1/3] Menjalankan Cargo integration_m4 test suite..."
cargo test --manifest-path tools/dismoen-tools/Cargo.toml --test integration_m4

# 2. Jalankan Python BF16 test suite (test_m4_w4_bf16.py)
echo ">> [2/3] Menjalankan Python BF16 normative decode & F10-S suite..."
"$PYTHON_BIN" tests/unit/test_m4_w4_bf16.py

# 3. Evaluasi G-M4-1 & Tier-1 routing pada seluruh 5 prompt golden artifacts
echo ">> [3/3] Evaluasi Gate G-M4-1 & routing pada 5 prompt golden oracle..."
for i in {1..5}; do
    ref_bin="$FIXTURE_DIR/m4_prompt${i}_oracle.bin"
    r_dir="$FIXTURE_DIR/m4_prompt${i}_routing"
    [ -f "$ref_bin" ] || { echo "FAIL: $ref_bin tidak ditemukan"; exit 1; }

    # Identity check via compare
    REPORT=$("$COMPARE_BIN" compare \
        --ref "$ref_bin" \
        --cand "$ref_bin" \
        --gate G-M4-1 \
        --dim 151936 \
        --oracle-routing "$r_dir/routing_L0.json" \
        --cand-routing "$r_dir/routing_L0.json")

    STATUS=$(echo "$REPORT" | grep -o '"status": "[^"]*"' | head -n1 | cut -d'"' -f4)
    VERDICT=$(echo "$REPORT" | grep -o '"verdict": "[^"]*"' | head -n1 | cut -d'"' -f4)

    [ "$STATUS" = "MATCH" ] && [ "$VERDICT" = "PASS" ] || {
        echo "FAIL: Identity compare gagal pada prompt $i!"
        exit 1
    }
    echo "   PASS: Prompt $i golden oracle identity & routing L0 terverifikasi"
done

echo "======================================================================"
echo "SEMUA SUITE M4-W4 (VERDICT A->N->S & FAIL CATEGORIES) SUKSES (PASS)"
echo "======================================================================"
exit 0
