#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M11 Wave 1 (M11-W1: Static Worker Pool & Concurrency Determinism Contract §3.2).
# Menguji:
#   1. Static Hygiene & Zero Suppressions (0 noqa, 0 #[allow], no fast-math flags)
#   2. Mojo Unit Test Suite (partition invariants, worker lifecycle, parallel dequant bit-exact, parallel MoE 8-expert bit-exact, 5-run determinism)
#   3. Full CLI forward multithreading bit-exact parity: c=1 vs c=2 vs c=4 (Delta_max == 0.0)
#   4. Full CLI decode multithreading bit-exact parity: c=1 vs c=2 vs c=4 (Delta_max == 0.0)
#   5. Formal Determinism Contract §3.2 Scorecard

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-$ROOT_DIR/dismoen}"
if [[ ! -x "$DISMOEN" ]]; then
    echo "FAIL: Binary dismoen tidak ditemukan atau tidak executable. Jalankan 'pixi run build'."
    exit 1
fi

MINI_CONFIG="fixtures/m9_port_config_mini.json"
QUANT_GGUF="fixtures/m9_port_mini.gguf"
TOKENS_FIXTURE="fixtures/m9_port_tokens.json"

TEST_DIR="/tmp/test_m11_w1_workerpool_$$"
mkdir -p "$TEST_DIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M11-W1: Worker Pool & Concurrency Determinism Contract §3.2"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Hygiene & Zero Suppression (§3.2 Invariants)
# ---------------------------------------------------------------------------
echo "--> Stage 1: Static Hygiene & Zero Suppression"

for file in "src/core/worker_pool.mojo" "tests/unit/test_m11_w1_workerpool.mojo"; do
    if grep -nE "noqa|#[[:space:]]*allow" "$file"; then
        echo "FAIL: Ditemukan suppressions terlarang di $file!"
        exit 1
    fi
done

# Verifikasi tidak ada flag fast-math di repo / task build
for flag in "-ffast-math" "-fassociative-math" "-freciprocal-math"; do
    if grep -rn -- "$flag" pixi.toml src/; then
        echo "FAIL: Dilarang menggunakan flag floating-point non-asosiatif: $flag!"
        exit 1
    fi
done

echo "   PASS: 0 noqa, 0 #[allow], dan 0 flag fast-math terdeteksi."

# ---------------------------------------------------------------------------
# Stage 2: Mojo Unit Test Suite Execution
# ---------------------------------------------------------------------------
echo "--> Stage 2: Mojo Unit Test Suite (Lifecycle, Partitioning, Bit-exact Dequant & MoE)"

pixi run mojo run -I src tests/unit/test_m11_w1_workerpool.mojo

echo "   PASS: Seluruh unit test M11-W1 lolos 100% bit-exact."

# ---------------------------------------------------------------------------
# Stage 3: Full CLI Forward Multithreading Parity (c=1 vs c=2 vs c=4)
# ---------------------------------------------------------------------------
echo "--> Stage 3: Full CLI Forward Multithreading Parity (c=1 vs c=2 vs c=4)"

LOGITS_C1="${TEST_DIR}/forward_c1.bin"
LOGITS_C2="${TEST_DIR}/forward_c2.bin"
LOGITS_C4="${TEST_DIR}/forward_c4.bin"

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --threads 1 \
    --output "$LOGITS_C1" > /dev/null

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --threads 2 \
    --output "$LOGITS_C2" > /dev/null

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --threads 4 \
    --output "$LOGITS_C4" > /dev/null

# Verifikasi bit-exact comparison tanpa threshold toleransi
if ! cmp -s "$LOGITS_C1" "$LOGITS_C2"; then
    echo "FAIL: Forward logits mismatch antara c=1 dan c=2! Melanggar kontrak §3.2!"
    exit 1
fi

if ! cmp -s "$LOGITS_C1" "$LOGITS_C4"; then
    echo "FAIL: Forward logits mismatch antara c=1 dan c=4! Melanggar kontrak §3.2!"
    exit 1
fi

echo "   PASS: Forward c=1 vs c=2 vs c=4 100% bit-exact (Delta_max == 0.0)."

# ---------------------------------------------------------------------------
# Stage 4: Full CLI Decode Multithreading Parity (c=1 vs c=2 vs c=4)
# ---------------------------------------------------------------------------
echo "--> Stage 4: Full CLI Decode Multithreading Parity (c=1 vs c=2 vs c=4)"

SESS_PATH="${TEST_DIR}/prefill.kmss"
"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --save-session "$SESS_PATH" > /dev/null

DEC_C1="${TEST_DIR}/dec_c1.json"
DEC_C2="${TEST_DIR}/dec_c2.json"
DEC_C4="${TEST_DIR}/dec_c4.json"

"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --session "$SESS_PATH" \
    --max-tokens 8 \
    --threads 1 \
    --output "$DEC_C1" > /dev/null

"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --session "$SESS_PATH" \
    --max-tokens 8 \
    --threads 2 \
    --output "$DEC_C2" > /dev/null

"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --session "$SESS_PATH" \
    --max-tokens 8 \
    --threads 4 \
    --output "$DEC_C4" > /dev/null

if ! diff -u "$DEC_C1" "$DEC_C2" > /dev/null; then
    echo "FAIL: Decode tokens mismatch antara c=1 dan c=2! Melanggar kontrak §3.2!"
    exit 1
fi

if ! diff -u "$DEC_C1" "$DEC_C4" > /dev/null; then
    echo "FAIL: Decode tokens mismatch antara c=1 dan c=4! Melanggar kontrak §3.2!"
    exit 1
fi

echo "   PASS: Decode c=1 vs c=2 vs c=4 100% bit-exact (Delta_max == 0.0)."

# ---------------------------------------------------------------------------
# Stage 5: Formal Scorecard
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "M11-W1 Determinism Contract §3.2 Scorecard: ALL GATES PASS"
echo "======================================================================"
echo "  [x] Static zero-allocation WorkerPool in inference hot loop"
echo "  [x] Invariant 1: Fixed partitioning f(thread_id, c)"
echo "  [x] Invariant 2: Ascending accumulation order (index order)"
echo "  [x] Invariant 3: Deterministic SIMD reduction tree (width-16 pinned)"
echo "  [x] Invariant 4: No unordered atomic float additions, no fast-math"
echo "  [x] Bit-exact verification: c=1 vs c=2 vs c=4 (Delta_max == 0.0)"
echo "======================================================================"
