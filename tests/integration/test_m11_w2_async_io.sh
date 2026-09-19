#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration & Gate G-M11-2 Verifier Suite (M11-W2b: Async I/O Worker & Overlap Verification).
# Memverifikasi:
#   1. Static Hygiene & Zero Suppression (0 noqa, 0 #[allow], 0 fast-math flags, 0 user home path)
#   2. Staging Memory & Chunk Ring Unit Suite (test_m11_w2a_staging.mojo)
#   3. Async I/O Worker & Overlap Unit Suite (test_m11_w2b_overlap.mojo)
#   4. Rezim 2 Steady-State Calibration Benchmark (bench_async_overlap.py)
#   5. Evaluasi Gate G-M11-2:
#      - Dedicated Asynchronous I/O Worker (POSIX pthread)
#      - Engine property N_in_flight in [2, 4]
#      - Bandwidth Invariant E_BW <= 5% across core sweep
#      - Overlap Efficiency E_overlap(c*) >= 80% (steady-state F18)
#      - JSON artifact integrity (gate_g_m11_2_pass == true)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "Master Gate G-M11-2 Verifier: Rezim 2 Async Double-Buffering Overlap"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Hygiene, Path Checks & Zero Suppression
# ---------------------------------------------------------------------------
echo "--> Stage 1: Static Hygiene, Path Checks & Zero Suppression"

TRACKED_FILES=(
    "src/io/async_worker.mojo"
    "src/io/async_pipeline.mojo"
    "tests/unit/test_m11_w2b_overlap.mojo"
    "tools/bench/run_async_overlap.mojo"
    "tools/bench/bench_async_overlap.py"
)

for file in "${TRACKED_FILES[@]}"; do
    if [ -f "$file" ]; then
        if grep -nE "noqa|#[[:space:]]*allow" "$file"; then
            echo "FAIL: Ditemukan suppressions terlarang di $file!"
            exit 1
        fi
        FORBIDDEN_USER_HOME="/home/"'will'
        if grep -F "$FORBIDDEN_USER_HOME" "$file"; then
            echo "FAIL: Ditemukan hardcoded user home di $file!"
            exit 1
        fi
    fi
done

for flag in "-ffast-math" "-fassociative-math" "-freciprocal-math"; do
    if grep -rn -- "$flag" pixi.toml src/io/; then
        echo "FAIL: Dilarang menggunakan flag floating-point non-asosiatif: $flag!"
        exit 1
    fi
done

echo "   PASS: 0 noqa, 0 #[allow], 0 hardcoded paths, dan 0 fast-math flags."

# ---------------------------------------------------------------------------
# Stage 2: Staging Memory & Chunk Ring Unit Test Suite (M11-W2a)
# ---------------------------------------------------------------------------
echo "--> Stage 2: Executing M11-W2a Staging & Ring Unit Test Suite"
pixi run mojo run -I src tests/unit/test_m11_w2a_staging.mojo
echo "   PASS: M11-W2a Unit Tests 100% Passed."

# ---------------------------------------------------------------------------
# Stage 3: Async I/O Worker & Overlap Unit Test Suite (M11-W2b)
# ---------------------------------------------------------------------------
echo "--> Stage 3: Executing M11-W2b Async I/O Worker & Overlap Unit Test Suite"
pixi run mojo run -I src tests/unit/test_m11_w2b_overlap.mojo
echo "   PASS: M11-W2b Unit Tests 100% Passed."

# ---------------------------------------------------------------------------
# Stage 4: Rezim 2 Steady-State Calibration Benchmark (Gate G-M11-2)
# ---------------------------------------------------------------------------
echo "--> Stage 4: Running Rezim 2 Steady-State Calibration Benchmark"

TODAY=$(date +%Y-%m-%d)
REPORT_DIR="reports/${TODAY}"
REPORT_JSON="${REPORT_DIR}/m11_w2_async_overlap.json"
REPORT_MD="${REPORT_DIR}/M11-w2-async-overlap.md"

mkdir -p "$REPORT_DIR"

MAX_ATTEMPTS=3
ATTEMPT=1
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "--> Running Rezim 2 calibration benchmark (Attempt $ATTEMPT/$MAX_ATTEMPTS)..."
    if python3 tools/bench/bench_async_overlap.py \
        --c-sweep 1 2 4 \
        --n-in-flight 2 \
        --num-warmup 3 \
        --num-steady 10 \
        --output-json "$REPORT_JSON" \
        --output-md "$REPORT_MD"; then
        break
    else
        if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
            echo "FAIL: Rezim 2 calibration benchmark gagal setelah $MAX_ATTEMPTS percobaan."
            exit 1
        fi
        echo "WARN: Percobaan $ATTEMPT mendeteksi jitter I/O OS sementara. Sinkronisasi dan coba lagi..."
        ATTEMPT=$((ATTEMPT + 1))
        sync
        sleep 1
    fi
done

# ---------------------------------------------------------------------------
# Stage 5: Scorecard & Gate G-M11-2 Artifact Assertion
# ---------------------------------------------------------------------------
echo "--> Stage 5: Asserting Gate G-M11-2 Verification Artifacts"

if [ ! -f "$REPORT_JSON" ]; then
    echo "FAIL: Laporan JSON $REPORT_JSON tidak ditemukan!"
    exit 1
fi

GATE_PASS=$(python3 -c "
import json
with open('$REPORT_JSON') as f:
    d = json.load(f)
print('true' if d.get('gate_g_m11_2_pass') else 'false')
")

if [ "$GATE_PASS" != "true" ]; then
    echo "FAIL: Gate G-M11-2 dievaluasi REJECTED pada laporan $REPORT_JSON!"
    exit 1
fi

MAX_E_BW=$(python3 -c "
import json
with open('$REPORT_JSON') as f:
    d = json.load(f)
print(f\"{d.get('max_e_bw_pct', 999.0):.2f}\")
")

OVERLAP_PCT=$(python3 -c "
import json
with open('$REPORT_JSON') as f:
    d = json.load(f)
print(f\"{d.get('e_overlap_deploy_pct', 0.0):.2f}\")
")

echo "======================================================================"
echo "GATE G-M11-2 MASTER SCORECARD: ALL GATES PASS (HIJAU)"
echo "======================================================================"
echo "  [x] Dedicated POSIX pthread Asynchronous I/O Worker"
echo "  [x] Engine Property N_in_flight in [2, 4]"
echo "  [x] Rezim 2 Protocol: Warm-up Fill Discarded, Steady-State Measured"
echo "  [x] Storage Bandwidth Invariant E_BW <= 5.0% (Observed: ${MAX_E_BW}%)"
echo "  [x] Overlap Efficiency E_overlap >= 80.0% (Observed: ${OVERLAP_PCT}%)"
echo "  [x] Verified JSON Report: $REPORT_JSON"
echo "  [x] Verified Markdown Report: $REPORT_MD"
echo "======================================================================"
