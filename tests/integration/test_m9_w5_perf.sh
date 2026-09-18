#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M9 Wave 5 (M9-W5).
# Menguji:
#   1. Static formatting hygiene & zero suppression (Ruff, zero noqa/allow)
#   2. Binary build & CLI flags verification (--run-id, --timing-profile)
#   3. Gate G-M9-4 F2 KV cache scaling verification (e_KV <= 5%)
#   4. F1-Port bottom-up memory budget & F5 decode calibration (e_T <= 20%)
#   5. Performance benchmark protocol (Prefill N=5, Decode N=30, p50/p95, governor)
#   6. Sublayer breakdown telemetry (GDN, GatedAttn, MoE)
#   7. Bit-exact determinism invariant when profiling flags are disabled
#   8. Table §2.7 (TBM -> Measured) integration verification

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

DISMOEN="${DISMOEN:-$ROOT_DIR/dismoen}"
if [[ ! -x "$DISMOEN" && -x "$ROOT_DIR/build/bin/dismoen" ]]; then
    DISMOEN="$ROOT_DIR/build/bin/dismoen"
fi

MINI_CONFIG="fixtures/m9_port_config_mini.json"
TOKENS_FIXTURE="fixtures/m9_port_tokens.json"

TEST_DIR="/tmp/test_m9_w5_$$"
mkdir -p "$TEST_DIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "=== M9-W5 Master Integration Test Suite: Performance, KV Cache F2, & Calibration ==="

# ---------------------------------------------------------------------------
# 1. Static Formatting Hygiene & Zero Suppression
# ---------------------------------------------------------------------------
echo "--> Test 1: Static Formatting Hygiene & Zero Suppression"

# 1a. Ruff linting on benchmark & calibration tools
pixi run pre-commit run ruff --files tools/bench/bench_port_perf.py tools/bench/verify_port_calibration.py

# 1b. Check for forbidden suppressions (noqa, allow)
if grep -nE "noqa|#[[:space:]]*allow" tools/bench/bench_port_perf.py tools/bench/verify_port_calibration.py; then
    echo "FAIL: Ditemukan suppressions terlarang di tools/bench!"
    exit 1
fi

echo "PASS: Test 1 (Static hygiene & zero suppression valid)"

# ---------------------------------------------------------------------------
# 2. Binary Build & CLI Flags Verification
# ---------------------------------------------------------------------------
echo "--> Test 2: Binary Build & CLI Flags Verification"

pixi run build

if [ ! -x "$DISMOEN" ]; then
    echo "FAIL: Binary dismoen tidak ditemukan atau tidak executable di $DISMOEN!"
    exit 1
fi

# Pastikan flag --run-id dan --timing-profile menghasilkan metrics block yang valid
CLI_TEST_OUT=$("$DISMOEN" forward-port \
    --architecture qwen3.6 \
    --model-dir "$MINI_CONFIG" \
    --tokens "$TOKENS_FIXTURE" \
    --run-id "M9-CLI-CHECK" \
    --timing-profile)

echo "$CLI_TEST_OUT" | grep -q '"run_id": "M9-CLI-CHECK"' || {
    echo "FAIL: CLI flag --run-id tidak terdeteksi pada forward-port!"
    exit 1
}
echo "$CLI_TEST_OUT" | grep -q '"timing_profile"' || {
    echo "FAIL: CLI flag --timing-profile tidak terdeteksi pada forward-port!"
    exit 1
}
echo "$CLI_TEST_OUT" | grep -q '"gdn_percent"' || {
    echo "FAIL: Timing profile tidak berisi sublayer breakdown gdn_percent!"
    exit 1
}

echo "PASS: Test 2 (Binary build & CLI profiling flags siap dan terverifikasi)"

# ---------------------------------------------------------------------------
# 3. Gate G-M9-4, F1-Port Memory, & F5 Calibration Verification
# ---------------------------------------------------------------------------
echo "--> Test 3: Gate G-M9-4 F2 KV Cache Scaling & F5 Calibration"

CALIB_OUT=$("$PYTHON" tools/bench/verify_port_calibration.py)
echo "$CALIB_OUT"

# Verifikasi teks evaluasi
echo "$CALIB_OUT" | grep -q "STATUS KALIBRASI M9: SELURUH AMBANG BATAS LULUS 100% \[PASS\]" || {
    echo "FAIL: verify_port_calibration.py tidak lulus 100%!"
    exit 1
}

# Verifikasi laporan kalibrasi markdown dihasilkan hari ini
TODAY_STR=$(date +%Y-%m-%d)
CALIB_REPORT="reports/${TODAY_STR}/M9-calibration-f1-f2-f5.md"
if [ ! -f "$CALIB_REPORT" ]; then
    echo "FAIL: Laporan kalibrasi $CALIB_REPORT tidak ditemukan!"
    exit 1
fi

grep -F -q "**Verdict Gate G-M9-4**: **[PASS]**" "$CALIB_REPORT" || {
    echo "FAIL: Gate G-M9-4 tidak PASS di $CALIB_REPORT!"
    exit 1
}
grep -F -q "**Verdict Gate G-M9-2**: **[PASS]**" "$CALIB_REPORT" || {
    echo "FAIL: Gate G-M9-2 tidak PASS di $CALIB_REPORT!"
    exit 1
}
grep -F -q "**Verdict**: **[PASS]**" "$CALIB_REPORT" || {
    echo "FAIL: Kalibrasi F5 tidak PASS di $CALIB_REPORT!"
    exit 1
}

echo "PASS: Test 3 (Gate G-M9-4, F1-Port, dan F5 calibration PASS)"

# ---------------------------------------------------------------------------
# 4. Performance Protocol Execution (Prefill N=5, Decode N=30)
# ---------------------------------------------------------------------------
echo "--> Test 4: Performance Protocol (Prefill N=5, Decode N=30, p50/p95)"

BENCH_OUT=$("$PYTHON" tools/bench/bench_port_perf.py \
    --dismoen-bin "$DISMOEN" \
    --config-path "$MINI_CONFIG" \
    --warmup 2 \
    --n-prefill 5 \
    --n-decode 30)

echo "$BENCH_OUT"

# Verifikasi statistik p50/p95 dicetak
echo "$BENCH_OUT" | grep -q "Prefill walltime p50" || {
    echo "FAIL: Output benchmark tidak mengandung statistik prefill p50!"
    exit 1
}
echo "$BENCH_OUT" | grep -q "Decode walltime p50" || {
    echo "FAIL: Output benchmark tidak mengandung statistik decode p50!"
    exit 1
}
echo "$BENCH_OUT" | grep -q "Sublayer Breakdown" || {
    echo "FAIL: Output benchmark tidak mengandung breakdown sublayer!"
    exit 1
}

# Verifikasi CSV dan Markdown reports
PERF_CSV="reports/${TODAY_STR}/m9_port_perf.csv"
PERF_MD="reports/${TODAY_STR}/M9-port-performance.md"

if [ ! -f "$PERF_CSV" ] || [ ! -f "$PERF_MD" ]; then
    echo "FAIL: File laporan performa tidak dibuat di reports/${TODAY_STR}!"
    exit 1
fi

grep -q "M9-" "$PERF_CSV" || {
    echo "FAIL: Format Run-ID M9- tidak tercatat di $PERF_CSV!"
    exit 1
}
grep -F -q "Sublayer Breakdown" "$PERF_MD" || {
    echo "FAIL: Sublayer breakdown tidak ada di $PERF_MD!"
    exit 1
}
grep -F -q "p95" "$PERF_MD" || {
    echo "FAIL: Statistik p95 tidak ditemukan di $PERF_MD!"
    exit 1
}

echo "PASS: Test 4 (Performance protocol selesai dengan laporan p50/p95 lengkap)"

# ---------------------------------------------------------------------------
# 5. Bit-Exact Determinism Invariant (Profiling Flags Disabled)
# ---------------------------------------------------------------------------
echo "--> Test 5: Invarian Determinisme Output Stdout Tanpa Flag Profiling"

RUN1_OUT=$("$DISMOEN" forward-port \
    --architecture qwen3.6 \
    --model-dir "$MINI_CONFIG" \
    --tokens "$TOKENS_FIXTURE" \
    --threads 1)

RUN2_OUT=$("$DISMOEN" forward-port \
    --architecture qwen3.6 \
    --model-dir "$MINI_CONFIG" \
    --tokens "$TOKENS_FIXTURE" \
    --threads 1)

diff -u <(echo "$RUN1_OUT") <(echo "$RUN2_OUT") || {
    echo "FAIL: Output stdout forward-port tidak deterministik saat profiling dimatikan!"
    exit 1
}

echo "PASS: Test 5 (Output forward-port 100% bitwise deterministik saat profiling off)"

# ---------------------------------------------------------------------------
# 6. Verifikasi Table §2.7 di docs/01-architecture.md
# ---------------------------------------------------------------------------
echo "--> Test 6: Verifikasi Table §2.7 (TBM -> Measured)"

ARCH_FILE="docs/01-architecture.md"

grep -E -q "Measured.*Run-ID" "$ARCH_FILE" || {
    echo "FAIL: Table §2.7 tidak memiliki kolom Measured dan Run-ID!"
    exit 1
}
grep -F -q "0,50 GiB nominal" "$ARCH_FILE" || {
    echo "FAIL: W_res measured tidak ditemukan di Table §2.7!"
    exit 1
}
grep -F -q "10.240 B/tok" "$ARCH_FILE" || {
    echo "FAIL: M_KV measured tidak ditemukan di Table §2.7!"
    exit 1
}
grep -F -q "1,066 GB/tok" "$ARCH_FILE" || {
    echo "FAIL: B_tok measured tidak ditemukan di Table §2.7!"
    exit 1
}
grep -F -q "0,214 ms" "$ARCH_FILE" || {
    echo "FAIL: T_tok measured tidak ditemukan di Table §2.7!"
    exit 1
}
grep -F -q "13,5–16,8 GB GGUF" "$ARCH_FILE" || {
    echo "FAIL: Ukuran disk actual tidak ditemukan di Table §2.7!"
    exit 1
}

echo "PASS: Test 6 (Table §2.7 terisi angka nyata dan Run-ID lengkap)"

echo "==================================================================="
echo "ALL TESTS PASSED: Milestone M9 Wave 5 (M9-W5) IS COMPLETE!"
echo "==================================================================="
