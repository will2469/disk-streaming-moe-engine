#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Certification Test Suite: Milestone M8 Wave 6
# Formal Certification of Gates G-M8-1, G-M8-2, G-M8-3,
# Real Model & M9 Integration Readiness, and Quality Phase DoD.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "======================================================================="
echo "=== MILESTONE M8 WAVE 6: MASTER QUALITY GATES & READINESS SUITE     ==="
echo "======================================================================="

# ----------------------------------------------------------------------
# Stage 1: Static Hygiene & Formatting Verification
# ----------------------------------------------------------------------
echo ">> [1/6] Memeriksa kepatuhan formatting & static hygiene..."
pixi run pre-commit run --all-files || {
    echo "FAIL: Pre-commit hooks gagal!"
    exit 1
}
echo "   PASS: Seluruh file lolos pre-commit tanpa supresi noqa / allow."

# ----------------------------------------------------------------------
# Stage 2: Gate G-M8-1 (100 Random Sequences Numerical Equivalence)
# ----------------------------------------------------------------------
echo ">> [2/6] [CORRECTNESS-FIRST] Sertifikasi Gate G-M8-1 (100 sekuens acak)..."
"$PYTHON" tools/bench/test_gdn_random100.py || {
    echo "FAIL: Gate G-M8-1 gagal pada 100 sekuens acak!"
    exit 1
}
echo "   PASS: Gate G-M8-1 TERSERTIFIKASI HIJAU (100/100 PASS, Delta_max <= 1e-3, threads=1)."

# ----------------------------------------------------------------------
# Stage 3: Gate G-M8-2 (O(1) Constant Peak Memory & 32K Context Stability)
# ----------------------------------------------------------------------
echo ">> [3/6] [O(1) MEMORY] Sertifikasi Gate G-M8-2 (Peak Memory 1K..32K)..."
"$PYTHON" tools/bench/bench_gdn_stability.py || {
    echo "FAIL: Gate G-M8-2 gagal pada pengujian stabilitas 32K / O(1) memory!"
    exit 1
}
echo "   PASS: Gate G-M8-2 TERSERTIFIKASI HIJAU (Delta VmHWM <= 10 MB, O(1) konstan)."

# ----------------------------------------------------------------------
# Stage 4: Gate G-M8-3 (Core Speedup >= 2.0x in-memory scan-only)
# ----------------------------------------------------------------------
echo ">> [4/6] [THROUGHPUT] Sertifikasi Gate G-M8-3 (Speedup Core >= 2.0x)..."
"$PYTHON" tools/bench/bench_gdn_perf.py || {
    echo "FAIL: Gate G-M8-3 gagal pada benchmark baseline performa!"
    exit 1
}
echo "   PASS: Gate G-M8-3 TERSERTIFIKASI HIJAU (speedup_core p50 >= 2.0x)."

# ----------------------------------------------------------------------
# Stage 5: Integration Readiness (Real Model & M9 Hybrid Architecture)
# ----------------------------------------------------------------------
echo ">> [5/6] [READINESS] Memverifikasi kesiapan model asli & arsitektur M9..."
"$PYTHON" tools/bench/test_gdn_real_model.py || {
    echo "FAIL: Integration readiness test gagal!"
    exit 1
}
echo "   PASS: Integration Readiness model asli & M9 40-layer sketch TERVERIFIKASI."

# ----------------------------------------------------------------------
# Stage 6: Quality Gate Documentation & Definition of Done (§5.4)
# ----------------------------------------------------------------------
echo ">> [6/6] [GOVERNANCE] Memverifikasi kelengkapan dokumen deviasi & DoD Fase GDN..."
# 1. Pastikan dokumen deviasi R9 ada dan non-kosong
test -f "docs/milestones/M8-deviations.md" || {
    echo "FAIL: docs/milestones/M8-deviations.md tidak ditemukan!"
    exit 1
}
grep -q "Deviasi 1:" "docs/milestones/M8-deviations.md" || {
    echo "FAIL: Format deviasi R9 tidak lengkap!"
    exit 1
}

# 2. Pastikan docs/04-quality.md §5.4 mencatat Fase GDN selesai
grep -q "Fase GDN" "docs/04-quality.md" || {
    echo "FAIL: Fase GDN tidak ditemukan di docs/04-quality.md!"
    exit 1
}

echo "   PASS: Seluruh dokumentasi deviasi dan DoD Fase GDN sesuai standar."

echo "======================================================================="
echo "=== SELURUH SERTIFIKASI M8-W6 SUKSES 100%: GATES & READINESS HIJAU  ==="
echo "======================================================================="
