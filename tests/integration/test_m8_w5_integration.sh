#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite: Milestone M8 Wave 5
# Integrasi M7 (O_DIRECT+LRU), M9 (Sketsa 40L), Continuation Stress (20x),
# Performance Baseline (p50/p95, N=10), dan Deviasi R9.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DISMOEN="./dismoen"
FIXTURES_DIR="fixtures"
TMP_DIR="$(mktemp -d -t dismoen_m8_w5_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "======================================================================="
echo "=== M8-W5 Master Test Suite: M7/M9 Integration, Stress, Perf, & SEC ==="
echo "======================================================================="

# Pastikan binary dismoen siap
if [ ! -f "$DISMOEN" ]; then
    echo "Binary dismoen tidak ditemukan, mengompilasi via pixi build..."
    pixi run build
fi

# -----------------------------------------------------------------------------
# Stage 1: Static Hygiene & Formatting Compliance
# -----------------------------------------------------------------------------
echo ">> [1/7] Memeriksa kepatuhan formatting, ruff, clippy, dan no-noqa..."
pixi run pre-commit run --all-files
echo "   PASS: Seluruh static hygiene checks (ruff, clippy, mojo format, no-noqa) lolos 100%."

# -----------------------------------------------------------------------------
# Stage 2: Integrasi M7 (O_DIRECT + LRU Cache Aktif, Gate G-M8-1 & SEC-4)
# -----------------------------------------------------------------------------
echo ">> [2/7] Menguji integrasi M7 (O_DIRECT + LRU Cache)..."
OUT_M7="$TMP_DIR/m7_gdn_state.bin"
STDOUT_M7="$TMP_DIR/m7_gdn_stdout.json"

"$DISMOEN" gdn \
    --model-dir "$FIXTURES_DIR" \
    --tokens "$FIXTURES_DIR/m8_tokens.json" \
    --output "$OUT_M7" \
    --layers 2 \
    --dk 32 \
    --dv 32 \
    --chunk-size 8 \
    --threads 1 \
    --use-odirect \
    --lru-capacity 100 \
    --run-id "M8-M7-INTEG" \
    --timing-profile > "$STDOUT_M7"

# Validasi flag M7 dan memori SEC-4 pada output JSON
python3 -c '
import json
with open("'"$STDOUT_M7"'") as f:
    d = json.load(f)
assert d["status"] == "success"
assert d["io_config"]["use_odirect"] is True
assert d["io_config"]["lru_capacity"] == 100
assert d["run_id"] == "M8-M7-INTEG"
# SEC-4: Memory peak <= 6 GiB
assert d["metrics"]["vmhwm_bytes"] <= 6 * 1024 * 1024 * 1024, "VmHWM exceeds 6 GiB"
assert "phases" in d
assert "timing_profile" in d
'

# Verifikasi numerik terhadap golden reference (Gate G-M8-1)
"$DISMOEN" compare \
    --reference "$FIXTURES_DIR/m8_state_naive.bin" \
    --candidate "$OUT_M7" \
    --gate G-M8-1 >/dev/null

echo "   PASS: Integrasi M7 lolos (O_DIRECT+LRU aktif, G-M8-1 PASS, VmHWM <= 6G)."

# -----------------------------------------------------------------------------
# Stage 3: Integrasi M9 (Sketsa Arsitektur Hybrid 40L: 10 GatedAttn + 30 GDN)
# -----------------------------------------------------------------------------
echo ">> [3/7] Memverifikasi sketsa arsitektur hybrid M9 (10 GatedAttn + 30 GDN)..."
python3 tools/arch/verify_m9_hybrid_sketch.py
echo "   PASS: Sketsa arsitektur hybrid M9 terverifikasi (invarian 30L GDN independen valid)."

# -----------------------------------------------------------------------------
# Stage 4: Multi-Chunk & Boundary Stress Test Matrix (20 Kombinasi)
# -----------------------------------------------------------------------------
echo ">> [4/7] Menjalankan matriks boundary stress test (C in {64..1024} x 4 split)..."
python3 tools/bench/test_gdn_boundary_stress.py
echo "   PASS: Seluruh 20 kombinasi boundary stress test terbukti memenuhi Gate G-M8-1."

# -----------------------------------------------------------------------------
# Stage 5: Long-Sequence Numerical Stability & O(1) Memory (s in {1K..32K})
# -----------------------------------------------------------------------------
echo ">> [5/7] Menguji stabilitas long-sequence (1K..32K) & scaling memori O(1)..."
python3 tools/bench/bench_gdn_stability.py
echo "   PASS: Stabilitas numerik 32K (Delta_max <= 1e-3) & Gate G-M8-2 terbukti."

# -----------------------------------------------------------------------------
# Stage 6: Performance Baseline Protocol & Profiling (Gate G-M8-3)
# -----------------------------------------------------------------------------
echo ">> [6/7] Menjalankan benchmark performa baseline (N=10 runs terukur, 2 warmup)..."
python3 tools/bench/bench_gdn_perf.py

# Verifikasi berkas artefak benchmark
TODAY_STR=$(date +%Y-%m-%d)
REPORT_MD="reports/${TODAY_STR}/M8-gdn-performance.md"
REPORT_CSV="reports/${TODAY_STR}/m8_gdn_perf.csv"

[ -f "$REPORT_MD" ] || { echo "FAIL: Laporan markdown $REPORT_MD tidak ditemukan"; exit 1; }
[ -f "$REPORT_CSV" ] || { echo "FAIL: Laporan CSV $REPORT_CSV tidak ditemukan"; exit 1; }
echo "   PASS: Benchmark performa selesai; Gate G-M8-3 PASS; laporan tersimpan di $REPORT_MD."

# -----------------------------------------------------------------------------
# Stage 7: Verifikasi Catatan Deviasi vs Paper Yang et al. [R9]
# -----------------------------------------------------------------------------
echo ">> [7/7] Memverifikasi dokumen catatan deviasi vs paper [R9]..."
DEV_DOC="docs/milestones/M8-deviations.md"
[ -f "$DEV_DOC" ] || { echo "FAIL: Dokumen deviasi $DEV_DOC tidak ditemukan"; exit 1; }

# Pastikan 4 deviasi tercatat dengan template normatif (Alasan, Trade-off, Mitigasi)
python3 -c '
doc = open("docs/milestones/M8-deviations.md").read()
assert "Deviasi 1" in doc, "Deviasi 1 hilang"
assert "Deviasi 2" in doc, "Deviasi 2 hilang"
assert "Deviasi 3" in doc, "Deviasi 3 hilang"
assert "Deviasi 4" in doc, "Deviasi 4 hilang"
assert "Alasan" in doc, "Bagian Alasan tidak ada"
assert "Trade-off" in doc, "Bagian Trade-off tidak ada"
assert "Mitigasi" in doc, "Bagian Mitigasi tidak ada"
'
echo "   PASS: Dokumen deviasi $DEV_DOC terverifikasi lengkap dan sesuai standar DoD M8."

echo "======================================================================="
echo "=== SEMUA PENGUJIAN M8-W5 LOLOS 100%: TERINTEGRASI, TERUKUR, & JUJUR ==="
echo "======================================================================="
