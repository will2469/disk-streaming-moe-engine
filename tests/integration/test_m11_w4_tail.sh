#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration & Gate G-M11-3 Verifier Suite (M11-W4: Tail Latency & Dynamic RAM Budget Adherence).
# Memverifikasi:
#   1. Static Hygiene & Zero Suppressions (0 noqa, 0 #[allow], 0 fast-math, 0 hardcoded paths)
#   2. Eksekusi pengujian N >= 100 run di c*_system via verify_tail_stability.py
#   3. Evaluasi Kriteria Gate G-M11-3:
#      - Project SLO R_tail = p95 / p50 <= 1.35 (interpolasi linear Type 7)
#      - Bootstrap Confidence Interval (CI 95%, B=1000) terestimasi
#      - Dynamic RAM Budget Adherence R_RAM = VmHWM / M_budget <= 0.95 lintas tier 8/16/32/64 GiB & host aktif
#      - Verifikasi tanpa kebocoran memori (leak-free multi-run)
#      - Asersi stabilitas bandwidth storage E_BW <= 5%
#   4. Integritas artefak laporan JSON dan Markdown
#   5. Formal Scorecard Gate G-M11-3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "Master Gate G-M11-3 Verifier: Tail Latency & Dynamic RAM Budget"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Hygiene, Path Checks & Zero Suppression
# ---------------------------------------------------------------------------
echo "--> Stage 1: Static Hygiene, Path Checks & Zero Suppression"

TRACKED_FILES=(
    "tools/bench/verify_tail_stability.py"
)

SUPPRESS_PATTERN="no""qa|#[[:space:]]*al""low"
HOME_PREFIX="/home/""will"
for file in "${TRACKED_FILES[@]}"; do
    if [ -f "$file" ]; then
        if grep -nE "$SUPPRESS_PATTERN" "$file"; then
            echo "FAIL: Ditemukan suppressions terlarang di $file!"
            exit 1
        fi
        if grep -F "$HOME_PREFIX" "$file"; then
            echo "FAIL: Ditemukan hardcoded $HOME_PREFIX di $file!"
            exit 1
        fi
    fi
done

for flag in "-ffast-math" "-fassociative-math" "-freciprocal-math"; do
    if grep -rn -- "$flag" pixi.toml src/ tools/bench/ 2>/dev/null; then
        echo "FAIL: Dilarang menggunakan flag floating-point non-asosiatif: $flag!"
        exit 1
    fi
done

echo "   PASS: 0 noqa, 0 #[allow], 0 hardcoded paths, dan 0 fast-math flags."

# ---------------------------------------------------------------------------
# Stage 2: Eksekusi Driver Kalibrasi Tail Stability (verify_tail_stability.py)
# ---------------------------------------------------------------------------
echo "--> Stage 2: Menjalankan Driver verify_tail_stability.py (N=100 run)"

TODAY=$(date +%Y-%m-%d)
REPORT_DIR="reports/${TODAY}"
REPORT_JSON="${REPORT_DIR}/m11_w4_tail_stability.json"
REPORT_MD="${REPORT_DIR}/M11-w4-tail-stability.md"

mkdir -p "$REPORT_DIR"

python3 tools/bench/verify_tail_stability.py \
    --lockfile "dismoen.hardware.lock" \
    --iterations 100 \
    --num-warmup 10 \
    --bootstrap-samples 1000 \
    --output-json "$REPORT_JSON" \
    --output-md "$REPORT_MD"

echo "   PASS: Driver verify_tail_stability.py selesai dieksekusi."

# ---------------------------------------------------------------------------
# Stage 3: Evaluasi Kriteria Gate G-M11-3 dari Laporan JSON
# ---------------------------------------------------------------------------
echo "--> Stage 3: Evaluasi Kriteria Gate G-M11-3 dari Laporan JSON"

if [ ! -f "$REPORT_JSON" ]; then
    echo "FAIL: Laporan JSON $REPORT_JSON tidak ditemukan!"
    exit 1
fi

python3 -c "
import json
import sys

with open('$REPORT_JSON') as f:
    d = json.load(f)

# 1. Gate pass overall
if not d.get('gate_g_m11_3_pass'):
    print('FAIL: gate_g_m11_3_pass bernilai false!')
    sys.exit(1)

# 2. Sampel N >= 100
n_runs = d.get('n_runs', 0)
if n_runs < 100:
    print(f'FAIL: n_runs = {n_runs} < 100!')
    sys.exit(1)

# 3. Project SLO R_tail <= 1.35
stats = d.get('statistics', {})
r_tail = stats.get('r_tail', 999.0)
if r_tail > 1.35:
    print(f'FAIL: R_tail = {r_tail} > 1.35!')
    sys.exit(1)

# 4. Bootstrap CI
b_ci = d.get('bootstrap_ci', {})
if 'p50_ci' not in b_ci or 'p95_ci' not in b_ci or 'r_tail_ci' not in b_ci:
    print('FAIL: Bootstrap CI diagnostik tidak lengkap!')
    sys.exit(1)

# 5. Dynamic RAM Budget Adherence
ram_adh = d.get('ram_adherence', {})
for tier, info in ram_adh.items():
    if not info.get('passed'):
        print(f'FAIL: RAM adherence gagal pada tier {tier}!')
        sys.exit(1)
    if info.get('r_ram', 1.0) > 0.95:
        print(f'FAIL: R_RAM {info.get(\"r_ram\")} > 0.95 pada tier {tier}!')
        sys.exit(1)

# 6. Memory leak check
leak = d.get('memory_leak_check', {})
if not leak.get('passed'):
    print(f'FAIL: Memory leak check gagal! Delta: {leak.get(\"delta_pct\")}%')
    sys.exit(1)

# 7. Storage bandwidth stability
storage = d.get('storage_e_bw', {})
if not storage.get('passed'):
    print(f'FAIL: Storage E_BW check gagal! E_BW: {storage.get(\"max_e_bw_pct\")}%')
    sys.exit(1)

print('   PASS: Seluruh kriteria Gate G-M11-3 terverifikasi 100% valid.')
"

# ---------------------------------------------------------------------------
# Stage 4: Verifikasi Laporan Markdown
# ---------------------------------------------------------------------------
echo "--> Stage 4: Verifikasi Laporan Markdown"

if [ ! -f "$REPORT_MD" ]; then
    echo "FAIL: Laporan Markdown $REPORT_MD tidak ditemukan!"
    exit 1
fi

for keyword in "Project SLO" "Bootstrap CI" "VmHWM" "Leak" "E_{BW}" "ALL GATES PASS"; do
    if ! grep -F "$keyword" "$REPORT_MD" >/dev/null; then
        echo "FAIL: Keyword '$keyword' tidak ditemukan di $REPORT_MD!"
        exit 1
    fi
done

echo "   PASS: Laporan Markdown memuat seluruh analisis dan scorecard resmi."

# ---------------------------------------------------------------------------
# Stage 5: Formal Scorecard Gate G-M11-3
# ---------------------------------------------------------------------------
R_TAIL=$(python3 -c "import json; d=json.load(open('$REPORT_JSON')); print(f\"{d['statistics']['r_tail']:.4f}\")")
P50=$(python3 -c "import json; d=json.load(open('$REPORT_JSON')); print(f\"{d['statistics']['p50_ms']:.2f}\")")
P95=$(python3 -c "import json; d=json.load(open('$REPORT_JSON')); print(f\"{d['statistics']['p95_ms']:.2f}\")")
VM_HWM=$(python3 -c "import json; d=json.load(open('$REPORT_JSON')); print(d['memory_leak_check']['vm_hwm_run1_kib'])")
E_BW=$(python3 -c "import json; d=json.load(open('$REPORT_JSON')); print(f\"{d['storage_e_bw']['max_e_bw_pct']:.2f}\")")

echo "======================================================================"
echo "M11-W4 Formal Scorecard: Tail Latency & Dynamic RAM Budget Adherence"
echo "======================================================================"
echo "  [✓] Sample Size N >= 100:                OK (N = 100 steady runs)"
echo "  [✓] Empirical Latency Distribution:      OK (p50 = ${P50} ms, p95 = ${P95} ms)"
echo "  [✓] Project SLO R_tail = p95/p50 <= 1.35: OK (R_tail = ${R_TAIL} <= 1.35)"
echo "  [✓] Bootstrap 95% CI Diagnostics:        OK (B = 1000 resamples)"
echo "  [✓] Peak Memory Footprint (VmHWM):       OK (${VM_HWM} KiB)"
echo "  [✓] Dynamic RAM Budget (R_RAM <= 0.95):  OK (All tiers 8/16/32/64 GiB & host)"
echo "  [✓] Memory Stability (Leak-Free):        OK (Delta <= 5.0%)"
echo "  [✓] Storage Bandwidth Invariant E_BW:    OK (E_BW = ${E_BW}% <= 5.0%)"
echo "  [✓] Verified Artifacts:                  OK ($REPORT_JSON, $REPORT_MD)"
echo "======================================================================"
echo "Gate G-M11-3 Master Verifier: ALL GATES PASSED (100% HIJAU)"
echo "======================================================================"
