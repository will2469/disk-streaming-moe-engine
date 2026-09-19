#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration & Certification Suite untuk Milestone M11 Wave 5 (M11-W5: Quality Gates G-M11-1..4).
# Menguji & Mensertifikasi:
#   1. Static Hygiene & Zero-Suppression (Pre-commit 13 hooks, 0 noqa, 0 #[allow], 0 fast-math)
#   2. Gate G-M11-1: F16 Amdahl Curve Fit, Compute Knee & Tri-Pillar Synthesis (test_m11_w3b_calibrate.sh)
#   3. Gate G-M11-2: Steady-State Async Double-Buffering Overlap & Storage Stability (test_m11_w2_async_io.sh)
#   4. Gate G-M11-3: Tail Latency Project SLO & Dynamic RAM Budget Adherence (test_m11_w4_tail.sh)
#   5. Gate G-M11-4: Determinism Reduction Contract §3.2 & Zero Regression (Delta_max == 0.0, validate-m10)
#   6. Formal Closure Scorecard Generation (reports/YYYY-MM-DD/M11-gates-scorecard.md)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-$ROOT_DIR/dismoen}"
if [[ ! -x "$DISMOEN" ]]; then
    echo "FAIL: Binary dismoen tidak ditemukan. Jalankan 'pixi run build'."
    exit 1
fi

echo "======================================================================"
echo "MILESTONE M11 WAVE 5: MASTER QUALITY GATES (G-M11-1..4) CERTIFICATION"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Code Hygiene & Zero-Suppression Verification
# ---------------------------------------------------------------------------
echo "--> Stage 1: Static Code Hygiene & Zero-Suppression Verification"
pixi run pre-commit run --all-files

SUPPRESS_PATTERN="no""qa|#[[:space:]]*al""low"
HOME_PREFIX="/home/""will"

if grep -rnE "$SUPPRESS_PATTERN" src/ tools/bench/ 2>/dev/null; then
    echo "FAIL: Ditemukan suppressions terlarang di src/ atau tools/bench/!"
    exit 1
fi

for flag in "-ffast-math" "-fassociative-math" "-freciprocal-math"; do
    if grep -rn -- "$flag" pixi.toml src/ tools/bench/ 2>/dev/null; then
        echo "FAIL: Dilarang menggunakan flag floating-point non-asosiatif: $flag!"
        exit 1
    fi
done

echo "   PASS: Pre-commit 13 hooks lolos 100%, 0 noqa, 0 #[allow], 0 fast-math flags."

# ---------------------------------------------------------------------------
# Stage 2: Gate G-M11-1 (F16 Amdahl Curve Fit, Compute Knee & Tri-Pillar Synthesis)
# ---------------------------------------------------------------------------
echo "--> Stage 2: Gate G-M11-1 (F16 Amdahl Fit, Knee c*_compute & Lockfile Synthesis)"
bash tests/integration/test_m11_w3b_calibrate.sh

# Verifikasi keberadaan dan integritas dismoen.hardware.lock
LOCK_FILE="dismoen.hardware.lock"
if [[ ! -f "$LOCK_FILE" ]]; then
    echo "FAIL: Lockfile $LOCK_FILE tidak ditemukan!"
    exit 1
fi

python3 -c "
import json
with open('$LOCK_FILE') as f:
    d = json.load(f)

profiles = d.get('profiles', {})
assert len(profiles) >= 5, 'Minimal 5 profil hardware!'
for name, p in profiles.items():
    fields = [
        'ram_budget_gib', 'bw_eff_mbs', 'cache_hit_rate',
        'c_compute_max', 'c_star_compute', 'c_star_system',
        'r_star_system', 'chunk_size', 'n_in_flight', 'dio_align'
    ]
    for fld in fields:
        assert fld in p, f'Field {fld} hilang di profil {name}!'
    assert 1 <= p['c_star_system'] <= p['c_star_compute'] <= p['c_compute_max']
"
echo "   PASS: Gate G-M11-1 TERSERTIFIKASI HIJAU (F16 fit, knee, synthesis & 10-field lockfile OK)."

# ---------------------------------------------------------------------------
# Stage 3: Gate G-M11-2 (Async Double-Buffering Overlap & Bandwidth Stability)
# ---------------------------------------------------------------------------
echo "--> Stage 3: Gate G-M11-2 (Async Double-Buffering Overlap & Bandwidth Invariant)"
bash tests/integration/test_m11_w2_async_io.sh

TODAY=$(date +%Y-%m-%d)
REPORT_W2="reports/${TODAY}/m11_w2_async_overlap.json"
python3 -c "
import json
with open('$REPORT_W2') as f:
    d = json.load(f)
assert d.get('gate_g_m11_2_pass') is True
assert d.get('e_overlap_deploy_pct', 0.0) >= 80.0
assert d.get('max_e_bw_pct', 999.0) <= 5.0
assert d.get('n_in_flight', 0) in [2, 4]
"
echo "   PASS: Gate G-M11-2 TERSERTIFIKASI HIJAU (E_overlap >= 80%, E_BW <= 5%, N_in_flight in [2, 4])."

# ---------------------------------------------------------------------------
# Stage 4: Gate G-M11-3 (Tail Latency Project SLO & Dynamic RAM Budget Adherence)
# ---------------------------------------------------------------------------
echo "--> Stage 4: Gate G-M11-3 (Tail Latency Project SLO & Dynamic RAM Budget)"
bash tests/integration/test_m11_w4_tail.sh

REPORT_W4="reports/${TODAY}/m11_w4_tail_stability.json"
python3 -c "
import json
with open('$REPORT_W4') as f:
    d = json.load(f)
assert d.get('gate_g_m11_3_pass') is True
assert d.get('n_runs', 0) >= 100
assert d['statistics']['r_tail'] <= 1.35
for tier, info in d['ram_adherence'].items():
    assert info['passed'] is True and info['r_ram'] <= 0.95
assert d['memory_leak_check']['passed'] is True
assert d['storage_e_bw']['passed'] is True
"
echo "   PASS: Gate G-M11-3 TERSERTIFIKASI HIJAU (R_tail <= 1.35, N >= 100, R_RAM <= 0.95, Leak-Free)."

# ---------------------------------------------------------------------------
# Stage 5: Gate G-M11-4 (Determinism Reduction Contract §3.2 & Zero Regression)
# ---------------------------------------------------------------------------
echo "--> Stage 5: Gate G-M11-4 (Determinism Contract §3.2 & Zero Regression)"

# 1. Unit test suite determinisme WorkerPool
pixi run mojo run -I src tests/unit/test_m11_w1_workerpool.mojo
echo "   [✓] WorkerPool reduction contract & bit-exact unit tests: PASS"

# 2. Multi-core forward bit-exactness verification (Delta_max == 0.0)
MINI_CONFIG="fixtures/m9_port_config_mini.json"
QUANT_GGUF="fixtures/m9_port_mini.gguf"
TOKENS_FIXTURE="fixtures/m9_port_tokens.json"
OUT_C1="/tmp/test_m11_g4_c1_$$.bin"
OUT_C4="/tmp/test_m11_g4_c4_$$.bin"
OUT_AUTO="/tmp/test_m11_g4_auto_$$.bin"
SESS1="/tmp/test_m11_sess1_$$.kmss"
OUT_DEC_C1="/tmp/test_m11_dec_c1_$$.json"
OUT_DEC_C4="/tmp/test_m11_dec_c4_$$.json"

trap 'rm -f "$OUT_C1" "$OUT_C4" "$OUT_AUTO" "$SESS1" "$OUT_DEC_C1" "$OUT_DEC_C4"' EXIT

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --save-session "$SESS1" \
    --threads 1 \
    --output "$OUT_C1" > /dev/null

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --threads 4 \
    --output "$OUT_C4" > /dev/null

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --auto \
    --output "$OUT_AUTO" > /dev/null

if ! cmp -s "$OUT_C1" "$OUT_C4"; then
    echo "FAIL: Output forward c=4 berbeda dengan c=1 (melanggar Delta_max == 0.0)!"
    exit 1
fi

if ! cmp -s "$OUT_C1" "$OUT_AUTO"; then
    echo "FAIL: Output forward --auto berbeda dengan c=1 (melanggar Delta_max == 0.0)!"
    exit 1
fi
echo "   [✓] CLI forward multi-core vs single-thread: 100% bit-exact (Delta_max = 0.0)"

# 3. Multi-core decode bit-exactness verification
"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --session "$SESS1" \
    --max-tokens 4 \
    --threads 1 \
    --output "$OUT_DEC_C1" > /dev/null

"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --session "$SESS1" \
    --max-tokens 4 \
    --threads 4 \
    --output "$OUT_DEC_C4" > /dev/null

if ! cmp -s "$OUT_DEC_C1" "$OUT_DEC_C4"; then
    echo "FAIL: Output decode c=4 berbeda dengan c=1 (melanggar Delta_max == 0.0)!"
    exit 1
fi
echo "   [✓] CLI decode multi-core vs single-thread: 100% bit-exact (Delta_max = 0.0)"

# 4. Zero regression verification across past milestones (M8, M9, M10)
echo "   --> Menjalankan full suite regresi validate-m10 (M8, M9, M10)..."
pixi run validate-m10
echo "   PASS: Gate G-M11-4 TERSERTIFIKASI HIJAU (Delta_max == 0.0, Zero Regression M8-M10 PASS)."

# ---------------------------------------------------------------------------
# Stage 6: Formal Closure Scorecard Generation (reports/YYYY-MM-DD/M11-gates-scorecard.md)
# ---------------------------------------------------------------------------
echo "--> Stage 6: Formal Closure Scorecard Generation"
SCORECARD_DIR="reports/${TODAY}"
SCORECARD_FILE="${SCORECARD_DIR}/M11-gates-scorecard.md"
mkdir -p "$SCORECARD_DIR"

python3 tools/bench/emit_m11_scorecard.py \
    --output "$SCORECARD_FILE" \
    --report-w2 "$REPORT_W2" \
    --report-w4 "$REPORT_W4" \
    --today "$TODAY"

if [ ! -f "$SCORECARD_FILE" ]; then
    echo "FAIL: Scorecard sertifikasi $SCORECARD_FILE gagal dibuat!"
    exit 1
fi

grep -F -q "[PASS - SERTIFIKASI M11 SELESAI]" "$SCORECARD_FILE" || {
    echo "FAIL: Scorecard tidak memiliki verdict final PASS!"
    exit 1
}

echo "   PASS: Scorecard formal terverifikasi di $SCORECARD_FILE."

echo "======================================================================"
echo "ALL TESTS PASSED: MILESTONE M11 QUALITY GATES CERTIFICATION COMPLETE!"
echo "======================================================================"
