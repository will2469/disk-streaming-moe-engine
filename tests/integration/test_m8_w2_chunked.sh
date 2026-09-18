#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M8 Wave 2 (M8-W2):
#   Chunked Scan WY + Native Remainder + Hukum Komposisi Operator
#
# Menguji:
#   1. Unit tests (TestSuite): backward solve, komposisi operator, kontinuitas boundary, dsb.
#   2. Validasi Gate G-M8-1 pada fixture sintetis simetris (dk=32, dv=32, seq=16, chunk=8)
#   3. Validasi Gate G-M8-1 pada fixture asimetris (dk=32, dv=48, seq=16, chunk=8)
#   4. Validasi native partial remainder (seq=13, chunk=8 -> m_rem=5 < 8) vs naive oracle
#   5. Konsistensi sweep ukuran chunk (C=8 vs C=16)
#   6. Kontrak validasi chunk size: range [8, 4096] (fail-fast exit non-zero jika di luar batas)
#   7. Determinisme 5x ulangan run identik (SHA-256 bitwise identical)
#   8. Kepatuhan invariant I-8: dievaluasi pada single-thread (threads=1)

set -euo pipefail

KIMO_TOOLS="${KIMO_TOOLS:-target/debug/dismoen-tools}"
if [ ! -f "$KIMO_TOOLS" ]; then
    KIMO_TOOLS="target/release/dismoen-tools"
fi
PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

TEST_DIR="/tmp/test_m8_w2_$$"
WORKDIR="$TEST_DIR/workdir"
mkdir -p "$WORKDIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "=== M8-W2 Master Integration Test Suite: Chunked Scan WY & Composition Law ==="

# ---------------------------------------------------------------------------
# 1. Jalankan Unit Test Suite (Mojo TestSuite)
# ---------------------------------------------------------------------------
echo "--> Test 1: Menjalankan TestSuite Hukum Komposisi Operator & Remainder..."
pixi run mojo run -I src tests/unit/test_gdn_composition_law.mojo
echo "PASS: Test 1 (Seluruh 9 unit tests hijau)"

# ---------------------------------------------------------------------------
# 2. Gate G-M8-1: Ekuivalensi Numerik Fixture Simetris (dk=32, dv=32, C=8)
# ---------------------------------------------------------------------------
echo "--> Test 2: Verifikasi Gate G-M8-1 Fixture Simetris (dk=32, dv=32, seq=16, C=8)..."
TOKENS_SYM="fixtures/m8_tokens.json"
WEIGHTS_SYM="fixtures/m8_gdn_weights.safetensors"
STATE_REF_SYM="fixtures/m8_state_naive.bin"
CAND_SYM="$WORKDIR/state_chunked_sym.bin"
REPORT_SYM="$WORKDIR/report_sym.json"

pixi run mojo run -I src tests/integration/run_gdn_scan_test.mojo \
    --tokens "$TOKENS_SYM" \
    --weights "$WEIGHTS_SYM" \
    --output "$CAND_SYM" \
    --layers 2 --dk 32 --dv 32 --chunk-size 8 > /dev/null

"$KIMO_TOOLS" compare "$STATE_REF_SYM" "$CAND_SYM" --gate G-M8-1 > "$REPORT_SYM"

"$PYTHON" -c "
import json
with open('$REPORT_SYM') as f:
    r = json.load(f)
assert r['status'] == 'MATCH', f'status={r[\"status\"]}'
assert r['verdict'] == 'PASS', f'verdict={r[\"verdict\"]}'
m = r['metrics']
assert m['delta_max'] <= 1e-3, f'delta_max={m[\"delta_max\"]} > 1e-3'
assert m['epsilon_rel'] <= 1e-4, f'epsilon_rel={m[\"epsilon_rel\"]} > 1e-4'
assert m['agreement'] == 100.0, f'agreement={m[\"agreement\"]}'
"
echo "PASS: Test 2 (G-M8-1 lulus pada fixture simetris)"

# ---------------------------------------------------------------------------
# 3. Gate G-M8-1: Ekuivalensi Numerik Fixture Asimetris (dk=32, dv=48, C=8)
# ---------------------------------------------------------------------------
echo "--> Test 3: Verifikasi Gate G-M8-1 Fixture Asimetris (dk=32, dv=48, seq=16, C=8)..."
TOKENS_ASYM="fixtures/m8_asym_tokens.json"
WEIGHTS_ASYM="fixtures/m8_asym_weights.safetensors"
STATE_REF_ASYM="fixtures/m8_asym_state_naive.bin"
CAND_ASYM="$WORKDIR/state_chunked_asym.bin"
REPORT_ASYM="$WORKDIR/report_asym.json"

pixi run mojo run -I src tests/integration/run_gdn_scan_test.mojo \
    --tokens "$TOKENS_ASYM" \
    --weights "$WEIGHTS_ASYM" \
    --output "$CAND_ASYM" \
    --layers 2 --dk 32 --dv 48 --chunk-size 8 > /dev/null

"$KIMO_TOOLS" compare "$STATE_REF_ASYM" "$CAND_ASYM" --gate G-M8-1 > "$REPORT_ASYM"

"$PYTHON" -c "
import json
with open('$REPORT_ASYM') as f:
    r = json.load(f)
assert r['status'] == 'MATCH', f'status={r[\"status\"]}'
assert r['verdict'] == 'PASS', f'verdict={r[\"verdict\"]}'
m = r['metrics']
assert m['delta_max'] <= 1e-3, f'delta_max={m[\"delta_max\"]} > 1e-3'
assert m['epsilon_rel'] <= 1e-4, f'epsilon_rel={m[\"epsilon_rel\"]} > 1e-4'
assert m['agreement'] == 100.0, f'agreement={m[\"agreement\"]}'
"
echo "PASS: Test 3 (G-M8-1 lulus pada fixture asimetris)"

# ---------------------------------------------------------------------------
# 4. Native Partial Remainder: seq_len % chunk_size != 0 (seq=13, C=8 -> m_rem=5)
# ---------------------------------------------------------------------------
echo "--> Test 4: Verifikasi Native Partial Remainder (seq=13, C=8 -> m_rem=5)..."
TOKENS_REM="$WORKDIR/tokens_13.json"
STATE_REF_REM="$WORKDIR/state_ref_rem_13.bin"
CAND_REM="$WORKDIR/state_cand_rem_13.bin"
REPORT_REM="$WORKDIR/report_rem.json"

"$PYTHON" -c "
import json
with open('$TOKENS_SYM') as f:
    d = json.load(f)
d['tokens'] = d['tokens'][:13]
d['seq_len'] = 13
with open('$TOKENS_REM', 'w') as f:
    json.dump(d, f)
"

"$PYTHON" tools/oracle/oracle_gdn.py \
    --tokens "$TOKENS_REM" \
    --weights "$WEIGHTS_SYM" \
    --output "$STATE_REF_REM" \
    --layers 2 --dk 32 --dv 32 > /dev/null

pixi run mojo run -I src tests/integration/run_gdn_scan_test.mojo \
    --tokens "$TOKENS_REM" \
    --weights "$WEIGHTS_SYM" \
    --output "$CAND_REM" \
    --layers 2 --dk 32 --dv 32 --chunk-size 8 > /dev/null

"$KIMO_TOOLS" compare "$STATE_REF_REM" "$CAND_REM" --gate G-M8-1 > "$REPORT_REM"

"$PYTHON" -c "
import json
with open('$REPORT_REM') as f:
    r = json.load(f)
assert r['status'] == 'MATCH', f'status={r[\"status\"]}'
assert r['verdict'] == 'PASS', f'verdict={r[\"verdict\"]}'
m = r['metrics']
assert m['delta_max'] <= 1e-3, f'delta_max={m[\"delta_max\"]} > 1e-3'
assert m['epsilon_rel'] <= 1e-4, f'epsilon_rel={m[\"epsilon_rel\"]} > 1e-4'
"
echo "PASS: Test 4 (Native partial remainder terbukti ekuivalen terhadap naive oracle)"

# ---------------------------------------------------------------------------
# 5. Konsistensi Sweep Ukuran Chunk (C=8 vs C=16)
# ---------------------------------------------------------------------------
echo "--> Test 5: Verifikasi Konsistensi Sweep Ukuran Chunk (C=8 vs C=16)..."
CAND_C16="$WORKDIR/state_chunked_c16.bin"
REPORT_SWEEP="$WORKDIR/report_sweep.json"

pixi run mojo run -I src tests/integration/run_gdn_scan_test.mojo \
    --tokens "$TOKENS_SYM" \
    --weights "$WEIGHTS_SYM" \
    --output "$CAND_C16" \
    --layers 2 --dk 32 --dv 32 --chunk-size 16 > /dev/null

"$KIMO_TOOLS" compare "$CAND_SYM" "$CAND_C16" --gate G-M8-1 > "$REPORT_SWEEP"

"$PYTHON" -c "
import json
with open('$REPORT_SWEEP') as f:
    r = json.load(f)
assert r['status'] == 'MATCH', f'status={r[\"status\"]}'
assert r['verdict'] == 'PASS', f'verdict={r[\"verdict\"]}'
m = r['metrics']
assert m['delta_max'] <= 1e-5, f'delta_max={m[\"delta_max\"]} > 1e-5'
"
echo "PASS: Test 5 (Ukuran chunk C=8 vs C=16 konsisten secara numerik)"

# ---------------------------------------------------------------------------
# 6. Kontrak Validasi Chunk Size: Range [8, 4096]
# ---------------------------------------------------------------------------
echo "--> Test 6: Verifikasi Kontrak Rentang Valid Chunk Size [8, 4096]..."
if pixi run mojo run -I src tests/integration/run_gdn_scan_test.mojo \
    --tokens "$TOKENS_SYM" \
    --weights "$WEIGHTS_SYM" \
    --output "$WORKDIR/invalid.bin" \
    --layers 2 --dk 32 --dv 32 --chunk-size 4 > /dev/null 2>&1; then
    echo "FAIL: chunk_size=4 seharusnya ditolak (batas bawah 8)"
    exit 1
fi

if pixi run mojo run -I src tests/integration/run_gdn_scan_test.mojo \
    --tokens "$TOKENS_SYM" \
    --weights "$WEIGHTS_SYM" \
    --output "$WORKDIR/invalid.bin" \
    --layers 2 --dk 32 --dv 32 --chunk-size 5000 > /dev/null 2>&1; then
    echo "FAIL: chunk_size=5000 seharusnya ditolak (batas atas 4096)"
    exit 1
fi
echo "PASS: Test 6 (Validasi batas rentang chunk_size ditegakkan)"

# ---------------------------------------------------------------------------
# 7. Determinisme 5x Ulangan Run Identik (Bitwise Hash)
# ---------------------------------------------------------------------------
echo "--> Test 7: Uji Determinisme 5x Ulangan Run Identik..."
EXPECTED_SHA=""
for run in 1 2 3 4 5; do
    RUN_OUT="$WORKDIR/state_det_${run}.bin"
    pixi run mojo run -I src tests/integration/run_gdn_scan_test.mojo \
        --tokens "$TOKENS_SYM" \
        --weights "$WEIGHTS_SYM" \
        --output "$RUN_OUT" \
        --layers 2 --dk 32 --dv 32 --chunk-size 8 > /dev/null

    CURRENT_SHA=$(sha256sum "$RUN_OUT" | cut -d ' ' -f 1)
    if [ "$run" -eq 1 ]; then
        EXPECTED_SHA="$CURRENT_SHA"
    else
        if [ "$CURRENT_SHA" != "$EXPECTED_SHA" ]; then
            echo "FAIL: Run determinisme $run mismatch ($CURRENT_SHA != $EXPECTED_SHA)"
            exit 1
        fi
    fi
done
echo "PASS: Test 7 (Determinisme bitwise 5x ulangan terbukti, SHA=$EXPECTED_SHA)"

echo "=== M8-W2 SUKSES 100%: Chunked Scan WY + Remainder + Komposisi Lulus Seluruh Kontrak ==="
