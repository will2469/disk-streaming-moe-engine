#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M8 Wave 1 (M8-W1).
# Menguji:
#   1. Integritas fixture sintetis (<10 MB, seq 16, vocab 512, seed 42)
#   2. Oracle naive F14 loop reference (GDNS v1 framed format, 8352 bytes)
#   3. SEC-6 Golden Hash & 5x Determinisme Reproducibility
#   4. Asymmetric config probe (dk=32, dv=48, 12448 bytes)
#   5. Kontrak Compare: dismoen-tools compare, dismoen compare, tools/compare.py
#   6. Gate G-M8-1 pre-validation di fixture sintetis
#   7. Deteksi mismatch (Exit 1, MISMATCH, FAIL)
#   8. Deteksi checksum korup (Exit 2, CORRUPT_STATE_CHECKSUM)
#   9. Deteksi layout mismatch (Exit 2, LAYOUT_MISMATCH)
#  10. Deteksi file not found (Exit 2, FILE_NOT_FOUND)
#  11. Oracle error handling (Exit 1 input invalid, Exit 2 config invalid)

set -euo pipefail

DISMOEN="${DISMOEN:-./dismoen}"
DISMOEN_TOOLS="${DISMOEN_TOOLS:-target/debug/dismoen-tools}"
if [ ! -f "$DISMOEN_TOOLS" ]; then
    DISMOEN_TOOLS="target/release/dismoen-tools"
fi
PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

TEST_DIR="/tmp/test_m8_w1_$$"
WORKDIR="$TEST_DIR/workdir"
mkdir -p "$WORKDIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "=== M8-W1 Master Integration Test Suite: Oracle Naive & Compare Contract ==="

# ---------------------------------------------------------------------------
# 1. Verifikasi Fixture Sintetis (<10 MB) & SEC-6 Golden Hash
# ---------------------------------------------------------------------------
echo "--> Test 1: Verifikasi Integritas Fixture Sintetis & Golden Hash"
TOKENS_FILE="fixtures/m8_tokens.json"
WEIGHTS_FILE="fixtures/m8_gdn_weights.safetensors"
STATE_FILE="fixtures/m8_state_naive.bin"
HASH_FILE="fixtures/m8_state_naive.bin.sha256"
if [ ! -f "$WEIGHTS_FILE" ] || [ ! -f "$TOKENS_FILE" ]; then
    "$PYTHON" tools/oracle/generate_gdn_fixture.py \
        --layers 2 --dk 32 --dv 32 --vocab 512 --seq-len 16 --seed 42 \
        --output "$WEIGHTS_FILE" --tokens-output "$TOKENS_FILE"
fi

test -f "$TOKENS_FILE"
test -f "$WEIGHTS_FILE"
test -f "$STATE_FILE"
test -f "$HASH_FILE"

# Verifikasi ukuran weights < 10 MB (DoD M8)
weights_size=$(stat -c %s "$WEIGHTS_FILE")
if [ "$weights_size" -ge 10485760 ]; then
    echo "FAIL: ukuran weights fixture >= 10 MB ($weights_size bytes)"
    exit 1
fi

# Verifikasi ukuran file state GDNS v1: 128B header + (2 * 32 * 32 * 4) + 32B hash = 8352 B
state_size=$(stat -c %s "$STATE_FILE")
if [ "$state_size" -ne 8352 ]; then
    echo "FAIL: ukuran file state $state_size != 8352 bytes"
    exit 1
fi

# Verifikasi SEC-6 golden hash
sha256sum -c "$HASH_FILE"
echo "PASS: Test 1 (Fixture sintetis & golden hash valid)"

# ---------------------------------------------------------------------------
# 2. Determinisme Test (5x Run Identik)
# ---------------------------------------------------------------------------
echo "--> Test 2: Uji Reproducibility & Determinisme Oracle (5x Ulangan)"
EXPECTED_SHA=$(cut -d ' ' -f 1 "$HASH_FILE")

for run in 1 2 3 4 5; do
    OUT_RUN="$WORKDIR/state_det_${run}.bin"
    "$PYTHON" tools/oracle/oracle_gdn.py \
        --tokens "$TOKENS_FILE" \
        --layers 2 --dk 32 --dv 32 \
        --weights "$WEIGHTS_FILE" \
        --output "$OUT_RUN" \
        --seed 42 > /dev/null

    RUN_SHA=$(sha256sum "$OUT_RUN" | cut -d ' ' -f 1)
    if [ "$RUN_SHA" != "$EXPECTED_SHA" ]; then
        echo "FAIL: determinisme run $run mismatch! ($RUN_SHA != $EXPECTED_SHA)"
        exit 1
    fi
done
echo "PASS: Test 2 (5x determinisme bit-for-bit identik)"

# ---------------------------------------------------------------------------
# 3. Asymmetric Config Layout Probe (dk=32, dv=48)
# ---------------------------------------------------------------------------
echo "--> Test 3: Asymmetric Config Layout Probe (dk=32, dv=48)"
ASYM_TOKENS="fixtures/m8_asym_tokens.json"
ASYM_WEIGHTS="fixtures/m8_asym_weights.safetensors"
ASYM_STATE="fixtures/m8_asym_state_naive.bin"
ASYM_HASH="fixtures/m8_asym_state_naive.bin.sha256"
if [ ! -f "$ASYM_WEIGHTS" ] || [ ! -f "$ASYM_TOKENS" ]; then
    "$PYTHON" tools/oracle/generate_gdn_fixture.py \
        --layers 2 --dk 32 --dv 48 --vocab 512 --seq-len 16 --seed 42 \
        --output "$ASYM_WEIGHTS" --tokens-output "$ASYM_TOKENS"
fi

test -f "$ASYM_TOKENS"
test -f "$ASYM_WEIGHTS"
test -f "$ASYM_STATE"
test -f "$ASYM_HASH"

# Ukuran asimetris: 128 + (2 * 48 * 32 * 4) + 32 = 12448 bytes
asym_size=$(stat -c %s "$ASYM_STATE")
if [ "$asym_size" -ne 12448 ]; then
    echo "FAIL: ukuran file state asimetris $asym_size != 12448 bytes"
    exit 1
fi
sha256sum -c "$ASYM_HASH"

# Compare asimetris vs itself
REPORT_ASYM="$WORKDIR/report_asym.json"
"$DISMOEN_TOOLS" compare \
    --reference "$ASYM_STATE" \
    --candidate "$ASYM_STATE" \
    --tolerance 1e-3 \
    --output "$REPORT_ASYM" > /dev/null

python3 -c "
import json
with open('$REPORT_ASYM') as f:
    r = json.load(f)
assert r['status'] == 'MATCH'
assert r['verdict'] == 'PASS'
assert r['metrics']['delta_max'] == 0.0
assert r['metrics']['epsilon_rel'] == 0.0
"
echo "PASS: Test 3 (Asymmetric layout probe MATCH)"

# ---------------------------------------------------------------------------
# 4. Compare Contract Happy Path (Gate G-M8-1: MATCH, PASS, Exit 0)
# ---------------------------------------------------------------------------
echo "--> Test 4: Compare Contract Happy Path (Gate G-M8-1)"

# 4a: dismoen-tools compare binary
REPORT_HAPPY_A="$WORKDIR/report_happy_a.json"
"$DISMOEN_TOOLS" compare \
    --reference "$STATE_FILE" \
    --candidate "$STATE_FILE" \
    --tolerance 1e-3 \
    --output "$REPORT_HAPPY_A" > /dev/null

python3 -c "
import json
with open('$REPORT_HAPPY_A') as f:
    r = json.load(f)
assert r['status'] == 'MATCH', f'status={r[\"status\"]}'
assert r['verdict'] == 'PASS', f'verdict={r[\"verdict\"]}'
assert r['threshold'] == 'delta_max <= 1e-3 && epsilon_rel <= 1e-4'
m = r['metrics']
assert m['delta_max'] <= 1e-3
assert m['epsilon_rel'] <= 1e-4
"

# 4b: dismoen CLI compare subcommand
REPORT_HAPPY_B="$WORKDIR/report_happy_b.json"
"$DISMOEN" compare \
    --reference "$STATE_FILE" \
    --candidate "$STATE_FILE" \
    --tolerance 1e-3 \
    --output "$REPORT_HAPPY_B" > /dev/null

python3 -c "
import json
with open('$REPORT_HAPPY_B') as f:
    r = json.load(f)
assert r['status'] == 'MATCH'
assert r['verdict'] == 'PASS'
"

# 4c: Python tools/compare.py wrapper
REPORT_HAPPY_C="$WORKDIR/report_happy_c.json"
python3 tools/compare.py \
    --reference "$STATE_FILE" \
    --candidate "$STATE_FILE" \
    --tolerance 1e-3 \
    --output "$REPORT_HAPPY_C" > /dev/null

python3 -c "
import json
with open('$REPORT_HAPPY_C') as f:
    r = json.load(f)
assert r['status'] == 'MATCH'
assert r['verdict'] == 'PASS'
"
echo "PASS: Test 4 (Compare contract happy path lulus di seluruh runner)"

# ---------------------------------------------------------------------------
# 5. Mismatch Detection (Exit 1, MISMATCH, FAIL)
# ---------------------------------------------------------------------------
echo "--> Test 5: Deteksi Numeric Mismatch (Exit 1, FAIL)"
MUTATED_STATE="$WORKDIR/mutated_state.bin"

python3 -c "
import hashlib
import struct
with open('$STATE_FILE', 'rb') as f:
    data = bytearray(f.read())
# Mutasi 1 float di dalam payload dengan menambah 1.0
val = struct.unpack_from('<f', data, 128 + 40)[0]
struct.pack_into('<f', data, 128 + 40, val + 1.0)
# Recompute checksum valid atas data termutasi agar lolos validasi header/checksum
payload_end = len(data) - 32
hasher = hashlib.sha256()
hasher.update(data[:payload_end])
data[payload_end:] = hasher.digest()
with open('$MUTATED_STATE', 'wb') as f:
    f.write(data)
"

REPORT_FAIL="$WORKDIR/report_fail.json"
set +e
"$DISMOEN_TOOLS" compare \
    --reference "$STATE_FILE" \
    --candidate "$MUTATED_STATE" \
    --tolerance 1e-3 \
    --output "$REPORT_FAIL" > /dev/null
status=$?
set -e

if [ $status -ne 1 ]; then
    echo "FAIL: ekspektasi exit 1 pada mismatch, didapat $status"
    exit 1
fi

python3 -c "
import json
with open('$REPORT_FAIL') as f:
    r = json.load(f)
assert r['status'] == 'MISMATCH'
assert r['verdict'] == 'FAIL'
assert r['metrics']['delta_max'] > 1e-3
assert r.get('fail_category') is not None
"
echo "PASS: Test 5 (Deteksi numeric mismatch menghasilkan Exit 1)"

# ---------------------------------------------------------------------------
# 6. Corrupted Checksum Detection (Exit 2, CORRUPT_STATE_CHECKSUM)
# ---------------------------------------------------------------------------
echo "--> Test 6: Deteksi Checksum Korup (Exit 2, CORRUPT_STATE_CHECKSUM)"
CORRUPT_CHECKSUM_STATE="$WORKDIR/corrupt_checksum.bin"

python3 -c "
with open('$STATE_FILE', 'rb') as f:
    data = bytearray(f.read())
# Rusak 1 byte di payload tanpa update checksum
data[128 + 10] ^= 0xFF
with open('$CORRUPT_CHECKSUM_STATE', 'wb') as f:
    f.write(data)
"

ERR_CORRUPT="$WORKDIR/err_corrupt.txt"
set +e
"$DISMOEN_TOOLS" compare \
    --reference "$STATE_FILE" \
    --candidate "$CORRUPT_CHECKSUM_STATE" > /dev/null 2> "$ERR_CORRUPT"
status=$?
set -e

if [ $status -ne 2 ]; then
    echo "FAIL: ekspektasi exit 2 pada checksum korup, didapat $status"
    exit 1
fi
grep -q '"error_type":"CORRUPT_STATE_CHECKSUM"' "$ERR_CORRUPT"
echo "PASS: Test 6 (Checksum korup ditolak dengan Exit 2)"

# ---------------------------------------------------------------------------
# 7. Truncated State File (Exit 2, LAYOUT_MISMATCH)
# ---------------------------------------------------------------------------
echo "--> Test 7: Deteksi File Truncated (Exit 2, LAYOUT_MISMATCH)"
TRUNCATED_STATE="$WORKDIR/truncated.bin"
head -c 100 "$STATE_FILE" > "$TRUNCATED_STATE"

ERR_TRUNC="$WORKDIR/err_trunc.txt"
set +e
"$DISMOEN_TOOLS" compare \
    --reference "$STATE_FILE" \
    --candidate "$TRUNCATED_STATE" > /dev/null 2> "$ERR_TRUNC"
status=$?
set -e

if [ $status -ne 2 ]; then
    echo "FAIL: ekspektasi exit 2 pada file terpotong, didapat $status"
    exit 1
fi
grep -q '"error_type":"LAYOUT_MISMATCH"' "$ERR_TRUNC"
echo "PASS: Test 7 (File terpotong ditolak dengan Exit 2)"

# ---------------------------------------------------------------------------
# 8. Missing File Detection (Exit 2, FILE_NOT_FOUND)
# ---------------------------------------------------------------------------
echo "--> Test 8: Deteksi File Not Found (Exit 2, FILE_NOT_FOUND)"
ERR_FNF="$WORKDIR/err_fnf.txt"
set +e
"$DISMOEN_TOOLS" compare \
    --reference "$STATE_FILE" \
    --candidate "$WORKDIR/nonexistent.bin" > /dev/null 2> "$ERR_FNF"
status=$?
set -e

if [ $status -ne 2 ]; then
    echo "FAIL: ekspektasi exit 2 pada file hilang, didapat $status"
    exit 1
fi
grep -q '"error_type":"FILE_NOT_FOUND"' "$ERR_FNF"
echo "PASS: Test 8 (File hilang ditolak dengan Exit 2)"

# ---------------------------------------------------------------------------
# 9. Oracle Input & Config Error Handling
# ---------------------------------------------------------------------------
echo "--> Test 9: Validasi Error Handling Oracle (Exit 1 & Exit 2)"

# Missing tokens file -> exit 1
set +e
"$PYTHON" tools/oracle/oracle_gdn.py \
    --tokens "$WORKDIR/nonexistent_tokens.json" \
    --output "$WORKDIR/tmp.bin" > /dev/null 2>&1
status=$?
set -e
if [ $status -ne 1 ]; then
    echo "FAIL: oracle missing tokens harus exit 1, didapat $status"
    exit 1
fi

# Invalid config (dk <= 0) -> exit 2
set +e
"$PYTHON" tools/oracle/oracle_gdn.py \
    --tokens "$TOKENS_FILE" \
    --dk -1 \
    --output "$WORKDIR/tmp.bin" > /dev/null 2>&1
status=$?
set -e
if [ $status -ne 2 ]; then
    echo "FAIL: oracle invalid config harus exit 2, didapat $status"
    exit 1
fi
echo "PASS: Test 9 (Oracle error handling valid)"

echo "=== SEMUA 9 TEST M8-W1 LULUS 100%! Gate G-M8-1 Siap untuk W2 Chunked Kernel ==="
