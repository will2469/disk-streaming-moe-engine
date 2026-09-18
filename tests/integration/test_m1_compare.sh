#!/bin/bash
# Integration test suite for dismoen-tools compare & F10 verdict (M1-W4)
# Covers:
# 1. Happy path: kimo head output vs oracle logits_ref.bin (exit 0, MATCH, PASS)
# 2. Positional shards head output vs oracle logits_ref.bin (exit 0, MATCH, PASS)
# 3. Numeric mismatch detection (exit 1, MISMATCH, FAIL)
# 4. Layout mismatch detection (exit 2, LAYOUT_MISMATCH)
# 5. File not found detection (exit 2, FILE_NOT_FOUND)

set -u

KIMO="${KIMO:-./dismoen}"
KIMO_TOOLS="${KIMO_TOOLS:-target/debug/dismoen-tools}"
TEST_DIR="/tmp/test_m1_compare_$$"
WORKDIR="$TEST_DIR/workdir"
OUT_BIN="$WORKDIR/logits_mojo.bin"
REF_BIN="fixtures/m1/logits_ref.bin"

fail=0

# Ensure dismoen-tools binary exists
if [ ! -f "$KIMO_TOOLS" ]; then
    cargo build --manifest-path tools/dismoen-tools/Cargo.toml --bin dismoen-tools > /dev/null 2>&1
fi

# shellcheck disable=SC2317
cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
[ "${1:-}" = "--cleanup" ] && cleanup

mkdir -p "$WORKDIR"

echo "== 1. Happy path: Mojo head vs PyTorch oracle (G-M1-1) =="
"$KIMO" head fixtures/m1/tokens.json --model-dir fixtures/m1 --output "$OUT_BIN" --workdir "$WORKDIR" > /dev/null 2>&1
status=$?
if [ $status -ne 0 ]; then
    echo "FAIL: kimo head failed with exit code $status"
    fail=1
fi

REPORT_JSON="$TEST_DIR/report_happy.json"
"$KIMO_TOOLS" compare "$REF_BIN" "$OUT_BIN" --gate G-M1-1 > "$REPORT_JSON" 2> "$TEST_DIR/err_happy.txt"
status=$?
if [ $status -ne 0 ]; then
    echo "FAIL: compare returned non-zero ($status) on identical oracle/mojo logits"
    fail=1
else
    python3 -c "
import json
with open('$REPORT_JSON') as f:
    r = json.load(f)
assert r['status'] == 'MATCH', f'status={r[\"status\"]}'
assert r['verdict'] == 'PASS', f'verdict={r[\"verdict\"]}'
m = r['metrics']
assert m['delta_max'] <= 1e-3, f'delta_max={m[\"delta_max\"]}'
assert m['epsilon_rel'] <= 1e-4, f'epsilon_rel={m[\"epsilon_rel\"]}'
assert m['agreement'] == 100.0, f'agreement={m[\"agreement\"]}'
" || { echo "FAIL: G-M1-1 gate assertions failed"; fail=1; }
fi

echo "== 2. Positional shards head output vs oracle =="
OUT_POS="$WORKDIR/logits_pos.bin"
"$KIMO" head fixtures/m1/tokens.json fixtures/m1/fixture-00001-of-00003.safetensors fixtures/m1/fixture-00002-of-00003.safetensors --output "$OUT_POS" --workdir "$WORKDIR" > /dev/null 2>&1
status=$?
if [ $status -ne 0 ]; then
    echo "FAIL: kimo head positional shards failed with exit code $status"
    fail=1
fi

REPORT_POS="$TEST_DIR/report_pos.json"
"$KIMO_TOOLS" compare "$REF_BIN" "$OUT_POS" --gate G-M1-1 > "$REPORT_POS" 2> "$TEST_DIR/err_pos.txt"
status=$?
if [ $status -ne 0 ]; then
    echo "FAIL: compare returned non-zero ($status) on positional shards"
    fail=1
fi

echo "== 3. Mismatch detection (exit 1, MISMATCH, FAIL) =="
MUTATED_BIN="$WORKDIR/logits_mutated.bin"
python3 -c "
with open('$OUT_BIN', 'rb') as f:
    data = bytearray(f.read())
data[0] ^= 0xFF
data[1] ^= 0xFF
data[2] ^= 0xFF
data[3] ^= 0xFF
with open('$MUTATED_BIN', 'wb') as f:
    f.write(data)
"
REPORT_MUT="$TEST_DIR/report_mutated.json"
"$KIMO_TOOLS" compare "$REF_BIN" "$MUTATED_BIN" --gate G-M1-1 > "$REPORT_MUT" 2> "$TEST_DIR/err_mut.txt"
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit code 1 on mismatch, got $status"
    fail=1
else
    python3 -c "
import json
with open('$REPORT_MUT') as f:
    r = json.load(f)
assert r['status'] == 'MISMATCH', f'status={r[\"status\"]}'
assert r['verdict'] == 'FAIL', f'verdict={r[\"verdict\"]}'
assert r.get('fail_category') is not None, 'missing fail_category'
" || { echo "FAIL: mismatch report JSON assertion failed"; fail=1; }
fi

echo "== 4. Layout mismatch detection (exit 2, LAYOUT_MISMATCH) =="
TRUNCATED_BIN="$WORKDIR/logits_truncated.bin"
python3 -c "
with open('$OUT_BIN', 'rb') as f:
    data = f.read(100)
with open('$TRUNCATED_BIN', 'wb') as f:
    f.write(data)
"
ERR_LAYOUT="$TEST_DIR/err_layout.txt"
"$KIMO_TOOLS" compare "$REF_BIN" "$TRUNCATED_BIN" > /dev/null 2> "$ERR_LAYOUT"
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit code 2 on layout mismatch, got $status"
    fail=1
fi
grep -q '"error_type":"LAYOUT_MISMATCH"' "$ERR_LAYOUT" || { echo "FAIL: expected error_type LAYOUT_MISMATCH"; fail=1; }

echo "== 5. File not found detection (exit 2, FILE_NOT_FOUND) =="
ERR_FNF="$TEST_DIR/err_fnf.txt"
"$KIMO_TOOLS" compare "$REF_BIN" "$WORKDIR/nonexistent.bin" > /dev/null 2> "$ERR_FNF"
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit code 2 on missing file, got $status"
    fail=1
fi
grep -q '"error_type":"FILE_NOT_FOUND"' "$ERR_FNF" || { echo "FAIL: expected error_type FILE_NOT_FOUND"; fail=1; }

if [ "$fail" -eq 0 ]; then
    echo "=== All M1 Compare integration tests PASSED! ==="
else
    echo "=== Some M1 Compare integration tests FAILED! ==="
fi

exit $fail
