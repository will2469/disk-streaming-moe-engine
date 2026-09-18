#!/bin/bash
# Integration test suite for dismoen-tools compare & Gate G-M3-1 verdict (M3-W4)
# Covers:
# 1. Happy path: dismoen layer --part moe (0, 12, 23) vs oracle moe_ref_*.bin (exit 0, MATCH, PASS, Gate G-M3-1)
# 2. Routing verification: matching routing produces MATCH and PASS
# 3. Gate G-M3-1 FAIL category classification (exit 1, MISMATCH, FAIL):
#    - router-selection (routing SET mismatch)
#    - dtype-layout (delta_max > 1.0 or cos_theta < 0.90)
#    - bias-placement (0.25 < delta_max <= 1.0)
#    - rope-style (0.05 <= delta_max <= 0.25)
#    - numeric-order (delta_max > 1e-3, delta_max < 0.05)
# 4. Dimension handling: auto-detect 2048 and explicit --dim 2048
# 5. Layout mismatch detection (exit 2, LAYOUT_MISMATCH)
# 6. File not found detection (exit 2, FILE_NOT_FOUND)

set -u

DISMOEN="${DISMOEN:-./dismoen}"
DISMOEN_TOOLS="${DISMOEN_TOOLS:-target/debug/dismoen-tools}"
if [ ! -f "$DISMOEN_TOOLS" ]; then
    DISMOEN_TOOLS="target/release/dismoen-tools"
fi

MODEL_DIR="${MODEL_DIR:-/home/will/models/qwen1.5-moe-a2.7b-chat}"
TEST_DIR="/tmp/test_m3_w4_compare_$$"
WORKDIR="$TEST_DIR/workdir"
ACT_BIN="fixtures/m3/activation.bin"

fail=0

# Ensure dismoen-tools binary exists
if [ ! -f "$DISMOEN_TOOLS" ]; then
    echo "Building dismoen-tools release binary..."
    cargo build --release --manifest-path tools/dismoen-tools/Cargo.toml --bin dismoen-tools > /dev/null 2>&1
    DISMOEN_TOOLS="target/release/dismoen-tools"
fi

# shellcheck disable=SC2317
cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
[ "${1:-}" = "--cleanup" ] && cleanup

mkdir -p "$WORKDIR"

# Check prerequisites
if [ ! -f "$ACT_BIN" ]; then
    echo "FAIL: activation fixture not found: $ACT_BIN"
    exit 1
fi

echo "=== 1. Happy path: Mojo layer --part moe vs Oracle reference (Gate G-M3-1) ==="
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    for lyr in 0 12 23; do
        REF_BIN="fixtures/m3/moe_ref_${lyr}.bin"
        ROUT_JSON="fixtures/m3/routing_info_${lyr}.json"
        if [ ! -f "$REF_BIN" ] || [ ! -f "$ROUT_JSON" ]; then
            echo "FAIL: reference fixture not found for layer $lyr: $REF_BIN or $ROUT_JSON"
            fail=1
            continue
        fi

        OUT_BIN="$WORKDIR/moe_mojo_${lyr}.bin"
        REPORT_DISMOEN="$TEST_DIR/dismoen_stdout_${lyr}.json"
        "$DISMOEN" layer --layer "$lyr" --part moe "$ACT_BIN" --model-dir "$MODEL_DIR" --workdir "$WORKDIR" --output "moe_mojo_${lyr}.bin" > "$REPORT_DISMOEN" 2>&1
        status=$?
        if [ $status -ne 0 ]; then
            echo "FAIL: dismoen layer --part moe $lyr failed with exit code $status"
            cat "$REPORT_DISMOEN"
            fail=1
            continue
        fi

        REPORT_JSON="$TEST_DIR/report_happy_${lyr}.json"
        "$DISMOEN_TOOLS" compare "$REF_BIN" "$OUT_BIN" --gate G-M3-1 --oracle-routing "$ROUT_JSON" --cand-routing "$ROUT_JSON" > "$REPORT_JSON" 2> "$TEST_DIR/err_happy_${lyr}.txt"
        status=$?
        if [ $status -ne 0 ]; then
            echo "FAIL: compare returned non-zero ($status) for layer $lyr against oracle reference"
            cat "$TEST_DIR/err_happy_${lyr}.txt"
            fail=1
        else
            python3 -c "
import json
with open('$REPORT_JSON') as f:
    r = json.load(f)
assert r['status'] == 'MATCH', f'layer $lyr status={r[\"status\"]}'
assert r['verdict'] == 'PASS', f'layer $lyr verdict={r[\"verdict\"]}'
m = r['metrics']
assert m['delta_max'] <= 1e-3, f'layer $lyr delta_max={m[\"delta_max\"]} > 1e-3'
assert m['epsilon_rel'] <= 1e-4, f'layer $lyr epsilon_rel={m[\"epsilon_rel\"]} > 1e-4'
assert m['agreement'] == 100.0, f'layer $lyr agreement={m[\"agreement\"]}'
" || { echo "FAIL: G-M3-1 gate assertions failed for layer $lyr"; fail=1; }
            echo "PASS: Layer $lyr passed Gate G-M3-1"
        fi
    done
else
    echo "SKIP: Happy path live model check skipped (model dir $MODEL_DIR not available)"
fi

echo "=== 2. Gate G-M3-1 FAIL category classification ==="
BASE_REF="fixtures/m3/moe_ref_0.bin"
ROUT_ORACLE="fixtures/m3/routing_info_0.json"

# Helper function to test failure category
test_fail_cat() {
    local label="$1"
    local delta="$2"
    local expected_cat="$3"
    local mut_bin="$WORKDIR/moe_mut_${label}.bin"
    local rpt_json="$TEST_DIR/report_${label}.json"

    python3 -c "
import sys, struct
with open(sys.argv[1], 'rb') as f:
    data = bytearray(f.read())
v = struct.unpack('<f', data[0:4])[0]
v += float(sys.argv[3])
struct.pack_into('<f', data, 0, v)
with open(sys.argv[2], 'wb') as f:
    f.write(data)
" "$BASE_REF" "$mut_bin" "$delta"

    "$DISMOEN_TOOLS" compare "$BASE_REF" "$mut_bin" --gate G-M3-1 > "$rpt_json" 2>/dev/null
    local rc=$?
    if [ $rc -ne 1 ]; then
        echo "FAIL: expected exit code 1 for $label, got $rc"
        fail=1
        return
    fi

    python3 -c "
import json
with open('$rpt_json') as f:
    r = json.load(f)
assert r['status'] == 'MISMATCH', f'expected MISMATCH, got {r[\"status\"]}'
assert r['verdict'] == 'FAIL', f'expected FAIL, got {r[\"verdict\"]}'
assert r.get('fail_category') == '$expected_cat', f'expected category $expected_cat, got {r.get(\"fail_category\")}'
" || { echo "FAIL: category assertion failed for $label"; fail=1; return; }
    echo "PASS: $label correctly categorized as $expected_cat"
}

# 2a. router-selection (routing SET mismatch)
ROUT_MISMATCH="$TEST_DIR/routing_mismatch.json"
python3 -c "
import json
with open('$ROUT_ORACLE') as f:
    r = json.load(f)
r['selected_experts'][0] = [99, 98, 97, 96]
with open('$ROUT_MISMATCH', 'w') as f:
    json.dump(r, f)
"
REPORT_ROUT="$TEST_DIR/report_rout_mismatch.json"
"$DISMOEN_TOOLS" compare "$BASE_REF" "$BASE_REF" --gate G-M3-1 --oracle-routing "$ROUT_ORACLE" --cand-routing "$ROUT_MISMATCH" > "$REPORT_ROUT" 2>/dev/null
rc_rout=$?
if [ $rc_rout -ne 1 ]; then
    echo "FAIL: expected exit code 1 on routing mismatch, got $rc_rout"
    fail=1
else
    python3 -c "
import json
with open('$REPORT_ROUT') as f:
    r = json.load(f)
assert r['status'] == 'MISMATCH', f'status={r[\"status\"]}'
assert r['verdict'] == 'FAIL', f'verdict={r[\"verdict\"]}'
assert r['fail_category'] == 'router-selection', f'fail_category={r[\"fail_category\"]}'
" || { echo "FAIL: router-selection categorization failed"; fail=1; }
    echo "PASS: routing SET mismatch correctly categorized as router-selection"
fi

# 2b. dtype-layout (delta_max > 1.0)
test_fail_cat "dtype_layout" "5.0" "dtype-layout"

# 2c. bias-placement (0.25 < delta_max <= 1.0)
test_fail_cat "bias_placement" "0.60" "bias-placement"

# 2d. rope-style (0.05 <= delta_max <= 0.25)
test_fail_cat "rope_style" "0.15" "rope-style"

# 2e. numeric-order (1e-3 < delta_max < 0.05)
test_fail_cat "numeric_order" "0.005" "numeric-order"

echo "=== 3. Dimension handling (--dim 2048 & autodetection) ==="
REPORT_AUTO="$TEST_DIR/report_auto.json"
REPORT_EXP="$TEST_DIR/report_explicit.json"

"$DISMOEN_TOOLS" compare "$BASE_REF" "$BASE_REF" --gate G-M3-1 > "$REPORT_AUTO"
rc_auto=$?
"$DISMOEN_TOOLS" compare "$BASE_REF" "$BASE_REF" --dim 2048 --gate G-M3-1 > "$REPORT_EXP"
rc_exp=$?

if [ $rc_auto -ne 0 ] || [ $rc_exp -ne 0 ]; then
    echo "FAIL: self-compare failed with exit codes auto=$rc_auto exp=$rc_exp"
    fail=1
else
    echo "PASS: Dimension auto-detection and explicit --dim 2048 succeed"
fi

echo "=== 4. Layout mismatch detection (exit 2, LAYOUT_MISMATCH) ==="
TRUNC_BIN="$WORKDIR/truncated.bin"
head -c 65536 "$BASE_REF" > "$TRUNC_BIN"
REPORT_TRUNC="$TEST_DIR/err_trunc.json"
"$DISMOEN_TOOLS" compare "$BASE_REF" "$TRUNC_BIN" --gate G-M3-1 > /dev/null 2> "$REPORT_TRUNC"
rc=$?
if [ $rc -ne 2 ]; then
    echo "FAIL: expected exit code 2 on size mismatch, got $rc"
    fail=1
else
    python3 -c "
import json
with open('$REPORT_TRUNC') as f:
    r = json.load(f)
assert r['error_type'] == 'LAYOUT_MISMATCH', f'error_type={r[\"error_type\"]}'
assert r['stage'] == 'compare', f'stage={r[\"stage\"]}'
" || { echo "FAIL: LAYOUT_MISMATCH assertion failed"; fail=1; }
    echo "PASS: Layout mismatch correctly returns code 2 and LAYOUT_MISMATCH"
fi

echo "=== 5. File not found detection (exit 2, FILE_NOT_FOUND) ==="
REPORT_FNF="$TEST_DIR/err_fnf.json"
"$DISMOEN_TOOLS" compare "$BASE_REF" "$WORKDIR/non_existent_file.bin" --gate G-M3-1 > /dev/null 2> "$REPORT_FNF"
rc=$?
if [ $rc -ne 2 ]; then
    echo "FAIL: expected exit code 2 on missing candidate file, got $rc"
    fail=1
else
    python3 -c "
import json
with open('$REPORT_FNF') as f:
    r = json.load(f)
assert r['error_type'] == 'FILE_NOT_FOUND', f'error_type={r[\"error_type\"]}'
assert r['stage'] == 'compare', f'stage={r[\"stage\"]}'
" || { echo "FAIL: FILE_NOT_FOUND assertion failed"; fail=1; }
    echo "PASS: Missing file correctly returns code 2 and FILE_NOT_FOUND"
fi

if [ $fail -eq 0 ]; then
    echo "=== ALL M3-W4 COMPARE INTEGRATION TESTS PASSED ==="
    exit 0
else
    echo "=== SOME M3-W4 COMPARE INTEGRATION TESTS FAILED ==="
    exit 1
fi
