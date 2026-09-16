#!/bin/bash
# E2E test suite for kimo forward CLI (M4-W1)
# Covers:
# 1. USAGE and CLI options validation (missing required, unknown options)
# 2. Happy path: stdout valid JSON, logits exact size s*151936*4, run_id format, runs/<run-id> cleaned
# 3. Output escape containment (SEC-5): ../ escape, absolute path outside workdir, symlink escape
# 4. Input bounds: MAX_TOKENS_FILE_BYTES (1 MiB), MAX_TOKENS (1024), empty tokens, vocab range (151936)
# 5. Token format strictness: nested array rejected, negative int rejected, float rejected, string rejected, leading zero rejected, trailing garbage rejected, trailing comma rejected
# 6. Model directory & workdir validation (non-existent model-dir, unwritable workdir)
# 7. Atomic rollback on output failure
# 8. Shared workdir isolation (IT-M4-13): 2 runs, distinct run_ids, both outputs intact, no orphan files
# 9. Error channel contract (stdout sterile, stderr JSON)
# 10. Negative path coverage: error codes 1-6 verification & cleanup verification

set -u

KIMO="${KIMO:-./kimo}"
TEST_DIR="/tmp/test_m4_w1_cli_$$"
WORKDIR="$TEST_DIR/workdir"
MODEL_DIR="$TEST_DIR/model"
TOKENS_DIR="$TEST_DIR/tokens"

fail=0

cleanup() {
    chmod -R 777 "$TEST_DIR" 2>/dev/null || true
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
[ "${1:-}" = "--cleanup" ] && cleanup

mkdir -p "$WORKDIR" "$MODEL_DIR" "$TOKENS_DIR"

# Valid 16-token prompt
TOKENS_16="$TOKENS_DIR/tokens_16.json"
python3 -c "
import json
json.dump([i * 10 for i in range(16)], open('$TOKENS_16', 'w'))
"

echo "== 1. USAGE & CLI option validation =="
ERR_OUT="$TEST_DIR/err_usage.txt"
"$KIMO" forward > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on empty forward command, got $status"
    fail=1
fi
grep -q '"code":"M4_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M4_ERR_INPUT"; fail=1; }

"$KIMO" forward --unknown-flag > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on unknown option, got $status"
    fail=1
fi
grep -q '"code":"M4_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M4_ERR_INPUT"; fail=1; }

echo "== 2. Happy path: forward CLI =="
OUT_LOGITS="$WORKDIR/logits_happy.bin"
STDOUT_JSON="$TEST_DIR/happy_stdout.json"
STDERR_TXT="$TEST_DIR/happy_stderr.txt"

"$KIMO" forward \
  --model-dir "$MODEL_DIR" \
  --tokens "$TOKENS_16" \
  --output "$OUT_LOGITS" \
  --workdir "$WORKDIR" \
  --threads 1 \
  > "$STDOUT_JSON" 2> "$STDERR_TXT"
status=$?

if [ $status -ne 0 ]; then
    echo "FAIL: happy path forward returned exit $status"
    cat "$STDERR_TXT"
    fail=1
else
    # Verify stderr is empty
    if [ -s "$STDERR_TXT" ]; then
        echo "FAIL: stderr not empty on happy path"
        cat "$STDERR_TXT"
        fail=1
    fi

    # Verify JSON schema in stdout
    python3 -c "
import json, re
with open('$STDOUT_JSON') as f:
    d = json.load(f)
assert d['status'] == 'success'
assert re.match(r'^M4-\d{8}-\d{3}$', d['run_id']), f'invalid run_id: {d[\"run_id\"]}'
assert d['model'] == 'qwen1.5-moe-a2.7b-chat'
assert d['num_tokens'] == 16
assert d['num_layers'] == 24
assert d['logits_path'] == '$OUT_LOGITS'
m = d['metrics']
assert m['walltime_sec'] >= 0.0
assert m['vmhwm_bytes'] > 0
assert m['logical_bytes_read'] == 0
assert m['physical_read_bytes'] >= 0
assert m['cgroup_peak_bytes'] >= 0
assert m['cgroup_oom_kills'] == 0
p = m['phases']
for k in ['index_load_sec', 'embedding_sec', 'layer_forward_sec', 'final_norm_sec', 'lm_head_sec', 'write_sec']:
    assert k in p, f'missing phase {k}'
" || { echo "FAIL: happy path JSON schema validation failed"; fail=1; }

    # Verify binary size: 16 * 151936 * 4 = 9723904 bytes
    actual_size=$(wc -c < "$OUT_LOGITS")
    if [ "$actual_size" -ne 9723904 ]; then
        echo "FAIL: expected logits size 9723904 bytes, got $actual_size"
        fail=1
    fi

    # Verify workdir/runs has no leftover files
    if [ -d "$WORKDIR/runs" ] && [ -n "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ]; then
        echo "FAIL: leftover files in $WORKDIR/runs after happy path"
        fail=1
    fi
fi

echo "== 3. Output escape containment (SEC-5) =="
# Escape via ../
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_16" --output "$WORKDIR/../escape.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on ../ output escape, got $status"
    fail=1
fi
grep -q '"code":"M4_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M4_ERR_INPUT"; fail=1; }
[ ! -e "$WORKDIR/../escape.bin" ] || { echo "FAIL: escape file was written!"; fail=1; }

# Escape via absolute path outside workdir
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_16" --output "/tmp/escape_$$.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on absolute output escape, got $status"
    fail=1
fi
grep -q '"code":"M4_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M4_ERR_INPUT"; fail=1; }
[ ! -e "/tmp/escape_$$.bin" ] || { echo "FAIL: escape file was written!"; fail=1; }

# Escape via symlink in workdir pointing outside
mkdir -p "$WORKDIR/sub"
ln -s /tmp "$WORKDIR/sub/sym_link"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_16" --output "sub/sym_link/escape.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on symlink output escape, got $status"
    fail=1
fi
grep -q '"code":"M4_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M4_ERR_INPUT"; fail=1; }
[ ! -e "/tmp/escape.bin" ] || { echo "FAIL: symlink escape file was written!"; fail=1; }

# Escape via --layer-timing
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_16" --output "$WORKDIR/out.bin" --workdir "$WORKDIR" --layer-timing "$WORKDIR/../timing_escape.json" > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on layer-timing escape, got $status"
    fail=1
fi
grep -q '"code":"M4_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M4_ERR_INPUT"; fail=1; }

echo "== 4. Input bounds validation =="
# 4.1 MAX_TOKENS_FILE_BYTES (1 MiB = 1048576)
BIG_FILE="$TOKENS_DIR/oversize_file.json"
python3 -c "
# Generate JSON larger than 1 MiB
with open('$BIG_FILE', 'w') as f:
    f.write('[' + '0,' * 600000 + '0]')
"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$BIG_FILE" --output "$WORKDIR/big_file.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on file size > 1 MiB, got $status"
    fail=1
fi
grep -q 'MAX_TOKENS_FILE_BYTES' "$ERR_OUT" || { echo "FAIL: error message missing MAX_TOKENS_FILE_BYTES"; fail=1; }

# 4.2 MAX_TOKENS (1024)
BIG_TOKENS="$TOKENS_DIR/tokens_1025.json"
python3 -c "
import json
json.dump(list(range(1025)), open('$BIG_TOKENS', 'w'))
"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$BIG_TOKENS" --output "$WORKDIR/big_tok.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on tokens count > 1024, got $status"
    fail=1
fi
grep -q 'MAX_TOKENS' "$ERR_OUT" || { echo "FAIL: error message missing MAX_TOKENS"; fail=1; }

# 4.3 Empty tokens array []
EMPTY_TOKENS="$TOKENS_DIR/empty.json"
echo "[]" > "$EMPTY_TOKENS"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$EMPTY_TOKENS" --output "$WORKDIR/empty.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on empty tokens array, got $status"
    fail=1
fi
grep -q '"code":"M4_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M4_ERR_INPUT"; fail=1; }

# 4.4 Token ID out of vocab (151936)
OUT_OF_VOCAB="$TOKENS_DIR/out_of_vocab.json"
python3 -c "
import json
json.dump([10, 20, 151936], open('$OUT_OF_VOCAB', 'w'))
"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$OUT_OF_VOCAB" --output "$WORKDIR/oov.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ]; then
    echo "FAIL: expected exit 1 on out of vocab token, got $status"
    fail=1
fi
grep -q '"token_id":151936' "$ERR_OUT" || { echo "FAIL: token_id missing in error details"; fail=1; }
grep -q '"vocab_size":151936' "$ERR_OUT" || { echo "FAIL: vocab_size missing in error details"; fail=1; }

echo "== 5. Token syntax and format strictness =="
# 5.1 Nested array [[1, 2]]
NESTED_TOK="$TOKENS_DIR/nested.json"
echo "[[1, 2]]" > "$NESTED_TOK"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$NESTED_TOK" --output "$WORKDIR/nested.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: nested array not rejected"; fail=1; }

# 5.2 Negative integer
NEG_TOK="$TOKENS_DIR/neg.json"
echo "[-1, 2]" > "$NEG_TOK"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$NEG_TOK" --output "$WORKDIR/neg.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: negative token not rejected"; fail=1; }

# 5.3 Float
FLOAT_TOK="$TOKENS_DIR/float.json"
echo "[1.5, 2]" > "$FLOAT_TOK"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$FLOAT_TOK" --output "$WORKDIR/float.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: float token not rejected"; fail=1; }

# 5.4 String element
STR_TOK="$TOKENS_DIR/str.json"
echo '["123", 2]' > "$STR_TOK"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$STR_TOK" --output "$WORKDIR/str.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: string token not rejected"; fail=1; }

# 5.5 Leading zero integer
ZERO_TOK="$TOKENS_DIR/zero.json"
echo '[01, 2]' > "$ZERO_TOK"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$ZERO_TOK" --output "$WORKDIR/zero.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: leading zero not rejected"; fail=1; }

# 5.6 Trailing garbage
GARB_TOK="$TOKENS_DIR/garb.json"
echo '[1, 2]GARBAGE' > "$GARB_TOK"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$GARB_TOK" --output "$WORKDIR/garb.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: trailing garbage not rejected"; fail=1; }

# 5.7 Trailing comma
COMMA_TOK="$TOKENS_DIR/comma.json"
echo '[1, 2,]' > "$COMMA_TOK"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$COMMA_TOK" --output "$WORKDIR/comma.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: trailing comma not rejected"; fail=1; }

# 5.8 Unterminated array
UNTERM_TOK="$TOKENS_DIR/unterm.json"
printf '[' > "$UNTERM_TOK"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$UNTERM_TOK" --output "$WORKDIR/unterm.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: unterminated array not rejected"; fail=1; }

echo "== 6. Model directory & workdir validation =="
# Missing model directory
"$KIMO" forward --model-dir "/nonexistent_model_dir_$$" --tokens "$TOKENS_16" --output "$WORKDIR/out.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: nonexistent model dir not rejected"; fail=1; }
grep -q '"code":"M4_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M4_ERR_INPUT"; fail=1; }

# Unwritable workdir
RO_WORKDIR="$TEST_DIR/ro_workdir"
mkdir -p "$RO_WORKDIR"
chmod 555 "$RO_WORKDIR"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_16" --output "out.bin" --workdir "$RO_WORKDIR" > "$ERR_OUT" 2>&1
[ $? -eq 1 ] || { echo "FAIL: unwritable workdir not rejected with exit 1"; fail=1; }
grep -q '"code":"M4_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M4_ERR_INPUT"; fail=1; }
chmod 777 "$RO_WORKDIR"

echo "== 7. Atomic rollback on output failure =="
RO_OUTDIR="$WORKDIR/readonly_outdir"
mkdir -p "$RO_OUTDIR"
chmod 555 "$RO_OUTDIR"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_16" --output "readonly_outdir/out.bin" --workdir "$WORKDIR" > "$ERR_OUT" 2>&1
status=$?
if [ $status -ne 1 ] && [ $status -ne 6 ]; then
    echo "FAIL: expected exit 1 or 6 on unwritable output destination, got $status"
    fail=1
fi
chmod 777 "$RO_OUTDIR"
if [ -f "$RO_OUTDIR/out.bin" ] || compgen -G "$RO_OUTDIR/out.bin.tmp.*" > /dev/null; then
    echo "FAIL: leftover file in failed output destination!"
    fail=1
fi

echo "== 8. Shared workdir isolation (IT-M4-13) =="
OUT_ISO_A="$WORKDIR/iso_a.bin"
OUT_ISO_B="$WORKDIR/iso_b.bin"
JSON_A="$TEST_DIR/iso_a.json"
JSON_B="$TEST_DIR/iso_b.json"

"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_16" --output "$OUT_ISO_A" --workdir "$WORKDIR" --run-id "M4-20260916-001" > "$JSON_A" 2>&1
[ $? -eq 0 ] || { echo "FAIL: run A failed"; fail=1; }

"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_16" --output "$OUT_ISO_B" --workdir "$WORKDIR" --run-id "M4-20260916-002" > "$JSON_B" 2>&1
[ $? -eq 0 ] || { echo "FAIL: run B failed"; fail=1; }

# Both output files must exist
[ -f "$OUT_ISO_A" ] && [ -f "$OUT_ISO_B" ] || { echo "FAIL: output files missing in shared workdir"; fail=1; }

# Verify run_ids are distinct
RUN_A=$(python3 -c "import json; print(json.load(open('$JSON_A'))['run_id'])")
RUN_B=$(python3 -c "import json; print(json.load(open('$JSON_B'))['run_id'])")
if [ "$RUN_A" = "$RUN_B" ]; then
    echo "FAIL: run IDs are identical across runs: $RUN_A"
    fail=1
fi

# Verify no orphan files in runs/
if [ -d "$WORKDIR/runs" ] && [ -n "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ]; then
    echo "FAIL: orphan temp remaining in $WORKDIR/runs after both runs"; fail=1;
fi

echo "== 9. Error channel contract (stdout sterile, stderr JSON) =="
STDOUT_ERR_TEST="$TEST_DIR/sterile_stdout.txt"
STDERR_ERR_TEST="$TEST_DIR/sterile_stderr.txt"
"$KIMO" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_16" --output "$WORKDIR/../escape.bin" --workdir "$WORKDIR" > "$STDOUT_ERR_TEST" 2> "$STDERR_ERR_TEST" || true
if [ -s "$STDOUT_ERR_TEST" ]; then
    echo "FAIL: stdout is not sterile on error:"
    cat "$STDOUT_ERR_TEST"
    fail=1
fi
python3 -c "
import json
d = json.load(open('$STDERR_ERR_TEST'))
assert d['status'] == 'error'
assert 'code' in d['error']
assert 'stage' in d['error']
assert 'message' in d['error']
" || { echo "FAIL: stderr does not contain valid error JSON"; fail=1; }

echo "== 10. Negative path coverage: error codes 1-6 & cleanup =="
declare -A EXPECTED_CODES=(
    ["M4_ERR_INPUT"]="1"
    ["M4_ERR_INDEX"]="2"
    ["M4_ERR_MEMORY"]="3"
    ["M4_ERR_SHARD_IO"]="4"
    ["M4_ERR_LAYER_FORWARD"]="5"
    ["M4_ERR_OUTPUT"]="6"
)

for err_code in "${!EXPECTED_CODES[@]}"; do
    expected_exit="${EXPECTED_CODES[$err_code]}"
    ERR_JSON="$TEST_DIR/mock_${err_code}.json"
    "$KIMO" forward \
      --model-dir "$MODEL_DIR" \
      --tokens "$TOKENS_16" \
      --output "$WORKDIR/out_${err_code}.bin" \
      --workdir "$WORKDIR" \
      --mock-error "$err_code" \
      > /dev/null 2> "$ERR_JSON"
    actual_exit=$?
    if [ "$actual_exit" -ne "$expected_exit" ]; then
        echo "FAIL: mock error $err_code expected exit $expected_exit, got $actual_exit"
        fail=1
    fi
    # Verify code in JSON
    grep -q "\"code\":\"$err_code\"" "$ERR_JSON" || { echo "FAIL: error JSON code mismatch for $err_code"; fail=1; }
    # Verify no orphan files in runs/
    if [ -d "$WORKDIR/runs" ] && [ -n "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ]; then
        echo "FAIL: orphan temp remaining in $WORKDIR/runs after error $err_code"
        fail=1
    fi
    # Verify target file was not committed
    if [ -e "$WORKDIR/out_${err_code}.bin" ]; then
        echo "FAIL: target output file exists after error $err_code"
        fail=1
    fi
done

if [ $fail -eq 0 ]; then
    echo "ALL M4-W1 INTEGRATION TESTS PASSED"
    exit 0
else
    echo "SOME M4-W1 INTEGRATION TESTS FAILED"
    exit 1
fi
