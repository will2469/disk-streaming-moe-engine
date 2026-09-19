#!/bin/bash
# E2E test suite for dismoen decode CLI (M5-W3: Decode CLI + Error + Rollback + Sampling)
#
# Covers:
# 1. Formatting compliance: mojo format check
# 2. CLI options & USAGE validation (unknown option, missing required flags, invalid numbers)
# 3. Context bounds chain enforcement (S + N <= ctx <= s_max):
#    - IT-M5-4: Context size > s_max (8192 > 4096) -> exit 2
#    - IT-M5-11: S + N > ctx overflow before KV alloc -> exit 2
# 4. Error codes 1-6 verification & cleanup:
#    - Exit 1: M5_ERR_INPUT (stage: input)
#    - Exit 2: M5_ERR_CONTEXT_SIZE (stage: kv_alloc)
#    - Exit 3: M5_ERR_KV_ALLOC (stage: kv_alloc)
#    - Exit 4: M5_ERR_PREFILL (stage: prefill)
#    - Exit 5: M5_ERR_DECODE (stage: decode)
#    - Exit 6: M5_ERR_OUTPUT (stage: output)
# 5. Output containment & atomic rollback (no partial files on failure)
# 6. Sampling honesty:
#    - Greedy mode (temp == 0.0): seed is null in JSON even if --seed 42 is passed
#    - Sample mode (temp > 0.0): seed is recorded in JSON
# 7. Reproducibility (IT-M5-8): run-sama -> byte-sama (SHA-256 match)
# 8. Happy path stdout JSON schema & output tokens.json formatting

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-./dismoen}"
TEST_DIR="/tmp/test_m5_w3_cli_$$"
WORKDIR="$TEST_DIR/workdir"
MODEL_DIR="$TEST_DIR/model"
OUTPUT_DIR="$TEST_DIR/output"

cleanup() {
    chmod -R 777 "$TEST_DIR" 2>/dev/null || true
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

mkdir -p "$WORKDIR" "$MODEL_DIR" "$OUTPUT_DIR"

# Fix #3: mock model-dir dibekali tokenizer BPE REAL (fixture) agar
# --prompt menghasilkan ID BPE sebenarnya (tanpa hash fallback).
cp fixtures/m12_tokenizer/tokenizer.json "$MODEL_DIR/"

echo "============================================================"
echo "M5-W3: Decode CLI Integration Verification"
echo "============================================================"

# Step 1: Format check
echo "== 1. Checking Mojo formatting =="
FORMAT_OUTPUT=$(pixi run mojo format \
    src/cli/cmd_decode.mojo \
    src/cli/io_utils.mojo \
    src/cli/m5_errors.mojo \
    src/cli/sys_utils.mojo \
    src/main.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi
echo "PASS: Formatting is clean."

# Step 2: USAGE and basic option validation
echo "== 2. Testing USAGE and CLI option validation =="
ERR_OUT="$TEST_DIR/err_usage.txt"

# Empty decode args -> exit 1, M5_ERR_INPUT
if "$DISMOEN" decode > "$ERR_OUT" 2>&1; then
    echo "FAIL: expected failure on empty decode command"
    exit 1
fi
grep -q '"code":"M5_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M5_ERR_INPUT on empty decode"; exit 1; }

# Unknown flag -> exit 1, M5_ERR_INPUT
if "$DISMOEN" decode --unknown-flag > "$ERR_OUT" 2>&1; then
    echo "FAIL: expected failure on unknown flag"
    exit 1
fi
grep -q '"code":"M5_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M5_ERR_INPUT on unknown flag"; exit 1; }

# Missing model-dir -> exit 1
if "$DISMOEN" decode --prompt "Test" > "$ERR_OUT" 2>&1; then
    echo "FAIL: expected failure on missing --model-dir"
    exit 1
fi
grep -q '"code":"M5_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M5_ERR_INPUT on missing model-dir"; exit 1; }

# Invalid max-tokens = 0 -> exit 1
if "$DISMOEN" decode --mock-decode --model-dir "$MODEL_DIR" --prompt "Test" --max-tokens 0 > "$ERR_OUT" 2>&1; then
    echo "FAIL: expected failure on max-tokens=0"
    exit 1
fi
grep -q '"code":"M5_ERR_INPUT"' "$ERR_OUT" || { echo "FAIL: error code not M5_ERR_INPUT on max-tokens=0"; exit 1; }

# Step 3: Context size bounds chain (S + N <= ctx <= s_max)
echo "== 3. Testing context bounds chain enforcement =="

# IT-M5-4: Context size > s_max (8192 > 4096) -> exit 2
set +e
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test" \
  --max-tokens 64 \
  --context-size 8192 \
  --workdir "$WORKDIR" > /dev/null 2> "$ERR_OUT"
status=$?
set -e
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on ctx > s_max, got $status"
    exit 1
fi
grep -q '"code":"M5_ERR_CONTEXT_SIZE"' "$ERR_OUT" || { echo "FAIL: error code not M5_ERR_CONTEXT_SIZE on ctx > s_max"; exit 1; }
echo "PASS: IT-M5-4 context size > s_max rejected with exit 2."

# IT-M5-11: S + N > ctx overflow before KV alloc -> exit 2
BIG_PROMPT=$(python3 -c "print('hello ' * 5000)")
set +e
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "$BIG_PROMPT" \
  --max-tokens 64 \
  --context-size 2048 \
  --workdir "$WORKDIR" > /dev/null 2> "$ERR_OUT"
status=$?
set -e
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on S + N > ctx, got $status"
    exit 1
fi
grep -q '"code":"M5_ERR_CONTEXT_SIZE"' "$ERR_OUT" || { echo "FAIL: error code not M5_ERR_CONTEXT_SIZE on S+N>ctx"; exit 1; }
echo "PASS: IT-M5-11 prompt context overflow rejected with exit 2."

# Step 4: Error codes 1-6 stage behavior & sterile stdout
echo "== 4. Testing error codes 1-6 stage behavior =="
for err_type in M5_ERR_INPUT M5_ERR_CONTEXT_SIZE M5_ERR_KV_ALLOC M5_ERR_PREFILL M5_ERR_DECODE M5_ERR_OUTPUT; do
    case "$err_type" in
        M5_ERR_INPUT) exp_code=1 ;;
        M5_ERR_CONTEXT_SIZE) exp_code=2 ;;
        M5_ERR_KV_ALLOC) exp_code=3 ;;
        M5_ERR_PREFILL) exp_code=4 ;;
        M5_ERR_DECODE) exp_code=5 ;;
        M5_ERR_OUTPUT) exp_code=6 ;;
    esac

    STDOUT_ERR="$TEST_DIR/stdout_${err_type}.txt"
    STDERR_ERR="$TEST_DIR/stderr_${err_type}.txt"

    set +e
    "$DISMOEN" decode \
      --mock-decode \
      --mock-error "$err_type" \
      --model-dir "$MODEL_DIR" \
      --prompt "Hello world" \
      --max-tokens 64 \
      --context-size 2048 \
      --workdir "$WORKDIR" > "$STDOUT_ERR" 2> "$STDERR_ERR"
    ret=$?
    set -e

    if [ $ret -ne $exp_code ]; then
        echo "FAIL: $err_type expected exit code $exp_code, got $ret"
        cat "$STDERR_ERR"
        exit 1
    fi

    # Stdout must remain empty on error
    if [ -s "$STDOUT_ERR" ]; then
        echo "FAIL: stdout not empty on error $err_type"
        exit 1
    fi

    # Stderr must be valid JSON with error schema
    python3 -c "
import json
with open('$STDERR_ERR') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == '$err_type'
assert 'stage' in d['error']
assert 'message' in d['error']
assert 'details' in d['error']
"
    echo "PASS: $err_type -> exit $exp_code verified with valid JSON schema."
done

# Step 5: Sampling honesty: greedy seed null vs sample seed recorded
echo "== 5. Testing sampling honesty =="
STDOUT_GREEDY="$TEST_DIR/stdout_greedy.json"
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "The quick brown fox" \
  --max-tokens 32 \
  --context-size 2048 \
  --workdir "$WORKDIR" \
  --output "$WORKDIR/tokens_greedy.json" \
  --seed 42 > "$STDOUT_GREEDY" 2>/dev/null

python3 -c "
import json
with open('$STDOUT_GREEDY') as f:
    d = json.load(f)
assert d['status'] == 'success'
assert d['sampling']['mode'] == 'greedy'
assert d['sampling']['temperature'] == 0.0
assert d['sampling']['seed'] is None, f'greedy seed must be null, got {d[\"sampling\"][\"seed\"]}'
"
echo "PASS: Greedy mode ignores --seed and records 'seed': null."

STDOUT_SAMPLE="$TEST_DIR/stdout_sample.json"
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "The quick brown fox" \
  --max-tokens 32 \
  --context-size 2048 \
  --workdir "$WORKDIR" \
  --output "$WORKDIR/tokens_sample.json" \
  --temperature 0.8 \
  --seed 12345 > "$STDOUT_SAMPLE" 2>/dev/null

python3 -c "
import json
with open('$STDOUT_SAMPLE') as f:
    d = json.load(f)
assert d['status'] == 'success'
assert d['sampling']['mode'] == 'sample'
assert d['sampling']['temperature'] == 0.8
assert d['sampling']['seed'] == 12345, f'sample seed must be 12345, got {d[\"sampling\"][\"seed\"]}'
"
echo "PASS: Sample mode records effective seed."

# Step 6: Atomic rollback on output failure
echo "== 6. Testing atomic rollback on output failure =="
UNWRITABLE_OUTPUT="$TEST_DIR/non_existent_dir/tokens.json"
set +e
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test rollback" \
  --max-tokens 16 \
  --context-size 2048 \
  --workdir "$WORKDIR" \
  --output "$UNWRITABLE_OUTPUT" > /dev/null 2> "$ERR_OUT"
status=$?
set -e
if [ $status -ne 1 ] && [ $status -ne 6 ]; then
    echo "FAIL: expected failure on unwritable output path, got $status"
    exit 1
fi
if [ -f "$UNWRITABLE_OUTPUT" ]; then
    echo "FAIL: output file should not exist on rollback failure"
    exit 1
fi
echo "PASS: Atomic rollback preserved cleanly without partial output."

# Step 7: Reproducibility A (IT-M5-8)
echo "== 7. Testing reproducibility A (IT-M5-8: run-sama -> byte-sama) =="
TOKENS_RUN1="$WORKDIR/tokens_run1.json"
TOKENS_RUN2="$WORKDIR/tokens_run2.json"

"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Deterministic Test Prompt" \
  --max-tokens 64 \
  --context-size 2048 \
  --output "$TOKENS_RUN1" \
  --workdir "$WORKDIR" \
  --threads 1 \
  --seed 42 > /dev/null 2>&1

"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Deterministic Test Prompt" \
  --max-tokens 64 \
  --context-size 2048 \
  --output "$TOKENS_RUN2" \
  --workdir "$WORKDIR" \
  --threads 1 \
  --seed 42 > /dev/null 2>&1

SHA1=$(sha256sum "$TOKENS_RUN1" | cut -d' ' -f1)
SHA2=$(sha256sum "$TOKENS_RUN2" | cut -d' ' -f1)

if [ "$SHA1" != "$SHA2" ]; then
    echo "FAIL: SHA mismatch in IT-M5-8 reproducibility ($SHA1 vs $SHA2)"
    exit 1
fi
echo "PASS: IT-M5-8 identical SHA256 match ($SHA1)."

# Step 8: Happy path stdout JSON schema validation
echo "== 8. Testing happy path stdout JSON schema and tokens file formatting =="
STDOUT_FINAL="$TEST_DIR/stdout_final.json"
FINAL_TOKENS="$WORKDIR/tokens_final.json"

"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "What is the capital of France?" \
  --max-tokens 64 \
  --context-size 2048 \
  --output "$FINAL_TOKENS" \
  --workdir "$WORKDIR" > "$STDOUT_FINAL" 2>&1

python3 -c "
import json, re

with open('$STDOUT_FINAL') as f:
    d = json.load(f)

assert d['status'] == 'success'
assert re.match(r'^M5-\d{8}-\d{3}$', d['run_id']), f'invalid run_id: {d[\"run_id\"]}'
assert d['model'] in ('qwen3.6-35b-a3b', 'qwen1.5-moe-a2.7b-chat')
assert d['prompt'] == 'What is the capital of France?'
assert d['prompt_tokens'] > 0
assert d['generated_tokens'] == 64
assert d['context_size'] == 2048
assert d['kv_cache_bytes'] > 0

m = d['metrics']
assert m['prefill_time_sec'] >= 0.0
assert m['decode_time_sec'] >= 0.0
assert m['total_time_sec'] >= 0.0
assert m['tokens_per_sec'] >= 0.0
assert m['vmhwm_bytes'] > 0
assert 'bytes_read_prefill' in m
assert 'bytes_read_decode' in m

with open('$FINAL_TOKENS') as f:
    tokens = json.load(f)
assert isinstance(tokens, list)
assert len(tokens) == 64
for t in tokens:
    assert isinstance(t, int)
    assert 0 <= t < 151936
"
echo "PASS: Full schema and token formatting verified."

echo "============================================================"
echo "M5-W3 VERIFICATION COMPLETE: ALL CHECKS PASSED (GREEN)"
echo "============================================================"
