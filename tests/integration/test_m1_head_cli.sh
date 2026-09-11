#!/bin/bash
# E2E test suite for kimo head CLI (M1-W3)
# Covers:
# 1. Happy path (--model-dir)
# 2. Happy path (positional shards)
# 3. Allowlist enforcement (missing required shards/tensors)
# 4. Missing shard file on disk
# 5. Invalid token ID (exceeds vocab)
# 6. Invalid token ID (negative)
# 7. Invalid prompt length (!= 16)
# 8. Invalid prompt count (!= 3)
# 9. Missing tokens file & malformed JSON
# 10. Config error (missing rms_norm_eps)
# 11. Config error (non-positive rms_norm_eps)
# 12. Output path escapes workdir via ../
# 13. Output path escapes workdir via symlink
# 14. Atomic write rollback on unwritable directory

set -u

KIMO="${KIMO:-./kimo}"
TEST_DIR="/tmp/test_m1_head_cli_$$"
FX_DIR="$TEST_DIR/fixture"
WORKDIR="$TEST_DIR/workdir"
OUT_DIR="$WORKDIR/output"

fail=0

# shellcheck disable=SC2317
cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
# Avoid unused warning for analyzers that miss trap
[ "${1:-}" = "--cleanup" ] && cleanup

mkdir -p "$FX_DIR" "$WORKDIR" "$OUT_DIR"

# 1. Generate clean synthetic fixture for M1 testing
python3 -c "
import struct, json, os

out = '$FX_DIR'
os.makedirs(out, exist_ok=True)

def make_shard(tensors, filename):
    hdr = {}
    blobs = []
    off = 0
    for name, shape, val_fn in tensors:
        n = 1
        for d in shape: n *= d
        raw = bytearray(n * 2)
        for i in range(n):
            v = val_fn(i)
            u32 = struct.unpack('<I', struct.pack('<f', v))[0]
            struct.pack_into('<H', raw, i * 2, (u32 >> 16) & 0xFFFF)
        hdr[name] = {'dtype': 'BF16', 'shape': shape, 'data_offsets': [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(hdr, separators=(',', ':')).encode()
    with open(os.path.join(out, filename), 'wb') as f:
        f.write(struct.pack('<Q', len(hb)) + hb + b''.join(blobs))

make_shard([
    ('model.embed_tokens.weight', [512, 64], lambda i: (i % 17 - 8) * 0.05),
    ('lm_head.weight', [512, 64], lambda i: (i % 13 - 6) * 0.02),
], 'fixture-00001-of-00003.safetensors')

make_shard([
    ('model.norm.weight', [64], lambda i: 1.0 + (i % 5) * 0.01),
], 'fixture-00002-of-00003.safetensors')

make_shard([], 'fixture-00003-of-00003.safetensors')

weight_map = {
    'model.embed_tokens.weight': 'fixture-00001-of-00003.safetensors',
    'lm_head.weight': 'fixture-00001-of-00003.safetensors',
    'model.norm.weight': 'fixture-00002-of-00003.safetensors',
}

with open(os.path.join(out, 'model.safetensors.index.json'), 'w') as f:
    json.dump({'metadata': {'total_size': 0}, 'weight_map': weight_map}, f)

with open(os.path.join(out, 'model_config.json'), 'w') as f:
    json.dump({
        'hidden_size': 64,
        'vocab_size': 512,
        'rms_norm_eps': 1e-6
    }, f, indent=2)
"

# Generate standard 3x16 tokens
TOKENS_VALID="$TEST_DIR/tokens_3x16.json"
python3 -c "
import json
tokens = [
    [i for i in range(16)],
    [i + 16 for i in range(16)],
    [i + 32 for i in range(16)]
]
with open('$TOKENS_VALID', 'w') as f:
    json.dump(tokens, f)
"

echo "== 1. Happy path: --model-dir =="
OUT_BIN="$OUT_DIR/logits_model_dir.bin"
STDOUT_JSON="$TEST_DIR/out_m1.json"
"$KIMO" head "$TOKENS_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "$OUT_BIN" > "$STDOUT_JSON" 2>&1
status=$?
if [ $status -ne 0 ]; then
    echo "FAIL: happy path --model-dir returned exit code $status"
    fail=1
else
    # Verify JSON report structure
    python3 -c "
import json, sys
with open('$STDOUT_JSON') as f:
    data = json.load(f)
assert data['status'] == 'success'
assert data['num_prompts'] == 3
assert data['tokens_per_prompt'] == 16
assert data['num_tokens_total'] == 48
assert data['vocab_size'] == 512
assert data['parse_time_ms'] > 0
assert data['compute_time_ms'] > 0
assert data['memory']['resident_target_bytes'] == 262400
assert data['memory']['vmhwm_bytes'] > 0
" || { echo "FAIL: JSON report schema check failed"; fail=1; }

    # Verify binary size: 3 * 16 * 512 * 4 = 98304 bytes
    actual_size=$(wc -c < "$OUT_BIN")
    if [ "$actual_size" -ne 98304 ]; then
        echo "FAIL: expected binary size 98304, got $actual_size"
        fail=1
    fi
fi

echo "== 2. Happy path: positional shards =="
OUT_BIN_POS="$OUT_DIR/logits_pos.bin"
"$KIMO" head "$TOKENS_VALID" "$FX_DIR/fixture-00001-of-00003.safetensors" "$FX_DIR/fixture-00002-of-00003.safetensors" --workdir "$WORKDIR" --output "$OUT_BIN_POS" > /dev/null 2>&1
status=$?
if [ $status -ne 0 ]; then
    echo "FAIL: happy path positional shards returned exit code $status"
    fail=1
else
    if ! cmp -s "$OUT_BIN" "$OUT_BIN_POS"; then
        echo "FAIL: positional output differs from model-dir output"
        fail=1
    fi
fi

echo "== 3. Allowlist: missing required shard in positional mode =="
ERR_TXT="$TEST_DIR/err_allowlist.txt"
"$KIMO" head "$TOKENS_VALID" "$FX_DIR/fixture-00001-of-00003.safetensors" --workdir "$WORKDIR" --output "$OUT_DIR/out_bad.bin" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on allowlist violation, got $status"
    fail=1
fi
grep -q '"error_type":"WEIGHT_LOAD_FAILED"' "$ERR_TXT" || { echo "FAIL: error_type not WEIGHT_LOAD_FAILED"; fail=1; }
grep -q 'model\.norm\.weight' "$ERR_TXT" || { echo "FAIL: missing_tensors not reported"; fail=1; }
grep -q 'fixture-00002-of-00003\.safetensors' "$ERR_TXT" || { echo "FAIL: expected_shards not reported"; fail=1; }

echo "== 4. Missing shard file on disk =="
ERR_TXT="$TEST_DIR/err_missing_disk.txt"
"$KIMO" head "$TOKENS_VALID" "$FX_DIR/fixture-00001-of-00003.safetensors" "$FX_DIR/nonexistent.safetensors" --workdir "$WORKDIR" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on missing disk shard, got $status"
    fail=1
fi
grep -q '"error_type":"FILE_NOT_FOUND"' "$ERR_TXT" || { echo "FAIL: error_type not FILE_NOT_FOUND"; fail=1; }

echo "== 5. Token ID exceeds vocab =="
TOKENS_BAD_ID="$TEST_DIR/tokens_bad_id.json"
python3 -c "
import json
tokens = [[i for i in range(16)], [i + 16 for i in range(16)], [i + 32 for i in range(16)]]
tokens[1][4] = 512
with open('$TOKENS_BAD_ID', 'w') as f:
    json.dump(tokens, f)
"
ERR_TXT="$TEST_DIR/err_bad_id.txt"
"$KIMO" head "$TOKENS_BAD_ID" --model-dir "$FX_DIR" --workdir "$WORKDIR" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on token ID out of bounds, got $status"
    fail=1
fi
grep -q '"error_type":"TOKEN_INVALID"' "$ERR_TXT" || { echo "FAIL: error_type not TOKEN_INVALID"; fail=1; }
grep -q '"stage":"embedding"' "$ERR_TXT" || { echo "FAIL: stage not embedding"; fail=1; }
grep -q '"prompt_idx":1' "$ERR_TXT" || { echo "FAIL: prompt_idx not 1"; fail=1; }
grep -q '"token_pos":4' "$ERR_TXT" || { echo "FAIL: token_pos not 4"; fail=1; }

echo "== 6. Negative token ID =="
TOKENS_NEG="$TEST_DIR/tokens_neg.json"
python3 -c "
import json
tokens = [[i for i in range(16)], [i + 16 for i in range(16)], [i + 32 for i in range(16)]]
tokens[2][3] = -5
with open('$TOKENS_NEG', 'w') as f:
    json.dump(tokens, f)
"
ERR_TXT="$TEST_DIR/err_neg.txt"
"$KIMO" head "$TOKENS_NEG" --model-dir "$FX_DIR" --workdir "$WORKDIR" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on negative token ID, got $status"
    fail=1
fi
grep -q '"error_type":"TOKEN_INVALID"' "$ERR_TXT" || { echo "FAIL: error_type not TOKEN_INVALID"; fail=1; }
grep -q '"prompt_idx":2' "$ERR_TXT" || { echo "FAIL: prompt_idx not 2"; fail=1; }

echo "== 7. Invalid prompt length != 16 =="
TOKENS_SHORT="$TEST_DIR/tokens_short.json"
python3 -c "
import json
tokens = [[i for i in range(15)], [i + 16 for i in range(16)], [i + 32 for i in range(16)]]
with open('$TOKENS_SHORT', 'w') as f:
    json.dump(tokens, f)
"
ERR_TXT="$TEST_DIR/err_short.txt"
"$KIMO" head "$TOKENS_SHORT" --model-dir "$FX_DIR" --workdir "$WORKDIR" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on prompt length != 16, got $status"
    fail=1
fi
grep -q '"error_type":"TOKEN_INVALID"' "$ERR_TXT" || { echo "FAIL: error_type not TOKEN_INVALID"; fail=1; }

echo "== 8. Invalid prompt count != 3 =="
TOKENS_COUNT="$TEST_DIR/tokens_count.json"
python3 -c "
import json
tokens = [[i for i in range(16)], [i + 16 for i in range(16)]]
with open('$TOKENS_COUNT', 'w') as f:
    json.dump(tokens, f)
"
ERR_TXT="$TEST_DIR/err_count.txt"
"$KIMO" head "$TOKENS_COUNT" --model-dir "$FX_DIR" --workdir "$WORKDIR" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on prompt count != 3, got $status"
    fail=1
fi
grep -q '"error_type":"TOKEN_INVALID"' "$ERR_TXT" || { echo "FAIL: error_type not TOKEN_INVALID"; fail=1; }

echo "== 9. Missing tokens file =="
ERR_TXT="$TEST_DIR/err_no_tokens.txt"
"$KIMO" head "$TEST_DIR/no_such_file.json" --model-dir "$FX_DIR" --workdir "$WORKDIR" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on missing tokens file, got $status"
    fail=1
fi
grep -q '"error_type":"FILE_NOT_FOUND"' "$ERR_TXT" || { echo "FAIL: error_type not FILE_NOT_FOUND"; fail=1; }

echo "== 10. Config error: missing rms_norm_eps =="
# fixtures/m0 lacks rms_norm_eps
ERR_TXT="$TEST_DIR/err_cfg_missing.txt"
"$KIMO" head "$TOKENS_VALID" --model-dir "fixtures/m0" --workdir "$WORKDIR" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on missing rms_norm_eps, got $status"
    fail=1
fi
grep -q '"error_type":"CONFIG_ERROR"' "$ERR_TXT" || { echo "FAIL: error_type not CONFIG_ERROR"; fail=1; }
grep -q '"stage":"config"' "$ERR_TXT" || { echo "FAIL: stage not config"; fail=1; }

echo "== 11. Config error: non-positive rms_norm_eps =="
BAD_CFG_DIR="$TEST_DIR/bad_cfg"
mkdir -p "$BAD_CFG_DIR"
cp "$FX_DIR"/* "$BAD_CFG_DIR/"
python3 -c "
import json
p = '$BAD_CFG_DIR/model_config.json'
with open(p) as f: d = json.load(f)
d['rms_norm_eps'] = 0.0
with open(p, 'w') as f: json.dump(d, f)
"
ERR_TXT="$TEST_DIR/err_cfg_zero.txt"
"$KIMO" head "$TOKENS_VALID" --model-dir "$BAD_CFG_DIR" --workdir "$WORKDIR" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on zero rms_norm_eps, got $status"
    fail=1
fi
grep -q '"error_type":"CONFIG_ERROR"' "$ERR_TXT" || { echo "FAIL: error_type not CONFIG_ERROR"; fail=1; }

echo "== 12. Workdir confinement: escape via ../ =="
ERR_TXT="$TEST_DIR/err_escape.txt"
"$KIMO" head "$TOKENS_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "$WORKDIR/../escaped.bin" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on workdir escape, got $status"
    fail=1
fi
grep -q '"error_type":"OUTPUT_WRITE_FAILED"' "$ERR_TXT" || { echo "FAIL: error_type not OUTPUT_WRITE_FAILED"; fail=1; }
if [ -f "$WORKDIR/../escaped.bin" ]; then
    echo "FAIL: escaped file was created!"
    rm -f "$WORKDIR/../escaped.bin"
    fail=1
fi

echo "== 13. Workdir confinement: escape via symlink =="
mkdir -p "$WORKDIR/sub"
ln -s /tmp "$WORKDIR/sub/sym_link"
ERR_TXT="$TEST_DIR/err_symlink.txt"
"$KIMO" head "$TOKENS_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "sub/sym_link/escaped.bin" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on symlink escape, got $status"
    fail=1
fi
grep -q '"error_type":"OUTPUT_WRITE_FAILED"' "$ERR_TXT" || { echo "FAIL: error_type not OUTPUT_WRITE_FAILED"; fail=1; }

echo "== 14. Atomic write rollback =="
RO_DIR="$WORKDIR/readonly_dir"
mkdir -p "$RO_DIR"
chmod 555 "$RO_DIR"
ERR_TXT="$TEST_DIR/err_ro.txt"
"$KIMO" head "$TOKENS_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "$RO_DIR/out.bin" > "$ERR_TXT" 2>&1
status=$?
if [ $status -ne 2 ]; then
    echo "FAIL: expected exit 2 on unwritable output path, got $status"
    fail=1
fi
grep -q '"error_type":"OUTPUT_WRITE_FAILED"' "$ERR_TXT" || { echo "FAIL: error_type not OUTPUT_WRITE_FAILED"; fail=1; }
# Verify no partial or tmp file exists
chmod 777 "$RO_DIR"
if [ -f "$RO_DIR/out.bin" ] || [ -f "$RO_DIR/out.bin.tmp.bin" ]; then
    echo "FAIL: leftover file in readonly directory!"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "=== All M1 Head CLI integration tests PASSED! ==="
else
    echo "=== Some M1 Head CLI integration tests FAILED! ==="
fi

exit $fail
