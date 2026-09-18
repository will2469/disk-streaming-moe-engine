#!/bin/bash
# E2E test suite for kimo layer CLI (M2-W4)
# Covers:
# 1. Happy path (--model-dir) for layer 0, 12, 23
# 2. Happy path (positional shards)
# 3. Output file format (exact byte size, fp32 LE, finite)
# 4. Residual check (output differs from input due to attention)
# 5. Output JSON schema & timing metrics
# 6. Layer validation (missing --layer, layer 5, layer -1, layer 24, layer abc) -> LAYER_INVALID
# 7. Activation validation (missing path, file not found, truncated, NaN/Inf, outlier) -> ACT_LOAD_FAILED / FILE_NOT_FOUND
# 8. Shard & model validation (missing shard, missing weight tensor, bias count mismatch) -> FILE_NOT_FOUND / WEIGHT_LOAD_FAILED
# 9. Workdir containment (escaping workdir via ../) -> OUTPUT_WRITE_FAILED
# 10. Atomic write rollback on unwritable directory -> OUTPUT_WRITE_FAILED, clean dir

set -u

KIMO="${KIMO:-./dismoen}"
TEST_DIR="/tmp/test_m2_w4_layer_cli_$$"
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

# 1. Generate synthetic fixture for M2 testing (24 layers, 72 biases, layers 0, 12, 23 fully populated)
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

# shard 1: layer 0 & 12
tensors_1 = []
for l in [0, 12]:
    pfx = f'model.layers.{l}.'
    tensors_1.append((f'{pfx}input_layernorm.weight', [64], lambda i: 1.0))
    tensors_1.append((f'{pfx}self_attn.q_proj.weight', [64, 64], lambda i: 0.05 if i % 65 == 0 else 0.0))
    tensors_1.append((f'{pfx}self_attn.k_proj.weight', [64, 64], lambda i: 0.05 if i % 65 == 0 else 0.0))
    tensors_1.append((f'{pfx}self_attn.v_proj.weight', [64, 64], lambda i: 0.05 if i % 65 == 0 else 0.0))
    tensors_1.append((f'{pfx}self_attn.o_proj.weight', [64, 64], lambda i: 0.05 if i % 65 == 0 else 0.0))

# shard 2: layer 23 + all 72 biases
tensors_2 = []
l = 23
pfx = f'model.layers.{l}.'
tensors_2.append((f'{pfx}input_layernorm.weight', [64], lambda i: 1.0))
tensors_2.append((f'{pfx}self_attn.q_proj.weight', [64, 64], lambda i: 0.05 if i % 65 == 0 else 0.0))
tensors_2.append((f'{pfx}self_attn.k_proj.weight', [64, 64], lambda i: 0.05 if i % 65 == 0 else 0.0))
tensors_2.append((f'{pfx}self_attn.v_proj.weight', [64, 64], lambda i: 0.05 if i % 65 == 0 else 0.0))
tensors_2.append((f'{pfx}self_attn.o_proj.weight', [64, 64], lambda i: 0.05 if i % 65 == 0 else 0.0))

for i in range(24):
    p = f'model.layers.{i}.self_attn.'
    tensors_2.append((f'{p}q_proj.bias', [64], lambda j: 0.01))
    tensors_2.append((f'{p}k_proj.bias', [64], lambda j: 0.01))
    tensors_2.append((f'{p}v_proj.bias', [64], lambda j: 0.01))

make_shard(tensors_1, 'shard-00001-of-00002.safetensors')
make_shard(tensors_2, 'shard-00002-of-00002.safetensors')

wmap = {}
for name, _, _ in tensors_1:
    wmap[name] = 'shard-00001-of-00002.safetensors'
for name, _, _ in tensors_2:
    wmap[name] = 'shard-00002-of-00002.safetensors'

with open(os.path.join(out, 'model.safetensors.index.json'), 'w') as f:
    json.dump({'metadata': {'total_size': 0}, 'weight_map': wmap}, f)

with open(os.path.join(out, 'model_config.json'), 'w') as f:
    json.dump({
        'hidden_size': 64,
        'num_hidden_layers': 24,
        'num_attention_heads': 2,
        'vocab_size': 512,
        'rms_norm_eps': 1e-6
    }, f)
"

ACT_VALID="$TEST_DIR/activation_valid.bin"
python3 -c "
import struct, math
with open('$ACT_VALID', 'wb') as f:
    for i in range(16 * 64):
        v = math.sin(float(i + 1)) * 0.1
        f.write(struct.pack('<f', v))
"

echo "=== 1. Happy path: Layer 0, 12, 23 with --model-dir ==="
for lyr in 0 12 23; do
    out_file="attn_out_${lyr}.bin"
    res=$("$KIMO" layer --layer "$lyr" "$ACT_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "$out_file")
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAIL: layer $lyr returned exit code $rc"
        fail=1
    else
        echo "PASS: layer $lyr exited 0"
    fi

    # Verify JSON output
    echo "$res" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
assert d['status'] == 'success', f'expected success, got {d}'
assert d['layer'] == $lyr, f'expected layer $lyr, got {d}'
assert d['num_tokens'] == 16, f'expected 16 tokens, got {d}'
assert d['output_file'] == '$out_file', f'expected $out_file, got {d}'
assert d['parse_time_ms'] > 0, 'parse_time_ms must be > 0'
assert d['compute_time_ms'] > 0, 'compute_time_ms must be > 0'
" || { echo "FAIL: layer $lyr output JSON invalid"; fail=1; }

    # Verify output file size: 16 * 64 * 4 = 4096 bytes
    out_path="$WORKDIR/$out_file"
    if [ ! -f "$out_path" ]; then
        echo "FAIL: output file not found: $out_path"
        fail=1
    else
        sz=$(stat -c %s "$out_path")
        if [ "$sz" -ne 4096 ]; then
            echo "FAIL: output file size $sz != 4096"
            fail=1
        else
            echo "PASS: output file size 4096 bytes verified"
        fi
    fi

    # Residual check: output ≈ input + attention (diff != 0, finite)
    python3 -c "
import struct, math
with open('$ACT_VALID', 'rb') as f:
    in_b = f.read()
with open('$out_path', 'rb') as f:
    out_b = f.read()
assert len(in_b) == len(out_b) == 16 * 64 * 4
max_diff = 0.0
for i in range(16 * 64):
    (v_in,) = struct.unpack_from('<f', in_b, i * 4)
    (v_out,) = struct.unpack_from('<f', out_b, i * 4)
    assert math.isfinite(v_out), f'non-finite value at {i}'
    d = abs(v_out - v_in)
    if d > max_diff:
        max_diff = d
assert max_diff > 1e-7, f'residual check failed: diff too small {max_diff}'
assert max_diff < 100.0, f'residual check failed: diff too large {max_diff}'
" || { echo "FAIL: residual check failed for layer $lyr"; fail=1; }
done

echo "=== 2. Happy path: Positional shards ==="
out_shards="attn_out_shards.bin"
res=$("$KIMO" layer --layer 0 "$ACT_VALID" "$FX_DIR/shard-00001-of-00002.safetensors" "$FX_DIR/shard-00002-of-00002.safetensors" --workdir "$WORKDIR" --output "$out_shards")
rc=$?
if [ $rc -ne 0 ]; then
    echo "FAIL: positional shards mode returned exit code $rc"
    fail=1
else
    echo "PASS: positional shards mode exited 0"
fi
echo "$res" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
assert d['status'] == 'success'
assert d['layer'] == 0
" || { echo "FAIL: positional shards JSON invalid"; fail=1; }

echo "=== 3. Layer validation: Invalid layer values ==="
# Missing --layer
err=$("$KIMO" layer "$ACT_VALID" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"LAYER_INVALID"'; then
    echo "PASS: missing --layer rejected with LAYER_INVALID (exit 2)"
else
    echo "FAIL: missing --layer not handled correctly: rc=$rc, err=$err"
    fail=1
fi

# Invalid layer 5 (not in {0, 12, 23})
err=$("$KIMO" layer --layer 5 "$ACT_VALID" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"LAYER_INVALID"'; then
    echo "PASS: layer 5 rejected with LAYER_INVALID (exit 2)"
else
    echo "FAIL: layer 5 not rejected: rc=$rc, err=$err"
    fail=1
fi

# Negative layer -1
err=$("$KIMO" layer --layer -1 "$ACT_VALID" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"LAYER_INVALID"'; then
    echo "PASS: layer -1 rejected with LAYER_INVALID (exit 2)"
else
    echo "FAIL: layer -1 not rejected: rc=$rc, err=$err"
    fail=1
fi

# Layer 24 (out of bounds)
err=$("$KIMO" layer --layer 24 "$ACT_VALID" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"LAYER_INVALID"'; then
    echo "PASS: layer 24 rejected with LAYER_INVALID (exit 2)"
else
    echo "FAIL: layer 24 not rejected: rc=$rc, err=$err"
    fail=1
fi

# Non-numeric layer 'abc'
err=$("$KIMO" layer --layer abc "$ACT_VALID" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"LAYER_INVALID"'; then
    echo "PASS: layer abc rejected with LAYER_INVALID (exit 2)"
else
    echo "FAIL: layer abc not rejected: rc=$rc, err=$err"
    fail=1
fi

echo "=== 4. Activation validation ==="
# Missing activation file argument
err=$("$KIMO" layer --layer 0 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"ACT_LOAD_FAILED"'; then
    echo "PASS: missing activation path rejected with ACT_LOAD_FAILED (exit 2)"
else
    echo "FAIL: missing activation path not handled: rc=$rc, err=$err"
    fail=1
fi

# Activation file does not exist
err=$("$KIMO" layer --layer 0 "$TEST_DIR/nonexistent.bin" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"FILE_NOT_FOUND"'; then
    echo "PASS: nonexistent activation rejected with FILE_NOT_FOUND (exit 2)"
else
    echo "FAIL: nonexistent activation not handled: rc=$rc, err=$err"
    fail=1
fi

# Truncated activation file
ACT_SHORT="$TEST_DIR/act_short.bin"
head -c 100 "$ACT_VALID" > "$ACT_SHORT"
err=$("$KIMO" layer --layer 0 "$ACT_SHORT" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"ACT_LOAD_FAILED"'; then
    echo "PASS: truncated activation rejected with ACT_LOAD_FAILED (exit 2)"
else
    echo "FAIL: truncated activation not handled: rc=$rc, err=$err"
    fail=1
fi

# Activation with NaN
ACT_NAN="$TEST_DIR/act_nan.bin"
python3 -c "
import struct, math
with open('$ACT_NAN', 'wb') as f:
    for i in range(16 * 64):
        v = float('nan') if i == 5 else math.sin(float(i + 1)) * 0.1
        f.write(struct.pack('<f', v))
"
err=$("$KIMO" layer --layer 0 "$ACT_NAN" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"ACT_LOAD_FAILED"'; then
    echo "PASS: NaN activation rejected with ACT_LOAD_FAILED (exit 2)"
else
    echo "FAIL: NaN activation not handled: rc=$rc, err=$err"
    fail=1
fi

# Activation with Inf
ACT_INF="$TEST_DIR/act_inf.bin"
python3 -c "
import struct, math
with open('$ACT_INF', 'wb') as f:
    for i in range(16 * 64):
        v = float('inf') if i == 10 else math.sin(float(i + 1)) * 0.1
        f.write(struct.pack('<f', v))
"
err=$("$KIMO" layer --layer 0 "$ACT_INF" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"ACT_LOAD_FAILED"'; then
    echo "PASS: Inf activation rejected with ACT_LOAD_FAILED (exit 2)"
else
    echo "FAIL: Inf activation not handled: rc=$rc, err=$err"
    fail=1
fi

# Activation with outlier (> 1e6)
ACT_OUTLIER="$TEST_DIR/act_outlier.bin"
python3 -c "
import struct, math
with open('$ACT_OUTLIER', 'wb') as f:
    for i in range(16 * 64):
        v = 2e6 if i == 10 else math.sin(float(i + 1)) * 0.1
        f.write(struct.pack('<f', v))
"
err=$("$KIMO" layer --layer 0 "$ACT_OUTLIER" --model-dir "$FX_DIR" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"ACT_LOAD_FAILED"'; then
    echo "PASS: outlier activation rejected with ACT_LOAD_FAILED (exit 2)"
else
    echo "FAIL: outlier activation not handled: rc=$rc, err=$err"
    fail=1
fi

echo "=== 5. Shard & Model Validation ==="
# Missing shard file on disk
err=$("$KIMO" layer --layer 0 "$ACT_VALID" "$TEST_DIR/shard_ghost.safetensors" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"FILE_NOT_FOUND"'; then
    echo "PASS: missing shard on disk rejected with FILE_NOT_FOUND (exit 2)"
else
    echo "FAIL: missing shard on disk not handled: rc=$rc, err=$err"
    fail=1
fi

# Shards missing required layer weights (shard 2 only has layer 23, but we request layer 0)
err=$("$KIMO" layer --layer 0 "$ACT_VALID" "$FX_DIR/shard-00002-of-00002.safetensors" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"WEIGHT_LOAD_FAILED"'; then
    echo "PASS: shard missing layer weights rejected with WEIGHT_LOAD_FAILED (exit 2)"
else
    echo "FAIL: shard missing layer weights not handled: rc=$rc, err=$err"
    fail=1
fi

# Checkpoint with bias count mismatch
FX_BAD_BIAS="$TEST_DIR/fx_bad_bias"
mkdir -p "$FX_BAD_BIAS"
cp "$FX_DIR/shard-00001-of-00002.safetensors" "$FX_BAD_BIAS/"
cp "$FX_DIR/shard-00002-of-00002.safetensors" "$FX_BAD_BIAS/"
cp "$FX_DIR/model_config.json" "$FX_BAD_BIAS/"
python3 -c "
import json
with open('$FX_DIR/model.safetensors.index.json') as f:
    idx = json.load(f)
# delete one bias
del idx['weight_map']['model.layers.0.self_attn.q_proj.bias']
with open('$FX_BAD_BIAS/model.safetensors.index.json', 'w') as f:
    json.dump(idx, f)
"
err=$("$KIMO" layer --layer 0 "$ACT_VALID" --model-dir "$FX_BAD_BIAS" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"WEIGHT_LOAD_FAILED"'; then
    echo "PASS: bias count mismatch rejected with WEIGHT_LOAD_FAILED (exit 2)"
else
    echo "FAIL: bias count mismatch not handled: rc=$rc, err=$err"
    fail=1
fi

echo "=== 6. Workdir Containment & Atomic Rollback ==="
# Workdir escape
err=$("$KIMO" layer --layer 0 "$ACT_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "../escaped.bin" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"OUTPUT_WRITE_FAILED"'; then
    echo "PASS: workdir escape rejected with OUTPUT_WRITE_FAILED (exit 2)"
else
    echo "FAIL: workdir escape not handled: rc=$rc, err=$err"
    fail=1
fi

# Atomic write rollback on unwritable directory
RO_DIR="$TEST_DIR/ro_dir"
mkdir -p "$RO_DIR"
chmod 555 "$RO_DIR"
err=$("$KIMO" layer --layer 0 "$ACT_VALID" --model-dir "$FX_DIR" --workdir "$RO_DIR" --output "locked.bin" 2>&1)
rc=$?
chmod 755 "$RO_DIR"
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"OUTPUT_WRITE_FAILED"'; then
    # Verify no .tmp file left (tmp pattern: <target>.tmp.<pid>.<ns>.<attempt>)
    tmp_count=$(find "$RO_DIR" -name "*.tmp.*" | wc -l)
    if [ "$tmp_count" -eq 0 ]; then
        echo "PASS: atomic write failed cleanly with rollback (exit 2, 0 tmp files)"
    else
        echo "FAIL: atomic write left lingering tmp files: $tmp_count"
        fail=1
    fi
else
    echo "FAIL: unwritable output dir not handled: rc=$rc, err=$err"
    fail=1
fi

# Ambiguous invocation: --model-dir forbids positional shards
err=$("$KIMO" layer --layer 0 "$ACT_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" "$FX_DIR/shard-00001-of-00002.safetensors" 2>&1)
rc=$?
if [ $rc -eq 2 ] && echo "$err" | grep -q '"error_type":"USAGE"'; then
    echo "PASS: ambiguous --model-dir + positional shards rejected (exit 2)"
else
    echo "FAIL: ambiguous invocation not handled: rc=$rc, err=$err"
    fail=1
fi

echo "=== 7. Real Qwen Checkpoint (Optional Sanity Check) ==="
REAL_MODEL="/home/will/models/qwen1.5-moe-a2.7b-chat"
if [ -d "$REAL_MODEL" ]; then
    echo "Testing real model at $REAL_MODEL..."
    ACT_REAL="$TEST_DIR/act_real_16x2048.bin"
    python3 -c "
import struct, math
with open('$ACT_REAL', 'wb') as f:
    for i in range(16 * 2048):
        v = math.sin(float(i + 1)) * 0.1
        f.write(struct.pack('<f', v))
"
    for lyr in 0 12 23; do
        res=$("$KIMO" layer --layer "$lyr" "$ACT_REAL" --model-dir "$REAL_MODEL" --workdir "$WORKDIR" --output "real_out_${lyr}.bin")
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "PASS: real model layer $lyr succeeded (exit 0)"
            sz=$(stat -c %s "$WORKDIR/real_out_${lyr}.bin")
            if [ "$sz" -ne 131072 ]; then
                echo "FAIL: real model output size $sz != 131072"
                fail=1
            fi
        else
            echo "FAIL: real model layer $lyr failed with rc=$rc: $res"
            fail=1
        fi
    done
else
    echo "SKIP: real model directory $REAL_MODEL not found"
fi

if [ $fail -eq 0 ]; then
    echo "ALL M2-W4 TESTS PASSED!"
    exit 0
else
    echo "SOME M2-W4 TESTS FAILED!"
    exit 1
fi
