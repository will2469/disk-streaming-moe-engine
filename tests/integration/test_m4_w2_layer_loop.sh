#!/bin/bash
# ==============================================================================
# test_m4_w2_layer_loop.sh — Integration Test Suite M4-W2 (Streaming Layer Loop)
#
# Memverifikasi DoD M4-W2:
# 1. Full forward 24-layer streaming inference (l = 0..23) pada checkpoint asli
#    Qwen1.5-MoE-A2.7B-Chat (exit 0).
# 2. Output logits FP32 binary tepat 16 * 151936 * 4 = 9.723.904 byte,
#    semua float finite (tanpa NaN/Inf).
# 3. Memory budget bound: Peak VmHWM <= 5 GiB gate (margin ~1.5 GiB, terukur ~3.4 GiB),
#    cgroup_oom_kills == 0 (zero leak antar-layer).
# 4. Routing dump Tier-1: tepat 24 file routing_L0.json .. routing_L23.json
#    dengan format {"selected_experts": [[4 ID] x 16]}.
# 5. Layer timing: memuat rincian pread_sec, attention_sec, moe_sec, total_sec
#    untuk ke-24 layer transformer.
# ==============================================================================

set -u

KIMO="${KIMO:-./kimo}"
MODEL_DIR="${MODEL_DIR:-/home/will/models/qwen1.5-moe-a2.7b-chat}"

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "SKIP: model directory not found: $MODEL_DIR"
    exit 0
fi

TEST_DIR="/tmp/test_m4_w2_layer_loop_$$"
WORKDIR="$TEST_DIR/work"
ROUTING_DIR="$TEST_DIR/routing"
TOKENS="$TEST_DIR/tokens_16.json"
OUT_LOGITS="$WORKDIR/logits_m4_w2.bin"
TIMING_FILE="$WORKDIR/timing_m4_w2.json"
STDOUT_JSON="$TEST_DIR/stdout.json"
STDERR_TXT="$TEST_DIR/stderr.txt"

fail=0

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
[ "${1:-}" = "--cleanup" ] && cleanup

mkdir -p "$WORKDIR" "$ROUTING_DIR"

echo "======================================================================"
echo "M4-W2: Full Forward 24-Layer Streaming Loop & Residual Order"
echo "======================================================================"

# 1. Siapkan prompt 16 token
python3 -c "
import json
tokens = [1234, 5678, 9012, 3456, 7890, 2345, 6789, 123, 4567, 8901, 2345, 6789, 123, 4567, 8901, 2345]
with open('$TOKENS', 'w') as f:
    json.dump(tokens, f)
"

# 2. Jalankan forward 24 layer streaming
echo ">> Menjalankan kimo forward 24 layer..."
"$KIMO" forward \
  --model-dir "$MODEL_DIR" \
  --tokens "$TOKENS" \
  --output "$OUT_LOGITS" \
  --workdir "$WORKDIR" \
  --dump-routing "$ROUTING_DIR" \
  --layer-timing "$TIMING_FILE" \
  --threads 1 \
  > "$STDOUT_JSON" 2> "$STDERR_TXT"
status=$?

if [ $status -ne 0 ]; then
    echo "FAIL: kimo forward returned exit code $status (expected 0)"
    cat "$STDERR_TXT"
    exit 1
fi

if [ -s "$STDERR_TXT" ]; then
    echo "FAIL: stderr not empty on success:"
    cat "$STDERR_TXT"
    fail=1
fi

echo ">> Memverifikasi stdout JSON schema dan metrik..."
python3 -c "
import json, re

with open('$STDOUT_JSON') as f:
    d = json.load(f)

assert d['status'] == 'success', f'status not success: {d}'
assert re.match(r'^M4-\d{8}-\d{3}$', d['run_id']), f'invalid run_id: {d[\"run_id\"]}'
assert d['model'] == 'qwen1.5-moe-a2.7b-chat'
assert d['num_tokens'] == 16, f'num_tokens mismatch: {d[\"num_tokens\"]}'
assert d['num_layers'] == 24, f'num_layers mismatch: {d[\"num_layers\"]}'
assert d['logits_path'] == '$OUT_LOGITS'

m = d['metrics']
# Waktu total harus wajar (< 300 detik)
assert m['walltime_sec'] > 0.0 and m['walltime_sec'] < 300.0, f'walltime invalid: {m[\"walltime_sec\"]}'

# Peak memory bound: VmHWM <= 5 GiB (5368709120 bytes)
assert m['vmhwm_bytes'] <= 5368709120, f'VmHWM exceeded 5 GiB gate: {m[\"vmhwm_bytes\"]} bytes'
assert m['cgroup_oom_kills'] == 0, f'cgroup oom kills occurred: {m[\"cgroup_oom_kills\"]}'

# Akuntansi byte logis I/O: headers + resident + streamed layers > 15 GB
assert m['logical_bytes_read'] > 15000000000, f'logical bytes read suspiciously low: {m[\"logical_bytes_read\"]}'

p = m['phases']
for k in ['index_load_sec', 'embedding_sec', 'layer_forward_sec', 'final_norm_sec', 'lm_head_sec', 'write_sec']:
    assert k in p, f'missing phase: {k}'
    assert p[k] > 0.0, f'phase timing not positive: {k} = {p[k]}'

print(f'   PASS: VmHWM = {m[\"vmhwm_bytes\"] / (1024**3):.2f} GiB (<= 5 GiB gate)')
print(f'   PASS: Walltime = {m[\"walltime_sec\"]:.1f}s, Layers = {p[\"layer_forward_sec\"]:.1f}s')
" || { echo "FAIL: stdout metrics validation failed"; fail=1; }

echo ">> Memverifikasi format biner dan nilai numerik logits..."
actual_size=$(wc -c < "$OUT_LOGITS")
expected_size=9723904
if [ "$actual_size" -ne "$expected_size" ]; then
    echo "FAIL: logits binary size $actual_size != expected $expected_size bytes"
    fail=1
else
    echo "   PASS: Ukuran file logits tepat $actual_size byte (16 x 151936 x 4)"
fi

python3 -c "
import struct, math

with open('$OUT_LOGITS', 'rb') as f:
    data = f.read()

assert len(data) == 16 * 151936 * 4, f'file size mismatch: {len(data)}'

count = 0
min_val = 1e9
max_val = -1e9
sum_val = 0.0

for (val,) in struct.iter_unpack('<f', data):
    assert not math.isnan(val), f'NaN found at index {count}'
    assert not math.isinf(val), f'Inf found at index {count}'
    if val < min_val:
        min_val = val
    if val > max_val:
        max_val = val
    sum_val += val
    count += 1

assert count == 16 * 151936, f'elements count mismatch: {count}'
assert min_val > -50.0, f'suspicious low minimum logit: {min_val}'
assert max_val < 50.0, f'suspicious high maximum logit: {max_val}'

mean_val = sum_val / count
print(f'   PASS: Logits finite, min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}')
" || { echo "FAIL: logits numerical sanity failed"; fail=1; }

echo ">> Memverifikasi Tier-1 Routing Dumps (24 layer)..."
routing_count=$(find "$ROUTING_DIR" -name "routing_L*.json" | wc -l)
if [ "$routing_count" -ne 24 ]; then
    echo "FAIL: expected 24 routing dump files, found $routing_count"
    fail=1
else
    echo "   PASS: Tepat 24 file routing_L0..routing_L23 ditemukan"
fi

python3 -c "
import json, os

for l in range(24):
    path = os.path.join('$ROUTING_DIR', f'routing_L{l}.json')
    assert os.path.exists(path), f'missing routing file: {path}'
    with open(path) as f:
        data = json.load(f)
    assert 'selected_experts' in data, f'missing selected_experts in {path}'
    exp = data['selected_experts']
    assert len(exp) == 16, f'layer {l} selected_experts length != 16 tokens'
    for t, tok_exp in enumerate(exp):
        assert len(tok_exp) == 4, f'layer {l} token {t} does not have exactly 4 experts: {tok_exp}'
        for eid in tok_exp:
            assert 0 <= eid < 60, f'expert ID {eid} out of range 0..59 in layer {l}'

print('   PASS: Semua 24 file routing dump valid (16 token x 4 expert ID per layer)')
" || { echo "FAIL: routing dump validation failed"; fail=1; }

echo ">> Memverifikasi Layer Timing Breakdown..."
if [ ! -f "$TIMING_FILE" ]; then
    echo "FAIL: layer timing file not created: $TIMING_FILE"
    fail=1
else
    python3 -c "
import json

with open('$TIMING_FILE') as f:
    d = json.load(f)

assert 'layer_timing' in d, 'missing layer_timing in timing file'
lt = d['layer_timing']
assert len(lt) == 24, f'layer_timing count {len(lt)} != 24'

for idx, entry in enumerate(lt):
    assert entry['layer'] == idx, f'layer index mismatch at {idx}: {entry[\"layer\"]}'
    assert entry['pread_sec'] > 0.0, f'pread_sec not positive in layer {idx}'
    assert entry['attention_sec'] > 0.0, f'attention_sec not positive in layer {idx}'
    assert entry['moe_sec'] > 0.0, f'moe_sec not positive in layer {idx}'
    assert entry['total_sec'] > 0.0, f'total_sec not positive in layer {idx}'

assert d['total_layer_forward_sec'] > 0.0, 'total_layer_forward_sec not positive'
print(f'   PASS: 24 layer timing entries valid, total compute+pread={d[\"total_layer_forward_sec\"]:.1f}s')
" || { echo "FAIL: layer timing validation failed"; fail=1; }
fi

echo ">> Memverifikasi kebersihan workdir (zero orphan temp files)..."
if [ -d "$WORKDIR/runs" ] && [ -n "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ]; then
    echo "FAIL: orphan files left in $WORKDIR/runs"
    find "$WORKDIR/runs"
    fail=1
else
    echo "   PASS: workdir/runs bersih (semua temp file dibersihkan)"
fi

if [ $fail -ne 0 ]; then
    echo "SOME M4-W2 INTEGRATION TESTS FAILED"
    exit 1
fi

echo "ALL M4-W2 INTEGRATION TESTS PASSED"
exit 0
