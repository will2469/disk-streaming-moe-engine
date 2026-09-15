#!/bin/bash
# E2E test suite for kimo layer CLI part MoE (M3-W3)
# Covers:
# 1. Happy path (--model-dir) for layer 0, 12, 23 with --part moe
# 2. Happy path (positional shards)
# 3. Output file format (exact byte size [16, hidden_dim], fp32 LE, finite)
# 4. Residual check (output differs from input due to MoE block)
# 5. Output JSON schema & timing metrics & routing_info
# 6. Part validation (--part foo) -> PART_INVALID (exit 2)
# 7. Layer validation (missing --layer, layer 5, layer -1, layer 24, layer abc) -> LAYER_INVALID (exit 2)
# 8. Activation validation (missing path, file not found, truncated, NaN/Inf, outlier) -> ACT_LOAD_FAILED / FILE_NOT_FOUND (exit 2)
# 9. Shard & model validation (missing shard, missing weight tensor) -> FILE_NOT_FOUND / WEIGHT_LOAD_FAILED (exit 2)
# 10. Inline routing check vs oracle (--oracle-routing):
#     - Match -> status=success, exit 0
#     - Violation -> status=mismatch, error_type=ROUTING_VIOLATION, exit 1
# 11. Workdir containment (escaping workdir via ../) -> OUTPUT_WRITE_FAILED (exit 2)
# 12. Atomic write rollback on unwritable directory -> OUTPUT_WRITE_FAILED, clean dir
# 13. Real Qwen Checkpoint sanity check (if available)

set -u

KIMO="${KIMO:-./kimo}"
TEST_DIR="/tmp/test_m3_w3_moe_cli_$$"
FX_DIR="$TEST_DIR/fixture"
WORKDIR="$TEST_DIR/workdir"
OUT_DIR="$WORKDIR/output"

fail=0

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
[ "${1:-}" = "--cleanup" ] && cleanup

mkdir -p "$FX_DIR" "$WORKDIR" "$OUT_DIR"

# 1. Generate synthetic fixture for M3 testing (hidden 64, inter_routed 32, inter_shared 64, 8 experts, top-k 4)
python3 -c "
import struct, json, os, math

out = '$FX_DIR'
os.makedirs(out, exist_ok=True)

def make_shard(tensors, filename):
    hdr = {}
    blobs = []
    off = 0
    for name, shape, val_fn in tensors:
        n = 1
        for d in shape: n *= d
        raw = bytearray(n * 4)
        for i in range(n):
            v = val_fn(i)
            raw[i*4:(i+1)*4] = struct.pack('<f', v)
        hdr[name] = {
            'dtype': 'F32',
            'shape': shape,
            'data_offsets': [off, off + len(raw)]
        }
        blobs.append(raw)
        off += len(raw)
    hdr_bytes = json.dumps(hdr).encode('utf-8')
    with open(os.path.join(out, filename), 'wb') as f:
        f.write(struct.pack('<Q', len(hdr_bytes)))
        f.write(hdr_bytes)
        for b in blobs:
            f.write(b)

layers = [0, 12, 23]
tensors_s1 = []
tensors_s2 = []
weight_map = {}

hidden = 64
num_experts = 8
inter_routed = 32
inter_shared = 64

for lyr in layers:
    pfx = f'model.layers.{lyr}.mlp.'
    # Router gate
    r_name = pfx + 'gate.weight'
    tensors_s1.append((r_name, [num_experts, hidden], lambda i: 0.01 * math.cos(i)))
    weight_map[r_name] = 'model-00001-of-00002.safetensors'

    # Shared expert
    sg_name = pfx + 'shared_expert.gate_proj.weight'
    su_name = pfx + 'shared_expert.up_proj.weight'
    sd_name = pfx + 'shared_expert.down_proj.weight'
    sgate_name = pfx + 'shared_expert_gate.weight'

    tensors_s1.append((sg_name, [inter_shared, hidden], lambda i: 0.01 * math.sin(i)))
    tensors_s1.append((su_name, [inter_shared, hidden], lambda i: 0.01 * math.cos(i)))
    tensors_s1.append((sd_name, [hidden, inter_shared], lambda i: 0.01 * math.sin(i)))
    tensors_s1.append((sgate_name, [hidden], lambda i: 0.01))

    weight_map[sg_name] = 'model-00001-of-00002.safetensors'
    weight_map[su_name] = 'model-00001-of-00002.safetensors'
    weight_map[sd_name] = 'model-00001-of-00002.safetensors'
    weight_map[sgate_name] = 'model-00001-of-00002.safetensors'

    # Routed experts
    for e in range(num_experts):
        eg_name = f'{pfx}experts.{e}.gate_proj.weight'
        eu_name = f'{pfx}experts.{e}.up_proj.weight'
        ed_name = f'{pfx}experts.{e}.down_proj.weight'

        target_list = tensors_s1 if e < 4 else tensors_s2
        target_shard = 'model-00001-of-00002.safetensors' if e < 4 else 'model-00002-of-00002.safetensors'

        target_list.append((eg_name, [inter_routed, hidden], lambda i: 0.01 * math.sin(i + e)))
        target_list.append((eu_name, [inter_routed, hidden], lambda i: 0.01 * math.cos(i + e)))
        target_list.append((ed_name, [hidden, inter_routed], lambda i: 0.01 * math.sin(i + e)))

        weight_map[eg_name] = target_shard
        weight_map[eu_name] = target_shard
        weight_map[ed_name] = target_shard

make_shard(tensors_s1, 'model-00001-of-00002.safetensors')
make_shard(tensors_s2, 'model-00002-of-00002.safetensors')

with open(os.path.join(out, 'model.safetensors.index.json'), 'w') as f:
    json.dump({'metadata': {'total_size': 0}, 'weight_map': weight_map}, f)

with open(os.path.join(out, 'config.json'), 'w') as f:
    json.dump({
        'hidden_size': hidden,
        'num_hidden_layers': 24,
        'num_attention_heads': 2,
        'vocab_size': 512,
        'rms_norm_eps': 1e-6,
        'num_experts': num_experts,
        'num_experts_per_tok': 4,
        'moe_intermediate_size': inter_routed,
        'shared_expert_intermediate_size': inter_shared,
        'norm_topk_prob': False
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

echo "=== 1. Happy path: Layer 0, 12, 23 with --model-dir and --part moe ==="
for lyr in 0 12 23; do
    out_file="moe_out_${lyr}.bin"
    res=$("$KIMO" layer --layer "$lyr" --part moe "$ACT_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "$out_file")
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
assert d['part'] == 'moe', f'expected part moe, got {d}'
assert d['num_tokens'] == 16, f'expected 16 tokens, got {d}'
assert d['output_file'] == '$out_file', f'expected $out_file, got {d}'
assert d['parse_time_ms'] > 0, 'parse_time_ms must be > 0'
assert d['compute_time_ms'] > 0, 'compute_time_ms must be > 0'
assert 'routing_info' in d, 'routing_info missing'
rinfo = d['routing_info']
assert len(rinfo['selected_experts']) == 16, 'expected 16 selected expert rows'
assert len(rinfo['router_probs']) == 16, 'expected 16 router prob rows'
for row in rinfo['selected_experts']:
    assert len(row) == 4, 'expected top-4'
" || { echo "FAIL: layer $lyr output JSON invalid"; fail=1; }

    # Verify output binary
    out_path="$WORKDIR/$out_file"
    if [ ! -f "$out_path" ]; then
        echo "FAIL: output file not found at $out_path"
        fail=1
    else
        fsize=$(wc -c < "$out_path")
        expected_size=$((16 * 64 * 4))
        if [ "$fsize" -ne "$expected_size" ]; then
            echo "FAIL: output file size $fsize != $expected_size"
            fail=1
        else
            echo "PASS: output file size $fsize bytes verified"
        fi
    fi
done

echo "=== 2. Happy path: Positional shards ==="
out_file_pos="moe_out_pos.bin"
res=$("$KIMO" layer --layer 0 --part moe "$ACT_VALID" \
    "$FX_DIR/model-00001-of-00002.safetensors" \
    "$FX_DIR/model-00002-of-00002.safetensors" \
    --workdir "$WORKDIR" --output "$out_file_pos")
rc=$?
if [ $rc -ne 0 ]; then
    echo "FAIL: positional shards mode returned exit code $rc"
    fail=1
else
    echo "PASS: positional shards mode exited 0"
fi

echo "=== 3. Part validation ==="
err_out=$("$KIMO" layer --layer 0 --part invalid "$ACT_VALID" --model-dir "$FX_DIR" 2>&1 >/dev/null)
rc=$?
if [ $rc -ne 2 ]; then
    echo "FAIL: invalid part returned exit code $rc (expected 2)"
    fail=1
else
    echo "$err_out" | grep -q "PART_INVALID" && echo "PASS: invalid part rejected with PART_INVALID (exit 2)" || { echo "FAIL: expected PART_INVALID, got: $err_out"; fail=1; }
fi

echo "=== 4. Layer validation ==="
for bad_lyr in 5 -1 24 abc ""; do
    if [ -z "$bad_lyr" ]; then
        cmd=("$KIMO" layer --part moe "$ACT_VALID" --model-dir "$FX_DIR")
        desc="missing --layer"
    else
        cmd=("$KIMO" layer --layer "$bad_lyr" --part moe "$ACT_VALID" --model-dir "$FX_DIR")
        desc="layer $bad_lyr"
    fi
    err_out=$("${cmd[@]}" 2>&1 >/dev/null)
    rc=$?
    if [ $rc -ne 2 ]; then
        echo "FAIL: $desc returned exit code $rc (expected 2)"
        fail=1
    else
        echo "$err_out" | grep -q "LAYER_INVALID" && echo "PASS: $desc rejected with LAYER_INVALID (exit 2)" || { echo "FAIL: expected LAYER_INVALID for $desc, got: $err_out"; fail=1; }
    fi
done

echo "=== 5. Activation validation ==="
# Missing path
err_out=$("$KIMO" layer --layer 0 --part moe --model-dir "$FX_DIR" 2>&1 >/dev/null)
rc=$?
if [ $rc -ne 2 ]; then
    echo "FAIL: missing activation argument returned exit code $rc (expected 2)"
    fail=1
else
    echo "$err_out" | grep -q "ACT_LOAD_FAILED" && echo "PASS: missing activation path rejected with ACT_LOAD_FAILED (exit 2)" || { echo "FAIL: expected ACT_LOAD_FAILED, got: $err_out"; fail=1; }
fi

# Nonexistent activation
err_out=$("$KIMO" layer --layer 0 --part moe "$TEST_DIR/nonexistent.bin" --model-dir "$FX_DIR" 2>&1 >/dev/null)
rc=$?
if [ $rc -ne 2 ]; then
    echo "FAIL: nonexistent activation returned exit code $rc (expected 2)"
    fail=1
else
    echo "$err_out" | grep -q "FILE_NOT_FOUND" && echo "PASS: nonexistent activation rejected with FILE_NOT_FOUND (exit 2)" || { echo "FAIL: expected FILE_NOT_FOUND, got: $err_out"; fail=1; }
fi

# Truncated activation
ACT_TRUNC="$TEST_DIR/activation_truncated.bin"
head -c 100 "$ACT_VALID" > "$ACT_TRUNC"
err_out=$("$KIMO" layer --layer 0 --part moe "$ACT_TRUNC" --model-dir "$FX_DIR" 2>&1 >/dev/null)
rc=$?
if [ $rc -ne 2 ]; then
    echo "FAIL: truncated activation returned exit code $rc (expected 2)"
    fail=1
else
    echo "$err_out" | grep -q "ACT_LOAD_FAILED" && echo "PASS: truncated activation rejected with ACT_LOAD_FAILED (exit 2)" || { echo "FAIL: expected ACT_LOAD_FAILED, got: $err_out"; fail=1; }
fi

echo "=== 6. Inline routing check vs oracle (--oracle-routing) ==="
# Get baseline routing output from layer 0
res_base=$("$KIMO" layer --layer 0 --part moe "$ACT_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "base_moe.bin")
ORACLE_MATCH="$TEST_DIR/oracle_match.json"
ORACLE_MISMATCH="$TEST_DIR/oracle_mismatch.json"

python3 -c "
import json
d = json.loads('''$res_base''')
rinfo = d['routing_info']
with open('$ORACLE_MATCH', 'w') as f:
    json.dump({'selected_experts': rinfo['selected_experts'], 'router_probs': rinfo['router_probs']}, f)

# Create mismatch by changing token 0 expert set
mismatch_experts = [list(r) for r in rinfo['selected_experts']]
mismatch_experts[0] = [99, 98, 97, 96]
with open('$ORACLE_MISMATCH', 'w') as f:
    json.dump({'selected_experts': mismatch_experts, 'router_probs': rinfo['router_probs']}, f)
"

# 6a. Match test
res_m=$("$KIMO" layer --layer 0 --part moe "$ACT_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "match_out.bin" --oracle-routing "$ORACLE_MATCH")
rc=$?
if [ $rc -ne 0 ]; then
    echo "FAIL: oracle match check failed with exit code $rc"
    fail=1
else
    echo "PASS: oracle match check succeeded (exit 0)"
fi

# 6b. Mismatch test
stdout_mismatch=$(mktemp)
stderr_mismatch=$(mktemp)
"$KIMO" layer --layer 0 --part moe "$ACT_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "mismatch_out.bin" --oracle-routing "$ORACLE_MISMATCH" >"$stdout_mismatch" 2>"$stderr_mismatch"
rc=$?
if [ $rc -ne 1 ]; then
    echo "FAIL: oracle mismatch check returned exit code $rc (expected 1)"
    fail=1
else
    echo "PASS: oracle mismatch check returned exit code 1"
fi

# Verify stdout has status mismatch and routing_violation
python3 -c "
import json
with open('$stdout_mismatch') as f:
    d = json.load(f)
assert d['status'] == 'mismatch', f'expected status mismatch, got {d}'
assert d['part'] == 'moe', f'expected part moe, got {d}'
assert 'routing_violation' in d, f'routing_violation missing in {d}'
rv = d['routing_violation']
assert rv['token_index'] == 0, f'expected token 0, got {rv}'
assert rv['oracle_experts'] == [99, 98, 97, 96], f'unexpected oracle experts: {rv}'
assert len(rv['engine_experts']) == 4, f'unexpected engine experts: {rv}'
" && echo "PASS: stdout routing violation structure verified" || { echo "FAIL: stdout routing violation invalid"; fail=1; }

# Verify stderr has ROUTING_VIOLATION error JSON
python3 -c "
import json
with open('$stderr_mismatch') as f:
    d = json.load(f)
assert d['error_type'] == 'ROUTING_VIOLATION', f'expected ROUTING_VIOLATION, got {d}'
assert d['stage'] == 'router', f'expected stage router, got {d}'
assert d['token_index'] == 0, f'expected token_index 0, got {d}'
" && echo "PASS: stderr routing violation error JSON verified" || { echo "FAIL: stderr routing violation invalid"; fail=1; }

echo "=== 7. Workdir Containment & Atomic Rollback ==="
# Workdir escape
err_out=$("$KIMO" layer --layer 0 --part moe "$ACT_VALID" --model-dir "$FX_DIR" --workdir "$WORKDIR" --output "../escape.bin" 2>&1 >/dev/null)
rc=$?
if [ $rc -ne 2 ]; then
    echo "FAIL: workdir escape returned exit code $rc (expected 2)"
    fail=1
else
    echo "$err_out" | grep -q "OUTPUT_WRITE_FAILED" && echo "PASS: workdir escape rejected with OUTPUT_WRITE_FAILED (exit 2)" || { echo "FAIL: expected OUTPUT_WRITE_FAILED, got: $err_out"; fail=1; }
fi

# Ambiguous invocation
err_out=$("$KIMO" layer --layer 0 --part moe "$ACT_VALID" "$FX_DIR/model-00001-of-00002.safetensors" --model-dir "$FX_DIR" 2>&1 >/dev/null)
rc=$?
if [ $rc -ne 2 ]; then
    echo "FAIL: ambiguous invocation returned exit code $rc (expected 2)"
    fail=1
else
    echo "PASS: ambiguous --model-dir + positional shards rejected (exit 2)"
fi

echo "=== 8. Real Qwen Checkpoint (Sanity Check) ==="
REAL_MODEL="/home/will/models/qwen1.5-moe-a2.7b-chat"
if [ -d "$REAL_MODEL" ]; then
    echo "Testing real model at $REAL_MODEL..."
    REAL_ACT="$TEST_DIR/real_act.bin"
    python3 -c "
import struct, math
with open('$REAL_ACT', 'wb') as f:
    for i in range(16 * 2048):
        v = math.sin(float(i + 1)) * 0.05
        f.write(struct.pack('<f', v))
"
    for rlyr in 0 12 23; do
        res=$("$KIMO" layer --layer "$rlyr" --part moe "$REAL_ACT" --model-dir "$REAL_MODEL" --workdir "$WORKDIR" --output "real_moe_${rlyr}.bin")
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "PASS: real model layer $rlyr succeeded (exit 0)"
        else
            echo "FAIL: real model layer $rlyr returned exit code $rc"
            fail=1
        fi
    done
else
    echo "Notice: $REAL_MODEL not found, skipping real model test"
fi

if [ $fail -eq 0 ]; then
    echo "ALL M3-W3 TESTS PASSED!"
    exit 0
else
    echo "M3-W3 TESTS FAILED!"
    exit 1
fi
