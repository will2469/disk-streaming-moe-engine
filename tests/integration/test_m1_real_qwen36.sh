#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Integration Test Suite untuk Milestone M1 (Head Path Qwen3.6-35B-A3B).
# Menguji:
#   1. Resolusi bobot model.language_model.embed_tokens.weight, model.language_model.norm.weight, lm_head.weight
#   2. Eksekusi head path: embedding lookup -> final RMSNorm -> lm_head matmul
#   3. Verifikasi ukuran output logits (3 x 16 x 248320 x 4 = 47.677.440 byte)
#   4. Verifikasi telemetri memori M1-B (resident ~3,79 GiB, conversion buffer <= 64 MiB, VmHWM <= 5,0 GiB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-$ROOT_DIR/dismoen}"
if [[ ! -x "$DISMOEN" && -x "$ROOT_DIR/build/bin/dismoen" ]]; then
    DISMOEN="$ROOT_DIR/build/bin/dismoen"
fi

if [[ ! -x "$DISMOEN" ]]; then
    echo "FAIL: Binary dismoen tidak ditemukan atau tidak executable. Jalankan 'pixi run build'."
    exit 1
fi

MODEL_DIR="${MODEL_DIR:-/home/will/models/qwen3.6-35b-a3b}"

if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) tidak ditemukan. Melewati uji real head."
    exit 0
fi

TEST_DIR="/tmp/test_m1_real_qwen36_$$"
mkdir -p "$TEST_DIR"
LOGITS_FILE="qwen36_head_logits.bin"
CAND_LOGITS="${TEST_DIR}/${LOGITS_FILE}"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M1: Head Path Execution on Real Checkpoint (Qwen3.6-35B-A3B)"
echo "======================================================================"

echo "--> Stage 1: Run dismoen head (3 prompt x 16 token, vocab 248.320)"
REPORT=$("$DISMOEN" head "fixtures/m1/tokens.json" \
    --model-dir "$MODEL_DIR" \
    --output "$LOGITS_FILE" \
    --workdir "$TEST_DIR")

echo "$REPORT"

python3 -c "
import json, struct, os, sys

data = json.loads('''$REPORT''')
assert data['status'] == 'success', f'Status not success: {data[\"status\"]}'
assert data['num_prompts'] == 3, f'Expected 3 prompts, got {data[\"num_prompts\"]}'
assert data['tokens_per_prompt'] == 16, f'Expected 16 tokens/prompt, got {data[\"tokens_per_prompt\"]}'
assert data['num_tokens_total'] == 48, f'Expected 48 tokens total, got {data[\"num_tokens_total\"]}'
assert data['vocab_size'] == 248320, f'Expected vocab 248320, got {data[\"vocab_size\"]}'

mem = data['memory']
assert mem['resident_target_bytes'] == 4068483072, f'Unexpected resident bytes: {mem[\"resident_target_bytes\"]}'
assert mem['conversion_buffer_bytes'] <= 67108864, f'Conversion buffer exceeded 64MB: {mem[\"conversion_buffer_bytes\"]}'
assert mem['source_buffer_bytes'] <= 67108864, f'Source buffer exceeded 64MB: {mem[\"source_buffer_bytes\"]}'
assert mem['vmhwm_bytes'] <= 5368709120, f'VmHWM exceeded 5.0 GiB: {mem[\"vmhwm_bytes\"]}'

logits_path = '$CAND_LOGITS'
expected_bytes = 3 * 16 * 248320 * 4
actual_bytes = os.path.getsize(logits_path)
assert actual_bytes == expected_bytes, f'Expected {expected_bytes} bytes, got {actual_bytes}'

# Verifikasi sample nilai logits finite (bukan NaN atau Inf)
with open(logits_path, 'rb') as f:
    sample_bytes = f.read(100 * 4)
    floats = struct.unpack(f'<{len(sample_bytes)//4}f', sample_bytes)
    for v in floats:
        assert not (v != v or v == float('inf') or v == float('-inf')), 'Non-finite value in logits'

print(f'   PASS: Head path valid. Output: {actual_bytes} bytes, VmHWM: {mem[\"vmhwm_bytes\"] / (1024**3):.2f} GiB.')
"

echo "======================================================================"
echo "Verdict: GATE G-M1-1 / G-M1-2 PASS (Qwen3.6-35B-A3B SSOT)"
echo "======================================================================"
