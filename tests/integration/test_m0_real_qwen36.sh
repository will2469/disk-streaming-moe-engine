#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Integration Test Suite untuk Milestone M0 (Gate G-M0-1: Reader Safetensors Multi-Shard Qwen3.6-35B-A3B).
# Menguji:
#   1. Parsing & indexing 26 shard Safetensors asli (71,90 GB, 1.045 tensor)
#   2. Full-scope match vs model.safetensors.index.json (1045/1045 tensor)
#   3. Subset-scope check (1 shard)
#   4. Ambang batas waktu parsing (parse_time_ms < 1.000 ms)

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
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) tidak ditemukan. Melewati uji real shard."
    exit 0
fi

echo "======================================================================"
echo "M0: Gate G-M0-1 Safetensors Multi-Shard Indexing (Qwen3.6-35B-A3B)"
echo "======================================================================"

# Stage 1: Full check-index 26 shards
echo "--> Stage 1: Full check-index (26 shard, 1.045 tensor)"
REPORT=$("$DISMOEN" check-index "$MODEL_DIR"/model-*.safetensors)
echo "$REPORT"

python3 -c "
import json, sys

data = json.loads('''$REPORT''')
assert data['status'] == 'match', f'Expected match, got {data[\"status\"]}'
assert data['scope'] == 'full', f'Expected full scope, got {data[\"scope\"]}'
assert data['total_tensors'] == 1045, f'Expected 1045 tensors, got {data[\"total_tensors\"]}'
assert data['assessed_tensors'] == 1045, f'Expected 1045 assessed, got {data[\"assessed_tensors\"]}'
assert data['matched_tensors'] == 1045, f'Expected 1045 matched, got {data[\"matched_tensors\"]}'
assert len(data['mismatches']) == 0, f'Expected 0 mismatches, got {data[\"mismatches\"]}'
assert data['parse_time_ms'] < 1000.0, f'Parse time exceeded 1000ms: {data[\"parse_time_ms\"]}ms'
print(f'   PASS: Gate G-M0-1 terpenuhi. 1045/1045 tensor cocok dalam {data[\"parse_time_ms\"]} ms.')
"

# Stage 2: Subset check-index (1 shard)
echo "--> Stage 2: Subset check-index (1 shard)"
SUBSET_REPORT=$("$DISMOEN" check-index "$MODEL_DIR"/model-00001-of-00026.safetensors)
python3 -c "
import json, sys

data = json.loads('''$SUBSET_REPORT''')
assert data['status'] == 'match', f'Expected match, got {data[\"status\"]}'
assert data['scope'] == 'subset', f'Expected subset scope, got {data[\"scope\"]}'
assert data['total_tensors'] == 1045, f'Expected 1045 total tensors, got {data[\"total_tensors\"]}'
assert data['assessed_tensors'] > 0 and data['assessed_tensors'] < 1045, f'Unexpected assessed count: {data[\"assessed_tensors\"]}'
assert data['matched_tensors'] == data['assessed_tensors'], 'Matched does not equal assessed'
assert len(data['mismatches']) == 0, f'Expected 0 mismatches, got {data[\"mismatches\"]}'
print(f'   PASS: Subset check valid ({data[\"matched_tensors\"]}/{data[\"assessed_tensors\"]} tensor di shard 1).')
"

echo "======================================================================"
echo "Verdict: GATE G-M0-1 PASS (Qwen3.6-35B-A3B SSOT)"
echo "======================================================================"
