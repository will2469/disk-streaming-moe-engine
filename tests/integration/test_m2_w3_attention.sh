#!/bin/bash
# Integration test suite for M2-W3: Causal Mask + MHA + Softmax Stabil + o_proj + Residual
# Verifies:
# 1. Mojo model.mojo unit & property tests (causal mask triangular, autoregressive property, softmax max-shift, MHA oracle, full block)
# 2. PyTorch oracle slice property tests
# 3. Binary fixture generation and validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TMP_DIR="/tmp/test_m2_w3_$$"
mkdir -p "$TMP_DIR"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "=== 1. Mojo Model Attention Unit & Property Tests (TestSuite) ==="
pixi run mojo run -I src src/model.mojo

echo "=== 2. PyTorch Oracle Slice Property Tests ==="
uv run python3 tools/oracle/oracle_slice_attention.py --test

echo "=== 3. PyTorch Oracle Binary Generation & Validation ==="
uv run python3 tools/oracle/oracle_slice_attention.py \
    --seq-len 16 \
    --num-heads 16 \
    --head-dim 128 \
    --base 1000000.0 \
    --dump-dir "$TMP_DIR"

python3 -c "
import numpy as np
import os

attn_in = np.fromfile('$TMP_DIR/attn_in.bin', dtype=np.float32)
attn_ref = np.fromfile('$TMP_DIR/attn_ref.bin', dtype=np.float32)

assert len(attn_in) == 16 * 16 * 128, f'Expected 32768 elements, got {len(attn_in)}'
assert len(attn_ref) == 16 * 16 * 128
assert np.all(np.isfinite(attn_ref)), 'Non-finite values found in attention block ref'

print('PyTorch Attention Block fixtures verified: OK')
"

echo "=== All M2-W3 Attention Block tests PASSED! ==="
