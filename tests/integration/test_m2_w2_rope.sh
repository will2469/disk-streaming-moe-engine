#!/bin/bash
# Integration test suite for M2-W2: RoPE rotate_half F7 + Invariant Isometry (P-3)
# Verifies:
# 1. Mojo model.mojo unit & property tests (RoPE rotate_half, isometry invariant, linearity, oracle slice)
# 2. PyTorch oracle slice test (Property P-3, rotate_half vs interleaved distinction)
# 3. Binary fixture generation and isometry cross-check

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TMP_DIR="/tmp/test_m2_w2_$$"
mkdir -p "$TMP_DIR"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "=== 1. Mojo Model RoPE Unit & Property Tests (TestSuite) ==="
pixi run mojo run -I src src/model.mojo

echo "=== 2. PyTorch Oracle Slice Property Tests (Property P-3) ==="
uv run python3 tools/oracle/oracle_slice_rope.py --test

echo "=== 3. PyTorch Oracle Binary Generation & Validation ==="
uv run python3 tools/oracle/oracle_slice_rope.py \
    --seq-len 16 \
    --num-heads 16 \
    --head-dim 128 \
    --base 1000000.0 \
    --dump-dir "$TMP_DIR"

python3 -c "
import numpy as np
import os

rope_in = np.fromfile('$TMP_DIR/rope_in.bin', dtype=np.float32)
rope_out = np.fromfile('$TMP_DIR/rope_out_ref.bin', dtype=np.float32)

assert len(rope_in) == 16 * 16 * 128, f'Expected 32768 elements, got {len(rope_in)}'
assert len(rope_out) == 16 * 16 * 128
assert np.all(np.isfinite(rope_out)), 'Non-finite values found in RoPE ref'

# Per-head isometry check
x_in = rope_in.reshape(16, 16, 128)
x_out = rope_out.reshape(16, 16, 128)
for s in range(16):
    for h in range(16):
        n_in = np.linalg.norm(x_in[s, h])
        n_out = np.linalg.norm(x_out[s, h])
        diff = abs(n_in - n_out)
        assert diff < 1e-4 * max(1.0, n_in), f'Isometry violation at s={s}, h={h}: {n_in} vs {n_out}'

print('PyTorch RoPE fixtures verified with strict isometry: OK')
"

echo "=== All M2-W2 RoPE rotate_half tests PASSED! ==="
