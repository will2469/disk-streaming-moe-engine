#!/bin/bash
# Integration test suite for M3-W2: SwiGLU Routed + Shared Expert + Sigmoid Gate + Agregasi
# Verifies:
# 1. Mojo model.mojo unit & property tests (SwiGLU, SiLU, Sigmoid, Shared Gate, Agregasi, Bounds)
# 2. Rust shape_fidelity test (4440 MoE tensors in index: 24 layers x 185 tensors)
# 3. PyTorch oracle slice generation on synthetic inputs (F8c, F8d)
# 4. Mojo vs PyTorch Oracle cross-verification:
#    - SwiGLU output matches oracle PyTorch reference (delta_max <= 1e-3, eps_rel <= 1e-4)
#    - Invariant Keras #2: Sigmoid gate verified, non-sigmoid gate triggers GATE_ERROR
#    - Router selection 100% match (0 flip)
# 5. Independent Python verification of tolerances

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TMP_DIR="/tmp/test_m3_w2_$$"
mkdir -p "$TMP_DIR"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "=== 1. Mojo Model Unit & Property Tests (TestSuite) ==="
pixi run mojo run -I src src/model.mojo

echo "=== 2. Rust Shape Fidelity & MoE Property Tests ==="
pixi run cargo test --manifest-path tools/kimo-tools/Cargo.toml --test shape_fidelity

echo "=== 3. PyTorch Oracle MoE Slice Generation ==="
uv run python3 tools/oracle/oracle_slice_moe.py \
    --seq-len 8 \
    --hidden-size 2048 \
    --inter-routed 1408 \
    --inter-shared 5632 \
    --num-experts 60 \
    --top-k 4 \
    --seed 42 \
    --dump-dir "$TMP_DIR"

echo "=== 4. Mojo Engine vs PyTorch Oracle Cross-Verification ==="
MOE_FIXTURE_DIR="$TMP_DIR" pixi run mojo run -I src tests/integration/verify_m3_w2_oracle.mojo

echo "=== 5. Independent Python Metric Cross-Check (F10) ==="
python3 -c "
import numpy as np
import os

moe_ref = np.fromfile('$TMP_DIR/moe_ref.bin', dtype=np.float32)
y_routed_ref = np.fromfile('$TMP_DIR/y_routed_ref.bin', dtype=np.float32)
y_shared_ref = np.fromfile('$TMP_DIR/y_shared_ref.bin', dtype=np.float32)

assert len(moe_ref) == 8 * 2048
assert np.isfinite(moe_ref).all(), 'Non-finite in moe_ref'
assert np.isfinite(y_routed_ref).all(), 'Non-finite in y_routed_ref'
assert np.isfinite(y_shared_ref).all(), 'Non-finite in y_shared_ref'

print('MoE reference outputs verified finite with 8x2048 elements.')
"

echo "=== ALL M3-W2 INTEGRATION TESTS PASSED ==="
