#!/bin/bash
# Integration test suite for M3-W1: Router Softmax fp32 + Top-4 No-Renorm + Invariant SET
# Verifies:
# 1. Mojo model.mojo unit tests (RouterConfig, projection, stable softmax, top-4 no-renorm, weight load)
# 2. Rust shape_fidelity test (24 router gate weights in index, no bias, norm_topk_prob=false)
# 3. PyTorch oracle slice generation on 256 random inputs (F8a, F8b)
# 4. Mojo vs Oracle cross-verification:
#    - 100% SET matching on 256 tokens (0 flip)
#    - Numerical tolerance atol <= 1e-5 on unrenormalized probabilities
#    - Invariant sum(p_i) <= 1.0 (strictly < 1.0 per token)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TMP_DIR="/tmp/test_m3_w1_$$"
mkdir -p "$TMP_DIR"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "=== 1. Mojo Model Unit & Property Tests (TestSuite) ==="
pixi run mojo run -I src src/model.mojo

echo "=== 2. Rust Shape Fidelity & Property Tests ==="
pixi run cargo test --manifest-path tools/dismoen-tools/Cargo.toml --test shape_fidelity

echo "=== 3. PyTorch Oracle Slice Generation (256 random tokens) ==="
uv run python3 tools/oracle/oracle_slice_router.py \
    --seq-len 256 \
    --hidden-size 2048 \
    --num-experts 60 \
    --top-k 4 \
    --seed 42 \
    --dump-dir "$TMP_DIR"

echo "=== 4. Mojo Engine vs PyTorch Oracle Cross-Verification (0 Flip Assertion) ==="
ROUTER_FIXTURE_DIR="$TMP_DIR" pixi run mojo run -I src tests/integration/verify_m3_w1_oracle.mojo

echo "=== 5. Numerical Accuracy & Property Cross-Check (Python) ==="
python3 -c "
import numpy as np
import os

topk_indices = np.fromfile('$TMP_DIR/topk_indices_ref.bin', dtype=np.int32).reshape(256, 4)
topk_probs = np.fromfile('$TMP_DIR/topk_probs_ref.bin', dtype=np.float32).reshape(256, 4)

assert topk_indices.shape == (256, 4)
assert topk_probs.shape == (256, 4)
assert np.all((topk_indices >= 0) & (topk_indices < 60)), 'Expert indices out of [0, 59]'
assert np.all(np.isfinite(topk_probs)), 'Non-finite values found in topk probs'

# Property norm_topk_prob=false
sums = topk_probs.sum(axis=-1)
assert np.all(sums <= 1.0), 'Top-4 prob sum exceeds 1.0'
assert np.all(sums < 1.0), 'Top-4 prob sum equals 1.0 (renormalization leak)'

# Strictly sorted descending
for t in range(256):
    for k in range(3):
        assert topk_probs[t, k] >= topk_probs[t, k + 1], f'Probabilities not descending at token {t}'

print('Oracle reference fixtures verified: OK')
"

echo "=== All M3-W1 Router Softmax fp32 + Top-4 No-Renorm tests PASSED! ==="
