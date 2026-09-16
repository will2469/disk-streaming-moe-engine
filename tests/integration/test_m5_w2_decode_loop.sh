#!/bin/bash
# E2E test suite for M5-W2: Decode Loop (Posisi + Recompute + Numerik + F3b/F5)
#
# Covers:
# 1. Formatting compliance: mojo format check
# 2. Positional contract invariants & assertions (p = S + i, no t-1 label)
# 3. Incremental decode vs full recompute equivalence (mha_decode_step)
# 4. Positional semantic keys (RoPE at p, causal mask [0, p], 16x128 heads, M4 residuals)
# 5. Layer localization harness (S=4, N=4 per layer per position tensor comparison)
# 6. Strict rejection of hard-fail categories (router-selection, rope-style, bias-placement)
# 7. Locked F3b traffic subscript formulas and F5 v1 latency model verification
# 8. Regression check against M5-W1 kv_cache, attention, and forward unit tests

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "============================================================"
echo "M5-W2: Decode Loop Integration Verification"
echo "============================================================"

# Step 1: Format check
echo "== 1. Checking Mojo formatting =="
FORMAT_OUTPUT=$(pixi run mojo format \
    src/core/f3b_f5.mojo \
    src/core/__init__.mojo \
    src/layers/mha.mojo \
    src/layers/forward_layer.mojo \
    src/layers/decode_loop.mojo \
    src/layers/__init__.mojo \
    tests/unit/test_m5_decode_loop.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi
echo "PASS: Formatting is clean."

# Step 2: Run M5-W2 unit tests
echo "== 2. Running M5-W2 decode loop unit test suite =="
pixi run mojo run -I src tests/unit/test_m5_decode_loop.mojo
echo "PASS: All 6 M5-W2 unit tests passed."

# Step 3: Regression test on KV cache
echo "== 3. Running regression check on KV cache (M5-W1) =="
pixi run mojo run -I src tests/unit/test_m5_kv_cache.mojo
echo "PASS: M5-W1 KV cache unit tests passed without regression."

# Step 4: Regression test on attention
echo "== 4. Running regression check on attention =="
pixi run mojo run -I src tests/unit/test_attention.mojo
echo "PASS: Attention unit tests passed without regression."

# Step 5: Regression test on forward layer
echo "== 5. Running regression check on forward layer =="
pixi run mojo run -I src tests/unit/test_m4_forward_layer.mojo
echo "PASS: Forward layer unit tests passed without regression."

# Step 6: Python pytest tests
echo "== 6. Running Python test suite =="
uv run --python 3.12 --with pytest pytest -q
echo "PASS: Python tests passed."

echo "============================================================"
echo "M5-W2 VERIFICATION COMPLETE: ALL CHECKS PASSED (GREEN)"
echo "============================================================"
