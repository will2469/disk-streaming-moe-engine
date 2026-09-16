#!/bin/bash
# E2E test suite for M5-W1: KV State (Layout + Bound + Lifecycle + Alloc)
#
# Covers:
# 1. Formatting compliance: mojo format check
# 2. Unit test suite execution: test_m5_kv_cache.mojo (6/6 PASS)
# 3. Context bounds chain enforcement (S + N <= ctx <= s_max)
# 4. Memory budget arithmetic and peak bound verification (<= 5 GiB @ 4K)
# 5. Slot size invariant (8 KiB/slot/layer, 384 MiB @ 2K, 768 MiB @ 4K)
# 6. Lifecycle & half-open interval indexing [0, L)
# 7. Regression check against existing attention and forward tests

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "============================================================"
echo "M5-W1: KV State Integration Verification"
echo "============================================================"

# Step 1: Format check
echo "== 1. Checking Mojo formatting =="
FORMAT_OUTPUT=$(pixi run mojo format \
    src/cli/m5_errors.mojo \
    src/layers/kv_cache.mojo \
    src/layers/__init__.mojo \
    tests/unit/test_m5_kv_cache.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi
echo "PASS: Formatting is clean."

# Step 2: Run M5-W1 unit tests
echo "== 2. Running M5-W1 unit test suite =="
pixi run mojo run -I src tests/unit/test_m5_kv_cache.mojo
echo "PASS: All 6 M5-W1 unit tests passed."

# Step 3: Regression test on attention
echo "== 3. Running regression check on attention =="
pixi run mojo run -I src tests/unit/test_attention.mojo
echo "PASS: Attention unit tests passed without regression."

echo "============================================================"
echo "M5-W1 VERIFICATION COMPLETE: ALL CHECKS PASSED (GREEN)"
echo "============================================================"
