#!/bin/bash
# E2E test suite for M5-W5: Perf 30-Run Baseline + F16 Core Scaling + Bandwidth Floor
#
# Covers:
# 1. Code hygiene & formatting:
#    - mojo format verification
#    - ruff check & format verification on tools/bench/*.py
#    - Spec hygiene: no-absolute-cores-in-docs
#    - Attribute hygiene: no allow/noqa suppressions
# 2. Gate G-M5-6: Floor bandwidth RAM >= 10.0 GB/s (STREAM Copy single-thread)
# 3. Benchmark runner execution:
#    - N=30 measurement runs + 2 warmup under cgroup MemoryMax=6G
#    - Context 4K verification (Gate G-M5-3: VmHWM <= 4.50 GiB)
#    - F16 core scaling sweep & Amdahl fit (Gate G-M5-5: e_{T,core} <= 20%, monotonic non-regression, knee c*)
#    - F5 calibration pipeline v0 -> v1 frozen (Gate G-M5-4: e_T <= 30%)
#    - F2 KV cache size calibration (Gate G-M5-2: e_KV <= 5%)
# 4. JSON schema and gate scorecard validation

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "============================================================"
echo "M5-W5: Performance, Core Scaling & Bandwidth Verification"
echo "============================================================"

# Step 1: Format and hygiene check
echo "== 1. Checking formatting and hygiene =="

# Mojo format check
MOJO_FMT=$(pixi run mojo format src/cli/cmd_decode.mojo 2>&1)
if echo "$MOJO_FMT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified src/cli/cmd_decode.mojo:"
    echo "$MOJO_FMT"
    exit 1
fi

# Python ruff check
uvx ruff@0.8.4 check tools/bench/bench_bw_stream.py tools/bench/bench_m5_real.py
uvx ruff@0.8.4 format --check tools/bench/bench_bw_stream.py tools/bench/bench_m5_real.py

# Spec hygiene: no absolute core counts in docs/
if grep -rniE "(^|[[:space:]])[0-9]+[- ]cores?\b" docs/ 2>/dev/null; then
    echo "FAIL: absolute core count detected in docs/"
    exit 1
fi

# No noqa comments
if grep -rn "noqa" tools/bench/bench_bw_stream.py tools/bench/bench_m5_real.py 2>/dev/null; then
    echo "FAIL: noqa suppression detected"
    exit 1
fi

echo "PASS: Formatting and hygiene checks passed cleanly."

# Step 2: Gate G-M5-6 RAM Bandwidth Floor
echo "== 2. Running STREAM Copy memory bandwidth benchmark (Gate G-M5-6) =="
STREAM_JSON="reports/2026-09-17/m5_stream_bw.json"
uv run --python .venv python tools/bench/bench_bw_stream.py --output-json "$STREAM_JSON"

python3 -c "
import json
with open('$STREAM_JSON') as f:
    d = json.load(f)

assert d['gate'] == 'G-M5-6'
assert d['verdict'] == 'PASS', f'G-M5-6 failed: {d[\"verdict\"]}'
med_bw = d['measured_read_equiv_bw_gb_s']['median']
assert med_bw >= 10.0, f'RAM bandwidth {med_bw:.2f} GB/s < 10.0 GB/s floor'
assert d['parameters']['llc_multiple'] >= 4.0, 'Array must be >= 4x LLC'
print(f'PASS: Gate G-M5-6 verified with sustained bandwidth {med_bw:.2f} GB/s >= 10.0 GB/s.')
"

# Step 3: Run M5 Benchmark Baseline (30 runs + 2 warmup)
echo "== 3. Running M5 decode benchmark suite (N=30 + 2 warmup) =="
BENCH_JSON="reports/2026-09-17/m5_benchmark_raw.json"
uv run --python .venv python tools/bench/bench_m5_real.py \
    --runs 30 \
    --warmup 2 \
    --max-tokens 64 \
    --context-size 2048 \
    --output-json "$BENCH_JSON"

# Step 4: Validate gates scorecard and calibration invariants
echo "== 4. Validating M5 benchmark gates scorecard and calibration invariants =="
python3 -c "
import json

with open('$BENCH_JSON') as f:
    data = json.load(f)

assert data['benchmark'] == 'M5 KV Cache & Autoregressive Decode'
scorecard = data['gate_scorecard']

# Gate G-M5-2: F2 KV cache size
assert scorecard['G-M5-2'] == 'PASS', f'Gate G-M5-2 failed: {scorecard}'
e_kv = data['calibration_f5_f2']['calibration_errors']['e_KV_cache_error']
assert e_kv <= 0.05, f'e_KV {e_kv:.4f} > 0.05'

# Gate G-M5-3: Memory @4K ctx <= 4.50 GiB and 0 OOM kills
assert scorecard['G-M5-3'] == 'PASS', f'Gate G-M5-3 failed: {scorecard}'
vmhwm_4k = data['context_4k_test']['vmhwm_gib']
assert vmhwm_4k <= 4.50, f'VmHWM @4K {vmhwm_4k:.2f} GiB > 4.50 GiB'
assert data['summary_n30']['cgroup_oom_kills_total'] == 0, 'OOM kills detected'

# Gate G-M5-4: F5 latency calibration e_T <= 30% against v1 frozen
assert scorecard['G-M5-4'] == 'PASS', f'Gate G-M5-4 failed: {scorecard}'
e_t = data['calibration_f5_f2']['calibration_errors']['e_T_latency_error']
assert e_t <= 0.30, f'e_T {e_t:.4f} > 0.30'
v1_pred = data['calibration_f5_f2']['v1_prediction_frozen']
assert v1_pred['label'] == 'v1_frozen_calibrated'

# Gate G-M5-5: F16 core scaling curve consistency e_{T,core} <= 20% & non-regression
assert scorecard['G-M5-5'] == 'PASS', f'Gate G-M5-5 failed: {scorecard}'
f16 = data['f16_core_scaling']['fit_parameters']
assert f16['f16_consistency_pass'], f'F16 consistency failed: max e_T_core {f16[\"max_e_t_core\"]:.4f} > 0.20'
assert data['f16_core_scaling']['monotonic_pass'], 'Monotonic non-regression failed'
assert data['f16_core_scaling']['s_tok_pass'], 'S_tok >= 1 non-regression failed'
assert f16['knee_operating_point_c_star'] >= 1
assert 0.0 <= f16['safe_ratio_r_star'] <= 1.0

# Gate G-M5-6: Bandwidth floor
assert scorecard['G-M5-6'] == 'PASS', f'Gate G-M5-6 failed: {scorecard}'
assert scorecard['overall_status'] == 'PASS', 'Overall benchmark status is not PASS'

print('PASS: All M5 gates (G-M5-2, G-M5-3, G-M5-4, G-M5-5, G-M5-6) verified!')
"

echo "============================================================"
echo "M5-W5 VERIFICATION COMPLETE: ALL GATES PASSED (GREEN)"
echo "============================================================"
