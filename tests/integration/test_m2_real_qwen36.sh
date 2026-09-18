#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Integration Test Suite untuk Milestone M2 (Attention Layer Qwen3.6-35B-A3B).
# Menguji probe layers:
#   - Layer 3  (Early attention)
#   - Layer 23 (Middle attention)
#   - Layer 39 (Late attention)
# Arsitektur Attention Qwen3.6:
#   - GQA (16 Q heads, 2 KV heads, head_dim=256)
#   - QK-Norm per-head (q_norm [256], k_norm [256])
#   - Partial RoPE (factor 0.25 = 64 dims rotary, 192 pass-through, theta=1e7)
#   - Output Sigmoid Gating (dari q_proj split Q & Gate)
#   - 0 bias tensors (attention_bias: false)
# Gate G-M2-1: delta_max <= 1e-3, epsilon_rel <= 1e-4, agreement == 100%.

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

DISMOEN_TOOLS="${DISMOEN_TOOLS:-$ROOT_DIR/target/debug/dismoen-tools}"
if [[ ! -x "$DISMOEN_TOOLS" && -x "$ROOT_DIR/target/release/dismoen-tools" ]]; then
    DISMOEN_TOOLS="$ROOT_DIR/target/release/dismoen-tools"
fi

if [[ ! -x "$DISMOEN_TOOLS" ]]; then
    echo "Membangun binary dismoen-tools..."
    cargo build --manifest-path tools/dismoen-tools/Cargo.toml --bin dismoen-tools
    DISMOEN_TOOLS="$ROOT_DIR/target/debug/dismoen-tools"
fi

MODEL_DIR="${MODEL_DIR:-/home/will/models/qwen3.6-35b-a3b}"
if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) tidak ditemukan. Melewati uji real attention."
    exit 0
fi

ACT_BIN="$ROOT_DIR/fixtures/m2/activation.bin"
if [[ ! -f "$ACT_BIN" ]]; then
    echo "FAIL: Activation fixture ($ACT_BIN) tidak ditemukan."
    exit 1
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if command -v uv >/dev/null 2>&1; then
        PYTHON="uv run python"
    else
        PYTHON="python3"
    fi
fi

TEST_DIR="/tmp/test_m2_real_qwen36_$$"
mkdir -p "$TEST_DIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M2: Attention Layer Execution & Gate G-M2-1 on Real Model"
echo "Model: Qwen3.6-35B-A3B ($MODEL_DIR)"
echo "Probe Layers: Layer 3 (early), Layer 23 (middle), Layer 39 (late)"
echo "======================================================================"

PROBE_LAYERS=(3 23 39)

for lyr in "${PROBE_LAYERS[@]}"; do
    echo ""
    echo "--> [Layer $lyr] Menghitung output PyTorch Oracle..."
    ORACLE_BIN="$TEST_DIR/oracle_attn_${lyr}.bin"
    $PYTHON tools/oracle/oracle_layer.py \
        --part attn \
        --layer "$lyr" \
        --activation "$ACT_BIN" \
        --model-dir "$MODEL_DIR" \
        --output "$ORACLE_BIN" > /dev/null

    echo "--> [Layer $lyr] Menjalankan Dismoen Mojo engine..."
    MOJO_FILE="mojo_attn_${lyr}.bin"
    MOJO_BIN="$TEST_DIR/$MOJO_FILE"
    REPORT=$("$DISMOEN" layer \
        --layer "$lyr" \
        "$ACT_BIN" \
        --model-dir "$MODEL_DIR" \
        --workdir "$TEST_DIR" \
        --output "$MOJO_FILE")
    echo "    $REPORT"

    echo "--> [Layer $lyr] Memverifikasi Gate G-M2-1 via dismoen-tools compare..."
    COMP_JSON="$TEST_DIR/comp_${lyr}.json"
    "$DISMOEN_TOOLS" compare "$ORACLE_BIN" "$MOJO_BIN" --gate G-M2-1 > "$COMP_JSON"

    python3 -c "
import json
with open('$COMP_JSON') as f:
    r = json.load(f)
assert r['status'] == 'MATCH', f'Layer $lyr status={r[\"status\"]}'
assert r['verdict'] == 'PASS', f'Layer $lyr verdict={r[\"verdict\"]}'
m = r['metrics']
assert m['delta_max'] <= 1e-3, f'Layer $lyr delta_max={m[\"delta_max\"]} > 1e-3'
assert m['epsilon_rel'] <= 1e-4, f'Layer $lyr epsilon_rel={m[\"epsilon_rel\"]} > 1e-4'
assert m['agreement'] == 100.0, f'Layer $lyr agreement={m[\"agreement\"]}'
print(f'    PASS: Layer $lyr Gate G-M2-1 MATCH (delta_max={m[\"delta_max\"]:.2e}, eps_rel={m[\"epsilon_rel\"]:.2e}, cos_theta={m[\"cos_theta\"]:.10f})')
"
done

echo ""
echo "======================================================================"
echo "Verdict: GATE G-M2-1 PASS (Semua probe layer 3, 23, 39 terverifikasi 100%)"
echo "======================================================================"
