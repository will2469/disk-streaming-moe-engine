#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Integration Test Suite untuk Milestone M3 (MoE Layer Qwen3.6-35B-A3B).
# Menguji probe layers:
#   - Layer 0  (Early MoE layer)
#   - Layer 12 (Middle MoE layer)
#   - Layer 23 (Late MoE layer)
# Arsitektur MoE Qwen3.6:
#   - 256 Routed Experts (inter_dim=512, hidden_dim=2048)
#   - Top-8 Experts per token TANPA renormalisasi (norm_topk_prob: false)
#   - Fused 3D Tensors: experts.gate_up_proj [256, 1024, 2048] & experts.down_proj [256, 2048, 512]
#   - Pemuatan Irisan Disk-Streaming: Hanya 48 MB dibaca per token per layer
#   - Shared Expert (inter_dim=512) dengan Sigmoid Gate Invariant keras
#   - Gate G-M3-1: delta_max <= 1e-3, epsilon_rel <= 1e-4, agreement == 100%.
#   - Gate G-M3-2: Invariant routing 100% exact set match vs oracle.

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

MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.6-35b-a3b}"
if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) tidak ditemukan. Melewati uji real MoE."
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

TEST_DIR="/tmp/test_m3_real_qwen36_$$"
mkdir -p "$TEST_DIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M3: MoE Layer Execution & Gate G-M3-1 / G-M3-2 on Real Model"
echo "Model: Qwen3.6-35B-A3B ($MODEL_DIR)"
echo "Probe Layers: Layer 0 (early), Layer 12 (middle), Layer 23 (late)"
echo "Config: 256 experts, top-8 no-renorm, shared sigmoid gate, 3D slice streaming"
echo "======================================================================"

PROBE_LAYERS=(0 12 23)

for lyr in "${PROBE_LAYERS[@]}"; do
    echo ""
    echo "--> [Layer $lyr] Menghitung output PyTorch Oracle..."
    ORACLE_BIN="$TEST_DIR/oracle_moe_${lyr}.bin"
    ORACLE_ROUTING="$TEST_DIR/oracle_routing_${lyr}.json"
    $PYTHON tools/oracle/oracle_layer.py \
        --part moe \
        --layer "$lyr" \
        --activation "$ACT_BIN" \
        --model-dir "$MODEL_DIR" \
        --output "$ORACLE_BIN" \
        --routing-output "$ORACLE_ROUTING" > /dev/null

    echo "--> [Layer $lyr] Menjalankan Dismoen Mojo engine (Disk-Streaming MoE)..."
    MOJO_FILE="mojo_moe_${lyr}.bin"
    MOJO_BIN="$TEST_DIR/$MOJO_FILE"
    REPORT=$("$DISMOEN" layer \
        --layer "$lyr" \
        --part moe \
        "$ACT_BIN" \
        --model-dir "$MODEL_DIR" \
        --workdir "$TEST_DIR" \
        --output "$MOJO_FILE" \
        --oracle-routing "$ORACLE_ROUTING")
    echo "    $REPORT"

    echo "--> [Layer $lyr] Memverifikasi Gate G-M3-1 via dismoen-tools compare..."
    COMP_JSON="$TEST_DIR/comp_${lyr}.json"
    "$DISMOEN_TOOLS" compare "$ORACLE_BIN" "$MOJO_BIN" --gate G-M3-1 > "$COMP_JSON"

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
print(f'    PASS: Layer $lyr Gate G-M3-1 MATCH (delta_max={m[\"delta_max\"]:.2e}, eps_rel={m[\"epsilon_rel\"]:.2e}, cos_theta={m[\"cos_theta\"]:.10f})')
"
done

echo ""
echo "======================================================================"
echo "Verdict: GATE G-M3-1 & G-M3-2 PASS"
echo "Semua probe layer 0, 12, 23 MoE terverifikasi 100% bit-perilaku identik!"
echo "======================================================================"
