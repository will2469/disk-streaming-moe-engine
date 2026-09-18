#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M10 Wave 4a (Gate G-M10-3: Unified Forward Numerical Parity & Decode Continuation).
# Menguji:
#   1. Static formatting & zero suppression (0 noqa, 0 #[allow])
#   2. Paritas numerik dismoen forward terhadap reference logits M9 (delta_max <= 1e-7, Gate G-M10-3)
#   3. Forward prefill session generation (KMSS v1 serialization) & metric verification
#   4. Autoregressive decode continuation tanpa recompute historis (historical_recompute_tokens == 0, gdn_reused == true)
#   5. Chained decode continuation (multi-step session persistence)
#   6. Formal Quality Gate G-M10-3 Scorecard

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-$ROOT_DIR/dismoen}"
if [[ ! -x "$DISMOEN" && -x "$ROOT_DIR/kimo" ]]; then
    DISMOEN="$ROOT_DIR/kimo"
fi

if [[ ! -x "$DISMOEN" ]]; then
    echo "FAIL: Binary dismoen/kimo tidak ditemukan atau tidak executable. Jalankan 'pixi run build'."
    exit 1
fi

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

MINI_CONFIG="fixtures/m9_port_config_mini.json"
TOKENS_FIXTURE="fixtures/m9_port_tokens.json"
REF_LOGITS="fixtures/m9_port_logits_naive.bin"

TEST_DIR="/tmp/test_m10_w3_forward_decode_$$"
mkdir -p "$TEST_DIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M10-W4a: Gate G-M10-3 Unified Forward Parity & Decode Continuation"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Hygiene & Zero Suppression (§5 G-M10-4, G-M10-3)
# ---------------------------------------------------------------------------
echo "--> Stage 1: Static Hygiene & Zero Suppression"

for file in "src/cli/cmd_decode.mojo" "src/cli/cmd_forward.mojo" "tools/compare.py" "tools/kimo-tools/src/compare.rs"; do
    if grep -nE "noqa|#[[:space:]]*allow" "$file"; then
        echo "FAIL: Ditemukan suppressions terlarang di $file!"
        exit 1
    fi
done

echo "   PASS: 0 noqa, 0 #[allow] di seluruh modul M10 forward/decode."

# ---------------------------------------------------------------------------
# Stage 2: Gate G-M10-3 Numerical Parity (delta_max <= 1e-7)
# ---------------------------------------------------------------------------
echo "--> Stage 2: Gate G-M10-3 Forward Numerical Parity (delta_max <= 1e-7)"

CAND_LOGITS="${TEST_DIR}/cand_forward_logits.bin"

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --tokens "$TOKENS_FIXTURE" \
    --output "$CAND_LOGITS" > /dev/null

if [[ ! -s "$CAND_LOGITS" ]]; then
    echo "FAIL: Candidate logits tidak dihasilkan atau berukuran 0!"
    exit 1
fi

# Evaluasi via dismoen compare Rust engine
COMPARE_OUT=$("$DISMOEN" compare \
    --reference "$REF_LOGITS" \
    --candidate "$CAND_LOGITS" \
    --gate G-M10-3)

echo "$COMPARE_OUT"

echo "$COMPARE_OUT" | grep -q '"verdict": "PASS"' || {
    echo "FAIL: Gate G-M10-3 gagal memenuhi ambang batas delta_max <= 1e-7 via dismoen compare!"
    exit 1
}

# Cross-evaluasi via python compare oracle
PY_COMPARE_OUT=$("$PYTHON" tools/compare.py \
    --ref "$REF_LOGITS" \
    --cand "$CAND_LOGITS" \
    --gate G-M10-3 \
    --dim 1024)

echo "$PY_COMPARE_OUT" | grep -q '"verdict": "PASS"' || {
    echo "FAIL: Gate G-M10-3 gagal memenuhi ambang batas delta_max <= 1e-7 via python compare!"
    exit 1
}

echo "   PASS: Gate G-M10-3 Numerical Parity terverifikasi (delta_max <= 1e-7)."

# ---------------------------------------------------------------------------
# Stage 3: Forward Prefill Session Generation & Metric Verification
# ---------------------------------------------------------------------------
echo "--> Stage 3: Forward Prefill Session Generation & Metrics"

SESS1="${TEST_DIR}/session1.kmss"

FWD_OUT=$("$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --tokens "$TOKENS_FIXTURE" \
    --save-session "$SESS1")

if [[ ! -s "$SESS1" ]]; then
    echo "FAIL: Sesi KMSS tidak tersimpan di $SESS1!"
    exit 1
fi

echo "$FWD_OUT" | grep -q '"historical_recompute_tokens": 8' || {
    echo "FAIL: dismoen forward output missing historical_recompute_tokens: 8!"
    exit 1
}

echo "$FWD_OUT" | grep -q '"gdn_reused": false' || {
    echo "FAIL: dismoen forward prefill missing gdn_reused: false!"
    exit 1
}

echo "   PASS: Sesi prefill tersimpan valid dan metrik presisi ter-emit."

# ---------------------------------------------------------------------------
# Stage 4: Autoregressive Decode Continuation (G-M10-3 historical_recompute_tokens == 0)
# ---------------------------------------------------------------------------
echo "--> Stage 4: Autoregressive Decode Continuation (historical_recompute_tokens == 0)"

DEC_OUT_FILE="${TEST_DIR}/tokens_gen_1.json"

DEC_STDOUT=$("$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --session "$SESS1" \
    --max-tokens 4 \
    --output "$DEC_OUT_FILE")

echo "$DEC_STDOUT"

echo "$DEC_STDOUT" | grep -q '"historical_recompute_tokens": 0' || {
    echo "FAIL: Gate G-M10-3 FAIL: historical_recompute_tokens != 0!"
    exit 1
}

echo "$DEC_STDOUT" | grep -q '"recompute_tokens": 0' || {
    echo "FAIL: Gate G-M10-3 FAIL: backward-compatible recompute_tokens != 0!"
    exit 1
}

echo "$DEC_STDOUT" | grep -q '"gdn_reused": true' || {
    echo "FAIL: Gate G-M10-3 FAIL: gdn_reused != true!"
    exit 1
}

echo "$DEC_STDOUT" | grep -q '"gdn_state_reused": true' || {
    echo "FAIL: Gate G-M10-3 FAIL: gdn_state_reused != true!"
    exit 1
}

echo "$DEC_STDOUT" | grep -q '"prompt_tokens": 8' || {
    echo "FAIL: Gate G-M10-3 FAIL: prompt_tokens != 8!"
    exit 1
}

echo "$DEC_STDOUT" | grep -q '"generated_tokens": 4' || {
    echo "FAIL: Gate G-M10-3 FAIL: generated_tokens != 4!"
    exit 1
}

if [[ ! -s "$DEC_OUT_FILE" ]]; then
    echo "FAIL: File output token decode $DEC_OUT_FILE tidak ditemukan!"
    exit 1
fi

"$PYTHON" -c '
import json, sys
with open("'"$DEC_OUT_FILE"'") as f:
    data = json.load(f)
assert isinstance(data, list), "Output must be JSON list"
assert len(data) == 4, f"Expected 4 generated tokens, got {len(data)}"
for tok in data:
    assert isinstance(tok, int), f"Token {tok} must be int"
'

echo "   PASS: Gate G-M10-3 Autoregressive Continuation terverifikasi (recompute=0, gdn_reused=true, 4 tokens valid)."

# ---------------------------------------------------------------------------
# Stage 5: Chained Multi-Step Decode Continuation (KMSS session save & restore)
# ---------------------------------------------------------------------------
echo "--> Stage 5: Chained Multi-Step Decode Continuation"

SESS2="${TEST_DIR}/session2.kmss"
DEC_OUT_FILE2="${TEST_DIR}/tokens_gen_2.json"
DEC_OUT_FILE3="${TEST_DIR}/tokens_gen_3.json"

# Decode 4 token dari session 1 dan simpan ke session 2
"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --session "$SESS1" \
    --save-session "$SESS2" \
    --max-tokens 4 \
    --output "$DEC_OUT_FILE2" > /dev/null

if [[ ! -s "$SESS2" ]]; then
    echo "FAIL: Chained session tidak tersimpan di $SESS2!"
    exit 1
fi

# Decode 4 token berikutnya dari session 2 (total konteks harus 12 token)
CHAIN_STDOUT=$("$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --session "$SESS2" \
    --max-tokens 4 \
    --output "$DEC_OUT_FILE3")

echo "$CHAIN_STDOUT" | grep -q '"prompt_tokens": 12' || {
    echo "FAIL: Chained decode prompt_tokens != 12!"
    exit 1
}

echo "$CHAIN_STDOUT" | grep -q '"historical_recompute_tokens": 0' || {
    echo "FAIL: Chained decode historical_recompute_tokens != 0!"
    exit 1
}

echo "$CHAIN_STDOUT" | grep -q '"gdn_reused": true' || {
    echo "FAIL: Chained decode gdn_reused != true!"
    exit 1
}

echo "   PASS: Chained continuation dari KMSS session v1 sukses (konteks 12 token, 0 recompute)."

# ---------------------------------------------------------------------------
# Stage 6: Gate G-M10-3 Quality Gate Certification
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "GATE G-M10-3 CERTIFICATION SCORECARD: PASS"
echo "======================================================================"
echo "  [x] Static Code Hygiene              : 0 noqa, 0 #[allow]"
echo "  [x] Forward Numerical Parity         : delta_max <= 1e-7 (PASS)"
echo "  [x] Zero Historical Recomputation    : historical_recompute_tokens == 0"
echo "  [x] GDN State Reuse Invariant        : gdn_reused == true"
echo "  [x] KMSS v1 Autoregressive Session   : Chained decode continuation OK"
echo "======================================================================"
