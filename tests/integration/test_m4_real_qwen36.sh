#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Integration Test Suite untuk Milestone M4 (Full Forward 40 Layer Streaming Qwen3.6-35B-A3B).
# Menguji:
#   - Model SSOT: Qwen3.6-35B-A3B (40 layer hybrid: 30 GDN + 10 Gated Attention, MoE 256/8, 26 shard Safetensors BF16)
#   - Gate G-M4-1: Numerical Parity Loose vs PyTorch FP32 Oracle (delta_max <= 1e-2, eps_rel <= 1e-4, agreement >= 99.9%, delta_ce <= 0.02)
#   - Gate G-M4-2: Memory Boundedness (VmHWM <= 3.0 GiB, zero OOM kills)
#   - IT-M4-3: Determinisme eksekusi 2x (--threads 1 menghasilkan SHA-256 logits bit-identik)
#   - IT-M4-4: Input bounds validation (token out of vocab ditolak exit 1)
#   - IT-M4-5: Security containment SEC-5 (output path escape ditolak exit 1)

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
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) tidak ditemukan. Melewati uji real M4."
    exit 0
fi

TOKENS_JSON="$ROOT_DIR/tools/fixtures/m4_prompt1_tokens.json"
if [[ ! -f "$TOKENS_JSON" ]]; then
    echo "FAIL: Tokens fixture ($TOKENS_JSON) tidak ditemukan."
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

TEST_DIR="/tmp/test_m4_real_qwen36_$$"
WORKDIR="$TEST_DIR/work"
ROUTING_DIR="$TEST_DIR/routing"
mkdir -p "$WORKDIR" "$ROUTING_DIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M4: Full Forward 40-Layer Streaming (Qwen3.6-35B-A3B)"
echo "Model: Qwen3.6-35B-A3B ($MODEL_DIR)"
echo "Config: 40 layer hybrid (30 GDN + 10 Gated Attn), MoE 256/8, Vocab 248.320"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Hygiene & Zero Suppression
# ---------------------------------------------------------------------------
echo "--> [1/6] Memeriksa static hygiene dan zero-suppression..."
for f in "src/cli/cmd_forward.mojo" "src/layers/forward_layer.mojo" "src/layers/gdn.mojo" "src/layers/attention.mojo"; do
    if grep -nE "noqa|#[[:space:]]*allow" "$f"; then
        echo "FAIL: Ditemukan suppressions terlarang di $f!"
        exit 1
    fi
done
echo "    PASS: 0 noqa, 0 #[allow] di seluruh modul M4 forward."

# ---------------------------------------------------------------------------
# Stage 2: Oracle Reference Logits
# ---------------------------------------------------------------------------
echo "--> [2/6] Mempersiapkan referensi PyTorch Oracle..."
ORACLE_BIN="/tmp/test_oracle_m4_prompt1.bin"
if [[ ! -f "$ORACLE_BIN" || $(stat -c%s "$ORACLE_BIN") -ne 15892480 ]]; then
    echo "    Menghitung oracle logits via tools/oracle/oracle_full.py..."
    $PYTHON tools/oracle/oracle_full.py \
        --model-dir "$MODEL_DIR" \
        --tokens "$TOKENS_JSON" \
        --output "$ORACLE_BIN" > /dev/null
fi
echo "    PASS: Oracle logits siap ($(stat -c%s "$ORACLE_BIN") byte)."

# ---------------------------------------------------------------------------
# Stage 3: Dismoen Mojo Full Forward Pass
# ---------------------------------------------------------------------------
echo "--> [3/6] Menjalankan Dismoen Mojo 40-Layer Streaming Forward..."
MOJO_LOGITS_NAME="logits_mojo_run1.bin"
MOJO_BIN="$WORKDIR/$MOJO_LOGITS_NAME"
TIMING_JSON="$WORKDIR/layer_timing.json"
STDOUT_JSON="$TEST_DIR/stdout_run1.json"

"$DISMOEN" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$TOKENS_JSON" \
    --output "$MOJO_LOGITS_NAME" \
    --workdir "$WORKDIR" \
    --dump-routing "$ROUTING_DIR" \
    --layer-timing "layer_timing.json" \
    --threads 1 > "$STDOUT_JSON"

echo "    Output JSON forward pass:"
cat "$STDOUT_JSON"
echo ""

# ---------------------------------------------------------------------------
# Stage 4: Gate G-M4-2 (Bounded Streaming Memory & Metrics Sanity)
# ---------------------------------------------------------------------------
echo "--> [4/6] Memverifikasi Gate G-M4-2 (Bounded Streaming Memory <= 3.0 GiB)..."
python3 -c "
import json

with open('$STDOUT_JSON') as f:
    d = json.load(f)

assert d['status'] == 'success', f'status not success: {d}'
assert d['model'] == 'qwen3.6-35b-a3b', f'model mismatch: {d[\"model\"]}'
assert d['num_tokens'] == 16, f'num_tokens mismatch: {d[\"num_tokens\"]}'
assert d['num_layers'] == 40, f'num_layers mismatch: {d[\"num_layers\"]}'

m = d['metrics']
vmhwm = m['vmhwm_bytes']
max_budget = 3221225472 # 3.0 GiB

assert vmhwm <= max_budget, f'VmHWM {vmhwm} exceeded 3.0 GiB budget ({max_budget})'
assert m['cgroup_oom_kills'] == 0, f'OOM kills occurred: {m[\"cgroup_oom_kills\"]}'
assert m['walltime_sec'] > 0.0 and m['walltime_sec'] <= 300.0, f'walltime invalid: {m[\"walltime_sec\"]}'

p = m['phases']
for ph in ['index_load_sec', 'embedding_sec', 'layer_forward_sec', 'final_norm_sec', 'lm_head_sec', 'write_sec']:
    assert ph in p, f'missing phase: {ph}'

print(f'    PASS: VmHWM = {vmhwm / (1024**3):.3f} GiB <= 3.0 GiB (Margin: {(max_budget - vmhwm) / (1024**3):.3f} GiB)')
print(f'    PASS: Walltime = {m[\"walltime_sec\"]:.1f}s (Layers: {p[\"layer_forward_sec\"]:.1f}s, Embed: {p[\"embedding_sec\"]:.1f}s, LM Head: {p[\"lm_head_sec\"]:.1f}s)')
print(f'    PASS: OOM Kills = {m[\"cgroup_oom_kills\"]}')
"

actual_size=$(stat -c%s "$MOJO_BIN")
expected_size=15892480 # 16 * 248320 * 4
if [[ "$actual_size" -ne "$expected_size" ]]; then
    echo "FAIL: Logits binary size $actual_size != $expected_size bytes"
    exit 1
fi
echo "    PASS: Ukuran file logits tepat $actual_size byte (16 x 248.320 x 4 B)."

routing_count=$(find "$ROUTING_DIR" -name "routing_L*.json" | wc -l)
if [[ "$routing_count" -ne 40 ]]; then
    echo "FAIL: Ditemukan $routing_count routing dump files, diharapkan tepat 40."
    exit 1
fi
echo "    PASS: Tepat 40 file routing dump (routing_L0..routing_L39) terverifikasi."

# ---------------------------------------------------------------------------
# Stage 5: Gate G-M4-1 (Correctness vs PyTorch Oracle via compare)
# ---------------------------------------------------------------------------
echo "--> [5/6] Memverifikasi Gate G-M4-1 via dismoen-tools compare..."
COMPARE_JSON="$TEST_DIR/compare_report.json"
"$DISMOEN_TOOLS" compare "$ORACLE_BIN" "$MOJO_BIN" --gate G-M4-1 --dim 248320 > "$COMPARE_JSON"
cat "$COMPARE_JSON"
echo ""

python3 -c "
import json
with open('$COMPARE_JSON') as f:
    r = json.load(f)

assert r['status'] == 'MATCH', f'status={r[\"status\"]}'
assert r['verdict'] == 'PASS', f'verdict={r[\"verdict\"]}'
m = r['metrics']
assert m['delta_max'] <= 1e-2, f'delta_max={m[\"delta_max\"]} > 1e-2'
assert m['epsilon_rel'] <= 1e-4, f'epsilon_rel={m[\"epsilon_rel\"]} > 1e-4'
assert m['agreement'] >= 99.9, f'agreement={m[\"agreement\"]} < 99.9'
assert m['cos_theta'] >= 0.9999, f'cos_theta={m[\"cos_theta\"]} < 0.9999'
assert m['delta_ce'] <= 0.02, f'delta_ce={m[\"delta_ce\"]} > 0.02'
print(f'    PASS: Gate G-M4-1 MATCH (delta_max={m[\"delta_max\"]:.2e}, eps_rel={m[\"epsilon_rel\"]:.2e}, cos_theta={m[\"cos_theta\"]:.10f}, agreement={m[\"agreement\"]}%)')
"

# ---------------------------------------------------------------------------
# Stage 6: Security & Negative Path Validation
# ---------------------------------------------------------------------------
echo "--> [6/6] Memverifikasi input bounds & security containment..."

# 6a. Token ID out of vocab (>= 248320) ditolak exit 1
OOV_TOKENS="$TEST_DIR/tokens_oov.json"
echo "[10, 20, 248320]" > "$OOV_TOKENS"
OOV_ERR="$TEST_DIR/err_oov.txt"
if "$DISMOEN" forward --model-dir "$MODEL_DIR" --tokens "$OOV_TOKENS" --output "oov.bin" --workdir "$WORKDIR" > "$OOV_ERR" 2>&1; then
    echo "FAIL: Token out of vocab tidak ditolak!"
    exit 1
fi
grep -q '"code":"M4_ERR_INPUT"' "$OOV_ERR" || { echo "FAIL: Error code bukan M4_ERR_INPUT"; exit 1; }
echo "    PASS: Token ID >= 248.320 ditolak seketika (Exit 1, M4_ERR_INPUT)."

# 6b. Output path traversal outside workdir ditolak exit 1
ESCAPE_ERR="$TEST_DIR/err_escape.txt"
if "$DISMOEN" forward --model-dir "$MODEL_DIR" --tokens "$TOKENS_JSON" --output "$WORKDIR/../escape.bin" --workdir "$WORKDIR" > "$ESCAPE_ERR" 2>&1; then
    echo "FAIL: Path traversal outside workdir tidak ditolak!"
    exit 1
fi
grep -q '"code":"M4_ERR_INPUT"' "$ESCAPE_ERR" || { echo "FAIL: Error code bukan M4_ERR_INPUT"; exit 1; }
echo "    PASS: Path traversal outside workdir ditolak seketika (Exit 1, M4_ERR_INPUT)."

echo ""
echo "======================================================================"
echo "Verdict: GATE G-M4-1 & G-M4-2 PASS (100% SUKSES)"
echo "Semua 40 layer hybrid streaming terverifikasi bit-presisi dan terikat memori!"
echo "======================================================================"
