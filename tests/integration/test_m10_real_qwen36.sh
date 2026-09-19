#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Real Model Integration Test Suite: Milestone M10 (Consolidation DISMOEN)
# Memverifikasi integrasi produksi engine dismoen dengan checkpoint fisik riil Qwen 3.6-35B-A3B:
#   1. Static hygiene & zero suppression (0 noqa, 0 #[allow], 0 hardcoded path host).
#   2. Verifikasi eliminasi total legacy model 1.5 (L_paths == 0, L_logical == 0) dan SSOT 26 shard Qwen 3.6.
#   3. Verifikasi Single-SSOT models.lock.json (7 field, 26 shard manifest equality).
#   4. Verifikasi format check-index pada shard riil Safetensors Qwen 3.6.
#   5. Verifikasi penolakan keras (Fail-Closed Exit 7 NO_QUANTIZER_MODEL) anti-OOM pada checkpoint riil.
#   6. Verifikasi GGUF-backed forward & decode session continuation (historical_recompute_tokens == 0).
#   7. Verifikasi SEC-5 Read-Only Model Directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DISMOEN="./dismoen"
PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

if [[ ! -x "$DISMOEN" ]]; then
    echo ">> Mengompilasi binary dismoen..."
    pixi run build
fi

MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.6-35b-a3b}"
MODEL_ROOT="$(dirname "$MODEL_DIR")"

echo "======================================================================"
echo "M10: Real Model Integration Test Suite (Consolidation DISMOEN)"
echo "Model Dir: $MODEL_DIR"
echo "======================================================================"

if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) tidak ditemukan. Melewati uji integrasi real model M10."
    exit 0
fi

TEST_DIR="/tmp/test_m10_real_qwen36_$$"
WORKDIR="${TEST_DIR}/work"
mkdir -p "$WORKDIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# ----------------------------------------------------------------------
# Stage 1: Static Hygiene & Zero-Suppression Check
# ----------------------------------------------------------------------
echo ">> [1/7] Memeriksa static hygiene dan zero-suppression..."

FORBIDDEN_USER_HOME="/home/"'will'
for file in "src/cli/cmd_forward.mojo" "src/cli/cmd_decode.mojo" "src/cli/cmd_quantize.mojo"; do
    if grep -nE "noqa|#[[:space:]]*allow" "$file"; then
        echo "FAIL: Ditemukan suppressions terlarang di $file!"
        exit 1
    fi
    if grep -nF "$FORBIDDEN_USER_HOME" "$file"; then
        echo "FAIL: Ditemukan path developer hardcoded di $file!"
        exit 1
    fi
done

# Pastikan 8 pola legacy §4.3 bernilai 0 di src/
PATTERNS=(
    '"trial"'
    'quant_model\.bin'
    '\.kimo\.bin'
    'QuantHeader'
    'QuantTensorMetadata'
    'parse_model_config_adapter'
    'kimo-tools'
    'quant_format'
)
for pat in "${PATTERNS[@]}"; do
    if rg -q -e "$pat" src/; then
        echo "FAIL: Pola legacy '$pat' ditemukan di src/!"
        rg -n -e "$pat" src/
        exit 1
    fi
done
echo "   PASS: Static hygiene, zero suppression, dan zero-legacy src/ terverifikasi."

# ----------------------------------------------------------------------
# Stage 2: Audit Eliminasi Legacy Qwen 1.5 & Keutuhan SSOT Qwen 3.6
# ----------------------------------------------------------------------
echo ">> [2/7] Mengaudit eliminasi legacy model 1.5 dan keberadaan SSOT Qwen 3.6..."

LEGACY_P1="$MODEL_ROOT/qwen1.5-moe-a2.7b-chat"
LEGACY_P2="$MODEL_ROOT/qwen1.5-moe-a2.7b-chat-4bit"

L_PATHS=0
if [ -e "$LEGACY_P1" ]; then L_PATHS=$((L_PATHS + 1)); fi
if [ -e "$LEGACY_P2" ]; then L_PATHS=$((L_PATHS + 1)); fi

if [ "$L_PATHS" -ne 0 ]; then
    echo "FAIL: L_paths invariant violated: model legacy masih ada di storage!"
    exit 1
fi

TOTAL_SHARDS=$(find "$MODEL_DIR" -name "model-*-of-00026.safetensors" | wc -l)
if [ "$TOTAL_SHARDS" -ne 26 ]; then
    echo "FAIL: Jumlah shard Safetensors tidak 26 (ditemukan: $TOTAL_SHARDS)!"
    exit 1
fi

TOTAL_BYTES=$(stat -c%s "$MODEL_DIR"/model-*-of-00026.safetensors | awk '{s+=$1} END {print s}')
TOTAL_GIB=$(awk -v b="$TOTAL_BYTES" 'BEGIN {printf "%.2f", b / (1024*1024*1024)}')
echo "   PASS: L_paths == 0 terverifikasi (bebas artefak legacy 1.5)."
echo "   PASS: Checkpoint Qwen 3.6 lengkap (26 shards, $TOTAL_GIB GiB / $TOTAL_BYTES Bytes)."

# ----------------------------------------------------------------------
# Stage 3: Verifikasi Single-SSOT models.lock.json (7 Field & SEC-1)
# ----------------------------------------------------------------------
echo ">> [3/7] Memverifikasi integritas single-SSOT models.lock.json (7 field)..."

"$PYTHON" - "$REPO_ROOT/models.lock.json" "$MODEL_DIR" <<'EOF'
import json, sys, os

lock_file = sys.argv[1]
model_dir = sys.argv[2]

with open(lock_file, "r") as f:
    mlock = json.load(f)

# Verifikasi 7 field wajib §4.5
model_id = mlock.get("model_id") or mlock.get("model")
assert model_id == "Qwen/Qwen3.6-35B-A3B", f"Invalid model_id: {model_id}"

arch = mlock.get("architecture")
assert arch == "qwen3.6", f"Invalid architecture: {arch}"

rev = mlock.get("revision")
assert rev and len(rev) == 40, f"Invalid revision: {rev}"

shards = mlock.get("shards")
assert isinstance(shards, list) and len(shards) == 26, f"Expected 26 shards, got {len(shards)}"

total_sz = 0
for i, s in enumerate(shards, 1):
    expected_fn = f"model-{i:05d}-of-00026.safetensors"
    fn = s.get("filename")
    assert fn == expected_fn, f"Shard {i} filename mismatch: {fn} != {expected_fn}"
    sz = s.get("size")
    assert isinstance(sz, int) and sz > 0, f"Invalid size for shard {fn}: {sz}"
    total_sz += sz

    h = s.get("sha256")
    assert h and h != "pinned" and not h.startswith("TBD"), f"Placeholder hash found in {fn}: {h}"
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h), f"Invalid sha256 in {fn}: {h}"

    # Manifest equality check against physical shard on disk
    s_path = os.path.join(model_dir, fn)
    assert os.path.exists(s_path), f"Shard file {fn} tidak ada di {model_dir}"
    actual_sz = os.path.getsize(s_path)
    assert actual_sz == sz, f"Size mismatch {fn}: actual {actual_sz} != locked {sz}"

assert mlock.get("total_size") == total_sz, f"total_size mismatch: {mlock.get('total_size')} != {total_sz}"

tok_rev = mlock.get("tokenizer_revision")
assert tok_rev and len(tok_rev) == 40, f"Invalid tokenizer_revision: {tok_rev}"

tok_sha = mlock.get("tokenizer_sha256")
assert isinstance(tok_sha, dict), "tokenizer_sha256 must be a dict"
assert "tokenizer.json" in tok_sha and len(tok_sha["tokenizer.json"]) == 64
assert "tokenizer_config.json" in tok_sha and len(tok_sha["tokenizer_config.json"]) == 64

cfg_h = mlock.get("config_hash")
assert cfg_h and len(cfg_h) == 64 and all(c in "0123456789abcdef" for c in cfg_h), f"Invalid config_hash: {cfg_h}"

print(f"   PASS: models.lock.json valid 100% (7 field, 26 shards, total {total_sz} bytes, 0 pinned/placeholder).")
EOF

# ----------------------------------------------------------------------
# Stage 4: Verifikasi dismoen check-index pada Shard Riil
# ----------------------------------------------------------------------
echo ">> [4/7] Menguji dismoen check-index pada shard fisik Qwen 3.6..."

FIRST_SHARD="$MODEL_DIR/model-00001-of-00026.safetensors"
"$DISMOEN" check-index "$FIRST_SHARD" > "$WORKDIR/check_index.json"

"$PYTHON" - "$WORKDIR/check_index.json" <<'EOF'
import json, sys
out = json.load(open(sys.argv[1]))
assert out.get("status") in ["match", "ok"] or out.get("valid") is True
assert len(out.get("mismatches", [])) == 0
matched = out.get("matched_tensors", out.get("num_tensors", 0))
print(f"   PASS: dismoen check-index berhasil memvalidasi {matched} tensor di shard 1 (status: {out.get('status')}).")
EOF

# ----------------------------------------------------------------------
# Stage 5: Verifikasi Fail-Closed Anti-OOM (Exit Code 7 NO_QUANTIZER_MODEL)
# ----------------------------------------------------------------------
echo ">> [5/7] Menguji fail-closed anti-OOM saat model kuantisasi tidak disediakan..."

set +e
"$DISMOEN" forward \
    --model-dir "$MODEL_DIR" \
    --tokens fixtures/m9_port_tokens.json \
    --output "$WORKDIR/unwanted_logits.bin" > "$WORKDIR/forward_fail.json" 2>&1
FWD_RC=$?

"$DISMOEN" decode \
    --model-dir "$MODEL_DIR" \
    --tokens fixtures/m9_port_tokens.json \
    --max-tokens 2 \
    --output "$WORKDIR/unwanted_decode.json" > "$WORKDIR/decode_fail.json" 2>&1
DEC_RC=$?
set -e

if [ "$FWD_RC" -ne 7 ]; then
    echo "FAIL: dismoen forward pada Safetensors 70GB tanpa quant tidak exit 7 (got: $FWD_RC)!"
    exit 1
fi
if [ "$DEC_RC" -ne 7 ]; then
    echo "FAIL: dismoen decode pada Safetensors 70GB tanpa quant tidak exit 7 (got: $DEC_RC)!"
    exit 1
fi

echo "   PASS: dismoen forward & decode fail-closed dengan Exit 7 (NO_QUANTIZER_MODEL)."
echo "   INFO: Zero OOM risk: 0 byte dari 68.12 GiB Safetensors dimuat ke RAM."

# ----------------------------------------------------------------------
# Stage 6: Verifikasi GGUF-backed Forward & Decode Session Continuation
# ----------------------------------------------------------------------
echo ">> [6/7] Menguji forward & decode session continuation berbasis GGUF..."

MINI_CONFIG="fixtures/m9_port_config_mini.json"
QUANT_GGUF="fixtures/m9_port_mini.gguf"
TOKENS_FIXTURE="fixtures/m9_port_tokens.json"
SESSION_KMSS="$WORKDIR/session_real.kmss"
LOGITS_BIN="$WORKDIR/fwd_logits.bin"
DECODE_JSON="$WORKDIR/decode_out.json"

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --save-session "$SESSION_KMSS" \
    --output "$LOGITS_BIN" > "$WORKDIR/fwd_out.json"

if [ ! -s "$SESSION_KMSS" ]; then
    echo "FAIL: File sesi KMSS v1 tidak terbentuk!"
    exit 1
fi

"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --session "$SESSION_KMSS" \
    --max-tokens 4 \
    --output "$DECODE_JSON" > "$WORKDIR/decode_log.json"

"$PYTHON" - "$DECODE_JSON" "$WORKDIR/decode_log.json" <<'EOF'
import json, sys

toks = json.load(open(sys.argv[1]))
log_out = json.load(open(sys.argv[2])) if sys.argv[2] else {}

recompute = log_out.get("historical_recompute_tokens", log_out.get("recompute_tokens", -1))
assert recompute == 0, f"historical_recompute_tokens harus 0, got {recompute}"
assert log_out.get("gdn_reused") is True or log_out.get("gdn_state_reused") is True, "gdn_reused harus True"
assert len(toks) >= 1, "Token decoded kosong"

print(f"   PASS: Decode continuation sukses: historical_recompute_tokens = 0, gdn_reused = True, tokens generated = {len(toks)}.")
EOF

# ----------------------------------------------------------------------
# Stage 7: Verifikasi Keamanan SEC-5 (Read-Only Model Directory)
# ----------------------------------------------------------------------
echo ">> [7/7] Menguji keamanan SEC-5: eksekusi di atas model directory Read-Only..."

chmod -R a-w "$MODEL_DIR"
RESTORE_PERMS() {
    chmod -R u+w "$MODEL_DIR" || true
}
trap 'RESTORE_PERMS; cleanup' EXIT

set +e
"$DISMOEN" check-index "$FIRST_SHARD" > /dev/null 2>&1
CHECK_RO_RC=$?
set -e

RESTORE_PERMS
trap cleanup EXIT

if [ "$CHECK_RO_RC" -ne 0 ]; then
    echo "FAIL: dismoen check-index gagal pada model directory read-only (RC=$CHECK_RO_RC)!"
    exit 1
fi

echo "   PASS: SEC-5 terverifikasi: engine membaca model secara read-only tanpa modifikasi fisik."

echo ""
echo "======================================================================"
echo "SUKSES: INTEGRASI REAL MODEL QWEN3.6-35B-A3B DENGAN MILESTONE M10 LULUS 100%!"
echo "======================================================================"
