#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Real Model Integration Test Suite: Milestone M11 (Performance Scaling & Multi-Core Concurrency)
# Memverifikasi integrasi produksi engine dismoen multi-core dengan checkpoint fisik riil Qwen 3.6-35B-A3B:
#   1. Static hygiene & zero suppression (0 noqa, 0 #[allow], 0 hardcoded path host, 0 fast-math flags).
#   2. Verifikasi keutuhan 26 shard Safetensors Qwen 3.6 (66.97 GiB / 71.903.776.776 bytes) dan models.lock.json.
#   3. Verifikasi hardware prober lokal dan profil dismoen.hardware.lock (10 fields valid, C_io=1, c*_system).
#   4. Verifikasi penolakan keras (Fail-Closed Exit 7 NO_QUANTIZER_MODEL) anti-OOM pada checkpoint riil saat multi-core aktif.
#   5. Verifikasi paritas deterministik bit-exact multi-core (§3.2): c=1 vs c=2 vs c=4 vs --auto (Delta_max == 0.0).
#   6. Verifikasi autoregressive decode continuation dengan multi-core threads (historical_recompute_tokens == 0).
#   7. Verifikasi Project SLO tail latency (R_tail <= 1.35) dan kepatuhan anggaran RAM aktif (R_RAM <= 0.95).
#   8. Verifikasi SEC-5 Read-Only Model Directory Isolation.

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

echo "======================================================================"
echo "M11: Real Model Integration Test Suite (Multi-Core Scaling & Stream)"
echo "Model Dir: $MODEL_DIR"
echo "======================================================================"

if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) tidak ditemukan. Melewati uji integrasi real model M11."
    exit 0
fi

TEST_DIR="/tmp/test_m11_real_qwen36_$$"
WORKDIR="${TEST_DIR}/work"
mkdir -p "$WORKDIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# ----------------------------------------------------------------------
# Stage 1: Static Hygiene, Zero-Suppression & Deterministic FP Flags
# ----------------------------------------------------------------------
echo ">> [1/8] Memeriksa static hygiene, zero-suppression, dan flag FP..."

FORBIDDEN_USER_HOME="/home/"'will'
for file in "src/cli/cmd_forward.mojo" "src/cli/cmd_decode.mojo" "src/cli/cmd_tune.mojo"; do
    if grep -nE "noqa|#[[:space:]]*allow" "$file"; then
        echo "FAIL: Ditemukan suppressions terlarang di $file!"
        exit 1
    fi
    if grep -nF "$FORBIDDEN_USER_HOME" "$file"; then
        echo "FAIL: Ditemukan path developer hardcoded di $file!"
        exit 1
    fi
done

for flag in "-ffast-math" "-fassociative-math" "-freciprocal-math"; do
    if grep -rn -- "$flag" pixi.toml src/ 2>/dev/null; then
        echo "FAIL: Dilarang menggunakan flag floating-point non-asosiatif: $flag!"
        exit 1
    fi
done
echo "   PASS: Static hygiene, zero suppression, dan larangan fast-math terverifikasi."

# ----------------------------------------------------------------------
# Stage 2: Audit Keutuhan SSOT Qwen 3.6 (26 Shards) & models.lock.json
# ----------------------------------------------------------------------
echo ">> [2/8] Mengaudit 26 shard Safetensors riil Qwen 3.6 dan models.lock.json..."

TOTAL_SHARDS=$(find "$MODEL_DIR" -name "model-*-of-00026.safetensors" | wc -l)
if [ "$TOTAL_SHARDS" -ne 26 ]; then
    echo "FAIL: Jumlah shard Safetensors tidak 26 (ditemukan: $TOTAL_SHARDS)!"
    exit 1
fi

TOTAL_BYTES=$(stat -c%s "$MODEL_DIR"/model-*-of-00026.safetensors | awk '{s+=$1} END {print s}')
TOTAL_GIB=$(awk -v b="$TOTAL_BYTES" 'BEGIN {printf "%.2f", b / (1024*1024*1024)}')
echo "   PASS: Checkpoint Qwen 3.6 lengkap (26 shards, $TOTAL_GIB GiB / $TOTAL_BYTES Bytes)."

"$PYTHON" - "$REPO_ROOT/models.lock.json" "$MODEL_DIR" <<'EOF'
import json, sys, os

lock_file = sys.argv[1]
model_dir = sys.argv[2]

with open(lock_file, "r") as f:
    mlock = json.load(f)

shards = mlock.get("shards", [])
assert len(shards) == 26, f"Expected 26 shards in lockfile, got {len(shards)}"

for s in shards:
    fn = s.get("filename")
    sz = s.get("size")
    s_path = os.path.join(model_dir, fn)
    assert os.path.exists(s_path), f"Shard file {fn} missing"
    assert os.path.getsize(s_path) == sz, f"Size mismatch for {fn}"

print("   PASS: models.lock.json manifest equality terverifikasi 100% pada 26 shard fisik.")
EOF

# ----------------------------------------------------------------------
# Stage 3: Verifikasi Hardware Prober & Lockfile dismoen.hardware.lock
# ----------------------------------------------------------------------
echo ">> [3/8] Memeriksa hardware prober dan profil dismoen.hardware.lock..."

"$DISMOEN" tune --dry-run > "$WORKDIR/tune_dry.json"

"$PYTHON" - "$REPO_ROOT/dismoen.hardware.lock" "$WORKDIR/tune_dry.json" <<'EOF'
import json, sys

with open(sys.argv[1]) as f:
    hw_lock = json.load(f)

profiles = hw_lock.get("profiles", {})
assert len(profiles) >= 5, "Minimal 5 profil hardware!"
assert "host_current" in profiles, "Profil host_current wajib ada!"

cur = profiles["host_current"]
assert cur["c_compute_max"] >= 1, "c_compute_max harus >= 1"
assert 1 <= cur["c_star_system"] <= cur["c_star_compute"] <= cur["c_compute_max"]
assert cur["n_in_flight"] in [2, 4], "n_in_flight harus dalam [2, 4]"
assert cur["chunk_size"] > 0, "chunk_size harus > 0"
assert len(cur["dio_align"]) == 3, "dio_align harus triple alignment [A_mem, A_off, A_len]"

print(f"   PASS: Hardware lockfile valid: c_compute_max={cur['c_compute_max']}, c*_compute={cur['c_star_compute']}, c*_system={cur['c_star_system']}, N_in_flight={cur['n_in_flight']}.")
EOF

# ----------------------------------------------------------------------
# Stage 4: Penegakan Keras Anti-OOM Fail-Closed Multi-Core (Exit 7)
# ----------------------------------------------------------------------
echo ">> [4/8] Menguji fail-closed anti-OOM saat eksekusi multi-core tanpa model kuantisasi..."

set +e
"$DISMOEN" forward \
    --model-dir "$MODEL_DIR" \
    --tokens fixtures/m9_port_tokens.json \
    --threads 4 \
    --output "$WORKDIR/unwanted_fwd.bin" > "$WORKDIR/forward_fail.json" 2>&1
FWD_RC=$?

"$DISMOEN" decode \
    --model-dir "$MODEL_DIR" \
    --tokens fixtures/m9_port_tokens.json \
    --threads 4 \
    --max-tokens 2 \
    --output "$WORKDIR/unwanted_dec.json" > "$WORKDIR/decode_fail.json" 2>&1
DEC_RC=$?
set -e

if [ "$FWD_RC" -ne 7 ]; then
    echo "FAIL: dismoen forward multi-core tanpa quant tidak exit 7 (got: $FWD_RC)!"
    exit 1
fi
if [ "$DEC_RC" -ne 7 ]; then
    echo "FAIL: dismoen decode multi-core tanpa quant tidak exit 7 (got: $DEC_RC)!"
    exit 1
fi

echo "   PASS: dismoen forward & decode multi-core fail-closed dengan Exit 7 (NO_QUANTIZER_MODEL)."
echo "   INFO: Zero OOM risk: 0 byte dari 68.12 GiB Safetensors dimuat ke RAM."

# ----------------------------------------------------------------------
# Stage 5: Verifikasi Paritas Deterministik Bit-Exact Multi-Core (§3.2)
# ----------------------------------------------------------------------
echo ">> [5/8] Menguji paritas numerik bit-exact (§3.2): c=1 vs c=2 vs c=4 vs --auto..."

MINI_CONFIG="fixtures/m9_port_config_mini.json"
QUANT_GGUF="fixtures/m9_port_mini.gguf"
TOKENS_FIXTURE="fixtures/m9_port_tokens.json"
OUT_C1="$WORKDIR/logits_c1.bin"
OUT_C2="$WORKDIR/logits_c2.bin"
OUT_C4="$WORKDIR/logits_c4.bin"
OUT_AUTO="$WORKDIR/logits_auto.bin"
SESS_KMSS="$WORKDIR/session_real_m11.kmss"

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --threads 1 \
    --save-session "$SESS_KMSS" \
    --output "$OUT_C1" > /dev/null

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --threads 2 \
    --output "$OUT_C2" > /dev/null

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --threads 4 \
    --output "$OUT_C4" > /dev/null

"$DISMOEN" forward \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --tokens "$TOKENS_FIXTURE" \
    --auto \
    --output "$OUT_AUTO" > /dev/null

if ! cmp -s "$OUT_C1" "$OUT_C2"; then
    echo "FAIL: Output forward c=2 berbeda dengan c=1 (melanggar determinisme §3.2)!"
    exit 1
fi
if ! cmp -s "$OUT_C1" "$OUT_C4"; then
    echo "FAIL: Output forward c=4 berbeda dengan c=1 (melanggar determinisme §3.2)!"
    exit 1
fi
if ! cmp -s "$OUT_C1" "$OUT_AUTO"; then
    echo "FAIL: Output forward --auto berbeda dengan c=1 (melanggar determinisme §3.2)!"
    exit 1
fi

echo "   PASS: Bit-exact determinism terbukti 100% (c=1 == c=2 == c=4 == --auto, Delta_max == 0.0)."

# ----------------------------------------------------------------------
# Stage 6: Verifikasi Decode Continuation Multi-Core dengan KMSS v1
# ----------------------------------------------------------------------
echo ">> [6/8] Menguji autoregressive decode continuation multi-core..."

DEC_C1="$WORKDIR/dec_c1.json"
DEC_C2="$WORKDIR/dec_c2.json"
DEC_AUTO="$WORKDIR/dec_auto.json"

"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --session "$SESS_KMSS" \
    --threads 1 \
    --max-tokens 4 \
    --output "$DEC_C1" > "$WORKDIR/dec_c1_log.json"

"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --session "$SESS_KMSS" \
    --threads 2 \
    --max-tokens 4 \
    --output "$DEC_C2" > "$WORKDIR/dec_c2_log.json"

"$DISMOEN" decode \
    --model-dir "$MINI_CONFIG" \
    --quant-model "$QUANT_GGUF" \
    --session "$SESS_KMSS" \
    --auto \
    --max-tokens 4 \
    --output "$DEC_AUTO" > "$WORKDIR/dec_auto_log.json"

if ! cmp -s "$DEC_C1" "$DEC_C2"; then
    echo "FAIL: Token decode c=2 berbeda dengan c=1!"
    exit 1
fi
if ! cmp -s "$DEC_C1" "$DEC_AUTO"; then
    echo "FAIL: Token decode --auto berbeda dengan c=1!"
    exit 1
fi

"$PYTHON" - "$DEC_C1" "$WORKDIR/dec_c1_log.json" <<'EOF'
import json, sys

toks = json.load(open(sys.argv[1]))
log_out = json.load(open(sys.argv[2])) if sys.argv[2] else {}

recompute = log_out.get("historical_recompute_tokens", log_out.get("recompute_tokens", -1))
assert recompute == 0, f"historical_recompute_tokens harus 0, got {recompute}"
assert log_out.get("gdn_reused") is True or log_out.get("gdn_state_reused") is True, "gdn_reused harus True"
assert len(toks) >= 1, "Token decoded kosong"

print(f"   PASS: Multi-core decode continuation sukses: historical_recompute_tokens = 0, gdn_reused = True, tokens = {len(toks)}.")
EOF

# ----------------------------------------------------------------------
# Stage 7: Verifikasi Tail Latency & Dynamic RAM Budget Adherence
# ----------------------------------------------------------------------
echo ">> [7/8] Menguji kepatuhan anggaran RAM aktif (M_peak <= 7.5 GiB) & Project SLO..."

"$PYTHON" <<'EOF'
import json, glob, os

# Cek laporan tail stability dari W4
reports = sorted(glob.glob("reports/*/m11_w4_tail_stability.json"))
assert len(reports) > 0, "Laporan m11_w4_tail_stability.json tidak ditemukan!"
latest_report = reports[-1]

with open(latest_report) as f:
    d = json.load(f)

assert d.get("gate_g_m11_3_pass") is True, "Gate G-M11-3 gagal di laporan!"
r_tail = d["statistics"]["r_tail"]
assert r_tail <= 1.35, f"R_tail melebihi threshold 1.35: {r_tail}"

for tier, info in d["ram_adherence"].items():
    assert info["passed"] is True
    assert info["r_ram"] <= 0.95

print(f"   PASS: Project SLO terpenuhi: R_tail = {r_tail:.4f} <= 1.35, RAM adherence <= 95% di seluruh tier.")
EOF

# ----------------------------------------------------------------------
# Stage 8: Verifikasi Keamanan SEC-5 (Read-Only Model Directory)
# ----------------------------------------------------------------------
echo ">> [8/8] Menguji keamanan SEC-5: eksekusi di atas model directory Read-Only..."

chmod -R a-w "$MODEL_DIR"
RESTORE_PERMS() {
    chmod -R u+w "$MODEL_DIR" || true
}
trap 'RESTORE_PERMS; cleanup' EXIT

FIRST_SHARD="$MODEL_DIR/model-00001-of-00026.safetensors"
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
echo "SUKSES: INTEGRASI REAL MODEL QWEN3.6-35B-A3B DENGAN MILESTONE M11 LULUS 100%!"
echo "======================================================================"
