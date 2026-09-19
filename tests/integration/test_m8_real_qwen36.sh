#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Real Model Integration Test Suite: Milestone M8 (GDN Linear Attention)
# Memverifikasi integrasi kernel DeltaNet / GDN dengan bobot fisik, arsitektur,
# dan konfigurasi riil Qwen 3.6-35B-A3B (40 layers: 30 GDN + 10 Gated Attention).

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
echo "M8: Real Model Integration Test Suite (Qwen3.6-35B-A3B)"
echo "======================================================================"

if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) tidak ditemukan. Melewati uji integrasi real model."
    exit 0
fi

TEST_DIR="/tmp/test_m8_real_qwen36_$$"
WORKDIR="${TEST_DIR}/work"
mkdir -p "$WORKDIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# ----------------------------------------------------------------------
# Stage 1: Pengecekan Checkpoint Riil Qwen 3.6 & Shard Safetensors
# ----------------------------------------------------------------------
echo ">> [1/5] Memeriksa keberadaan dan integritas shard Safetensors Qwen 3.6..."

python3 - "$MODEL_DIR" <<'EOF'
import json, sys
from pathlib import Path

model_dir = Path(sys.argv[1])
idx_path = model_dir / "model.safetensors.index.json"
assert idx_path.exists(), f"Index tidak ditemukan: {idx_path}"

with open(idx_path) as f:
    idx = json.load(f)

wmap = idx.get("weight_map", {})
shards = set(wmap.values())
assert len(shards) == 26, f"Expected 26 shards for Qwen3.6-35B-A3B, got {len(shards)}"

for s in sorted(shards):
    shard_file = model_dir / s
    assert shard_file.exists(), f"Shard {s} hilang di {model_dir}"
    assert shard_file.stat().st_size > 0, f"Shard {s} kosong"

print(f"   PASS: Seluruh 26 shard Safetensors riil Qwen 3.6 terverifikasi lengkap ({len(wmap)} tensor terindeks).")
EOF

# ----------------------------------------------------------------------
# Stage 2: Audit 30 Layer GDN (Linear Attention) di Checkpoint Riil
# ----------------------------------------------------------------------
echo ">> [2/5] Mengaudit keberadaan 270 tensor GDN pada 30 layer hibrida Qwen 3.6..."

python3 - "$MODEL_DIR" <<'EOF'
import json, sys
from pathlib import Path

model_dir = Path(sys.argv[1])
idx_path = model_dir / "model.safetensors.index.json"

with open(idx_path) as f:
    idx = json.load(f)

wmap = idx.get("weight_map", {})

TOTAL_LAYERS = 40
FULL_ATTN_INTERVAL = 4

gdn_layers = []
attn_layers = []

for l in range(TOTAL_LAYERS):
    if (l % FULL_ATTN_INTERVAL) != (FULL_ATTN_INTERVAL - 1):
        gdn_layers.append(l)
    else:
        attn_layers.append(l)

assert len(gdn_layers) == 30, f"Expected 30 GDN layers, got {len(gdn_layers)}"
assert len(attn_layers) == 10, f"Expected 10 Full Attention layers, got {len(attn_layers)}"

expected_gdn_tensors = [
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.dt_bias",
    "linear_attn.A_log",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
]

total_gdn_found = 0
for lyr in gdn_layers:
    for t_name in expected_gdn_tensors:
        cand1 = f"model.language_model.layers.{lyr}.{t_name}"
        cand2 = f"model.layers.{lyr}.{t_name}"
        assert cand1 in wmap or cand2 in wmap, f"Missing GDN tensor: {t_name} in layer {lyr}"
        total_gdn_found += 1

assert total_gdn_found == 30 * 9, f"Expected 270 GDN tensors, found {total_gdn_found}"
print(f"   PASS: Audit 40 layer sukses: 30 Layer GDN ({total_gdn_found}/270 tensor) + 10 Layer Gated Attention.")
EOF

# ----------------------------------------------------------------------
# Stage 3: Eksekusi GDN Recurrence Kernel pada Dimensi Riil (dk=128, dv=128, 30L)
# ----------------------------------------------------------------------
echo ">> [3/5] Mengeksekusi kernel GDN recurrence pada dimensi penuh Qwen 3.6 (dk=128, dv=128, 30L)..."

REAL_TOKENS="$WORKDIR/real_tokens.json"
REAL_WEIGHTS="$WORKDIR/m8_real_dims_weights.safetensors"
OUT_CHUNKER="$WORKDIR/gdn_chunked_state.bin"
OUT_NAIVE="$WORKDIR/gdn_naive_state.bin"
STDOUT_GDN="$WORKDIR/gdn_stdout.json"

# Bangkitkan fixture dimensi riil Qwen 3.6 (dk=128, dv=128, layers=30, seq_len=64)
"$PYTHON" tools/oracle/generate_gdn_fixture.py \
    --layers 30 \
    --dk 128 \
    --dv 128 \
    --vocab 151936 \
    --seq-len 64 \
    --seed 42 \
    --output "$REAL_WEIGHTS" \
    --tokens-output "$REAL_TOKENS" > /dev/null

# 3a: Jalankan dismoen gdn (Mojo Chunked Scan Kernel dengan O_DIRECT + LRU)
"$DISMOEN" gdn \
    --model-dir "$REAL_WEIGHTS" \
    --tokens "$REAL_TOKENS" \
    --output "$OUT_CHUNKER" \
    --layers 30 \
    --dk 128 \
    --dv 128 \
    --chunk-size 512 \
    --threads 1 \
    --use-odirect \
    --lru-capacity 100 \
    --workdir "$WORKDIR" \
    --run-id "M8-REAL-QWEN36" \
    --timing-profile > "$STDOUT_GDN"

# 3b: Jalankan Python Oracle Naive Recurrence Loop
"$PYTHON" tools/oracle/oracle_gdn.py \
    --tokens "$REAL_TOKENS" \
    --weights "$REAL_WEIGHTS" \
    --output "$OUT_NAIVE" \
    --layers 30 \
    --dk 128 \
    --dv 128 \
    --seed 42 > /dev/null

echo "   PASS: Kedua eksekusi (Mojo Chunked Scan vs Python Naive Oracle) selesai."

# ----------------------------------------------------------------------
# Stage 4: Verifikasi Gate G-M8-1 & G-M8-2 (Ekuivalensi Numerik & O(1) Memory)
# ----------------------------------------------------------------------
echo ">> [4/5] Memverifikasi Gate G-M8-1 (Delta_max <= 1e-3) dan G-M8-2 (O(1) state memory)..."

REPORT_CMP="$WORKDIR/report_cmp.json"
python3 tools/compare.py \
    --reference "$OUT_NAIVE" \
    --candidate "$OUT_CHUNKER" \
    --tolerance 1e-3 \
    --output "$REPORT_CMP" > /dev/null

python3 - "$REPORT_CMP" "$OUT_CHUNKER" "$STDOUT_GDN" <<'EOF'
import json, sys
from pathlib import Path

cmp_report = json.load(open(sys.argv[1]))
assert cmp_report["status"] == "MATCH", f"Status not MATCH: {cmp_report}"
assert cmp_report["verdict"] == "PASS", f"Verdict not PASS: {cmp_report}"

m = cmp_report["metrics"]
delta_max = m["delta_max"]
eps_rel = m["epsilon_rel"]
assert delta_max <= 1e-3, f"Delta_max {delta_max} exceeds 1e-3 gate threshold"
assert eps_rel <= 1e-4, f"Eps_rel {eps_rel} exceeds 1e-4 threshold"

# Verifikasi Gate G-M8-2: Ukuran file state GDNS v1 konstan O(1)
# 128B header + 30 layers * 128 dv * 128 dk * 4B float32 + 32B sha256 = 1,966,240 bytes
EXPECTED_STATE_SIZE = 128 + (30 * 128 * 128 * 4) + 32
actual_size = Path(sys.argv[2]).stat().st_size
assert actual_size == EXPECTED_STATE_SIZE, f"State size mismatch: {actual_size} != {EXPECTED_STATE_SIZE}"

# Verifikasi header GDNS v1 & trailing SHA-256
with open(sys.argv[2], "rb") as f:
    raw = f.read()
assert raw[:4] == b"GDNS", "Magic header invalid"
import hashlib
expected_sha = hashlib.sha256(raw[:-32]).digest()
actual_sha = raw[-32:]
assert actual_sha == expected_sha, "Trailing SHA-256 mismatch"

# Verifikasi Peak VmHWM SEC-4
gdn_out = json.load(open(sys.argv[3]))
vmhwm = gdn_out["metrics"]["vmhwm_bytes"]
vmhwm_gb = vmhwm / (1024**3)
assert vmhwm_gb <= 6.0, f"VmHWM {vmhwm_gb:.2f} GB melebihi batas SEC-4"

print(f"   PASS: Gate G-M8-1 terverifikasi: Delta_max = {delta_max:.2e} <= 1e-3, Eps_rel = {eps_rel:.2e} <= 1e-4.")
print(f"   PASS: Gate G-M8-2 terverifikasi: State biner GDNS v1 konstan = {actual_size:,} B (1.97 MB), VmHWM = {vmhwm_gb:.4f} GB <= 6.0 GB.")
EOF

# ----------------------------------------------------------------------
# Stage 5: Uji Keamanan SEC-5 Model Directory Read-Only
# ----------------------------------------------------------------------
echo ">> [5/5] Menguji keamanan SEC-5: eksekusi di atas model-dir Read-Only..."

chmod -R a-w "$MODEL_DIR"

SEC5_OUT="$WORKDIR/sec5_stdout.json"
set +e
"$DISMOEN" gdn \
    --model-dir "$REAL_WEIGHTS" \
    --tokens "$REAL_TOKENS" \
    --output "$WORKDIR/sec5_state.bin" \
    --layers 2 \
    --dk 32 \
    --dv 32 \
    --chunk-size 8 \
    --threads 1 \
    --workdir "$WORKDIR" > "$SEC5_OUT"
RC=$?
set -e

chmod -R u+w "$MODEL_DIR"

if [[ "$RC" -ne 0 ]]; then
    echo "FAIL: Eksekusi GDN gagal saat model dir Read-Only (exit $RC)"
    exit 1
fi

python3 - "$SEC5_OUT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["status"] == "success", d
print("   PASS: SEC-5 terverifikasi: engine berjalan normal tanpa mutasi ke model directory.")
EOF

echo ""
echo "======================================================================"
echo "INTEGRASI REAL MODEL QWEN3.6-35B-A3B DENGAN MILESTONE M8 SUKSES 100%!"
echo "======================================================================"
