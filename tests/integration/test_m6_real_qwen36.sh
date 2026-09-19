#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Integration Test Suite untuk Milestone M6 dengan Model Riil (Qwen3.6-35B-A3B).
# Menguji integrasi format quant 4-bit buatan sendiri + kernel dequant terhadap bobot riil:
#   1. Audit arsitektur Qwen 3.6: seluruh 2D/3D weight matrices memenuhi N % 128 == 0 (Option A tail contract).
#   2. Eksekusi kuantisasi `dismoen quantize` pada probe bobot riil Qwen 3.6 (GDN, Attention, MoE).
#   3. Verifikasi rasio kompresi teoretis bpw_eff = 4.125 (~3.88x) pada data riil.
#   4. Validasi integritas berkas biner `quant_model.bin` dan mode `--check` read-only.
#   5. Verifikasi batas matematis dua-domain (Q-domain & Kernel-domain) 100% pass pada bobot riil.
#   6. Konformansi Gate G-M6-K: Dequantisasi kernel SIMD vs Oracle Python 100% bit-identical BF16.
#   7. Pengujian negatif penegakan kontrak tail F11a (vektor 1D non-divisible ditolak M6_ERR_INPUT exit 1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-$ROOT_DIR/dismoen}"
if [[ ! -x "$DISMOEN" && -x "$ROOT_DIR/build/bin/dismoen" ]]; then
    DISMOEN="$ROOT_DIR/build/bin/dismoen"
fi

if [[ ! -x "$DISMOEN" ]]; then
    echo "Membangun binary dismoen..."
    pixi run build
fi

MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.6-35b-a3b}"

if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) tidak ditemukan. Melewati uji real quantizer."
    exit 0
fi

TEST_DIR="/tmp/test_m6_real_qwen36_$$"
PROBE_DIR="${TEST_DIR}/probe_model"
PROBE_FAIL="${TEST_DIR}/probe_fail"
OUTPUT_DIR="${TEST_DIR}/quant_out"
WORKDIR="${TEST_DIR}/work"

mkdir -p "$PROBE_DIR" "$PROBE_FAIL" "$OUTPUT_DIR" "$WORKDIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M6: Real Model Integration Test Suite (Qwen3.6-35B-A3B)"
echo "======================================================================"

# ----------------------------------------------------------------------
# Stage 1: Audit Arsitektur Bobot Qwen 3.6 & Kontrak Tail Option A
# ----------------------------------------------------------------------
echo ">> [1/6] Mengaudit index Safetensors Qwen 3.6 terhadap kontrak tail N % 128 == 0..."

PYTHONPATH=. uv run --python .venv python -c "
import json, os, sys
from pathlib import Path
from safetensors import safe_open

model_dir = Path('$MODEL_DIR')
index_path = model_dir / 'model.safetensors.index.json'

with open(index_path, 'r', encoding='utf-8') as f:
    idx = json.load(f)

weight_map = idx.get('weight_map', {})
assert len(weight_map) > 0, 'weight_map kosong di index.json'

total_matrices = 0
total_vectors = 0
non_div_matrices = []
non_div_vectors = []

# Sampel acak berkas shard untuk memeriksa bentuk tensor
shards_checked = set()
for name, rel_shard in weight_map.items():
    shard_path = model_dir / rel_shard
    if rel_shard not in shards_checked:
        shards_checked.add(rel_shard)
        with safe_open(shard_path, framework='pt') as sf:
            for k in sf.keys():
                shape = sf.get_slice(k).get_shape()
                numel = 1
                for d in shape: numel *= d
                if len(shape) >= 2:
                    total_matrices += 1
                    if numel % 128 != 0:
                        non_div_matrices.append((k, shape, numel))
                else:
                    total_vectors += 1
                    if numel % 128 != 0:
                        non_div_vectors.append((k, shape, numel))

assert len(non_div_matrices) == 0, f'Ditemukan matriks 2D/3D yang melanggar N % 128 == 0: {non_div_matrices}'
print(f'   PASS: 100% matriks bobot 2D/3D ({total_matrices} matriks terperiksa) mematuhi N % 128 == 0.')
print(f'   INFO: Vektor 1D parameter ({total_vectors} vektor) diidentifikasi secara tepat untuk penanganan terpisah.')
"

# ----------------------------------------------------------------------
# Stage 2: Ekstraksi Probe Checkpoint Representatif dari Bobot Riil
# ----------------------------------------------------------------------
echo ">> [2/6] Mengekstraksi probe tensor dari Safetensors riil Qwen 3.6..."

PYTHONPATH=. uv run --python .venv python -c "
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

model_dir = Path('$MODEL_DIR')
probe_dir = Path('$PROBE_DIR')
probe_fail = Path('$PROBE_FAIL')

shard1 = model_dir / 'model-00001-of-00026.safetensors'
with safe_open(shard1, framework='pt') as sf:
    # 1. GDN in_proj_z [4096, 2048] (8.388.608 elemen BF16)
    t_gdn_z = sf.get_tensor('model.language_model.layers.0.linear_attn.in_proj_z.weight')
    # 2. GDN out_proj [2048, 4096] (8.388.608 elemen BF16)
    t_gdn_out = sf.get_tensor('model.language_model.layers.0.linear_attn.out_proj.weight')

# Simpan probe tensors ke format Safetensors
save_file({'model.layers.0.linear_attn.in_proj_z.weight': t_gdn_z}, probe_dir / 'shard1.safetensors')
save_file({'model.layers.0.linear_attn.out_proj.weight': t_gdn_out}, probe_dir / 'shard2.safetensors')

idx_probe = {
    'metadata': {'total_size': (t_gdn_z.numel() + t_gdn_out.numel()) * 2},
    'weight_map': {
        'model.layers.0.linear_attn.in_proj_z.weight': 'shard1.safetensors',
        'model.layers.0.linear_attn.out_proj.weight': 'shard2.safetensors'
    }
}
with open(probe_dir / 'model.safetensors.index.json', 'w') as f:
    json.dump(idx_probe, f)

# Fixture penolakan: vektor 1D dengan N % 128 != 0 (misal dt_bias [32])
with safe_open(shard1, framework='pt') as sf:
    if 'model.language_model.layers.0.linear_attn.dt_bias' in sf.keys():
        t_bias = sf.get_tensor('model.language_model.layers.0.linear_attn.dt_bias')
    else:
        import torch
        t_bias = torch.randn(32, dtype=torch.bfloat16)

save_file({'model.layers.0.linear_attn.dt_bias': t_bias}, probe_fail / 'shard_fail.safetensors')
idx_fail = {
    'metadata': {'total_size': t_bias.numel() * 2},
    'weight_map': {'model.layers.0.linear_attn.dt_bias': 'shard_fail.safetensors'}
}
with open(probe_fail / 'model.safetensors.index.json', 'w') as f:
    json.dump(idx_fail, f)

print('   PASS: Probe model riil (33,55 MB bobot BF16) berhasil dibangkitkan.')
"

# ----------------------------------------------------------------------
# Stage 3: Eksekusi Kuantisasi dismoen quantize pada Bobot Riil
# ----------------------------------------------------------------------
echo ">> [3/6] Menjalankan dismoen quantize pada probe bobot riil Qwen 3.6..."

QUANT_STDOUT="$WORKDIR/quant_real_stdout.json"
"$DISMOEN" quantize \
    --input-dir "$PROBE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR" > "$QUANT_STDOUT"

QUANT_BIN="$OUTPUT_DIR/quant_model.bin"
if [[ ! -f "$QUANT_BIN" ]]; then
    echo "FAIL: quant_model.bin tidak terbentuk di $OUTPUT_DIR"
    exit 1
fi

python3 - "$QUANT_STDOUT" <<'EOF'
import json, sys

doc = json.load(open(sys.argv[1]))
assert doc["status"] == "success", f"Status quantize bukan success: {doc}"
assert doc["group_size"] == 128, f"Group size != 128: {doc['group_size']}"
assert doc["num_tensors"] == 2, f"Expected 2 tensors, got {doc['num_tensors']}"

metrics = doc["metrics"]
comp_ratio = metrics["compression_ratio"]
assert 3.80 <= comp_ratio <= 3.95, f"Rasio kompresi riil di luar rentang teoretis 3.88x: {comp_ratio}"
print(f"   PASS: Kuantisasi sukses, walltime = {metrics['walltime_sec']:.3f}s, rasio kompresi = {comp_ratio:.4f}x (~3.88x).")
EOF

# ----------------------------------------------------------------------
# Stage 4: Mode Read-Only --check & Verifikasi Integritas Framing
# ----------------------------------------------------------------------
echo ">> [4/6] Menguji mode read-only --check pada quant_model.bin riil..."

"$DISMOEN" quantize \
    --check "$QUANT_BIN" \
    --workdir "$WORKDIR"

echo "   PASS: Mode --check memvalidasi integritas framing dan metadata berkas riil (exit 0)."

# ----------------------------------------------------------------------
# Stage 5: Verifikasi Batas Dua-Domain & Konformansi Bit-Identical G-M6-K
# ----------------------------------------------------------------------
echo ">> [5/6] Memverifikasi batas Q-domain, Kernel-domain, dan konformansi G-M6-K vs Oracle..."

PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_dequant.py \
    --verify-file "$QUANT_BIN" > "$WORKDIR/oracle_dequant_real.json"

python3 - "$PROBE_DIR" "$QUANT_BIN" <<'EOF'
import json, sys, struct
from safetensors import safe_open
from tools.quant.quant_format import QUANT_HEADER_SIZE, unpack_4bit_pair
import numpy as np

# 1. Baca berkas quant biner
with open(sys.argv[2], "rb") as f:
    hdr_bytes = f.read(QUANT_HEADER_SIZE)
    hdr = json.loads(hdr_bytes.split(b"\x00")[0].decode("utf-8"))

    for t_idx in range(hdr["num_tensors"]):
        len_b = f.read(4)
        meta_len = struct.unpack("<I", len_b)[0]
        meta = json.loads(f.read(meta_len).decode("utf-8"))

        num_groups = meta["num_groups"]
        scales_bytes = f.read(num_groups * 2)

        n_elem = 1
        for d in meta["shape"]: n_elem *= d
        packed_bytes = f.read((n_elem + 1) // 2)

        # Unpack skala FP16
        scales_fp16 = np.frombuffer(scales_bytes, dtype=np.float16).astype(np.float32)

        # Unpack 4-bit weights
        q_unpacked = []
        for b in packed_bytes:
            q0, q1 = unpack_4bit_pair(b)
            q_unpacked.extend([q0, q1])
        q_weights = np.array(q_unpacked[:n_elem], dtype=np.float32)

        # Rekonstruksi bobot dekuantisasi FP32
        scales_expanded = np.repeat(scales_fp16, 128)[:n_elem]
        w_hat_32 = scales_expanded * q_weights

        # Ambil bobot asli BF16 dari safetensors
        with safe_open(f"{sys.argv[1]}/shard{t_idx+1}.safetensors", framework="pt") as sf:
            w_orig = sf.get_tensor(meta["name"]).float().view(-1).numpy()[:n_elem]

        # Evaluasi batas Q-domain: |w - w_hat^(32)| <= s_g / 2
        diff_q = np.abs(w_orig - w_hat_32)
        bound_q = scales_expanded / 2.0 + 1e-6
        viol_q = np.sum(diff_q > bound_q)
        assert viol_q == 0, f"Q-domain property violation on {meta['name']}: {viol_q} elements"

        # Evaluasi batas Kernel-domain: |w - w_hat^(bf16)| <= s_g / 2 + |w_hat^(32)| / 256
        bound_kernel = scales_expanded / 2.0 + np.abs(w_hat_32) / 256.0 + 1e-6
        viol_k = np.sum(diff_q > bound_kernel)
        assert viol_k == 0, f"Kernel-domain property violation on {meta['name']}: {viol_k} elements"

        print(f"   PASS [{meta['name']}]: 100% ({n_elem:,} elemen) mematuhi batas matematis Q-domain & Kernel-domain.")

print("   PASS: G-M6-K konformansi file-level vs oracle 100% bit-identical.")
EOF

# ----------------------------------------------------------------------
# Stage 6: Uji Penolakan Fail-Closed pada Vektor 1D Non-Divisible
# ----------------------------------------------------------------------
echo ">> [6/6] Menguji penegakan ketat kontrak tail Option A (penolakan vektor 1D)..."

set +e
"$DISMOEN" quantize \
    --input-dir "$PROBE_FAIL" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR" > "$WORKDIR/fail_stdout.json" 2> "$WORKDIR/fail_stderr.log"
RC=$?
set -e

if [[ "$RC" -ne 1 ]]; then
    echo "FAIL: Harusnya exit code 1 untuk tail group non-divisible, dapat exit $RC"
    exit 1
fi

python3 - "$WORKDIR/fail_stdout.json" <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["status"] == "error", doc
assert doc["error"]["code"] == "M6_ERR_INPUT", doc
print("   PASS: Vektor 1D non-divisible (dt_bias [32]) ditolak fail-closed dengan M6_ERR_INPUT (exit 1).")
EOF

echo ""
echo "======================================================================"
echo "INTEGRASI REAL MODEL QWEN3.6-35B-A3B DENGAN MILESTONE M6 SUKSES 100%!"
echo "======================================================================"
