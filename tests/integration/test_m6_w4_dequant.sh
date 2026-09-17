#!/bin/bash
# ==============================================================================
# test_m6_w4_dequant.sh — Integration Test Suite M6-W4 (Dequant Kernel & Forward)
#
# Memverifikasi DoD M6-W4:
# 1. Kernel Dequant SIMD: Unpack nibble -> tolak 0b1000 -> signed -7..7 -> x scale -> BF16.
# 2. Validasi: Skala finite, nibble reserved ditolak, bound Kernel-domain terpenuhi 100%.
# 3. SEC-4: Parser hardening (caps max_tensors, max_name, max_ndim, overflow-safe, non-overlap).
# 4. G-M6-K: Konformansi vs oracle (seed-42 fixture 131.072 elemen 100% bit-identical + tolak identik).
# 5. Integrasi: pread quant -> dequant buffer -> forward M2/M3 -> discard (streaming tetap).
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "M6-W4: Dequant Kernel SIMD + SEC-4 Hardening + G-M6-K Conformance"
echo "======================================================================"

FIXTURES_DIR="/tmp/test_dequant_fixtures"
BF16_MODEL_DIR="/tmp/test_m6_w4_bf16"
QUANT_OUT_DIR="/tmp/test_m6_w4_out"
WORKDIR="/tmp/test_m6_w4_work"

rm -rf "$FIXTURES_DIR" "$BF16_MODEL_DIR" "$QUANT_OUT_DIR" "$WORKDIR"
mkdir -p "$FIXTURES_DIR" "$BF16_MODEL_DIR" "$QUANT_OUT_DIR" "$WORKDIR"

# ----------------------------------------------------------------------
# 1. Formatting & Code Quality
# ----------------------------------------------------------------------
echo ">> [1/6] Memeriksa formatting Mojo dan Python..."

# Mojo format check
FORMAT_OUTPUT=$(pixi run mojo format \
    src/quant/dequant_kernel.mojo \
    src/quant/__init__.mojo \
    src/format/quant_reader.mojo \
    src/format/quant_format.mojo \
    src/format/__init__.mojo \
    src/layers/quant_loader.mojo \
    tests/unit/test_m6_dequant.mojo \
    tests/integration/test_m6_w4_forward.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi

# Python ruff check
uvx ruff@0.8.4 check tools/oracle/oracle_dequant.py
uvx ruff@0.8.4 format --check tools/oracle/oracle_dequant.py
echo "   PASS: Formatting Mojo dan Python bersih 100%."

# ----------------------------------------------------------------------
# 2. Build Binary kimo
# ----------------------------------------------------------------------
echo ">> [2/6] Membangun binary kimo via pixi build..."
pixi run build
echo "   PASS: Binary kimo siap dijalankan."

# ----------------------------------------------------------------------
# 3. Pembangkitan Fixture Conformance G-M6-K (Seed-42)
# ----------------------------------------------------------------------
echo ">> [3/6] Membangkitkan fixture G-M6-K (seed-42, 1024 grup x 128 = 131.072 elemen)..."
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_dequant.py \
    --generate-fixtures "$FIXTURES_DIR" \
    --num-groups 1024 \
    --group-size 128 > "$WORKDIR/fixture_meta.json"

if [ ! -f "$FIXTURES_DIR/fixture_seed42_golden_bf16.bin" ]; then
    echo "FAIL: Golden BF16 fixture oracle tidak terbentuk"
    exit 1
fi
echo "   PASS: Fixture seed-42 berhasil dibangkitkan oleh Python oracle."

# ----------------------------------------------------------------------
# 4. Mojo Unit Test Suite (SIMD, SEC-4, G-M6-K Bit-Identical)
# ----------------------------------------------------------------------
echo ">> [4/6] Menjalankan Mojo TestSuite untuk Dequant, SEC-4 & G-M6-K..."
pixi run mojo run -I src tests/unit/test_m6_dequant.mojo
echo "   PASS: Seluruh unit test Mojo lulus 100% (G-M6-K bit-identical terverifikasi)."

# ----------------------------------------------------------------------
# 5. File-Level Conformance (Oracle vs kimo quantize output)
# ----------------------------------------------------------------------
echo ">> [5/6] Menyiapkan model BF16 dan menguji konformansi file-level..."

uv run --python .venv python - "$BF16_MODEL_DIR" <<'EOF'
import os, sys, torch
from safetensors.torch import save_file

model_dir = sys.argv[1]
os.makedirs(model_dir, exist_ok=True)

# Bobot routed expert layer 0 expert 0
torch.manual_seed(42)
w_gate = torch.randn(1408, 2048, dtype=torch.bfloat16) * 0.02
w_up = torch.randn(1408, 2048, dtype=torch.bfloat16) * 0.02
w_down = torch.randn(2048, 1408, dtype=torch.bfloat16) * 0.02

tensors = {
    "model.layers.0.mlp.experts.0.gate_proj.weight": w_gate,
    "model.layers.0.mlp.experts.0.up_proj.weight": w_up,
    "model.layers.0.mlp.experts.0.down_proj.weight": w_down,
}

save_file(tensors, os.path.join(model_dir, "model.safetensors"))
EOF

# Kuantisasi model via CLI kimo
./kimo quantize \
    --input-dir "$BF16_MODEL_DIR" \
    --output-dir "$QUANT_OUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR" > "$WORKDIR/quant_stdout.json"

QUANT_MODEL_BIN="$QUANT_OUT_DIR/quant_model.bin"
if [ ! -f "$QUANT_MODEL_BIN" ]; then
    echo "FAIL: quant_model.bin tidak ditemukan setelah kuantisasi"
    exit 1
fi

# Verifikasi file-level dengan oracle python
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_dequant.py \
    --verify-file "$QUANT_MODEL_BIN" > "$WORKDIR/file_conformance.json"
echo "   PASS: Konformansi file-level G-M6-K terverifikasi pada quant_model.bin."

# ----------------------------------------------------------------------
# 6. Integrasi Layer Forward On-the-Fly & SEC-4 Fault Injection
# ----------------------------------------------------------------------
echo ">> [6/6] Menguji integrasi Layer Forward streaming on-the-fly & SEC-4 fault injection..."

# Jalankan forward streaming
FWD_OUT=$(pixi run mojo run -I src tests/integration/test_m6_w4_forward.mojo --quant-file "$QUANT_MODEL_BIN")
if ! echo "$FWD_OUT" | grep -q '"forward_ok":true'; then
    echo "FAIL: Streaming forward gagal: $FWD_OUT"
    exit 1
fi
echo "   PASS: Forward SwiGLU berjalan sukses di atas bobot quant streaming (on-the-fly dequant)."

# SEC-4 Fault Injection: truncated file
cp "$QUANT_MODEL_BIN" "$QUANT_OUT_DIR/model_trunc.bin"
truncate -s -16 "$QUANT_OUT_DIR/model_trunc.bin"

TRUNC_FAILED=0
set +e
pixi run mojo run -I src tests/integration/test_m6_w4_forward.mojo --quant-file "$QUANT_OUT_DIR/model_trunc.bin" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    TRUNC_FAILED=1
fi
set -e
if [ "$TRUNC_FAILED" -ne 1 ]; then
    echo "FAIL: SEC-4 gagal menolak file terpotong"
    exit 1
fi
echo "   PASS: SEC-4 berhasil menolak berkas quant terpotong (truncated)."

# SEC-4 Fault Injection: inject reserved nibble 0x8 ke dalam model.kimo.bin
cp "$QUANT_MODEL_BIN" "$QUANT_OUT_DIR/model_corrupt.bin"
python3 - "$QUANT_OUT_DIR/model_corrupt.bin" <<'EOF'
import sys
raw = bytearray(open(sys.argv[1], "rb").read())
# Ganti byte bobot pertama di payload record pertama
# Header 256B, meta_len 4B, json ~200B, scales ~22K, lalu bobot
raw[-100] = (raw[-100] & 0x0F) | 0x80  # inject 0b1000
open(sys.argv[1], "wb").write(bytes(raw))
EOF

NIBBLE_FAILED=0
set +e
pixi run mojo run -I src tests/integration/test_m6_w4_forward.mojo --quant-file "$QUANT_OUT_DIR/model_corrupt.bin" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    NIBBLE_FAILED=1
fi
set -e
if [ "$NIBBLE_FAILED" -ne 1 ]; then
    echo "FAIL: Kernel gagal menolak reserved nibble 0x8 pada forward streaming"
    exit 1
fi
echo "   PASS: Kernel SIMD berhasil menolak reserved nibble 0x8 pada forward streaming."

# Bersihkan direktori temporary
rm -rf "$FIXTURES_DIR" "$BF16_MODEL_DIR" "$QUANT_OUT_DIR" "$WORKDIR"

echo "======================================================================"
echo "M6-W4 VERIFIKASI LENGKAP: DEQUANT SIMD, SEC-4 & G-M6-K HIJAU (GREEN)"
echo "======================================================================"
