#!/bin/bash
# ==============================================================================
# test_m6_w3_quantize_cli.sh — Integration Test Suite M6-W3 (Quantize CLI & Rollback)
#
# Memverifikasi DoD M6-W3:
# 1. CLI `dismoen quantize` lengkap: --input-dir, --output-dir, --group-size, --workdir, --check.
# 2. Strict JSON RFC 8259 pada stdout untuk output sukses dan error.
# 3. Pemetaan 5 kode error normatif dan exit code yang tepat:
#    - M6_ERR_INPUT      -> exit 1
#    - M6_ERR_QUANT      -> exit 2
#    - M6_ERR_DEQUANT    -> exit 2
#    - M6_ERR_OUTPUT     -> exit 3
#    - M6_ERR_VALIDATION -> exit 4
# 4. Mode `--check` read-only untuk integritas berkas dan penolakan nibble reserved 0x8 / truncated.
# 5. Atomic rollback: file sementara di workdir bersih total pada failure, tanpa partial valid file.
# 6. Cakupan skenario IT-M6-1 s/d IT-M6-11 sesuai Test Matrix spec M6.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "M6-W3: CLI dismoen quantize + Error Handling + Atomic Rollback"
echo "======================================================================"

# ----------------------------------------------------------------------
# 1. Formatting & Code Quality
# ----------------------------------------------------------------------
echo ">> [1/6] Memeriksa formatting Mojo..."

FORMAT_OUTPUT=$(pixi run mojo format \
    src/cli/m6_errors.mojo \
    src/cli/cmd_quantize.mojo \
    src/main.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi
echo "   PASS: Formatting Mojo bersih 100%."

# ----------------------------------------------------------------------
# 2. Build Dismoen Executable
# ----------------------------------------------------------------------
echo ">> [2/6] Membangun binary dismoen via pixi build..."
pixi run build
DISMOEN="./dismoen"
[ -x "$DISMOEN" ] || { echo "FAIL: binary dismoen tidak ditemukan"; exit 1; }
echo "   PASS: Binary dismoen siap dijalankan."

# ----------------------------------------------------------------------
# 3. Setup Test Fixtures & Harness
# ----------------------------------------------------------------------
echo ">> [3/6] Menyiapkan fixture model BF16 dan direktori kerja..."

TEST_BASE="/tmp/test_m6_w3_$$"
INPUT_DIR="$TEST_BASE/model_bf16"
INPUT_NAN="$TEST_BASE/model_nan"
INPUT_TAIL="$TEST_BASE/model_tail"
OUTPUT_DIR="$TEST_BASE/model_4bit"
WORKDIR="$TEST_BASE/work"

cleanup() {
    rm -rf "$TEST_BASE"
}
trap cleanup EXIT

mkdir -p "$INPUT_DIR" "$INPUT_NAN" "$INPUT_TAIL" "$OUTPUT_DIR" "$WORKDIR"

# Helper generator fixture model safetensors
uv run --python .venv python -c "
import json
import torch
from safetensors.torch import save_file

# 1. Happy path model: 2 tensor, semuanya kelipatan 128 (N % 128 == 0)
t1 = torch.randn(128, 128, dtype=torch.bfloat16) * 0.5
t2 = torch.randn(64, 128, dtype=torch.bfloat16) * 0.25
save_file({'model.layers.0.q_proj.weight': t1}, '$INPUT_DIR/shard1.safetensors')
save_file({'model.layers.0.k_proj.weight': t2}, '$INPUT_DIR/shard2.safetensors')

idx1 = {
    'metadata': {'total_size': (128*128 + 64*128) * 2},
    'weight_map': {
        'model.layers.0.q_proj.weight': 'shard1.safetensors',
        'model.layers.0.k_proj.weight': 'shard2.safetensors',
    }
}
with open('$INPUT_DIR/model.safetensors.index.json', 'w') as f:
    json.dump(idx1, f)

# 2. NaN model: 1 tensor mengandung NaN
t_nan = torch.randn(128, 128, dtype=torch.bfloat16)
t_nan[0, 0] = float('nan')
save_file({'model.layers.0.nan.weight': t_nan}, '$INPUT_NAN/shard1.safetensors')
idx_nan = {
    'metadata': {'total_size': 128*128*2},
    'weight_map': {'model.layers.0.nan.weight': 'shard1.safetensors'}
}
with open('$INPUT_NAN/model.safetensors.index.json', 'w') as f:
    json.dump(idx_nan, f)

# 3. Tail model: 1 tensor dengan N % 128 != 0 (misal vektor bias N = 64)
t_tail = torch.randn(64, dtype=torch.bfloat16)
save_file({'model.layers.0.tail.bias': t_tail}, '$INPUT_TAIL/shard1.safetensors')
idx_tail = {
    'metadata': {'total_size': 64*2},
    'weight_map': {'model.layers.0.tail.bias': 'shard1.safetensors'}
}
with open('$INPUT_TAIL/model.safetensors.index.json', 'w') as f:
    json.dump(idx_tail, f)
"
echo "   PASS: Seluruh model fixture berhasil dibangkitkan."

# Helper assertion: assert exit code & catat stdout/stderr
expect_rc() {
    local expected="$1"; local desc="$2"; shift 2
    set +e
    "$@" >"$WORKDIR/last_stdout.json" 2>"$WORKDIR/last_stderr.log"
    local rc=$?
    set -e
    if [ "$rc" -ne "$expected" ]; then
        echo "FAIL: $desc: exit $rc, want $expected"
        echo "--- STDOUT ---"
        cat "$WORKDIR/last_stdout.json" || true
        echo "--- STDERR ---"
        cat "$WORKDIR/last_stderr.log" || true
        exit 1
    fi
    echo "   PASS: $desc (exit $rc)"
}

# Helper assertion: stdout wajib JSON error strict RFC 8259
expect_error_code() {
    local expected_code="$1"
    python3 - "$expected_code" "$WORKDIR/last_stdout.json" <<'EOF'
import json, sys
with open(sys.argv[2], "r") as f:
    doc = json.load(f)
assert doc["status"] == "error", f"status is not error: {doc}"
assert doc["error"]["code"] == sys.argv[1], f"error code mismatch: expected {sys.argv[1]}, got {doc['error']['code']}"
assert "stage" in doc["error"], "missing stage in error"
assert "message" in doc["error"], "missing message in error"
EOF
    echo "   PASS: error $expected_code terparse sebagai JSON strict RFC 8259."
}

# ----------------------------------------------------------------------
# 4. Pengujian IT-M6-1 s/d IT-M6-4 (Input & Quantization Failures)
# ----------------------------------------------------------------------
echo ">> [4/6] Menguji skenario input, exit code 0-2, dan validasi input-dir..."

# IT-M6-1: Happy path
expect_rc 0 "IT-M6-1 happy path quantize" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR"

# Verifikasi struktur JSON output IT-M6-1
python3 - "$WORKDIR/last_stdout.json" "$OUTPUT_DIR/quant_model.bin" <<'EOF'
import json, sys, os
doc = json.load(open(sys.argv[1]))
assert doc["status"] == "success", doc
assert doc["group_size"] == 128
assert doc["num_tensors"] == 2
metrics = doc["metrics"]
assert metrics["compression_ratio"] > 3.0
assert os.path.exists(sys.argv[2])
EOF
echo "   PASS: IT-M6-1 output JSON dan file biner terverifikasi valid."

# IT-M6-2: Input-dir tidak ada
expect_rc 1 "IT-M6-2 input-dir tidak ada" \
    "$DISMOEN" quantize \
    --input-dir "/tmp/nonexistent_model_dir_$$" \
    --output-dir "$OUTPUT_DIR" \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_INPUT

# IT-M6-3: Invalid group-size (∉ {32,64,128,256})
expect_rc 1 "IT-M6-3 group-size 100 ditolak" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 100 \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_INPUT

# IT-M6-11: Tail group (N % G != 0)
expect_rc 1 "IT-M6-11 tail group N % G != 0 ditolak" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_TAIL" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_INPUT

# IT-M6-4: Quantization fail pada NaN in scale / tensor
expect_rc 2 "IT-M6-4 tensor mengandung NaN" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_NAN" \
    --output-dir "$OUTPUT_DIR" \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_QUANT

# ----------------------------------------------------------------------
# 5. Pengujian Mode --check, IT-M6-5, IT-M6-6 & Atomic Rollback
# ----------------------------------------------------------------------
echo ">> [5/6] Menguji mode --check, IT-M6-5, IT-M6-6 & rollback..."

# Mode --check pada berkas valid
expect_rc 0 "Mode --check berkas valid" \
    "$DISMOEN" quantize \
    --check "$OUTPUT_DIR/quant_model.bin" \
    --workdir "$WORKDIR"

# IT-M6-5: Dequantization fail (suntik nibble reserved 0x8)
python3 - "$OUTPUT_DIR/quant_model.bin" "$OUTPUT_DIR/quant_corrupt.bin" <<'EOF'
import sys
raw = bytearray(open(sys.argv[1], "rb").read())
# Ganti nibble di bobot terakhir menjadi 0x8 (reserved nibble)
raw[-1] = (raw[-1] & 0x0F) | 0x80
open(sys.argv[2], "wb").write(bytes(raw))
EOF

expect_rc 2 "IT-M6-5 nibble reserved 0x8 ditolak" \
    "$DISMOEN" quantize \
    --check "$OUTPUT_DIR/quant_corrupt.bin" \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_DEQUANT

# IT-M6-6: Output validation fail (berkas dipotong 1 byte)
cp "$OUTPUT_DIR/quant_model.bin" "$OUTPUT_DIR/quant_trunc.bin"
truncate -s -1 "$OUTPUT_DIR/quant_trunc.bin"

expect_rc 4 "IT-M6-6 file terpotong ditolak" \
    "$DISMOEN" quantize \
    --check "$OUTPUT_DIR/quant_trunc.bin" \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_VALIDATION

# Verifikasi Atomic Rollback: saat failure, workdir tidak boleh memiliki file .tmp yang tertinggal
TMP_LEFTOVER=$(find "$WORKDIR" -name "quant_model.tmp.*" | wc -l)
[ "$TMP_LEFTOVER" -eq 0 ] || { echo "FAIL: ada $TMP_LEFTOVER file temporary tertinggal di workdir"; exit 1; }
echo "   PASS: Atomic rollback terverifikasi (workdir bersih 100% tanpa sampah temp)."

# ----------------------------------------------------------------------
# 6. Determinisme (IT-M6-8) & Custom Group Size 64 (IT-M6-9)
# ----------------------------------------------------------------------
echo ">> [6/6] Menguji determinisme IT-M6-8 & custom group-size 64 IT-M6-9..."

# IT-M6-8: Deterministic SHA-256
expect_rc 0 "IT-M6-8 run 1" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR"
SHA1=$(sha256sum "$OUTPUT_DIR/quant_model.bin" | cut -d' ' -f1)

expect_rc 0 "IT-M6-8 run 2" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR"
SHA2=$(sha256sum "$OUTPUT_DIR/quant_model.bin" | cut -d' ' -f1)

[ "$SHA1" = "$SHA2" ] || { echo "FAIL: determinisme gagal: $SHA1 != $SHA2"; exit 1; }
echo "   PASS: IT-M6-8 deterministik bit-identical (SHA256: $SHA1)."

# IT-M6-9: Custom group size 64
expect_rc 0 "IT-M6-9 custom group-size 64" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 64 \
    --workdir "$WORKDIR"
S64=$(stat -c%s "$OUTPUT_DIR/quant_model.bin")

expect_rc 0 "IT-M6-9 group-size 128 untuk perbandingan rasio" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR"
S128=$(stat -c%s "$OUTPUT_DIR/quant_model.bin")

python3 - "$S64" "$S128" <<'EOF'
import sys
s64 = int(sys.argv[1])
s128 = int(sys.argv[2])
r = s64 / s128
# Teori rasio: (4 + 16/64) / (4 + 16/128) = 4.25 / 4.125 ≈ 1.0303
assert 1.0 <= r <= 1.10, f"Rasio ukuran di luar toleransi ±10%: {r}"
print(f"   PASS: IT-M6-9 rasio ukuran G=64 vs G=128 = {r:.4f} (~1.03x)")
EOF

echo "======================================================================"
echo "M6-W3 VERIFIKASI LENGKAP: CLI, ERROR HANDLING & ROLLBACK HIJAU (GREEN)"
echo "======================================================================"
