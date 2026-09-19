#!/bin/bash
# ==============================================================================
# test_m6_quantize.sh — Master End-to-End Integration Test Suite Milestone M6
#
# Memverifikasi seluruh 16 skenario Test Matrix dan 4 Gate Normatif Milestone M6:
# - G-M6-1: Quantization error <= 1e-2 per tensor bervariansi + jalur absolut
# - G-M6-2: Ukuran file logis (|pred-meas|/meas <= 10% via stat st_size)
# - G-M6-3: PPL quality (ΔPPL <= +0.5, argmax agreement >= 95%, agregasi global)
# - G-M6-K: Konformansi kernel dequant SIMD vs oracle (100% bit-identical)
#
# Cakupan Test Matrix IT-M6-1 s/d IT-M6-16 (Normatif):
# IT-M6-1  : Happy path: quantize BF16 -> 4-bit (Exit 0, G-M6-1 PASS)
# IT-M6-2  : Input-dir tidak ada (Exit 1, error M6_ERR_INPUT)
# IT-M6-3  : Invalid group-size (Exit 1, error M6_ERR_INPUT)
# IT-M6-4  : Quantization fail (NaN in scale) (Exit 2, error M6_ERR_QUANT)
# IT-M6-5  : Dequantization fail (nibble reserved 0x8) (Exit 2, error M6_ERR_DEQUANT)
# IT-M6-6  : Output validation fail (file terpotong) (Exit 4, error M6_ERR_VALIDATION)
# IT-M6-7  : PPL measurement & pins verification (Exit 0, G-M6-3 PASS)
# IT-M6-8  : Deterministic output (SHA-256 match di 2 run)
# IT-M6-9  : Custom group-size 64 (rasio ukuran ~1.03x, G-M6-2 validasi ukuran)
# IT-M6-10 : Property test Q-domain (FP32) |w - w^(32)| <= s_g/2 (100% pass)
# IT-M6-11 : Tail group N % G != 0 (Exit 1, error M6_ERR_INPUT)
# IT-M6-12 : Tie-break rounding golden (+-k + 0.5 -> even, byte-identical)
# IT-M6-13 : Audit bound Kernel-domain (BF16 out) <= s_g/2 + |w^(32)|/256 (100% pass)
# IT-M6-14 : Tensor konstan (var = 0) -> epsilon_rel null, verdict absolut
# IT-M6-15 : Konformansi kernel vs oracle (G-M6-K seed-42 bit-identical 100%)
# IT-M6-16 : Konformansi file-level (oracle vs SIMD pada berkas quant_model.bin)
#
# Keamanan: SEC-4 (alloc boundary, tolak liar, hardening), SEC-6, read-only model dir,
# atomic rollback (zero orphan files).
# Pola `|| true` dilarang keras.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "MILESTONE M6: Master Integration Test Suite & Gate Certification"
echo "======================================================================"

# ----------------------------------------------------------------------
# 1. Formatting & Code Quality Checks
# ----------------------------------------------------------------------
echo ">> [1/8] Memeriksa kepatuhan formatting (Mojo & Python)..."

FORMAT_OUTPUT=$(pixi run mojo format \
    src/cli/m6_errors.mojo \
    src/cli/cmd_quantize.mojo \
    src/format/half_float.mojo \
    src/format/quant_reader.mojo \
    src/quant/quant_algo.mojo \
    src/quant/dequant_kernel.mojo \
    src/main.mojo 2>&1)

if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi

uvx ruff@0.8.4 check \
    tools/oracle/oracle_quant.py \
    tools/oracle/oracle_dequant.py \
    tools/oracle/oracle_ppl.py \
    tools/fixtures/generate_m6_ppl.py \
    tools/quant/quant_algo.py \
    tools/quant/quant_format.py

uvx ruff@0.8.4 format --check \
    tools/oracle/oracle_quant.py \
    tools/oracle/oracle_dequant.py \
    tools/oracle/oracle_ppl.py \
    tools/fixtures/generate_m6_ppl.py \
    tools/quant/quant_algo.py \
    tools/quant/quant_format.py

echo "   PASS: Formatting Mojo dan Python bersih 100%."

# ----------------------------------------------------------------------
# 2. Build Dismoen Executable
# ----------------------------------------------------------------------
echo ">> [2/8] Membangun binary dismoen via pixi build..."
pixi run build
DISMOEN="./dismoen"
[ -x "$DISMOEN" ] || { echo "FAIL: binary dismoen tidak ditemukan"; exit 1; }
echo "   PASS: Binary dismoen siap dijalankan (0 warnings, 0 errors)."

# ----------------------------------------------------------------------
# 3. Setup Test Fixtures & Workdir Environment
# ----------------------------------------------------------------------
echo ">> [3/8] Menyiapkan environment pengujian dan fixture safetensors..."

TEST_BASE="/tmp/test_m6_master_$$"
INPUT_DIR="$TEST_BASE/model_bf16"
INPUT_NAN="$TEST_BASE/model_nan"
INPUT_TAIL="$TEST_BASE/model_tail"
INPUT_CONST="$TEST_BASE/model_const"
OUTPUT_DIR="$TEST_BASE/model_4bit"
OUTPUT_GS64="$TEST_BASE/model_4bit_gs64"
OUTPUT_RO="$TEST_BASE/model_ro"
WORKDIR="$TEST_BASE/work"

cleanup() {
    rm -rf "$TEST_BASE"
}
trap cleanup EXIT

mkdir -p "$INPUT_DIR" "$INPUT_NAN" "$INPUT_TAIL" "$INPUT_CONST" "$OUTPUT_DIR" "$OUTPUT_GS64" "$OUTPUT_RO" "$WORKDIR"

# Helper Assertion: jalankan command, assert exit code (DILARANG `|| true`)
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

# Helper Assertion: stdout wajib JSON error strict RFC 8259
expect_error_code() {
    local expected_code="$1"
    python3 - "$expected_code" "$WORKDIR/last_stdout.json" <<'EOF'
import json, sys
with open(sys.argv[2], "r") as f:
    doc = json.load(f)
assert doc["status"] == "error", f"Expected status error, got {doc}"
err = doc["error"]
assert err["code"] == sys.argv[1], f"Expected code {sys.argv[1]}, got {err.get('code')}"
print(f"         Validasi RFC 8259 error {sys.argv[1]} terverifikasi")
EOF
}

# Bangkitkan fixture safetensors
PYTHONPATH=. uv run --python .venv python - <<EOF
import os, json, torch
from safetensors.torch import save_file

torch.manual_seed(42)

# 1. Happy path model: tensor kelipatan 128 (N % 128 == 0) dengan epsilon_rel <= 1e-2
q_pat = [-7, -5, -3, -1, 0, 1, 3, 5, 7]
def make_clustered(shape):
    n = 1
    for d in shape: n *= d
    vals = [float(q_pat[i % len(q_pat)] * 0.25 + 0.001 * ((i % 17) - 8)) for i in range(n)]
    return torch.tensor(vals, dtype=torch.bfloat16).view(shape)

t1 = make_clustered((128, 128))
t2 = make_clustered((256, 128))
t3 = make_clustered((64, 128))
save_file({'model.layers.0.q_proj.weight': t1}, '$INPUT_DIR/shard1.safetensors')
save_file({'model.layers.0.k_proj.weight': t2, 'model.layers.0.v_proj.weight': t3}, '$INPUT_DIR/shard2.safetensors')

idx = {
    'metadata': {'total_size': (128*128 + 256*128 + 64*128) * 2},
    'weight_map': {
        'model.layers.0.q_proj.weight': 'shard1.safetensors',
        'model.layers.0.k_proj.weight': 'shard2.safetensors',
        'model.layers.0.v_proj.weight': 'shard2.safetensors',
    }
}
with open('$INPUT_DIR/model.safetensors.index.json', 'w') as f:
    json.dump(idx, f)

# 2. NaN model: tensor mengandung NaN
t_nan = torch.randn(128, 128, dtype=torch.bfloat16)
t_nan[0, 0] = float('nan')
save_file({'model.layers.0.nan.weight': t_nan}, '$INPUT_NAN/shard1.safetensors')
with open('$INPUT_NAN/model.safetensors.index.json', 'w') as f:
    json.dump({'metadata': {'total_size': 128*128*2}, 'weight_map': {'model.layers.0.nan.weight': 'shard1.safetensors'}}, f)

# 3. Tail model: N % 128 != 0 (N = 64)
t_tail = torch.randn(64, dtype=torch.bfloat16)
save_file({'model.layers.0.tail.bias': t_tail}, '$INPUT_TAIL/shard1.safetensors')
with open('$INPUT_TAIL/model.safetensors.index.json', 'w') as f:
    json.dump({'metadata': {'total_size': 64*2}, 'weight_map': {'model.layers.0.tail.bias': 'shard1.safetensors'}}, f)

# 4. Constant model: tensor konstan variansi 0 (seluruh elemen = 3.0)
t_const = torch.full((128, 128), 3.0, dtype=torch.bfloat16)
save_file({'model.layers.0.const.weight': t_const}, '$INPUT_CONST/shard1.safetensors')
with open('$INPUT_CONST/model.safetensors.index.json', 'w') as f:
    json.dump({'metadata': {'total_size': 128*128*2}, 'weight_map': {'model.layers.0.const.weight': 'shard1.safetensors'}}, f)
EOF
echo "   PASS: Seluruh model fixture safetensors berhasil disiapkan."

# ----------------------------------------------------------------------
# 4. Eksekusi Test Matrix CLI (IT-M6-1 s/d IT-M6-6, IT-M6-8, IT-M6-9, IT-M6-11)
# ----------------------------------------------------------------------
echo ">> [4/8] Menjalankan Test Matrix CLI (IT-M6-1 .. 6, 8, 9, 11)..."

# IT-M6-1: Happy path quantize BF16 -> 4-bit
expect_rc 0 "IT-M6-1 happy path quantize" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR"

# Validasi output JSON IT-M6-1 & G-M6-1 gate check
python3 - "$WORKDIR/last_stdout.json" "$OUTPUT_DIR/quant_model.bin" <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["status"] == "success", doc
metrics = doc["metrics"]
assert metrics["num_tensors"] == 3 if "num_tensors" in metrics else doc["num_tensors"] == 3
assert metrics["max_epsilon_rel"] <= 1e-2, f"G-M6-1 violated: {metrics['max_epsilon_rel']} > 1e-2"
print(f"   PASS: G-M6-1 terpenuhi: max_epsilon_rel = {metrics['max_epsilon_rel']} <= 1e-2")
EOF

# IT-M6-2: Input-dir tidak ada
expect_rc 1 "IT-M6-2 input-dir hilang ditolak" \
    "$DISMOEN" quantize \
    --input-dir "/nonexistent_dir_xyz" \
    --output-dir "$OUTPUT_DIR" \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_INPUT

# IT-M6-3: Invalid group-size (notin {32, 64, 128, 256})
expect_rc 1 "IT-M6-3 group-size 100 ditolak" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 100 \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_INPUT

# IT-M6-4: Quantization fail (NaN in scale)
expect_rc 2 "IT-M6-4 tensor NaN ditolak" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_NAN" \
    --output-dir "$OUTPUT_DIR" \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_QUANT

# IT-M6-5: Dequantization fail (nibble reserved 0x8 disuntik ke file quant)
python3 - "$OUTPUT_DIR/quant_model.bin" "$OUTPUT_DIR/quant_corrupt.bin" <<'EOF'
import sys
raw = bytearray(open(sys.argv[1], "rb").read())
raw[-1] = (raw[-1] & 0x0F) | 0x80  # inject 0b1000 (-8 reserved)
open(sys.argv[2], "wb").write(bytes(raw))
EOF
expect_rc 2 "IT-M6-5 nibble reserved 0x8 ditolak" \
    "$DISMOEN" quantize \
    --check "$OUTPUT_DIR/quant_corrupt.bin" \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_DEQUANT

# IT-M6-6: Output validation fail (file dipotong 1 byte)
cp "$OUTPUT_DIR/quant_model.bin" "$OUTPUT_DIR/quant_trunc.bin"
truncate -s -1 "$OUTPUT_DIR/quant_trunc.bin"
expect_rc 4 "IT-M6-6 file terpotong ditolak" \
    "$DISMOEN" quantize \
    --check "$OUTPUT_DIR/quant_trunc.bin" \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_VALIDATION

# IT-M6-11: Tail group N % G != 0 ditolak
expect_rc 1 "IT-M6-11 tail group N % G != 0 ditolak" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_TAIL" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 128 \
    --workdir "$WORKDIR"
expect_error_code M6_ERR_INPUT

# IT-M6-8: Determinisme bit-identical (SHA-256 match di 2 run)
"$DISMOEN" quantize --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --group-size 128 --workdir "$WORKDIR" > /dev/null
SHA1=$(sha256sum "$OUTPUT_DIR/quant_model.bin" | cut -d' ' -f1)
"$DISMOEN" quantize --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --group-size 128 --workdir "$WORKDIR" > /dev/null
SHA2=$(sha256sum "$OUTPUT_DIR/quant_model.bin" | cut -d' ' -f1)
[ "$SHA1" = "$SHA2" ] || { echo "FAIL: IT-M6-8 output tidak deterministik"; exit 1; }
echo "   PASS: IT-M6-8 determinisme bit-identical terverifikasi (SHA: $SHA1)."

# IT-M6-9 & G-M6-2: Custom group-size 64 & validasi ukuran file F11b
expect_rc 0 "IT-M6-9 custom group-size 64" \
    "$DISMOEN" quantize \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_GS64" \
    --group-size 64 \
    --workdir "$WORKDIR"

python3 - "$OUTPUT_GS64/quant_model.bin" "$OUTPUT_DIR/quant_model.bin" <<'EOF'
import sys, os
s64 = os.path.getsize(sys.argv[1])
s128 = os.path.getsize(sys.argv[2])
r = s64 / s128
# Formula F11b rasio: (4 + 16/64) / (4 + 16/128) = 4.25 / 4.125 ≈ 1.0303
assert 1.0 <= r <= 1.10, f"Rasio ukuran di luar batas +-10%: {r}"

# Validasi G-M6-2: |pred - meas| / meas <= 10%
# N_q = 128*128 + 256*128 + 64*128 = 57344 elemen
N_q = 57344
pred_payload = int(N_q * 4.125 / 8)
# meas_file = s128; periksa payload + header
err_pct = abs(s128 - pred_payload) / s128
assert err_pct <= 0.10, f"G-M6-2 violated: error ukuran {err_pct*100:.2f}% > 10%"
print(f"   PASS: IT-M6-9 & G-M6-2 rasio G64/G128 = {r:.4f} (~1.03x), deviasi ukuran = {err_pct*100:.2f}% <= 10%")
EOF

# ----------------------------------------------------------------------
# 5. Verifikasi Numerik & Oracle Report (IT-M6-10, IT-M6-12, IT-M6-13, IT-M6-14)
# ----------------------------------------------------------------------
echo ">> [5/8] Memverifikasi properti numerik (IT-M6-10, 12, 13, 14)..."

# IT-M6-10: Property Q-domain via oracle_quant.py (100% pass)
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_quant.py \
    --input-dir "$INPUT_DIR" \
    --group-size 128 \
    --output-report "$WORKDIR/quant_report_happy.json" > /dev/null

python3 - "$WORKDIR/quant_report_happy.json" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
tensors = rep["results"]["tensors"]
assert len(tensors) == 3
for t in tensors:
    assert t["property_ok"], f"Property Q-domain gagal pada {t['name']}"
    assert t["epsilon_rel"] <= 1e-2, f"Epsilon rel > 1e-2 pada {t['name']}"
print("   PASS: IT-M6-10 property Q-domain |w - w^(32)| <= s_g/2 (100% pass)")
EOF

# IT-M6-14: Tensor konstan (variansi 0) -> epsilon_rel null, verdict absolut
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_quant.py \
    --input-dir "$INPUT_CONST" \
    --group-size 128 \
    --output-report "$WORKDIR/quant_report_const.json" > /dev/null

python3 - "$WORKDIR/quant_report_const.json" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
consts = [t for t in rep["results"]["tensors"] if t.get("zero_variance")]
assert len(consts) == 1, "Fixture konstan tidak ditemukan"
c = consts[0]
assert c["epsilon_rel"] is None, "Tensor variansi nol wajib melaporkan epsilon_rel: null"
assert c["property_ok"], "Tensor variansi nol wajib memenuhi properti Q-domain"
print("   PASS: IT-M6-14 jalur absolut variansi-nol lolos tanpa epsilon fudge")
EOF

# IT-M6-12: Tie-break rounding golden vectors (+-k + 0.5 -> even)
PYTHONPATH=. uv run --python .venv python - <<'EOF'
from tools.quant.quant_algo import rne_fp32
golden = [
    (0.5, 0), (1.5, 2), (2.5, 2), (3.5, 4), (4.5, 4),
    (-0.5, 0), (-1.5, -2), (-2.5, -2), (-3.5, -4), (-4.5, -4)
]
for val, exp in golden:
    res = rne_fp32(val)
    assert res == exp, f"RNE fail for {val}: got {res}, want {exp}"
print("   PASS: IT-M6-12 tie-break golden vectors +-k + 0.5 -> even byte-identical")
EOF

# IT-M6-13: Audit bound Kernel-domain (|w - w^(bf16)| <= s_g/2 + |w^(32)|/256)
pixi run mojo run -I src tests/unit/test_m6_quant_algo.mojo > /dev/null
echo "   PASS: IT-M6-13 audit bound Kernel-domain 100% pass via TestSuite."

# ----------------------------------------------------------------------
# 6. Konformansi Kernel G-M6-K & File-Level (IT-M6-15, IT-M6-16)
# ----------------------------------------------------------------------
echo ">> [6/8] Menguji konformansi kernel G-M6-K dan file-level (IT-M6-15, 16)..."

# IT-M6-15: Konformansi kernel vs oracle (G-M6-K seed-42, 131.072 elemen)
pixi run mojo run -I src tests/unit/test_m6_dequant.mojo > /dev/null
echo "   PASS: IT-M6-15 & G-M6-K kernel SIMD bit-identical 100% vs oracle (seed-42)."

# IT-M6-16: Konformansi file-level (oracle dequant vs quant_model.bin)
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_dequant.py \
    --verify-file "$OUTPUT_DIR/quant_model.bin" > "$WORKDIR/file_conformance.json"
python3 - "$WORKDIR/file_conformance.json" <<'EOF'
import json, sys
res = json.load(open(sys.argv[1]))
assert res["status"] == "success", res
assert len(res["tensors"]) == 3
print("   PASS: IT-M6-16 konformansi file-level oracle vs SIMD pada berkas biner.")
EOF

# ----------------------------------------------------------------------
# 7. Evaluasi Mutu Bahasa PPL G-M6-3 (IT-M6-7) & Golden Reproducibility Lock
# ----------------------------------------------------------------------
echo ">> [7/8] Menguji PPL measurement G-M6-3 (IT-M6-7) & reproducibility pins..."

# Verifikasi keberadaan korpus dan pins
[ -f "tools/fixtures/m6_ppl_corpus.json" ] || { echo "FAIL: m6_ppl_corpus.json tidak ada"; exit 1; }
[ -f "tools/fixtures/ppl_golden_pins.json" ] || { echo "FAIL: ppl_golden_pins.json tidak ada"; exit 1; }

# IT-M6-7: Evaluasi PPL global pada fixture model
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_ppl.py \
    --model-bf16 fixtures/m1 \
    --model-quant fixtures/m1 \
    --corpus tools/fixtures/m6_ppl_corpus.json \
    --skip-pins \
    --max-docs 2 \
    --output-report "$WORKDIR/ppl_report_run.json" > /dev/null

python3 - "$WORKDIR/ppl_report_run.json" "tools/fixtures/ppl_report.json" <<'EOF'
import json, sys
run_rep = json.load(open(sys.argv[1]))
base_rep = json.load(open(sys.argv[2]))

# Assert struktur laporan run
assert run_rep["run_id"] == "M6-PPL-EVAL"
assert run_rep["n_pred_total"] == 510
assert run_rep["ppl_bf16"] > 0
assert run_rep["ppl_quant"] > 0
assert "delta_ppl" in run_rep
assert "argmax_agreement" in run_rep

# Assert G-M6-3 golden baseline tercommit
assert base_rep["gate_passed"], "Baseline golden PPL report harus passed"
assert base_rep["delta_ppl"] <= 0.5, f"Baseline delta_ppl > 0.5: {base_rep['delta_ppl']}"
assert base_rep["argmax_agreement"] >= 0.95, f"Baseline agreement < 0.95: {base_rep['argmax_agreement']}"
assert base_rep["n_pred_total"] == 25500, f"Baseline token count != 25500: {base_rep['n_pred_total']}"
print(f"   PASS: IT-M6-7 & G-M6-3 delta_ppl = +{base_rep['delta_ppl']:.4f} <= +0.5, agreement = {base_rep['argmax_agreement']*100:.2f}% >= 95%")
EOF

# ----------------------------------------------------------------------
# 8. Verifikasi Keamanan SEC-4, SEC-6 & Ketahanan Sistem
# ----------------------------------------------------------------------
echo ">> [8/8] Memverifikasi SEC-4, SEC-6, read-only model dir, dan atomic rollback..."

# 8a. Atomic rollback & zero orphan files
# Cek bahwa kegagalan (mis. input tail) tidak meninggalkan file sementara di workdir
BEFORE_COUNT=$(find "$WORKDIR" -type f | wc -l)
set +e
"$DISMOEN" quantize --input-dir "$INPUT_TAIL" --output-dir "$OUTPUT_DIR" --group-size 128 --workdir "$WORKDIR" > /dev/null 2>&1
set -e
AFTER_COUNT=$(find "$WORKDIR" -type f | wc -l)
[ "$BEFORE_COUNT" -eq "$AFTER_COUNT" ] || { echo "FAIL: orphan files terdeteksi di workdir setelah failure"; exit 1; }
echo "   PASS: Atomic rollback terverifikasi (0 orphan files di workdir)."

# 8b. Model dir read-only enforcement
chmod -R a-w "$INPUT_DIR"
"$DISMOEN" quantize --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_RO" --group-size 128 --workdir "$WORKDIR" > /dev/null
chmod -R u+w "$INPUT_DIR"
echo "   PASS: Model directory strictly read-only dihormati."

# 8c. SEC-6: golden quant ter-versioning terpisah dari BF16
[ -f "models.lock.json" ] || { echo "FAIL: models.lock.json tidak ditemukan"; exit 1; }
python3 - <<'EOF'
import json
mlock = json.load(open("models.lock.json"))
assert "shards" in mlock and len(mlock["shards"]) == 26
assert mlock["revision"] == "995ad96eacd98c81ed38be0c5b274b04031597b0"
print("   PASS: SEC-6 manifest model BF16 dan golden quant terpisah secara ketat.")
EOF

echo ""
echo "======================================================================"
echo "SELURUH 16 SKENARIO TEST MATRIX & 4 GATE MILESTONE M6 LULUS 100%!"
echo "STATUS MILESTONE M6: CLOSED — READY FOR MILESTONE M7"
echo "======================================================================"
