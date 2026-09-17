#!/bin/bash
# ==============================================================================
# test_m6_w5_oracle_ppl.sh — Integration Test Suite M6-W5 (Oracle Roundtrip & PPL)
#
# Memverifikasi DoD M6-W5:
# 1. oracle_quant.py: roundtrip per tensor, epsilon_rel, property Q-domain 100%,
#    jalur absolut variansi-nol (epsilon_rel: null, zero_variance: true),
#    penolakan tail group dan invalid group-size, serta report JSON.
# 2. tools/fixtures/generate_m6_ppl.py & m6_ppl_corpus.json:
#    100 dokumen x tepat 256 token ID, token < 151.936, no pad/truncate,
#    ppl_golden_pins.json & baseline artifacts terverifikasi.
# 3. kimo quantize --error-report:
#    per-tensor error breakdown (epsilon_rel, mse, property_ok, max_abs_error,
#    num_groups, summary).
# 4. oracle_ppl.py:
#    verifikasi proteksi golden pins (pin mismatch -> status INVALID),
#    evaluasi PPL global dan argmax agreement pada N_pred_total token.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "M6-W5: Oracle Roundtrip + Korpus PPL + Baseline Evaluation"
echo "======================================================================"

# ----------------------------------------------------------------------
# 1. Formatting & Code Quality
# ----------------------------------------------------------------------
echo ">> [1/6] Memeriksa formatting Mojo dan Python..."

FORMAT_OUTPUT=$(pixi run mojo format \
    src/cli/cmd_quantize.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi

uvx ruff@0.8.4 check \
    tools/oracle/oracle_quant.py \
    tools/oracle/oracle_ppl.py \
    tools/fixtures/generate_m6_ppl.py

uvx ruff@0.8.4 format --check \
    tools/oracle/oracle_quant.py \
    tools/oracle/oracle_ppl.py \
    tools/fixtures/generate_m6_ppl.py

echo "   PASS: Formatting Mojo dan Python bersih 100%."

# ----------------------------------------------------------------------
# 2. Build Kimo Executable
# ----------------------------------------------------------------------
echo ">> [2/6] Membangun binary kimo via pixi build..."
pixi run build
KIMO="./kimo"
[ -x "$KIMO" ] || { echo "FAIL: binary kimo tidak ditemukan"; exit 1; }
echo "   PASS: Binary kimo siap dijalankan."

# ----------------------------------------------------------------------
# 3. Setup Test Fixtures & Working Directories
# ----------------------------------------------------------------------
echo ">> [3/6] Menyiapkan fixture model untuk pengujian oracle..."

TEST_BASE="/tmp/test_m6_w5_$$"
INPUT_HAPPY="$TEST_BASE/model_happy"
INPUT_TAIL="$TEST_BASE/model_tail"
INPUT_CONST="$TEST_BASE/model_const"
OUTPUT_DIR="$TEST_BASE/model_4bit"
WORKDIR="$TEST_BASE/work"

cleanup() {
    rm -rf "$TEST_BASE"
}
trap cleanup EXIT

mkdir -p "$INPUT_HAPPY" "$INPUT_TAIL" "$INPUT_CONST" "$OUTPUT_DIR" "$WORKDIR"

PYTHONPATH=. uv run --python .venv python -c "
import json
import torch
from safetensors.torch import save_file

# 1. Happy model (kelipatan 128)
t1 = torch.randn(128, 128, dtype=torch.bfloat16) * 0.5
t2 = torch.randn(256, 128, dtype=torch.bfloat16) * 0.25
save_file({'layer.0.weight': t1, 'layer.1.weight': t2}, '$INPUT_HAPPY/model.safetensors')

# 2. Tail model (N % 128 != 0, misal N = 64)
t_tail = torch.randn(64, dtype=torch.bfloat16)
save_file({'layer.tail.weight': t_tail}, '$INPUT_TAIL/model.safetensors')

# 3. Constant model (variansi 0)
t_const = torch.full((128, 128), 3.0, dtype=torch.bfloat16)
save_file({'layer.const.weight': t_const}, '$INPUT_CONST/model.safetensors')
"
echo "   PASS: Fixture model (happy, tail, const) berhasil dibangkitkan."

# ----------------------------------------------------------------------
# 4. Pengujian oracle_quant.py
# ----------------------------------------------------------------------
echo ">> [4/6] Menguji tools/oracle/oracle_quant.py..."

# 4a. Happy path roundtrip
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_quant.py \
    --input-dir "$INPUT_HAPPY" \
    --group-size 128 \
    --output-report "$WORKDIR/oracle_happy_report.json" > /dev/null

PYTHONPATH=. uv run --python .venv python - "$WORKDIR/oracle_happy_report.json" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
assert rep["status"] if "status" in rep else True
assert rep["num_tensors"] == 2
assert len(rep["results"]["tensors"]) == 2
for t in rep["results"]["tensors"]:
    assert t["property_ok"], f"Property failed on {t['name']}"
    assert t["epsilon_rel"] is not None and t["epsilon_rel"] > 0
assert rep["results"]["max_epsilon_rel"] > 0
print("   PASS: Happy path roundtrip & property Q-domain 100%")
EOF

# 4b. Jalur absolut variansi-nol (IT-M6-14)
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_quant.py \
    --input-dir "$INPUT_CONST" \
    --group-size 128 \
    --output-report "$WORKDIR/oracle_const_report.json" > /dev/null

PYTHONPATH=. uv run --python .venv python - "$WORKDIR/oracle_const_report.json" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
consts = [t for t in rep["results"]["tensors"] if t.get("zero_variance")]
assert len(consts) == 1, "Fixture konstan tidak terdeteksi"
assert consts[0]["epsilon_rel"] is None, "epsilon_rel harus None untuk tensor variansi nol"
assert consts[0]["property_ok"], "Property Q-domain harus terpenuhi pada jalur absolut"
print("   PASS: IT-M6-14 jalur absolut variansi nol lulus (epsilon_rel: null)")
EOF

# 4c. Tail group ditolak (exit 1, M6_ERR_INPUT)
set +e
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_quant.py \
    --input-dir "$INPUT_TAIL" \
    --group-size 128 \
    --output-report "$WORKDIR/oracle_tail.json" 2>"$WORKDIR/tail_err.json"
RC_TAIL=$?
set -e
[ "$RC_TAIL" -eq 1 ] || { echo "FAIL: tail group want exit 1, got $RC_TAIL"; exit 1; }
grep -q "M6_ERR_INPUT" "$WORKDIR/tail_err.json" || { echo "FAIL: error code must be M6_ERR_INPUT"; exit 1; }
echo "   PASS: Tail group ditolak exit 1 (M6_ERR_INPUT)."

# 4d. Group size tidak valid ditolak (exit 1, M6_ERR_INPUT)
set +e
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_quant.py \
    --input-dir "$INPUT_HAPPY" \
    --group-size 50 \
    --output-report "$WORKDIR/oracle_gs50.json" 2>"$WORKDIR/gs50_err.json"
RC_GS=$?
set -e
[ "$RC_GS" -eq 1 ] || { echo "FAIL: group-size 50 want exit 1, got $RC_GS"; exit 1; }
grep -q "M6_ERR_INPUT" "$WORKDIR/gs50_err.json" || { echo "FAIL: error code must be M6_ERR_INPUT"; exit 1; }
echo "   PASS: Group size tidak valid ditolak exit 1 (M6_ERR_INPUT)."

# ----------------------------------------------------------------------
# 5. Pengujian Korpus PPL & Golden Pins
# ----------------------------------------------------------------------
echo ">> [5/6] Memverifikasi integritas korpus PPL dan golden pins..."

# Verifikasi m6_ppl_corpus.json dan ppl_golden_pins.json
PYTHONPATH=. uv run --python .venv python - <<'EOF'
import json, hashlib

corpus_path = "tools/fixtures/m6_ppl_corpus.json"
pins_path = "tools/fixtures/ppl_golden_pins.json"

with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)

docs = corpus.get("corpus", [])
assert len(docs) == 100, f"Expected 100 documents, got {len(docs)}"

for d in docs:
    tids = d["token_ids"]
    assert len(tids) == 256, f"Doc {d['id']} length != 256"
    assert all(0 <= tid < 151936 for tid in tids), f"Doc {d['id']} token out of bounds"

# Check hash
h = hashlib.sha256()
with open(corpus_path, "rb") as f:
    while chunk := f.read(65536):
        h.update(chunk)
corpus_sha = h.hexdigest()

with open(pins_path, "r", encoding="utf-8") as f:
    pins = json.load(f)

assert pins["corpus"]["sha256"] == corpus_sha, "Pins corpus SHA mismatch!"
assert pins["corpus"]["num_documents"] == 100
assert pins["corpus"]["tokens_per_doc"] == 256
assert pins["corpus"]["n_pred_total"] == 25500

# Verifikasi baseline artifacts
for f_base in ["ppl_report.json", "ppl_bf16_baseline.json", "ppl_quant_baseline.json", "delta_ppl.json"]:
    with open(f"tools/fixtures/{f_base}", "r", encoding="utf-8") as bf:
        b_data = json.load(bf)
        assert b_data, f"Baseline {f_base} empty"

print("   PASS: Korpus 100x256, golden pins, dan seluruh baseline artifacts valid.")
EOF

# ----------------------------------------------------------------------
# 6. Pengujian kimo quantize --error-report & oracle_ppl.py
# ----------------------------------------------------------------------
echo ">> [6/6] Menguji kimo quantize --error-report dan oracle_ppl.py..."

# 6a. kimo quantize dengan --error-report
"$KIMO" quantize \
    --input-dir fixtures/m1 \
    --output-dir "$OUTPUT_DIR" \
    --group-size 64 \
    --error-report "$WORKDIR/kimo_error_report.json" \
    --workdir "$WORKDIR" > "$WORKDIR/quant_stdout.json"

PYTHONPATH=. uv run --python .venv python - "$WORKDIR/kimo_error_report.json" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
assert rep["run_id"] == "M6-QUANT-CLI"
summary = rep["summary"]
assert summary["num_tensors"] == 19
assert summary["max_epsilon_rel"] > 0
assert summary["min_epsilon_rel"] > 0
assert "num_high_error" in summary

tensors = rep["tensors"]
assert len(tensors) == 19
for t in tensors:
    assert "name" in t and "shape" in t
    assert "epsilon_rel" in t
    assert "mse" in t
    assert t["property_ok"], f"Property Q-domain failed on {t['name']}"
    assert "max_abs_error" in t
    assert "num_groups" in t
print("   PASS: kimo quantize --error-report menghasilkan schema per-tensor lengkap.")
EOF

# 6b. oracle_ppl.py: proteksi golden pins (pin mismatch -> status INVALID)
set +e
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_ppl.py \
    --model-bf16 fixtures/m1 \
    --model-quant "$OUTPUT_DIR" \
    --corpus tools/fixtures/m6_ppl_corpus.json \
    --golden-pins tools/fixtures/m4_golden.json \
    --output-report "$WORKDIR/invalid_pins_report.json" 2>"$WORKDIR/invalid_pins_err.log"
RC_INVALID=$?
set -e
[ "$RC_INVALID" -ne 0 ] || { echo "FAIL: pin mismatch want non-zero exit, got 0"; exit 1; }

PYTHONPATH=. uv run --python .venv python - "$WORKDIR/invalid_pins_report.json" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
assert rep["status"] == "INVALID", f"Expected status INVALID, got {rep['status']}"
assert rep["reason"] == "corpus_sha256_mismatch"
print("   PASS: Pin mismatch memicu status INVALID (bukan FAIL).")
EOF

# 6c. oracle_ppl.py: evaluasi model PPL dan konformansi
PYTHONPATH=. uv run --python .venv python tools/oracle/oracle_ppl.py \
    --model-bf16 fixtures/m1 \
    --model-quant "$OUTPUT_DIR" \
    --corpus tools/fixtures/m6_ppl_corpus.json \
    --skip-pins \
    --max-docs 2 \
    --output-report "$WORKDIR/ppl_eval_report.json" > "$WORKDIR/ppl_eval_stdout.json"

PYTHONPATH=. uv run --python .venv python - "$WORKDIR/ppl_eval_report.json" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
assert rep["run_id"] == "M6-PPL-EVAL"
assert rep["n_pred_total"] == 510  # 2 doc x 255
assert rep["ppl_bf16"] > 0
assert rep["ppl_quant"] > 0
assert "delta_ppl" in rep
assert "argmax_agreement" in rep
assert len(rep["diagnostics"]) == 2
for d in rep["diagnostics"]:
    assert d["n_pred"] == 255
    assert d["ppl_bf16"] > 0
    assert d["ppl_quant"] > 0
print("   PASS: Evaluasi PPL global dan diagnostik per dokumen bekerja 100%.")
EOF

echo "======================================================================"
echo "SEMUA PENGUJIAN M6-W5 LULUS 100%!"
echo "======================================================================"
