#!/bin/bash
# ==============================================================================
# test_m4_w3_oracle.sh — Integration Test Suite M4-W3 (Oracle & Golden Set)
#
# Memverifikasi DoD M4-W3:
# 1. Dataset golden tools/fixtures/m4_golden.json terverifikasi utuh dengan
#    tools/fixtures/m4_golden.sha256 (5 prompt x 16 token, range [0, 151936)).
# 2. File individual tools/fixtures/m4_prompt{1..5}_tokens.json valid & match.
# 3. Artefak oracle tools/fixtures/m4_prompt{1..5}_oracle.bin (tepat 9.723.904 B)
#    dan hash SHA-256 terpin (.sha256).
# 4. Routing dump Tier-1 untuk ke-5 prompt: direktori m4_prompt{1..5}_routing
#    berisi tepat 24 file routing_L0..23.json format {"selected_experts": [[4 ID] x 16]}.
# 5. Ekivalensi numerik G-M4-1: dismoen forward (Mojo) vs oracle_full (PyTorch)
#    menghasilkan MATCH pada loose gate G-M4-1 dan SET equality routing Tier-1.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-./dismoen}"
COMPARE_BIN="${COMPARE_BIN:-target/debug/dismoen-tools}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.6-35b-a3b}"
FIXTURE_DIR="tools/fixtures"

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "SKIP: model directory not found: $MODEL_DIR"
    exit 0
fi

if [ ! -f "$COMPARE_BIN" ]; then
    echo ">> Membangun dismoen-tools..."
    cargo build --manifest-path tools/dismoen-tools/Cargo.toml --bin dismoen-tools
fi

if [ ! -f "$DISMOEN" ]; then
    echo ">> Membangun engine dismoen..."
    pixi run build
fi

TEST_DIR="/tmp/test_m4_w3_oracle_$$"
WORKDIR="$TEST_DIR/work"
CAND_ROUTING="$TEST_DIR/routing"
OUT_LOGITS="$WORKDIR/mojo_prompt1_logits.bin"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
mkdir -p "$WORKDIR" "$CAND_ROUTING"

echo "======================================================================"
echo "M4-W3: Oracle Reference, Golden Prompts, and F10 Gate Verification"
echo "======================================================================"

# ----------------------------------------------------------------------
# 1. Validasi m4_golden.json dan SHA-256
# ----------------------------------------------------------------------
echo ">> [1/5] Verifikasi hash dan struktur tools/fixtures/m4_golden.json..."
[ -f "$FIXTURE_DIR/m4_golden.json" ] || { echo "FAIL: $FIXTURE_DIR/m4_golden.json tidak ditemukan"; exit 1; }
[ -f "$FIXTURE_DIR/m4_golden.sha256" ] || { echo "FAIL: $FIXTURE_DIR/m4_golden.sha256 tidak ditemukan"; exit 1; }

(cd "$FIXTURE_DIR" && sha256sum -c m4_golden.sha256) || {
    echo "FAIL: Checksum mismatch pada m4_golden.json!"
    exit 1
}

python3 -c "
import json
with open('$FIXTURE_DIR/m4_golden.json', 'r') as f:
    data = json.load(f)
assert data['name'] == 'M4 golden set'
prompts = data['prompts']
assert len(prompts) == 5, f'Expected 5 prompts, got {len(prompts)}'
for i, p in enumerate(prompts, start=1):
    assert p['id'] == f'prompt{i}'
    tokens = p['tokens']
    assert len(tokens) == 16, f'Prompt {p[\"id\"]} length != 16'
    for t in tokens:
        assert isinstance(t, int) and 0 <= t < 151936, f'Invalid token {t}'
print('   PASS: m4_golden.json valid (5 prompts x 16 tokens dalam range [0, 151936))')
"

# ----------------------------------------------------------------------
# 2. Validasi File Token Individual
# ----------------------------------------------------------------------
echo ">> [2/5] Verifikasi individual tokens JSON m4_prompt{1..5}_tokens.json..."
for i in {1..5}; do
    tok_file="$FIXTURE_DIR/m4_prompt${i}_tokens.json"
    [ -f "$tok_file" ] || { echo "FAIL: $tok_file tidak ditemukan"; exit 1; }
    python3 -c "
import json
with open('$tok_file') as f:
    t = json.load(f)
assert isinstance(t, list) and len(t) == 16, 'Tokens must be list of 16 ints'
"
done
echo "   PASS: 5 file token individual valid dan siap dikonsumsi CLI"

# ----------------------------------------------------------------------
# 3. Validasi Artefak Golden Oracle Biner dan Checksum
# ----------------------------------------------------------------------
echo ">> [3/5] Verifikasi artefak binary oracle m4_prompt{1..5}_oracle.bin..."
EXPECTED_BYTES=$((16 * 151936 * 4)) # 9.723.904 bytes

for i in {1..5}; do
    bin_file="$FIXTURE_DIR/m4_prompt${i}_oracle.bin"
    sha_file="$FIXTURE_DIR/m4_prompt${i}_oracle.bin.sha256"
    [ -f "$bin_file" ] || { echo "FAIL: $bin_file tidak ditemukan"; exit 1; }
    [ -f "$sha_file" ] || { echo "FAIL: $sha_file tidak ditemukan"; exit 1; }

    sz=$(stat -c%s "$bin_file")
    [ "$sz" -eq "$EXPECTED_BYTES" ] || {
        echo "FAIL: $bin_file ukuran $sz != $EXPECTED_BYTES byte!"
        exit 1
    }

    (cd "$FIXTURE_DIR" && sha256sum -c "m4_prompt${i}_oracle.bin.sha256") || {
        echo "FAIL: Checksum mismatch pada $bin_file!"
        exit 1
    }
done
echo "   PASS: 5 binary oracle terverifikasi ukuran ($EXPECTED_BYTES byte) & SHA-256"

# ----------------------------------------------------------------------
# 4. Validasi Routing Dump Tier-1 untuk 5 Prompt
# ----------------------------------------------------------------------
echo ">> [4/5] Verifikasi routing dumps m4_prompt{1..5}_routing/routing_L*.json..."
for i in {1..5}; do
    rout_dir="$FIXTURE_DIR/m4_prompt${i}_routing"
    [ -d "$rout_dir" ] || { echo "FAIL: $rout_dir tidak ditemukan"; exit 1; }
    count=$(find "$rout_dir" -name "routing_L*.json" | wc -l)
    [ "$count" -eq 24 ] || {
        echo "FAIL: $rout_dir memuat $count file (harus tepat 24 layer)!"
        exit 1
    }
    python3 -c "
import json, os
for l in range(24):
    p = os.path.join('$rout_dir', f'routing_L{l}.json')
    assert os.path.exists(p)
    with open(p) as f:
        d = json.load(f)
    assert 'selected_experts' in d
    assert len(d['selected_experts']) == 16
    for row in d['selected_experts']:
        assert len(row) == 4
"
done
echo "   PASS: 5 prompt memiliki tepat 24 routing dump Tier-1 terstruktur"

# ----------------------------------------------------------------------
# 5. Ekivalensi Numerik F10 (Gate G-M4-1) & Routing Equality Check
# ----------------------------------------------------------------------
echo ">> [5/5] Menjalankan forward engine Mojo pada Prompt 1 dan mengevaluasi Gate G-M4-1..."
"$DISMOEN" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$OUT_LOGITS" \
    --workdir "$WORKDIR" \
    --dump-routing "$CAND_ROUTING" > "$WORKDIR/forward_stdout.json"

[ -f "$OUT_LOGITS" ] || { echo "FAIL: Output logits tidak tercipta"; exit 1; }

echo ">> Membandingkan Mojo vs PyTorch Oracle via dismoen-tools compare (Gate G-M4-1)..."
COMPARE_REPORT=$("$COMPARE_BIN" compare \
    --ref "$FIXTURE_DIR/m4_prompt1_oracle.bin" \
    --cand "$OUT_LOGITS" \
    --gate G-M4-1 \
    --dim 151936)

echo "$COMPARE_REPORT"

VERDICT=$(echo "$COMPARE_REPORT" | grep -o '"verdict": "[^"]*"' | head -n1 | cut -d'"' -f4)
[ "$VERDICT" = "PASS" ] || {
    echo "FAIL: Gate G-M4-1 menghasilkan verdict $VERDICT (harus PASS)!"
    exit 1
}

echo ">> Verifikasi SET equality routing dump Tier-1 pada layer 0, 12, dan 23..."
for l in 0 12 23; do
    "$COMPARE_BIN" compare \
        --ref "$FIXTURE_DIR/m4_prompt1_oracle.bin" \
        --cand "$OUT_LOGITS" \
        --gate G-M4-1 \
        --dim 151936 \
        --oracle-routing "$FIXTURE_DIR/m4_prompt1_routing/routing_L${l}.json" \
        --cand-routing "$CAND_ROUTING/routing_L${l}.json" >/dev/null || {
            echo "FAIL: Routing Tier-1 mismatch pada layer $l!"
            exit 1
        }
done
echo "   PASS: Routing SET equality lolos pada layer 0, 12, 23"

echo "======================================================================"
echo "SEMUA SUITE M4-W3 (ORACLE & GOLDEN SET) SUKSES (PASS)"
echo "======================================================================"
exit 0
