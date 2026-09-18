#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M9 Wave 4 (M9-W4).
# Menguji:
#   1. Integritas fixture sintetis (<10 MB, 4 layer 3+1, vocab 1024, seed 42)
#   2. Oracle port reference determinisme reproducibility (5x identik)
#   3. GGUF oracle execution (on-demand dequant ke FP32)
#   4. Kontrak compare F10 & evaluasi Gate G-M9-1 (MATCH, FAIL, ERROR)
#   5. G-M9-1 layer-by-layer intermediate activation dump
#   6. Korpus PPL port & golden baseline integrity (100 docs x 256 tokens)
#   7. Code quality & Ruff linter zero warnings

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

COMPARE_BIN="${COMPARE_BIN:-target/debug/dismoen-tools}"
if [ ! -f "$COMPARE_BIN" ]; then
    COMPARE_BIN="target/release/dismoen-tools"
fi

TEST_DIR="/tmp/test_m9_w4_$$"
WORKDIR="$TEST_DIR/work"
DUMP_LAYERS_DIR="$TEST_DIR/layers_dump"
DUMP_ROUTING_DIR="$TEST_DIR/routing_dump"
mkdir -p "$WORKDIR" "$DUMP_LAYERS_DIR" "$DUMP_ROUTING_DIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "=== M9-W4 Master Integration Test Suite: Oracle Port, Fixture Mini, & Korpus PPL ==="

# ---------------------------------------------------------------------------
# 1. Verifikasi Fixture Sintetis & Golden Hash
# ---------------------------------------------------------------------------
echo "--> Test 1: Verifikasi Fixture Sintetis Mini (<10 MB) & SHA-256 Pin"
TOKENS_FILE="fixtures/m9_port_tokens.json"
WEIGHTS_FILE="fixtures/m9_port_weights.safetensors"
WEIGHTS_SHA_FILE="fixtures/m9_port_weights.safetensors.sha256"
LOGITS_FILE="fixtures/m9_port_logits_naive.bin"
LOGITS_SHA_FILE="fixtures/m9_port_logits_naive.bin.sha256"

if [ ! -f "$WEIGHTS_FILE" ] || [ ! -f "$TOKENS_FILE" ]; then
    "$PYTHON" tools/oracle/generate_port_fixture.py \
        --layers 4 --routed-experts 8 --topk 2 --inter-dim 64 --vocab 1024 \
        --dh 32 --gqa-q 4 --gqa-kv 1 --seed 42 \
        --output "$WEIGHTS_FILE" --tokens-output "$TOKENS_FILE"
fi

test -f "$TOKENS_FILE"
test -f "$WEIGHTS_FILE"
test -f "$WEIGHTS_SHA_FILE"
test -f "$LOGITS_FILE"
test -f "$LOGITS_SHA_FILE"

# Verifikasi ukuran weights < 10 MB (DoD M9)
weights_size=$(stat -c %s "$WEIGHTS_FILE")
if [ "$weights_size" -ge 10485760 ]; then
    echo "FAIL: ukuran weights fixture >= 10 MB ($weights_size bytes)"
    exit 1
fi

# Verifikasi ukuran file logits: 8 tokens * 1024 vocab * 4 bytes = 32768 B
logits_size=$(stat -c %s "$LOGITS_FILE")
if [ "$logits_size" -ne 32768 ]; then
    echo "FAIL: ukuran file logits $logits_size != 32768 bytes"
    exit 1
fi

# Verifikasi SHA256 weights & logits
(cd fixtures && sha256sum -c m9_port_weights.safetensors.sha256)
(cd fixtures && sha256sum -c m9_port_logits_naive.bin.sha256)
echo "PASS: Test 1 (Fixture sintetis mini & golden hash valid)"

# ---------------------------------------------------------------------------
# 2. Determinisme Test (5x Run Identik)
# ---------------------------------------------------------------------------
echo "--> Test 2: Uji Reproducibility & Determinisme Oracle Port (5x Ulangan)"
EXPECTED_SHA=$(cut -d ' ' -f 1 "$LOGITS_SHA_FILE")

for run in 1 2 3 4 5; do
    OUT_RUN="$WORKDIR/logits_det_${run}.bin"
    "$PYTHON" tools/oracle/oracle_port.py \
        --tokens "$TOKENS_FILE" \
        --weights "$WEIGHTS_FILE" \
        --architecture qwen3.6 \
        --output "$OUT_RUN" \
        --seed 42 > /dev/null

    RUN_SHA=$(sha256sum "$OUT_RUN" | cut -d ' ' -f 1)
    if [ "$RUN_SHA" != "$EXPECTED_SHA" ]; then
        echo "FAIL: determinisme run $run mismatch! ($RUN_SHA != $EXPECTED_SHA)"
        exit 1
    fi
done
echo "PASS: Test 2 (100% deterministik & bit-exact pada 5x eksekusi)"

# ---------------------------------------------------------------------------
# 3. GGUF Oracle Execution (On-Demand Dequantization Path)
# ---------------------------------------------------------------------------
echo "--> Test 3: Eksekusi Oracle GGUF Dequantization Path"
GGUF_FIXTURE="fixtures/m9_port_mini.gguf"
test -f "$GGUF_FIXTURE"

GGUF_OUT="$WORKDIR/logits_gguf.bin"
"$PYTHON" tools/oracle/oracle_port.py \
    --tokens "$TOKENS_FILE" \
    --weights "$GGUF_FIXTURE" \
    --architecture qwen3.6 \
    --output "$GGUF_OUT" \
    --seed 42 > /dev/null

test -f "$GGUF_OUT"
gguf_size=$(stat -c %s "$GGUF_OUT")
if [ "$gguf_size" -ne 32768 ]; then
    echo "FAIL: ukuran file logits GGUF $gguf_size != 32768 bytes"
    exit 1
fi
echo "PASS: Test 3 (Oracle GGUF dequantization path berhasil menghasilkan logits)"

# ---------------------------------------------------------------------------
# 4. Kontrak Compare F10 (Gate G-M9-1: MATCH, FAIL, ERROR)
# ---------------------------------------------------------------------------
echo "--> Test 4: Verifikasi Kontrak Compare F10 & Gate G-M9-1"
CAND_FILE="$WORKDIR/logits_det_1.bin"

# 4a. Identity Check (MATCH / PASS)
COMPARE_OUT=$("$PYTHON" tools/compare.py \
    --ref "$LOGITS_FILE" \
    --cand "$CAND_FILE" \
    --gate G-M9-1 \
    --dim 1024)

echo "$COMPARE_OUT" | grep -q '"verdict": "PASS"' || {
    echo "FAIL: compare identity gagal menghasilkan verdict PASS"
    exit 1
}
echo "$COMPARE_OUT" | grep -q '"status": "MATCH"' || {
    echo "FAIL: compare identity gagal menghasilkan status MATCH"
    exit 1
}

# Jika dismoen-tools binary tersedia, uji juga via binary
if [ -x "$COMPARE_BIN" ]; then
    BIN_COMPARE=$("$COMPARE_BIN" compare "$LOGITS_FILE" "$CAND_FILE" --gate G-M9-1 --dim 1024)
    echo "$BIN_COMPARE" | grep -q '"verdict": "PASS"' || {
        echo "FAIL: binary compare identity gagal"
        exit 1
    }
fi

# 4b. Perturbation Check (MISMATCH / FAIL, Exit 1)
CORRUPT_CAND="$WORKDIR/logits_corrupt.bin"
cp "$CAND_FILE" "$CORRUPT_CAND"
# Modifikasi beberapa byte di tengah logits
printf "\xFF\xFF\x7F\x7F" | dd of="$CORRUPT_CAND" bs=1 seek=1024 count=4 conv=notrunc status=none

set +e
"$PYTHON" tools/compare.py \
    --ref "$LOGITS_FILE" \
    --cand "$CORRUPT_CAND" \
    --gate G-M9-1 \
    --dim 1024 > "$WORKDIR/corrupt_report.json" 2>&1
FAIL_RC=$?
set -e

if [ "$FAIL_RC" -ne 1 ]; then
    echo "FAIL: deteksi perturbasi harus exit code 1, dapat $FAIL_RC"
    exit 1
fi
grep -q '"status": "MISMATCH"' "$WORKDIR/corrupt_report.json" || {
    echo "FAIL: perturbasi tidak menghasilkan status MISMATCH"
    exit 1
}

# 4c. Non-existent file (ERROR, Exit 2)
set +e
"$PYTHON" tools/compare.py \
    --ref "$LOGITS_FILE" \
    --cand "$WORKDIR/non_existent.bin" \
    --gate G-M9-1 \
    --dim 1024 > "$WORKDIR/missing_report.json" 2>&1
ERR_RC=$?
set -e

if [ "$ERR_RC" -ne 2 ]; then
    echo "FAIL: berkas tidak ditemukan harus exit code 2, dapat $ERR_RC"
    exit 1
fi
echo "PASS: Test 4 (Kontrak compare F10: MATCH=0, FAIL=1, ERROR=2 terverifikasi)"

# ---------------------------------------------------------------------------
# 5. G-M9-1 Layer-by-Layer Activation Dump
# ---------------------------------------------------------------------------
echo "--> Test 5: Verifikasi G-M9-1 Layer-by-Layer Intermediate Activation Dump"
"$PYTHON" tools/oracle/oracle_port.py \
    --tokens "$TOKENS_FILE" \
    --weights "$WEIGHTS_FILE" \
    --architecture qwen3.6 \
    --output "$WORKDIR/logits_dump_test.bin" \
    --dump-layers "$DUMP_LAYERS_DIR" \
    --dump-routing "$DUMP_ROUTING_DIR" \
    --seed 42 > /dev/null

# Verifikasi embedding & final
test -s "$DUMP_LAYERS_DIR/00_embedding.bin"
test -s "$DUMP_LAYERS_DIR/98_final_norm.bin"
test -s "$DUMP_LAYERS_DIR/99_logits.bin"

# Verifikasi 4 layer blok (0..3)
for l in 0 1 2 3; do
    test -s "$DUMP_LAYERS_DIR/block_${l}_01_input.bin"
    test -s "$DUMP_LAYERS_DIR/block_${l}_02_mixer_norm.bin"
    test -s "$DUMP_LAYERS_DIR/block_${l}_03_mixer_out.bin"
    test -s "$DUMP_LAYERS_DIR/block_${l}_04_post_mixer.bin"
    test -s "$DUMP_LAYERS_DIR/block_${l}_05_post_norm.bin"
    test -s "$DUMP_LAYERS_DIR/block_${l}_06_router_logits.bin"
    test -s "$DUMP_LAYERS_DIR/block_${l}_07_topk_indices.bin"
    test -s "$DUMP_LAYERS_DIR/block_${l}_08_moe_out.bin"
    test -s "$DUMP_LAYERS_DIR/block_${l}_09_block_out.bin"

    if [ "$l" -ne 3 ]; then
        test -s "$DUMP_LAYERS_DIR/block_${l}_gdn_state.bin"
    else
        test -s "$DUMP_LAYERS_DIR/block_${l}_kv_cache.bin"
    fi
    test -s "$DUMP_ROUTING_DIR/routing_L${l}.json"
done

# Verifikasi ukuran salah satu aktivasi [8, 128] * 4 B = 4096 B
act_size=$(stat -c %s "$DUMP_LAYERS_DIR/block_0_01_input.bin")
if [ "$act_size" -ne 4096 ]; then
    echo "FAIL: ukuran aktivasi $act_size != 4096 bytes"
    exit 1
fi
echo "PASS: Test 5 (G-M9-1 layer-by-layer intermediate dump lengkap & valid)"

# ---------------------------------------------------------------------------
# 6. Korpus PPL Port & Baseline Integrity
# ---------------------------------------------------------------------------
echo "--> Test 6: Verifikasi Korpus PPL Port & Golden Baseline"
PPL_CORPUS="fixtures/m9_ppl_corpus.json"
PPL_PINS="fixtures/m9_ppl_golden_pins.json"
PPL_BF16_BASE="fixtures/m9_ppl_bf16_baseline.json"
PPL_QUANT_BASE="fixtures/m9_ppl_quant_baseline.json"
PPL_DELTA="fixtures/m9_delta_ppl.json"

test -f "$PPL_CORPUS"
test -f "$PPL_PINS"
test -f "$PPL_BF16_BASE"
test -f "$PPL_QUANT_BASE"
test -f "$PPL_DELTA"

"$PYTHON" -c "
import json
with open('$PPL_CORPUS') as f:
    corpus = json.load(f)
assert len(corpus['corpus']) == 100, f'Expected 100 docs, got {len(corpus[\"corpus\"])}'
for d in corpus['corpus']:
    assert len(d['token_ids']) == 256, f'Doc {d[\"id\"]} length != 256'
    for t in d['token_ids']:
        assert 0 <= t < 248320, f'Token {t} out of range [0, 248320)'

with open('$PPL_DELTA') as f:
    delta = json.load(f)
assert delta['global_delta_ppl'] <= 1.0, f'Delta PPL {delta[\"global_delta_ppl\"]} > 1.0'
assert delta['global_argmax_agreement'] >= 0.95, f'Agreement {delta[\"global_argmax_agreement\"]} < 0.95'
print('   PASS: Korpus 100 docs x 256 tokens valid dan baseline delta PPL memenuhi ambang batas')
"
echo "PASS: Test 6 (Korpus PPL port & baseline integrity terverifikasi)"

# ---------------------------------------------------------------------------
# 7. Code Quality & Ruff Linter
# ---------------------------------------------------------------------------
echo "--> Test 7: Code Quality & Ruff Linter (Zero Warnings)"
uvx ruff@0.8.4 check \
    tools/oracle/generate_port_fixture.py \
    tools/oracle/oracle_port.py \
    tools/fixtures/generate_m9_ppl.py

uvx ruff@0.8.4 format --check \
    tools/oracle/generate_port_fixture.py \
    tools/oracle/oracle_port.py \
    tools/fixtures/generate_m9_ppl.py

echo "PASS: Test 7 (Ruff check & format lulus 100% tanpa warning)"

echo "=== Seluruh 7 Pengujian M9-W4 Oracle Port & Fixture Mini Lulus 100%! ==="
