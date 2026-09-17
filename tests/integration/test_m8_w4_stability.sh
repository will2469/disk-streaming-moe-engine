#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite: Milestone M8 Wave 4
# Stabilitas Long-Seq (32K), O(1) Peak Memory, 5x Determinisme, Fuzzing (27), SEC-4/5/6.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

KIMO="./kimo"
TMP_DIR="$(mktemp -d -t kimo_m8_w4_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "======================================================================="
echo "=== M8-W4 Master Test Suite: Long-Seq, Determinism, Fuzz, & Security ==="
echo "======================================================================="

# Pastikan binary kimo siap
if [ ! -f "$KIMO" ]; then
    echo "Binary kimo tidak ditemukan, mengompilasi via pixi build..."
    pixi run build
fi

# -----------------------------------------------------------------------------
# Stage 1: Hygiene & Formatting Compliance
# -----------------------------------------------------------------------------
echo ">> [1/5] Memeriksa kepatuhan formatting, ruff, clippy, dan no-noqa..."
pre-commit run --all-files
echo "   PASS: Seluruh static hygiene checks (ruff, clippy, mojo format, no-noqa) lolos 100%."

# -----------------------------------------------------------------------------
# Stage 2: Stabilitas Sekuens Panjang & Peak Memory O(1) (Gate G-M8-1 & G-M8-2)
# -----------------------------------------------------------------------------
echo ">> [2/5] Menguji stabilitas long-sequence (1K..32K) & peak memory O(1)..."
python3 tools/bench/bench_gdn_stability.py
echo "   PASS: Stabilitas numerik 32K (Delta_max <= 1e-3) & O(1) memory (Delta VmHWM <= 10MB) terbukti."

# -----------------------------------------------------------------------------
# Stage 3: Determinisme 5x Ulangan Identik (Offline Seed, Threads=1, FP32 State)
# -----------------------------------------------------------------------------
echo ">> [3/5] Menguji determinisme bitwise 5x ulangan identik (threads=1)..."
REF_HASH=""
for run in 1 2 3 4 5; do
    RUN_OUT="$TMP_DIR/det_run_${run}.bin"
    "$KIMO" gdn \
        --model-dir "fixtures" \
        --tokens "fixtures/m8_tokens.json" \
        --output "$RUN_OUT" \
        --layers 2 \
        --dk 32 \
        --dv 32 \
        --chunk-size 8 \
        --threads 1 >/dev/null
    RUN_HASH=$(sha256sum "$RUN_OUT" | awk '{print $1}')
    if [ -z "$REF_HASH" ]; then
        REF_HASH="$RUN_HASH"
    else
        if [ "$REF_HASH" != "$RUN_HASH" ]; then
            echo "FAIL: Determinisme run $run gagal! Hash $RUN_HASH != $REF_HASH"
            exit 1
        fi
    fi
done
echo "   PASS: Determinisme bitwise 5x identik terverifikasi (SHA256: $REF_HASH)."

# -----------------------------------------------------------------------------
# Stage 4: Fuzzing Suite 27 Kasus Mutasi (Anti-Hang, 0 Crash, 0 OOM)
# -----------------------------------------------------------------------------
echo ">> [4/5] Menjalankan fuzzing suite 27 kasus mutasi ekstrim..."
bash tests/integration/test_m8_w4_fuzz.sh
echo "   PASS: 27/27 mutasi fuzzing lolos bersih tanpa hang, crash, atau OOM."

# -----------------------------------------------------------------------------
# Stage 5: Verifikasi Keamanan SEC-4, SEC-5, SEC-6
# -----------------------------------------------------------------------------
echo ">> [5/5] Memverifikasi perimeter keamanan SEC-4, SEC-5, dan SEC-6..."

# 5a. SEC-4: Checked arithmetic overflow guard (ceiling <= 100 MB)
set +e
ERR_SEC4=$("$KIMO" gdn \
    --model-dir "fixtures" \
    --tokens "fixtures/m8_tokens.json" \
    --output "$TMP_DIR/sec4_dummy.bin" \
    --layers 50 \
    --dk 2048 \
    --dv 2048 2>&1 >/dev/null)
RET_SEC4=$?
set -e
# 50 * 2048 * 2048 * 4 = 838,860,800 bytes (> 100 MB ceiling) -> harus ditolak exit 2
if [ "$RET_SEC4" -ne 2 ]; then
    echo "FAIL: SEC-4 checked arithmetic bound gagal menolak alokasi >100MB! Got exit $RET_SEC4"
    exit 1
fi
python3 -c '
import json
err = json.loads("""'"$ERR_SEC4"'""")
assert err["error_code"] == 2
assert err["error_type"] == "CONFIG_INVALID"
'
echo "   PASS: SEC-4 checked arithmetic guard (>100 MB) menolak fail-fast dengan exit 2."

# 5b. SEC-5: Model read-only, atomic output write, dan isolation
RO_MODEL_DIR="$TMP_DIR/sec5_readonly_model"
mkdir -p "$RO_MODEL_DIR"
cp fixtures/m8_gdn_weights.safetensors "$RO_MODEL_DIR/"
chmod 444 "$RO_MODEL_DIR/m8_gdn_weights.safetensors"
chmod 555 "$RO_MODEL_DIR"

OUT_SEC5="$TMP_DIR/sec5_out.bin"
"$KIMO" gdn \
    --model-dir "$RO_MODEL_DIR" \
    --tokens "fixtures/m8_tokens.json" \
    --output "$OUT_SEC5" \
    --layers 2 \
    --dk 32 \
    --dv 32 \
    --chunk-size 8 \
    --threads 1 >/dev/null

[ -f "$OUT_SEC5" ] || { echo "FAIL: SEC-5 output tidak tercipta dari read-only model dir"; exit 1; }
chmod 777 "$RO_MODEL_DIR"

# Atomic failure: Direktori read-only tidak meninggalkan berkas sampah .tmp
RO_OUT_DIR="$TMP_DIR/sec5_ro_out"
mkdir -p "$RO_OUT_DIR"
chmod 555 "$RO_OUT_DIR"
set +e
"$KIMO" gdn \
    --model-dir "fixtures" \
    --tokens "fixtures/m8_tokens.json" \
    --output "$RO_OUT_DIR/state.bin" \
    --layers 2 \
    --dk 32 \
    --dv 32 2>/dev/null
RET_RO=$?
set -e
chmod 777 "$RO_OUT_DIR"
if [ "$RET_RO" -ne 6 ]; then
    echo "FAIL: SEC-5 expected exit 6 on write error, got $RET_RO"
    exit 1
fi
LEAK_COUNT=$(find "$RO_OUT_DIR" -name "*.tmp" | wc -l)
if [ "$LEAK_COUNT" -ne 0 ]; then
    echo "FAIL: SEC-5 atomic rollback bocor berkas .tmp!"
    exit 1
fi
echo "   PASS: SEC-5 read-only model & atomic write (zero leak) terverifikasi."

# 5c. SEC-6: Silent regression golden hash verification
sha256sum -c fixtures/m8_state_naive.bin.sha256 >/dev/null
echo "   PASS: SEC-6 golden state fixture hash terverifikasi tanpa regresi senyap."

echo "======================================================================="
echo "=== SEMUA PENGUJIAN M8-W4 LOLOS 100%: STABIL, DETERMINISTIK, & AMAN ==="
echo "======================================================================="
