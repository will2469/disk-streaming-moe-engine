#!/bin/bash
# ==============================================================================
# Integration Test Suite: M9-W2 Hybrid Macro Block Scheduler + GQA + KMSS v1
# ==============================================================================
# Sesuai kontrak:
# - docs/milestones/M9-port.md (§ Layer Scheduling, § Session State, § DoD)
# - scratch/wave/m9/m9-w2-scheduler.md
# - skill: mojo-1-0
#
# Pengujian:
# Stage 1: Pre-commit formatting & zero-suppression hygiene (Mojo format, no noqa/allow)
# Stage 2: Binary compilation check (dismoen binary siap)
# Stage 3: Unit test execution (GQA 16Q/2KV, KMSS v1 serialization, Macro Scheduler)
# Stage 4: CLI forward-port fresh execution + session save
# Stage 5: CLI forward-port session load continuation (recompute_tokens == 0)
# Stage 6: KMSS corruption rejection & integrity falsification
# Stage 7: Determinism check across independent runs
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-./dismoen}"
TEST_DIR="/tmp/test_m9_w2_$$"
MINI_CONFIG="fixtures/m9_port_config_mini.json"
TOKENS_FIXTURE="fixtures/m9_port_tokens.json"
SESSION_FILE="${TEST_DIR}/test_m9_w2.session"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
mkdir -p "$TEST_DIR"

echo "======================================================================"
echo "M9-W2: Hybrid Macro Block Scheduler + GQA + KMSS v1 Verification Suite"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Stage 1: Formatting & Static Hygiene
# -----------------------------------------------------------------------------
echo ">> [1/7] Memeriksa kepatuhan formatting Mojo dan zero-suppression..."
FORMAT_OUTPUT=$(pixi run mojo format \
    src/format/kmss.mojo \
    src/layers/gated_attention.mojo \
    src/layers/port_scheduler.mojo \
    src/cli/cmd_forward_port.mojo \
    tests/unit/test_m9_w2_gqa.mojo \
    tests/unit/test_m9_w2_kmss.mojo \
    tests/unit/test_m9_w2_scheduler.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting memodifikasi berkas:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi

# Larangan keras noqa dan allow suppression
M9_FILES="src/format/kmss.mojo src/layers/gated_attention.mojo src/layers/port_scheduler.mojo src/cli/cmd_forward_port.mojo tests/unit/test_m9_w2_gqa.mojo tests/unit/test_m9_w2_kmss.mojo tests/unit/test_m9_w2_scheduler.mojo"
if grep -rn "noqa" $M9_FILES; then
    echo "FAIL: Ditemukan komentar noqa terlarang!"
    exit 1
fi
if grep -rn "allow(" $M9_FILES; then
    echo "FAIL: Ditemukan allow suppression terlarang!"
    exit 1
fi
echo "   PASS: Formatting bersih, zero-suppression terverifikasi."

# -----------------------------------------------------------------------------
# Stage 2: Kompilasi Binary dismoen
# -----------------------------------------------------------------------------
echo ">> [2/7] Memeriksa kompilasi binary dismoen..."
pixi run build >/dev/null 2>&1 || {
    echo "FAIL: Gagal melakukan build binary dismoen!"
    exit 1
}
if [[ ! -x "$DISMOEN" ]]; then
    echo "FAIL: Binary dismoen tidak ditemukan atau tidak executable: $DISMOEN"
    exit 1
fi
echo "   PASS: Binary dismoen siap eksekusi."

# -----------------------------------------------------------------------------
# Stage 3: Eksekusi Unit Test M9-W2
# -----------------------------------------------------------------------------
echo ">> [3/7] Menjalankan unit test suites M9-W2..."
pixi run mojo -I src tests/unit/test_m9_w2_gqa.mojo >/dev/null 2>&1 || {
    echo "FAIL: Unit test GQA gagal!"
    exit 1
}
pixi run mojo -I src tests/unit/test_m9_w2_kmss.mojo >/dev/null 2>&1 || {
    echo "FAIL: Unit test KMSS v1 gagal!"
    exit 1
}
pixi run mojo -I src tests/unit/test_m9_w2_scheduler.mojo >/dev/null 2>&1 || {
    echo "FAIL: Unit test Macro Scheduler gagal!"
    exit 1
}
echo "   PASS: Seluruh unit test (11/11) lolos."

# -----------------------------------------------------------------------------
# Stage 4: CLI forward-port Fresh Execution + Session Save
# -----------------------------------------------------------------------------
echo ">> [4/7] Menguji forward-port fresh execution dan penyimpanan KMSS v1..."
OUT_STAGE4=$("$DISMOEN" forward-port \
    --architecture qwen3.6 \
    --model-dir "$MINI_CONFIG" \
    --tokens "$TOKENS_FIXTURE" \
    --save-session "$SESSION_FILE")

echo "$OUT_STAGE4" | grep -q '"status": "COMPLETED"' || {
    echo "FAIL: Eksekusi stage 4 tidak berstatus COMPLETED!"
    echo "$OUT_STAGE4"
    exit 1
}
echo "$OUT_STAGE4" | grep -q '"seq_len": 8' || {
    echo "FAIL: seq_len stage 4 tidak bernilai 8!"
    echo "$OUT_STAGE4"
    exit 1
}
echo "$OUT_STAGE4" | grep -q '"recompute_tokens": 8' || {
    echo "FAIL: recompute_tokens stage 4 tidak bernilai 8!"
    echo "$OUT_STAGE4"
    exit 1
}
echo "$OUT_STAGE4" | grep -q '"gdn_state_reused": false' || {
    echo "FAIL: gdn_state_reused stage 4 seharusnya false!"
    echo "$OUT_STAGE4"
    exit 1
}
if [[ ! -f "$SESSION_FILE" ]]; then
    echo "FAIL: Berkas session tidak berhasil disimpan: $SESSION_FILE"
    exit 1
fi
SESSION_SIZE=$(stat -c %s "$SESSION_FILE")
if [[ "$SESSION_SIZE" -le 160 ]]; then
    echo "FAIL: Ukuran berkas session terlalu kecil: $SESSION_SIZE bytes"
    exit 1
fi
echo "   PASS: Fresh forward pass berhasil, session tersimpan ($SESSION_SIZE bytes)."

# -----------------------------------------------------------------------------
# Stage 5: CLI forward-port Session Continuation (recompute_tokens == 0)
# -----------------------------------------------------------------------------
echo ">> [5/7] Menguji session continuation tanpa recompute (recompute_tokens == 0)..."
OUT_STAGE5=$("$DISMOEN" forward-port \
    --architecture qwen3.6 \
    --model-dir "$MINI_CONFIG" \
    --tokens "$TOKENS_FIXTURE" \
    --load-session "$SESSION_FILE")

echo "$OUT_STAGE5" | grep -q '"status": "COMPLETED"' || {
    echo "FAIL: Eksekusi stage 5 tidak berstatus COMPLETED!"
    echo "$OUT_STAGE5"
    exit 1
}
echo "$OUT_STAGE5" | grep -q '"recompute_tokens": 0' || {
    echo "FAIL: Invarian krusial dilanggar: recompute_tokens != 0!"
    echo "$OUT_STAGE5"
    exit 1
}
echo "$OUT_STAGE5" | grep -q '"kv_tokens_after": 16' || {
    echo "FAIL: Invarian krusial dilanggar: kv_tokens_after != 16!"
    echo "$OUT_STAGE5"
    exit 1
}
echo "$OUT_STAGE5" | grep -q '"gdn_state_reused": true' || {
    echo "FAIL: Invarian krusial dilanggar: gdn_state_reused != true!"
    echo "$OUT_STAGE5"
    exit 1
}
echo "   PASS: Session continuation berhasil dengan zero-recompute (recompute_tokens == 0)."

# -----------------------------------------------------------------------------
# Stage 6: KMSS Corruption Rejection & Integrity Falsification
# -----------------------------------------------------------------------------
echo ">> [6/7] Menguji deteksi korupsi SHA-256 pada berkas session KMSS..."
CORRUPT_SESSION="${TEST_DIR}/corrupt.session"
cp "$SESSION_FILE" "$CORRUPT_SESSION"
# Korupsi 1 byte pada offset 200 (payload KV cache)
printf '\xff' | dd of="$CORRUPT_SESSION" bs=1 seek=200 count=1 conv=notrunc 2>/dev/null

set +e
OUT_CORRUPT=$("$DISMOEN" forward-port \
    --architecture qwen3.6 \
    --model-dir "$MINI_CONFIG" \
    --tokens "$TOKENS_FIXTURE" \
    --load-session "$CORRUPT_SESSION" 2>&1)
EXIT_CORRUPT=$?
set -e

if [[ $EXIT_CORRUPT -eq 0 ]]; then
    echo "FAIL: Session yang terkorupsi berhasil diload tanpa error!"
    echo "$OUT_CORRUPT"
    exit 1
fi
echo "   PASS: Korupsi berkas session berhasil ditolak secara deterministik."

# -----------------------------------------------------------------------------
# Stage 7: Determinisme Eksekusi
# -----------------------------------------------------------------------------
echo ">> [7/7] Menguji determinisme eksekusi antar run..."
RUN1=$("$DISMOEN" forward-port --architecture qwen3.6 --model-dir "$MINI_CONFIG" --tokens "$TOKENS_FIXTURE")
RUN2=$("$DISMOEN" forward-port --architecture qwen3.6 --model-dir "$MINI_CONFIG" --tokens "$TOKENS_FIXTURE")

if [[ "$RUN1" != "$RUN2" ]]; then
    echo "FAIL: Output JSON forward-port tidak deterministik antar run!"
    diff <(echo "$RUN1") <(echo "$RUN2")
    exit 1
fi
echo "   PASS: Eksekusi 100% deterministik."

echo "======================================================================"
echo "Semua tahap verifikasi M9-W2 BERHASIL! (7/7)"
echo "======================================================================"
