#!/bin/bash
# ==============================================================================
# test_m5_w4_oracle.sh — Integration Test Suite M5-W4 (Oracle, Fixtures & Golden)
#
# Memverifikasi DoD M5-W4:
# 1. Dataset fixture M5 (m5_kv_decode.json, m5_kv_decode_4k.json, tokens_prompt.json)
#    teruji utuh dan mematuhi rantai bound S + N <= ctx <= s_max serta token < 151936.
# 2. Artefak golden ter-commit (logits_kv_decode.bin, logits_recompute.bin, SHA keduanya).
#    Ukuran tepat 64 * 151936 * 4 = 38.895.616 byte dan SHA identity pin terverifikasi.
# 3. Rust compare wiring KV-vs-recompute (Gate G-M5-1, F10 loose thresholds)
#    menghasilkan MATCH / PASS.
# 4. tools/compare.py CLI wrapper berfungsi mulus dengan flag --ref/--cand maupun --mojo/--oracle.
# 5. Fault injection: verifikasi deteksi mutasi (exit 1) dan layout corruption (exit 2).
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COMPARE_BIN="${COMPARE_BIN:-target/debug/kimo-tools}"
FIXTURE_DIR="tools/fixtures"
EXPECTED_BYTES=$((64 * 151936 * 4)) # 38.895.616 bytes

if [ ! -f "$COMPARE_BIN" ]; then
    echo ">> Membangun kimo-tools..."
    cargo build --manifest-path tools/kimo-tools/Cargo.toml
fi

TEST_DIR="/tmp/test_m5_w4_oracle_$$"
cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
mkdir -p "$TEST_DIR"

echo "======================================================================"
echo "M5-W4: Oracle KV vs Recompute, Fixture & Golden Verification"
echo "======================================================================"

# ----------------------------------------------------------------------
# 1. Validasi Fixture M5 dan Rantai Bound
# ----------------------------------------------------------------------
echo ">> [1/5] Verifikasi fixture M5 (2K, 4K, dan tokens_prompt.json)..."

[ -f "$FIXTURE_DIR/m5_kv_decode.json" ] || { echo "FAIL: $FIXTURE_DIR/m5_kv_decode.json tidak ditemukan"; exit 1; }
[ -f "$FIXTURE_DIR/m5_kv_decode_4k.json" ] || { echo "FAIL: $FIXTURE_DIR/m5_kv_decode_4k.json tidak ditemukan"; exit 1; }
[ -f "$FIXTURE_DIR/tokens_prompt.json" ] || { echo "FAIL: $FIXTURE_DIR/tokens_prompt.json tidak ditemukan"; exit 1; }

python3 -c "
import json

# Validasi 2K
with open('$FIXTURE_DIR/m5_kv_decode.json') as f:
    d2k = json.load(f)
p2k = d2k['prompt']
s2k = d2k['token_count']
exp2k = p2k['expected_tokens']
ctx2k = p2k['context_size']
assert s2k + exp2k <= ctx2k <= 4096, f'Bound violation in 2K: {s2k} + {exp2k} <= {ctx2k}'
assert exp2k == 64
assert ctx2k == 2048

# Validasi 4K
with open('$FIXTURE_DIR/m5_kv_decode_4k.json') as f:
    d4k = json.load(f)
p4k = d4k['prompt']
s4k = d4k['token_count']
exp4k = p4k['expected_tokens']
ctx4k = p4k['context_size']
assert s4k + exp4k <= ctx4k <= 4096, f'Bound violation in 4K: {s4k} + {exp4k} <= {ctx4k}'
assert exp4k == 64
assert ctx4k == 4096

# Validasi tokens
with open('$FIXTURE_DIR/tokens_prompt.json') as f:
    toks = json.load(f)
assert isinstance(toks, list)
assert len(toks) == s2k
for t in toks:
    assert isinstance(t, int) and 0 <= t < 151936, f'Token ID invalid: {t}'

print(f'   PASS: Fixture valid (2K prompt: {s2k} tokens, 4K prompt: {s4k} tokens, all < 151936)')
"

# ----------------------------------------------------------------------
# 2. Validasi Artefak Golden Biner & SHA Identity Pin
# ----------------------------------------------------------------------
echo ">> [2/5] Verifikasi artefak golden oracle biner & SHA-256 identity pins..."

KV_BIN="$FIXTURE_DIR/logits_kv_decode.bin"
REC_BIN="$FIXTURE_DIR/logits_recompute.bin"
KV_SHA="$FIXTURE_DIR/oracle_kv_decode.sha256"
REC_SHA="$FIXTURE_DIR/oracle_recompute.sha256"

[ -f "$KV_BIN" ] || { echo "FAIL: $KV_BIN tidak ditemukan"; exit 1; }
[ -f "$REC_BIN" ] || { echo "FAIL: $REC_BIN tidak ditemukan"; exit 1; }
[ -f "$KV_SHA" ] || { echo "FAIL: $KV_SHA tidak ditemukan"; exit 1; }
[ -f "$REC_SHA" ] || { echo "FAIL: $REC_SHA tidak ditemukan"; exit 1; }

sz_kv=$(stat -c%s "$KV_BIN")
sz_rec=$(stat -c%s "$REC_BIN")

[ "$sz_kv" -eq "$EXPECTED_BYTES" ] || {
    echo "FAIL: $KV_BIN size $sz_kv != $EXPECTED_BYTES byte!"
    exit 1
}
[ "$sz_rec" -eq "$EXPECTED_BYTES" ] || {
    echo "FAIL: $REC_BIN size $sz_rec != $EXPECTED_BYTES byte!"
    exit 1
}

(cd "$FIXTURE_DIR" && sha256sum -c oracle_kv_decode.sha256) || {
    echo "FAIL: Checksum mismatch pada logits_kv_decode.bin!"
    exit 1
}
(cd "$FIXTURE_DIR" && sha256sum -c oracle_recompute.sha256) || {
    echo "FAIL: Checksum mismatch pada logits_recompute.bin!"
    exit 1
}
echo "   PASS: Artefak golden biner ($EXPECTED_BYTES B) & SHA-256 identity pins valid"

# ----------------------------------------------------------------------
# 3. Evaluasi Gate G-M5-1 via kimo-tools compare (Rust)
# ----------------------------------------------------------------------
echo ">> [3/5] Evaluasi Gate G-M5-1 via kimo-tools compare (Rust)..."

COMPARE_OUT=$("$COMPARE_BIN" compare \
    --ref "$REC_BIN" \
    --cand "$KV_BIN" \
    --gate G-M5-1 \
    --dim 151936)

STATUS=$(echo "$COMPARE_OUT" | grep -o '"status": "[^"]*"' | head -n1 | cut -d'"' -f4)
VERDICT=$(echo "$COMPARE_OUT" | grep -o '"verdict": "[^"]*"' | head -n1 | cut -d'"' -f4)

[ "$STATUS" = "MATCH" ] && [ "$VERDICT" = "PASS" ] || {
    echo "FAIL: Gate G-M5-1 GAGAL!"
    echo "$COMPARE_OUT"
    exit 1
}
echo "   PASS: Gate G-M5-1 MATCH & PASS (F10 loose threshold terpenuhi)"

# ----------------------------------------------------------------------
# 4. Evaluasi Wrapper tools/compare.py (Python CLI)
# ----------------------------------------------------------------------
echo ">> [4/5] Verifikasi tools/compare.py CLI wrapper..."

# Test dengan --mojo dan --oracle
WRAPPER_OUT=$(python3 tools/compare.py \
    --mojo "$KV_BIN" \
    --oracle "$REC_BIN" \
    --gate G-M5-1 \
    --dim 151936)

W_STATUS=$(echo "$WRAPPER_OUT" | grep -o '"status": "[^"]*"' | head -n1 | cut -d'"' -f4)
W_VERDICT=$(echo "$WRAPPER_OUT" | grep -o '"verdict": "[^"]*"' | head -n1 | cut -d'"' -f4)

[ "$W_STATUS" = "MATCH" ] && [ "$W_VERDICT" = "PASS" ] || {
    echo "FAIL: Wrapper compare.py gagal!"
    echo "$WRAPPER_OUT"
    exit 1
}
echo "   PASS: tools/compare.py --mojo/--oracle sukses"

# ----------------------------------------------------------------------
# 5. Fault Injection: Deteksi Mutasi & Layout Corruption
# ----------------------------------------------------------------------
echo ">> [5/5] Uji ketahanan terhadap mutasi numerik & corrupt layout..."

# 5a. Mutasi numerik buatan
MUTATED_BIN="$TEST_DIR/mutated_kv.bin"
cp "$KV_BIN" "$MUTATED_BIN"
python3 -c "
import struct
with open('$MUTATED_BIN', 'r+b') as f:
    f.seek(4000)
    val = struct.unpack('<f', f.read(4))[0]
    f.seek(4000)
    f.write(struct.pack('<f', val + 5.0))
"

set +e
"$COMPARE_BIN" compare --ref "$REC_BIN" --cand "$MUTATED_BIN" --gate G-M5-1 --dim 151936 >/dev/null 2>&1
RC_MUT=$?
set -e

[ "$RC_MUT" -eq 1 ] || {
    echo "FAIL: Mutasi numerik diharapkan menghasilkan exit code 1, dapat $RC_MUT"
    exit 1
}
echo "   PASS: Mutasi numerik terdeteksi sebagai MISMATCH (exit code 1)"

# 5b. Truncated layout corruption
TRUNC_BIN="$TEST_DIR/truncated.bin"
head -c 1024 "$KV_BIN" > "$TRUNC_BIN"

set +e
"$COMPARE_BIN" compare --ref "$REC_BIN" --cand "$TRUNC_BIN" --gate G-M5-1 --dim 151936 >/dev/null 2>&1
RC_TRUNC=$?
set -e

[ "$RC_TRUNC" -eq 2 ] || {
    echo "FAIL: File terpotong diharapkan menghasilkan exit code 2, dapat $RC_TRUNC"
    exit 1
}
echo "   PASS: Truncated layout terdeteksi sebagai ERROR (exit code 2)"

echo "======================================================================"
echo "SEMUA SUITE M5-W4 (ORACLE, FIXTURES & GOLDEN) SUKSES (PASS)"
echo "======================================================================"
exit 0
