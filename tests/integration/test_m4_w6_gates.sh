#!/bin/bash
# ==============================================================================
# test_m4_w6_gates.sh — Integration Test Suite M4-W6 (Gates G-M4-1, G-M4-2, Perf)
#
# Memverifikasi DoD M4-W6:
# 1. Gate G-M4-1: Evaluasi 5 prompt x 16 token (FP32 vs FP32, loose N, hard-fail A)
# 2. Tier-1 Routing Proof: SET equality kesetaraan top-4 experts layer 0, 12, 23
# 3. Short-circuit A -> N -> S: Deteksi kegagalan router-selection sebelum metrik numerik
# 4. Gate G-M4-2: Sanity memory (VmHWM <= 5 GiB, oom_kill == 0) & runtime (<= 300 s)
# 5. Determinisme: SHA-256 bit-exact match pada 2 run berulang
# 6. Kebersihan direktori: Nol orphan files di workdir/runs/
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

KIMO="${KIMO:-./dismoen}"
COMPARE_BIN="${COMPARE_BIN:-target/debug/dismoen-tools}"
MODEL_DIR="${MODEL_DIR:-/home/will/models/qwen1.5-moe-a2.7b-chat}"
FIXTURE_DIR="tools/fixtures"

PYTHON_BIN="python3"
if [ -f ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
fi

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "SKIP: model directory tidak ditemukan: $MODEL_DIR"
    exit 0
fi

if [ ! -f "$COMPARE_BIN" ]; then
    echo ">> Membangun dismoen-tools..."
    cargo build --manifest-path tools/dismoen-tools/Cargo.toml --bin dismoen-tools
fi

if [ ! -f "$KIMO" ]; then
    echo ">> Membangun engine dismoen..."
    pixi run build
fi

TEST_DIR="/tmp/test_m4_w6_$$"
WORKDIR="$TEST_DIR/work"
ROUTING_BASE="$TEST_DIR/routing"

cleanup() {
    chmod -R 777 "$TEST_DIR" 2>/dev/null || true
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

mkdir -p "$WORKDIR" "$ROUTING_BASE"

echo "======================================================================"
echo "M4-W6: GATES (G-M4-1, G-M4-2, ROUTING TIER-1, DETERMINISME)"
echo "======================================================================"

PROMPT_LIST=${M4_PROMPTS:-"1 2 3 4 5"}
CGROUP_PREFIX=""
if command -v systemd-run >/dev/null 2>&1; then
    if systemd-run --user --scope -p MemoryMax=6G true >/dev/null 2>&1; then
        CGROUP_PREFIX="systemd-run --user --scope -p MemoryMax=6G"
        echo ">> Mengaktifkan isolasi cgroup: $CGROUP_PREFIX"
    fi
fi

# ----------------------------------------------------------------------
# BAGIAN 1: Gate G-M4-1 & Tier-1 Routing SET Proof pada 5 Prompt Golden Set
# ----------------------------------------------------------------------
echo ">> [Bagian 1] Evaluasi Gate G-M4-1 pada 5 Prompt Golden Set..."

for p in $PROMPT_LIST; do
    echo "   --- Memproses Prompt $p ---"
    TOKENS="$FIXTURE_DIR/m4_prompt${p}_tokens.json"
    ORACLE_BIN="$FIXTURE_DIR/m4_prompt${p}_oracle.bin"
    ORACLE_ROUTING="$FIXTURE_DIR/m4_prompt${p}_routing"
    OUTPUT="$WORKDIR/prompt${p}_logits.bin"
    CAND_ROUTING="$ROUTING_BASE/prompt${p}"
    STDOUT_JSON="$WORKDIR/prompt${p}_stdout.json"

    [ -f "$TOKENS" ] || { echo "FAIL: Tokens $TOKENS tidak ditemukan"; exit 1; }
    [ -f "$ORACLE_BIN" ] || { echo "FAIL: Oracle $ORACLE_BIN tidak ditemukan"; exit 1; }

    mkdir -p "$CAND_ROUTING"

    # Jalankan forward pass (atau reuse logits valid jika M4_REUSE_LOGITS=1)
    if [ "${M4_REUSE_LOGITS:-0}" -eq 1 ] && [ -f "$OUTPUT" ] && [ "$(stat -c%s "$OUTPUT")" -eq 9723904 ]; then
        echo "   (Menggunakan logits terkomputasi sebelumnya: $OUTPUT)"
    else
        $CGROUP_PREFIX "$KIMO" forward \
            --model-dir "$MODEL_DIR" \
            --tokens "$TOKENS" \
            --output "$OUTPUT" \
            --workdir "$WORKDIR" \
            --dump-routing "$CAND_ROUTING" \
            --threads 1 > "$STDOUT_JSON"
    fi

    # Verifikasi ukuran logits tepat: 16 tokens x 151936 vocab x 4 bytes = 9.723.904 bytes
    SZ_OUT=$(stat -c%s "$OUTPUT")
    [ "$SZ_OUT" -eq 9723904 ] || {
        echo "FAIL: Ukuran file output Prompt $p ($SZ_OUT bytes) != 9723904 bytes!"
        exit 1
    }

    # Evaluasi Gate G-M4-1 via dismoen-tools compare
    COMPARE_REPORT=$("$COMPARE_BIN" compare \
        --ref "$ORACLE_BIN" \
        --cand "$OUTPUT" \
        --gate G-M4-1 \
        --dim 151936)

    STATUS=$(echo "$COMPARE_REPORT" | grep -o '"status": "[^"]*"' | head -n1 | cut -d'"' -f4)
    VERDICT=$(echo "$COMPARE_REPORT" | grep -o '"verdict": "[^"]*"' | head -n1 | cut -d'"' -f4)

    [ "$STATUS" = "MATCH" ] && [ "$VERDICT" = "PASS" ] || {
        echo "FAIL: Gate G-M4-1 GAGAL pada Prompt $p!"
        echo "$COMPARE_REPORT"
        exit 1
    }
    echo "   PASS: Gate G-M4-1 MATCH (loose numeric) pada Prompt $p"

    # Evaluasi Tier-1 Routing SET equality per layer kunci (0, 12, 23)
    if [ -d "$CAND_ROUTING" ] && [ -f "$CAND_ROUTING/routing_L0.json" ]; then
        for l in 0 12 23; do
            ROUTING_REP=$("$COMPARE_BIN" compare \
                --ref "$ORACLE_BIN" \
                --cand "$OUTPUT" \
                --gate G-M4-1 \
                --dim 151936 \
                --oracle-routing "$ORACLE_ROUTING/routing_L${l}.json" \
                --cand-routing "$CAND_ROUTING/routing_L${l}.json")

            R_VERDICT=$(echo "$ROUTING_REP" | grep -o '"verdict": "[^"]*"' | head -n1 | cut -d'"' -f4)
            [ "$R_VERDICT" = "PASS" ] || {
                echo "FAIL: Tier-1 Routing SET equality GAGAL pada Prompt $p Layer $l!"
                echo "$ROUTING_REP"
                exit 1
            }
        done
        echo "   PASS: Tier-1 Routing SET equality terverifikasi pada layer 0, 12, 23 (Prompt $p)"
    fi

    # ------------------------------------------------------------------
    # BAGIAN 2: Gate G-M4-2 Sanity (Memory & Waktu)
    # ------------------------------------------------------------------
    if [ -f "$STDOUT_JSON" ]; then
        $PYTHON_BIN -c "
import json
with open('$STDOUT_JSON') as f:
    d = json.load(f)
assert d['status'] == 'success'
m = d['metrics']
walltime = m.get('walltime_sec', 0.0)
vmhwm = m.get('vmhwm_bytes', 0)
oom_kills = m.get('cgroup_oom_kills', 0)
print(f'   [G-M4-2 Telemetri] Walltime: {walltime:.2f} s (<= 300 s) | VmHWM: {vmhwm / (1024**3):.2f} GiB (<= 5 GiB) | OOM Kills: {oom_kills} (== 0)')
assert walltime <= 300.0, f'Walltime {walltime} s melebihi batas 300 s!'
assert vmhwm <= 5368709120, f'VmHWM {vmhwm} bytes melebihi batas 5 GiB!'
assert oom_kills == 0, f'OOM kills {oom_kills} terdeteksi!'
"
        echo "   PASS: Gate G-M4-2 lolos pada Prompt $p"
    fi
done

# ----------------------------------------------------------------------
# BAGIAN 3: Short-Circuit A -> N -> S Falsification Test
# ----------------------------------------------------------------------
echo ">> [Bagian 3] Menguji mekanisme Short-Circuit A -> N -> S..."
# Jika routing memiliki anomali, perbandingan harus langsung FAIL di verdict A
# dengan kategori 'router-selection', meskipun file logits identik 100%.

SIM_CORRUPT_ROUTING="$TEST_DIR/sim_corrupt_routing.json"
$PYTHON_BIN -c "
import json
# Bikin routing yang berbeda dari oracle L0
corrupt = {'selected_experts': [[99, 98, 97, 96]] * 16}
with open('$SIM_CORRUPT_ROUTING', 'w') as f:
    json.dump(corrupt, f)
"

set +e
SC_REPORT=$("$COMPARE_BIN" compare \
    --ref "$FIXTURE_DIR/m4_prompt1_oracle.bin" \
    --cand "$FIXTURE_DIR/m4_prompt1_oracle.bin" \
    --gate G-M4-1 \
    --dim 151936 \
    --oracle-routing "$FIXTURE_DIR/m4_prompt1_routing/routing_L0.json" \
    --cand-routing "$SIM_CORRUPT_ROUTING" 2>&1)
SC_STATUS=$?
set -e

[ "$SC_STATUS" -eq 1 ] || {
    echo "FAIL: Short-circuit diharapkan exit 1, tetapi menghasilkan $SC_STATUS"
    exit 1
}

$PYTHON_BIN -c "
import json
d = json.loads('''$SC_REPORT''')
assert d['status'] in ('MISMATCH', 'FAIL')
assert d['verdict'] == 'FAIL'
assert d['fail_category'] == 'router-selection', f'Expected fail_category router-selection, got {d.get(\"fail_category\")}'
"
echo "   PASS: Short-Circuit A -> N -> S terverifikasi: anomali routing langsung memicu FAIL 'router-selection'"

# ----------------------------------------------------------------------
# BAGIAN 4: Determinisme Output (threads=1)
# ----------------------------------------------------------------------
echo ">> [Bagian 4] Menguji determinisme output bit-exact (threads=1)..."
DET_RUN1="$WORKDIR/prompt1_det1.bin"
DET_RUN2="$WORKDIR/prompt1_det2.bin"

if [ -f "$WORKDIR/prompt1_logits.bin" ]; then
    cp "$WORKDIR/prompt1_logits.bin" "$DET_RUN1"
else
    "$KIMO" forward \
        --model-dir "$MODEL_DIR" \
        --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
        --output "$DET_RUN1" \
        --workdir "$WORKDIR" \
        --threads 1 >/dev/null
fi

if [ "${M4_REUSE_LOGITS:-0}" -eq 1 ] && [ -f "$DET_RUN2" ] && [ "$(stat -c%s "$DET_RUN2")" -eq 9723904 ]; then
    echo "   (Menggunakan logits determinisme Run 2 terkomputasi sebelumnya: $DET_RUN2)"
else
    "$KIMO" forward \
        --model-dir "$MODEL_DIR" \
        --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
        --output "$DET_RUN2" \
        --workdir "$WORKDIR" \
        --threads 1 >/dev/null
fi

SHA1=$(sha256sum "$DET_RUN1" | cut -d' ' -f1)
SHA2=$(sha256sum "$DET_RUN2" | cut -d' ' -f1)

[ "$SHA1" = "$SHA2" ] || {
    echo "FAIL: Output determinisme mismatch antara Run 1 ($SHA1) dan Run 2 ($SHA2)!"
    exit 1
}
echo "   PASS: Output deterministik 100% bit-exact (SHA-256: $SHA1)"

# ----------------------------------------------------------------------
# BAGIAN 5: Verifikasi Kebersihan Direktori (Zero Orphan Files)
# ----------------------------------------------------------------------
echo ">> [Bagian 5] Memverifikasi ketiadaan file orphan di workdir/runs..."
[ -z "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ] || {
    echo "FAIL: Ditemukan file orphan tersisa di $WORKDIR/runs/!"
    find "$WORKDIR/runs"
    exit 1
}
echo "   PASS: Direktori $WORKDIR/runs bersih (nol orphan files)"

echo "======================================================================"
echo "SELURUH PENGUJIAN GATE M4-W6 (G-M4-1, G-M4-2, TIER-1 ROUTING) SUKSES!"
echo "======================================================================"
exit 0
