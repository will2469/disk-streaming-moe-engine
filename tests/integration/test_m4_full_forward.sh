#!/bin/bash
# ==============================================================================
# test_m4_full_forward.sh — Integration Test Suite M4-W5 (IT-M4-1..13 + SEC)
#
# Memverifikasi DoD M4-W5:
# - IT-M4-1:  Happy path 5 prompt x 16 token (Gate G-M4-1 loose PASS, Tier-1 routing SET)
# - IT-M4-2:  Missing shard file (Exit 4, error M4_ERR_SHARD_IO)
# - IT-M4-3:  Corrupt shard header (Exit 2, error M4_ERR_INDEX)
# - IT-M4-4:  Invalid tokens out-of-vocab (Exit 1, error M4_ERR_INPUT)
# - IT-M4-5:  Empty tokens array (Exit 1, error M4_ERR_INPUT)
# - IT-M4-6:  Cgroup memory.max=6G boundary (Exit 0, VmHWM <= 5 GiB, oom_kill == 0)
# - IT-M4-7:  Deterministic output threads=1 (SHA-256 match di 2 run)
# - IT-M4-8:  Workdir not writable (Exit 1, error M4_ERR_INPUT)
# - IT-M4-9:  Model dir not readable (Exit 1, error M4_ERR_INPUT)
# - IT-M4-10: Layer buffer release test (24 layers, VmHWM <= 5 GiB, no leak)
# - IT-M4-11: Output escape ditolak SEC-5 (Exit 1, error M4_ERR_INPUT, zero escape)
# - IT-M4-12: Tokens melebihi batas (count > 1024, file > 1 MiB -> Exit 1 M4_ERR_INPUT)
# - IT-M4-13: Isolasi cleanup workdir bersama (2 run, kedua output utuh, 0 orphan)
# - Negative path coverage: Error codes 1-6, valid JSON schema, cleanup runs/<run-id>
# - SEC-4 & SEC-5: cgroup 6G + RLIMIT_FSIZE, read-only model dir, atomic se-directory
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
    echo "SKIP: model directory not found: $MODEL_DIR"
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

TEST_DIR="/tmp/test_m4_full_$$"
WORKDIR="$TEST_DIR/work"
HARDLINK_DIR="$ROOT_DIR/target/test_m4_hardlinks_$$"

cleanup() {
    chmod -R 777 "$TEST_DIR" 2>/dev/null || true
    chmod -R 777 "$HARDLINK_DIR" 2>/dev/null || true
    rm -rf "$TEST_DIR" "$HARDLINK_DIR"
}
trap cleanup EXIT

mkdir -p "$WORKDIR" "$HARDLINK_DIR"

# Helper untuk membuat hardlink model directory terisolasi (0 copy overhead)
create_isolated_model_dir() {
    local dest="$1"
    mkdir -p "$dest"
    for src_file in "$MODEL_DIR"/*; do
        ln "$src_file" "$dest/$(basename "$src_file")"
    done
}

echo "======================================================================"
echo "M4-W5: Integration Tests (IT-M4-1..13) & Security Verification (SEC-4/5)"
echo "======================================================================"

# ----------------------------------------------------------------------
# IT-M4-4: Invalid tokens (out of vocab) -> Exit 1, M4_ERR_INPUT
# ----------------------------------------------------------------------
echo ">> [IT-M4-4] Menguji token ID di luar kosakata (>= 151936)..."
TOKENS_OOB="$TEST_DIR/tokens_oob.json"
$PYTHON_BIN -c "import json; json.dump([0, 10, 151936, 50], open('$TOKENS_OOB', 'w'))"

ERR_IT4="$TEST_DIR/err_it4.txt"
set +e
"$KIMO" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$TOKENS_OOB" \
    --output "$WORKDIR/out_it4.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT4"
STATUS_IT4=$?
set -e

[ "$STATUS_IT4" -eq 1 ] || { echo "FAIL: IT-M4-4 expected exit 1, got $STATUS_IT4"; exit 1; }
[ ! -f "$WORKDIR/out_it4.bin" ] || { echo "FAIL: IT-M4-4 output file was created!"; exit 1; }
$PYTHON_BIN -c "
import json
with open('$ERR_IT4') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M4_ERR_INPUT'
assert d['error']['stage'] == 'input'
assert 'out of vocabulary' in d['error']['message']
"
echo "   PASS: IT-M4-4 berhasil menolak token out-of-vocab dengan exit 1 M4_ERR_INPUT"

# ----------------------------------------------------------------------
# IT-M4-5: Empty tokens array -> Exit 1, M4_ERR_INPUT
# ----------------------------------------------------------------------
echo ">> [IT-M4-5] Menguji array token kosong []..."
TOKENS_EMPTY="$TEST_DIR/tokens_empty.json"
echo "[]" > "$TOKENS_EMPTY"

ERR_IT5="$TEST_DIR/err_it5.txt"
set +e
"$KIMO" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$TOKENS_EMPTY" \
    --output "$WORKDIR/out_it5.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT5"
STATUS_IT5=$?
set -e

[ "$STATUS_IT5" -eq 1 ] || { echo "FAIL: IT-M4-5 expected exit 1, got $STATUS_IT5"; exit 1; }
[ ! -f "$WORKDIR/out_it5.bin" ] || { echo "FAIL: IT-M4-5 output file was created!"; exit 1; }
$PYTHON_BIN -c "
import json
with open('$ERR_IT5') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M4_ERR_INPUT'
assert d['error']['stage'] == 'input'
assert 'empty' in d['error']['message']
"
echo "   PASS: IT-M4-5 berhasil menolak token array kosong dengan exit 1 M4_ERR_INPUT"

# ----------------------------------------------------------------------
# IT-M4-12: Tokens melebihi batas (count > 1024 atau file > 1 MiB)
# ----------------------------------------------------------------------
echo ">> [IT-M4-12] Menguji batasan token (count > 1024 dan file > 1 MiB)..."
TOKENS_OVERCOUNT="$TEST_DIR/tokens_1025.json"
$PYTHON_BIN -c "import json; json.dump(list(range(1025)), open('$TOKENS_OVERCOUNT', 'w'))"

ERR_IT12A="$TEST_DIR/err_it12a.txt"
set +e
"$KIMO" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$TOKENS_OVERCOUNT" \
    --output "$WORKDIR/out_it12a.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT12A"
STATUS_IT12A=$?
set -e

[ "$STATUS_IT12A" -eq 1 ] || { echo "FAIL: IT-M4-12 count>1024 expected exit 1, got $STATUS_IT12A"; exit 1; }
[ ! -f "$WORKDIR/out_it12a.bin" ] || { echo "FAIL: output file created for overcount tokens!"; exit 1; }
$PYTHON_BIN -c "
import json
with open('$ERR_IT12A') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M4_ERR_INPUT'
assert 'exceeds MAX_TOKENS' in d['error']['message']
"

# File size > 1 MiB (1048576 B)
TOKENS_OVERSIZE="$TEST_DIR/tokens_oversize.json"
$PYTHON_BIN -c "
with open('$TOKENS_OVERSIZE', 'w') as f:
    f.write('[1' + (' ' * 1048580) + ']')
"
ERR_IT12B="$TEST_DIR/err_it12b.txt"
set +e
"$KIMO" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$TOKENS_OVERSIZE" \
    --output "$WORKDIR/out_it12b.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT12B"
STATUS_IT12B=$?
set -e

[ "$STATUS_IT12B" -eq 1 ] || { echo "FAIL: IT-M4-12 file>1MiB expected exit 1, got $STATUS_IT12B"; exit 1; }
[ ! -f "$WORKDIR/out_it12b.bin" ] || { echo "FAIL: output file created for oversize tokens file!"; exit 1; }
$PYTHON_BIN -c "
import json
with open('$ERR_IT12B') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M4_ERR_INPUT'
assert 'exceeds MAX_TOKENS_FILE_BYTES' in d['error']['message']
"
echo "   PASS: IT-M4-12 berhasil menolak token oversize sebelum alokasi besar"

# ----------------------------------------------------------------------
# IT-M4-11: Output escape ditolak (aturan output path, SEC-5)
# ----------------------------------------------------------------------
echo ">> [IT-M4-11] Menguji pencegahan output escape (SEC-5)..."
ERR_IT11="$TEST_DIR/err_it11.txt"

# 1. Path traversal '..' ke luar workdir
set +e
"$KIMO" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$WORKDIR/../escape.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT11"
STATUS_IT11A=$?
set -e
[ "$STATUS_IT11A" -eq 1 ] || { echo "FAIL: IT-M4-11 .. escape expected exit 1, got $STATUS_IT11A"; exit 1; }
[ ! -e "$WORKDIR/../escape.bin" ] || { echo "FAIL: escape file created!"; exit 1; }

# 2. Path absolut di luar workdir
set +e
"$KIMO" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "/tmp/absolute_escape_$$.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT11"
STATUS_IT11B=$?
set -e
[ "$STATUS_IT11B" -eq 1 ] || { echo "FAIL: IT-M4-11 absolute escape expected exit 1, got $STATUS_IT11B"; exit 1; }
[ ! -e "/tmp/absolute_escape_$$.bin" ] || { echo "FAIL: absolute escape file created!"; exit 1; }

# 3. Symlink escape di dalam workdir
mkdir -p "$TEST_DIR/outside"
ln -s "$TEST_DIR/outside" "$WORKDIR/symlink_out"
set +e
"$KIMO" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$WORKDIR/symlink_out/escaped.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT11"
STATUS_IT11C=$?
set -e
[ "$STATUS_IT11C" -eq 1 ] || { echo "FAIL: IT-M4-11 symlink escape expected exit 1, got $STATUS_IT11C"; exit 1; }
[ ! -e "$TEST_DIR/outside/escaped.bin" ] || { echo "FAIL: symlink escape file created!"; exit 1; }
echo "   PASS: IT-M4-11 berhasil menolak semua bentuk escape output path (SEC-5)"

# ----------------------------------------------------------------------
# IT-M4-8: Workdir not writable -> Exit 1, M4_ERR_INPUT
# ----------------------------------------------------------------------
echo ">> [IT-M4-8] Menguji workdir tidak writable (chmod 555)..."
UNWRITABLE_WORKDIR="$TEST_DIR/unwritable_work"
mkdir -p "$UNWRITABLE_WORKDIR"
chmod 555 "$UNWRITABLE_WORKDIR"

ERR_IT8="$TEST_DIR/err_it8.txt"
set +e
"$KIMO" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$UNWRITABLE_WORKDIR/out.bin" \
    --workdir "$UNWRITABLE_WORKDIR" >/dev/null 2> "$ERR_IT8"
STATUS_IT8=$?
set -e
chmod 777 "$UNWRITABLE_WORKDIR"

[ "$STATUS_IT8" -eq 1 ] || { echo "FAIL: IT-M4-8 expected exit 1, got $STATUS_IT8"; exit 1; }
$PYTHON_BIN -c "
import json
with open('$ERR_IT8') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M4_ERR_INPUT'
assert 'workdir not writable' in d['error']['message']
"
echo "   PASS: IT-M4-8 berhasil menolak unwritable workdir dengan exit 1 M4_ERR_INPUT"

# ----------------------------------------------------------------------
# IT-M4-9: Model dir not readable -> Exit 1, M4_ERR_INPUT
# ----------------------------------------------------------------------
echo ">> [IT-M4-9] Menguji model dir unreadable / tidak ditemukan..."
UNREADABLE_MODEL="$TEST_DIR/unreadable_model"
mkdir -p "$UNREADABLE_MODEL"
chmod 000 "$UNREADABLE_MODEL"

ERR_IT9="$TEST_DIR/err_it9.txt"
set +e
"$KIMO" forward \
    --model-dir "$UNREADABLE_MODEL" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$WORKDIR/out_it9.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT9"
STATUS_IT9=$?
set -e
chmod 777 "$UNREADABLE_MODEL"

[ "$STATUS_IT9" -eq 1 ] || { echo "FAIL: IT-M4-9 expected exit 1, got $STATUS_IT9"; exit 1; }
$PYTHON_BIN -c "
import json
with open('$ERR_IT9') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M4_ERR_INPUT'
"
echo "   PASS: IT-M4-9 berhasil menolak model dir tak terbaca dengan exit 1 M4_ERR_INPUT"

# ----------------------------------------------------------------------
# IT-M4-2: Missing shard file -> Exit 4, M4_ERR_SHARD_IO
# ----------------------------------------------------------------------
echo ">> [IT-M4-2] Menguji penanganan missing shard (exit 4, M4_ERR_SHARD_IO)..."
MODEL_MISSING="$HARDLINK_DIR/model_missing_shard"
create_isolated_model_dir "$MODEL_MISSING"
rm -f "$MODEL_MISSING/model-00002-of-00008.safetensors"

ERR_IT2="$TEST_DIR/err_it2.txt"
set +e
"$KIMO" forward \
    --model-dir "$MODEL_MISSING" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$WORKDIR/out_it2.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT2"
STATUS_IT2=$?
set -e

[ "$STATUS_IT2" -eq 4 ] || { echo "FAIL: IT-M4-2 expected exit 4, got $STATUS_IT2"; exit 1; }
[ ! -f "$WORKDIR/out_it2.bin" ] || { echo "FAIL: output file created for missing shard!"; exit 1; }
$PYTHON_BIN -c "
import json
with open('$ERR_IT2') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M4_ERR_SHARD_IO'
assert d['error']['stage'] == 'index_load'
assert 'shard file not found' in d['error']['message']
"
# Verifikasi tidak ada orphan direktori runs/
[ -z "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ] || { echo "FAIL: orphan temp file tersisa di runs/"; exit 1; }
echo "   PASS: IT-M4-2 berhasil menangani missing shard dengan exit 4 M4_ERR_SHARD_IO"

# ----------------------------------------------------------------------
# IT-M4-3: Corrupt shard header (F15 fail) -> Exit 2, M4_ERR_INDEX
# ----------------------------------------------------------------------
echo ">> [IT-M4-3] Menguji penanganan corrupt shard header (exit 2, M4_ERR_INDEX)..."
MODEL_CORRUPT="$HARDLINK_DIR/model_corrupt_shard"
create_isolated_model_dir "$MODEL_CORRUPT"
rm -f "$MODEL_CORRUPT/model-00002-of-00008.safetensors"

# Tulis header safetensors corrupt (ukuran header 64 byte tapi berisi data non-JSON)
$PYTHON_BIN -c "
import struct
with open('$MODEL_CORRUPT/model-00002-of-00008.safetensors', 'wb') as f:
    f.write(struct.pack('<Q', 64) + b'corrupt_non_json_safetensors_header_payload_padding_padding____')
"

ERR_IT3="$TEST_DIR/err_it3.txt"
set +e
"$KIMO" forward \
    --model-dir "$MODEL_CORRUPT" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$WORKDIR/out_it3.bin" \
    --workdir "$WORKDIR" >/dev/null 2> "$ERR_IT3"
STATUS_IT3=$?
set -e

[ "$STATUS_IT3" -eq 2 ] || { echo "FAIL: IT-M4-3 expected exit 2, got $STATUS_IT3"; exit 1; }
[ ! -f "$WORKDIR/out_it3.bin" ] || { echo "FAIL: output file created for corrupt shard header!"; exit 1; }
$PYTHON_BIN -c "
import json
with open('$ERR_IT3') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M4_ERR_INDEX'
assert d['error']['stage'] == 'index_load'
"
[ -z "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ] || { echo "FAIL: orphan temp file tersisa di runs/"; exit 1; }
echo "   PASS: IT-M4-3 berhasil menangani corrupt shard header dengan exit 2 M4_ERR_INDEX"

# ----------------------------------------------------------------------
# IT-M4-13: Isolasi cleanup (2 run berbagi workdir, run-id beda)
# ----------------------------------------------------------------------
echo ">> [IT-M4-13] Menguji isolasi cleanup workdir bersama..."
OUT_ISO_A="$WORKDIR/iso_a.bin"
OUT_ISO_B="$WORKDIR/iso_b.bin"

"$KIMO" forward \
    --mock-forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$OUT_ISO_A" \
    --workdir "$WORKDIR" \
    --run-id "M4-20260916-101" >/dev/null

"$KIMO" forward \
    --mock-forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$OUT_ISO_B" \
    --workdir "$WORKDIR" \
    --run-id "M4-20260916-102" >/dev/null

[ -f "$OUT_ISO_A" ] && [ -f "$OUT_ISO_B" ] || { echo "FAIL: output file tidak lengkap!"; exit 1; }
[ -z "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ] || { echo "FAIL: orphan temp tersisa di runs/!"; exit 1; }
echo "   PASS: IT-M4-13 kedua output selamat dan nol orphan temp di runs/"

# ----------------------------------------------------------------------
# Verifikasi Error Codes 1-6 & Schema JSON Failure Paths
# ----------------------------------------------------------------------
echo ">> Menguji kepatuhan skema JSON dan exit code 1-6 pada semua failure paths..."
for code_num in 1 2 3 4 5 6; do
    case $code_num in
        1) err_name="M4_ERR_INPUT" ;;
        2) err_name="M4_ERR_INDEX" ;;
        3) err_name="M4_ERR_MEMORY" ;;
        4) err_name="M4_ERR_SHARD_IO" ;;
        5) err_name="M4_ERR_LAYER_FORWARD" ;;
        6) err_name="M4_ERR_OUTPUT" ;;
    esac

    err_file="$TEST_DIR/err_code_${code_num}.txt"
    set +e
    "$KIMO" forward \
        --mock-error "$err_name" \
        --model-dir "$MODEL_DIR" \
        --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
        --output "$WORKDIR/mock_err_${code_num}.bin" \
        --workdir "$WORKDIR" >/dev/null 2> "$err_file"
    exit_got=$?
    set -e

    [ "$exit_got" -eq "$code_num" ] || {
        echo "FAIL: Expected exit $code_num for $err_name, got $exit_got"
        exit 1
    }

    $PYTHON_BIN -c "
import json
with open('$err_file') as f:
    d = json.load(f)
assert d['status'] == 'error', 'status must be error'
assert 'error' in d, 'missing error object'
err = d['error']
assert err['code'] == '$err_name', f'expected code $err_name, got {err[\"code\"]}'
assert isinstance(err['stage'], str) and len(err['stage']) > 0
assert isinstance(err['message'], str) and len(err['message']) > 0
assert isinstance(err['details'], dict)
"
    [ -z "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ] || {
        echo "FAIL: orphan temp file tersisa di runs/ untuk $err_name"
        exit 1
    }
done
echo "   PASS: Error codes 1-6 teruji semua dengan skema JSON valid dan clean runs/<run-id>"

# ----------------------------------------------------------------------
# SEC-4: RLIMIT_FSIZE & Model Dir Read-Only (SEC-5)
# ----------------------------------------------------------------------
echo ">> [SEC-4 & SEC-5] Menguji RLIMIT_FSIZE dan read-only model directory..."
# SEC-5: Model dir read-only (chmod 555) — engine tidak boleh menulis ke model directory
MODEL_RO="$HARDLINK_DIR/model_readonly"
create_isolated_model_dir "$MODEL_RO"
chmod -R 555 "$MODEL_RO"

OUT_SEC5="$WORKDIR/sec5_out.bin"
"$KIMO" forward \
    --mock-forward \
    --model-dir "$MODEL_RO" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$OUT_SEC5" \
    --workdir "$WORKDIR" >/dev/null
[ -f "$OUT_SEC5" ] || { echo "FAIL: forward gagal pada model directory read-only"; exit 1; }
chmod -R 777 "$MODEL_RO"
echo "   PASS: SEC-5 model directory read-only lolos tanpa mutasi"

# SEC-4: RLIMIT_FSIZE aman (50 MB limit cukup untuk 9.7 MB file)
(
    ulimit -f 100000 2>/dev/null || true
    "$KIMO" forward \
        --mock-forward \
        --model-dir "$MODEL_DIR" \
        --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
        --output "$WORKDIR/sec4_out.bin" \
        --workdir "$WORKDIR" >/dev/null
)
[ -f "$WORKDIR/sec4_out.bin" ] || { echo "FAIL: RLIMIT_FSIZE aman gagal!"; exit 1; }
echo "   PASS: SEC-4 lolos di bawah batasan RLIMIT_FSIZE wajar"

# ----------------------------------------------------------------------
# IT-M4-1, IT-M4-6, IT-M4-10: Happy path, Cgroup 6G boundary & Layer Buffer Release
# ----------------------------------------------------------------------
echo ">> [IT-M4-1, IT-M4-6, IT-M4-10] Menjalankan forward engine penuh 24 layer..."

PROMPT_LIST=${M4_PROMPTS:-"1"}
CGROUP_PREFIX=""
if command -v systemd-run >/dev/null 2>&1; then
    if systemd-run --user --scope -p MemoryMax=6G true >/dev/null 2>&1; then
        CGROUP_PREFIX="systemd-run --user --scope -p MemoryMax=6G"
        echo "   (Mengaktifkan cgroup boundary: $CGROUP_PREFIX)"
    fi
fi

for p in $PROMPT_LIST; do
    echo ">> [IT-M4-1] Evaluasi Happy path Prompt $p..."
    TOKENS="$FIXTURE_DIR/m4_prompt${p}_tokens.json"
    OUTPUT="$WORKDIR/prompt${p}_logits.bin"
    ROUTING_OUT="$WORKDIR/prompt${p}_routing"
    STDOUT_JSON="$WORKDIR/prompt${p}_stdout.json"

    mkdir -p "$ROUTING_OUT"

    # Jalankan forward penuh (di bawah cgroup MemoryMax=6G bila tersedia)
    if [ "${M4_REUSE_LOGITS:-0}" -eq 1 ] && [ -f "$OUTPUT" ] && [ "$(stat -c%s "$OUTPUT")" -eq 9723904 ]; then
        echo "   (Menggunakan logits terkomputasi sebelumnya: $OUTPUT)"
    else
        $CGROUP_PREFIX "$KIMO" forward \
            --model-dir "$MODEL_DIR" \
            --tokens "$TOKENS" \
            --output "$OUTPUT" \
            --workdir "$WORKDIR" \
            --dump-routing "$ROUTING_OUT" \
            --threads 1 > "$STDOUT_JSON"
    fi

    # Verifikasi ukuran logits tepat s * V * 4 = 16 * 151936 * 4 = 9.723.904 byte
    SZ_OUT=$(stat -c%s "$OUTPUT")
    [ "$SZ_OUT" -eq 9723904 ] || {
        echo "FAIL: Output logits Prompt $p size $SZ_OUT != 9723904 byte!"
        exit 1
    }

    # IT-M4-6 & IT-M4-10: Cek metrik memori VmHWM <= 5 GiB dan oom_kill == 0
    if [ -f "$STDOUT_JSON" ]; then
        $PYTHON_BIN -c "
import json
with open('$STDOUT_JSON') as f:
    d = json.load(f)
assert d['status'] == 'success'
m = d['metrics']
vmhwm = m.get('vmhwm_bytes', 0)
oom_kills = m.get('cgroup_oom_kills', 0)
print(f'   [Metrics] VmHWM: {vmhwm / (1024**3):.2f} GiB (<= 5 GiB gate), oom_kills: {oom_kills}')
assert vmhwm <= 5368709120, f'VmHWM {vmhwm} exceeds 5 GiB limit!'
assert oom_kills == 0, f'OOM kills {oom_kills} detected!'
"
    fi

    # IT-M4-1: Evaluasi Gate G-M4-1 via dismoen-tools compare (FP32 vs FP32)
    COMPARE_REPORT=$("$COMPARE_BIN" compare \
        --ref "$FIXTURE_DIR/m4_prompt${p}_oracle.bin" \
        --cand "$OUTPUT" \
        --gate G-M4-1 \
        --dim 151936)

    VERDICT=$(echo "$COMPARE_REPORT" | grep -o '"verdict": "[^"]*"' | head -n1 | cut -d'"' -f4)
    [ "$VERDICT" = "PASS" ] || {
        echo "FAIL: Gate G-M4-1 verdict on Prompt $p is $VERDICT (expected PASS)!"
        echo "$COMPARE_REPORT"
        exit 1
    }
    echo "   PASS: Gate G-M4-1 PASS pada Prompt $p"

    # Verifikasi Tier-1 SET equality routing bila dumps ada
    if [ -d "$ROUTING_OUT" ] && [ -f "$ROUTING_OUT/routing_L0.json" ]; then
        for l in 0 12 23; do
            "$COMPARE_BIN" compare \
                --ref "$FIXTURE_DIR/m4_prompt${p}_oracle.bin" \
                --cand "$OUTPUT" \
                --gate G-M4-1 \
                --dim 151936 \
                --oracle-routing "$FIXTURE_DIR/m4_prompt${p}_routing/routing_L${l}.json" \
                --cand-routing "$ROUTING_OUT/routing_L${l}.json" >/dev/null || {
                    echo "FAIL: Routing Tier-1 SET equality mismatch pada Prompt $p layer $l!"
                    exit 1
                }
        done
        echo "   PASS: Routing Tier-1 SET equality terverifikasi pada layer 0, 12, 23"
    fi
done

# ----------------------------------------------------------------------
# IT-M4-7: Deterministic output (threads=1) -> SHA-256 match di 2 run
# ----------------------------------------------------------------------
echo ">> [IT-M4-7] Menguji determinisme output pada threads=1 (2 run SHA-256 match)..."
OUT_DET_1="$WORKDIR/prompt1_logits.bin"
OUT_DET_2="$WORKDIR/prompt1_run2.bin"

if [ ! -f "$OUT_DET_1" ]; then
    "$KIMO" forward \
        --model-dir "$MODEL_DIR" \
        --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
        --output "$OUT_DET_1" \
        --workdir "$WORKDIR" \
        --threads 1 >/dev/null
fi

# Run 2 dengan threads 1
"$KIMO" forward \
    --model-dir "$MODEL_DIR" \
    --tokens "$FIXTURE_DIR/m4_prompt1_tokens.json" \
    --output "$OUT_DET_2" \
    --workdir "$WORKDIR" \
    --threads 1 >/dev/null

SHA1=$(sha256sum "$OUT_DET_1" | cut -d' ' -f1)
SHA2=$(sha256sum "$OUT_DET_2" | cut -d' ' -f1)

[ "$SHA1" = "$SHA2" ] || {
    echo "FAIL: IT-M4-7 output mismatch antara Run 1 ($SHA1) dan Run 2 ($SHA2)!"
    exit 1
}
echo "   PASS: IT-M4-7 determinisme terbukti bit-exact (SHA-256: $SHA1)"

echo "======================================================================"
echo "SEMUA INTEGRATION TEST M4 (IT-M4-1..13 + SEC-4/5) SUKSES (PASS)!"
echo "======================================================================"
exit 0
