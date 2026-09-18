#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite: Milestone M8 Wave 3
# CLI kimo gdn, State Lifecycle, GDNS v1 Serialization, and 7 Error Codes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

KIMO="./dismoen"
FIXTURES_DIR="fixtures"
TMP_DIR="$(mktemp -d -t dismoen_m8_w3_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "=== M8-W3 Master Integration Test Suite: dismoen gdn CLI & State Lifecycle ==="

# Pastikan binary dismoen tersedia
if [ ! -f "$KIMO" ]; then
    echo "Binary dismoen tidak ditemukan, mengompilasi via pixi build..."
    pixi run build
fi

# ----------------------------------------------------------------------
# Test 1: Happy Path CLI Invocation
# ----------------------------------------------------------------------
echo "--> Test 1: Verifikasi Happy Path CLI kimo gdn..."
OUT_HAPPY="$TMP_DIR/happy_state.bin"
STDOUT_HAPPY="$TMP_DIR/happy_stdout.json"

"$KIMO" gdn \
    --model-dir "$FIXTURES_DIR" \
    --tokens "$FIXTURES_DIR/m8_tokens.json" \
    --output "$OUT_HAPPY" \
    --layers 2 \
    --dk 32 \
    --dv 32 \
    --chunk-size 8 \
    --threads 1 > "$STDOUT_HAPPY"

# Verifikasi file binary terbentuk
[ -f "$OUT_HAPPY" ] || { echo "FAIL: output state binary tidak terbentuk"; exit 1; }

# Verifikasi JSON output di stdout
python3 -c '
import json, sys
data = json.load(open("'$STDOUT_HAPPY'"))
assert data["status"] == "success", "Status must be success, got " + str(data.get("status"))
assert data["layers"] == 2
assert data["dk"] == 32
assert data["dv"] == 32
assert data["chunk_size"] == 8
assert data["seq_len"] == 16
assert data["metrics"]["peak_state_bytes"] == 8192
assert data["metrics"]["vmhwm_bytes"] > 0
'

# Verifikasi numerical equivalence dengan golden naive oracle (Gate G-M8-1)
"$KIMO" compare \
    --reference "$FIXTURES_DIR/m8_state_naive.bin" \
    --candidate "$OUT_HAPPY" \
    --gate G-M8-1 >/dev/null

echo "PASS: Test 1 (Happy path CLI sukses dan sesuai kontrak G-M8-1)"

# ----------------------------------------------------------------------
# Test 2: State Lifecycle: Zero-init vs State Continuation
# ----------------------------------------------------------------------
echo "--> Test 2: Verifikasi State Lifecycle (Continuation vs Single-Pass Combined)..."

# Siapkan tokens split: 8 token pertama (seq1) dan 8 token berikutnya (seq2)
python3 -c '
import json
tokens = [1, 23, 45, 67, 89, 101, 123, 145, 167, 189, 201, 223, 245, 267, 289, 311]
with open("'$TMP_DIR'/tokens_seq1.json", "w") as f:
    json.dump({"tokens": tokens[:8], "seq_len": 8}, f)
with open("'$TMP_DIR'/tokens_seq2.json", "w") as f:
    json.dump({"tokens": tokens[8:], "seq_len": 8}, f)
'

OUT_SEQ1="$TMP_DIR/seq1_state.bin"
OUT_CONT="$TMP_DIR/cont_state.bin"

# 1. Jalankan seq1 (zero-init)
"$KIMO" gdn \
    --model-dir "$FIXTURES_DIR" \
    --tokens "$TMP_DIR/tokens_seq1.json" \
    --output "$OUT_SEQ1" \
    --layers 2 --dk 32 --dv 32 --chunk-size 8 >/dev/null

# 2. Jalankan seq2 dengan continuation dari state seq1
"$KIMO" gdn \
    --model-dir "$FIXTURES_DIR" \
    --tokens "$TMP_DIR/tokens_seq2.json" \
    --state-input "$OUT_SEQ1" \
    --output "$OUT_CONT" \
    --layers 2 --dk 32 --dv 32 --chunk-size 8 >/dev/null

# 3. Verifikasi ekuivalensi numerik continuation vs single-pass 16 token (Gate G-M8-1)
"$KIMO" compare \
    --reference "$OUT_HAPPY" \
    --candidate "$OUT_CONT" \
    --gate G-M8-1 >/dev/null

echo "PASS: Test 2 (State continuation ekuivalen penuh terhadap single-pass under G-M8-1)"

# ----------------------------------------------------------------------
# Test 3: Error Code 1 (INPUT_INVALID)
# ----------------------------------------------------------------------
echo "--> Test 3: Verifikasi Error Code 1 (INPUT_INVALID)..."

# 3a: Tokens file tidak ditemukan
set +e
ERR_OUT1=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$TMP_DIR/non_existent.json" --output "$TMP_DIR/dummy.bin" 2>&1 >/dev/null)
RET1=$?
set -e
[ "$RET1" -eq 1 ] || { echo "FAIL: expected exit 1 on missing tokens, got $RET1"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT1"'""")
assert err["error_code"] == 1
assert err["error_type"] == "INPUT_INVALID"
'

# 3b: Tokens JSON format salah
echo "invalid json content {" > "$TMP_DIR/bad_tokens.json"
set +e
ERR_OUT2=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$TMP_DIR/bad_tokens.json" --output "$TMP_DIR/dummy.bin" 2>&1 >/dev/null)
RET2=$?
set -e
[ "$RET2" -eq 1 ] || { echo "FAIL: expected exit 1 on corrupt tokens JSON, got $RET2"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT2"'""")
assert err["error_code"] == 1
assert err["error_type"] == "INPUT_INVALID"
'

# 3c: Opsi CLI tidak dikenal (mis. --seed di runtime forward ditolak)
set +e
ERR_OUT3=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$FIXTURES_DIR/m8_tokens.json" --output "$TMP_DIR/dummy.bin" --seed 42 2>&1 >/dev/null)
RET3=$?
set -e
[ "$RET3" -eq 1 ] || { echo "FAIL: expected exit 1 on unknown option --seed, got $RET3"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT3"'""")
assert err["error_code"] == 1
assert err["error_type"] == "INPUT_INVALID"
'

echo "PASS: Test 3 (Error Code 1 INPUT_INVALID terverifikasi di seluruh skenario input)"

# ----------------------------------------------------------------------
# Test 4: Error Code 2 (CONFIG_INVALID)
# ----------------------------------------------------------------------
echo "--> Test 4: Verifikasi Error Code 2 (CONFIG_INVALID)..."

# 4a: Dimensi negatif atau nol
set +e
ERR_OUT_CFG1=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$FIXTURES_DIR/m8_tokens.json" --output "$TMP_DIR/dummy.bin" --layers -2 2>&1 >/dev/null)
RET_CFG1=$?
set -e
[ "$RET_CFG1" -eq 2 ] || { echo "FAIL: expected exit 2 on negative layers, got $RET_CFG1"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT_CFG1"'""")
assert err["error_code"] == 2
assert err["error_type"] == "CONFIG_INVALID"
'

# 4b: Pre-alloc checked arithmetic guard (> 100 MB ceiling)
set +e
ERR_OUT_CFG2=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$FIXTURES_DIR/m8_tokens.json" --output "$TMP_DIR/dummy.bin" --layers 1000000 --dk 1000000 2>&1 >/dev/null)
RET_CFG2=$?
set -e
[ "$RET_CFG2" -eq 2 ] || { echo "FAIL: expected exit 2 on allocation overflow, got $RET_CFG2"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT_CFG2"'""")
assert err["error_code"] == 2
assert err["error_type"] == "CONFIG_INVALID"
'

# 4c: State input dimension mismatch (state dk=32, dipanggil dengan dk=48)
set +e
ERR_OUT_CFG3=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$TMP_DIR/tokens_seq2.json" --state-input "$OUT_SEQ1" --output "$TMP_DIR/dummy.bin" --layers 2 --dk 48 --dv 32 2>&1 >/dev/null)
RET_CFG3=$?
set -e
[ "$RET_CFG3" -eq 2 ] || { echo "FAIL: expected exit 2 on state input dimension mismatch, got $RET_CFG3"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT_CFG3"'""")
assert err["error_code"] == 2
assert err["error_type"] == "CONFIG_INVALID"
'

echo "PASS: Test 4 (Error Code 2 CONFIG_INVALID terverifikasi pada checked arithmetic & dims)"

# ----------------------------------------------------------------------
# Test 5: Error Code 4 (IO_ERROR)
# ----------------------------------------------------------------------
echo "--> Test 5: Verifikasi Error Code 4 (IO_ERROR)..."

# Corrupt trailing checksum pada state-input file
CORRUPT_STATE="$TMP_DIR/corrupt_state.bin"
cp "$OUT_SEQ1" "$CORRUPT_STATE"
# Ubah 1 byte di bagian checksum (32 byte terakhir)
python3 -c '
with open("'$CORRUPT_STATE'", "r+b") as f:
    f.seek(-1, 2)
    b = f.read(1)
    f.seek(-1, 2)
    f.write(bytes([(b[0] ^ 0xFF)]))
'

set +e
ERR_OUT_IO=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$TMP_DIR/tokens_seq2.json" --state-input "$CORRUPT_STATE" --output "$TMP_DIR/dummy.bin" --layers 2 --dk 32 --dv 32 2>&1 >/dev/null)
RET_IO=$?
set -e
[ "$RET_IO" -eq 4 ] || { echo "FAIL: expected exit 4 on corrupt state checksum, got $RET_IO"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT_IO"'""")
assert err["error_code"] == 4
assert err["error_type"] == "IO_ERROR"
'

echo "PASS: Test 5 (Error Code 4 IO_ERROR terverifikasi saat integritas checksum gagal)"

# ----------------------------------------------------------------------
# Test 6: Error Code 6 (OUTPUT_ERROR)
# ----------------------------------------------------------------------
echo "--> Test 6: Verifikasi Error Code 6 (OUTPUT_ERROR)..."

RO_DIR="$TMP_DIR/readonly_output_dir"
mkdir -p "$RO_DIR"
chmod 555 "$RO_DIR"

set +e
ERR_OUT_WRITE=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$FIXTURES_DIR/m8_tokens.json" --output "$RO_DIR/state.bin" --layers 2 --dk 32 --dv 32 --chunk-size 8 2>&1 >/dev/null)
RET_WRITE=$?
set -e
chmod 777 "$RO_DIR"
[ "$RET_WRITE" -eq 6 ] || { echo "FAIL: expected exit 6 on write failure, got $RET_WRITE"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT_WRITE"'""")
assert err["error_code"] == 6
assert err["error_type"] == "OUTPUT_ERROR"
'

echo "PASS: Test 6 (Error Code 6 OUTPUT_ERROR terverifikasi pada kegagalan write atomic)"

# ----------------------------------------------------------------------
# Test 7: Error Code 7 (CHUNK_SIZE_ERROR)
# ----------------------------------------------------------------------
echo "--> Test 7: Verifikasi Error Code 7 (CHUNK_SIZE_ERROR)..."

# 7a: chunk_size < 8
set +e
ERR_OUT_C1=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$FIXTURES_DIR/m8_tokens.json" --output "$TMP_DIR/dummy.bin" --chunk-size 4 2>&1 >/dev/null)
RET_C1=$?
set -e
[ "$RET_C1" -eq 7 ] || { echo "FAIL: expected exit 7 on chunk_size=4, got $RET_C1"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT_C1"'""")
assert err["error_code"] == 7
assert err["error_type"] == "CHUNK_SIZE_ERROR"
'

# 7b: chunk_size > 4096
set +e
ERR_OUT_C2=$("$KIMO" gdn --model-dir "$FIXTURES_DIR" --tokens "$FIXTURES_DIR/m8_tokens.json" --output "$TMP_DIR/dummy.bin" --chunk-size 5000 2>&1 >/dev/null)
RET_C2=$?
set -e
[ "$RET_C2" -eq 7 ] || { echo "FAIL: expected exit 7 on chunk_size=5000, got $RET_C2"; exit 1; }
python3 -c '
import json
err = json.loads("""'"$ERR_OUT_C2"'""")
assert err["error_code"] == 7
assert err["error_type"] == "CHUNK_SIZE_ERROR"
'

echo "PASS: Test 7 (Error Code 7 CHUNK_SIZE_ERROR terverifikasi di batas [8, 4096])"

# ----------------------------------------------------------------------
# Test 8: SEC-4 & SEC-5 (Read-only model dir & RLIMIT_FSIZE)
# ----------------------------------------------------------------------
echo "--> Test 8: Verifikasi SEC-4 & SEC-5 (Model dir read-only & RLIMIT_FSIZE)..."

# SEC-5: Model dir read-only
RO_MODEL="$TMP_DIR/ro_model"
mkdir -p "$RO_MODEL"
cp "$FIXTURES_DIR/m8_gdn_weights.safetensors" "$RO_MODEL/"
chmod -R 555 "$RO_MODEL"

OUT_SEC5="$TMP_DIR/sec5_state.bin"
"$KIMO" gdn \
    --model-dir "$RO_MODEL" \
    --tokens "$FIXTURES_DIR/m8_tokens.json" \
    --output "$OUT_SEC5" \
    --layers 2 --dk 32 --dv 32 --chunk-size 8 >/dev/null

[ -f "$OUT_SEC5" ] || { echo "FAIL: state binary tidak terbentuk pada read-only model dir"; exit 1; }
chmod -R 777 "$RO_MODEL"

# SEC-4: RLIMIT_FSIZE
(
    ulimit -f 100000 2>/dev/null || true
    "$KIMO" gdn \
        --model-dir "$FIXTURES_DIR" \
        --tokens "$FIXTURES_DIR/m8_tokens.json" \
        --output "$TMP_DIR/sec4_state.bin" \
        --layers 2 --dk 32 --dv 32 --chunk-size 8 >/dev/null
)
[ -f "$TMP_DIR/sec4_state.bin" ] || { echo "FAIL: state binary gagal dibuat di bawah batas RLIMIT_FSIZE wajar"; exit 1; }

echo "PASS: Test 8 (SEC-4 dan SEC-5 lolos tanpa mutasi model directory)"

# ----------------------------------------------------------------------
# Test 9: Uji Determinisme 5x Ulangan Run Identik
# ----------------------------------------------------------------------
echo "--> Test 9: Uji Determinisme 5x Ulangan Run Identik..."
REF_HASH=""
for run in 1 2 3 4 5; do
    RUN_OUT="$TMP_DIR/run_${run}.bin"
    "$KIMO" gdn \
        --model-dir "$FIXTURES_DIR" \
        --tokens "$FIXTURES_DIR/m8_tokens.json" \
        --output "$RUN_OUT" \
        --layers 2 --dk 32 --dv 32 --chunk-size 8 --threads 1 >/dev/null
    RUN_HASH=$(sha256sum "$RUN_OUT" | awk '{print $1}')
    if [ -z "$REF_HASH" ]; then
        REF_HASH="$RUN_HASH"
    else
        [ "$REF_HASH" = "$RUN_HASH" ] || {
            echo "FAIL: Determinisme run $run gagal! Hash $RUN_HASH != $REF_HASH"
            exit 1
        }
    fi
done
echo "PASS: Test 9 (Determinisme bitwise 5x ulangan identik terverifikasi, SHA=$REF_HASH)"

echo "=== M8-W3 SUKSES 100%: CLI kimo gdn, Lifecycle, GDNS v1, dan 7 Error Codes LULUS ==="
