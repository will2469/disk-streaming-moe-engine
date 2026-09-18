#!/bin/bash
# ==============================================================================
# Integration Test Suite: M7-W4 Config CLI + Validasi + 8 Error + FS/Thermal Log
# ==============================================================================
# Sesuai kontrak:
# - docs/milestones/M7-odirect-lru.md (§ Error Handling, § O_DIRECT Configuration)
# - scratch/wave/m7/m7-w4-config-error.md
#
# Pengujian:
# 1. Formatting compliance (Mojo format bersih)
# 2. Binary compilation (dismoen binary siap)
# 3. CLI flag & parameter validation:
#    - block-size {512, 4096, 8192} vs invalid (1024 -> exit 1)
#    - queue-depth {1, 2, 4, 8, 16} vs invalid (32 -> exit 1)
#    - memory budget check (resident + kv + io + dequant + headroom + cache <= limit) -> exit 4
# 4. Phase-split: probe-fallback warning vs pasca-probe hard fail
# 5. Verifikasi 8 normative error types, exit codes (1..4), dan RFC 8259 schema
# 6. Stage behavior & atomic rollback (workdir bersih, zero leaked files)
# 7. Readahead policy [R25] (NONE untuk O_DIRECT vs POSIX_FADV_SEQUENTIAL untuk buffered)
# 8. Telemetri lingkungan & termal per run (fs, mount, block, alignment, ssd_temp, durasi)
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-./dismoen}"
TEST_DIR="/tmp/test_m7_w4_$$"
WORKDIR="$TEST_DIR/workdir"
MODEL_DIR="$TEST_DIR/model"
OUTPUT_DIR="$TEST_DIR/output"
REAL_MODEL_FILE="${REAL_MODEL_FILE:-$HOME/models/qwen3.6-35b-a3b/model-00001-of-00026.safetensors}"

cleanup() {
    chmod -R 777 "$TEST_DIR" 2>/dev/null || true
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

mkdir -p "$WORKDIR" "$MODEL_DIR" "$OUTPUT_DIR"

echo "======================================================================"
echo "M7-W4: Flag CLI + Validasi + 8 Error + FS/Thermal Logging Verification"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Stage 1: Kepatuhan formatting Mojo
# -----------------------------------------------------------------------------
echo ">> [1/8] Memeriksa kepatuhan formatting Mojo..."
FORMAT_OUTPUT=$(pixi run mojo format \
    src/io/telemetry.mojo \
    src/io/odirect.mojo \
    src/cli/m7_errors.mojo \
    src/cli/cmd_decode.mojo \
    src/main.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting memodifikasi berkas:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi
echo "   PASS: Formatting Mojo bersih 100%."

# -----------------------------------------------------------------------------
# Stage 2: Membangun binary dismoen
# -----------------------------------------------------------------------------
echo ">> [2/8] Membangun binary dismoen via pixi build..."
pixi run build
echo "   PASS: Binary dismoen siap dijalankan."

# -----------------------------------------------------------------------------
# Stage 3: Flag CLI & Validasi Parameter Fail-Fast
# -----------------------------------------------------------------------------
echo ">> [3/8] Menguji validasi flag CLI & parameter fail-fast..."
ERR_FILE="$TEST_DIR/err_out.json"

# 3a. Block size invalid (1024 bukan kelipatan diizinkan {512, 4096, 8192}) -> exit 1
set +e
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test" \
  --max-tokens 16 \
  --block-size 1024 \
  --workdir "$WORKDIR" > "$ERR_FILE" 2>&1
status=$?
set -e
if [ $status -ne 1 ]; then
    echo "FAIL: block-size 1024 diharapkan exit 1, got $status"
    exit 1
fi
python3 -c "
import json
with open('$ERR_FILE') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M7_ERR_ODIRECT_ALIGNMENT'
assert d['error']['stage'] == 'io_direct'
"
echo "   PASS: block-size 1024 ditolak fail-fast dengan M7_ERR_ODIRECT_ALIGNMENT (exit 1)."

# 3b. Queue depth invalid (32 bukan diizinkan {1, 2, 4, 8, 16}) -> exit 1
set +e
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test" \
  --max-tokens 16 \
  --queue-depth 32 \
  --workdir "$WORKDIR" > "$ERR_FILE" 2>&1
status=$?
set -e
if [ $status -ne 1 ]; then
    echo "FAIL: queue-depth 32 diharapkan exit 1, got $status"
    exit 1
fi
python3 -c "
import json
with open('$ERR_FILE') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M7_ERR_ODIRECT_ALIGNMENT'
assert d['error']['stage'] == 'io_direct'
"
echo "   PASS: queue-depth 32 ditolak fail-fast dengan M7_ERR_ODIRECT_ALIGNMENT (exit 1)."

# 3c. Valid parameters {512, 4096, 8192} & QD {1, 2, 4, 8, 16} diterima
OUT_OK="$TEST_DIR/out_ok.json"
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test" \
  --max-tokens 16 \
  --block-size 4096 \
  --queue-depth 8 \
  --workdir "$WORKDIR" > "$OUT_OK" 2>&1
python3 -c "
import json
with open('$OUT_OK') as f:
    d = json.load(f)
assert d['status'] == 'success'
assert d['io_config']['block_size'] == 4096
assert d['io_config']['queue_depth'] == 8
"
echo "   PASS: Parameter valid diterima dan tercatat di io_config."

# 3d. Budget memori terlampaui -> exit 4 (M7_ERR_LRU_ALLOC)
set +e
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test" \
  --max-tokens 16 \
  --cache-capacity 999999 \
  --memory-limit 1024 \
  --workdir "$WORKDIR" > "$ERR_FILE" 2>&1
status=$?
set -e
if [ $status -ne 4 ]; then
    echo "FAIL: memory budget overflow diharapkan exit 4, got $status"
    exit 1
fi
python3 -c "
import json
with open('$ERR_FILE') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M7_ERR_LRU_ALLOC'
assert d['error']['stage'] == 'lru_cache'
"
echo "   PASS: Pelanggaran budget memori ditolak fail-fast dengan M7_ERR_LRU_ALLOC (exit 4)."

# -----------------------------------------------------------------------------
# Stage 4: Phase-Split: Probe-Fallback vs Pasca-Probe Hard Fail
# -----------------------------------------------------------------------------
echo ">> [4/8] Menguji phase-split probe-fallback warning vs pasca-probe hard fail..."

# 4a. Probe-fallback pada filesystem yang tidak mendukung O_DIRECT
FALLBACK_STDOUT="$TEST_DIR/fallback_stdout.json"
FALLBACK_STDERR="$TEST_DIR/fallback_stderr.txt"

"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test" \
  --max-tokens 16 \
  --o-direct \
  --mock-fallback \
  --workdir "$WORKDIR" > "$FALLBACK_STDOUT" 2> "$FALLBACK_STDERR"

# Verifikasi warning tercetak ke stderr
grep -q "WARNING: O_DIRECT unsupported on filesystem, falling back to buffered I/O" "$FALLBACK_STDERR"
# Verifikasi fallback terkonfirmasi di JSON stdout
python3 -c "
import json
with open('$FALLBACK_STDOUT') as f:
    d = json.load(f)
assert d['status'] == 'success'
assert d['io_config']['io_path'] == 'BUFFERED'
assert d['io_config']['readahead_policy'] == 'POSIX_FADV_SEQUENTIAL'
assert d['environment']['probe_status'] == 'fallback_buffered'
"
echo "   PASS: Fase probe yang gagal memicu warning stderr + fallback buffered yang sah."

# 4b. Pasca-probe hard fail: fault injection M7_ERR_FORMAT_ALIGNMENT dilarang fallback diam-diam
POST_PROBE_ERR="$TEST_DIR/post_probe_err.json"
set +e
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test" \
  --max-tokens 16 \
  --o-direct \
  --mock-error "M7_ERR_FORMAT_ALIGNMENT" \
  --workdir "$WORKDIR" > "$POST_PROBE_ERR" 2>&1
status=$?
set -e
if [ $status -ne 2 ]; then
    echo "FAIL: Pasca-probe alignment fault diharapkan exit 2, got $status"
    exit 1
fi
python3 -c "
import json
with open('$POST_PROBE_ERR') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == 'M7_ERR_FORMAT_ALIGNMENT'
assert d['error']['stage'] == 'io_direct'
"
echo "   PASS: Pasca-probe alignment fault hard fail dengan M7_ERR_FORMAT_ALIGNMENT (exit 2, tanpa fallback)."

# -----------------------------------------------------------------------------
# Stage 5: Verifikasi 8 Error Codes, Exit Codes, dan RFC 8259 Schema
# -----------------------------------------------------------------------------
echo ">> [5/8] Memverifikasi seluruh 8 error codes M7, exit code SSOT, dan JSON schema..."

declare -A EXPECTED_EXITS=(
    ["M7_ERR_ODIRECT_ALIGNMENT"]=1
    ["M7_ERR_FORMAT_ALIGNMENT"]=2
    ["M7_ERR_ODIRECT_SHORT_READ"]=2
    ["M7_ERR_ODIRECT_ENOSPC"]=3
    ["M7_ERR_ODIRECT_EIO"]=3
    ["M7_ERR_LRU_NO_VICTIM"]=4
    ["M7_ERR_LRU_ALLOC"]=4
    ["M7_ERR_LRU_CORRUPT"]=4
)

for err_code in "${!EXPECTED_EXITS[@]}"; do
    expected_exit=${EXPECTED_EXITS[$err_code]}
    E_OUT="$TEST_DIR/err_${err_code}.json"

    set +e
    "$DISMOEN" decode \
      --mock-decode \
      --model-dir "$MODEL_DIR" \
      --prompt "Test" \
      --max-tokens 16 \
      --mock-error "$err_code" \
      --workdir "$WORKDIR" > "$E_OUT" 2>&1
    actual_exit=$?
    set -e

    if [ $actual_exit -ne $expected_exit ]; then
        echo "FAIL: $err_code exit code $actual_exit != expected $expected_exit"
        exit 1
    fi

    python3 -c "
import json
with open('$E_OUT') as f:
    d = json.load(f)
assert d['status'] == 'error'
assert d['error']['code'] == '$err_code', f'code mismatch: {d[\"error\"][\"code\"]}'
assert len(d['error']['message']) > 0
assert 'stage' in d['error']
assert 'details' in d['error']
"
    echo "   PASS: $err_code terverifikasi -> exit $expected_exit (RFC 8259 valid)."
done

# -----------------------------------------------------------------------------
# Stage 6: Stage Behavior & Atomic Rollback
# -----------------------------------------------------------------------------
echo ">> [6/8] Memverifikasi stage behavior dan atomic rollback..."

# Pada setiap kegagalan, run_dir dan temporary files harus bersih dari workdir
DIR_COUNT_BEFORE=$(find "$WORKDIR" -type d | wc -l)
FILE_COUNT_BEFORE=$(find "$WORKDIR" -type f | wc -l)

set +e
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test Rollback" \
  --max-tokens 16 \
  --mock-error "M7_ERR_ODIRECT_ENOSPC" \
  --workdir "$WORKDIR" > /dev/null 2>&1
set -e

FILE_COUNT_AFTER=$(find "$WORKDIR" -type f | wc -l)
if [ "$FILE_COUNT_AFTER" -ne "$FILE_COUNT_BEFORE" ]; then
    echo "FAIL: File bocor di workdir setelah error! Before: $FILE_COUNT_BEFORE, After: $FILE_COUNT_AFTER"
    exit 1
fi
echo "   PASS: Atomic rollback terbukti: zero leaked temporary files di workdir."

# -----------------------------------------------------------------------------
# Stage 7: Readahead Policy [R25] Verification
# -----------------------------------------------------------------------------
echo ">> [7/8] Memverifikasi kebijakan readahead [R25]..."

# O_DIRECT path: readahead NONE
STDOUT_OD="$TEST_DIR/stdout_odirect.json"
if [ -f "$REAL_MODEL_FILE" ]; then
    REAL_DIR="$(dirname "$REAL_MODEL_FILE")"
    "$DISMOEN" decode \
      --mock-decode \
      --model-dir "$REAL_DIR" \
      --prompt "Readahead Test" \
      --max-tokens 16 \
      --o-direct \
      --workdir "$WORKDIR" > "$STDOUT_OD" 2>&1
    python3 -c "
import json
with open('$STDOUT_OD') as f:
    d = json.load(f)
assert d['io_config']['o_direct'] is True
assert d['io_config']['readahead_policy'] == 'NONE'
assert d['io_config']['io_path'] == 'O_DIRECT'
"
    echo "   PASS: Jalur O_DIRECT mencatat readahead NONE pada berkas fisik."
fi

# Buffered path: readahead POSIX_FADV_SEQUENTIAL
STDOUT_BUF="$TEST_DIR/stdout_buffered.json"
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Buffered Test" \
  --max-tokens 16 \
  --workdir "$WORKDIR" > "$STDOUT_BUF" 2>&1
python3 -c "
import json
with open('$STDOUT_BUF') as f:
    d = json.load(f)
assert d['io_config']['o_direct'] is False
assert d['io_config']['readahead_policy'] == 'POSIX_FADV_SEQUENTIAL'
assert d['io_config']['io_path'] == 'BUFFERED'
"
echo "   PASS: Jalur Buffered mencatat readahead POSIX_FADV_SEQUENTIAL [R25]."

# -----------------------------------------------------------------------------
# Stage 8: Telemetri Lingkungan & Termal Per Run
# -----------------------------------------------------------------------------
echo ">> [8/8] Memverifikasi telemetri filesystem dan sensor termal per run..."

STDOUT_TELEMETRY="$TEST_DIR/stdout_telemetry.json"
"$DISMOEN" decode \
  --mock-decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Telemetry Test" \
  --max-tokens 32 \
  --workdir "$WORKDIR" > "$STDOUT_TELEMETRY" 2>&1

python3 -c "
import json
with open('$STDOUT_TELEMETRY') as f:
    d = json.load(f)

# Validasi blok environment
env = d['environment']
assert isinstance(env['fs_type'], str) and len(env['fs_type']) > 0
assert isinstance(env['mount_options'], str) and len(env['mount_options']) > 0
assert env['fs_block_size'] in [512, 1024, 2048, 4096, 8192]
assert env['dio_alignment'] in [512, 4096]
assert env['layout_scan'] == 'verified_m6_v1'
assert env['probe_status'] in ['ok', 'fallback_buffered', 'not_requested']
assert 'duration_sec' in env and env['duration_sec'] >= 0.0
assert isinstance(env['sustained_valid'], bool)
# ssd_temp_c boleh float (misal 39.85) atau None (jika sensor tidak tersedia)
if env['ssd_temp_c'] is not None:
    assert isinstance(env['ssd_temp_c'], (int, float))
    assert 0.0 < env['ssd_temp_c'] < 120.0

# Validasi blok cache_stats
cs = d['cache_stats']
assert 'cache_hit_requests' in cs
assert 'cache_miss_requests' in cs
assert 'hit_bytes' in cs
assert 'miss_bytes' in cs
assert 'disk_bytes' in cs
assert 'ram_bytes' in cs
assert 'evictions' in cs
assert 'pinned_experts' in cs
print('   Telemetri terverifikasi: FS=' + env['fs_type'] + ', block=' + str(env['fs_block_size']) + ', SSD Temp=' + str(env['ssd_temp_c']) + ' C')
"
echo "   PASS: Seluruh field telemetri sistem dan termal SSD terkonfirmasi valid."

echo "======================================================================"
echo "SEMUA PENGUJIAN M7-W4 (CONFIG + VALIDASI + 8 ERROR + TELEMETRI) LULUS!"
echo "STATUS M7-W4: DONE — SIAP LANJUT KE M7-W5 (BENCHMARK CORE + F16)"
echo "======================================================================"
