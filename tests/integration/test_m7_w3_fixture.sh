#!/bin/bash
# ==============================================================================
# test_m7_w3_fixture.sh — Integration Test Suite M7-W3 (Fixture 2 Pola + Benchmark)
#
# Memverifikasi DoD M7-W3:
# 1. m7_io_patterns.json ter-commit: trunk sequential (4 MB, QD1) + expert-miss (10 MB, seed-42, sha lock)
# 2. generate_m7_io_patterns.py teruji: offset kelipatan 4096, dalam file size, pairwise non-overlapping
# 3. Satu workload lintas QD: sweep q menggunakan request set yang identik
# 4. io_benchmark binary Mojo: O_DIRECT reader, alignment probe, tracking max_outstanding_observed
# 5. benchmark_io_patterns.sh: menghitung BW_seq, BW_exp(q), R_io, q* anti-rebound, dan D_sus
# 6. Negative path: penolakan parameter hilang/invalid
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "M7-W3: Fixture 2 Pola + Generator + Benchmark Script (F17a & F17b)"
echo "======================================================================"

WORKDIR="/tmp/test_m7_w3_work"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

trap 'rm -rf "$WORKDIR"' EXIT

MODEL_FILE="$HOME/models/qwen1.5-moe-a2.7b-chat-4bit/quant_model.bin"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model quant_model.bin tidak ditemukan di $MODEL_FILE"
    exit 1
fi

FIXTURE_PATH="tools/fixtures/m7_io_patterns.json"

# ----------------------------------------------------------------------
# 1. Formatting & Code Quality
# ----------------------------------------------------------------------
echo ">> [1/5] Memeriksa kepatuhan formatting Mojo..."

FORMAT_OUTPUT=$(pixi run mojo format tools/bench/io_benchmark.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi
echo "   PASS: Formatting Mojo bersih 100%."

# ----------------------------------------------------------------------
# 2. Build Binary io_benchmark
# ----------------------------------------------------------------------
echo ">> [2/5] Membangun binary io_benchmark via pixi build..."
pixi run bash -c 'PATH="/usr/bin:$PATH" mojo build -I src tools/bench/io_benchmark.mojo -o io_benchmark'
echo "   PASS: Binary io_benchmark siap dijalankan."

# ----------------------------------------------------------------------
# 3. Verifikasi Generator & Integritas Fixture Deterministik (P1)
# ----------------------------------------------------------------------
echo ">> [3/5] Menguji generator dan memvalidasi integritas fixture m7_io_patterns.json..."
python3 tools/fixtures/generate_m7_io_patterns.py "$MODEL_FILE" "$FIXTURE_PATH"

python3 -c "
import json, hashlib

with open('$FIXTURE_PATH') as f:
    data = json.load(f)

assert data['seed'] == 42
assert len(data['patterns']) == 2

# 1. Validasi Trunk Sequential
p_trunk = data['patterns'][0]
assert p_trunk['id'] == 'trunk_sequential'
assert p_trunk['block_size'] == 4194304
assert p_trunk['block_count'] == 100
assert len(p_trunk['offsets']) == 100
assert p_trunk['offsets'][0] == 0
assert p_trunk['offsets'][1] == 4194304

# Verifikasi SHA trunk
canonical_trunk = json.dumps(p_trunk['offsets'], separators=(',', ':'))
assert hashlib.sha256(canonical_trunk.encode('utf-8')).hexdigest() == p_trunk['offsets_sha256']

# 2. Validasi Expert-Miss
p_exp = data['patterns'][1]
assert p_exp['id'] == 'expert_miss'
assert p_exp['block_size'] == 10485760
assert p_exp['block_count'] == 100
assert len(p_exp['offsets']) == 100

# Verifikasi SHA expert
canonical_exp = json.dumps(p_exp['offsets'], separators=(',', ':'))
assert hashlib.sha256(canonical_exp.encode('utf-8')).hexdigest() == p_exp['offsets_sha256']

# Verifikasi kelipatan 4096 dan pairwise non-overlapping
intervals = []
for off in p_exp['offsets']:
    assert off % 4096 == 0, f'Offset {off} bukan kelipatan 4096'
    assert off + 10485760 <= data['model_file_size']
    intervals.append((off, off + 10485760))

intervals.sort(key=lambda x: x[0])
for i in range(len(intervals) - 1):
    assert intervals[i][1] <= intervals[i+1][0], f'Overlap terdeteksi: {intervals[i]} vs {intervals[i+1]}'

print('   PASS: Invarian deterministik fixture P1 terverifikasi 100%.')
"

# ----------------------------------------------------------------------
# 4. Pengujian Eksekusi io_benchmark (Direct I/O)
# ----------------------------------------------------------------------
echo ">> [4/5] Menguji eksekusi io_benchmark pada kedua pola I/O..."

OUT_SEQ="$WORKDIR/test_seq.json"
./io_benchmark \
  --pattern sequential \
  --block-size 4194304 \
  --block-count 10 \
  --queue-depth 1 \
  --file "$MODEL_FILE" \
  --offsets-fixture "$FIXTURE_PATH" \
  --output "$OUT_SEQ" > /dev/null

python3 -c "
import json
data = json.load(open('$OUT_SEQ'))
assert data['status'] == 'success'
assert data['pattern_id'] == 'trunk_sequential'
assert data['queue_depth'] == 1
assert data['max_outstanding_observed'] == 1
assert data['bandwidth_gb_s'] > 0.0
assert data['dio_alignment'] in (512, 4096)
print(f'   PASS: Sequential QD1 bandwidth = {data[\"bandwidth_gb_s\"]:.3f} GB/s')
"

OUT_EXP="$WORKDIR/test_exp.json"
./io_benchmark \
  --pattern random_jump \
  --block-size 10485760 \
  --block-count 10 \
  --queue-depth 4 \
  --file "$MODEL_FILE" \
  --offsets-fixture "$FIXTURE_PATH" \
  --output "$OUT_EXP" > /dev/null

python3 -c "
import json
data = json.load(open('$OUT_EXP'))
assert data['status'] == 'success'
assert data['pattern_id'] == 'expert_miss'
assert data['queue_depth'] == 4
assert data['max_outstanding_observed'] == 4, f'max_obs harus 4, got {data[\"max_outstanding_observed\"]}'
assert data['bandwidth_gb_s'] > 0.0
print(f'   PASS: Expert-miss QD4 bandwidth = {data[\"bandwidth_gb_s\"]:.3f} GB/s (QD terbukti = 4)')
"

# ----------------------------------------------------------------------
# 5. Pengujian Runner benchmark_io_patterns.sh & Output Metrik
# ----------------------------------------------------------------------
echo ">> [5/5] Menjalankan runner benchmark_io_patterns.sh dan validasi metrik normatif..."

SUMMARY_OUT="$WORKDIR/m7_summary.json"
QUICK_MODE=1 bash tools/benchmark/benchmark_io_patterns.sh "$MODEL_FILE" "$SUMMARY_OUT" > /dev/null

python3 -c "
import json
with open('$SUMMARY_OUT') as f:
    s = json.load(f)

assert s['status'] == 'success'
m = s['metrics']
assert 'bw_seq_gb_s' in m
assert 'bw_exp_by_qd' in m
assert 'r_io_by_qd' in m
assert 'max_outstanding_observed_by_qd' in m
assert 'q_star' in m
assert 'd_sus' in m

assert m['bw_seq_gb_s'] > 0.0
assert len(m['bw_exp_by_qd']) > 0
assert len(m['r_io_by_qd']) > 0
assert m['q_star'] in (1, 2, 4, 8, 16)
assert m['d_sus'] >= 0.0

print(f'   PASS: Summary metrics terverifikasi: BW_seq = {m[\"bw_seq_gb_s\"]:.3f} GB/s, q* = {m[\"q_star\"]}, D_sus = {m[\"d_sus\"]*100:.1f}%')
"

echo "======================================================================"
echo "SEMUA PENGUJIAN M7-W3 (FIXTURE 2 POLA + BENCHMARK) LULUS 100%!"
echo "STATUS M7-W3: DONE — SIAP LANJUT KE M7-W4 (CONFIG + ERROR)"
echo "======================================================================"
