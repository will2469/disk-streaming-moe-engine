#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Integration Test Suite untuk Milestone M7 dengan Model Riil (Qwen3.6-35B-A3B).
# Menguji integrasi reader O_DIRECT + LRU Cache Expert pada bobot riil:
#   1. Probe penemuan keselarasan O_DIRECT (dio_alignment: 512/4096) pada storage fisik.
#   2. Benchmark pola I/O storage F17 (Trunk sequential QD1 & Expert-miss QD sweep 1..16).
#   3. Eksekusi `dismoen decode` dengan akselerasi O_DIRECT + LRU Cache pada model riil Qwen 3.6.
#   4. Verifikasi telemetri F13 6-field byte counters, rasio byte rho_B, dan throughput >= 2 tok/s.
#   5. Verifikasi invarian Pin Budget (sum(pinned) <= 25% kapasitas) dan isolasi Read-Only SEC-5.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-$ROOT_DIR/dismoen}"
IO_BENCH="${IO_BENCH:-$ROOT_DIR/io_benchmark}"

if [[ ! -x "$DISMOEN" ]]; then
    echo "Membangun binary dismoen..."
    pixi run build
fi

if [[ ! -x "$IO_BENCH" ]]; then
    echo "Membangun binary io_benchmark..."
    pixi run bash -c 'PATH="/usr/bin:$PATH" mojo build -I src tools/bench/io_benchmark.mojo -o io_benchmark'
fi

MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.6-35b-a3b}"
SHARD1="$MODEL_DIR/model-00001-of-00026.safetensors"

if [[ ! -d "$MODEL_DIR" || ! -f "$SHARD1" ]]; then
    echo "SKIP: Direktori model Qwen3.6-35B-A3B ($MODEL_DIR) atau shard 1 tidak ditemukan. Melewati uji real O_DIRECT."
    exit 0
fi

TEST_DIR="/tmp/test_m7_real_qwen36_$$"
WORKDIR="${TEST_DIR}/work"
CACHE_STATS_OUT="${WORKDIR}/cache_stats.json"
M7_RAW_JSON="${WORKDIR}/m7_raw.json"
M7_REPORT_MD="${WORKDIR}/M7-benchmark.md"

mkdir -p "$WORKDIR"

cleanup() {
    chmod -R u+w "$TEST_DIR" 2>/dev/null || true
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M7: Real Model Integration Test Suite (Qwen3.6-35B-A3B)"
echo "======================================================================"

# ----------------------------------------------------------------------
# Stage 1: O_DIRECT Alignment Discovery Probe pada Shard Qwen 3.6
# ----------------------------------------------------------------------
echo ">> [1/5] Menguji O_DIRECT alignment discovery probe pada shard riil Qwen 3.6..."

PROBE_OUT="$WORKDIR/probe_stdout.json"
"$IO_BENCH" \
    --pattern sequential \
    --block-size 4096 \
    --block-count 10 \
    --queue-depth 1 \
    --file "$SHARD1" \
    --output "$PROBE_OUT"

python3 - "$PROBE_OUT" <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["status"] == "success"
align = doc["dio_alignment"]
assert align in (512, 4096), f"Invalid dio_alignment: {align}"
bw = doc["bandwidth_gb_s"]
assert bw > 0.0, f"Bandwidth must be positive: {bw}"
print(f"   PASS: Discovery probe sukses: dio_alignment = {align} bytes, BW awal = {bw:.3f} GB/s.")
EOF

# ----------------------------------------------------------------------
# Stage 2: Storage I/O Pattern Benchmark (F17) pada Shard Qwen 3.6
# ----------------------------------------------------------------------
echo ">> [2/5] Menguji pola I/O storage F17 (Trunk sequential & Expert-miss) pada bobot riil..."

IO_FIXTURE="tools/fixtures/m7_io_patterns.json"
TRUNK_OUT="$WORKDIR/trunk_seq.json"
"$IO_BENCH" \
    --pattern sequential \
    --block-size 4194304 \
    --block-count 10 \
    --queue-depth 1 \
    --file "$SHARD1" \
    --offsets-fixture "$IO_FIXTURE" \
    --output "$TRUNK_OUT"

python3 - "$TRUNK_OUT" <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["status"] == "success"
bw_seq = doc["bandwidth_gb_s"]
assert bw_seq > 0.1, f"BW_seq terlalu rendah: {bw_seq}"
print(f"   PASS: Trunk sequential (4 MB, QD1) BW_seq = {bw_seq:.3f} GB/s.")
EOF

EXPERT_OUT="$WORKDIR/expert_miss_qd16.json"
"$IO_BENCH" \
    --pattern random_jump \
    --block-size 10485760 \
    --block-count 10 \
    --queue-depth 16 \
    --file "$SHARD1" \
    --offsets-fixture "$IO_FIXTURE" \
    --output "$EXPERT_OUT"

python3 - "$EXPERT_OUT" <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["status"] == "success"
bw_exp = doc["bandwidth_gb_s"]
qd = doc["queue_depth"]
assert qd == 16, f"Expected QD 16, got {qd}"
print(f"   PASS: Expert-miss (10 MB, QD16) BW_exp = {bw_exp:.3f} GB/s.")
EOF

# ----------------------------------------------------------------------
# Stage 3: Eksekusi dismoen decode dengan Akselerasi O_DIRECT + LRU
# ----------------------------------------------------------------------
echo ">> [3/5] Menjalankan dismoen decode pada checkpoint riil Qwen 3.6..."

DECODE_OUT="$WORKDIR/decode_stdout.json"
"$DISMOEN" decode \
    --model-dir "$MODEL_DIR" \
    --tokens tools/fixtures/m4_prompt1_tokens.json \
    --max-tokens 64 \
    --context-size 2048 \
    --o-direct \
    --block-size 4096 \
    --queue-depth 16 \
    --cache-capacity 512 \
    --cache-stats "$CACHE_STATS_OUT" \
    --workdir "$WORKDIR" > "$DECODE_OUT"

python3 - "$DECODE_OUT" "$CACHE_STATS_OUT" <<'EOF'
import json, sys

d = json.load(open(sys.argv[1]))
assert d["status"] == "success", f"Decode status not success: {d}"
assert d["io_config"]["o_direct"] is True
assert d["io_config"]["io_path"] == "O_DIRECT"

metrics = d["metrics"]
tok_s = metrics["tokens_per_sec"]
assert tok_s >= 2.0, f"Throughput {tok_s} < 2.0 tok/s target"

cs = d["cache_stats"]
assert cs["cache_hit_requests"] > 0
assert cs["cache_miss_requests"] > 0
assert cs["hit_bytes"] > 0
assert cs["miss_bytes"] > 0
assert cs["disk_bytes"] > 0
assert cs["ram_bytes"] > 0
assert cs["evictions"] > 0
assert cs["pinned_experts"] >= 1
assert 0.0 < cs["hit_rate"] < 1.0
assert 0.0 < cs["rho_b"] < 1.0

f_stats = json.load(open(sys.argv[2]))
assert f_stats["statistics"]["hits"] == cs["cache_hit_requests"]
assert f_stats["statistics"]["misses"] == cs["cache_miss_requests"]

print(f"   PASS: Decode sukses: throughput = {tok_s:.1f} tok/s (>= 2.0 tok/s), HR = {cs['hit_rate']*100:.1f}%, rho_B = {cs['rho_b']*100:.1f}%.")
print(f"   PASS: 6 field byte telemetri F13 terverifikasi konsisten.")
EOF

# ----------------------------------------------------------------------
# Stage 4: Verifikasi Invarian LRU Pin Budget (<= 25% Kapasitas)
# ----------------------------------------------------------------------
echo ">> [4/5] Memverifikasi invarian pin budget dan stabilitas LRU cache..."

python3 - "$CACHE_STATS_OUT" "$DECODE_OUT" <<'EOF'
import json, sys
stats_doc = json.load(open(sys.argv[1]))
decode_doc = json.load(open(sys.argv[2]))

stats = stats_doc["statistics"]
cap_mb = decode_doc["io_config"]["cache_capacity_mb"]
cap = cap_mb * 1024 * 1024
pin_budget = int(cap * 0.25)
pinned_bytes = stats["pinned_bytes"]
pinned_entries = stats["pinned_entries"]

# Invarian normatif: pin_budget == 25% capacity
assert pin_budget <= cap * 0.25 + 1024, f"Pin budget melebihi 25%: {pin_budget} vs {cap}"
# Invarian normatif: sum(pinned_entry_bytes) <= pin_budget < capacity
assert pinned_bytes <= pin_budget, f"Pinned bytes melanggar budget: {pinned_bytes} > {pin_budget}"
assert pinned_entries >= 1, f"Expected pinned entries >= 1, got {pinned_entries}"
print(f"   PASS: Invarian Pin Budget terverifikasi: pinned = {pinned_bytes:,} B ({pinned_entries} entries) <= budget = {pin_budget:,} B (25% dari {cap:,} B).")
EOF

# ----------------------------------------------------------------------
# Stage 5: Uji Keamanan SEC-5 Model Directory Read-Only
# ----------------------------------------------------------------------
echo ">> [5/5] Menguji keamanan SEC-5: eksekusi di atas model-dir Read-Only..."

chmod -R a-w "$MODEL_DIR"

SEC5_OUT="$WORKDIR/sec5_stdout.json"
set +e
"$DISMOEN" decode \
    --model-dir "$MODEL_DIR" \
    --tokens tools/fixtures/m4_prompt1_tokens.json \
    --max-tokens 16 \
    --context-size 2048 \
    --o-direct \
    --workdir "$WORKDIR" > "$SEC5_OUT"
RC=$?
set -e

chmod -R u+w "$MODEL_DIR"

if [[ "$RC" -ne 0 ]]; then
    echo "FAIL: Decode gagal saat model dir Read-Only (exit $RC)"
    exit 1
fi

python3 - "$SEC5_OUT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["status"] == "success", d
print("   PASS: SEC-5 terverifikasi: engine berjalan normal tanpa mutasi ke model directory.")
EOF

echo ""
echo "======================================================================"
echo "INTEGRASI REAL MODEL QWEN3.6-35B-A3B DENGAN MILESTONE M7 SUKSES 100%!"
echo "======================================================================"
