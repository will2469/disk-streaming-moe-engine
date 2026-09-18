#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M11 Wave 2a (M11-W2a: Staging Buffer, Chunk Ring & O_DIRECT Probe Alignment §3.1).
# Menguji:
#   1. Static Hygiene & Zero Suppression (0 noqa, 0 #[allow], no fast-math flags)
#   2. Mojo Unit Test Suite (Probe, Triple Alignment, Round-up, Upfront Staging >= 128MB, Ring State Machine, Chunk Completion)
#   3. O_DIRECT Probe Alignment pada File Riil Host (statx STATX_DIOALIGN + fallback constraint)
#   4. Invarian Staging Memory upfront (M_staging >= 128 MiB, buffer_count == 2)
#   5. Formal Scorecard M11-W2a (§3.1 & P1-3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "M11-W2a: Staging Memory, Chunk Ring & Probe Alignment (§3.1 & P1-3)"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Hygiene & Zero Suppression (§3.1 Invariants)
# ---------------------------------------------------------------------------
echo "--> Stage 1: Static Hygiene & Zero Suppression"

for file in "src/io/dio_probe.mojo" "src/io/staging_ring.mojo" "tests/unit/test_m11_w2a_staging.mojo"; do
    if grep -nE "noqa|#[[:space:]]*allow" "$file"; then
        echo "FAIL: Ditemukan suppressions terlarang di $file!"
        exit 1
    fi
done

for flag in "-ffast-math" "-fassociative-math" "-freciprocal-math"; do
    if grep -rn -- "$flag" pixi.toml src/io/; then
        echo "FAIL: Dilarang menggunakan flag floating-point non-asosiatif: $flag!"
        exit 1
    fi
done

echo "   PASS: 0 noqa, 0 #[allow], dan 0 flag fast-math terdeteksi."

# ---------------------------------------------------------------------------
# Stage 2: Mojo Unit Test Suite Execution
# ---------------------------------------------------------------------------
echo "--> Stage 2: Mojo Unit Test Suite (Probe, Triple Alignment, Staging Upfront, Ring State Machine)"

pixi run mojo run -I src tests/unit/test_m11_w2a_staging.mojo

echo "   PASS: Seluruh unit test M11-W2a lolos 100%."

TMP_DIR="/tmp/test_m11_w2a_staging_$$"
mkdir -p "$TMP_DIR"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Stage 3: Live O_DIRECT Probe Verification on Storage Assets
# ---------------------------------------------------------------------------
echo "--> Stage 3: Live O_DIRECT Probe Verification"

cat << 'EOF' > "$TMP_DIR/probe_runner.mojo"
from io.dio_probe import (
    calculate_buffer_capacity,
    calculate_chunk_size,
    probe_dio_alignment,
)

def main() raises:
    var align = probe_dio_alignment("fixtures/m9_port_config_mini.json")
    print("A_mem:", align.mem_align)
    print("A_off:", align.offset_align)
    print("A_len:", align.length_align)
    print("source:", align.source)
    print("chunk_size:", calculate_chunk_size(align.required_alignment()))
    print("buffer_capacity:", calculate_buffer_capacity(align.required_alignment()))
EOF

PROBE_OUTPUT=$(pixi run mojo run -I src "$TMP_DIR/probe_runner.mojo")

echo "$PROBE_OUTPUT"

echo "$PROBE_OUTPUT" | grep -q "A_mem:" || {
    echo "FAIL: Probe A_mem tidak ter-emit!"
    exit 1
}

echo "$PROBE_OUTPUT" | grep -q "chunk_size: 3493888" || {
    echo "FAIL: Target chunk size profil 4096B bukan 3,493,888 B!"
    exit 1
}

echo "$PROBE_OUTPUT" | grep -q "buffer_capacity: 67108864" || {
    echo "FAIL: Target buffer capacity bukan 64 MiB (67,108,864 B)!"
    exit 1
}

echo "   PASS: Live probe O_DIRECT storage berhasil dan konsisten terhadap profil target."

# ---------------------------------------------------------------------------
# Stage 4: Formal Scorecard M11-W2a (§3.1 & P1-3)
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "M11-W2a Staging & Ring Scheduling Scorecard: ALL GATES PASS"
echo "======================================================================"
echo "  [x] M_staging >= 128 MiB (64 MiB x 2) upfront contiguous, required-aligned"
echo "  [x] buffer_count = 2 (Slot tahap Buffer A dan Buffer B)"
echo "  [x] Invarian §3.1: Ring slots tidak mengalokasikan memori di luar staging"
echo "  [x] State machine ring: EMPTY -> IO_IN_FLIGHT -> READY -> COMPUTING -> EMPTY"
echo "  [x] Exclusive ownership: I/O write (IO_IN_FLIGHT) vs CPU read (COMPUTING)"
echo "  [x] Per-chunk completion condition (bytes_transferred == chunk_size)"
echo "  [x] Target profile S_chunk = 3,493,888 B (853 x 4096 B)"
echo "  [x] O_DIRECT probe (A_mem, A_off, A_len) via statx STATX_DIOALIGN + fallback"
echo "  [x] Triple alignment verification (addr, offset, length)"
echo "  [x] Round-up chunk & buffer capacity jika probe != 4096 B"
echo "======================================================================"
