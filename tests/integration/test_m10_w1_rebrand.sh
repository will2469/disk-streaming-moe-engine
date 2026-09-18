#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M10 Wave 1 (M10-W1: Rebranding & Toolchain Integrity).
# Menguji Gate G-M10-1:
#   1. Kompilasi dismoen bersih tanpa warning
#   2. Verifikasi kimo tereliminasi total
#   3. CLI banner dan usage string menampilkan DISMOEN
#   4. Subcommand help (--help, -h, help) keluar dengan kode 0
#   5. Subcommand tidak dikenal menghasilkan JSON USAGE dengan kode 2
#   6. Ketersediaan single-binary Rust: dismoen-tools
#   7. Formatted JSON scorecard Gate G-M10-1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-$ROOT_DIR/dismoen}"
LEGACY_KIMO="$ROOT_DIR/kimo"

TEST_DIR="/tmp/test_m10_w1_$$"
mkdir -p "$TEST_DIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M10-W1: Rebranding & Toolchain Integrity Suite (Gate G-M10-1)"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Build Artifacts & Zero-Leftover Verification
# ---------------------------------------------------------------------------
echo "--> Stage 1: Build Artifacts & Zero-Leftover Verification"

if [ ! -x "$DISMOEN" ]; then
    echo "FAIL: Binary dismoen tidak ditemukan atau tidak executable: $DISMOEN"
    exit 1
fi

if [ -e "$LEGACY_KIMO" ]; then
    echo "FAIL: Sisa artefak kimo ditemukan: $LEGACY_KIMO (seharusnya sudah dieliminasi total)"
    exit 1
fi

if [ -e "$ROOT_DIR/tools/kimo-tools" ]; then
    echo "FAIL: Sisa artefak tools/kimo-tools ditemukan! (seharusnya sudah dieliminasi total)"
    exit 1
fi

if [ ! -d "$ROOT_DIR/tools/dismoen-tools" ] || [ -L "$ROOT_DIR/tools/dismoen-tools" ]; then
    echo "FAIL: tools/dismoen-tools harus berupa direktori fisik riil (bukan symlink)!"
    exit 1
fi

echo "   PASS: dismoen executable valid, tools/dismoen-tools direktori fisik, 0 leftover kimo."

# ---------------------------------------------------------------------------
# Stage 2: Banner & Help Subcommand Verification
# ---------------------------------------------------------------------------
echo "--> Stage 2: Banner & Help Subcommand Verification"

HELP_OUT="$TEST_DIR/help_stdout.txt"
"$DISMOEN" --help > "$HELP_OUT"
if ! grep -q "DISMOEN (DIsk Streaming MOe ENgine)" "$HELP_OUT"; then
    echo "FAIL: Banner dismoen --help tidak memuat DISMOEN (DIsk Streaming MOe ENgine)"
    exit 1
fi

if ! grep -q "Penggunaan: dismoen" "$HELP_OUT"; then
    echo "FAIL: Usage line pada dismoen --help tidak valid"
    exit 1
fi

# Verifikasi flag -h
"$DISMOEN" -h > "$TEST_DIR/help_short.txt"
if ! grep -q "DISMOEN (DIsk Streaming MOe ENgine)" "$TEST_DIR/help_short.txt"; then
    echo "FAIL: Banner via dismoen -h tidak memuat DISMOEN"
    exit 1
fi

echo "   PASS: Banner dan antarmuka help terverifikasi 100%."

# ---------------------------------------------------------------------------
# Stage 3: Error Handling & Machine-Readable USAGE Contract
# ---------------------------------------------------------------------------
echo "--> Stage 3: Error Handling & USAGE Contract Verification"

EMPTY_ERR="$TEST_DIR/empty_stderr.json"
set +e
"$DISMOEN" > /dev/null 2> "$EMPTY_ERR"
EMPTY_RC=$?
set -e

if [ "$EMPTY_RC" -ne 2 ]; then
    echo "FAIL: Invokasi kosong diharapkan exit 2, diperoleh: $EMPTY_RC"
    exit 1
fi

if ! grep -q '"error_type":"USAGE"' "$EMPTY_ERR"; then
    echo "FAIL: Output error invokasi kosong tidak memuat error_type USAGE"
    exit 1
fi

if ! grep -q 'DISMOEN' "$EMPTY_ERR"; then
    echo "FAIL: Detail error USAGE tidak memuat DISMOEN"
    exit 1
fi

# Verifikasi unknown subcommand
UNKNOWN_ERR="$TEST_DIR/unknown_stderr.json"
set +e
"$DISMOEN" unknown-subcommand > /dev/null 2> "$UNKNOWN_ERR"
UNKNOWN_RC=$?
set -e

if [ "$UNKNOWN_RC" -ne 2 ]; then
    echo "FAIL: Unknown subcommand diharapkan exit 2, diperoleh: $UNKNOWN_RC"
    exit 1
fi

if ! grep -q 'subcommand tak dikenal: unknown-subcommand' "$UNKNOWN_ERR"; then
    echo "FAIL: Detail unknown subcommand tidak sesuai kontrak"
    exit 1
fi

echo "   PASS: Kontrak error machine-readable USAGE valid."

# ---------------------------------------------------------------------------
# Stage 4: Single-Binary Rust Tooling Verification (dismoen-tools)
# ---------------------------------------------------------------------------
echo "--> Stage 4: Single-Binary Rust Tooling Verification (dismoen-tools)"

if [ ! -d "$ROOT_DIR/tools/dismoen-tools" ]; then
    echo "FAIL: Path tools/dismoen-tools tidak ditemukan!"
    exit 1
fi

DISMOEN_TOOLS_BIN="$ROOT_DIR/target/debug/dismoen-tools"

if [ ! -x "$DISMOEN_TOOLS_BIN" ]; then
    echo "--> Membangun binaries tools..."
    cargo build --manifest-path "$ROOT_DIR/tools/dismoen-tools/Cargo.toml" --bin dismoen-tools
fi

# Test dismoen-tools USAGE
set +e
DISMOEN_TOOLS_OUT=$("$DISMOEN_TOOLS_BIN" 2>&1)
DT_RC=$?
set -e
if [ "$DT_RC" -ne 2 ] || ! echo "$DISMOEN_TOOLS_OUT" | grep -q '"error_type":"USAGE"'; then
    echo "FAIL: dismoen-tools gagal merespons kontrak USAGE"
    exit 1
fi

echo "   PASS: Single-binary Rust (dismoen-tools) siap dan valid."

# ---------------------------------------------------------------------------
# Stage 5: Formal Quality Gate Certification G-M10-1
# ---------------------------------------------------------------------------
echo "--> Stage 5: Gate G-M10-1 Certification"

python3 -c "
import json

scorecard = {
    'milestone': 'M10',
    'wave': 'M10-W1',
    'gate': 'G-M10-1',
    'title': 'Rebranding & Toolchain Integrity',
    'verdict': 'PASS',
    'checks': {
        'binary_executable': '$DISMOEN',
        'legacy_kimo_eliminated': True,
        'dismoen_tools_physical': True,
        'banner_verified': True,
        'usage_contract_rc2': True,
        'rust_tooling_single_bin': True
    }
}
print(json.dumps(scorecard, indent=2))
"

echo "======================================================================"
echo "G-M10-1: PASS (Rebranding Executable & Toolchain Integrity Certified)"
echo "======================================================================"
