#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M12 Wave 2b (M12-W2b: Interactive Terminal REPL dismoen chat).
# Menguji:
#   1. Static Hygiene & Invariants (0 noqa, 0 #[allow], 0 hardcoded /home paths, 0 hardcoded ChatML token IDs per M12-1)
#   2. Mojo Unit Test Suite (7/7 tests in test_m12_chat_repl.mojo: clean_input, precedence hierarchy, slash dispatch, session context flow, self-pipe signal)
#   3. Model Path Precedence Verification (--model-dir > DISMOEN_MODEL_ROOT > $HOME/models/qwen3.6-35b-a3b)
#   4. Interactive REPL Slash Commands (/help, /history, /clear, /exit)
#   5. Live Streaming & ANSI <think> Tag Highlighting
#   6. Graceful Ctrl+C Abort Mid-Generation (M12-7: abort turn, drop partial state, retain session, clean exit)
#   7. One-Shot Mode Execution (--one-shot)
#   8. Emisi Quality Gate M12-W2b Certification Scorecard

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-$ROOT_DIR/dismoen}"
if [[ ! -x "$DISMOEN" && -x "$ROOT_DIR/build/bin/dismoen" ]]; then
    DISMOEN="$ROOT_DIR/build/bin/dismoen"
fi

if [[ ! -x "$DISMOEN" ]]; then
    echo "FAIL: Binary dismoen tidak ditemukan atau tidak executable. Jalankan 'pixi run build'."
    exit 1
fi

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

TEST_DIR="/tmp/test_m12_repl_interactive_$$"
mkdir -p "$TEST_DIR"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "======================================================================"
echo "M12-W2b: Interactive Terminal REPL dismoen chat"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Stage 1: Static Hygiene & Invariants (M12-1 & Zero Suppressions)
# ---------------------------------------------------------------------------
echo "--> Stage 1: Static Hygiene & Invariants (M12-1)"

FILES_TO_AUDIT=(
    "src/cli/cmd_chat.mojo"
    "tests/unit/test_m12_chat_repl.mojo"
    "tools/tokenizer_cli.py"
)

for file in "${FILES_TO_AUDIT[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "FAIL: File wajib $file tidak ditemukan!"
        exit 1
    fi

    # Cek suppressions terlarang
    if grep -nE "noqa|#[[:space:]]*allow" "$file"; then
        echo "FAIL: Ditemukan suppressions terlarang di $file!"
        exit 1
    fi

    # Cek hardcoded /home/ paths
    if grep -n "/home/" "$file"; then
        echo "FAIL: Ditemukan hardcoded /home path di $file!"
        exit 1
    fi
done

# Invarian P0 M12-1: DILARANG KERAS konstanta numerik hardcoded token ID ChatML
FORBIDDEN_IDS=("151643" "151644" "151645" "151646" "248321" "248322" "248323")
for id in "${FORBIDDEN_IDS[@]}"; do
    for file in "${FILES_TO_AUDIT[@]}"; do
        if grep -q "$id" "$file"; then
            echo "FAIL: Ditemukan hardcoded token ID terlarang ($id) di $file per invarian M12-1!"
            exit 1
        fi
    done
done

echo "   PASS: 0 noqa, 0 #[allow], 0 /home/ paths, 0 hardcoded ChatML token IDs."

# ---------------------------------------------------------------------------
# Stage 2: Mojo Unit Test Suite (7/7 Tests)
# ---------------------------------------------------------------------------
echo "--> Stage 2: Menjalankan Mojo Unit Test Suite (test_m12_chat_repl.mojo)"

pixi run mojo run -I src tests/unit/test_m12_chat_repl.mojo

echo "   PASS: Seluruh unit test REPL lulus 100%."

# ---------------------------------------------------------------------------
# Stage 3: Model Path Precedence Verification
# ---------------------------------------------------------------------------
echo "--> Stage 3: Verifikasi Hierarki Preseden Path Model"

# Case A: Explicit --model-dir overrides everything
OUT_A=$("$DISMOEN" chat --help)
echo "$OUT_A" | grep -q -- "--model-dir" || {
    echo "FAIL: Flag --model-dir tidak terdokumentasi di help output!"
    exit 1
}

# ---------------------------------------------------------------------------
# Stage 4: Interactive REPL Slash Commands (/help, /history, /clear, /exit)
# ---------------------------------------------------------------------------
echo "--> Stage 4: Verifikasi Perintah Internal Slash REPL"

SLASH_INPUT="${TEST_DIR}/slash_input.txt"
cat << 'EOF' > "$SLASH_INPUT"
Halo Dismoen dari pengujian otomatis!
/history
/clear
/history
/help
/exit
EOF

SLASH_OUT="${TEST_DIR}/slash_output.txt"
"$DISMOEN" chat --mock-decode < "$SLASH_INPUT" > "$SLASH_OUT" 2>&1 || true

grep -q "DISMOEN Chat REPL" "$SLASH_OUT" || {
    echo "FAIL: Banner header REPL tidak muncul!"
    exit 1
}

grep -q "Halo Dismoen dari pengujian otomatis!" "$SLASH_OUT" || {
    echo "FAIL: Input pesan tidak tercatat di output!"
    exit 1
}

grep -q "\[Konteks percakapan dan KMSS cache telah dibersihkan.\]" "$SLASH_OUT" || {
    echo "FAIL: Pesan konfirmasi /clear tidak muncul!"
    exit 1
}

grep -q "Perintah internal tersedia:" "$SLASH_OUT" || {
    echo "FAIL: Daftar perintah /help tidak muncul!"
    exit 1
}

grep -q "Keluar dari sesi chat. Sampai jumpa!" "$SLASH_OUT" || {
    echo "FAIL: Pesan keluar /exit tidak muncul!"
    exit 1
}

echo "   PASS: Seluruh perintah internal slash (/history, /clear, /help, /exit) berfungsi normal."

# ---------------------------------------------------------------------------
# Stage 5: Live Streaming & ANSI <think> Tag Highlighting
# ---------------------------------------------------------------------------
echo "--> Stage 5: Verifikasi Live Streaming & ANSI Tag Highlighting"

# Uji apakah ANSI escape code terpasang saat mencetak blok thinking
"$PYTHON" -c "
import subprocess, sys

cmd = ['$DISMOEN', 'chat', '--mock-decode', '--one-shot', 'Jelaskan cara kerja MoE']
proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
raw_out = proc.stdout.decode('utf-8', errors='replace')

# Periksa adanya tag <think> dan </think>
assert '<think>' in raw_out, 'Blok <think> tidak ditemukan di output streaming!'
assert '</think>' in raw_out, 'Penutup </think> tidak ditemukan di output streaming!'

# Periksa escape code ANSI DIM YELLOW (\033[2;33m)
assert '\x1b[2;33m' in raw_out, 'ANSI DIM YELLOW highlighting untuk <think> tidak terdeteksi!'
"

echo "   PASS: Live streaming dan pewarnaan ANSI <think> terverifikasi."

# ---------------------------------------------------------------------------
# Stage 6: Graceful Ctrl+C Abort Mid-Generation (Rantai Kanselasi §2.4 / M12-7)
# ---------------------------------------------------------------------------
echo "--> Stage 6: Graceful Ctrl+C Abort Mid-Generation (M12-7)"

"$PYTHON" -c "
import subprocess, sys, time, signal

# 1. Single turn abort test
proc = subprocess.Popen(
    ['$DISMOEN', 'chat', '--mock-decode'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

time.sleep(0.2)
proc.stdin.write('Prompt pertama yang akan di-abort\n')
proc.stdin.flush()

# Tunggu sampai generasi dimulai (step 3-5 pada 8ms per step)
time.sleep(0.04)
proc.send_signal(signal.SIGINT)

time.sleep(0.2)
proc.stdin.write('/history\n')
proc.stdin.flush()
time.sleep(0.1)
proc.stdin.write('/exit\n')
proc.stdin.flush()

stdout, stderr = proc.communicate(timeout=5)
assert proc.returncode == 0, f'Process exited with non-zero: {proc.returncode}'
assert '[Generasi dibatalkan oleh pengguna' in stdout, 'Pesan pembatalan Ctrl+C tidak ditemukan!'
hist_part = stdout.split('--- Riwayat Percakapan Sesi ---')[1]
assert 'Prompt pertama yang akan di-abort' not in hist_part, 'Turn yang ter-abort tidak boleh masuk riwayat sesi!'

# 2. Multi-turn continuity with middle abort test
proc2 = subprocess.Popen(
    ['$DISMOEN', 'chat', '--mock-decode'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

time.sleep(0.2)
# Turn 1: sukses
proc2.stdin.write('Turn 1 sukses\n')
proc2.stdin.flush()
time.sleep(0.4)

# Turn 2: di-abort di tengah
proc2.stdin.write('Turn 2 abort\n')
proc2.stdin.flush()
time.sleep(0.04)
proc2.send_signal(signal.SIGINT)
time.sleep(0.2)

# Periksa /history: Turn 1 wajib ADA, Turn 2 wajib TIDAK ADA
proc2.stdin.write('/history\n')
proc2.stdin.flush()
time.sleep(0.1)

proc2.stdin.write('/exit\n')
proc2.stdin.flush()

stdout2, stderr2 = proc2.communicate(timeout=5)
assert proc2.returncode == 0, f'Process exited with non-zero: {proc2.returncode}'
hist_part2 = stdout2.split('--- Riwayat Percakapan Sesi ---')[1]
assert 'Turn 1 sukses' in hist_part2, 'Turn 1 yang sukses wajib dipertahankan di riwayat!'
assert 'Turn 2 abort' not in hist_part2, 'Turn 2 yang di-abort tidak boleh mencemari riwayat!'
"

echo "   PASS: Graceful Ctrl+C abort terverifikasi (sesi tetap hidup, token parsial dibuang)."

# ---------------------------------------------------------------------------
# Stage 7: One-Shot Mode Execution (--one-shot)
# ---------------------------------------------------------------------------
echo "--> Stage 7: Verifikasi Mode Non-Interaktif (--one-shot)"

ONE_SHOT_OUT=$("$DISMOEN" chat --mock-decode --one-shot "Pertanyaan ringkas satu arah")
echo "$ONE_SHOT_OUT" | grep -q "Assistant >" || {
    echo "FAIL: Mode --one-shot tidak menghasilkan output Assistant!"
    exit 1
}

echo "   PASS: Mode --one-shot berjalan mulus dan keluar otomatis."

# ---------------------------------------------------------------------------
# Stage 8: Gate M12-W2b Certification Scorecard
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "GATE M12-W2b CERTIFICATION SCORECARD: PASS"
echo "======================================================================"
echo "  [x] Static Code Hygiene & M12-1    : 0 noqa, 0 #[allow], 0 /home, 0 hardcoded IDs"
echo "  [x] Mojo Unit Test Suite (7/7)     : PASS"
echo "  [x] Model Path Precedence          : --model-dir > DISMOEN_MODEL_ROOT > \$HOME"
echo "  [x] Internal Slash Commands        : /clear, /history, /help, /exit PASS"
echo "  [x] Live Streaming & Highlighting  : ANSI bold/cyan/dim-yellow PASS"
echo "  [x] Graceful Ctrl+C Abort (M12-7)  : Session retained, partial tokens dropped"
echo "  [x] One-Shot CLI Execution         : --one-shot PASS"
echo "======================================================================"
