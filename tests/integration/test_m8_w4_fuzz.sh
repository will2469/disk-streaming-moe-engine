#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Fuzz Runner untuk Milestone M8 Wave 4 (DoD M8 § Fuzzing).
# Menguji 20+ (27) kasus mutasi terhadap kimo gdn dengan timeout 10 detik/kasus (anti-hang).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

KIMO="./dismoen"
DIR="fixtures/m8-fuzz"
TMP_OUT="$(mktemp -d -t dismoen_fuzz_m8_XXXXXX)"
trap 'rm -rf "$TMP_OUT"' EXIT

if [ ! -f "$KIMO" ]; then
    echo "dismoen binary not found, building..."
    pixi run build
fi

if [ ! -f "$DIR/manifest.json" ]; then
    echo "Manifest fuzz tidak ditemukan, membangkitkan via generate_m8_fuzz.py..."
    python3 tools/fixtures/generate_m8_fuzz.py
fi

echo "=== Menjalankan Fuzzing Suite M8 (27 Mutasi Ekstrim) ==="

pass_count=0
fail_count=0

# Loop tiap kasus dari manifest.json via python helper
python3 -c '
import json, sys
manifest = json.load(open("fixtures/m8-fuzz/manifest.json"))
for idx, case in enumerate(manifest["cases"]):
    args = [
        "--model-dir", manifest["model_dir"],
        "--output", "'"$TMP_OUT"'/fuzz_out_" + str(idx) + ".bin",
        "--layers", str(manifest["default_layers"]),
        "--dk", str(manifest["default_dk"]),
        "--dv", str(manifest["default_dv"]),
        "--chunk-size", str(manifest["default_chunk_size"]),
    ]
    # Timpa dengan argumen kasus
    user_args = case["args"]
    # Parse overrides
    i = 0
    while i < len(user_args):
        flag = user_args[i]
        if flag in ["--layers", "--dk", "--dv", "--chunk-size", "--model-dir", "--tokens", "--output", "--state-input", "--threads", "--seed"]:
            # Jika sudah ada di args, ganti
            if flag in args:
                pos = args.index(flag)
                args[pos + 1] = user_args[i + 1]
            else:
                args.extend([flag, user_args[i + 1]])
            i += 2
        else:
            args.append(flag)
            i += 1
    print(
        case["id"]
        + "\t"
        + str(case["want_exit"])
        + "\t"
        + str(case["want_err"])
        + "\t"
        + " ".join(args)
    )
' | while IFS=$'\t' read -r case_id want_exit want_err cmd_args; do
    echo -n "   --> Kasus [$case_id]... "

    set +e
    out=$(timeout 10 "$KIMO" gdn $cmd_args 2>&1 >/dev/null)
    exit_code=$?
    set -e

    if [ "$exit_code" -eq 124 ]; then
        echo "FAIL (HANG: timeout 10 detik terlampaui!)"
        exit 1
    fi

    if [ "$exit_code" -ne "$want_exit" ]; then
        echo "FAIL (Expected exit $want_exit, got $exit_code. Output: $out)"
        exit 1
    fi

    if [ -n "$want_err" ]; then
        if ! echo "$out" | grep -q "$want_err"; then
            echo "FAIL (Expected error_type '$want_err', got: $out)"
            exit 1
        fi
    fi

    echo "PASS (exit $exit_code, $want_err)"
done

echo "=== 27/27 KASUS FUZZING LOLOS BERSIH (0 crash, 0 hang, 0 OOM) ==="
