#!/usr/bin/env bash
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
#
# Master Integration Test Suite untuk Milestone M10 Wave 2 (M10-W2: Storage Sanitization & Deterministic Byte Verification).
# Menguji Gate G-M10-2:
#   1. L_paths == 0 (kedua direktori model legacy tidak ada)
#   2. L_logical == 0 (sum stat(st_size) atas path legacy == 0)
#   3. Observational cross-check Delta B_free (toleran noise filesystem)
#   4. Pre-write operational margin B_free - B_required >= B_reserved (2 GiB)
#   5. Standar satuan SSOT GiB (IEC 2^30) dengan decimal GB SI parentetis
#   6. Formatted JSON scorecard Gate G-M10-2

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN_MODEL_ROOT="${DISMOEN_MODEL_ROOT:-$HOME/models}"
B_REQUIRED_GIB="${B_REQUIRED_GIB:-15.20}"
B_RESERVED_GIB="${B_RESERVED_GIB:-2.00}"
RECORD_FILE="${RECORD_FILE:-$ROOT_DIR/reports/2026-09-18/m10_w2_sanitization.json}"

echo "======================================================================"
echo "M10-W2: Storage Sanitization & Byte Verification (Gate G-M10-2)"
echo "======================================================================"
echo "Model root directory : $DISMOEN_MODEL_ROOT"
echo "Required buffer      : $B_REQUIRED_GIB GiB (target GGUF runtime)"
echo "Reserved margin      : $B_RESERVED_GIB GiB (safety headroom)"
echo "Record output path   : $RECORD_FILE"
echo "----------------------------------------------------------------------"

if [ ! -d "$DISMOEN_MODEL_ROOT" ]; then
    echo "ERROR: DISMOEN_MODEL_ROOT tidak ditemukan: $DISMOEN_MODEL_ROOT"
    exit 1
fi

python3 - "$DISMOEN_MODEL_ROOT" "$RECORD_FILE" "$B_REQUIRED_GIB" "$B_RESERVED_GIB" <<'EOF'
import os
import sys
import stat
import shutil
import json
import time

model_root = os.path.abspath(sys.argv[1])
record_file = os.path.abspath(sys.argv[2])
b_required_gib = float(sys.argv[3])
b_reserved_gib = float(sys.argv[4])

b_required_bytes = int(b_required_gib * (1024**3))
b_reserved_bytes = int(b_reserved_gib * (1024**3))

p1_name = "qwen1.5-moe-a2.7b-chat"
p2_name = "qwen1.5-moe-a2.7b-chat-4bit"
p1_path = os.path.join(model_root, p1_name)
p2_path = os.path.join(model_root, p2_name)

def gib(b):
    return b / (1024**3)

def gb_si(b):
    return b / (10**9)

def fmt_bytes(b):
    return f"{gib(b):.4f} GiB ({gb_si(b):.4f} GB SI) [{b:,} Bytes]"

def get_fs_avail_bytes(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize

def compute_logical_bytes(path):
    if not os.path.exists(path):
        return 0
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total

def ensure_writable_and_remove(path):
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path):
        for d in dirs:
            dp = os.path.join(root, d)
            try:
                os.chmod(dp, stat.S_IRWXU)
            except OSError:
                pass
        for f in files:
            fp = os.path.join(root, f)
            try:
                os.chmod(fp, stat.S_IRWXU)
            except OSError:
                pass
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:
        pass
    shutil.rmtree(path)

# ---------------------------------------------------------------------------
# Stage 1: Pre-Sanitization Audit & Baseline Measurement
# ---------------------------------------------------------------------------
print("--> Stage 1: Pre-Sanitization Audit & Baseline Measurement")

b_avail_before = get_fs_avail_bytes(model_root)
print(f"   Available space before sanitization : {fmt_bytes(b_avail_before)}")

p1_exists_initially = os.path.exists(p1_path)
p2_exists_initially = os.path.exists(p2_path)
p1_logical_bytes = compute_logical_bytes(p1_path)
p2_logical_bytes = compute_logical_bytes(p2_path)
total_legacy_bytes = p1_logical_bytes + p2_logical_bytes

prior_record = {}
prior_metrics = {}
if os.path.exists(record_file):
    try:
        with open(record_file, "r") as rf:
            prior_record = json.load(rf)
            prior_metrics = prior_record.get("storage_metrics", {})
    except Exception:
        pass

if p1_exists_initially or p2_exists_initially:
    print(f"   Legacy P1 ({p1_name}): {fmt_bytes(p1_logical_bytes)}")
    print(f"   Legacy P2 ({p2_name}): {fmt_bytes(p2_logical_bytes)}")
    print(f"   Total legacy models size : {fmt_bytes(total_legacy_bytes)}")
    expected_freed_threshold = 33 * (1024**3)
    if total_legacy_bytes >= expected_freed_threshold:
        print(f"   PASS: Legacy payload >= 33 GiB ({fmt_bytes(total_legacy_bytes)})")
    else:
        print(f"   WARNING: Legacy payload smaller than expected 33 GiB: {fmt_bytes(total_legacy_bytes)}")
else:
    print("   Legacy paths are already absent from filesystem.")
    prior_metrics = prior_record.get("storage_metrics", {})
    if prior_metrics.get("freed_logical_bytes"):
        total_legacy_bytes = prior_metrics["freed_logical_bytes"]
        print(f"   Prior recorded freed bytes : {fmt_bytes(total_legacy_bytes)}")

# ---------------------------------------------------------------------------
# Stage 2: Physical Removal of Legacy Directories
# ---------------------------------------------------------------------------
print("--> Stage 2: Physical Removal of Legacy Directories")

if p1_exists_initially:
    print(f"   Deleting {p1_path}...")
    ensure_writable_and_remove(p1_path)
    print("   P1 successfully removed.")
else:
    print(f"   P1 already purged: {p1_path}")

if p2_exists_initially:
    print(f"   Deleting {p2_path}...")
    ensure_writable_and_remove(p2_path)
    print("   P2 successfully removed.")
else:
    print(f"   P2 already purged: {p2_path}")

# ---------------------------------------------------------------------------
# Stage 3: Verification of Deterministic Invariants (L_paths == 0 & L_logical == 0)
# ---------------------------------------------------------------------------
print("--> Stage 3: Verification of Deterministic Invariants (L_paths & L_logical)")

p1_remains = os.path.exists(p1_path)
p2_remains = os.path.exists(p2_path)

l_paths = 0
if p1_remains:
    l_paths += 1
if p2_remains:
    l_paths += 1

l1_bytes = compute_logical_bytes(p1_path)
l2_bytes = compute_logical_bytes(p2_path)
l_logical = l1_bytes + l2_bytes

print(f"   Invariant L_paths   = {l_paths} (threshold == 0)")
print(f"   Invariant L_logical = {l_logical} Bytes (threshold == 0)")

if l_paths != 0:
    print(f"   FAIL: L_paths invariant violated: {l_paths} legacy paths still exist!", file=sys.stderr)
    sys.exit(1)
print("   PASS: L_paths == 0 satisfied (zero legacy directory exists).")

if l_logical != 0:
    print(f"   FAIL: L_logical invariant violated: {l_logical} bytes remaining in legacy paths!", file=sys.stderr)
    sys.exit(1)
print("   PASS: L_logical == 0 satisfied (zero logical bytes remaining in legacy paths).")

# ---------------------------------------------------------------------------
# Stage 4: Observational Cross-Check (Delta B_free)
# ---------------------------------------------------------------------------
print("--> Stage 4: Observational Cross-Check (Delta B_free)")

b_avail_after = get_fs_avail_bytes(model_root)
print(f"   Available space post-sanitization   : {fmt_bytes(b_avail_after)}")

if p1_exists_initially or p2_exists_initially:
    delta_b_free = b_avail_after - b_avail_before
else:
    delta_b_free = prior_metrics.get("delta_b_free_bytes", 0)

print(f"   Delta B_free (observational)        : {fmt_bytes(delta_b_free)}")
print("   (Note: Delta B_free is an observational metric subject to filesystem noise, not a gate failure criterion per §4.4)")

# ---------------------------------------------------------------------------
# Stage 5: Pre-Write Operational Margin Verification (§4.2)
# ---------------------------------------------------------------------------
print("--> Stage 5: Pre-Write Operational Margin Verification (§4.2)")

b_free_before_write = b_avail_after
b_free_expected_after = b_free_before_write - b_required_bytes
headroom_surplus = b_free_expected_after - b_reserved_bytes

print(f"   Current free space (B_free_before)  : {fmt_bytes(b_free_before_write)}")
print(f"   Declared write upper bound (B_req) : {fmt_bytes(b_required_bytes)}")
print(f"   Expected free after write           : {fmt_bytes(b_free_expected_after)}")
print(f"   Required reserve margin (B_res)     : {fmt_bytes(b_reserved_bytes)}")
print(f"   Headroom surplus over reserve       : {fmt_bytes(headroom_surplus)}")

if b_free_expected_after < b_reserved_bytes:
    print(f"   FAIL: Pre-write margin violation: {fmt_bytes(b_free_expected_after)} < {fmt_bytes(b_reserved_bytes)}", file=sys.stderr)
    sys.exit(1)

print(f"   PASS: Pre-write margin satisfied: B_free_after - B_required >= B_reserved ({fmt_bytes(b_free_expected_after)} >= {fmt_bytes(b_reserved_bytes)}).")

# ---------------------------------------------------------------------------
# Stage 6: Structured Scorecard Emission
# ---------------------------------------------------------------------------
print("--> Stage 6: Structured Scorecard Emission")

scorecard = {
    "gate": "G-M10-2",
    "status": "PASS",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_root": model_root,
    "legacy_paths_deleted": [p1_path, p2_path],
    "invariants": {
        "l_paths": l_paths,
        "l_logical_bytes": l_logical,
        "l_paths_pass": (l_paths == 0),
        "l_logical_pass": (l_logical == 0)
    },
    "storage_metrics": {
        "freed_logical_bytes": total_legacy_bytes,
        "freed_logical_gib": round(gib(total_legacy_bytes), 4),
        "freed_logical_gb_si": round(gb_si(total_legacy_bytes), 4),
        "delta_b_free_bytes": delta_b_free,
        "delta_b_free_gib": round(gib(delta_b_free), 4),
        "delta_b_free_gb_si": round(gb_si(delta_b_free), 4),
        "b_free_current_bytes": b_avail_after,
        "b_free_current_gib": round(gib(b_avail_after), 4),
        "b_free_current_gb_si": round(gb_si(b_avail_after), 4),
        "b_required_bytes": b_required_bytes,
        "b_required_gib": round(b_required_gib, 4),
        "b_reserved_bytes": b_reserved_bytes,
        "b_reserved_gib": round(b_reserved_gib, 4),
        "expected_free_after_write_bytes": b_free_expected_after,
        "expected_free_after_write_gib": round(gib(b_free_expected_after), 4),
        "headroom_surplus_bytes": headroom_surplus,
        "headroom_surplus_gib": round(gib(headroom_surplus), 4),
        "margin_satisfied": (b_free_expected_after >= b_reserved_bytes)
    }
}

os.makedirs(os.path.dirname(record_file), exist_ok=True)
with open(record_file, "w") as out_f:
    json.dump(scorecard, out_f, indent=2)
    out_f.write("\n")

print(f"   Scorecard successfully written to: {record_file}")
print("   Scorecard summary:")
print(json.dumps(scorecard, indent=2))
print("======================================================================")
print("GATE G-M10-2 (STORAGE SANITIZATION & BYTE VERIFICATION) PASSED 100%!")
print("======================================================================")
EOF
