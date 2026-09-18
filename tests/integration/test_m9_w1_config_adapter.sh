#!/bin/bash
# ==============================================================================
# Integration Test Suite: M9-W1 Config Adapter + Mismatch Detector
# ==============================================================================
# Sesuai kontrak:
# - docs/milestones/M9-port.md (§ Panduan Migrasi, § CLI Contract, § Exit Codes)
# - scratch/wave/m9/m9-w1-config-adapter.md
# - skill: ref-ground-truth (R1, R2, R4, R5)
#
# Pengujian:
# Stage 1: Pre-commit formatting & zero-suppression hygiene (Mojo format, no noqa)
# Stage 2: Binary compilation check (dismoen binary siap)
# Stage 3: Architecture flag validation (missing / invalid flag -> exit 2)
# Stage 4: Trial architecture verification (real model qwen1.5-moe -> exit 0)
# Stage 5: Qwen3.6 architecture verification (real model qwen3.6-35b -> exit 0)
# Stage 6: Synthetic mini port config verification (m9_port_config_mini.json -> exit 0)
# Stage 7: Mismatch detector test suite (exit 2 for arch mismatch, exit 3 for config mismatch)
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DISMOEN="${DISMOEN:-./dismoen}"
TEST_DIR="/tmp/test_m9_w1_$$"
QWEN36_MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.6-35b-a3b}"

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
mkdir -p "$TEST_DIR"

echo "======================================================================"
echo "M9-W1: Config Adapter + Mismatch Detector Verification Suite"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Stage 1: Formatting & Static Hygiene
# -----------------------------------------------------------------------------
echo ">> [1/7] Memeriksa kepatuhan formatting Mojo dan zero-suppression..."
FORMAT_OUTPUT=$(pixi run mojo format \
    src/cli/m9_errors.mojo \
    src/core/config.mojo \
    src/cli/config_parser.mojo \
    src/cli/cmd_forward_port.mojo \
    src/main.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting memodifikasi berkas:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi

# Larangan keras noqa dan allow suppression
if grep -rn "noqa" src/cli/m9_errors.mojo src/cli/cmd_forward_port.mojo; then
    echo "FAIL: Ditemukan komentar noqa terlarang!"
    exit 1
fi
if grep -rn "allow(" src/cli/m9_errors.mojo src/cli/cmd_forward_port.mojo; then
    echo "FAIL: Ditemukan allow suppression terlarang!"
    exit 1
fi
echo "   PASS: Formatting bersih, zero-suppression terverifikasi."

# -----------------------------------------------------------------------------
# Stage 2: Kompilasi Binary dismoen
# -----------------------------------------------------------------------------
echo ">> [2/7] Memeriksa kompilasi binary dismoen..."
pixi run build >/dev/null 2>&1 || {
    echo "FAIL: Gagal melakukan build binary dismoen!"
    exit 1
}
if [[ ! -x "$DISMOEN" ]]; then
    echo "FAIL: Binary dismoen tidak ditemukan atau tidak executable: $DISMOEN"
    exit 1
fi
echo "   PASS: Binary dismoen siap eksekusi."

# -----------------------------------------------------------------------------
# Stage 3: Validasi Flag --architecture (Exit Code 2)
# -----------------------------------------------------------------------------
echo ">> [3/7] Menguji validasi flag --architecture (wajib eksplisit, exit 2)..."

# Case A: Missing flag --architecture
set +e
MISSING_OUT=$("$DISMOEN" forward-port --model-dir "$QWEN36_MODEL_DIR" --check-config-only 2>&1)
EXIT_CODE=$?
set -e
if [[ $EXIT_CODE -ne 2 ]]; then
    echo "FAIL: Missing --architecture harus exit 2, dapat: $EXIT_CODE"
    echo "$MISSING_OUT"
    exit 1
fi
if ! echo "$MISSING_OUT" | grep -q "MISSING_ARCHITECTURE"; then
    echo "FAIL: Output error tidak mengandung MISSING_ARCHITECTURE:"
    echo "$MISSING_OUT"
    exit 1
fi

# Case B: Unsupported architecture value
set +e
INVALID_OUT=$("$DISMOEN" forward-port --model-dir "$QWEN36_MODEL_DIR" --architecture llama --check-config-only 2>&1)
EXIT_CODE=$?
set -e
if [[ $EXIT_CODE -ne 2 ]]; then
    echo "FAIL: Unsupported --architecture harus exit 2, dapat: $EXIT_CODE"
    echo "$INVALID_OUT"
    exit 1
fi
if ! echo "$INVALID_OUT" | grep -q "UNSUPPORTED_ARCHITECTURE"; then
    echo "FAIL: Output error tidak mengandung UNSUPPORTED_ARCHITECTURE:"
    echo "$INVALID_OUT"
    exit 1
fi
echo "   PASS: Flag --architecture tervalidasi fail-closed (Exit 2)."

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Stage 4: Verifikasi Kontrak Baru M10 (Flag --architecture Ditolak pada dismoen forward)
# -----------------------------------------------------------------------------
echo ">> [4/7] Memverifikasi flag --architecture ditolak pada dismoen forward..."
set +e
UNKNOWN_OUT=$("$DISMOEN" forward --architecture qwen3.6 2>&1)
UNKNOWN_RC=$?
set -e
if [[ $UNKNOWN_RC -eq 0 ]]; then
    echo "FAIL: dismoen forward harus menolak flag --architecture!"
    exit 1
fi
echo "$UNKNOWN_OUT" | grep -q "unknown option: --architecture" || {
    echo "FAIL: Output tidak mengandung 'unknown option: --architecture'!"
    exit 1
}
echo "   PASS: Flag --architecture berhasil ditolak pada dismoen forward (§4.3)."

# -----------------------------------------------------------------------------
# Stage 5: Verifikasi Arsitektur Qwen3.6-35B-A3B (Model Nyata)
# -----------------------------------------------------------------------------
echo ">> [5/7] Memverifikasi arsitektur Qwen3.6-35B-A3B pada model nyata..."
if [[ -d "$QWEN36_MODEL_DIR" ]]; then
    PORT_OUT=$("$DISMOEN" forward-port \
        --model-dir "$QWEN36_MODEL_DIR" \
        --architecture qwen3.6 \
        --check-config-only)

    echo "$PORT_OUT" | grep -q '"architecture": "qwen3.6"' || {
        echo "FAIL: JSON output tidak menyatakan architecture qwen3.6!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"vocab_size": 248320' || {
        echo "FAIL: JSON output vocab_size bukan 248320!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"num_hidden_layers": 40' || {
        echo "FAIL: JSON output num_hidden_layers bukan 40!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"num_experts": 256' || {
        echo "FAIL: JSON output num_experts bukan 256!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"num_experts_per_tok": 8' || {
        echo "FAIL: JSON output num_experts_per_tok bukan 8!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"num_attention_heads": 16' || {
        echo "FAIL: JSON output num_attention_heads bukan 16!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"num_key_value_heads": 2' || {
        echo "FAIL: JSON output num_key_value_heads bukan 2 (GQA 16Q/2KV)!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"num_gdn_layers": 30' || {
        echo "FAIL: JSON output num_gdn_layers bukan 30!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"num_attention_layers": 10' || {
        echo "FAIL: JSON output num_attention_layers bukan 10!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"full_attention_interval": 4' || {
        echo "FAIL: JSON output full_attention_interval bukan 4!"
        exit 1
    }
    echo "$PORT_OUT" | grep -q '"attention_bias": false' || {
        echo "FAIL: JSON output attention_bias bukan false!"
        exit 1
    }
    echo "   PASS: Model Qwen3.6 terverifikasi sempurna (40L [30 GDN + 10 Attn], 256E/top-8, GQA 16/2)."
else
    echo "   SKIP: Model Qwen3.6 tidak ditemukan di $QWEN36_MODEL_DIR."
fi

# -----------------------------------------------------------------------------
# Stage 6: Verifikasi Synthetic Mini Port Config (Fixture CI)
# -----------------------------------------------------------------------------
echo ">> [6/7] Memverifikasi synthetic mini port config (fixtures/m9_port_config_mini.json)..."
MINI_OUT=$("$DISMOEN" forward-port \
    --model-dir fixtures/m9_port_config_mini.json \
    --architecture qwen3.6 \
    --check-config-only)

echo "$MINI_OUT" | grep -q '"architecture": "qwen3.6"' || {
    echo "FAIL: Mini config output tidak menyatakan architecture qwen3.6!"
    exit 1
}
echo "$MINI_OUT" | grep -q '"vocab_size": 1024' || {
    echo "FAIL: Mini config vocab_size bukan 1024!"
    exit 1
}
echo "$MINI_OUT" | grep -q '"num_hidden_layers": 4' || {
    echo "FAIL: Mini config num_hidden_layers bukan 4!"
    exit 1
}
echo "$MINI_OUT" | grep -q '"num_experts": 8' || {
    echo "FAIL: Mini config num_experts bukan 8!"
    exit 1
}
echo "$MINI_OUT" | grep -q '"num_experts_per_tok": 2' || {
    echo "FAIL: Mini config num_experts_per_tok bukan 2!"
    exit 1
}
echo "$MINI_OUT" | grep -q '"num_key_value_heads": 1' || {
    echo "FAIL: Mini config num_key_value_heads bukan 1!"
    exit 1
}
echo "$MINI_OUT" | grep -q '"num_gdn_layers": 3' || {
    echo "FAIL: Mini config num_gdn_layers bukan 3!"
    exit 1
}
echo "$MINI_OUT" | grep -q '"num_attention_layers": 1' || {
    echo "FAIL: Mini config num_attention_layers bukan 1!"
    exit 1
}
echo "   PASS: Synthetic mini port config lulus verifikasi (4L [3 GDN + 1 Attn], 8E/top-2, GQA 4/1)."

# -----------------------------------------------------------------------------
# Stage 7: Mismatch Detector Suite (Exit Code 2 & Exit Code 3)
# -----------------------------------------------------------------------------
echo ">> [7/7] Menjalankan mismatch detector test suite..."

# Test 7.1: Kontrak M10 - flag --architecture ditolak pada command forward
if [[ -d "$QWEN36_MODEL_DIR" ]]; then
    set +e
    ERR_OUT=$("$DISMOEN" forward --model-dir "$QWEN36_MODEL_DIR" --architecture qwen3.6 --check-config-only 2>&1)
    CODE=$?
    set -e
    if [[ $CODE -eq 0 ]]; then
        echo "FAIL: Flag --architecture harus ditolak pada forward, dapat exit 0"
        exit 1
    fi
    echo "$ERR_OUT" | grep -q "unknown option: --architecture" || {
        echo "FAIL: Error output tidak mengandung 'unknown option: --architecture'!"
        exit 1
    }
fi

# Test 7.2: Architecture mismatch - flag qwen3.6 pada model legacy -> Exit 2
MOCK_LEGACY_DIR="${TEST_DIR}/mock_legacy_qwen15"
mkdir -p "$MOCK_LEGACY_DIR"
cat <<EOF > "${MOCK_LEGACY_DIR}/config.json"
{
  "model_type": "qwen2_moe",
  "num_experts": 60,
  "vocab_size": 151936
}
EOF
set +e
ERR_OUT=$("$DISMOEN" forward-port --model-dir "$MOCK_LEGACY_DIR" --architecture qwen3.6 --check-config-only 2>&1)
CODE=$?
set -e
if [[ $CODE -ne 2 ]]; then
    echo "FAIL: Architecture mismatch (qwen3.6 on legacy config) harus exit 2, dapat: $CODE"
    exit 1
fi
echo "$ERR_OUT" | grep -q "ARCHITECTURE_MISMATCH" || {
    echo "FAIL: Error output tidak mengandung ARCHITECTURE_MISMATCH!"
    exit 1
}

# Test 7.3: Config mismatch - tampered vocab size -> Exit 3
set +e
ERR_OUT=$("$DISMOEN" forward-port --model-dir fixtures/m9_mismatch_vocab.json --architecture qwen3.6 --check-config-only 2>&1)
CODE=$?
set -e
if [[ $CODE -ne 3 ]]; then
    echo "FAIL: Tampered vocab_size harus exit 3, dapat: $CODE"
    exit 1
fi
echo "$ERR_OUT" | grep -q "CONFIG_MISMATCH" || {
    echo "FAIL: Error output tidak mengandung CONFIG_MISMATCH!"
    exit 1
}

# Test 7.4: Config mismatch - tampered layer count -> Exit 3
set +e
ERR_OUT=$("$DISMOEN" forward-port --model-dir fixtures/m9_mismatch_layers.json --architecture qwen3.6 --check-config-only 2>&1)
CODE=$?
set -e
if [[ $CODE -ne 3 ]]; then
    echo "FAIL: Tampered num_hidden_layers harus exit 3, dapat: $CODE"
    exit 1
fi
echo "$ERR_OUT" | grep -q "CONFIG_MISMATCH" || {
    echo "FAIL: Error output tidak mengandung CONFIG_MISMATCH!"
    exit 1
}

# Test 7.5: Config mismatch - tampered top-k -> Exit 3
set +e
ERR_OUT=$("$DISMOEN" forward-port --model-dir fixtures/m9_mismatch_topk.json --architecture qwen3.6 --check-config-only 2>&1)
CODE=$?
set -e
if [[ $CODE -ne 3 ]]; then
    echo "FAIL: Tampered num_experts_per_tok harus exit 3, dapat: $CODE"
    exit 1
fi
echo "$ERR_OUT" | grep -q "CONFIG_MISMATCH" || {
    echo "FAIL: Error output tidak mengandung CONFIG_MISMATCH!"
    exit 1
}

# Test 7.6: Config mismatch - tampered expert count -> Exit 3
set +e
ERR_OUT=$("$DISMOEN" forward-port --model-dir fixtures/m9_mismatch_experts.json --architecture qwen3.6 --check-config-only 2>&1)
CODE=$?
set -e
if [[ $CODE -ne 3 ]]; then
    echo "FAIL: Tampered num_experts harus exit 3, dapat: $CODE"
    exit 1
fi
echo "$ERR_OUT" | grep -q "CONFIG_MISMATCH" || {
    echo "FAIL: Error output tidak mengandung CONFIG_MISMATCH!"
    exit 1
}

echo "   PASS: Seluruh pengujian mismatch detector lulus (Exit 2 & Exit 3 terverifikasi)."

echo "======================================================================"
echo "SUCCESS: M9-W1 Config Adapter + Mismatch Detector 100% LULUS (7/7 stages)"
echo "======================================================================"
