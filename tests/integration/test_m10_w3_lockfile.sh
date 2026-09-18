#!/usr/bin/env bash
# ==============================================================================
# test_m10_w3_lockfile.sh — M10-W3b: Lockfile Identity (7-Field & Retire Port)
#
# Memverifikasi:
# 1. Single SSOT: models.lock.json 7-field ada di repo root; models.lock.port.json retired.
# 2. Zero legacy symbol: 0 referensi models.lock.port di src/.
# 3. 5 Uji Negatif Manifest Equality Fail-Closed:
#    - Shard hilang (missing shard)
#    - Shard ekstra (extra unexpected shard)
#    - Nama file salah (filename naming pattern mismatch)
#    - Count salah (shard count != 26)
#    - Size mismatch (file size differs from locked size)
#    - Anti-placeholder ("pinned" / "TBD" rejected)
# 4. Happy Path Manifest Equality.
# 5. Verifikasi Direktori Model Riil (bila ada).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DISMOEN="./dismoen"
if [[ ! -x "$DISMOEN" ]]; then
    echo "FAIL: Binary dismoen tidak ditemukan. Jalankan 'pixi run build' terlebih dahulu."
    exit 1
fi

PYTHON="${PYTHON:-python3}"
TEST_DIR="$(mktemp -d -t m10_w3_lockfile_XXXXXX)"
trap 'rm -rf "${TEST_DIR}"' EXIT

echo "======================================================================"
echo "M10-W3b: Lockfile Identity (7-Field & Retire Port) Verification Suite"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Stage 1: Single SSOT & Zero-Symbol Invariant (§4.5)
# -----------------------------------------------------------------------------
echo ">> [1/5] Memeriksa Single SSOT dan ketiadaan models.lock.port.json..."

if [[ ! -f "models.lock.json" ]]; then
    echo "FAIL: models.lock.json tidak ditemukan di root repositori!"
    exit 1
fi

if [[ -f "models.lock.port.json" ]]; then
    echo "FAIL: models.lock.port.json belum di-retire dari root repositori!"
    exit 1
fi

# Audit zero symbols in src/
PORT_REFS=$(git grep -n "models\.lock\.port" src/ 2>/dev/null || true)
if [[ -n "$PORT_REFS" ]]; then
    echo "FAIL: Ditemukan referensi 'models.lock.port' di src/:"
    echo "$PORT_REFS"
    exit 1
fi

echo "   PASS: Single SSOT aktif, models.lock.port.json retired, 0 referensi di src/."

# -----------------------------------------------------------------------------
# Stage 2: Validasi Skema 7-Field models.lock.json (§4.5)
# -----------------------------------------------------------------------------
echo ">> [2/5] Memvalidasi struktur 7-field dan ketiadaan placeholder pada models.lock.json..."

"$PYTHON" - <<'EOF'
import json, sys

with open("models.lock.json") as f:
    data = json.load(f)

# Field 1: model_id / model
model_id = data.get("model_id") or data.get("model")
assert model_id == "Qwen/Qwen3.6-35B-A3B", f"Invalid model_id: {model_id}"

# Field 2: architecture
arch = data.get("architecture")
assert arch == "qwen3.6", f"Invalid architecture: {arch}"

# Field 3: revision (HF commit)
rev = data.get("revision")
assert rev and len(rev) == 40, f"Invalid revision: {rev}"

# Field 4 & 5: shards manifest & measured sha256 (26 shards)
shards = data.get("shards")
assert isinstance(shards, list), "shards must be a list"
assert len(shards) == 26, f"Expected 26 shards, got {len(shards)}"

total_sz = 0
for i, s in enumerate(shards, 1):
    expected_fn = f"model-{i:05d}-of-00026.safetensors"
    fn = s.get("filename")
    assert fn == expected_fn, f"Shard {i} filename mismatch: {fn} != {expected_fn}"

    sz = s.get("size")
    assert isinstance(sz, int) and sz > 0, f"Invalid size for shard {fn}: {sz}"
    total_sz += sz

    h = s.get("sha256")
    assert h and h != "pinned" and not h.startswith("TBD"), f"Placeholder hash found in {fn}: {h}"
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h), f"Invalid sha256 in {fn}: {h}"

assert data.get("total_size") == total_sz, f"total_size mismatch: {data.get('total_size')} != {total_sz}"

# Field 6: tokenizer_revision & tokenizer_sha256
tok_rev = data.get("tokenizer_revision")
assert tok_rev and len(tok_rev) == 40, f"Invalid tokenizer_revision: {tok_rev}"
tok_sha = data.get("tokenizer_sha256")
assert isinstance(tok_sha, dict), "tokenizer_sha256 must be a dict"
assert "tokenizer.json" in tok_sha and len(tok_sha["tokenizer.json"]) == 64
assert "tokenizer_config.json" in tok_sha and len(tok_sha["tokenizer_config.json"]) == 64

# Field 7: config_hash
cfg_h = data.get("config_hash")
assert cfg_h and len(cfg_h) == 64 and all(c in "0123456789abcdef" for c in cfg_h), f"Invalid config_hash: {cfg_h}"

print(f"   PASS: 7-field schema valid (26 shards, total {total_sz} bytes, 0 pinned/TBD).")
EOF

# -----------------------------------------------------------------------------
# Stage 3: Setup Mock Model & 5 Uji Negatif Manifest Equality (§4.5)
# -----------------------------------------------------------------------------
echo ">> [3/5] Menyiapkan mock model dan mengeksekusi 5 uji negatif manifest equality..."

MOCK_MODEL_DIR="${TEST_DIR}/mock_qwen36"
mkdir -p "${MOCK_MODEL_DIR}"

# Mock config.json dengan vocab port 248320 dan text_config
if [[ -f "/home/will/models/qwen3.6-35b-a3b/config.json" ]]; then
    cp "/home/will/models/qwen3.6-35b-a3b/config.json" "${MOCK_MODEL_DIR}/config.json"
else
    cat <<EOF > "${MOCK_MODEL_DIR}/config.json"
{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "text_config": {
    "attention_bias": false,
    "full_attention_interval": 4,
    "head_dim": 32,
    "hidden_size": 128,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "model_type": "qwen3_5_moe_text",
    "moe_intermediate_size": 64,
    "num_attention_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 4,
    "num_key_value_heads": 1,
    "rms_norm_eps": 1e-6,
    "shared_expert_intermediate_size": 64,
    "vocab_size": 248320
  }
}
EOF
fi

# Buat 26 mock shard file masing-masing berukuran 1024 bytes
MOCK_SHARD_SIZE=1024
for i in $(seq 1 26); do
    fn=$(printf "model-%05d-of-00026.safetensors" "$i")
    head -c "$MOCK_SHARD_SIZE" /dev/zero > "${MOCK_MODEL_DIR}/${fn}"
done

# Generator mock lockfile
generate_mock_lock() {
    local target="$1"
    local count="${2:-26}"
    local size="${3:-1024}"
    local placeholder="${4:-}"
    "$PYTHON" -c "
import json
shards = []
for i in range(1, $count + 1):
    fn = f'model-{i:05d}-of-00026.safetensors'
    h = '${placeholder}' if '${placeholder}' else f'{i:064x}'
    shards.append({'filename': fn, 'size': $size, 'sha256': h})
data = {
    'model_id': 'Qwen/Qwen3.6-35B-A3B',
    'architecture': 'qwen3.6',
    'revision': '0'*40,
    'total_size': $count * $size,
    'config_hash': '0'*64,
    'tokenizer_revision': '0'*40,
    'tokenizer_sha256': {'tokenizer.json': '0'*64, 'tokenizer_config.json': '0'*64},
    'shards': shards
}
with open('$target', 'w') as f:
    json.dump(data, f, indent=2)
"
}

VALID_LOCK="${TEST_DIR}/valid.lock.json"
generate_mock_lock "$VALID_LOCK" 26 "$MOCK_SHARD_SIZE"

# --- Kasus 1: Shard hilang (Missing shard) ---
rm -f "${MOCK_MODEL_DIR}/model-00005-of-00026.safetensors"
set +e
OUT_ERR1=$("$DISMOEN" forward --check-config-only --model-dir "$MOCK_MODEL_DIR" --lock "$VALID_LOCK" 2>&1)
RET_ERR1=$?
set -e
if [[ $RET_ERR1 -eq 0 ]] || [[ "$OUT_ERR1" != *"MODEL_LOCK_TAMPER_DETECTED"* ]]; then
    echo "FAIL: Kasus 1 (Shard hilang) tidak memicu MODEL_LOCK_TAMPER_DETECTED!"
    echo "$OUT_ERR1"
    exit 1
fi
echo "   [3.1] PASS: Kasus 1 (Shard hilang) ditolak fail-closed."

# Pulihkan shard 5
head -c "$MOCK_SHARD_SIZE" /dev/zero > "${MOCK_MODEL_DIR}/model-00005-of-00026.safetensors"

# --- Kasus 2: Shard ekstra (Extra shard) ---
head -c "$MOCK_SHARD_SIZE" /dev/zero > "${MOCK_MODEL_DIR}/model-00027-of-00026.safetensors"
set +e
OUT_ERR2=$("$DISMOEN" forward --check-config-only --model-dir "$MOCK_MODEL_DIR" --lock "$VALID_LOCK" 2>&1)
RET_ERR2=$?
set -e
if [[ $RET_ERR2 -eq 0 ]] || [[ "$OUT_ERR2" != *"MODEL_LOCK_TAMPER_DETECTED"* ]]; then
    echo "FAIL: Kasus 2 (Shard ekstra) tidak memicu MODEL_LOCK_TAMPER_DETECTED!"
    echo "$OUT_ERR2"
    exit 1
fi
echo "   [3.2] PASS: Kasus 2 (Shard ekstra) ditolak fail-closed."

# Hapus shard ekstra
rm -f "${MOCK_MODEL_DIR}/model-00027-of-00026.safetensors"

# --- Kasus 3: Nama file salah (Filename mismatch) ---
mv "${MOCK_MODEL_DIR}/model-00001-of-00026.safetensors" "${MOCK_MODEL_DIR}/corrupt_name.safetensors"
set +e
OUT_ERR3=$("$DISMOEN" forward --check-config-only --model-dir "$MOCK_MODEL_DIR" --lock "$VALID_LOCK" 2>&1)
RET_ERR3=$?
set -e
if [[ $RET_ERR3 -eq 0 ]] || [[ "$OUT_ERR3" != *"MODEL_LOCK_TAMPER_DETECTED"* ]]; then
    echo "FAIL: Kasus 3 (Nama file salah) tidak memicu MODEL_LOCK_TAMPER_DETECTED!"
    echo "$OUT_ERR3"
    exit 1
fi
echo "   [3.3] PASS: Kasus 3 (Nama file salah) ditolak fail-closed."

# Pulihkan nama file
mv "${MOCK_MODEL_DIR}/corrupt_name.safetensors" "${MOCK_MODEL_DIR}/model-00001-of-00026.safetensors"

# --- Kasus 4: Jumlah shard salah (Count mismatch: 25 shard di lock) ---
LOCK_COUNT25="${TEST_DIR}/count25.lock.json"
generate_mock_lock "$LOCK_COUNT25" 25 "$MOCK_SHARD_SIZE"
set +e
OUT_ERR4=$("$DISMOEN" forward --check-config-only --model-dir "$MOCK_MODEL_DIR" --lock "$LOCK_COUNT25" 2>&1)
RET_ERR4=$?
set -e
if [[ $RET_ERR4 -eq 0 ]] || [[ "$OUT_ERR4" != *"MODEL_LOCK_TAMPER_DETECTED"* ]]; then
    echo "FAIL: Kasus 4 (Jumlah shard salah) tidak memicu MODEL_LOCK_TAMPER_DETECTED!"
    echo "$OUT_ERR4"
    exit 1
fi
echo "   [3.4] PASS: Kasus 4 (Jumlah shard salah) ditolak fail-closed."

# --- Kasus 5: Size mismatch ---
# Modifikasi ukuran shard 1 menjadi 2048 bytes
head -c 2048 /dev/zero > "${MOCK_MODEL_DIR}/model-00001-of-00026.safetensors"
set +e
OUT_ERR5=$("$DISMOEN" forward --check-config-only --model-dir "$MOCK_MODEL_DIR" --lock "$VALID_LOCK" 2>&1)
RET_ERR5=$?
set -e
if [[ $RET_ERR5 -eq 0 ]] || [[ "$OUT_ERR5" != *"MODEL_LOCK_TAMPER_DETECTED"* ]]; then
    echo "FAIL: Kasus 5 (Size mismatch) tidak memicu MODEL_LOCK_TAMPER_DETECTED!"
    echo "$OUT_ERR5"
    exit 1
fi
echo "   [3.5] PASS: Kasus 5 (Size mismatch) ditolak fail-closed."

# Pulihkan ukuran shard 1
head -c "$MOCK_SHARD_SIZE" /dev/zero > "${MOCK_MODEL_DIR}/model-00001-of-00026.safetensors"

# --- Kasus Tambahan: Anti-placeholder hash ('pinned') ---
LOCK_PINNED="${TEST_DIR}/pinned.lock.json"
generate_mock_lock "$LOCK_PINNED" 26 "$MOCK_SHARD_SIZE" "pinned"
set +e
OUT_PINNED=$("$DISMOEN" forward --check-config-only --model-dir "$MOCK_MODEL_DIR" --lock "$LOCK_PINNED" 2>&1)
RET_PINNED=$?
set -e
if [[ $RET_PINNED -eq 0 ]] || [[ "$OUT_PINNED" != *"MODEL_LOCK_TAMPER_DETECTED"* ]]; then
    echo "FAIL: Placeholder 'pinned' tidak ditolak fail-closed!"
    echo "$OUT_PINNED"
    exit 1
fi
echo "   [3.6] PASS: Anti-placeholder hash ('pinned') ditolak fail-closed."

# -----------------------------------------------------------------------------
# Stage 4: Happy Path Manifest Equality (§4.5)
# -----------------------------------------------------------------------------
echo ">> [4/5] Menguji Happy Path Manifest Equality pada mock model..."

OUT_HAPPY=$("$DISMOEN" forward --check-config-only --model-dir "$MOCK_MODEL_DIR" --lock "$VALID_LOCK" 2>&1)
if [[ "$OUT_HAPPY" != *"status"* ]] || [[ "$OUT_HAPPY" != *"success"* ]]; then
    echo "FAIL: Happy path manifest equality gagal!"
    echo "$OUT_HAPPY"
    exit 1
fi
echo "   PASS: Happy path manifest equality lulus 100%."

# -----------------------------------------------------------------------------
# Stage 5: Verifikasi Terhadap Model Riil Host (Bila Ada)
# -----------------------------------------------------------------------------
REAL_MODEL_DIR="/home/will/models/qwen3.6-35b-a3b"
if [[ -d "$REAL_MODEL_DIR" ]] && [[ -f "${REAL_MODEL_DIR}/config.json" ]]; then
    echo ">> [5/5] Memverifikasi model riil di ${REAL_MODEL_DIR} terhadap models.lock.json..."
    OUT_REAL=$("$DISMOEN" forward --check-config-only --model-dir "$REAL_MODEL_DIR" 2>&1)
    if [[ "$OUT_REAL" != *"status"* ]] || [[ "$OUT_REAL" != *"success"* ]]; then
        echo "FAIL: Verifikasi model riil terhadap models.lock.json gagal!"
        echo "$OUT_REAL"
        exit 1
    fi
    echo "   PASS: Model riil lolos verifikasi manifest equality SEC-1."
else
    echo ">> [5/5] Direktori model riil tidak ditemukan, dilewati secara jujur."
fi

echo ""
echo "======================================================================"
echo "SEMUA TAHAP VERIFIKASI M10-W3b LULUS 100%! Single SSOT 7-Field SEC-1 SIAP"
echo "======================================================================"
