#!/bin/bash
# ==============================================================================
# test_m6_w1_format.sh — Integration Test Suite M6-W1 (Custom Quant Format)
#
# Memverifikasi DoD M6-W1:
# 1. Header JSON tepat 256 byte terpadatkan (version, model, quantization, num_tensors, total_bytes).
# 2. Metadata per tensor (name, shape, dtype, group_size, num_groups, scale_offset, data_offset).
# 3. Layout biner: scales FP16 kontigu + weights 4-bit packed kontigu.
# 4. Packing 4-bit Little-Endian 2 bobot/byte ([w1][w0], range -8..7); skala s_g = ceil_FP16(max|w|/7).
# 5. Validasi integritas format: header valid, nama/shape/group_size cocok, ukuran file eksak,
#    scales finite, values dalam rentang [-8, 7].
# 6. Contoh 1 tensor [2048, 2048] -> 2.162.688 B (rasio ~3.88x) terhitung tangan dan teruji.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TEST_DIR="/tmp/test_m6_w1_format_$$"
cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT
mkdir -p "$TEST_DIR"

echo "======================================================================"
echo "M6-W1: Custom Quantization File Format, Packing & Validation"
echo "======================================================================"

# ----------------------------------------------------------------------
# 1. Formatting & Code Quality
# ----------------------------------------------------------------------
echo ">> [1/5] Memeriksa kepatuhan formatting (Mojo & Python)..."

# Mojo format check
FORMAT_OUTPUT=$(pixi run mojo format \
    src/format/quant_format.mojo \
    src/format/__init__.mojo \
    tests/unit/test_m6_quant_format.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi

# Python ruff check
uvx ruff@0.8.4 check tools/quant/quant_format.py
uvx ruff@0.8.4 format --check tools/quant/quant_format.py

echo "   PASS: Formatting Mojo dan Python bersih 100%."

# ----------------------------------------------------------------------
# 2. Mojo Unit Test Suite
# ----------------------------------------------------------------------
echo ">> [2/5] Menjalankan Mojo TestSuite untuk format, packing, dan skala..."

pixi run mojo run -I src tests/unit/test_m6_quant_format.mojo
echo "   PASS: Seluruh unit test Mojo lulus tanpa error."

# ----------------------------------------------------------------------
# 3. Verifikasi Perhitungan Tangan [2048, 2048] -> 2.162.688 B
# ----------------------------------------------------------------------
echo ">> [3/5] Verifikasi eksak hitungan tangan tensor [2048, 2048]..."

uv run --python .venv python -c "
from tools.quant.quant_format import calculate_tensor_quant_size

shape = [2048, 2048]
num_groups, scales_bytes, weights_bytes, total_payload = calculate_tensor_quant_size(shape, 128)

assert num_groups == 32768, f'num_groups {num_groups} != 32768'
assert scales_bytes == 65536, f'scales_bytes {scales_bytes} != 65536 (64 KiB)'
assert weights_bytes == 2097152, f'weights_bytes {weights_bytes} != 2097152 (2 MiB)'
assert total_payload == 2162688, f'total_payload {total_payload} != 2162688'

bf16_bytes = 2048 * 2048 * 2
assert bf16_bytes == 8388608, f'bf16_bytes {bf16_bytes} != 8388608 (8 MiB)'

ratio = bf16_bytes / total_payload
assert 3.87 < ratio < 3.89, f'Ratio {ratio:.4f} deviates from expected ~3.88x'

print(f'   PASS: [2048, 2048] -> Scales: {scales_bytes} B, Weights: {weights_bytes} B, Total: {total_payload} B, Rasio: {ratio:.4f}x (~3.88x)')
"

# ----------------------------------------------------------------------
# 4. Pembuatan & Validasi Berkas Biner Kuantisasi (Cross-Language)
# ----------------------------------------------------------------------
echo ">> [4/5] Uji coba pembentukan berkas biner dan validasi roundtrip biner..."

QUANT_BIN="$TEST_DIR/sample_quant_model.bin"

uv run --python .venv python -c "
from tools.quant.quant_format import (
    make_quant_header,
    make_tensor_record,
    parse_quant_header,
    read_tensor_record,
    QUANT_HEADER_SIZE,
)

# Buat tensor 1: [2048, 2048]
shape1 = [2048, 2048]
num_groups1 = (2048 * 2048) // 128
scales1 = [0.125] * num_groups1
weights1 = [1, -2, 3, -4, 5, -6, 7, 0] * ((2048 * 2048) // 8)
rec1 = make_tensor_record('model.layers.0.self_attn.q_proj.weight', shape1, scales1, weights1)

# Buat tensor 2: [128, 256]
shape2 = [128, 256]
num_groups2 = (128 * 256) // 128
scales2 = [0.25] * num_groups2
weights2 = [-7, 6, -5, 4, -3, 2, -1, 0] * ((128 * 256) // 8)
rec2 = make_tensor_record('model.layers.0.self_attn.k_proj.weight', shape2, scales2, weights2)

total_file_bytes = QUANT_HEADER_SIZE + len(rec1) + len(rec2)
hdr = make_quant_header('qwen1.5-moe-a2.7b-chat', 2, total_file_bytes)

with open('$QUANT_BIN', 'wb') as f:
    f.write(hdr)
    f.write(rec1)
    f.write(rec2)

# Baca kembali dan validasi
with open('$QUANT_BIN', 'rb') as f:
    raw = f.read()

assert len(raw) == total_file_bytes, f'Ukuran berkas {len(raw)} != {total_file_bytes}'
hdr_data = parse_quant_header(raw[:QUANT_HEADER_SIZE])
assert hdr_data['num_tensors'] == 2
assert hdr_data['total_bytes'] == total_file_bytes

meta1, s1, w1, off1 = read_tensor_record(raw, QUANT_HEADER_SIZE)
assert meta1['name'] == 'model.layers.0.self_attn.q_proj.weight'
assert meta1['shape'] == [2048, 2048]
assert len(s1) == num_groups1
assert len(w1) == 2048 * 2048
assert w1[:8] == [1, -2, 3, -4, 5, -6, 7, 0]

meta2, s2, w2, off2 = read_tensor_record(raw, off1)
assert meta2['name'] == 'model.layers.0.self_attn.k_proj.weight'
assert meta2['shape'] == [128, 256]
assert len(s2) == num_groups2
assert len(w2) == 128 * 256
assert w2[:8] == [-7, 6, -5, 4, -3, 2, -1, 0]
assert off2 == total_file_bytes

print(f'   PASS: Berkas quant model biner ({total_file_bytes} B) valid dan terbaca sempurna.')
"

# ----------------------------------------------------------------------
# 5. Negative Path Validation (Fault Injection)
# ----------------------------------------------------------------------
echo ">> [5/5] Uji ketahanan validasi terhadap header corrupt dan nilai di luar bound..."

uv run --python .venv python -c "
from tools.quant.quant_format import parse_quant_header, pack_4bit_pair, read_tensor_record

# Uji header tidak 256 byte
try:
    parse_quant_header(b'short header')
    assert False, 'Harus gagal jika header != 256 byte'
except ValueError:
    pass

# Uji nilai di luar range [-8, 7]
try:
    pack_4bit_pair(8, 0)
    assert False, 'Harus gagal jika w0 > 7'
except ValueError:
    pass

try:
    pack_4bit_pair(0, -9)
    assert False, 'Harus gagal jika w1 < -8'
except ValueError:
    pass

print('   PASS: Seluruh skenario negative/fault injection tertolak secara aman.')
"

echo "======================================================================"
echo "M6-W1 VERIFIKASI LENGKAP: FORMAT BEKU, PACKING & VALIDASI SUKSES (GREEN)"
echo "======================================================================"
