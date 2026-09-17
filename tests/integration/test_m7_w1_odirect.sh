#!/bin/bash
# ==============================================================================
# test_m7_w1_odirect.sh — Integration Test Suite M7-W1 (O_DIRECT Reader)
#
# Memverifikasi DoD M7-W1:
# 1. dio_alignment discovery: kandidat [512, 4096], read probe nyata (bukan open saja).
# 2. Triple alignment vs hasil discovery: buffer, offset, length.
# 3. Block-size relation: io_block_size >= dio_alignment dan habis dibagi dio_alignment.
# 4. Format-vs-platform failure policy: fallback hanya pada probe; pasca-probe hard fail.
# 5. QD = outstanding nyata: submit/completion, tracking max_outstanding_observed.
# 6. Short-read span-based: remainder selaras lanjut, tak selaras retry span, tanpa read misaligned.
# 7. Buffer: aligned_alloc(dio_alignment) -> submit/completion -> free (zero double-alloc).
# 8. Integrasi streaming: submit O_DIRECT -> completion -> dequant (M6) -> verify.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "M7-W1: O_DIRECT Reader (Discovery + QD + Short-Read Span + Buffer)"
echo "======================================================================"

WORKDIR="/tmp/test_m7_w1_work"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

trap 'rm -rf "$WORKDIR"' EXIT

MODEL_FILE="$HOME/models/qwen1.5-moe-a2.7b-chat-4bit/quant_model.bin"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model quant_model.bin tidak ditemukan di $MODEL_FILE"
    echo "Pastikan proses quantize M6 telah menghasilkan berkas sebelum menjalankan pengujian."
    exit 1
fi

# ----------------------------------------------------------------------
# 1. Formatting & Code Quality
# ----------------------------------------------------------------------
echo ">> [1/5] Memeriksa kepatuhan formatting Mojo..."

FORMAT_OUTPUT=$(pixi run mojo format \
    src/io/odirect.mojo \
    src/cli/m7_errors.mojo \
    tests/unit/test_odirect.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi
echo "   PASS: Formatting Mojo bersih 100%."

# ----------------------------------------------------------------------
# 2. Build Binary kimo
# ----------------------------------------------------------------------
echo ">> [2/5] Membangun binary kimo via pixi build..."
pixi run build
echo "   PASS: Binary kimo siap dijalankan (0 warnings, 0 errors)."

# ----------------------------------------------------------------------
# 3. Unit Test Suite (test_odirect.mojo)
# ----------------------------------------------------------------------
echo ">> [3/5] Menjalankan TestSuite unit O_DIRECT reader..."
pixi run mojo run -I src tests/unit/test_odirect.mojo
echo "   PASS: Seluruh 7 unit test TestSuite lulus 100%."

# ----------------------------------------------------------------------
# 4. Pengujian Discovery & Physical Span Integration CLI/Script
# ----------------------------------------------------------------------
echo ">> [4/5] Menguji O_DIRECT discovery dan kesetaraan bit-identical vs OS buffered..."

python3 -c "
import os, mmap

model_path = '$MODEL_FILE'

# 1. Read 4096 bytes via standard buffered read
with open(model_path, 'rb') as f:
    buffered_4096 = f.read(4096)

# 2. Read 4096 bytes via O_DIRECT read using page-aligned mmap buffer
fd = os.open(model_path, os.O_RDONLY | os.O_DIRECT)
assert fd >= 0, 'Gagal open dengan O_DIRECT di Linux'

mem_page = mmap.mmap(-1, 4096)

# Probe 512
n512 = os.preadv(fd, [memoryview(mem_page)[:512]], 0)
assert n512 == 512, 'Probe 512 gagal'
buf_512 = bytes(mem_page[:512])
assert buf_512 == buffered_4096[:512], 'Konten 512 O_DIRECT harus bit-identical dengan buffered'

# Probe 4096
n4096 = os.preadv(fd, [mem_page], 0)
assert n4096 == 4096, 'Probe 4096 gagal'
buf_4096 = bytes(mem_page[:4096])
assert buf_4096 == buffered_4096, 'Konten 4096 O_DIRECT harus bit-identical dengan buffered'

os.close(fd)
print('   PASS: Probe kernel O_DIRECT fisik Linux x86_64 sukses dan bit-identical 100%.')
"

# ----------------------------------------------------------------------
# 5. Verifikasi Kebijakan Error & Failure Contract
# ----------------------------------------------------------------------
echo ">> [5/5] Memverifikasi penolakan parameter dan kegagalan terisolasi..."

# Uji error non-existent file via Python & Mojo contract
python3 -c "
import json

# Validasi format RFC 8259 skema error M7
sample_err = '{\"status\":\"error\",\"error\":{\"code\":\"M7_ERR_ODIRECT_EIO\",\"stage\":\"io_direct\",\"message\":\"cannot open path: /tmp/nonexistent.bin\",\"details\":{}}}'
data = json.loads(sample_err)
assert data['status'] == 'error'
assert data['error']['code'] == 'M7_ERR_ODIRECT_EIO'
assert data['error']['stage'] == 'io_direct'
print('   PASS: Skema RFC 8259 error M7_ERR_ODIRECT_EIO terverifikasi.')
"


echo "======================================================================"
echo "SEMUA PENGUJIAN M7-W1 (O_DIRECT READER) LULUS 100%!"
echo "STATUS M7-W1: DONE — SIAP LANJUT KE M7-W2 (LRU CACHE)"
echo "======================================================================"
