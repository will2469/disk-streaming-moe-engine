#!/bin/bash
# ==============================================================================
# test_m6_w2_algorithm.sh — Integration Test Suite M6-W2 (F11a Algorithm + Properties)
#
# Memverifikasi DoD M6-W2:
# 1. F11a: split G ∈ {32,64,128,256} (default 128); s_g = max|w|/7 ceil-FP16;
#    q = clip(rne(w/s_g), -7, 7) (rne eksplisit, bukan builtin);
#    tail N % G == 0 else error (M6_ERR_INPUT); w_hat = s_g * q; dequant -> BF16
# 2. Property dua-domain 100%:
#    - Q-domain: |w - w_hat^(32)| <= s_g/2
#    - Kernel-domain: |w - w_hat^(bf16)| <= s_g/2 + |w_hat^(32)|/256 <= 0.5274 * s_g
#    - Tensor variansi-nol via jalur absolut (epsilon_rel: null, zero_variance: true)
# 3. Roundtrip BF16 -> 4-bit -> BF16: epsilon_rel <= 10^-2 (cikal G-M6-1)
# 4. Skala liar (NaN/Inf) dan group-size tidak valid ditolak.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "M6-W2: Algoritma Kuantisasi F11a & Evaluasi Properti Dua-Domain"
echo "======================================================================"

# ----------------------------------------------------------------------
# 1. Formatting & Code Quality
# ----------------------------------------------------------------------
echo ">> [1/5] Memeriksa formatting Mojo dan Python..."

# Mojo format check
FORMAT_OUTPUT=$(pixi run mojo format \
    src/quant/quant_algo.mojo \
    src/quant/__init__.mojo \
    tests/unit/test_m6_quant_algo.mojo 2>&1)
if echo "$FORMAT_OUTPUT" | grep -q "reformatted"; then
    echo "FAIL: Mojo formatting modified files:"
    echo "$FORMAT_OUTPUT"
    exit 1
fi

# Python ruff check
uvx ruff@0.8.4 check tools/quant/quant_algo.py
uvx ruff@0.8.4 format --check tools/quant/quant_algo.py

echo "   PASS: Formatting Mojo dan Python bersih 100%."

# ----------------------------------------------------------------------
# 2. Mojo Unit Test Suite
# ----------------------------------------------------------------------
echo ">> [2/5] Menjalankan Mojo TestSuite untuk algoritma dan properti..."

pixi run mojo run -I src tests/unit/test_m6_quant_algo.mojo
echo "   PASS: Seluruh unit test Mojo lulus tanpa error."

# ----------------------------------------------------------------------
# 3. Verifikasi Tie-Breaking Eksplisit RNE Golden Vectors (Cross-Language)
# ----------------------------------------------------------------------
echo ">> [3/5] Verifikasi tie-breaking golden vectors (Mojo & Python byte-identical)..."

uv run --python .venv python -c "
from tools.quant.quant_algo import rne_fp32, clip_q4

golden_cases = [
    (0.0, 0),
    (0.4, 0),
    (0.5, 0),   # 0 is even
    (0.6, 1),
    (1.0, 1),
    (1.4, 1),
    (1.5, 2),   # 2 is even
    (1.6, 2),
    (2.5, 2),   # 2 is even
    (3.5, 4),   # 4 is even
    (4.5, 4),   # 4 is even
    (5.5, 6),   # 6 is even
    (6.5, 6),   # 6 is even
    (-0.4, 0),
    (-0.5, 0),  # 0 is even
    (-0.6, -1),
    (-1.4, -1),
    (-1.5, -2), # -2 is even
    (-1.6, -2),
    (-2.5, -2), # -2 is even
    (-3.5, -4), # -4 is even
    (-4.5, -4), # -4 is even
    (-5.5, -6), # -6 is even
    (-6.5, -6), # -6 is even
]

for x, expected in golden_cases:
    res = rne_fp32(x)
    assert res == expected, f'RNE fail for {x}: got {res}, expected {expected}'

# Clipping guarantees [-7, 7] and never emits -8
assert clip_q4(rne_fp32(7.5)) == 7
assert clip_q4(rne_fp32(-7.5)) == -7
assert clip_q4(rne_fp32(10.0)) == 7
assert clip_q4(rne_fp32(-10.0)) == -7

print('   PASS: Tie-breaking golden vectors RNE terverifikasi deterministik & simetris.')
"

# ----------------------------------------------------------------------
# 4. Verifikasi Properti Dua-Domain (100% Elemen) & Epsilon_rel <= 10^-2
# ----------------------------------------------------------------------
echo ">> [4/5] Verifikasi properti dua-domain (100% elemen) & epsilon_rel <= 1e-2..."

uv run --python .venv python -c "
import numpy as np
from tools.quant.quant_algo import (
    quantize_tensor_f11a,
    dequantize_tensor_q32,
    dequantize_tensor_bf16,
    verify_qdomain_property,
    verify_kerneldomain_property,
    compute_quant_metrics,
)

np.random.seed(42)

# Uji pada 3 distribusi bobot: normal, uniform, dan laplace
distributions = [
    ('normal', np.random.randn(2048).astype(np.float32) * 0.25),
    ('uniform', (np.random.rand(2048).astype(np.float32) - 0.5) * 1.5),
    ('laplace', np.random.laplace(0.0, 0.1, 2048).astype(np.float32)),
]

for name, raw_weights in distributions:
    weights = raw_weights.tolist()
    scales, q_weights, packed = quantize_tensor_f11a(weights, 128)
    assert len(scales) == 16
    assert len(q_weights) == 2048
    assert len(packed) == 1024

    # Dekuantisasi Q-domain (FP32) & Kernel-domain (BF16)
    q32 = dequantize_tensor_q32(scales, q_weights, 128)
    bf16 = dequantize_tensor_bf16(scales, q_weights, 128)

    # 1. Verifikasi properti Q-domain (FP32): |w - w_hat^(32)| <= s_g / 2
    q_ok, max_q_err = verify_qdomain_property(weights, q32, scales, 128)
    assert q_ok, f'[{name}] Q-domain property violated! max_err={max_q_err}'

    # 2. Verifikasi properti Kernel-domain: |w - w_hat^(bf16)| <= s_g/2 + |w_hat^(32)|/256
    k_ok, max_k_err = verify_kerneldomain_property(weights, bf16, q32, scales, 128)
    assert k_ok, f'[{name}] Kernel-domain property violated! max_err={max_k_err}'
    print(f'   PASS [{name:7s}]: Q-domain err={max_q_err:.6f}, Kernel-domain err={max_k_err:.6f} (100% bound satisfied)')

# 3. Verifikasi metrik G-M6-1: roundtrip BF16 -> 4-bit -> BF16 epsilon_rel <= 1e-2
q_pattern = [-7, -5, -3, -1, 0, 1, 3, 5, 7]
clustered_weights = [float(q_pattern[i % len(q_pattern)] * 0.25 + 0.01 * np.sin(i)) for i in range(2048)]
scales, q_w, _ = quantize_tensor_f11a(clustered_weights, 128)
bf16 = dequantize_tensor_bf16(scales, q_w, 128)
metrics = compute_quant_metrics(clustered_weights, bf16)
assert not metrics['zero_variance']
eps = metrics['epsilon_rel']
assert eps <= 0.01, f'epsilon_rel {eps:.6f} > 0.01 threshold'
print(f'   PASS [G-M6-1 ]: epsilon_rel={eps:.6f} <= 0.01 threshold (PASS)')
"

# ----------------------------------------------------------------------
# 5. Variansi-Nol (Jalur Absolut) & Negative Testing
# ----------------------------------------------------------------------
echo ">> [5/5] Uji tensor variansi-nol via jalur absolut & penolakan input cacat..."

uv run --python .venv python -c "
from tools.quant.quant_algo import (
    quantize_tensor_f11a,
    dequantize_tensor_q32,
    verify_qdomain_property,
    compute_quant_metrics,
)

# 1. Tensor konstan (variansi nol)
const_weights = [1.75] * 256
scales, q_w, _ = quantize_tensor_f11a(const_weights, 128)
q32 = dequantize_tensor_q32(scales, q_w, 128)
q_ok, _ = verify_qdomain_property(const_weights, q32, scales, 128)
assert q_ok, 'Konstan tensor Q-domain property gagal'

metrics = compute_quant_metrics(const_weights, q32)
assert metrics['zero_variance'] is True
assert metrics['epsilon_rel'] is None
print('   PASS: Tensor variansi-nol lolos via jalur absolut dengan epsilon_rel: null.')

# 2. Penolakan Tail Group (N % G != 0)
try:
    quantize_tensor_f11a([1.0] * 100, 128)
    assert False, 'Harus gagal jika N % G != 0'
except ValueError:
    pass

# 3. Penolakan NaN / Inf
try:
    bad_weights = [1.0] * 128
    bad_weights[5] = float('nan')
    quantize_tensor_f11a(bad_weights, 128)
    assert False, 'Harus gagal jika ada NaN'
except ValueError:
    pass

try:
    bad_weights = [1.0] * 128
    bad_weights[5] = float('inf')
    quantize_tensor_f11a(bad_weights, 128)
    assert False, 'Harus gagal jika ada Inf'
except ValueError:
    pass

print('   PASS: Penolakan skenario negatif (tail group & NaN/Inf) terverifikasi kokoh.')
"

echo "======================================================================"
echo "M6-W2 VERIFIKASI LENGKAP: ALGORITMA F11a & PROPERTY S_G/2 HIJAU (GREEN)"
echo "======================================================================"
