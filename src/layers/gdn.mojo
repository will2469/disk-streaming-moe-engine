# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Gated DeltaNet (GDN) Recurrence Kernel Baseline — M8-W2.

Mengimplementasikan kernel recurrence GDN chunked scan representasi Woodbury (WY)
untuk F14:
    S_t = gamma_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
Layout kanonis: S in R^{dv x dk} (row = dv, col = dk).
Dalam satu chunk berukuran m in [1, C]:
    W_i = beta_i (K_i - sum_{j > i} (K_i . K_j) W_j)
    S_{next} = S - (S K^T) W + V^T W
Mendukung sembarang panjang chunk m secara native tanpa naive fallback.
"""

from core.config import LoadMemoryTelemetry, ModelConfig
from core.tensor_loader import ShardHeaderCache, _load_tensor_by_numel
from layers.residual import add_residual
from layers.rmsnorm import rmsnorm
from layers.swiglu import sigmoid_f32, silu_f32
from std.builtin.dtype import DType
from std.collections import Dict, List
from std.math import exp, isinf, isnan, log, sqrt


struct GDNConfig(Copyable, Movable):
    """Konfigurasi layer dan dimensi GDN."""

    var layers: Int
    var dk: Int
    var dv: Int
    var chunk_size: Int

    def __init__(
        out self, layers: Int, dk: Int, dv: Int, chunk_size: Int = 512
    ):
        self.layers = layers
        self.dk = dk
        self.dv = dv
        self.chunk_size = chunk_size

    def validate(self) raises:
        """Memvalidasi parameter konfigurasi sesuai kontrak M8."""
        if self.layers <= 0:
            raise Error(
                '{"error_type":"CONFIG_INVALID","detail":"layers must be'
                ' positive"}'
            )
        if self.dk <= 0:
            raise Error(
                '{"error_type":"CONFIG_INVALID","detail":"dk must be positive"}'
            )
        if self.dv <= 0:
            raise Error(
                '{"error_type":"CONFIG_INVALID","detail":"dv must be positive"}'
            )
        if self.chunk_size < 8 or self.chunk_size > 4096:
            raise Error(
                '{"error_type":"CONFIG_INVALID","detail":"chunk_size must be in'
                ' range [8, 4096]"}'
            )


struct GDNState(Copyable, Movable):
    """Pengelola tensor state kanonis GDN: [layers, dv, dk] row-major FP32."""

    var layers: Int
    var dv: Int
    var dk: Int
    var data: List[Float32]

    def __init__(out self, layers: Int, dv: Int, dk: Int):
        self.layers = layers
        self.dv = dv
        self.dk = dk
        var total_elements = layers * dv * dk
        self.data = List[Float32]()
        self.data.resize(total_elements, Float32(0.0))

    def zero(mut self):
        """Mereset seluruh state menjadi nol (S_0 = 0)."""
        var total = len(self.data)
        var p = self.data.unsafe_ptr()
        for i in range(total):
            p[unsafe_offset=i] = Float32(0.0)

    def get(self, layer: Int, row: Int, col: Int) -> Float32:
        """Mengambil elemen S[layer][row][col]."""
        var offset = layer * (self.dv * self.dk) + row * self.dk + col
        return self.data[offset]

    def set(mut self, layer: Int, row: Int, col: Int, val: Float32):
        """Menetapkan elemen S[layer][row][col]."""
        var offset = layer * (self.dv * self.dk) + row * self.dk + col
        self.data[offset] = val

    def check_finite(self, layer: Int = -1) raises:
        """Memeriksa apakah terdapat nilai NaN atau INF dalam state."""
        var start = 0
        var end = len(self.data)
        if layer >= 0 and layer < self.layers:
            start = layer * (self.dv * self.dk)
            end = start + (self.dv * self.dk)

        var p = self.data.unsafe_ptr()
        for i in range(start, end):
            var val = p[unsafe_offset=i]
            if isnan(val) or isinf(val):
                raise Error(
                    '{"error_type":"GDN_FORWARD_ERROR","detail":"non-finite'
                    ' value in GDN state","layer":'
                    + String(layer)
                    + "}"
                )

    def clone(self) -> GDNState:
        """Membuat salinan dalam (deep copy) dari GDNState."""
        var out = GDNState(self.layers, self.dv, self.dk)
        var total = len(self.data)
        var p_src = self.data.unsafe_ptr()
        var p_dst = out.data.unsafe_ptr()
        for i in range(total):
            p_dst[unsafe_offset=i] = p_src[unsafe_offset=i]
        return out^


def compute_wy_coefficients(
    k_mat: List[Float32],
    beta: List[Float32],
    m: Int,
    dk: Int,
) raises -> List[Float32]:
    """Menghitung koefisien representasi WY (W in R^{m x dk}) via backward substitution.

    Rumus analitis:
        W_i = beta_i (K_i - sum_{j > i} (K_i . K_j) W_j)
    di mana K_i in R^{dk} adalah baris key ke-i.
    """
    if m <= 0 or dk <= 0:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"m and dk must be positive"}'
        )
    if len(k_mat) != m * dk:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"k_mat size mismatch"}'
        )
    if len(beta) != m:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"beta size mismatch"}'
        )

    var w_mat = List[Float32]()
    w_mat.resize(m * dk, Float32(0.0))

    var p_k = k_mat.unsafe_ptr()
    var p_w = w_mat.unsafe_ptr()
    var p_beta = beta.unsafe_ptr()

    # Precompute upper triangular Gram matrix: G[i, j] = K_i . K_j untuk j > i
    var gram = List[Float32]()
    gram.resize(m * m, Float32(0.0))
    var p_gram = gram.unsafe_ptr()

    for i in range(m):
        var row_i_off = i * dk
        for j in range(i + 1, m):
            var row_j_off = j * dk
            var acc_simd = SIMD[DType.float32, 8](0.0)
            var c = 0
            while c + 8 <= dk:
                acc_simd += p_k.unsafe_load[width=8](
                    row_i_off + c
                ) * p_k.unsafe_load[width=8](row_j_off + c)
                c += 8
            var dot = acc_simd.reduce_add()
            while c < dk:
                dot += (
                    p_k[unsafe_offset=row_i_off + c]
                    * p_k[unsafe_offset=row_j_off + c]
                )
                c += 1
            p_gram[unsafe_offset=i * m + j] = dot

    # Backward substitution: dari i = m - 1 turun ke 0
    var i = m - 1
    while i >= 0:
        var b_i = p_beta[unsafe_offset=i]
        var row_i_k = i * dk
        var row_i_w = i * dk

        # Inisialisasi rhs dengan K_i
        var c = 0
        while c + 8 <= dk:
            var k_vec = p_k.unsafe_load[width=8](row_i_k + c)
            p_w.unsafe_store[width=8](row_i_w + c, k_vec)
            c += 8
        while c < dk:
            p_w[unsafe_offset=row_i_w + c] = p_k[unsafe_offset=row_i_k + c]
            c += 1

        # Kurangkan sum_{j > i} G[i, j] * W_j
        for j in range(i + 1, m):
            var g_ij = p_gram[unsafe_offset=i * m + j]
            var g_vec = SIMD[DType.float32, 8](g_ij)
            var row_j_w = j * dk

            c = 0
            while c + 8 <= dk:
                var curr_rhs = p_w.unsafe_load[width=8](row_i_w + c)
                var w_j_vec = p_w.unsafe_load[width=8](row_j_w + c)
                curr_rhs -= g_vec * w_j_vec
                p_w.unsafe_store[width=8](row_i_w + c, curr_rhs)
                c += 8
            while c < dk:
                p_w[unsafe_offset=row_i_w + c] -= (
                    g_ij * p_w[unsafe_offset=row_j_w + c]
                )
                c += 1

        # Kalikan dengan beta_i
        var b_vec = SIMD[DType.float32, 8](b_i)
        c = 0
        while c + 8 <= dk:
            var final_rhs = p_w.unsafe_load[width=8](row_i_w + c)
            p_w.unsafe_store[width=8](row_i_w + c, final_rhs * b_vec)
            c += 8
        while c < dk:
            p_w[unsafe_offset=row_i_w + c] *= b_i
            c += 1

        i -= 1

    return w_mat^


def apply_wy_chunk_update(
    mut s_layer: List[Float32],
    k_mat: List[Float32],
    v_mat: List[Float32],
    w_mat: List[Float32],
    m: Int,
    dk: Int,
    dv: Int,
) raises:
    """Menerapkan pembaruan state WY untuk satu chunk.

        S_{next} = S - (S K^T) W + V^T W
    di mana:
        S in R^{dv x dk}
        K in R^{m x dk}
        V in R^{m x dv}
        W in R^{m x dk}
    Operasi ini tidak pernah mematerialisasi matriks [dk x dk].
    """
    if len(s_layer) != dv * dk:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"s_layer size mismatch"}'
        )
    if len(k_mat) != m * dk:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"k_mat size mismatch"}'
        )
    if len(v_mat) != m * dv:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"v_mat size mismatch"}'
        )
    if len(w_mat) != m * dk:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"w_mat size mismatch"}'
        )

    var p_s = s_layer.unsafe_ptr()
    var p_k = k_mat.unsafe_ptr()
    var p_v = v_mat.unsafe_ptr()
    var p_w = w_mat.unsafe_ptr()

    # 1. A = S @ K^T in R^{dv x m}
    # A[r, i] = sum_{c=0}^{dk-1} S[r, c] * K[i, c]
    var a_mat = List[Float32]()
    a_mat.resize(dv * m, Float32(0.0))
    var p_a = a_mat.unsafe_ptr()

    for r in range(dv):
        var s_row_off = r * dk
        var a_row_off = r * m
        for i in range(m):
            var k_row_off = i * dk
            var acc_simd = SIMD[DType.float32, 8](0.0)
            var c = 0
            while c + 8 <= dk:
                acc_simd += p_s.unsafe_load[width=8](
                    s_row_off + c
                ) * p_k.unsafe_load[width=8](k_row_off + c)
                c += 8
            var dot = acc_simd.reduce_add()
            while c < dk:
                dot += (
                    p_s[unsafe_offset=s_row_off + c]
                    * p_k[unsafe_offset=k_row_off + c]
                )
                c += 1
            p_a[unsafe_offset=a_row_off + i] = dot

    # 2. B = A @ W in R^{dv x dk}
    # B[r, c] = sum_{i=0}^{m-1} A[r, i] * W[i, c]
    var b_mat = List[Float32]()
    b_mat.resize(dv * dk, Float32(0.0))
    var p_b = b_mat.unsafe_ptr()

    for r in range(dv):
        var a_row_off = r * m
        var b_row_off = r * dk
        for i in range(m):
            var a_val = p_a[unsafe_offset=a_row_off + i]
            var a_vec = SIMD[DType.float32, 8](a_val)
            var w_row_off = i * dk

            var c = 0
            while c + 8 <= dk:
                var curr_b = p_b.unsafe_load[width=8](b_row_off + c)
                var w_vec = p_w.unsafe_load[width=8](w_row_off + c)
                curr_b += a_vec * w_vec
                p_b.unsafe_store[width=8](b_row_off + c, curr_b)
                c += 8
            while c < dk:
                p_b[unsafe_offset=b_row_off + c] += (
                    a_val * p_w[unsafe_offset=w_row_off + c]
                )
                c += 1

    # 3. C = V^T @ W in R^{dv x dk}
    # C[r, c] = sum_{i=0}^{m-1} V[i, r] * W[i, c]
    var c_mat = List[Float32]()
    c_mat.resize(dv * dk, Float32(0.0))
    var p_c = c_mat.unsafe_ptr()

    for i in range(m):
        var v_row_off = i * dv
        var w_row_off = i * dk
        for r in range(dv):
            var v_val = p_v[unsafe_offset=v_row_off + r]
            var v_vec = SIMD[DType.float32, 8](v_val)
            var c_row_off = r * dk

            var c = 0
            while c + 8 <= dk:
                var curr_c = p_c.unsafe_load[width=8](c_row_off + c)
                var w_vec = p_w.unsafe_load[width=8](w_row_off + c)
                curr_c += v_vec * w_vec
                p_c.unsafe_store[width=8](c_row_off + c, curr_c)
                c += 8
            while c < dk:
                p_c[unsafe_offset=c_row_off + c] += (
                    v_val * p_w[unsafe_offset=w_row_off + c]
                )
                c += 1

    # 4. Update state: S_{next} = S - B + C
    for idx in range(dv * dk):
        var val = (
            p_s[unsafe_offset=idx]
            - p_b[unsafe_offset=idx]
            + p_c[unsafe_offset=idx]
        )
        if isnan(val) or isinf(val):
            raise Error(
                '{"error_type":"GDN_FORWARD_ERROR","detail":"non-finite'
                ' value in chunk update"}'
            )
        p_s[unsafe_offset=idx] = val


@fieldwise_init
struct ChunkOperator(Copyable, Movable):
    """Pasangan operator affine blok (M_A, B_A) untuk representasi WY."""

    var m_mat: List[Float32]
    var b_mat: List[Float32]


@fieldwise_init
struct ProjectedKVBeta(Copyable, Movable):
    """Hasil proyeksi input x ke K, V, dan Beta."""

    var k_mat: List[Float32]
    var v_mat: List[Float32]
    var beta: List[Float32]


def extract_chunk_operator(
    k_mat: List[Float32],
    v_mat: List[Float32],
    w_mat: List[Float32],
    m: Int,
    dk: Int,
    dv: Int,
) raises -> ChunkOperator:
    """Mengekstrak operator affine blok eksplisit (M_A, B_A) untuk verifikasi hukum komposisi.

    M_A = I - K^T W in R^{dk x dk}
    B_A = V^T W in R^{dv x dk}
    """
    var p_k = k_mat.unsafe_ptr()
    var p_v = v_mat.unsafe_ptr()
    var p_w = w_mat.unsafe_ptr()

    # M_A = I - K^T @ W
    var m_op = List[Float32]()
    m_op.resize(dk * dk, Float32(0.0))
    var p_m = m_op.unsafe_ptr()

    # Inisialisasi identitas I
    for i in range(dk):
        p_m[unsafe_offset=i * dk + i] = Float32(1.0)

    # Kurangkan K^T @ W: (K^T @ W)[i, j] = sum_{t=0}^{m-1} K[t, i] * W[t, j]
    for t in range(m):
        var k_row_off = t * dk
        var w_row_off = t * dk
        for i in range(dk):
            var k_ti = p_k[unsafe_offset=k_row_off + i]
            var k_vec = SIMD[DType.float32, 8](k_ti)
            var m_row_off = i * dk

            var j = 0
            while j + 8 <= dk:
                var curr_m = p_m.unsafe_load[width=8](m_row_off + j)
                var w_vec = p_w.unsafe_load[width=8](w_row_off + j)
                curr_m -= k_vec * w_vec
                p_m.unsafe_store[width=8](m_row_off + j, curr_m)
                j += 8
            while j < dk:
                p_m[unsafe_offset=m_row_off + j] -= (
                    k_ti * p_w[unsafe_offset=w_row_off + j]
                )
                j += 1

    # B_A = V^T @ W in R^{dv x dk}
    var b_op = List[Float32]()
    b_op.resize(dv * dk, Float32(0.0))
    var p_b = b_op.unsafe_ptr()

    for t in range(m):
        var v_row_off = t * dv
        var w_row_off = t * dk
        for r in range(dv):
            var v_tr = p_v[unsafe_offset=v_row_off + r]
            var v_vec = SIMD[DType.float32, 8](v_tr)
            var b_row_off = r * dk

            var c = 0
            while c + 8 <= dk:
                var curr_b = p_b.unsafe_load[width=8](b_row_off + c)
                var w_vec = p_w.unsafe_load[width=8](w_row_off + c)
                curr_b += v_vec * w_vec
                p_b.unsafe_store[width=8](b_row_off + c, curr_b)
                c += 8
            while c < dk:
                p_b[unsafe_offset=b_row_off + c] += (
                    v_tr * p_w[unsafe_offset=w_row_off + c]
                )
                c += 1

    return ChunkOperator(m_op^, b_op^)


def compose_chunk_operators(
    m_a: List[Float32],
    b_a: List[Float32],
    m_b: List[Float32],
    b_b: List[Float32],
    dk: Int,
    dv: Int,
) raises -> ChunkOperator:
    """Mengomposisikan dua operator affine blok berdampingan.

    (M_{A union B}, B_{A union B}) = (M_A M_B, B_A M_B + B_B).
    """
    var p_ma = m_a.unsafe_ptr()
    var p_ba = b_a.unsafe_ptr()
    var p_mb = m_b.unsafe_ptr()
    var p_bb = b_b.unsafe_ptr()

    # 1. M_{AB} = M_A @ M_B in R^{dk x dk}
    var m_res = List[Float32]()
    m_res.resize(dk * dk, Float32(0.0))
    var p_mres = m_res.unsafe_ptr()

    for i in range(dk):
        var ma_row_off = i * dk
        var mres_row_off = i * dk
        for k in range(dk):
            var ma_val = p_ma[unsafe_offset=ma_row_off + k]
            var ma_vec = SIMD[DType.float32, 8](ma_val)
            var mb_row_off = k * dk

            var j = 0
            while j + 8 <= dk:
                var curr = p_mres.unsafe_load[width=8](mres_row_off + j)
                var mb_vec = p_mb.unsafe_load[width=8](mb_row_off + j)
                curr += ma_vec * mb_vec
                p_mres.unsafe_store[width=8](mres_row_off + j, curr)
                j += 8
            while j < dk:
                p_mres[unsafe_offset=mres_row_off + j] += (
                    ma_val * p_mb[unsafe_offset=mb_row_off + j]
                )
                j += 1

    # 2. B_{AB} = B_A @ M_B + B_B in R^{dv x dk}
    var b_res = List[Float32]()
    b_res.resize(dv * dk, Float32(0.0))
    var p_bres = b_res.unsafe_ptr()

    for r in range(dv):
        var ba_row_off = r * dk
        var bres_row_off = r * dk
        for k in range(dk):
            var ba_val = p_ba[unsafe_offset=ba_row_off + k]
            var ba_vec = SIMD[DType.float32, 8](ba_val)
            var mb_row_off = k * dk

            var c = 0
            while c + 8 <= dk:
                var curr = p_bres.unsafe_load[width=8](bres_row_off + c)
                var mb_vec = p_mb.unsafe_load[width=8](mb_row_off + c)
                curr += ba_vec * mb_vec
                p_bres.unsafe_store[width=8](bres_row_off + c, curr)
                c += 8
            while c < dk:
                p_bres[unsafe_offset=bres_row_off + c] += (
                    ba_val * p_mb[unsafe_offset=mb_row_off + c]
                )
                c += 1

    # Tambahkan B_B
    for idx in range(dv * dk):
        p_bres[unsafe_offset=idx] += p_bb[unsafe_offset=idx]

    return ChunkOperator(m_res^, b_res^)


def chunked_gdn_scan(
    mut state: GDNState,
    layer: Int,
    k_all: List[Float32],
    v_all: List[Float32],
    beta_all: List[Float32],
    seq_len: Int,
    dk: Int,
    dv: Int,
    chunk_size: Int = 512,
) raises:
    """Menjalankan chunked scan GDN untuk satu layer melintasi seluruh sekuens.

    Mendukung sembarang seq_len; remainder m_rem < chunk_size diproses native
    menggunakan fungsi kernel WY yang sama tanpa naive fallback.
    """
    if layer < 0 or layer >= state.layers:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"layer index out of range"}'
        )
    if seq_len <= 0 or dk <= 0 or dv <= 0:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"seq_len, dk, dv must be'
            ' positive"}'
        )
    if chunk_size < 8 or chunk_size > 4096:
        raise Error(
            '{"error_type":"KERNEL_ERROR","detail":"chunk_size out of valid'
            ' range [8, 4096]"}'
        )

    var p_k_all = k_all.unsafe_ptr()
    var p_v_all = v_all.unsafe_ptr()
    var p_beta_all = beta_all.unsafe_ptr()

    # Ekstrak buffer layer saat ini
    var layer_offset = layer * (dv * dk)
    var s_layer = List[Float32]()
    s_layer.resize(dv * dk, Float32(0.0))
    var p_s_layer = s_layer.unsafe_ptr()
    var p_state_data = state.data.unsafe_ptr()

    for idx in range(dv * dk):
        p_s_layer[unsafe_offset=idx] = p_state_data[
            unsafe_offset=layer_offset + idx
        ]

    var offset = 0
    while offset < seq_len:
        var m = seq_len - offset
        if m > chunk_size:
            m = chunk_size

        # Potong sub-vektor untuk chunk aktif
        var k_chunk = List[Float32]()
        k_chunk.resize(m * dk, Float32(0.0))
        var p_k_chunk = k_chunk.unsafe_ptr()
        var k_start = offset * dk
        for idx in range(m * dk):
            p_k_chunk[unsafe_offset=idx] = p_k_all[unsafe_offset=k_start + idx]

        var v_chunk = List[Float32]()
        v_chunk.resize(m * dv, Float32(0.0))
        var p_v_chunk = v_chunk.unsafe_ptr()
        var v_start = offset * dv
        for idx in range(m * dv):
            p_v_chunk[unsafe_offset=idx] = p_v_all[unsafe_offset=v_start + idx]

        var beta_chunk = List[Float32]()
        beta_chunk.resize(m, Float32(0.0))
        var p_beta_chunk = beta_chunk.unsafe_ptr()
        for idx in range(m):
            p_beta_chunk[unsafe_offset=idx] = p_beta_all[
                unsafe_offset=offset + idx
            ]

        # 1. Hitung representasi koefisien WY
        var w_chunk = compute_wy_coefficients(k_chunk, beta_chunk, m, dk)

        # 2. Terapkan pembaruan WY ke state layer
        apply_wy_chunk_update(s_layer, k_chunk, v_chunk, w_chunk, m, dk, dv)

        # 3. Barrier antarchunk & validasi finite
        for idx in range(dv * dk):
            var val = p_s_layer[unsafe_offset=idx]
            if isnan(val) or isinf(val):
                raise Error(
                    '{"error_type":"GDN_FORWARD_ERROR","detail":"non-finite'
                    ' value after chunk scan","layer":'
                    + String(layer)
                    + ',"offset":'
                    + String(offset)
                    + "}"
                )

        offset += m

    # Tulis kembali hasil ke state utama
    for idx in range(dv * dk):
        p_state_data[unsafe_offset=layer_offset + idx] = p_s_layer[
            unsafe_offset=idx
        ]


def project_tokens_to_kv_beta(
    x: List[Float32],
    w_k: List[Float32],
    w_v: List[Float32],
    w_beta: List[Float32],
    seq_len: Int,
    embed_dim: Int,
    dk: Int,
    dv: Int,
) raises -> ProjectedKVBeta:
    """Memproyeksikan token embedding X ke K, V, dan Beta sesuai arsitektur GDN.

    K_t = normalize(W_k @ X_t) (L2 norm dengan epsilon 1e-6)
    V_t = W_v @ X_t
    Beta_t = sigmoid(W_beta @ X_t)
    """
    var p_x = x.unsafe_ptr()
    var p_wk = w_k.unsafe_ptr()
    var p_wv = w_v.unsafe_ptr()
    var p_wbeta = w_beta.unsafe_ptr()

    var k_out = List[Float32]()
    k_out.resize(seq_len * dk, Float32(0.0))
    var p_k_out = k_out.unsafe_ptr()

    var v_out = List[Float32]()
    v_out.resize(seq_len * dv, Float32(0.0))
    var p_v_out = v_out.unsafe_ptr()

    var beta_out = List[Float32]()
    beta_out.resize(seq_len, Float32(0.0))
    var p_beta_out = beta_out.unsafe_ptr()

    for t in range(seq_len):
        var x_off = t * embed_dim

        # 1. Proyeksi K: W_k in R^{dk x embed_dim}
        var k_off = t * dk
        var sum_sq_k = Float32(0.0)
        for i in range(dk):
            var wk_off = i * embed_dim
            var acc_simd = SIMD[DType.float32, 8](0.0)
            var c = 0
            while c + 8 <= embed_dim:
                acc_simd += p_wk.unsafe_load[width=8](
                    wk_off + c
                ) * p_x.unsafe_load[width=8](x_off + c)
                c += 8
            var dot = acc_simd.reduce_add()
            while c < embed_dim:
                dot += (
                    p_wk[unsafe_offset=wk_off + c]
                    * p_x[unsafe_offset=x_off + c]
                )
                c += 1
            p_k_out[unsafe_offset=k_off + i] = dot
            sum_sq_k += dot * dot

        # L2 normalize K_t: k_t / (norm + 1e-6)
        var norm_k = sqrt(sum_sq_k) + Float32(1e-6)
        var inv_norm_k = Float32(1.0) / norm_k
        var inv_norm_vec = SIMD[DType.float32, 8](inv_norm_k)
        var c_norm = 0
        while c_norm + 8 <= dk:
            var vec = p_k_out.unsafe_load[width=8](k_off + c_norm)
            p_k_out.unsafe_store[width=8](k_off + c_norm, vec * inv_norm_vec)
            c_norm += 8
        while c_norm < dk:
            p_k_out[unsafe_offset=k_off + c_norm] *= inv_norm_k
            c_norm += 1

        # 2. Proyeksi V: W_v in R^{dv x embed_dim}
        var v_off = t * dv
        for i in range(dv):
            var wv_off = i * embed_dim
            var acc_simd = SIMD[DType.float32, 8](0.0)
            var c = 0
            while c + 8 <= embed_dim:
                acc_simd += p_wv.unsafe_load[width=8](
                    wv_off + c
                ) * p_x.unsafe_load[width=8](x_off + c)
                c += 8
            var dot = acc_simd.reduce_add()
            while c < embed_dim:
                dot += (
                    p_wv[unsafe_offset=wv_off + c]
                    * p_x[unsafe_offset=x_off + c]
                )
                c += 1
            p_v_out[unsafe_offset=v_off + i] = dot

        # 3. Proyeksi Beta: W_beta in R^{1 x embed_dim}
        var acc_simd = SIMD[DType.float32, 8](0.0)
        var c = 0
        while c + 8 <= embed_dim:
            acc_simd += p_wbeta.unsafe_load[width=8](c) * p_x.unsafe_load[
                width=8
            ](x_off + c)
            c += 8
        var raw_beta = acc_simd.reduce_add()
        while c < embed_dim:
            raw_beta += p_wbeta[unsafe_offset=c] * p_x[unsafe_offset=x_off + c]
            c += 1

        # Sigmoid: 1 / (1 + exp(-raw))
        var beta_val = Float32(1.0) / (Float32(1.0) + exp(-raw_beta))
        p_beta_out[unsafe_offset=t] = beta_val

    return ProjectedKVBeta(k_out^, v_out^, beta_out^)


@fieldwise_init
struct GDNWeights(Copyable, Movable):
    """Bobot parameter Gated DeltaNet linear attention satu layer."""

    var norm_gamma: List[Float32]
    var w_qkv: List[Float32]
    var w_conv: List[Float32]
    var w_z: List[Float32]
    var w_a: List[Float32]
    var w_b: List[Float32]
    var dt_bias: List[Float32]
    var a_log: List[Float32]
    var norm_weight: List[Float32]
    var w_out: List[Float32]


def load_layer_gdn_weights(
    layer_idx: Int,
    model_root: String,
    weight_map: Dict[String, String],
    cfg: ModelConfig,
    mut cache: ShardHeaderCache,
    mut telemetry: LoadMemoryTelemetry,
) raises -> GDNWeights:
    """Memuat seluruh bobot blok GDN satu layer dari shard safetensors."""
    var hidden = cfg.hidden_size
    var prefix = "model.language_model.layers." + String(layer_idx) + "."
    if (prefix + "input_layernorm.weight") not in weight_map:
        prefix = "model.layers." + String(layer_idx) + "."

    var norm_name = prefix + "input_layernorm.weight"
    var qkv_name = prefix + "linear_attn.in_proj_qkv.weight"
    var conv_name = prefix + "linear_attn.conv1d.weight"
    var z_name = prefix + "linear_attn.in_proj_z.weight"
    var a_name = prefix + "linear_attn.in_proj_a.weight"
    var b_name = prefix + "linear_attn.in_proj_b.weight"
    var dt_bias_name = prefix + "linear_attn.dt_bias"
    var a_log_name = prefix + "linear_attn.A_log"
    var norm_w_name = prefix + "linear_attn.norm.weight"
    var out_name = prefix + "linear_attn.out_proj.weight"

    var norm_gamma = _load_tensor_by_numel(
        cache, model_root, weight_map[norm_name], norm_name, hidden, telemetry
    )
    var w_qkv = _load_tensor_by_numel(
        cache,
        model_root,
        weight_map[qkv_name],
        qkv_name,
        8192 * hidden,
        telemetry,
    )
    var w_conv = _load_tensor_by_numel(
        cache,
        model_root,
        weight_map[conv_name],
        conv_name,
        8192 * 4,
        telemetry,
    )
    var w_z = _load_tensor_by_numel(
        cache,
        model_root,
        weight_map[z_name],
        z_name,
        4096 * hidden,
        telemetry,
    )
    var w_a = _load_tensor_by_numel(
        cache,
        model_root,
        weight_map[a_name],
        a_name,
        32 * hidden,
        telemetry,
    )
    var w_b = _load_tensor_by_numel(
        cache,
        model_root,
        weight_map[b_name],
        b_name,
        32 * hidden,
        telemetry,
    )
    var dt_bias = _load_tensor_by_numel(
        cache,
        model_root,
        weight_map[dt_bias_name],
        dt_bias_name,
        32,
        telemetry,
    )
    var a_log = _load_tensor_by_numel(
        cache,
        model_root,
        weight_map[a_log_name],
        a_log_name,
        32,
        telemetry,
    )
    var norm_weight = _load_tensor_by_numel(
        cache,
        model_root,
        weight_map[norm_w_name],
        norm_w_name,
        128,
        telemetry,
    )
    var w_out = _load_tensor_by_numel(
        cache,
        model_root,
        weight_map[out_name],
        out_name,
        hidden * 4096,
        telemetry,
    )

    return GDNWeights(
        norm_gamma^,
        w_qkv^,
        w_conv^,
        w_z^,
        w_a^,
        w_b^,
        dt_bias^,
        a_log^,
        norm_weight^,
        w_out^,
    )


def _gdn_softplus(v: Float32) -> Float32:
    if v > Float32(20.0):
        return v
    if v < Float32(-20.0):
        return exp(v)
    return log(Float32(1.0) + exp(v))


def forward_gdn_block(
    x: List[Float32],
    weights: GDNWeights,
    seq_len: Int,
    cfg: ModelConfig,
    eps: Float32,
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Eksekusi token mixer Gated DeltaNet (WY scan recurrence + residual 1)."""
    var hidden = cfg.hidden_size

    # 1. RMSNorm input x
    var x_norm = List[Float32]()
    x_norm.reserve(seq_len * hidden)
    var tok_vec = List[Float32]()
    tok_vec.resize(hidden, Float32(0.0))
    var p_tok = tok_vec.unsafe_ptr()
    var p_x = x.unsafe_ptr()
    for t in range(seq_len):
        var row = t * hidden
        for k in range(hidden):
            p_tok[unsafe_offset=k] = p_x[unsafe_offset=row + k]
        var normed = rmsnorm(tok_vec, weights.norm_gamma, eps)
        var p_normed = normed.unsafe_ptr()
        for k in range(hidden):
            x_norm.append(p_normed[unsafe_offset=k])

    # 2. QKV projection: [seq_len, hidden] x [8192, hidden]^T -> [seq_len, 8192]
    var qkv = List[Float32]()
    qkv.resize(seq_len * 8192, Float32(0.0))
    var p_qkv = qkv.unsafe_ptr()
    var p_xnorm = x_norm.unsafe_ptr()
    var p_wqkv = weights.w_qkv.unsafe_ptr()

    for t in range(seq_len):
        var x_row = t * hidden
        var out_row = t * 8192
        for j in range(8192):
            var w_row = j * hidden
            var acc_simd = SIMD[DType.float32, 16](0.0)
            var k = 0
            while k + 16 <= hidden:
                acc_simd += p_xnorm.unsafe_load[width=16](
                    x_row + k
                ) * p_wqkv.unsafe_load[width=16](w_row + k)
                k += 16
            var acc = acc_simd.reduce_add()
            while k < hidden:
                acc += (
                    p_xnorm[unsafe_offset=x_row + k]
                    * p_wqkv[unsafe_offset=w_row + k]
                )
                k += 1
            p_qkv[unsafe_offset=out_row + j] = acc

    # 3. 1D Causal Convolution: w_conv [8192, 4], causal pad 3 + SiLU
    var qkv_act = List[Float32]()
    qkv_act.resize(seq_len * 8192, Float32(0.0))
    var p_qkv_act = qkv_act.unsafe_ptr()
    var p_wconv = weights.w_conv.unsafe_ptr()

    for t in range(seq_len):
        var out_row = t * 8192
        for c in range(8192):
            var conv_val = Float32(0.0)
            var w_off = c * 4
            for k in range(4):
                var t_in = t - 3 + k
                if t_in >= 0:
                    conv_val += (
                        p_wconv[unsafe_offset=w_off + k]
                        * p_qkv[unsafe_offset=t_in * 8192 + c]
                    )
            p_qkv_act[unsafe_offset=out_row + c] = silu_f32(conv_val)

    # 4. In_proj_z [4096, hidden], In_proj_a [32, hidden], In_proj_b [32, hidden]
    var z_act = List[Float32]()
    z_act.resize(seq_len * 4096, Float32(0.0))
    var p_z = z_act.unsafe_ptr()
    var p_wz = weights.w_z.unsafe_ptr()

    for t in range(seq_len):
        var x_row = t * hidden
        var out_row = t * 4096
        for j in range(4096):
            var w_row = j * hidden
            var acc_simd = SIMD[DType.float32, 16](0.0)
            var k = 0
            while k + 16 <= hidden:
                acc_simd += p_xnorm.unsafe_load[width=16](
                    x_row + k
                ) * p_wz.unsafe_load[width=16](w_row + k)
                k += 16
            var acc = acc_simd.reduce_add()
            while k < hidden:
                acc += (
                    p_xnorm[unsafe_offset=x_row + k]
                    * p_wz[unsafe_offset=w_row + k]
                )
                k += 1
            p_z[unsafe_offset=out_row + j] = silu_f32(acc)

    var a_proj = List[Float32]()
    a_proj.resize(seq_len * 32, Float32(0.0))
    var p_a = a_proj.unsafe_ptr()
    var p_wa = weights.w_a.unsafe_ptr()

    var b_proj = List[Float32]()
    b_proj.resize(seq_len * 32, Float32(0.0))
    var p_b = b_proj.unsafe_ptr()
    var p_wb = weights.w_b.unsafe_ptr()

    for t in range(seq_len):
        var x_row = t * hidden
        var out_row = t * 32
        for j in range(32):
            var w_row = j * hidden
            var acc_a_simd = SIMD[DType.float32, 16](0.0)
            var acc_b_simd = SIMD[DType.float32, 16](0.0)
            var k = 0
            while k + 16 <= hidden:
                var x_vec = p_xnorm.unsafe_load[width=16](x_row + k)
                acc_a_simd += x_vec * p_wa.unsafe_load[width=16](w_row + k)
                acc_b_simd += x_vec * p_wb.unsafe_load[width=16](w_row + k)
                k += 16
            var acc_a = acc_a_simd.reduce_add()
            var acc_b = acc_b_simd.reduce_add()
            while k < hidden:
                var xv = p_xnorm[unsafe_offset=x_row + k]
                acc_a += xv * p_wa[unsafe_offset=w_row + k]
                acc_b += xv * p_wb[unsafe_offset=w_row + k]
                k += 1
            p_a[unsafe_offset=out_row + j] = acc_a
            p_b[unsafe_offset=out_row + j] = acc_b

    # 5. DeltaNet Recurrence Scan per Head (32 heads of dim 128)
    var out_tensor = List[Float32]()
    out_tensor.resize(seq_len * 4096, Float32(0.0))
    var p_out_t = out_tensor.unsafe_ptr()

    var s_matrix = List[Float32]()
    s_matrix.resize(32 * 128 * 128, Float32(0.0))
    var p_s = s_matrix.unsafe_ptr()

    var kt_normed = List[Float32]()
    kt_normed.resize(128, Float32(0.0))
    var p_kt_norm = kt_normed.unsafe_ptr()

    var v_sk = List[Float32]()
    v_sk.resize(128, Float32(0.0))
    var p_vsk = v_sk.unsafe_ptr()

    for t in range(seq_len):
        var t_act_row = t * 8192
        var t_ab_row = t * 32
        var t_out_row = t * 4096

        for h in range(32):
            var s_h_off = h * (128 * 128)
            var src_h = h // 2

            # Compute decay and beta for head h
            var a_val = p_a[unsafe_offset=t_ab_row + h]
            var dt = weights.dt_bias[h]
            var al = weights.a_log[h]
            var g = exp(-exp(al) * _gdn_softplus(a_val + dt))

            var b_val = p_b[unsafe_offset=t_ab_row + h]
            var beta_val = sigmoid_f32(b_val)

            # Extract q, k, v pointers for head h
            var q_off = t_act_row + src_h * 128
            var k_off = t_act_row + 2048 + src_h * 128
            var v_off = t_act_row + 4096 + h * 128

            # Normalize k: kt / (norm(kt) + eps)
            var sum_sq = Float32(0.0)
            for d in range(128):
                var kd = p_qkv_act[unsafe_offset=k_off + d]
                sum_sq += kd * kd
            var inv_norm_k = Float32(1.0) / (sqrt(sum_sq) + eps)
            for d in range(128):
                p_kt_norm[unsafe_offset=d] = (
                    p_qkv_act[unsafe_offset=k_off + d] * inv_norm_k
                )

            # Compute v_sk = S @ kt_normed (128)
            for r in range(128):
                var r_off = s_h_off + r * 128
                var acc_simd = SIMD[DType.float32, 16](0.0)
                var d = 0
                while d + 16 <= 128:
                    acc_simd += p_s.unsafe_load[width=16](
                        r_off + d
                    ) * p_kt_norm.unsafe_load[width=16](d)
                    d += 16
                var acc = acc_simd.reduce_add()
                while d < 128:
                    acc += (
                        p_s[unsafe_offset=r_off + d]
                        * p_kt_norm[unsafe_offset=d]
                    )
                    d += 1
                p_vsk[unsafe_offset=r] = acc

            # Update S: S = g * (S - beta * v_sk * kt_norm^T) + beta * vt * kt_norm^T
            for r in range(128):
                var r_off = s_h_off + r * 128
                var vr = p_qkv_act[unsafe_offset=v_off + r]
                var vsk_r = p_vsk[unsafe_offset=r]
                for c in range(128):
                    var kc = p_kt_norm[unsafe_offset=c]
                    var s_old = p_s[unsafe_offset=r_off + c]
                    p_s[unsafe_offset=r_off + c] = (
                        g * (s_old - beta_val * vsk_r * kc) + beta_val * vr * kc
                    )

            # Compute ot = S @ qt (128)
            var out_h_off = t_out_row + h * 128
            for r in range(128):
                var r_off = s_h_off + r * 128
                var acc_simd = SIMD[DType.float32, 16](0.0)
                var d = 0
                while d + 16 <= 128:
                    acc_simd += p_s.unsafe_load[width=16](
                        r_off + d
                    ) * p_qkv_act.unsafe_load[width=16](q_off + d)
                    d += 16
                var acc = acc_simd.reduce_add()
                while d < 128:
                    acc += (
                        p_s[unsafe_offset=r_off + d]
                        * p_qkv_act[unsafe_offset=q_off + d]
                    )
                    d += 1
                p_out_t[unsafe_offset=out_h_off + r] = acc

    # 6. Head RMSNorm with norm_weight [128] + Gate Z [4096]
    var p_normw = weights.norm_weight.unsafe_ptr()
    for t in range(seq_len):
        var t_out_row = t * 4096
        for h in range(32):
            var h_off = t_out_row + h * 128
            var sum_sq = Float32(0.0)
            for d in range(128):
                var val = p_out_t[unsafe_offset=h_off + d]
                sum_sq += val * val
            var rsqrt = Float32(1.0) / sqrt(sum_sq / Float32(128.0) + eps)
            for d in range(128):
                var normed = (
                    p_out_t[unsafe_offset=h_off + d]
                    * rsqrt
                    * p_normw[unsafe_offset=d]
                )
                p_out_t[unsafe_offset=h_off + d] = (
                    normed * p_z[unsafe_offset=h_off + d]
                )

    # 7. Out projection: [seq_len, 4096] x [hidden, 4096]^T -> [seq_len, hidden]
    var y = List[Float32]()
    y.resize(seq_len * hidden, Float32(0.0))
    var p_y = y.unsafe_ptr()
    var p_wout = weights.w_out.unsafe_ptr()

    for t in range(seq_len):
        var in_row = t * 4096
        var out_row = t * hidden
        for j in range(hidden):
            var w_row = j * 4096
            var acc_simd = SIMD[DType.float32, 16](0.0)
            var k = 0
            while k + 16 <= 4096:
                acc_simd += p_out_t.unsafe_load[width=16](
                    in_row + k
                ) * p_wout.unsafe_load[width=16](w_row + k)
                k += 16
            var acc = acc_simd.reduce_add()
            while k < 4096:
                acc += (
                    p_out_t[unsafe_offset=in_row + k]
                    * p_wout[unsafe_offset=w_row + k]
                )
                k += 1
            p_y[unsafe_offset=out_row + j] = acc

    # 8. Residual 1: x + y
    return add_residual(y, x, layer_idx)
