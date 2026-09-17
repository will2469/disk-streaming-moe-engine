# Laporan Sertifikasi Gate G-M8-1: 100 Sekuens Acak

- **Tanggal**: 2026-09-18
- **Gate**: G-M8-1 (Ekuivalensi Numerik Chunked vs Naive Oracle)
- **Kriteria**: $\Delta_{\max} \le 10^{-3}$, $\epsilon_{\text{rel}} \le 10^{-4}$
- **Konfigurasi**: `--threads 1` (single-threaded invariant), seed 42
- **Total Sekuens**: 100
- **Hasil**: 100/100 PASS
- **Delta Max Ekstrem**: 2.682209e-06
- **Eps Rel Ekstrem**: 1.464706e-06

## Sampel Hasil Pengujian (10 Interval Terpilih)

| ID | Profil | Dims ($d_k \times d_v$) | Sekuens ($s$) | Chunk ($C$) | $\Delta_{\max}$ | $\epsilon_{\text{rel}}$ | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Symmetric  (32x32) | 32x32 | 768 | 8 | 1.7583e-06 | 1.3224e-06 | **PASS** |
| 10 | Symmetric  (32x32) | 32x32 | 512 | 32 | 1.1325e-06 | 1.0046e-06 | **PASS** |
| 20 | Symmetric  (32x32) | 32x32 | 8 | 32 | 8.1956e-08 | 2.3303e-07 | **PASS** |
| 30 | Symmetric  (32x32) | 32x32 | 32 | 32 | 2.9802e-07 | 4.0176e-07 | **PASS** |
| 40 | Symmetric  (32x32) | 32x32 | 13 | 512 | 1.3411e-07 | 2.9291e-07 | **PASS** |
| 50 | Symmetric  (32x32) | 32x32 | 1000 | 128 | 1.6689e-06 | 1.3125e-06 | **PASS** |
| 60 | Asymmetric (32x48) | 32x48 | 513 | 64 | 1.2517e-06 | 9.9868e-07 | **PASS** |
| 70 | Asymmetric (32x48) | 32x48 | 337 | 32 | 1.1921e-06 | 9.3004e-07 | **PASS** |
| 80 | Asymmetric (32x48) | 32x48 | 687 | 512 | 1.5199e-06 | 1.1674e-06 | **PASS** |
| 90 | Asymmetric (32x48) | 32x48 | 337 | 32 | 9.8348e-07 | 9.2031e-07 | **PASS** |
| 100 | Asymmetric (32x48) | 32x48 | 47 | 16 | 3.7253e-07 | 4.9684e-07 | **PASS** |

## Kesimpulan Gate G-M8-1

**GATE G-M8-1 VERDICT: [PASS]** — 100/100 sekuens acak lolos ekuivalensi numerik.
