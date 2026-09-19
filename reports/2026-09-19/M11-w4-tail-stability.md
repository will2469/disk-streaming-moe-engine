# M11-W4: Laporan Tail Latency & Dynamic RAM Budget Adherence

- **Tanggal**: 2026-09-19
- **CPU Model**: 12th Gen Intel(R) Core(TM) i3-1215U
- **Governor**: powersave
- **Titik Uji**: $c^*_{system} = 1$ threads (Profil `host_current`)
- **Jumlah Run**: $N = 100$ steady iterations

## 1. Distribusi Latensi Ekor & Project SLO

| Metrik | Nilai Empiris (ms) | 95% Bootstrap CI | Keterangan |
|:---|:---:|:---:|:---|
| $p50$ (Median) | **22.23** | [22.10, 22.40] | Interpolasi Linear Type 7 |
| $p90$ | 23.31 | - | Distribusi Ekor |
| $p95$ | **23.76** | [23.31, 25.22] | Evaluasi Project SLO |
| $p99$ | 25.70 | - | Ekor Ekstrem |
| Min / Max | 21.70 / 27.57 | - | Rentang Penuh |
| Mean (Std) | 22.49 (±0.87) | - | Statistik Agregat |
| **$R_{tail} = p95/p50$** | **1.0690** | [1.0486, 1.1343] | **Project SLO $\le 1{,}35$** |

## 2. Kepatuhan Dynamic RAM Budget ($\mathcal{R}_{RAM} \le 0{,}95$)

- **Puncak Pemakaian Fisik (VmHWM)**: **134392 KiB** (131.24 MiB)
- **Uji Stabilitas Leak**: Run 1 = 134392 KiB, Run 2 = 134032 KiB (Delta 0.27% $\le 5\%$, ✅ LEAK-FREE)

| Tier RAM Budget | Anggaran $M_{budget}$ | $\mathcal{R}_{RAM} = \text{VmHWM}/M_{budget}$ | Batas Maksimum | Status |
|:---|:---:|:---:|:---:|:---:|
| `tier_8gb` | 8.0 GiB | 1.60% | $\le 95\%$ | ✅ PASS |
| `tier_16gb` | 16.0 GiB | 0.80% | $\le 95\%$ | ✅ PASS |
| `tier_32gb` | 32.0 GiB | 0.40% | $\le 95\%$ | ✅ PASS |
| `tier_64gb` | 64.0 GiB | 0.20% | $\le 95\%$ | ✅ PASS |
| `host_active` | 8.8 GiB | 1.45% | $\le 95\%$ | ✅ PASS |

## 3. Asersi Stabilitas Throughput Storage ($E_{BW} \le 5\%$)

- **Laporan Sumber W2b**: `m11_w2_async_overlap.json`
- **Variasi Throughput Maksimum $E_{BW}$**: **4.70%**
- **Kriteria Stabilitas**: $E_{BW} \le 5.0\%$ (✅ TERPENUHI)

## 4. Formal Scorecard Gate G-M11-3

- [x] Ukuran Sampel $N \ge 100$: **PASS** ($N = 100$)
- [x] Project SLO $R_{tail} \le 1.35$: **PASS** ($R_{tail} = 1.0690$)
- [x] Dynamic RAM Budget Adherence $\mathcal{R}_{RAM} \le 0.95$: **PASS**
- [x] Verifikasi Tanpa Kebocoran Memori (Leak-Free): **PASS**
- [x] Stabilitas Bandwidth Storage $E_{BW} \le 5\%$: **PASS**
- **OVERALL VERDICT**: **ALL GATES PASS (HIJAU)**
