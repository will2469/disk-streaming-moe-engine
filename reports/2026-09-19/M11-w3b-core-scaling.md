# M11-W3b: Laporan Kalibrasi Core Scaling & Sintesis Hardware Profile

- **Tanggal**: 2026-09-19
- **CPU Model**: 12th Gen Intel(R) Core(TM) i3-1215U
- **Governor**: powersave
- **Topologi**: 8 logical CPUs (6 physical cores)
- **Plafon Komputasi**: $C_{compute\_max} = 7$, $C_{io} = 1$

## 1. Data Empiris Rezim 1 (Compute-Isolated)

| $c$ (Threads) | $T_{comp}$ p50 (ms) | $T_{comp}$ p95 (ms) | Speedup $S_{tok}$ | VmHWM (KiB) |
|:---:|:---:|:---:|:---:|:---:|
| **1** | 22.41 | 23.30 | 1.00x | 134536 |
| **2** | 13.56 | 16.16 | 1.65x | 134224 |
| **4** | 8.55 | 11.97 | 2.62x | 133836 |
| **7** | 6.85 | 10.61 | 3.27x | 134228 |

## 2. Fitting F16 Amdahl & Knee $c^*_{compute}$

- $T_1 = 22.7824\text{ ms}$
- $p = 0.8322$ ($83.22\%$ fraksi paralel)
- $\beta = 0.0350\text{ ms/thread}$ (overhead konkurensi)
- $e_{T,core} = 1.67\%$ (Ambang batas $\le 20\%$)
- **Knee Komputasi $c^*_{compute} = 7$ threads**

## 3. Sintesis Profil Sistem G-M11-1(b) (Tri-Pillar)

| Profile | $M_{budget}$ (GiB) | Hit Rate $h$ | $BW_{eff}$ (MB/s) | $c^*_{system}$ | $r^*_{system}$ | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `tier_8gb` | 8.0 | 10.0% | 2925 | **1** | 0.14 | ✅ VALID |
| `tier_16gb` | 16.0 | 35.0% | 2925 | **1** | 0.14 | ✅ VALID |
| `tier_32gb` | 32.0 | 65.0% | 2925 | **1** | 0.14 | ✅ VALID |
| `tier_64gb` | 64.0 | 90.0% | 2925 | **1** | 0.14 | ✅ VALID |
| `host_current` | 8.8 | 12.1% | 2925 | **1** | 0.14 | ✅ VALID |

## 4. Scorecard Gate G-M11-1

- Gate G-M11-1(a) (F16 Fit & Knee): **PASS**
- Gate G-M11-1(b) (Tri-Pillar Synthesis): **PASS**
- Artefak `dismoen.hardware.lock`: **10 Fields Valid**
- **OVERALL VERDICT**: **ALL GATES PASS (HIJAU)**
