# M11-W3b: Laporan Kalibrasi Core Scaling & Sintesis Hardware Profile

- **Tanggal**: 2026-09-18
- **CPU Model**: 12th Gen Intel(R) Core(TM) i3-1215U
- **Governor**: powersave
- **Topologi**: 8 logical CPUs (6 physical cores)
- **Plafon Komputasi**: $C_{compute\_max} = 7$, $C_{io} = 1$

## 1. Data Empiris Rezim 1 (Compute-Isolated)

| $c$ (Threads) | $T_{comp}$ p50 (ms) | $T_{comp}$ p95 (ms) | Speedup $S_{tok}$ | VmHWM (KiB) |
|:---:|:---:|:---:|:---:|:---:|
| **1** | 11.44 | 11.76 | 1.00x | 146248 |
| **2** | 6.10 | 6.56 | 1.87x | 146856 |
| **4** | 4.35 | 4.59 | 2.63x | 146344 |
| **7** | 2.57 | 3.48 | 4.45x | 146308 |

## 2. Fitting F16 Amdahl & Knee $c^*_{compute}$

- $T_1 = 11.9141\text{ ms}$
- $p = 0.8912$ ($89.12\%$ fraksi paralel)
- $\beta = 0.0000\text{ ms/thread}$ (overhead konkurensi)
- $e_{T,core} = 9.40\%$ (Ambang batas $\le 20\%$)
- **Knee Komputasi $c^*_{compute} = 7$ threads**

## 3. Sintesis Profil Sistem G-M11-1(b) (Tri-Pillar)

| Profile | $M_{budget}$ (GiB) | Hit Rate $h$ | $BW_{eff}$ (MB/s) | $c^*_{system}$ | $r^*_{system}$ | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `tier_8gb` | 8.0 | 10.0% | 2925 | **1** | 0.14 | ✅ VALID |
| `tier_16gb` | 16.0 | 35.0% | 2925 | **1** | 0.14 | ✅ VALID |
| `tier_32gb` | 32.0 | 65.0% | 2925 | **1** | 0.14 | ✅ VALID |
| `tier_64gb` | 64.0 | 90.0% | 2925 | **1** | 0.14 | ✅ VALID |
| `host_current` | 4.8 | 5.0% | 2925 | **1** | 0.14 | ✅ VALID |

## 4. Scorecard Gate G-M11-1

- Gate G-M11-1(a) (F16 Fit & Knee): **PASS**
- Gate G-M11-1(b) (Tri-Pillar Synthesis): **PASS**
- Artefak `dismoen.hardware.lock`: **10 Fields Valid**
- **OVERALL VERDICT**: **ALL GATES PASS (HIJAU)**
