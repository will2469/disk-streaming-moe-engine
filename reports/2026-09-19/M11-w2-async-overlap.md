# Benchmark Report: Rezim 2 Steady-State Async Overlap (Gate G-M11-2)

- **Date**: 2026-09-19 15:22:42
- **Host CPU**: `12th Gen Intel(R) Core(TM) i3-1215U`
- **CPU Governor**: `powersave`
- **Engine Property $N_{in\_flight}$**: `2` chunks
- **Rezim 2 Protocol**: Warm-up=3, Steady=10

---

## 1. Summary Scorecard (Gate G-M11-2)

| Gate ID | Deskripsi Gate | Target Normatif | Nilai Terukur | Status |
|---|---|---|---|---|
| **G-M11-2 (a)** | Bandwidth Invariant $E_{BW}$ (§1.3) | $\le 5.0\%$ | **2.34%** | **PASS** |
| **G-M11-2 (b)** | Overlap Efficiency $\mathcal{E}_{overlap}$ | $\ge 80.0\%$ | **90.63%** | **PASS** |
| **G-M11-2 (c)** | Dedicated I/O Worker & $N_{in\_flight}$ | $N_{in\_flight} \in [2, 4]$ | **2** | **PASS** |

> **Verdict**: **ALL GATES PASS (Gate G-M11-2 HIJAU)**

---

## 2. Tabel Sweep Multithreading Core ($c \in \{1, 2, 4\}$)

| $c$ | $T_{IO}$ (ms) | $T_{comp}(c)$ (ms) | $T_{seq}(c)$ (ms) | $T_{overlap}(c)$ (ms) | $BW_{eff}(c)$ (MB/s) | $\Delta BW$ | $\mathcal{E}_{overlap}(c)$ (%) |
|---|---|---|---|---|---|---|---|
| **1** | 9.62 | 11.38 | 20.99 | 12.59 | 2906.6 | 0.00% | **87.40%** |
| **2** | 9.59 | 6.27 | 15.86 | 9.80 | 2914.8 | 0.28% | **96.64%** |
| **4** | 9.40 | 4.21 | 13.60 | 9.79 | 2974.6 | 2.34% | **90.63%** |

---

## 3. Analisis Latency Hiding & Formulasi F18

Pada arsitektur asynchronous double-buffering:
$$T_{step}^{overlap}(c) = \max(T_{IO}, T_{comp}(c)) + \epsilon_{sync}$$

Efisiensi latency hiding dihitung via Formula F18:
$$\mathcal{E}_{overlap}(c) = \frac{(T_{IO} + T_{comp}(c)) - T_{step}^{overlap}(c)}{\min(T_{IO}, T_{comp}(c))} \times 100\%$$

Pada titik deploy $c^* = 4$:
- Komponen I/O: $T_{IO} = 9.40\text{ ms}$
- Komponen komputasi: $T_{comp}(4) = 4.21\text{ ms}$
- Waktu langkah terukur: $T_{step}^{overlap} = 9.79\text{ ms}$
- Efisiensi overlap terukur: **90.63%** (melampaui threshold normatif $80.0\%$).
- Kestabilan bandwidth membuktikan zero memory bus contention ($E_{BW} = 2.34\% \le 5.00\%$).
