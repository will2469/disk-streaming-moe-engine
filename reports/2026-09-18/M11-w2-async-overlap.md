# Benchmark Report: Rezim 2 Steady-State Async Overlap (Gate G-M11-2)

- **Date**: 2026-09-18 22:03:51
- **Host CPU**: `12th Gen Intel(R) Core(TM) i3-1215U`
- **CPU Governor**: `powersave`
- **Engine Property $N_{in\_flight}$**: `2` chunks
- **Rezim 2 Protocol**: Warm-up=3, Steady=10

---

## 1. Summary Scorecard (Gate G-M11-2)

| Gate ID | Deskripsi Gate | Target Normatif | Nilai Terukur | Status |
|---|---|---|---|---|
| **G-M11-2 (a)** | Bandwidth Invariant $E_{BW}$ (§1.3) | $\le 5.0\%$ | **1.59%** | **PASS** |
| **G-M11-2 (b)** | Overlap Efficiency $\mathcal{E}_{overlap}$ | $\ge 80.0\%$ | **80.55%** | **PASS** |
| **G-M11-2 (c)** | Dedicated I/O Worker & $N_{in\_flight}$ | $N_{in\_flight} \in [2, 4]$ | **2** | **PASS** |

> **Verdict**: **ALL GATES PASS (Gate G-M11-2 HIJAU)**

---

## 2. Tabel Sweep Multithreading Core ($c \in \{1, 2, 4\}$)

| $c$ | $T_{IO}$ (ms) | $T_{comp}(c)$ (ms) | $T_{seq}(c)$ (ms) | $T_{overlap}(c)$ (ms) | $BW_{eff}(c)$ (MB/s) | $\Delta BW$ | $\mathcal{E}_{overlap}(c)$ (%) |
|---|---|---|---|---|---|---|---|
| **1** | 10.67 | 12.02 | 22.69 | 14.82 | 2618.6 | 0.00% | **73.72%** |
| **2** | 10.51 | 6.18 | 16.69 | 10.93 | 2660.1 | 1.59% | **93.09%** |
| **4** | 10.55 | 4.24 | 14.79 | 11.37 | 2650.5 | 1.22% | **80.55%** |

---

## 3. Analisis Latency Hiding & Formulasi F18

Pada arsitektur asynchronous double-buffering:
$$T_{step}^{overlap}(c) = \max(T_{IO}, T_{comp}(c)) + \epsilon_{sync}$$

Efisiensi latency hiding dihitung via Formula F18:
$$\mathcal{E}_{overlap}(c) = \frac{(T_{IO} + T_{comp}(c)) - T_{step}^{overlap}(c)}{\min(T_{IO}, T_{comp}(c))} \times 100\%$$

Pada titik deploy $c^* = 4$:
- Komponen I/O: $T_{IO} = 10.55\text{ ms}$
- Komponen komputasi: $T_{comp}(4) = 4.24\text{ ms}$
- Waktu langkah terukur: $T_{step}^{overlap} = 11.37\text{ ms}$
- Efisiensi overlap terukur: **80.55%** (melampaui threshold normatif $80.0\%$).
- Kestabilan bandwidth membuktikan zero memory bus contention ($E_{BW} = 1.59\% \le 5.00\%$).
