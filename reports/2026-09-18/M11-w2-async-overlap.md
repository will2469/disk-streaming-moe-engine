# Benchmark Report: Rezim 2 Steady-State Async Overlap (Gate G-M11-2)

**Date**: 2026-09-18 20:32:13
**Host CPU**: `12th Gen Intel(R) Core(TM) i3-1215U`
**CPU Governor**: `powersave`
**Engine Property $N_{in\_flight}$**: `2` chunks outstanding
**Rezim 2 Protocol**: Warm-up discarded = 3 tokens, Steady measured = 10 tokens

---

## 1. Summary Scorecard (Gate G-M11-2)

| Gate ID | Deskripsi Gate | Target Normatif | Nilai Terukur | Status |
|---|---|---|---|---|
| **G-M11-2 (a)** | Bandwidth Invariant $E_{BW} = \max_c \frac{|BW_{eff}(c) - BW_{ref}|}{BW_{ref}}$ (§1.3) | $\le 5.0\%$ | **1.91%** | **PASS** |
| **G-M11-2 (b)** | Overlap Efficiency $\mathcal{E}_{overlap}(c^*)$ (F18, steady-state) | $\ge 80.0\%$ | **95.99%** | **PASS** |
| **G-M11-2 (c)** | Dedicated I/O Worker & Engine Property $N_{in\_flight} \in [2, 4]$ | $N_{in\_flight} \in [2, 4]$ | **2** | **PASS** |

> **Verdict**: **ALL GATES PASS (Gate G-M11-2 HIJAU)**

---

## 2. Tabel Sweep Multithreading Core ($c \in \{1, 2, 4\}$)

| $c$ (Threads) | $T_{IO}$ (ms) | $T_{comp}(c)$ (ms) | $T_{seq}(c)$ (ms) | $T_{overlap}(c)$ (ms) | $BW_{eff}(c)$ (MB/s) | $\Delta BW$ (vs Ref) | $\mathcal{E}_{overlap}(c)$ (%) |
|---|---|---|---|---|---|---|---|
| **1** | 9.50 | 11.11 | 20.61 | 12.58 | 2941.1 | 0.00% | **84.52%** |
| **2** | 9.56 | 6.11 | 15.67 | 9.87 | 2924.6 | 0.56% | **94.96%** |
| **4** | 9.69 | 4.27 | 13.96 | 9.86 | 2884.9 | 1.91% | **95.99%** |

---

## 3. Analisis Latency Hiding & Formulasi F18

Pada arsitektur asynchronous double-buffering, waktu langkah steady-state mengikuti:
$$T_{step}^{overlap}(c) = \max(T_{IO}, T_{comp}(c)) + \epsilon_{sync}$$

Efisiensi latency hiding dihitung via Formula F18:
$$\mathcal{E}_{overlap}(c) = \frac{(T_{IO} + T_{comp}(c)) - T_{step}^{overlap}(c)}{\min(T_{IO}, T_{comp}(c))} \times 100\%$$

Pada titik deploy $c^* = 4$:
- Komponen I/O: $T_{IO} = 9.69\text{ ms}$
- Komponen komputasi: $T_{comp}(4) = 4.27\text{ ms}$
- Waktu langkah terukur: $T_{step}^{overlap} = 9.86\text{ ms}$
- Efisiensi overlap terukur: **95.99%** (melampaui threshold normatif $80.0\%$).
- Penegakan kestabilan bandwidth storage membuktikan ketiadaan memory bus contention antara I/O streaming dan CPU multi-core ($E_{BW} = 1.91\% \le 5.00\%$).
