# Calibration Report: Formulas F1, F2, F5 & Gate G-M9-4

- **Date**: 2026-09-18
- **Target Architecture**: `Qwen3.6-35B-A3B`
- **Specification Document**: `docs/milestones/M9-port.md`

---

## 1. Gate G-M9-4: Formula F2 KV Cache Scaling

Formula F2 port:
$$M_{KV}(s) = 2 \cdot L_{att} \cdot H_{kv} \cdot d_h \cdot s \cdot b_{KV}$$
Dengan $b_{KV} = 2\text{ B}$ (BF16), untuk konfigurasi mini:
$M_{KV}(s) = 128 \times s\text{ B}$.
Untuk model penuh ($L_{att}=10, H_{kv}=2, d_h=128$):
$M_{KV}(s) = 10,240 \times s\text{ B} = 10\text{ KiB/token}$.

| Seq ($s$) | Pred F2 | Meas Payload | Allocated | e_KV (%) | Gate G-M9-4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | 1,024 B | 1,024 B | 65,536 B | 0.00% | [PASS] |
| 16 | 2,048 B | 2,048 B | 65,536 B | 0.00% | [PASS] |
| 64 | 8,192 B | 8,192 B | 65,536 B | 0.00% | [PASS] |
| 128 | 16,384 B | 16,384 B | 65,536 B | 0.00% | [PASS] |
| 256 | 32,768 B | 32,768 B | 65,536 B | 0.00% | [PASS] |
| 512 | 65,536 B | 65,536 B | 73,728 B | 0.00% | [PASS] |
| 1024 | 131,072 B | 131,072 B | 139,264 B | 0.00% | [PASS] |
| 4096 | 524,288 B | 524,288 B | 532,480 B | 0.00% | [PASS] |

**Verdict Gate G-M9-4**: **[PASS]** (Seluruh deviasi $e_{KV} \le 5\%$).

---

## 2. Gate G-M9-2: Formula F1-Port Bottom-Up Memory Budget

$$M_{{peak}}^{{M9}} = W_{{res}} + M_{{cache}} + M_{{expert}} + M_{{KV}} + M_{{GDN}} + M_{{scratch}} + M_{{ws}} \le 7{{,}}5\text{{ GiB}}$$

| Komponen | Nominal | Cap | Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| $W_{res}$ | 0.50 GiB | 1.00 GiB | $\le 1.00$ GiB | PASS |
| $M_{cache}$ | 1.00 GiB | 2.00 GiB | $\le 2.00$ GiB | PASS |
| $M_{expert}$ | 0.11 GiB | 0.15 GiB | $\le 0.15$ GiB | PASS |
| $M_{KV}(4K)$ | 0.039 GiB | 0.50 GiB | $\le 0.50$ GiB | PASS |
| $M_{GDN}$ | 0.0018 GiB | 0.005 GiB | $\le 0.005$ GiB | PASS |
| **$M_{peak}$** | **2.11 GiB** | **4.40 GiB** | **$\le 7.50$ GiB** | **[PASS]** |

**Verdict Gate G-M9-2**: **[PASS]** (Total batas keras $\le 4.41\text{ GiB} \ll 7.5\text{ GiB}$).

---

## 3. Formula F5: Decode Step Time & Kalibrasi $e_T$

$$T_{{tok}} = T_{{data}} + T_{{comp}} + T_{{ovh}}, \qquad e_T = \left| \frac{{T^{{pred}} - T^{{meas}}}}{{T^{{meas}}}} \right| \le 20\%$$

- **Measured Decode Step ($T^{meas}$)**: `0.159 ms`
- **Predicted Step ($T^{pred}$)**: `0.162 ms`
- **Observed Error ($e_T$)**: `2.00%` (Ambang batas $\le 20\%$)
- **Verdict**: **[PASS]** ($e_T \le 20\%$, konstanta terkalibrasi).
