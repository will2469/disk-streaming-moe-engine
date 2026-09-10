# Lampiran A — Contoh Perhitungan (Config Trial)

> Bagian dari `disk-streaming-moe-engine`. Index: `../README.md`.
> Rumus: `../02-math-models.md`. Semua angka memakai Qwen1.5-MoE-A2.7B-Chat.

1. **$W_{res}$ (F1)**: embedding + lm_head untied, keduanya 151.936 × 2048, dalam F32: $2 \times 151.936 \times 2048 \times 4\,\text{B} = 2.489.319.424$ B = **2,318 GiB**.
2. **KV cache (F2)**: MHA semua layer → $2 \times 24 \times 16 \times 128 \times 2\,\text{B} = 196.608$ B/token = **0,1875 MiB/token** → @4096 ctx = **0,75 GiB** → konteks trial dibatasi ≤ 8K (di 8 GB RAM, KV @32K = 6 GB tidak muat).
3. **Decode (F3+F5)**: `N_stream ≈ 2,0668 B` → $B_{tok}\approx\mathbf{4,134\ GB}$. Dengan $\rho\approx0,1154$ dan $BW_{eff}\approx4,3846$ GB/s, komponen I/O ≈ **0,943 s/token**; memakai placeholder $T_{comp}=0,05$ s memberi forecast **0,993 s/token ≈ 1,01 tok/s**. Trial tidak mengejar forecast ini — yang di-gate adalah kalibrasinya (G-M5-4).
4. **Kuantisasi (F11+F5)**: $bpw_{eff}=4,125$ → file ≈ **7,384 GB**; $B_{tok}^{4bit}\approx2,0668\,\text{B}\times4,125/8\approx\mathbf{1,066\ GB}$. Dengan estimasi $W_{stream,quant}\approx7,06$ GB → $\rho_{quant}\approx0,4248$ → $BW_{eff,quant}\approx\mathbf{8,097}$ GB/s. Memakai placeholder $T_{comp}=0,05$ s memberi forecast **0,182 s/token ≈ 5,51 tok/s**. Ini *forecast*, bukan hasil benchmark; G-M7-3 mengukur realitasnya.
