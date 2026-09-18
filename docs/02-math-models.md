# 02 — Model Matematika Inti

> Bagian dari `disk-streaming-moe-engine`. Index: `README.md`.
> Setiap rumus punya ID `F#` yang dirujuk oleh gate di `04-quality.md` dan kontrol security di `05-security.md`. Contoh angka memakai config trial; angka port ditandai TBM.

| ID  | Rumus                                                                        | Dipakai di                                 |
| --- | ---------------------------------------------------------------------------- | ------------------------------------------ |
| F1  | Anggaran memori peak                                                         | `01-architecture.md` §2.5, G-M1..M9, SEC-4 |
| F2  | Ukuran KV cache                                                              | M5, M9, Lampiran A                         |
| F3  | Bytes per forward / per token                                                | M5–M7, benchmark                           |
| F4  | Model roofline (memory- vs compute-bound)                                    | `03-testing.md` §4.4, M9                   |
| F5  | Waktu per token + bandwidth efektif                                          | M5–M7, M9                                  |
| F6  | RMSNorm                                                                      | M2                                         |
| F7  | RoPE rotate_half + invariant isometri                                        | M2, property test                          |
| F8  | Router top-k tanpa renorm + shared sigmoid gate + SwiGLU                     | M3 (paling kritis)                         |
| F9  | Diagnostik load-balance expert                                               | analisis routing, koreksi F5               |
| F10 | Metrik ekivalensi numerik + propagasi error                                  | `03-testing.md` §4.3, semua gate MATCH     |
| F11 | Kuantisasi simetris 4-bit (M6); Distorsi multi-format GGUF F11-GGUF (M9)     | M6, M9                                     |
| F12 | Perplexity & ΔPPL                                                            | M6, M9                                     |
| F13 | Hit rate LRU + bandwidth efektif                                             | M7                                         |
| F14 | Delta rule / Gated DeltaNet                                                  | M8                                         |
| F15 | Predikat validitas struktural safetensors (F15a/b/c; SHA = SEC-1, bukan F15) | `05-security.md` (SEC-1..3)                |
| F16 | Kurva skala core (rasio, device-agnostic)                                    | M5, M7, benchmark                          |
| F17 | Pola I/O storage (seq vs expert-size, QD, sustained)                         | M7, benchmark                              |
| F18 | Asynchronous double-buffering latency overlap & tail stability               | M11, benchmark                             |

## 3.1 Memori & Streaming

**F1 — Anggaran memori** (sudah ditulis di `01-architecture.md` §2.5):

- Model trial: $M_{peak}^{trial} = W_{res} + M_{KV} + M_{ws} + M_{io} \le M_{gate}$ ($M_{gate} \le 3{,}5\text{--}5\text{ GiB}$).
- Model port M9 (streaming hybrid 35B):
  $$M_{peak}^{port} = W_{res} + M_{cache} + M_{expert} + M_{KV} + M_{GDN} + M_{scratch} + M_{ws\_runtime} \le M_{gate} = 7{,}5\text{ GiB}$$
  dengan:
  - $W_{res}$: embedding + `lm_head` terkuantisasi (Q4_K/Q8_0) $\le 1{,}00\text{ GiB}$.
  - $M_{cache}$: cache LRU terikat (pola M7) bobot terkompresi Q3_K expert panas $\le 2{,}00\text{ GiB}$.
  - $M_{expert}$: buffer komputasi FP32 untuk 8 routed + 1 shared expert aktif $\le 0{,}15\text{ GiB}$ ($9 \times 3 \times 512 \times 2048 \times 4\text{ B} \approx 113{,}2\text{ MB}$).
  - $M_{KV}(s)$: KV cache GQA $10\text{ layer} \times 2\text{ KV heads} \times 128 \times s \times 2\text{ B} \approx 10\text{ KiB/tok} \le 0{,}50\text{ GiB}$ (@ $\le 50\text{K}$ ctx).
  - $M_{GDN}$: 30 recurrent states independen $30 \times 128 \times 128 \times 4\text{ B} = 1{,}875\text{ MiB} \approx 0{,}002\text{ GiB}$ (konstan $O(1)$).
  - $M_{scratch}$: O_DIRECT aligned pread buffer + scratchpad dequant layer $\le 0{,}25\text{ GiB}$.
  - $M_{ws\_runtime}$: buffer aktivasi token, GGUF metadata index ($<1\text{ MB}$), thread stacks & allocator pools $\le 0{,}50\text{ GiB}$.
    Total nominal streaming $M_{peak}^{port} \approx 2{,}11\text{ GiB}$; worst-case alokasi penuh $\le 4{,}41\text{ GiB} \ll 7{,}5\text{ GiB}$ (Gate G-M9-2).

**F2 — KV cache** (hanya layer dengan attention penuh):

$$M_{KV}(s) = 2 L_{att} H_{kv} d_h(\text{config}) s b_{KV}$$

Dengan:

- **Faktor 2**: Merepresentasikan sepasang tensor Key ($K$) dan Value ($V$) yang wajib hidup di data layout fisik: `[2, s, L_{att}, H_{kv}, d_h]` atau dual tensor `(K, V)`.
- $d_h(\text{config}) = \frac{\text{hidden\_size}}{\text{num\_attention\_heads}} = \frac{2048}{16} = 128$ (dihitung dinamis dari config model).
- $b_{KV}$ = byte per elemen **yang disimpan** di cache (trial: BF16, jadi 2 B; jangan otomatis mengikuti dtype bobot/dequant). Trial (MHA, semua layer): $2 \times 24 \times 16 \times 128 \times 2\,\text{B} = 196.608$ B/token = **0,1875 MiB/token** → @4096 ctx = **0,75 GiB** → konteks praktis trial ≤ 8K di RAM 8 GB. Port M9: berdasarkan layout resmi `10 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE))`, ada **10/40 = 1/4** layer full-attention; GQA = 2 KV heads. Maka F2 mengunci $L_{att}=10$, $H_{kv}=2$, $d_h=128$, dan $b_{KV}=2\text{ B}$ (BF16, SSOT runtime sesuai format sesi `KMSS v1` `kv_dtype = 2`), menghasilkan $M_{KV}(1) = 2 \times 10 \times 2 \times 128 \times 2 = \mathbf{10.240\text{ B/token}} = \mathbf{10\text{ KiB/tok}}$ (@4K ctx = 40 MiB; @32K ctx = 320 MiB). [R4]
- Evaluasi Gate G-M9-4 ($e_{KV} \le 5\%$) membandingkan $M_{KV}(s)$ langsung terhadap `metrics.kv_cache.kv_payload_bytes` yang diekspos engine (bukan diekstrak dari RSS / VmHWM).
  Dipakai di: `milestones/M5-kv-decode.md`, `milestones/M9-port.md`.

**F3 — Bytes yang dibaca dari disk:**

$$B_{fwd}(s) \approx W_{file}$$

Untuk trial, $W_{file}=28{,}63$ GB. Pada prefill streaming, bobot setiap layer dibaca sekali untuk seluruh $s$ token.

**F3b — Bytes bobot yang benar-benar disentuh per token decode.** `N_act` resmi bukan lagi dipakai sebagai sinonim untuk bytes yang di-stream. Karena embedding + lm_head sudah resident, F3b menghitung hanya bobot badan transformer yang dibaca ulang saat decode.

$$N_{stream}=L[N_{attn}+N_{router}+N_{routed,k}+N_{shared}+N_{norm}]$$

$$N_{stream}\approx2{,}0668\times10^9$$

$$B_{tok}=bN_{stream}$$

Untuk BF16 trial, $B_{tok}\approx4{,}134$ GB per token.
dengan `GB` = $10^9$ byte. `N_stream` adalah besaran **model-specific streaming set**, bukan angka “activated parameters” generik dari model card. [R1][R2]

Catatan penting: tanpa KV cache (M0–M4), menghasilkan $n$ token baru berarti mengulang prefill → boros $\times n$; itulah motivasi M5.

**F4 — Model roofline:**

$$I=F/B$$

$$P=\min(P_{peak}, I BW)$$

$$I_{decode}^{weight}\approx2N_{stream}/B_{tok}$$

Untuk trial, $I_{decode}^{weight}\approx1$ FLOP/byte.
Ini adalah intensitas **weight-only** (FMA dihitung 2 FLOP), bukan intensitas end-to-end: pembacaan KV/aktivasi dan kerja selain matmul belum dimodelkan. Transfer tambahan biasanya menurunkan intensitas, tetapi FLOPs tambahan harus dihitung juga; verdict tetap berasal dari benchmark.
Karena numerator dan denominator memakai himpunan bobot yang sama, rasio ini **self-canceling** terhadap perubahan definisi $N_{act}$ yang memasukkan embedding/lm_head. Kesimpulan memory-bound tetap bergantung pada posisi ridge point mesin target, sehingga verdict final tetap harus berasal dari benchmark `P_cpu` dan bandwidth nyata.

Implikasi arsitektural: optimasi performa = **kurangi bytes** (F3b, kuantisasi M6) atau **naikkan bandwidth efektif** (F5, F13), bukan optimasi FLOPs. Prefill dengan $s$ besar bergerak ke compute-bound — itu sebabnya konteks sangat panjang mahal di CPU meski KV-nya kecil.

**F5 — Waktu per token & bandwidth efektif:**

Jika fraksi byte $\rho_B$ datang dari RAM dan sisanya dari disk, waktu transfernya yang dijumlahkan:

$$T_{data}=B_{tok}(\rho_B/BW_{RAM}+(1-\rho_B)/BW_{SSD})$$

$$BW_{eff}=B_{tok}/T_{data}$$

Rata-rata aritmetika $\rho_B BW_{RAM}+(1-\rho_B)BW_{SSD}$ salah untuk sumber bytes serial; ia hanya berlaku bila bandwidth sumber benar-benar dipakai paralel untuk byte yang sama. Untuk jadwal tanpa overlap (forecast konservatif yang dipakai contoh di bawah),

$$T_{tok}=T_{data}+T_{comp}+T_{ovh}$$

Prefetch yang terukur dapat menumpangtindihkan compute dan I/O, sehingga batas idealnya $T_{tok}=\max(T_{data},T_{comp})+T_{ovh}$; laporan wajib menyebut jadwal yang dipakai, bukan mengklaim overlap dari rumus saja.

$$\rho_B=B_{RAM}/(B_{RAM}+B_{SSD})$$

$$\rho_C\approx\min(1,C_{pc}/W_{stream})$$

$\rho_B$ adalah fraksi **byte** terukur. $\rho_C$ hanya estimasi kapasitas awal untuk $\rho_B$, dengan asumsi blok yang direuse/routing seragam dan seluruh $C_{pc}$ tersedia; bias routing dikoreksi melalui F9 dan hasil ukur cache (F13).

Contoh trial (estimasi, bukan hasil ukur): $C_{pc} \approx 3$ GB, $W_{stream} \approx 26$ GB → $\rho_C=3/26\approx\mathbf{0{,}1154}$. Dengan asumsi $\rho_B=\rho_C$, $BW_{RAM}=15$ GB/s, dan $BW_{SSD}=3$ GB/s, diperoleh $BW_{eff}\approx\mathbf{3{,}305}$ GB/s. Dengan $B_{tok}=4{,}134$ GB, komponen I/O memberi $T_{data}\approx\mathbf{1{,}251}$ s/token. Untuk forecast serial dengan placeholder $T_{comp}=0{,}05$ s dan $T_{ovh}=0$, $T_{tok}\approx\mathbf{1{,}301}$ s/token ≈ **0,77 tok/s**. Placeholder `T_comp` wajib diganti hasil ukur sebelum dipakai untuk acceptance; angka ini hanya forecast engineering.

**Floor bandwidth (gate minimal, bukan estimasi):** $BW_{RAM}$ tipikal memakai 15 GB/s (kelas DDR4 umum), sedangkan **floor minimal $BW_{RAM} \ge 10$ GB/s** (G-M5-6, single-thread Copy read-equiv). Di floor ini target M7-3 tetap lolos dalam forecast serial: $BW_{eff,quant}=\left(0{,}4248/10+0{,}5752/2{,}5\right)^{-1}\approx\mathbf{3{,}67}$ GB/s → $T_{tok}\approx1{,}066/3{,}67+0{,}05\approx\mathbf{0{,}34}$ s/token ≈ **2,94 tok/s** ≥ 2 tok/s. Teoritis DDR4-3200 = 3200 MT/s × 8 B = 25,6 GB/s per kanal; yang di-gate adalah bandwidth _sustainable_ terukur ala STREAM [R21], bukan angka teoritis.

## 3.2 Komponen Model

**F6 — RMSNorm:**

$$y = \frac{x}{\mathrm{RMS}(x)} \odot \gamma, \qquad \mathrm{RMS}(x) = \sqrt{\tfrac{1}{d}\textstyle\sum_i x_i^2 + \varepsilon}$$

$\varepsilon$ dan dtype (fp32) wajib identik dengan oracle; nilai $\varepsilon$ diambil dari config dan diverifikasi silang ke kode modeling Qwen (TBM). Dipakai di M1/M2.

**F7 — RoPE (rotate_half):**

$$\theta_i = m \cdot \omega_i, \quad \omega_i = \mathrm{base}^{-2i/d_h}, \qquad \begin{pmatrix} q'_i \\ q'_{i+d_h/2} \end{pmatrix} = \begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix} \begin{pmatrix} q_i \\ q_{i+d_h/2} \end{pmatrix}$$

Invariant (property test P-3): $\lVert q' \rVert = \lVert q \rVert$ — rotasi ortogonal. Pelanggaran invariant = implementasi terpeleset ke gaya interleaved. Dipakai di `milestones/M2-attention.md`.

**F8 — Router + shared expert + SwiGLU (bagian paling kritis, M3) [R2][R3]:**

**F8a — Router:**

$$p=softmax(W_r x)$$

$p$ memiliki 60 elemen dan dihitung di fp32.

**F8b — Seleksi:**

$$\mathcal{A}=Top4(p)$$

Probabilitas expert terpilih dipakai apa adanya; tidak ada renormalisasi.

**F8c — Kombinasi:**

$$y_{routed}=\sum_{i\in\mathcal{A}}p_iE_i(x)$$

$$y=y_{routed}+\sigma(g_{sh})E_{sh}(x)$$

**F8d — Expert SwiGLU:**

$$u=W_{gate}x$$

$$v=W_{up}x$$

$$E(x)=W_{down}(SiLU(u)\odot v)$$

$$SiLU(z)=z\sigma(z)$$

Dua invariant keras: (a) **SET expert terpilih** ($\mathcal{A}$) harus identik dengan oracle untuk semua input uji — pergeseran probabilitas kecil boleh, flip seleksi = FAIL kategori `router-selection` (`03-testing.md` §4.3); (b) gate shared = **sigmoid** (F8c), bukan softmax. Dipakai di `milestones/M3-moe.md`.

**F9 — Diagnostik load-balance (analisis, bukan training):**

$$f_i = \frac{1}{T}\sum_t \mathbb{1}[i \in \mathcal{A}_t], \qquad P_i = \frac{1}{T}\sum_t p_i^{(t)}, \qquad \mathcal{L}_{lb} = N_e \sum_{i=1}^{N_e} f_i P_i, \qquad CV = \frac{\sigma_f}{\mu_f}$$

Dipakai untuk: (a) mendeteksi bias routing pada corpus golden; (b) mengoreksi asumsi $\rho$ seragam di F5 ketika $CV$ tinggi (expert panas bisa di-pin di page cache / LRU pada M7).

## 3.3 Ekivalensi Numerik (F10) — kontrak semua gate MATCH

**F10a — Error vektor:**

$$\Delta_{max}=\max_i|\hat{x}_i-x_i|$$

$$\varepsilon_{rel}=\lVert\hat{x}-x\rVert_2/\max(\lVert x\rVert_2,\tau)$$

$$cos=\langle\hat{x},x\rangle/(\lVert\hat{x}\rVert\lVert x\rVert)$$

**F10b — Keputusan output:**

$$\mathbb{A}=n^{-1}\sum_t\mathbb{1}[argmax\ \hat{x}_t=argmax\ x_t]$$

$$\Delta_{CE}=|CE(\hat{x})-CE(x)|$$

Untuk $\lVert x\rVert_2<\tau$, cosine tidak terdefinisi jika salah satu vektor nol; laporkan `N/A` dan gunakan $\Delta_{max}$/error absolut. Tetapkan $\tau$ di artefak benchmark agar metrik tidak berubah karena pembagian nol.

**Propagasi error antar-layer:** jika tiap layer menyumbang error relatif $\le \delta$, bound kasar untuk full forward adalah $\varepsilon_{full} \lesssim L \cdot \delta = 24\delta$ (asumsi amplifikasi kecil). Bound ini hanya penunjuk arah — yang di-gate adalah hasil ukur M4, bukan bound-nya.

**Softmax stabil wajib** (oracle dan engine sama-sama): $\mathrm{softmax}(z) = \dfrac{\exp(z - \max z)}{\sum \exp(z - \max z)}$ — tanpa pengurangan max, fp32 meluap pada logits besar dan verdict jadi tidak bermakna.

Threshold verdict per milestone ada di `03-testing.md` §4.3 dan diulang di tiap file `milestones/M*.md`.

## 3.4 Kuantisasi & Kualitas Output

**F11 — Kuantisasi simetris per-grup** (M6; grup $G$ = 128 bobot, skala fp16):

**F11a — Quantize dan dequantize:**

$$a_g = \max_{j \in G}|w_j|$$

Pilih $s_g$ sebagai nilai FP16 finite terkecil yang memenuhi $s_g \ge a_g/7$. Bila $a_g=0$, tetapkan $s_g=1$ dan seluruh $q_j=0$; bila $a_g/7$ tak dapat diwakili FP16 finite, kuantisasi gagal (`SCALE_OVERFLOW`). Pembulatan scale ke atas mencegah nilai maksimum tersaturasi.

$$m_j = \mathrm{rne}(w_j/s_g)$$

$$q_j = clamp(m_j, -7, 7)$$

$$\hat w_j = s_g q_j$$

`rne` berarti round-half-to-even dalam FP32 ($x=\mathrm{fp32}(w_j)/\mathrm{fp32}(s_g)$; pecahan tepat $\pm 0{,}5$ dibulatkan ke integer genap terdekat); `clamp` membatasi hasil ke interval tertutup $[-7,7]$.

Kode 4-bit bernilai $-8$ tetap representable, tetapi tidak dipancarkan oleh skema simetris ini.

**F11b — Error dan ukuran:**

$$\mathrm{MSE} = \frac{1}{n}\sum_j (w_j-\hat w_j)^2$$

$$\varepsilon_{rel}=\frac{\sqrt{\mathrm{MSE}}}{\sqrt{\tfrac{1}{n}\sum_j w_j^2}}$$

Bila penyebut nol ($\varepsilon_{rel}$ tak terdefinisi — tensor nol pada definisi RMS di atas, tensor konstan pada varian definisi-variansi di `milestones/M6-quantizer.md`): laporkan null dan putuskan via jalur absolut property Q-domain, tanpa epsilon fudge.

$$B_{payload}=\left\lceil\frac{N}{2}\right\rceil+2\left\lceil\frac{N}{G}\right\rceil$$

dengan $bpw_{eff}=4+16/128=\mathbf{4{,}125}$ bit/bobot secara asimtotik; ukuran file = payload + header/alignment. Untuk $N_{total}=14{,}32$ B, prediksi payload = $14{,}32\times4{,}125/8\approx\mathbf{7{,}384\ GB}$ sebelum metadata. Gate G-M6-2 tetap **±10%** terhadap ukuran terukur. Property test Q-domain (oracle, FP32): $|\mathrm{fp32}(w_j)-\hat w^{(32)}_j|\le s_g/2$ untuk semua $j$ dengan $\hat w^{(32)}_j=\mathrm{fp32}(s_g)q_j$. Kernel produksi mengeluarkan BF16 $\hat w^{(\mathrm{bf16})}_j=\mathrm{bf16}(\hat w^{(32)}_j)$; audit kernel memakai bound Kernel-domain $|\mathrm{fp32}(w_j)-\mathrm{fp32}(\hat w^{(\mathrm{bf16})}_j)|\le s_g/2+|\hat w^{(32)}_j|/256$ (segitiga: error kuantisasi + roundoff BF16 $2^{-8}$, ternormalisasi). Dipakai di `milestones/M6-quantizer.md`.

**F11-GGUF — Distorsi Kuantisasi Multi-Format GGUF (M9):**

Format GGUF v3 pada port M9 menggunakan beragam tipe kuantisasi GGML (`Q8_0`, `Q4_K_M`, `Q3_K_M`, `IQ3_S`) yang berbeda secara fundamental dari skema custom 4-bit M6. Kuantisasi 3-bit memiliki batasan teori laju-distorsi ($R(D)$) yang lebih longgar dibanding 4-bit atau 8-bit, sehingga threshold M6 ($\max \varepsilon_{rel} \le 10^{-2}$) **DILARANG** diwariskan begitu saja, dan angka toleransi ukuran file M6 ($\pm 10\%$) **DILARANG KERAS** disalahartikan sebagai threshold galat kuantisasi tensor.

Validasi kuantisasi GGUF M9 dibagi menjadi dua pilar:

1. **Tier 1 — Bit-Exact Decoder Reference**: Memverifikasi bahwa kernel dekuantisasi Mojo engine menghasilkan float yang identik bit-for-bit terhadap referensi C GGML resmi (`ggml-quants.c`):
   $$\Delta_{\max}(\hat{W}_{\text{engine}}, \hat{W}_{\text{ggml\_ref}}) \le 10^{-7} \quad (\text{exact up to FP32 rounding})$$
2. **Tier 2 — Distortion Metrics vs Checkpoint Asli Safetensors BF16** ($W_{\text{orig}}$):
   Dievaluasi pada tiga hierarki independen:
   - **Per-Tensor**: Untuk matriks bobot individual $T$ berukuran $N_T$ elemen:
     $$\text{MSE}(T) = \frac{1}{N_T}\sum_{j=1}^{N_T} (W_{\text{orig}, j}^{(T)} - \hat{W}_j^{(T)})^2, \quad \varepsilon_{rel}(T) = \frac{\sqrt{\text{MSE}(T)}}{\sqrt{\frac{1}{N_T}\sum_{j=1}^{N_T} (W_{\text{orig}, j}^{(T)})^2}}$$
     _(Bila tensor nol/konstan, laporkan `epsilon_rel: null` dan validasi via kesamaan nilai absolut tanpa fudge)._
   - **Per-Block**: Untuk transformer block $\ell \in [0, 39]$ dengan himpunan bobot terkuantisasi $\mathcal{T}_\ell$:
     $$\varepsilon_{rel}^{\text{block}}(\ell) = \frac{\sqrt{\sum_{T \in \mathcal{T}_\ell} \sum_{j=1}^{N_T} (W_{\text{orig}, j}^{(T)} - \hat{W}_j^{(T)})^2}}{\sqrt{\sum_{T \in \mathcal{T}_\ell} \sum_{j=1}^{N_T} (W_{\text{orig}, j}^{(T)})^2}}$$
   - **Global Model-Wide**: Agregasi seluruh bobot model terkuantisasi $\mathcal{T}_{\text{model}}$:
     $$\varepsilon_{rel}^{\text{global}} = \frac{\sqrt{\sum_{T \in \mathcal{T}_{\text{model}}} N_T \cdot \text{MSE}(T)}}{\sqrt{\sum_{T \in \mathcal{T}_{\text{model}}} \sum_{j=1}^{N_T} (W_{\text{orig}, j}^{(T)})^2}}$$

**Ambang Batas Spesifik Format & Peran (Format-Specific & Role-Specific Thresholds):**

| Peran Bobot / Komponen               | Format GGUF         | Referensi Bit-Exact ($\Delta_{\max}$)       | Per-Tensor Max ($\max_T \varepsilon_{rel}$) | Per-Tensor Mean ($\overline{\varepsilon_{rel}}$) | Global Model ($\varepsilon_{rel}^{\text{global}}$) |
| ------------------------------------ | ------------------- | ------------------------------------------- | ------------------------------------------- | ------------------------------------------------ | -------------------------------------------------- |
| **Norms, Biases, RoPE**              | FP32 / BF16         | Safetensors FP32 ($0$)                      | $\le 10^{-7}$                               | $\le 10^{-7}$                                    | $\le 10^{-7}$                                      |
| **Router Gating (`gate.weight`)**    | Q8_0 / BF16         | GGML `dequantize_row_q8_0` ($\le 10^{-7}$)  | $\le 0{,}008$ ($0{,}8\%$)                   | $\le 0{,}005$ ($0{,}5\%$)                        | $\le 0{,}005$ ($0{,}5\%$)                          |
| **Token Mixers (GDN / GatedAttn)**   | Q4_K_M              | GGML `dequantize_row_q4_K` ($\le 10^{-7}$)  | $\le 0{,}045$ ($4{,}5\%$)                   | $\le 0{,}030$ ($3{,}0\%$)                        | $\le 0{,}025$ ($2{,}5\%$)                          |
| **Shared Expert MLP**                | Q4_K_M              | GGML `dequantize_row_q4_K` ($\le 10^{-7}$)  | $\le 0{,}045$ ($4{,}5\%$)                   | $\le 0{,}030$ ($3{,}0\%$)                        | $\le 0{,}025$ ($2{,}5\%$)                          |
| **Routed MoE Experts** (256 experts) | **Q3_K_M**          | GGML `dequantize_row_q3_K` ($\le 10^{-7}$)  | $\le \mathbf{0{,}090}$ ($9{,}0\%$)          | $\le \mathbf{0{,}065}$ ($6{,}5\%$)               | $\le \mathbf{0{,}060}$ ($6{,}0\%$)                 |
| **Routed MoE Experts** (Alternatif)  | **IQ3_S / IQ3_XXS** | GGML `dequantize_row_iq3_s` ($\le 10^{-7}$) | $\le \mathbf{0{,}080}$ ($8{,}0\%$)          | $\le \mathbf{0{,}055}$ ($5{,}5\%$)               | $\le \mathbf{0{,}050}$ ($5{,}0\%$)                 |
| **LM Head Output**                   | Q4_K_M / Q6_K       | GGML dequantizer ($\le 10^{-7}$)            | $\le 0{,}035$ ($3{,}5\%$)                   | $\le 0{,}020$ ($2{,}0\%$)                        | $\le 0{,}020$ ($2{,}0\%$)                          |

Dipakai di `milestones/M9-port.md`.

**F11b-GGUF — Prediksi Ukuran Berkas GGUF Analitik dari Blok Tensor Nyata:**

Prediksi ukuran berkas GGUF port M9 **DILARANG KERAS** menggunakan estimasi kasar parameter global dikalikan bitrate rata-rata ($35\text{B} \times bpw / 8$), dan **DILARANG KERAS** mencampurkan formula bitrate M6 ($4{,}125\text{ bpw} \implies \sim 18{,}0\text{ GB}$) karena M6 hanya untuk trial 14.3B.

Ukuran berkas fisik GGUF dihitung secara deterministik berbasis struktur blok tensor GGML nyata:
$$\text{Size}_{\text{GGUF}}^{\text{expected}} = S_{\text{header}} + S_{\text{metadata\_kv}} + S_{\text{tensor\_info\_dir}} + \sum_{T \in \text{Tensors}} \text{PayloadBytes}(T) + S_{\text{alignment\_padding}}$$

dengan ukuran payload masing-masing tensor $T$ beranggotakan $N_T$ elemen:
$$\text{PayloadBytes}(T) = \left\lceil \frac{N_T}{\text{QK}_K(\text{type}_T)} \right\rceil \times \text{sizeof}(\text{block\_type}_T)$$
serta padding alokasi 32-byte pada setiap awal tensor payload:
$$\text{Offset}_{i+1} = \text{align\_to}(\text{Offset}_i + \text{PayloadBytes}(T_i), 32)$$

Spesifikasi ukuran blok GGML:

- `Q3_K`: $\text{QK}_K = 256$, ukuran blok = $114\text{ byte}$ ($3{,}5625\text{ bpw}$)
- `Q4_K`: $\text{QK}_K = 256$, ukuran blok = $144\text{ byte}$ ($4{,}5000\text{ bpw}$)
- `Q5_K`: $\text{QK}_K = 256$, ukuran blok = $176\text{ byte}$ ($5{,}5000\text{ bpw}$)
- `Q6_K`: $\text{QK}_K = 256$, ukuran blok = $210\text{ byte}$ ($6{,}5625\text{ bpw}$)
- `Q8_0`: $\text{QK}_K = 32$, ukuran blok = $34\text{ byte}$ ($8{,}5000\text{ bpw}$)
- `IQ3_S`: $\text{QK}_K = 256$, ukuran blok = $110\text{ byte}$ ($3{,}4375\text{ bpw}$)
- `IQ3_XXS`: $\text{QK}_K = 256$, ukuran blok = $98\text{ byte}$ ($3{,}0625\text{ bpw}$)
- `BF16`: $\text{QK}_K = 1$, ukuran blok = $2\text{ byte}$ ($16\text{ bpw}$)
- `FP32`: $\text{QK}_K = 1$, ukuran blok = $4\text{ byte}$ ($32\text{ bpw}$)

**Penjelasan Dispersi Ukuran Berkas 13–17 GB**:
Variasi rentang berkas GGUF resmi 13–17 GB berasal dari variasi mix tipe kuantisasi antar-lapisan:

1. `Q3_K_S` (MoE Q3_K, Attn Q3_K, Embed Q3_K): $\approx 13{,}5\text{ GB}$.
2. `Q3_K_M` (Standard: MoE Q3_K, Attn Q4_K, Shared Expert Q4_K, Embed Q4_K): $\approx 15{,}2\text{ GB}$.
3. `Q3_K_L` (Heavy: MoE Q3_K, Attn Q5_K, Shared Expert Q5_K, Embed Q8_0): $\approx 16{,}8\text{ GB}$.
4. `IQ3_XXS` / `IQ3_S` (Importance matrix vector quant): $\approx 13{,}5\text{--}14{,}8\text{ GB}$.

**Kontrak Gate Ukuran Berkas GGUF**:
Gate verifikasi berkas menguji **actual file bytes** (`stat(path).st_size`) terhadap nilai eksak `Size_GGUF_expected` dari header dan daftar blok tensor GGUF aktual:
$$\Delta_{\text{size}} = |\text{stat}(path).st\_size - \text{Size}_{\text{GGUF}}^{\text{expected}}| \equiv 0\text{ byte} \quad (\text{exact byte-match})$$
Penyimpangan $\Delta_{\text{size}} \ne 0$ byte menandakan berkas terpotong (truncated), memiliki trailing garbage, atau tabel offset korup (pelanggaran F15/SEC-1).

Dipakai di `milestones/M9-port.md`.

**F12 — Perplexity & degrade kuantisasi:**

Untuk setiap token target, $\ell_t$ adalah log-probability yang diprediksi engine.

$$NLL=-(\ell_1+\ell_2+\cdots+\ell_N)/N$$

$$PPL=\exp(NLL)$$

$$\Delta PPL=PPL_{quant}-PPL_{bf16}$$

$N$ adalah jumlah token yang benar-benar diprediksi (untuk satu urutan biasa, bukan token pertama). Untuk corpus, agregasikan NLL dan jumlah token dahulu agar PPL token-weighted. $\Delta\mathrm{PPL}$ dapat negatif, sehingga ambang degradasi adalah batas atas seperti G-M6-3, bukan nilai absolut.

Dipakai di M6 dan M9.

## 3.5 Cache & Linear Attention

**F13 — LRU hit rate & bandwidth efektif (M7):**

$$HR_{req}=hits/(hits+misses)$$

$$\rho_B=B_{RAM}/(B_{RAM}+B_{disk})$$

$$BW_{eff}=\left(\rho_B/BW_{RAM}+(1-\rho_B)/BW_{disk}\right)^{-1}$$

$HR_{req}$ adalah diagnostik frekuensi request. F5/F13 memakai $\rho_B$ karena ukuran request/blok dapat tidak seragam; hanya untuk blok sama besar $HR_{req}=\rho_B$. Hit/miss dan byte dari buffered cache maupun LRU aplikasi harus dicatat terpisah agar sumber data tidak terhitung dua kali.

Dipakai di `milestones/M7-odirect-lru.md`.

**F14 — Delta rule / Gated DeltaNet (M8) [R9]:**

$$A_t = I_{d_k} - \beta_t k_t k_t^\top \in \mathbb{R}^{d_k \times d_k}, \quad M_t = \gamma_t A_t, \quad B_t = \beta_t v_t k_t^\top \in \mathbb{R}^{d_v \times d_k}$$

$$S_t = S_{t-1} M_t + B_t, \quad S\in\mathbb{R}^{d_v\times d_k}$$

Di sini $k_t\in\mathbb{R}^{d_k}$ dan $v_t\in\mathbb{R}^{d_v}$; maka kedua suku pembaruan berukuran $d_v\times d_k$. Orientasi sebelumnya $d_k\times d_v$ tidak konsisten dengan perkalian kanan dan outer product $v_tk_t^\top$.

Setiap segmen token kontigu $A$ menginduksi operator affine $\text{ChunkOp}_A(S) = S \mathbf{M}_A + \mathbf{B}_A$. Hukum komposisi dua segmen berurutan $A$ dan $B$ membentuk monoid affine:
$$(\mathbf{M}_{AB}, \mathbf{B}_{AB}) = (\mathbf{M}_A, \mathbf{B}_A) \star (\mathbf{M}_B, \mathbf{B}_B) \coloneqq (\mathbf{M}_A \mathbf{M}_B, \; \mathbf{B}_A \mathbf{M}_B + \mathbf{B}_B)$$
yang mendasari validitas aljabar paralelisasi representasi Woodbury (WY) per chunk.

Versi tanpa gate: $\gamma_t = 1$. Oracle M8 = **loop rekuren naive** (Python fp32); implementasi Mojo = chunked scan (paralel per blok, representasi WY) dan wajib memenuhi ekuivalensi numerik (F10 numerical equivalence, $\Delta_{max} \le 10^{-3}$) terhadap naive. Sifat kunci yang diuji: peak memory runtime $M_{\text{peak}}(s, C) = O(C)$ dan ukuran tensor state konstan terhadap $s$ (G-M8-2, slope $|\Delta \text{PeakVmHWM}/\Delta s| \approx 0$ pada $s \in \{1\text{K}..32\text{K}\}$) — kontras langsung dengan pertumbuhan linier KV-cache F2 ($O(s)$). Dipakai di `milestones/M8-gdn.md`.

## 3.6 Integritas & Parsing Aman

**F15 — Format safetensors & predikat validitas:** file = `[len: u64 LE][header JSON sepanjang N][data buffer]`; $D=8+N$ (nilai `data_base`). `data_offsets = [BEGIN, END)` tiap tensor bersifat **relatif terhadap `data_base`, bukan koordinat absolut file** [R7]; koordinat file = `data_base + BEGIN`, `data_base + END`. Untuk tensor $t$, tulis awal offset sebagai $b_t$ dan akhir sebagai $e_t$. File sah jika dan hanya jika:

**F15a — Batas tensor:**

$$0\le b_t\le e_t$$

$$D+e_t\le filesize$$

$$dtype_t\in\mathcal{D}$$

Nama tensor harus unik dalam header.

**F15b — Buffer tanpa lubang atau overlap:** setelah diurutkan menurut $b$,

$$b_0=0$$

$$b_{i+1}=e_i$$

$$e_{last}=filesize-D$$

(F15b = buffer terindeks penuh tanpa lubang/overlap [R7]; tensor kosong BEGIN==END diizinkan. Membandingkan `data_offsets` mentah dengan `filesize` tanpa tambah `data_base` adalah bug — koordinatnya beda sistem.)

**F15c — Konsistensi dtype, shape, dan range:** ukuran byte harus cocok dengan deklarasi:

$$e_t-b_t=numel(shape_t)\,size(dtype_t)$$

dengan $\mathrm{size} = \{\mathrm{BF16}{:}2, \mathrm{F16}{:}2, \mathrm{F32}{:}4, \mathrm{F64}{:}8\}$ byte/elemen. Pelanggaran → error `LAYOUT_MISMATCH` (bukan `UNKNOWN_DTYPE`: dtype-nya dikenal, aritmetikanya tidak cocok).

**Validasi MERGE (lintas shard — bukan F15):** F15 berhenti di batas satu file. Gabungan semua header yang dipasok wajib memenuhi: nama unik global (tabrakan → `DUPLICATE_TENSOR_NAME`); lalu compare vs `weight_map` per mode (full: semua file rujukan dipasok; subset: hanya nama yang file harapannya termasuk subset yang dinilai).

dengan $\mathcal{D} = \{\mathrm{BF16}, \mathrm{F32}, \mathrm{F16}, \mathrm{F64}\}$ (himpunan eksak proyek — bukan "dll") dan batas keras **`header_len` ≤ 100 MB**, jumlah tensor ≤ 100.000, header JSON diawali `{`, tanpa rekursi parser. Himpunan R7 sendiri non-exhaustive dan kini mencakup BOOL, int/uint, F4/F6/F8\*, C64 (docs.rs `safetensors::tensor::Dtype`); proyek menolak semuanya — int/bool tanpa semantik engine, sub-byte bermasalah alignment [R7], kompleks tak dipakai — dan menolak varian masa depan yang tak dikenal. Checkpoint trial terbukti 100% BF16 (metadata HF `safetensors.parameters`), jadi F32/F16/F64 hanya untuk fixture/forward-compat. Batas 100 MB mengikuti implementasi `safetensors` saat ini; offset tensor divalidasi terhadap ukuran file **setelah dikonversi ke koordinat file**. [R7][R8] Dipakai di `milestones/M0-reader.md`.

**Batas lapisan (normatif):** F15 murni struktural — berhenti di "file ini safetensors yang well-formed". Integritas artefak ($\mathrm{SHA\text{-}256}(\text{file}) = d_{pinned}$, revision pin) adalah **SEC-1/K1, bukan F15**: file valid-struktural dengan hash salah = FAIL SEC-1 (tolak start), bukan FAIL F15 (malformed). Mencampur keduanya mengaburkan debugging (parser vs provenance).

## 3.7 Skala Core (F16) — rasio, tanpa angka device di spec

Simbol device (terdeteksi saat run, **tidak** dipatok di spec): $C_{max}$ = core logis tersedia; $c$ = thread pekerja yang diuji, $c \in \{1,2,4,\dots\} \cap [1, C_{max}]$; rasio $r = c / C_{max}$.

**F16a — Model waktu dan Amdahl:**

$$T_{tok}(c)=T_{IO}+T_{comp}(c)+T_{ovh}(c)$$

$$T_{comp}(c)=T_1/S(c)$$

$$S(c)=1/(1-p+p/c)$$

**F16b — Overhead:**

$$T_{ovh}(c)=\beta(c-1)$$

$$\beta\ge0$$

$T_{IO} = B_{tok}/BW_{eff}$ (F5) diasumsikan independen $c$ pada decode memory-bound — asumsi ini justru yang diuji di G-M7-4. $p$ = fraksi paralel, $T_1$ = komponen compute pada $c=1$, keduanya di-fit dari kurva ukur.

Turunan (dilaporkan, bukan di-gate kaku):

**F16c — Metrik turunan:**

$$S_{tok}(c)=T_{tok}(1)/T_{tok}(c)$$

$$\eta(c)=S_{tok}(c)/c$$

$$M(c\to2c)=[T(c)-T(2c)]/T(c)$$

**F16d — Knee dan rasio operasi:**

Pilih $c^*$ sebagai nilai $c$ terkecil yang diuji dan memenuhi gain marginal di bawah 10%.

$$M(c^*\to2c^*)<10\%$$

$$r^*=c^*/C_{max}$$

Kalibrasi kurva: $e_{T,core} = |T^{pred}(c)-T^{meas}(c)|/T^{meas}(c)$ untuk semua $c$ yang diuji. Monotonisitas: $T(c_2) \le T(c_1)(1+\varepsilon)$ untuk $c_2>c_1$, $\varepsilon=5\%$ (toleransi noise). Non-regresi: $S_{tok}(c) \ge 1$ (tak pernah melambat vs $c=1$ di luar noise). Bukti paper per pilihan: `appendices/D-core-scaling.md` ([R14]–[R20]). Dipakai di `milestones/M5-kv-decode.md` (G-M5-5) dan `milestones/M7-odirect-lru.md` (G-M7-4).

## 3.8 Pola I/O Storage (F17) — "NVMe" saja tidak cukup

Dua pola baca engine berbeda orde kecepatannya, jadi $BW_{SSD}$ tunggal dilarang dipakai untuk keduanya:

**F17a — Trunk sequential:** $BW_{seq}$ diukur dengan blok besar sekitar 4 MB, QD1, dan cache cold.

**F17b — Expert miss:** $BW_{exp}(q)$ diukur dengan blok seukuran expert sekitar 10 MB, offset meloncat, dan prefetch depth $q$.

**F17c — Rasio pola I/O:**

$$R_{io}(q)=BW_{exp}(q)/BW_{seq}$$

**F17d — Degradasi sustained:**

$$D_{sus}=(BW_{burst}-BW_{sustained})/BW_{burst}$$

Gate mensyaratkan $D_{sus}\le30\%$ pada satu pass penuh minimal sebesar $W_{file}$, bukan tembakan satu detik.

Aturan keras:

- **Alignment O_DIRECT:** triple (buffer, offset, panjang) wajib kelipatan block size; misaligned → `EINVAL` atau fallback buffered [R22]; `pread` pendek wajib di-loop (short read bukan error) [R23].
- **Antrean:** NVMe dirancang multi-queue (hingga 64K queue × 64K depth, dipetakan per core) [R24]; $q$ = prefetch depth yang diuji, $q \in \{1,2,4,\dots\}$; laporkan $q^*$ (knee marginal <10\%, analog F16d).
- **Readahead:** hanya berlaku di jalur buffered — advice SEQUENTIAL menggandakan window, RANDOM mematikannya [R25]; di jalur O_DIRECT readahead N/A (bypass page cache) — catat jalur mana yang diukur.
- **Sustained + suhu:** burst (cache SLC) vs sustained bisa jatuh abrupt [R27]; panas berlebih memicu throttle yang mendegradasi performa belasan persen [R26] — suhu dicatat, run pendek dilarang jadi bukti.
- $BW_{eff}$ di F5/F13 wajib memakai angka pola yang sesuai (trunk → $BW_{seq}$, expert-miss → $BW_{exp}$), bukan satu angka brosur. Dipakai di `milestones/M7-odirect-lru.md` (G-M7-5).

## 3.9 Asynchronous Double-Buffering & Latency Overlap (F18)

> Fondasi latency hiding pada inferensi disk-streaming memory-bound [R28][R29][R30].
> Memungkinkan komputasi CPU pada layer $\ell$ tumpang tindih (_overlapped_) dengan pembacaan disk O_DIRECT pada layer $\ell+1$.

**F18a — Model waktu langkah tumpang tindih (Overlapped Step Time):**

Pada eksekusi sekuensial naif (single buffer):
$$T_{step}^{serial} = T_{IO} + T_{comp}(c)$$

Dengan arsitektur asynchronous double-buffering ping-pong (buffer ganda bergiliran):
$$T_{step}^{overlap}(c) = \max\left(T_{IO}, \; T_{comp}(c)\right) + \epsilon_{sync}$$

dengan $T_{IO} = B_{tok}/BW_{eff}$ (F5), $T_{comp}(c) = T_1 / S(c)$ (F16a), dan $\epsilon_{sync} \ge 0$ adalah overhead sinkronisasi thread antrean I/O dan semafor worker pool.

Pada rezim decode disk-streaming memory-bound ($I_{decode} \ll I_{ridge}$, F4 [R17][R18]), $T_{IO} > T_{comp}(c)$ terpenuhi pada titik operasi $c^*$, sehingga waktu komputasi CPU tersembunyi (_hidden_) di balik transfer storage:
$$T_{step}^{overlap}(c^*) \approx T_{IO} + \epsilon_{sync}$$

**F18b — Efisiensi Latency Hiding (Overlap Efficiency):**

$$\mathcal{E}_{overlap}(c) = \frac{(T_{IO} + T_{comp}(c)) - T_{step}^{overlap}(c)}{\min\left(T_{IO}, \; T_{comp}(c)\right)} \times 100\%$$

Target kelayakan gate pada $c^*$: $\mathcal{E}_{overlap}(c^*) \ge 80\%$.

**F18c — Rasio Stabilitas Tail Latency (Tail Ratio):**

$$R_{tail} = \frac{p95}{p50}$$

Sesuai prinsip _The Tail at Scale_ [R19], utilisasi thread tidak boleh memicu saturasi yang merusak distribusi latensi ekor. Target kelayakan stabilitas pada $c^*$: $R_{tail} \le 1{,}35$ dengan toleransi kebisingan $\varepsilon = 5\%$. Dipakai di `milestones/M11-core-scaling.md` (G-M11-2, G-M11-3).
