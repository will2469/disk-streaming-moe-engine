# 02 — Model Matematika Inti

> Bagian dari `disk-streaming-moe-engine`. Index: `README.md`.
> Setiap rumus punya ID `F#` yang dirujuk oleh gate di `04-quality.md` dan kontrol security di `05-security.md`. Contoh angka memakai config trial; angka port ditandai TBM.

| ID | Rumus | Dipakai di |
|---|---|---|
| F1 | Anggaran memori peak | `01-architecture.md` §2.5, G-M1..M9, SEC-4 |
| F2 | Ukuran KV cache | M5, M9, Lampiran A |
| F3 | Bytes per forward / per token | M5–M7, benchmark |
| F4 | Model roofline (memory- vs compute-bound) | `03-testing.md` §4.4, M9 |
| F5 | Waktu per token + bandwidth efektif | M5–M7, M9 |
| F6 | RMSNorm | M2 |
| F7 | RoPE rotate_half + invariant isometri | M2, property test |
| F8 | Router top-k tanpa renorm + shared sigmoid gate + SwiGLU | M3 (paling kritis) |
| F9 | Diagnostik load-balance expert | analisis routing, koreksi F5 |
| F10 | Metrik ekivalensi numerik + propagasi error | `03-testing.md` §4.3, semua gate MATCH |
| F11 | Kuantisasi simetris 4-bit per-grup | M6 |
| F12 | Perplexity & ΔPPL | M6, M9 |
| F13 | Hit rate LRU + bandwidth efektif | M7 |
| F14 | Delta rule / Gated DeltaNet | M8 |
| F15 | Predikat validitas safetensors + SHA-256 | `05-security.md` (SEC-1..3) |
| F16 | Kurva skala core (rasio, device-agnostic) | M5, M7, benchmark |
| F17 | Pola I/O storage (seq vs expert-size, QD, sustained) | M7, benchmark |

## 3.1 Memori & Streaming

**F1 — Anggaran memori** (sudah ditulis di `01-architecture.md` §2.5): $M_{peak} = W_{res} + M_{KV} + M_{ws} + M_{io} \le M_{gate}$.

**F2 — KV cache** (hanya layer dengan attention penuh):

$$M_{KV}(s) = 2 \cdot L_{att} \cdot H_{kv} \cdot d_h \cdot s \cdot b \tag{F2}$$

Trial (MHA, semua layer): $2 \times 24 \times 16 \times 128 \times 2\,\text{B} = 196.608$ B/token = **0,1875 MiB/token** → @4096 ctx = **0,75 GiB** → konteks praktis trial ≤ 8K di RAM 8 GB. Port M9: berdasarkan layout resmi `10 × (3 × Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE)`, ada **10/40 = 1/4** layer full-attention; GQA = 2 KV heads. Maka F2 memakai $L_{att}=10$ dan $H_{kv}=2$ untuk KV attention. [R4]
Dipakai di: `milestones/M5-kv-decode.md`, `milestones/M9-port.md`.

**F3 — Bytes yang dibaca dari disk:**

$$B_{fwd}(s) \approx W_{file} = 28{,}63\ \text{GB} \quad \text{(prefill streaming: bobot tiap layer dibaca sekali untuk semua } s \text{ token)} \tag{F3a}$$

**F3b — Bytes bobot yang benar-benar disentuh per token decode.** `N_act` resmi bukan lagi dipakai sebagai sinonim untuk bytes yang di-stream. Karena embedding + lm_head sudah resident, F3b menghitung hanya bobot badan transformer yang dibaca ulang saat decode.

$$N_{stream}=L\,[N_{attn}+N_{router}+N_{routed,k}+N_{shared}+N_{norm}]\approx 2{,}0668\times10^9$$
$$B_{tok}^{decode}=b\cdot N_{stream}=2\times2{,}0668\times10^9\approx \mathbf{4{,}134\ GB/token}$$
dengan `GB` = $10^9$ byte. `N_stream` adalah besaran **model-specific streaming set**, bukan angka “activated parameters” generik dari model card. [R1][R2]

Catatan penting: tanpa KV cache (M0–M4), menghasilkan $n$ token baru berarti mengulang prefill → boros $\times n$; itulah motivasi M5.

**F4 — Model roofline:**

$$I = \frac{\text{FLOPs}}{\text{bytes dibaca}}, \qquad P = \min\left(P_{peak},\; I \cdot BW\right) \tag{F4}$$

$$I_{decode} \approx \frac{2 N_{stream}}{B_{tok}} = 1\ \text{FLOP/byte}$$
Karena numerator dan denominator memakai himpunan bobot yang sama, rasio ini **self-canceling** terhadap perubahan definisi $N_{act}$ yang memasukkan embedding/lm_head. Kesimpulan memory-bound tetap bergantung pada posisi ridge point mesin target, sehingga verdict final tetap harus berasal dari benchmark `P_cpu` dan bandwidth nyata.

Implikasi arsitektural: optimasi performa = **kurangi bytes** (F3b, kuantisasi M6) atau **naikkan bandwidth efektif** (F5, F13), bukan optimasi FLOPs. Prefill dengan $s$ besar bergerak ke compute-bound — itu sebabnya konteks sangat panjang mahal di CPU meski KV-nya kecil.

**F5 — Waktu per token & bandwidth efektif:**

$$T_{tok} = \frac{B_{tok}}{BW_{eff}} + T_{comp}, \qquad BW_{eff} = \rho \cdot BW_{RAM} + (1-\rho) \cdot BW_{SSD} \tag{F5}$$

$$\rho \approx \min\left(1, \frac{C_{pc}}{W_{stream}}\right) \quad \text{(asumsi routing seragam; jika bias, koreksi via F9)}$$

Contoh trial (estimasi, bukan hasil ukur): $C_{pc} \approx 3$ GB, $W_{stream} \approx 26$ GB → $\rho=3/26\approx\mathbf{0{,}1154}$ → $BW_{eff}\approx\mathbf{4{,}3846}$ GB/s. Dengan $B_{tok}=4{,}134$ GB, komponen I/O memberi $T_{I/O}\approx\mathbf{0{,}943}$ s/token. Bila sementara dipakai placeholder $T_{comp}=0{,}05$ s, maka $T_{tok}\approx\mathbf{0{,}993}$ s/token ≈ **1,01 tok/s**. Placeholder `T_comp` wajib diganti hasil ukur sebelum dipakai untuk acceptance; angka ini hanya forecast engineering.

**Floor bandwidth (gate minimal, bukan estimasi):** $BW_{RAM}$ tipikal memakai 15 GB/s (kelas DDR4 umum), sedangkan **floor minimal $BW_{RAM} \ge 10$ GB/s** (G-M5-6, single-thread Copy read-equiv). Di floor ini target M7-3 tetap lolos: $BW_{eff,quant}\approx0{,}4248\cdot10+0{,}5752\cdot2{,}5\approx5{,}69$ GB/s → $T_{tok}\approx1{,}066/5{,}69+0{,}05\approx0{,}24$ s/token ≈ 4,2 tok/s ≥ 2 tok/s. Teoritis DDR4-3200 = 3200 MT/s × 8 B = 25,6 GB/s per kanal; yang di-gate adalah bandwidth *sustainable* terukur ala STREAM [R21], bukan angka teoritis.

## 3.2 Komponen Model

**F6 — RMSNorm:**

$$y = \frac{x}{\operatorname{RMS}(x)} \odot \gamma, \qquad \operatorname{RMS}(x) = \sqrt{\tfrac{1}{d}\textstyle\sum_i x_i^2 + \varepsilon}$$

$\varepsilon$ dan dtype (fp32) wajib identik dengan oracle; nilai $\varepsilon$ diambil dari config dan diverifikasi silang ke kode modeling Qwen (TBM). Dipakai di M1/M2.

**F7 — RoPE (rotate_half):**

$$\theta_i = m \cdot \omega_i, \quad \omega_i = \mathrm{base}^{-2i/d_h}, \qquad \begin{pmatrix} q'_i \\ q'_{i+d_h/2} \end{pmatrix} = \begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix} \begin{pmatrix} q_i \\ q_{i+d_h/2} \end{pmatrix}$$

Invariant (property test P-3): $\lVert q' \rVert = \lVert q \rVert$ — rotasi ortogonal. Pelanggaran invariant = implementasi terpeleset ke gaya interleaved. Dipakai di `milestones/M2-attention.md`.

**F8 — Router + shared expert + SwiGLU (bagian paling kritis, M3) [R2][R3]:**

$$p = \operatorname{softmax}(W_r x) \in \mathbb{R}^{60} \quad \text{(dihitung di fp32)} \tag{F8a}$$

$$\mathcal{A} = \operatorname{Top-4}(p) \quad \text{TANPA renormalisasi } (\texttt{norm\_topk\_prob=false} \Rightarrow p_i \text{ dipakai apa adanya}) \tag{F8b}$$

$$y = \sum_{i \in \mathcal{A}} p_i \cdot E_i(x) \; + \; \sigma(g_{sh}) \cdot E_{sh}(x) \tag{F8c}$$

$$E(x) = W_{down}\left(\operatorname{SiLU}(W_{gate} x) \odot (W_{up} x)\right), \qquad \operatorname{SiLU}(z) = z \cdot \sigma(z) \tag{F8d}$$

Dua invariant keras: (a) **SET expert terpilih** ($\mathcal{A}$) harus identik dengan oracle untuk semua input uji — pergeseran probabilitas kecil boleh, flip seleksi = FAIL kategori `router-selection` (`03-testing.md` §4.3); (b) gate shared = **sigmoid** (F8c), bukan softmax. Dipakai di `milestones/M3-moe.md`.

**F9 — Diagnostik load-balance (analisis, bukan training):**

$$f_i = \frac{1}{T}\sum_t \mathbb{1}[i \in \mathcal{A}_t], \qquad P_i = \frac{1}{T}\sum_t p_i^{(t)}, \qquad \mathcal{L}_{lb} = N_e \sum_{i=1}^{N_e} f_i P_i, \qquad CV = \frac{\sigma_f}{\mu_f}$$

Dipakai untuk: (a) mendeteksi bias routing pada corpus golden; (b) mengoreksi asumsi $\rho$ seragam di F5 ketika $CV$ tinggi (expert panas bisa di-pin di page cache / LRU pada M7).

## 3.3 Ekivalensi Numerik (F10) — kontrak semua gate MATCH

$$\Delta_{max} = \max_i |\hat{x}_i - x_i|, \qquad \varepsilon_{rel} = \frac{\lVert \hat{x} - x \rVert_2}{\lVert x \rVert_2}, \qquad \cos\theta = \frac{\langle \hat{x}, x \rangle}{\lVert \hat{x} \rVert\, \lVert x \rVert} \tag{F10a}$$

$$\mathbb{A} = \frac{1}{n}\sum_t \mathbb{1}\left[\arg\max \hat{x}_t = \arg\max x_t\right], \qquad \Delta_{CE} = \left|\mathrm{CE}(\hat{x}) - \mathrm{CE}(x)\right| \tag{F10b}$$

**Propagasi error antar-layer:** jika tiap layer menyumbang error relatif $\le \delta$, bound kasar untuk full forward adalah $\varepsilon_{full} \lesssim L \cdot \delta = 24\delta$ (asumsi amplifikasi kecil). Bound ini hanya penunjuk arah — yang di-gate adalah hasil ukur M4, bukan bound-nya.

**Softmax stabil wajib** (oracle dan engine sama-sama): $\operatorname{softmax}(z) = \dfrac{\exp(z - \max z)}{\sum \exp(z - \max z)}$ — tanpa pengurangan max, fp32 meluap pada logits besar dan verdict jadi tidak bermakna.

Threshold verdict per milestone ada di `03-testing.md` §4.3 dan diulang di tiap file `milestones/M*.md`.

## 3.4 Kuantisasi & Kualitas Output

**F11 — Kuantisasi simetris per-grup** (M6; grup $G$ = 128 bobot, skala fp16):

$$s_g = \frac{\max_{j \in G} |w_j|}{7}, \qquad q_j = \operatorname{clip}\left(\operatorname{round}\left(\frac{w_j}{s_g}\right), -8, 7\right), \qquad \hat{w}_j = s_g \cdot q_j \tag{F11a}$$

$$\mathrm{MSE} = \frac{1}{n}\sum_j (w_j - \hat{w}_j)^2, \qquad \varepsilon_{rel} = \frac{\sqrt{\mathrm{MSE}}}{\sqrt{\tfrac{1}{n}\sum_j w_j^2}}, \qquad \text{bytes(file)} \approx N \cdot \frac{bpw_{eff}}{8} \tag{F11b}$$

dengan $bpw_{eff} = 4 + 16/128 = \mathbf{4{,}125}$ (+ metadata). Untuk $N_{total}=14{,}32$ B, prediksi file = $14{,}32\times4{,}125/8\approx\mathbf{7{,}384\ GB}$. Gate G-M6-2 tetap **±10%** terhadap ukuran terukur. Property test: $|w_j - \hat{w}_j| \le s_g/2$ untuk semua $j$. Dipakai di `milestones/M6-quantizer.md`.

**F12 — Perplexity & degrade kuantisasi:**

$$\mathrm{PPL} = \exp\left(-\frac{1}{N}\sum_{t=1}^{N} \ln p(x_t \mid x_{<t})\right), \qquad \Delta\mathrm{PPL} = \mathrm{PPL}_{quant} - \mathrm{PPL}_{bf16}$$

Dipakai di M6 dan M9.

## 3.5 Cache & Linear Attention

**F13 — LRU hit rate & bandwidth efektif (M7):**

$$HR = \frac{\text{hits}}{\text{hits} + \text{misses}}, \qquad BW_{eff} \approx HR \cdot BW_{RAM} + (1 - HR) \cdot BW_{disk}^{O\_DIRECT}$$

Dipakai di `milestones/M7-odirect-lru.md`.

**F14 — Delta rule / Gated DeltaNet (M8) [R9]:**

$$S_t = \gamma_t \, S_{t-1}\left(I - \beta_t\, k_t k_t^\top\right) + \beta_t\, v_t k_t^\top, \qquad S \in \mathbb{R}^{d_k \times d_v} \;\; \text{(ukuran tetap, tidak tumbuh dengan } s\text{)} \tag{F14}$$

Versi tanpa gate: $\gamma_t = 1$. Oracle M8 = **loop rekuren naive** (Python fp32); implementasi Mojo = chunked scan (paralel per blok, representasi WY) dan wajib MATCH strict (F10) terhadap naive. Sifat kunci yang diuji: ukuran state konstan terhadap $s$ (G-M8-2) — kontras langsung dengan F2. Dipakai di `milestones/M8-gdn.md`.

## 3.6 Integritas & Parsing Aman

**F15 — Format safetensors & predikat validitas:** file = `[len: u64][header JSON][data]`; file sah jika dan hanya jika:

$$\forall t:\quad \text{hdr\_end} \le \text{off}_t \;\wedge\; \text{off}_t + \text{len}_t \le \text{filesize} \;\wedge\; \text{dtype}_t \in \mathcal{D} \;\wedge\; \text{name}_t \text{ unik} \tag{F15}$$

dengan $\mathcal{D}$ = whitelist dtype (BF16, F32, F16, …) dan batas keras **`header_len` ≤ 100 MB**, jumlah tensor ≤ 100.000. Batas 100 MB mengikuti implementasi `safetensors` saat ini; offset tensor juga divalidasi terhadap ukuran file. [R7][R8] Integritas di luar format: $\operatorname{SHA-256}(\text{file}) = d_{pinned}$ (`05-security.md` §6.3-K1). Dipakai di `milestones/M0-reader.md`.

## 3.7 Skala Core (F16) — rasio, tanpa angka device di spec

Simbol device (terdeteksi saat run, **tidak** dipatok di spec): $C_{max}$ = core logis tersedia; $c$ = thread pekerja yang diuji, $c \in \{1,2,4,\dots\} \cap [1, C_{max}]$; rasio $r = c / C_{max}$.

$$T_{tok}(c) = T_{IO} + T_{comp}(c) + T_{ovh}(c), \qquad T_{comp}(c) = \frac{T_1}{S(c)}, \qquad S(c) = \frac{1}{(1-p) + p/c} \tag{F16a}$$

$$T_{ovh}(c) = \beta\,(c-1), \quad \beta \ge 0 \;\; \text{(kontensi/launch, di-fit)} \tag{F16b}$$

$T_{IO} = B_{tok}/BW_{eff}$ (F5) diasumsikan independen $c$ pada decode memory-bound — asumsi ini justru yang diuji di G-M7-4. $p$ = fraksi paralel, $T_1$ = komponen compute pada $c=1$, keduanya di-fit dari kurva ukur.

Turunan (dilaporkan, bukan di-gate kaku):

$$S_{tok}(c) = \frac{T_{tok}(1)}{T_{tok}(c)}, \qquad \eta(c) = \frac{S_{tok}(c)}{c}, \qquad M(c{\to}2c) = \frac{T(c)-T(2c)}{T(c)} \tag{F16c}$$

$$c^* = \min\{c : M(c{\to}2c) < 10\%\}, \qquad r^* = c^*/C_{max} \;\; \text{(titik operasi aman; sisakan } 1-r^* \text{ untuk OS)} \tag{F16d}$$

Kalibrasi kurva: $e_{T,core} = |T^{pred}(c)-T^{meas}(c)|/T^{meas}(c)$ untuk semua $c$ yang diuji. Monotonisitas: $T(c_2) \le T(c_1)(1+\varepsilon)$ untuk $c_2>c_1$, $\varepsilon=5\%$ (toleransi noise). Non-regresi: $S_{tok}(c) \ge 1$ (tak pernah melambat vs $c=1$ di luar noise). Bukti paper per pilihan: `appendices/D-core-scaling.md` ([R14]–[R20]). Dipakai di `milestones/M5-kv-decode.md` (G-M5-5) dan `milestones/M7-odirect-lru.md` (G-M7-4).

## 3.8 Pola I/O Storage (F17) — "NVMe" saja tidak cukup

Dua pola baca engine berbeda orde kecepatannya, jadi $BW_{SSD}$ tunggal dilarang dipakai untuk keduanya:

$$BW_{seq} = \text{sequential blok besar (≈4 MB), QD1, cold} \quad \text{(pola trunk per layer)} \tag{F17a}$$

$$BW_{exp}(q) = \text{blok seukuran expert (≈orde 10 MB) melompat antar offset, QD=}q \quad \text{(pola LRU-miss)} \tag{F17b}$$

$$R_{io}(q) = \frac{BW_{exp}(q)}{BW_{seq}} < 1 \;\; \text{(dilaporkan; mendokumentasikan jurang orde)} \tag{F17c}$$

$$D_{sus} = \frac{BW_{burst} - BW_{sustained}}{BW_{burst}} \le 30\% \quad \text{(satu pass penuh $\ge W_{file}$, bukan tembakan 1 detik)} \tag{F17d}$$

Aturan keras:

- **Alignment O_DIRECT:** triple (buffer, offset, panjang) wajib kelipatan block size; misaligned → `EINVAL` atau fallback buffered [R22]; `pread` pendek wajib di-loop (short read bukan error) [R23].
- **Antrean:** NVMe dirancang multi-queue (hingga 64K queue × 64K depth, dipetakan per core) [R24]; $q$ = prefetch depth yang diuji, $q \in \{1,2,4,\dots\}$; laporkan $q^*$ (knee marginal <10\%, analog F16d).
- **Readahead:** hanya berlaku di jalur buffered — advice SEQUENTIAL menggandakan window, RANDOM mematikannya [R25]; di jalur O_DIRECT readahead N/A (bypass page cache) — catat jalur mana yang diukur.
- **Sustained + suhu:** burst (cache SLC) vs sustained bisa jatuh abrupt [R27]; panas berlebih memicu throttle yang mendegradasi performa belasan persen [R26] — suhu dicatat, run pendek dilarang jadi bukti.
- $BW_{eff}$ di F5/F13 wajib memakai angka pola yang sesuai (trunk → $BW_{seq}$, expert-miss → $BW_{exp}$), bukan satu angka brosur. Dipakai di `milestones/M7-odirect-lru.md` (G-M7-5).
