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
| F15 | Predikat validitas struktural safetensors (F15a/b/c; SHA = SEC-1, bukan F15) | `05-security.md` (SEC-1..3) |
| F16 | Kurva skala core (rasio, device-agnostic) | M5, M7, benchmark |
| F17 | Pola I/O storage (seq vs expert-size, QD, sustained) | M7, benchmark |

## 3.1 Memori & Streaming

**F1 — Anggaran memori** (sudah ditulis di `01-architecture.md` §2.5): $M_{peak} = W_{res} + M_{KV} + M_{ws} + M_{io} \le M_{gate}$.

**F2 — KV cache** (hanya layer dengan attention penuh):

$$M_{KV}(s) = 2 L_{att} H_{kv} d_h s b_{KV}$$

Dengan $b_{KV}$ = byte per elemen **yang disimpan** di cache (trial: BF16, jadi 2 B; jangan otomatis mengikuti dtype bobot/dequant). Trial (MHA, semua layer): $2 \times 24 \times 16 \times 128 \times 2\,\text{B} = 196.608$ B/token = **0,1875 MiB/token** → @4096 ctx = **0,75 GiB** → konteks praktis trial ≤ 8K di RAM 8 GB. Port M9: berdasarkan layout resmi `10 × (3 × Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE)`, ada **10/40 = 1/4** layer full-attention; GQA = 2 KV heads. Maka F2 memakai $L_{att}=10$ dan $H_{kv}=2$ untuk KV attention. [R4]
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

**Floor bandwidth (gate minimal, bukan estimasi):** $BW_{RAM}$ tipikal memakai 15 GB/s (kelas DDR4 umum), sedangkan **floor minimal $BW_{RAM} \ge 10$ GB/s** (G-M5-6, single-thread Copy read-equiv). Di floor ini target M7-3 tetap lolos dalam forecast serial: $BW_{eff,quant}=\left(0{,}4248/10+0{,}5752/2{,}5\right)^{-1}\approx\mathbf{3{,}67}$ GB/s → $T_{tok}\approx1{,}066/3{,}67+0{,}05\approx\mathbf{0{,}34}$ s/token ≈ **2,94 tok/s** ≥ 2 tok/s. Teoritis DDR4-3200 = 3200 MT/s × 8 B = 25,6 GB/s per kanal; yang di-gate adalah bandwidth *sustainable* terukur ala STREAM [R21], bukan angka teoritis.

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

$$m_j = round(w_j/s_g)$$

$$q_j = clamp(m_j, -7, 7)$$

$$\hat w_j = s_g q_j$$

`round` berarti pembulatan ke integer terdekat; `clamp` membatasi hasil ke interval tertutup $[-7,7]$.

Kode 4-bit bernilai $-8$ tetap representable, tetapi tidak dipancarkan oleh skema simetris ini.

**F11b — Error dan ukuran:**

$$\mathrm{MSE} = \frac{1}{n}\sum_j (w_j-\hat w_j)^2$$

$$\varepsilon_{rel}=\frac{\sqrt{\mathrm{MSE}}}{\sqrt{\tfrac{1}{n}\sum_j w_j^2}}$$

$$B_{payload}=\left\lceil\frac{N}{2}\right\rceil+2\left\lceil\frac{N}{G}\right\rceil$$

dengan $bpw_{eff}=4+16/128=\mathbf{4{,}125}$ bit/bobot secara asimtotik; ukuran file = payload + header/alignment. Untuk $N_{total}=14{,}32$ B, prediksi payload = $14{,}32\times4{,}125/8\approx\mathbf{7{,}384\ GB}$ sebelum metadata. Gate G-M6-2 tetap **±10%** terhadap ukuran terukur. Property test membandingkan nilai yang dipromosikan ke FP32: $|w_j-\hat w_j|\le s_g/2$ untuk semua $j$. Bila output akhirnya dibulatkan lagi ke BF16, ukur error output itu terpisah. Dipakai di `milestones/M6-quantizer.md`.

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

$$A_t=I-\beta_tk_tk_t^\top$$

$$S_t=\gamma_tS_{t-1}A_t+\beta_tv_tk_t^\top$$

$$S\in\mathbb{R}^{d_v\times d_k}$$

Di sini $k_t\in\mathbb{R}^{d_k}$ dan $v_t\in\mathbb{R}^{d_v}$; maka kedua suku pembaruan berukuran $d_v\times d_k$. Orientasi sebelumnya $d_k\times d_v$ tidak konsisten dengan perkalian kanan dan outer product $v_tk_t^\top$.

Versi tanpa gate: $\gamma_t = 1$. Oracle M8 = **loop rekuren naive** (Python fp32); implementasi Mojo = chunked scan (paralel per blok, representasi WY) dan wajib MATCH strict (F10) terhadap naive. Sifat kunci yang diuji: ukuran state konstan terhadap $s$ (G-M8-2) — kontras langsung dengan F2. Dipakai di `milestones/M8-gdn.md`.

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

dengan $\mathcal{D} = \{\mathrm{BF16}, \mathrm{F32}, \mathrm{F16}, \mathrm{F64}\}$ (himpunan eksak proyek — bukan "dll") dan batas keras **`header_len` ≤ 100 MB**, jumlah tensor ≤ 100.000, header JSON diawali `{`, tanpa rekursi parser. Himpunan R7 sendiri non-exhaustive dan kini mencakup BOOL, int/uint, F4/F6/F8*, C64 (docs.rs `safetensors::tensor::Dtype`); proyek menolak semuanya — int/bool tanpa semantik engine, sub-byte bermasalah alignment [R7], kompleks tak dipakai — dan menolak varian masa depan yang tak dikenal. Checkpoint trial terbukti 100% BF16 (metadata HF `safetensors.parameters`), jadi F32/F16/F64 hanya untuk fixture/forward-compat. Batas 100 MB mengikuti implementasi `safetensors` saat ini; offset tensor divalidasi terhadap ukuran file **setelah dikonversi ke koordinat file**. [R7][R8] Dipakai di `milestones/M0-reader.md`.

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
