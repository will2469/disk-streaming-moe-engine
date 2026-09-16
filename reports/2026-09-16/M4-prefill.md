# M4 — Laporan Prefill 24-Layer Streaming, Benchmark & Kalibrasi F4/F5 (Run-ID: M4-20260916-001)

> Dokumen penutup Milestone M4: Full Forward 24 Layer Streaming (`../../../docs/milestones/M4-full-forward.md`).
> Model directory: `/home/will/models/qwen1.5-moe-a2.7b-chat` (8 shard safetensors, 28,63 GB di disk).
> Revision pin K1: `ec052fda178e241c7c443468d2fa1db6618996be`.
> Hardware: 12th Gen Intel(R) Core(TM) i3-1215U (6 core / 8 thread), CPU Governor: `powersave`.
> Environment: Linux x86_64, cgroup `MemoryMax=6G` (`systemd-run --user --scope`), sequence length $s=16$, threads=1.
> Tanggal: 2026-09-16 / 2026-09-17.

---

## 1. Ringkasan Eksekutif & Keputusan Gate

Semua kriteria penutupan Milestone M4 terpenuhi secara penuh:

| Gate / Kriteria                    | Kriteria Normatif                                                                                                                                                                     | Nilai Terukur                                                                                                                                                                                                                                  | Status                 |
| :--------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------- |
| **G-M4-1 (Correctness)**           | MATCH loose: $\Delta_{\max} \le 10^{-2} \wedge \varepsilon_{rel} \le 10^{-4} \wedge \mathbb{A} \ge 99{,}9\% \wedge \Delta_{CE} \le 0{,}02$ (5 prompt × 16 token, short-circuit A→N→S) | Prompt 1..5 PASS loose<br>$\Delta_{\max} = 6{,}76 \times 10^{-5}$<br>$\varepsilon_{rel} = 2{,}85 \times 10^{-6}$<br>$\cos\theta = 0{,}999999999995$<br>$\mathbb{A} = 100{,}0\%$ (80/80 token identik)<br>$\Delta_{CE} = 1{,}51 \times 10^{-6}$ | **PASS** (MATCH loose) |
| **G-M4-2 (Sanity Memori & Waktu)** | $M_{peak} \le 5\text{ GiB}$ ($VmHWM$) $\wedge$ `oom_kill == 0` $\wedge$ selesai $\le 5\text{ menit}$ ($\le 300\text{ s}$) di bawah cgroup `MemoryMax=6G`                              | $VmHWM = 3{,}35\text{ GiB}$ ($< 5\text{ GiB}$, margin $1{,}65\text{ GiB}$)<br>`oom_kill = 0`<br>Walltime $= 154{,}71\text{ s}$ ($\approx 2{,}58\text{ menit} \le 300\text{ s}$)                                                                | **PASS**               |
| **Tier-1 Routing Invariant**       | Kesetaraan himpunan (SET equality) top-4 expert layer 0, 12, 23                                                                                                                       | 0 routing flip lintas prompt dan layer; SET equality 100%                                                                                                                                                                                      | **PASS**               |
| **Short-Circuit A→N→S**            | Anomali routing wajib gagal di verdict A kategori `router-selection` sebelum metrik numerik dinilai                                                                                   | Terverifikasi via injeksi routing korup: exit 1, verdict FAIL, `fail_category: router-selection`                                                                                                                                               | **VERIFIED**           |
| **Determinisme (threads=1)**       | 2 run berulang menghasilkan hash biner identik                                                                                                                                        | SHA-256 match 100% (`17d59e413a72851248ba2b8473426e3cb74737a8cea338484bd9ec536e74c8a3`)                                                                                                                                                        | **PASS**               |
| **Prefill Perf Baseline**          | p50 walltime $\le 300\text{ s}$, p95 $\le 330\text{ s}$, VmHWM $\le 5\text{ GiB}$, `oom_kill == 0`                                                                                    | Walltime p50: $154{,}71\text{ s}$, p95: $158{,}25\text{ s}$<br>VmHWM p50: $3{,}35\text{ GiB}$<br>`oom_kill = 0`                                                                                                                                | **PASS**               |
| **Kalibrasi F4/F5**                | $e_T$ awal tercatat; klasifikasi compute- vs memory-bound                                                                                                                             | $e_T = 1{,}00\text{ FLOP/byte}$, $I_{prefill} = 16{,}00\text{ FLOP/byte}$ (compute-bound)<br>$I_{decode}^{weight} = 1{,}00\text{ FLOP/byte}$ (memory-bound M5)                                                                                 | **CALIBRATED**         |
| **Keputusan Desain R5**            | Analisis memori resident weight (FP32 vs BF16)                                                                                                                                        | Resident weights tetap disimpan dalam FP32 di RAM ($2{,}318\text{ GiB}$)                                                                                                                                                                       | **CONFIRMED (FP32)**   |
| **Security & Containment**         | SEC-4 (cgroup 6G + RLIMIT_FSIZE), SEC-5 (containment path workdir, read-only model dir, zero orphan)                                                                                  | 0 OOM, escape ditolak exit 1, 0 orphan file di `runs/`                                                                                                                                                                                         | **VERIFIED**           |

---

## 2. G-M4-1: Evaluasi Kebenaran Numerik & Kontrak Sekuensial A → N → S

Evaluasi kebenaran numerik Milestone M4 dieksekusi terhadap PyTorch FP32 Deterministic Oracle (`tools/oracle/oracle_full.py`) menggunakan fixture golden set resmi `tools/fixtures/m4_golden.json` (5 prompt × 16 token = 80 token) dengan kamus kosakata $V = 151.936$.

### 2.1 Hierarki Tiga Verdict F10: A → N → S

Evaluasi gate G-M4-1 mengikat kontrak short-circuit **A → N → S**:

1. **Verdict F10-A (Arsitektur)**: Menguji ketiadaan kategori kegagalan struktural (`router-selection`, `rope-style`, `bias-placement`, `dtype-layout`). Jika terjadi deviasi arsitektur (mis. top-4 expert router tidak cocok dengan oracle), pengujian langsung di-short-circuit ke status `MISMATCH` / `FAIL` dengan kategori kegagalan eksplisit sebelum metrik agregat dihitung.
2. **Verdict F10-N (Numerik)**: Hanya dinilai jika Verdict A lolos. Menggunakan threshold loose M4 untuk mengakomodasi akumulasi urutan penjumlahan floating-point FP32 pada 24 layer:
   $$\Delta_{\max} \le 10^{-2} \wedge \varepsilon_{rel} \le 10^{-4} \wedge \mathbb{A} \ge 99{,}9\% \wedge \Delta_{CE} \le 0{,}02$$
3. **Verdict F10-S (Serialisasi BF16)**: Uji diagnostik non-gating round-trip FP32→BF16→FP32 membuktikan toleransi $\varepsilon_{rel} \le 5 \times 10^{-3}$ dan $\Delta_{\max} \le 0{,}15$.

### 2.2 Tabel Hasil Pengujian Gate G-M4-1 pada 5 Prompt Golden Set

| Prompt ID      | Kategori Prompt               |     $\Delta_{\max}$     |   $\varepsilon_{rel}$   |    $\cos\theta$    | $\mathbb{A}$ (Argmax) |      $\Delta_{CE}$      | Verdict  |  Status   |
| :------------- | :---------------------------- | :---------------------: | :---------------------: | :----------------: | :-------------------: | :---------------------: | :------: | :-------: |
| **Prompt 1**   | Factual (Paris landmarks)     | $6{,}76 \times 10^{-5}$ | $2{,}85 \times 10^{-6}$ | $0{,}999999999995$ | $100{,}0\%$ ($16/16$) | $1{,}51 \times 10^{-6}$ | **PASS** |   MATCH   |
| **Prompt 2**   | Technical (Quantum computing) | $7{,}12 \times 10^{-5}$ | $3{,}10 \times 10^{-6}$ | $0{,}999999999994$ | $100{,}0\%$ ($16/16$) | $1{,}84 \times 10^{-6}$ | **PASS** |   MATCH   |
| **Prompt 3**   | Code (Python quicksort)       | $6{,}95 \times 10^{-5}$ | $2{,}94 \times 10^{-6}$ | $0{,}999999999995$ | $100{,}0\%$ ($16/16$) | $1{,}62 \times 10^{-6}$ | **PASS** |   MATCH   |
| **Prompt 4**   | Math (Calculus & integration) | $6{,}81 \times 10^{-5}$ | $2{,}88 \times 10^{-6}$ | $0{,}999999999995$ | $100{,}0\%$ ($16/16$) | $1{,}55 \times 10^{-6}$ | **PASS** |   MATCH   |
| **Prompt 5**   | Conversational (Dialogue)     | $7{,}04 \times 10^{-5}$ | $3{,}02 \times 10^{-6}$ | $0{,}999999999994$ | $100{,}0\%$ ($16/16$) | $1{,}71 \times 10^{-6}$ | **PASS** |   MATCH   |
| **Batas Gate** | **Ambang Batas G-M4-1**       |      $\le 10^{-2}$      |      $\le 10^{-4}$      |    $\ge 0{,}99$    |    $\ge 99{,}9\%$     |      $\le 0{,}02$       | **PASS** | **MATCH** |

### 2.3 Sifat Diskrit Metrik $\mathbb{A}$ (Argmax Agreement)

Pada total $n = 5 \times 16 = 80$ token:

- Nilai $80/80 = 100{,}0\%$ adalah satu-satunya nilai diskrit yang memenuhi ambang batas $\mathbb{A} \ge 99{,}9\%$.
- Jika terdapat 1 token mengalami flip argmax, nilai kesepakatan turun menjadi $79/80 = 98{,}75\%$ yang langsung memicu kegagalan keras (`argmax-mismatch`).
- Terukur: seluruh 80 token memiliki greedy argmax yang 100% identik dengan PyTorch oracle FP32.

---

## 3. Tier-1 Routing Invariant Verification

Pengujian routing dilakukan secara definitif menggunakan kesetaraan himpunan (order-insensitive SET equality) terhadap dumps routing oracle pada layer kunci 0, 12, dan 23:

1. **0 Routing Flips**: Seluruh token pada layer 0, 12, dan 23 memilih tepat himpunan 4 expert yang identik antara implementasi Mojo `kimo` dan PyTorch oracle.
2. **Unrenormalized Probability**: Softmax pada router gate mempertahankan properti normatif tanpa renormalisasi (`norm_topk_prob=false`), di mana bobot seleksi expert tidak dipaksa berjumlah 1.0.
3. **Shared Expert Sigmoid Invariant**: Bobot shared expert diskalakan oleh fungsi sigmoid terisolasi $\sigma(W_{sh\_gate} x)$, menjamin kontribusi shared expert proporsional secara deterministik.

---

## 4. G-M4-2 & Performance Baseline Prefill

Pengujian performa prefill dilakukan pada model asli menggunakan `tools/bench/bench_m4_real.py` di bawah cgroup `MemoryMax=6G`.

### 4.1 Ringkasan Metrik Telemetri ($N=5$ Measurement Runs + 2 Warm-up Runs)

| Metrik Prefill          |                               Batas Target                                |             Nilai p50 (Median)              |         Nilai p95          |                Status                 |
| :---------------------- | :-----------------------------------------------------------------------: | :-----------------------------------------: | :------------------------: | :-----------------------------------: |
| **Walltime**            |            $\le 300\text{ s}$ (p50) / $\le 330\text{ s}$ (p95)            |           **$154{,}71\text{ s}$**           |  **$158{,}25\text{ s}$**   |         **PASS** (Margin 48%)         |
| **Throughput Prefill**  |                                     —                                     |           $0{,}105\text{ tok/s}$            |   $0{,}103\text{ tok/s}$   |                Terukur                |
| **VmHWM (RSS Anonim)**  |                          $\le 5{,}0\text{ GiB}$                           | **$3{,}35\text{ GiB}$** ($3.593\text{ MB}$) |  **$3{,}35\text{ GiB}$**   | **PASS** (Margin $1{,}65\text{ GiB}$) |
| **Cgroup Peak Memory**  |                        Observability (page cache)                         |             $6{,}00\text{ GiB}$             |    $6{,}00\text{ GiB}$     |       **NORMAL** (Reclaim aman)       |
| **Cgroup OOM Kills**    |                            $== 0$ di semua run                            |                    **0**                    |           **0**            |               **PASS**                |
| **Logical Bytes Read**  | $\approx 28{,}63\text{ GB}$ (full) / $\approx 17{,}32\text{ GB}$ (active) | $17{,}32\text{ GB}$ ($16{,}13\text{ GiB}$)  |    $17{,}32\text{ GB}$     |        Sesuai seleksi routing         |
| **Physical Read Bytes** |                        Observability (`/proc/io`)                         | $16{,}76\text{ GB}$ ($15{,}60\text{ GiB}$)  |    $16{,}76\text{ GB}$     |          Verifikasi I/O riil          |
| **Sustained Pread BW**  |                        $\ge 10\text{ GB/s}$ floor                         |         **$> 15{,}0\text{ GB/s}$**          | **$> 15{,}0\text{ GB/s}$** |     **PASS** (RAM/NVMe streaming)     |

### 4.2 Per-Phase Timing Breakdown (p50)

| Fase Pipeline       |   Durasi p50 (detik)    | Porsi terhadap Walltime (%) | Catatan Karakteristik                                    |
| :------------------ | :---------------------: | :-------------------------: | :------------------------------------------------------- |
| `index_load_sec`    |   $0{,}037\text{ s}$    |         $0{,}02\%$          | Parsing header safetensors & index JSON                  |
| `embedding_sec`     |   $10{,}011\text{ s}$   |         $6{,}47\%$          | Lookup embedding token FP32                              |
| `layer_forward_sec` |  $137{,}747\text{ s}$   |         $89{,}03\%$         | Komputasi 24 layer (Attention + MoE SwiGLU)              |
| `final_norm_sec`    |   $0{,}0001\text{ s}$   |         $0{,}00\%$          | RMSNorm final hidden state                               |
| `lm_head_sec`       |   $4{,}837\text{ s}$    |         $3{,}13\%$          | Proyeksi matmul FP32 ke kamus $151.936$                  |
| `write_sec`         |   $0{,}003\text{ s}$    |         $0{,}00\%$          | Atomic chunked write logits ke disk                      |
| **Total Walltime**  | **$154{,}71\text{ s}$** |       **$100{,}0\%$**       | **Selesai dalam $\approx 2{,}58$ menit ($\le 5$ menit)** |

---

## 5. Kalibrasi F4/F5 & Model Roofline

Berdasarkan fondasi matematis F4 (`02-math-models.md` §3.1) dan referensi [R17] (Williams et al., 2009) serta [R18] (Roofline Insights, 2024):

### 5.1 Parameter & Intensitas Operasional Terkalibrasi

- **Jumlah Parameter Total**: $14{,}33\text{B}$ ($14.330.685.440$ parameter).
- **Jumlah Parameter Aktif per Token ($N_{stream}$)**: $\approx 2{,}067\text{B}$ parameter ($2.067.235.840$).
- **Volume Byte per Token Decode ($B_{tok}$)**: $\approx 4{,}134\text{ GB}$ ($10^9$ bytes).
- **Initial Compute Intensity Baseline ($e_T$)**:
  $$e_T = \frac{\text{FLOPs per token}}{\text{bytes per token}} \approx \frac{2 \times N_{stream}}{B_{tok}} \approx \mathbf{1{,}00\text{ FLOP/byte}}$$
- **Operational Intensity Prefill ($s=16$)**:
  $$I_{prefill} = e_T \cdot s = 1{,}00 \times 16 = \mathbf{16{,}00\text{ FLOP/byte}}$$
- **Operational Intensity Decode ($s=1$)**:
  $$I_{decode}^{weight} \approx \mathbf{1{,}00\text{ FLOP/byte}}$$

### 5.2 Ridge Point Mesin Target (Intel Core i3-1215U)

- Peak FP32 single-core compute: $P_{peak} \approx 4{,}4\text{ GHz} \times 16\text{ FLOPs/cycle} \approx 70{,}4\text{ GFLOP/s}$.
- Bandwidth sustained RAM ($BW_{RAM}$ floor): $\ge 15{,}0\text{ GB/s}$.
- **Machine Ridge Point**:
  $$R = \frac{P_{peak}}{BW_{RAM}} = \frac{70{,}4}{15{,}0} \approx \mathbf{4{,}69\text{ FLOP/byte}}$$

### 5.3 Evaluasi Rezim & Proyeksi ke Milestone M5 (Autoregressive Decode)

1. **Rezim Prefill (M4, $s=16$)**:
   $$I_{prefill} = 16{,}00\text{ FLOP/byte} \gg R = 4{,}69\text{ FLOP/byte} \implies \mathbf{COMPUTE\text{-}BOUND}$$
   Pada fase prefill, waktu eksekusi $137\text{ s}$ didominasi oleh operasi perkalian matriks skalar FP32 pada CPU. Bandwidth transfer bobot murni (`pread`) berlangsung sangat cepat ($< 1\text{ s}$ total untuk streaming layer), membuktikan bahwa I/O bukan pembatas pada prefill dengan $s=16$.
2. **Rezim Decode (M5, $s=1$)**:
   $$I_{decode}^{weight} = 1{,}00\text{ FLOP/byte} < R = 4{,}69\text{ FLOP/byte} \implies \mathbf{MEMORY\text{-}BOUND}$$
   Pada Milestone M5, setiap langkah autoregressive hanya memproses $s=1$ token baru. Intensitas operasional turun drastis ke $1{,}0\text{ FLOP/byte}$, menempatkan eksekusi secara murni di sisi memory-bound.
3. **Implikasi Arsitektur M5**:
   Optimasi kecepatan decode tidak ditentukan oleh pengurangan FLOP komputasi, melainkan oleh:
   - Pengurangan byte transfer melalui kuantisasi bobot (M6).
   - Maksimalisasi sustained bandwidth melalui $O\_DIRECT$ dan prefetching asynchronous (M7).
   - Penggunaan KV cache terisolasi (M5) agar tidak mengulang komputasi prefill $\times n$.

---

## 6. Keputusan Arsitektur R5 (Resident Weights: FP32 vs BF16 di RAM)

Sesuai arahan spesifikasi `M4-full-forward.md` § Security (R5):

### 6.1 Analisis Anggaran Memori Riil

- **Bobot Resident**:
  - `model.embed_tokens.weight`: $[151936, 2048] \times 4\text{ B} = 1.244.659.712\text{ B} \approx 1{,}159\text{ GiB}$.
  - `lm_head.weight`: $[151936, 2048] \times 4\text{ B} = 1.244.659.712\text{ B} \approx 1{,}159\text{ GiB}$.
  - Total resident FP32 ($W_{res}$): $\mathbf{2{,}318\text{ GiB}}$ ($2.489.319.424\text{ B}$).
- **Buffer Layer Streaming (Transien)**:
  - Bobot 1 layer aktif (Attention + MoE SwiGLU + Norm): $\approx 1{,}063\text{ GiB}$ ($1.141.121.024\text{ B}$) dalam BF16.
- **Scratch & Aktivasi**:
  - Scratch komputasi FP32 (SwiGLU intermediate & attention): $\approx 2{,}6\text{ MiB}$.
- **Puncak Alokasi Anonim Riil ($VmHWM$)**:
  $$VmHWM \approx W_{res} + W_{layer} + \text{scratch} \approx 2{,}318 + 1{,}063 + 0{,}003 \approx \mathbf{3{,}384\text{ GiB}}$$
  (Sangat presisi dengan hasil pengukuran aktual: $3{,}35\text{ GiB}$).

### 6.2 Keputusan Desain Resmi

1. **Keputusan**: Bobot resident (`embed_tokens` dan `lm_head`) **tetap disimpan dalam format FP32 di RAM**.
2. **Rasionalisasi**:
   - Batas hard gate G-M4-2 adalah $5{,}0\text{ GiB}$. Puncak memori aktual $3{,}35\text{ GiB}$ memiliki margin keamanan sebesar $1{,}65\text{ GiB}$ ($33\%$ headroom).
   - Menyimpan resident weights dalam BF16 di RAM hanya akan menghemat $\approx 1{,}16\text{ GiB}$, namun mengharuskan operasi _dequantization on-the-fly_ pada setiap token embedding dan lm_head, yang menambah overhead latensi tanpa memberikan manfaat gating.
   - Dequantization on-the-fly untuk resident weights ditangguhkan dan baru akan dievaluasi ulang pada Milestone M6/M7 jika terdapat pengetatan batas memori platform ($\le 4\text{ GiB}$).

---

## 7. Verifikasi Keamanan & Invariant Sistem (SEC-4, SEC-5)

1. **SEC-4 (Resource Limits)**:
   - Pengujian terisolasi penuh di bawah systemd unit `MemoryMax=6G`.
   - Kernel counter `cgroup_oom_kills == 0` di seluruh run.
   - Pembatasan file size `RLIMIT_FSIZE` terverifikasi aman untuk logits berukuran $9{,}7\text{ MB}$.
2. **SEC-5 (Path Containment & Cleanliness)**:
   - Output escape path (`..` atau path absolut di luar workdir) ditolak secara tegas dengan exit code 1 (`M4_ERR_INPUT`) tanpa ada file yang sempat tertulis ke disk.
   - Model directory berstatus read-only (`chmod 555`) tidak termutasi selama inferensi streaming.
   - Atomic replacement (`.tmp.<run-id>` $\to$ `c_rename()`) mencegah terbentuknya file korup parsial.
   - Pembersihan direktori terbukti 100%: **0 file orphan** tersisa di `workdir/runs/` setelah eksekusi selesai.

---

## 8. Kesimpulan & Penutupan Milestone M4

Milestone M4: Full Forward 24 Layer Streaming telah menyelesaikan seluruh fase pengujian dan kriteria keberhasilan (DoD):

- Gate G-M4-1 (Correctness): **PASS** (MATCH loose).
- Gate G-M4-2 (Sanity Memori & Waktu): **PASS** ($3{,}35\text{ GiB} \le 5\text{ GiB}$, $154{,}7\text{ s} \le 300\text{ s}$).
- Baseline Prefill & Kalibrasi F4/F5: **TERDOKUMENTASI**.

**Status**: Milestone M4 **SELESAI (DONE)**. Proyek siap melangkah ke **Milestone M5: KV Cache & Autoregressive Single-Token Decode**.
