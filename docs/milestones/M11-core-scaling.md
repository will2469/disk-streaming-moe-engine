# M11 — Performance Scaling: Multi-Core CPU (F16), Disk I/O Streaming (F17/F18), & RAM Budgeting (F1–F3)

> Proyek: `disk-streaming-moe-engine`. Fase: **Performance Scaling & Concurrency**. Index: `../README.md`.
> Implementasi dipecah menjadi waves: `../../scratch/wave/m11/README.md` (W1 worker pool → W5 gates).
> Landasan Ilmiah: Amdahl (1967) [R14], Gustafson (1988) [R15], Hill & Marty (2008) [R16], Roofline (2009) [R17], LLM Inference Unveiled (2024) [R18], Tail at Scale (2013) [R19], TaxBreak (2026) [R20], McCalpin STREAM (1995) [R21], LLM in a flash (2023) [R28], Fiddler (2024) [R29], Hennessy & Patterson (2017) [R30].

| Field           | Nilai                                                                                                      |
| :-------------- | :--------------------------------------------------------------------------------------------------------- |
| **Deliverable** | Engine `dismoen` teroptimasi multi-core, kurva Amdahl F16, async double-buffering F18, & RAM scaling F1-F3 |
| **Komponen**    | Worker Threadpool, Async I/O Double-Buffer, RAM Budget & LRU Allocator, Amdahl & Tail Profiler             |
| **Prasyarat**   | M10 hijau (Konsolidasi DISMOEN, unifikasi Qwen 3.6 SSOT, sanitasi storage 1.5)                             |
| **Next**        | M12 (Fast & Lightweight Chat CLI & OpenAI-Compatible API) $\to$ M13 (Multi-Token Prediction / MTP)         |
| **Gate**        | G-M11-1..G-M11-4                                                                                           |
| **Rumus**       | F1-F3 (RAM & Cache), F4 (Roofline), F5 (Throughput), F16 (Core Scaling), F17 (I/O), F18 (Async Overlap)    |

---

## 1. Landasan Teori Ilmiah (Scientific Peer-Reviewed Foundation)

Sesuai filosofi rekayasa proyek: _setiap klaim performa wajib berpasangan dengan paper ilmiah, rumus matematis, dan gate pengujian yang dapat diulang_. Milestone M11 berakar pada 11 publikasi ilmiah sistem bereputasi tinggi:

### 1.1 Fixed-Workload Scaling vs Scaled Speedup

- **Amdahl (1967) [R14] vs Gustafson (1988) [R15]**:
  - Pada autoregressive decode LLM, ukuran beban komputasi per langkah pembangkitan token adalah **tetap (fixed problem size)**: memproses tepat 1 token baru ($N=1$) dengan bobot aktif streaming yang konstan ($B_{tok} = 1{,}066\text{ GB}$ pada Q3_K).
  - Sesuai pembuktian Gustafson [R15], _scaled speedup_ (speedup linear) hanya berlaku jika ukuran masalah membesar seiring penambahan prosesor. Untuk masalah berukuran tetap, model yang berlaku secara mutlak adalah **Hukum Amdahl klasik [R14]**.
  - **Prinsip**: Kurva percepatan pasti cekung (_concave_) dan memiliki titik jenuh (_knee_). Klaim speedup linear hingga kapasitas prosesor penuh adalah anomali yang ditolak spesifikasi.

### 1.2 Titik Operasi Optimal ($c^*$) dan Alokasi Headroom OS

- **Hill & Marty (2008) — _Amdahl's Law in the Multicore Era_ [R16]**:
  - Di chip multicore modern, sumber daya seperti _memory controller_, _L3 cache_, dan _interconnect bus_ dibagi bersama antar-core.
  - Memaksa utilisasi seluruh core logis yang tersedia ($C_{max}$) menimbulkan saturasi bus memori dan _thread contention_, yang justru menurunkan efisiensi marginal.
  - **Prinsip**: Titik operasi optimal $c^*$ dipilih pada titik lutut kurva (_knee_) di mana kenaikan performa marjinal di bawah 10%. Sistem wajib menyisakan ruang bebas (_headroom_) sebesar $(1 - r^*)$, di mana $r^* = c^* / C_{max}$, untuk menjamin thread I/O kernel Linux dan background daemon tidak tercekik.

### 1.3 Rezim Memory-Bound & Independensi I/O

- **Roofline Model [R17] & LLM Inference Unveiled [R18]**:
  - Intensitas operasional decode disk-streaming:
    $$I_{decode} = \frac{2 \cdot N_{stream}}{B_{tok}} \approx 1{,}87\text{ FLOP/byte} \ll I_{ridge} \approx 10\text{--}20\text{ FLOP/byte}$$
  - Decode berada jauh di sisi **Memory-Bound (I/O Bound)**.
  - **Prinsip**: Kecepatan transfer data dari NVMe SSD ($T_{IO} = B_{tok} / BW_{eff}$) independen terhadap jumlah core komputasi CPU $c$. Menambah core komputasi hanya mempercepat fraksi $T_{comp}(c)$, bukan $T_{IO}$.

### 1.4 Latency Hiding via Asynchronous Double-Buffering

- **LLM in a flash [R28], Fiddler [R29], Hennessy & Patterson [R30]**:
  - Pada eksekusi sekuensial naif (single buffer), CPU menganggur selama I/O berlangsung ($T_{step} = T_{IO} + T_{comp}$).
  - Dengan arsitektur ping-pong asynchronous double-buffering (Buffer A dihitung CPU, Buffer B diisi secara asinkron oleh thread I/O dari NVMe via `O_DIRECT`), waktu langkah bertransformasi menjadi:
    $$T_{step}^{overlap}(c) = \max\left(T_{IO}, \; T_{comp}(c)\right) + \epsilon_{sync}$$
  - Karena pada titik operasi $c^*$ berlaku $T_{IO} > T_{comp}(c)$, maka seluruh komputasi CPU (dekuantisasi SIMD, perkalian matriks MoE, GDN linear recurrence) **tersembunyi 100% di balik waktu streaming storage**.

### 1.5 Pembatasan Variabilitas Latensi Ekor (Tail Latency)

- **The Tail at Scale [R19] & TaxBreak [R20]**:
  - Penambahan thread pekerja membawa penalti sinkronisasi mutex, _cache line bouncing_, dan antrean penjadwalan kernel: $T_{ovh}(c) = \beta(c - 1)$.
  - Utilisasi thread yang mendekati saturasi total menyebabkan lonjakan variabilitas latensi ekor ($p95$ dan $p99$).
  - **Prinsip**: Evaluasi performa multi-core wajib mengukur distribusi persentil ($p50$ dan $p95$), menetapkan batas kebisingan run-to-run $\varepsilon = 5\%$, dan membatasi rasio tail $R_{tail} = p95 / p50 \le 1{,}35$.

### 1.6 Sinergi Tri-Pilar: CPU, Disk I/O Streaming, & Skalabilitas RAM

- **McCalpin (1995) [R21] & Model Hierarki Memori**:
  - Dalam inferensi disk-streaming MoE, performa akhir tidak dapat dimaksimalkan hanya dari salah satu komponen saja. Ketiga pilar saling mengunci secara matematis:
    1. **Pilar CPU**: Menekan $T_{comp}(c)$ via SIMD parallel worker threads ($c^*$).
    2. **Pilar Disk I/O**: Memaksimalkan $BW_{eff}$ via Linux `O_DIRECT`, chunk alignment 4096B, Queue Depth (QD) NVMe optimal, dan _latency hiding_ (F18).
    3. **Pilar RAM**: Menentukan alokasi $M_{RAM}$ (F1). Ketika kapasitas DRAM host bertambah (skalabilitas RAM dari 8 GB $\to$ 16 GB $\to$ 32 GB $\to$ 64 GB), kapasitas LRU expert cache ($C_{exp}$, F2) membesar, melipatgandakan _cache hit rate_ $h(M_{RAM})$ (F3). Kenaikan $h$ langsung memangkas volume byte yang wajib di-stream dari disk:
       $$B_{tok}(M_{RAM}) = B_{trunk} + (1 - h(M_{RAM})) B_{moe}$$
       Penurunan $B_{tok}$ mempercepat $T_{IO} = B_{tok} / BW_{eff}$, menggeser keseimbangan sistem sehingga percepatan CPU multi-core ($c^*$) memberikan dampak throughput yang jauh lebih besar! Sebaliknya, saturasi bandwidth DRAM bus (diukur via STREAM benchmark [R21]) menjadi batas atas penambahan worker thread dekuantisasi.

---

## 2. Model Matematis Resmi (Formula F16 & F18)

### 2.1 Formula F16 — Skala Core Amdahl & Overhead

$$T_{tok}(c) = T_{IO} + T_{comp}(c) + T_{ovh}(c)$$

$$T_{comp}(c) = \frac{T_1}{(1 - p) + \frac{p}{c}}$$

$$T_{ovh}(c) = \beta (c - 1)$$

Di mana:

- $T_{IO} = \frac{B_{tok}}{BW_{eff}}$: Waktu pembacaan bobot dari storage via O_DIRECT.
- $T_1$: Komponen komputasi single-thread ($c = 1$).
- $p \in [0, 1]$: Fraksi beban kerja yang dapat diparalelkan (di-fit dari kurva regresi empiris).
- $\beta \ge 0$: Koefisien penalti sinkronisasi dan thread contention.

### 2.2 Penentuan Knee ($c^*$) dan Rasio Operasi Aman ($r^*$)

Pertambahan performa marjinal penggandaan core ($c \to 2c$):
$$M(c \to 2c) = \frac{T(c) - T(2c)}{T(c)}$$

Titik operasi optimal $c^*$ dipilih sebagai nilai $c$ terkecil yang diuji dan memenuhi:
$$M(c^* \to 2c^*) < 10\%$$
$$r^* = \frac{c^*}{C_{max}}$$

### 2.3 Formula F18 — Latency Overlap & Tail Ratio

Waktu langkah dengan double-buffering asinkron:
$$T_{step}^{overlap}(c) = \max\left(T_{IO}, \; T_{comp}(c)\right) + \epsilon_{sync}$$

Efisiensi latency hiding:
$$\mathcal{E}_{overlap}(c) = \frac{(T_{IO} + T_{comp}(c)) - T_{step}^{overlap}(c)}{\min\left(T_{IO}, \; T_{comp}(c)\right)} \times 100\%$$

Rasio stabilitas tail:
$$R_{tail} = \frac{p95}{p50} \le 1{,}35$$

### 2.4 Model Matematis Holistik Tri-Pillar (F1, F5, F16, F18)

Menggabungkan kapasitas DRAM host ($M_{total}$), throughput streaming NVMe ($BW_{eff}$), dan derajat konkurensi core CPU ($c$):

$$T_{tok}(c, M_{RAM}, BW_{eff}) = \max\left( \frac{B_{trunk} + (1 - h(M_{RAM})) B_{moe}}{BW_{eff}}, \; \frac{T_1}{(1 - p) + \frac{p}{c}} \right) + \beta(c - 1) + \epsilon_{sync}$$

Implikasi arsitektural:

- **RAM Scaling ($M_{RAM} \uparrow$)**: Menaikkan hit rate $h \to$ menurunkan pembilang $B_{tok} \to$ memangkas waktu transfer I/O ($T_{IO}$).
- **Disk I/O Scaling ($BW_{eff} \uparrow$)**: Mengoptimalkan $BW_{eff}$ via Linux `O_DIRECT`, chunk alignment 4096B, dan tuning QD NVMe $\to$ memangkas penyebut $T_{IO}$.
- **CPU Scaling ($c \uparrow$)**: Menurunkan waktu eksekusi $T_{comp}(c)$ hingga mencapai knee Amdahl ($c^*$).
- **Sinergi Overlap**: Pada sistem dengan RAM besar ($h \ge 50\%$), $T_{IO}$ turun drastis mendekati $T_{comp}(c)$, sehingga optimasi CPU multi-core menjadi penentu krusial latensi total end-to-end.

---

## 3. Komponen Arsitektur M11

```text
ARSITEKTUR MULTI-THREAD & ASYNC PIPELINE DISMOEN (M11)
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THREAD KOORDINATOR UTAMA (DISMOEN)                    │
└───────┬──────────────────────────────────┬──────────────────────────┬───────┘
        │                                  │                          │
        ▼ (Asinkron I/O Job)               ▼ (RAM Budget & LRU)       ▼ (Worker Task Dispatch)
┌───────────────────────────────┐  ┌────────────────────────┐ ┌───────────────────────────────┐
│     THREAD DEDIKASI I/O       │  │   RAM CACHE ALLOCATOR  │ │     WORKER THREADPOOL (c*)    │
│  - Linux O_DIRECT pread       │  │  - Static trunk pinned │ │  - SIMD AVX2/AVX-512 dequant  │
│  - Aligned block streaming    │  │  - Ping-pong buffers   │ │  - Parallel MoE Expert MatMul │
│  - NVMe Multi-Queue async     │  │  - Dynamic LRU cache   │ │  - Chunked GDN recurrence     │
└───────────────┬───────────────┘  └────────────────────────┘ └───────────────┬───────────────┘
                │                                                             │
                ▼ (Tukar Ping-Pong Tiap Batas Layer)                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STRUKTUR DATA DOUBLE-BUFFER PING-PONG                    │
│   [ BUFFER A ]: Sedang dihitung Worker Threads pada Layer L                 │
│   [ BUFFER B ]: Sedang di-stream oleh Thread I/O untuk Layer L + 1          │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Worker Threadpool**:
   - Threadpool bertipe statis bebas alokasi dinamis (_zero allocation_) di hot loop inferensi.
   - Pekerjaan dekuantisasi blok tensor GGUF Q3_K / Q4_K dibagi merata antar-worker.
   - Perkalian matriks MoE (8 expert aktif per token) dipartisi secara paralel antar-thread pekerja.
2. **Dedicated I/O Worker Thread**:
   - Berjalan pada core terisolasi (di luar pool komputasi $c^*$) untuk menjamin stream `O_DIRECT` tidak terinterupsi komputasi berat CPU.
   - Mengisi buffer layer berikutnya secara kontinu mendahului eksekusi layer saat ini via Linux `pread` aligned 4096B.
3. **RAM Budgeting & Dynamic LRU Cache Allocator**:
   - Mengatur partisi memori DRAM host via parameter `--ram-budget-gib <M>` atau autodetect.
   - Menjamin alokasi statis aman untuk trunk weights (~1.5 GB), double buffers (60 MB), dan working state context (KV cache + GDN state).
   - Memaksimalkan alokasi sisa DRAM untuk LRU expert cache ($C_{exp}$), memperbesar hit rate $h$ untuk mereduksi beban I/O NVMe.
4. **On-Device Hardware Prober & Subperintah `dismoen tune`**:
   - **Prinsip Zero-Assumption**: Engine tidak mematok konfigurasi statis untuk satu tipe perangkat tertentu, melainkan menginspeksi lingkungan hardware mesin lokal secara langsung (_runtime dynamic probing_):
     - **CPU Topology**: Membaca `sysconf(_SC_NPROCESSORS_ONLN)`, L3 cache, dan SIMD flags. Menghitung $C_{max}$ serta menentukan target $c^*$ dan $r^*$ agar OS kernel dan thread I/O tetap memiliki headroom bebas ($1 - r^*$).
     - **RAM Available**: Membaca `sysconf(_SC_AVPHYS_PAGES)` atau `/proc/meminfo`. Mengalokasikan $M_{RAM}$ optimal tanpa memicu swap atau OOM killer.
     - **Disk I/O Alignment**: Menguji sector alignment (4096B) dan mengukur bandwidth $BW_{eff}$ storage target.
   - **Mode Zero-Config**: Menyediakan flag `--auto` pada `dismoen forward` dan `dismoen decode` yang secara otomatis mengaplikasikan parameter optimal untuk mesin tersebut, serta subperintah `dismoen tune` yang menghasilkan profil lokal `dismoen.hardware.lock`.
5. **Automated Core & Resource Scaling Sweeper**:
   - Tool `tools/bench/bench_core_scaling.py`: Melakukan sweep thread $c \in \{1, 2, 4, \dots\}$, sweep budget RAM, mencatat metrik latensi p50/p95, melakukan curve fitting F16 & F18, serta memvalidasi VmHWM footprint.

---

## 4. Quality Gates (M11)

| Gate        | Kriteria Penilaian                                                                                                                                                                                        |                           Ambang Batas                           | Verifier Tool                               |
| :---------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------: | :------------------------------------------ |
| **G-M11-1** | **Amdahl Curve Fit & Knee Calibration**: Kurva F16 cocok terhadap data empiris, monotonik ($\varepsilon = 5\%$), tidak ada regresi vs single-thread ($S_{tok} \ge 1$), titik $c^*$ dan $r^*$ terkalibrasi | $e_{T,core} \le 20\%$, $M(c^* \to 2c^*) < 10\%$, $S_{tok} \ge 1$ | `tools/bench/bench_core_scaling.py`         |
| **G-M11-2** | **Async Double-Buffering Overlap Efficiency**: Komputasi CPU tersembunyi di balik I/O disk via ping-pong buffer                                                                                           |              $\mathcal{E}_{overlap}(c^*) \ge 80\%$               | `tests/integration/test_m11_w2_async_io.sh` |
| **G-M11-3** | **Tail Latency & Variance Stability**: Distribusi latensi ekor stabil di bawah multi-core, bebas thread thrashing                                                                                         |             $R_{tail} = \frac{p95}{p50} \le 1{,}35$              | `tools/bench/verify_tail_stability.py`      |
| **G-M11-4** | **Bitwise Determinism & Zero Regression**: Output logits pada multi-core $c = c^*$ bit-exact terhadap baseline $c = 1$, seluruh tes regresi M8–M10 100% hijau                                             |          $\Delta_{\max} \equiv 0{,}0$, zero regression           | `tests/integration/test_m11_w5_gates.sh`    |

---

## 5. Rencana Gelombang Kerja (Execution Waves)

- **Gelombang 1 (M11-W1: SIMD Parallel Dequant & MatMul)**:
  - Implementasi worker pool di Mojo / POSIX threads.
  - Paralelisasi dekuantisasi baris Q3_K / Q4_K dan MoE expert GEMM.
- **Gelombang 2 (M11-W2: Async Double-Buffering Storage Pipeline)**:
  - Pembangunan arsitektur ping-pong buffer ter-align 4096 byte.
  - Verifikasi latency hiding dan efisiensi overlap $\mathcal{E}_{overlap} \ge 80\%$ (Gate G-M11-2).
- **Gelombang 3 (M11-W3: Hardware Probing & On-Device Auto-Tuner `dismoen tune`)**:
  - Implementasi dynamic hardware prober (CPU cores, RAM available, direct I/O sector).
  - Eksekusi sweep $c \in \{1, 2, 4, \dots\} \cap [1, C_{max}]$, curve fitting nonlinear F16 ($p, \beta$), penentuan knee $c^*$, rasio aman $r^*$, serta alokasi otomatis $M_{RAM}^*$ (Gate G-M11-1).
- **Gelombang 4 (M11-W4: Tail Latency Profiling & Memory Bound)**:
  - Pengujian $N=30$ run di titik $c^*$, pemantauan variansi latensi ekor $p95 / p50 \le 1{,}35$, dan validasi VmHWM tetap bounded $\le 7{,}5\text{ GiB}$ (Gate G-M11-3).
- **Gelombang 5 (M11-W5: Master Certification Gates & Closure)**:
  - Uji determinisme bit-exact multi-core vs single-core (Gate G-M11-4).
  - Terbitkan laporan scorecard formal di `reports/YYYY-MM-DD/M11-gates-scorecard.md`.

---

## 6. Definisi Selesai (DoD M11)

- [ ] Worker threadpool terintegrasi ke dalam subperintah `dismoen forward` dan `dismoen decode` via parameter `--threads <c>` serta mode otomatis `--auto`.
- [ ] Subperintah `dismoen tune` berfungsi menginspeksi hardware lokal secara dinamis (CPU, RAM, Disk I/O) dan mengkalkulasi parameter optimal host tanpa hardcoding absolut.
- [ ] Pipeline asynchronous double-buffering I/O terbukti menyembunyikan komputasi dengan efisiensi overlap $\mathcal{E}_{overlap} \ge 80\%$ (Gate G-M11-2).
- [ ] Kurva kalibrasi core scaling F16 ter-fit dengan galat $e_{T,core} \le 20\%$ dan titik operasi aman $c^*$ serta rasio $r^*$ ter-commit (Gate G-M11-1).
- [ ] Rasio variabilitas tail latency memenuhi $p95 / p50 \le 1{,}35$ tanpa degradasi akibat saturasi core (Gate G-M11-3).
- [ ] Paritas numerik multi-core terbukti 100% bit-exact terhadap baseline single-threaded ($\Delta_{\max} = 0$) (Gate G-M11-4).
- [ ] Seluruh suite pengujian regresi (`validate-m10`, `validate-m9`, `validate-m8`) dan 13 hook `pre-commit` 100% hijau tanpa supresi (`# noqa`, `#[allow]`).
- [ ] Scorecard formal sertifikasi M11 ter-commit di `reports/YYYY-MM-DD/M11-gates-scorecard.md`.
