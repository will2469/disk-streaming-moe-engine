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
  - Memaksa utilisasi seluruh core logis yang tersedia menimbulkan saturasi bus memori dan _thread contention_, yang justru menurunkan efisiensi marginal.
  - **Prinsip**: Alokasi core wajib memperhitungkan dedicated I/O worker ($C_{io} \ge 1$) secara matematis. Batas atas pool komputasi adalah $C_{compute\_max} = \max(1, C_{online} - C_{io})$. Titik operasi optimal $c^*$ dipilih pada titik lutut kurva (_knee_) di mana kenaikan performa marjinal di bawah 10%, dengan rasio aman $r^* = c^* / C_{compute\_max}$, menyisakan ruang bebas $(1 - r^*)$ untuk menjamin thread I/O storage dan background daemon tidak tercekik.

### 1.3 Rezim Memory-Bound & Stabilitas I/O Terkendali ($T_{IO}(c) \approx \text{konstan}$)

- **Roofline Model [R17] & LLM Inference Unveiled [R18]**:
  - Intensitas operasional decode disk-streaming:
    $$I_{decode} = \frac{2 \cdot N_{stream}}{B_{tok}} \approx 1{,}87\text{ FLOP/byte} \ll I_{ridge} \approx 10\text{--}20\text{ FLOP/byte}$$
  - Decode berada jauh di sisi **Memory-Bound (I/O Bound)**.
  - **Prinsip & Batasan Fisik**:
    Dalam model kontrol analitis, transfer storage NVMe dimodelkan mendekati konstan terhadap penambahan core komputasi:
    $$T_{IO}(c) \approx \frac{B_{tok}}{BW_{eff}} \approx \text{konstan}$$
    Namun pada kenyataan sistem operasi Linux, throughput dan latensi NVMe rentan dipengaruhi oleh:
    1. _Interrupt & completion handling_ (latensi MSI-X / antrean `ksoftirqd`).
    2. _CPU contention_ (thread pekerja komputasi mencuri siklus CPU dari thread I/O).
    3. _I/O worker placement_ dan afinitas core (NUMA layout dan isolasi core).
    4. _Block layer & filesystem lock contention_.
    5. _Thermal throttling_ saat multi-core SIMD aktif memanaskan paket CPU.
  - **Konsekuensi Rekayasa**:
    Asumsi $T_{IO}(c) \approx \text{konstan}$ **hanya valid di bawah _controlled configuration_**: yaitu menyisakan _headroom_ OS $(1 - r^*)$, mengisolasi thread pekerja I/O di luar pool komputasi $c^*$, dan **wajib mengukur $BW_{eff}(c)$ secara empiris** pada setiap sweep untuk memverifikasi kestabilan throughput disk (slope $\approx 0$). Menambah core komputasi hanya mempercepat komponen komputasi $T_{comp}(c)$, bukan $T_{IO}$.

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
  - **Prinsip dari paper**: Evaluasi performa multi-core wajib mengukur distribusi persentil ($p50$ dan $p95$) dan menetapkan batas kebisingan run-to-run $\varepsilon = 5\%$. Paper [R19] memotivasi _mengukur_ tail, tetapi **tidak membuktikan angka threshold tertentu**.
  - **Project SLO (keputusan proyek, bukan hukum universal)**: $R_{tail} = p95 / p50 \le 1{,}35$. Angka $1{,}35$ adalah _acceptance criterion_ milik proyek ini; proyek lain dengan SLO berbeda tetap sah selama mengukur metodologi persentil yang sama.

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

$$T_{comp}(c) = T_1 \cdot \left((1 - p) + \frac{p}{c}\right)$$

$$T_{ovh}(c) = \beta (c - 1)$$

$$T_{seq}(c) = T_{IO} + T_{comp}(c) + T_{ovh}(c)$$

Di mana:

- $T_{IO} = \frac{B_{tok}}{BW_{eff}}$: Waktu pembacaan bobot dari storage via O_DIRECT.
- $T_1$: Komponen komputasi single-thread ($c = 1$).
- $p \in [0, 1]$: Fraksi beban kerja yang dapat diparalelkan (di-fit dari kurva regresi empiris).
- $\beta \ge 0$: Koefisien penalti sinkronisasi dan thread contention.

### 2.2 Penentuan Knee ($c^*$), Alokasi Core I/O, dan Rasio Operasi Aman ($r^*$)

Pertambahan performa marjinal penggandaan core komputasi ($c \to 2c$):
$$M(c \to 2c) = \frac{T(c) - T(2c)}{T(c)}$$

**Partisi Anggaran Core Sistem (Core Budget Accounting)**:
Di bawah arsitektur asynchronous double-buffering, thread I/O berjalan pada core terpisah di luar worker pool komputasi:

- $C_{online}$: Total processor logis online yang terdeteksi pada host.
- $C_{io}$: Jumlah core yang didedikasikan untuk pipeline I/O storage ($C_{io} \ge 1$).
- $C_{compute\_max} = \max(1, \; C_{online} - C_{io})$: Batas atas fisik untuk threadpool pekerja komputasi.

Titik operasi optimal $c^*$ dipilih sebagai nilai $c$ terkecil yang diuji dalam rentang $[1, C_{compute\_max}]$ dan memenuhi:
$$M(c^* \to 2c^*) < 10\%$$

Rasio operasi aman dievaluasi terhadap kapasitas komputasi efektif:
$$r^* = \frac{c^*}{C_{compute\_max}} \le 1{,}0$$

> **Tidak ada $c^*$ universal (M11-9).**
> Dari model holistik §2.4, $T_{tok} = T_{tok}(c, M_{RAM}, BW_{eff})$.
> Maka titik knee adalah fungsi bersyarat:
> $$c^* = c^*(M_{RAM}, BW_{eff}, h)$$
> di mana $h = h(M_{RAM})$ adalah cache-hit-rate LRU (F3).
> RAM bertambah $\to$ $h \uparrow$ $\to$ $B_{tok} \downarrow$ $\to$ $T_{IO} \downarrow$
> $\to$ titik ketika compute mulai dominan ikut bergeser.
> Setiap nilai $c^*$ / $r^*$ yang di-commit **wajib diberi label profile**
> $(M_{budget}, BW_{eff}, h)$ tempat ia dikalibrasi; memindahkan $c^*$
> antar mesin / antar tier RAM tanpa re-tune adalah INVALID.

**Invarian Kelayakan Alokasi (Feasibility Invariant)**:
Total alokasi thread sistem dijamin tidak pernah melebihi kapasitas fisik hardware:
$$C_{total} = c^* + C_{io} \le C_{online}$$
Hal ini mencegah kegagalan matematis alokasi tidak layak (misalnya jika $C_{online} = c^*$, lalu ditambahkan $C_{io} = 1$ yang memicu _oversubscription_ dan _thread thrashing_).

### 2.3 Formula F18 — Latency Overlap & Tail Ratio

Waktu langkah dengan double-buffering asinkron:
$$T_{step}^{overlap}(c) = \max\left(T_{IO}, \; T_{comp}(c)\right) + T_{ovh}(c) + \epsilon_{sync}$$

Efisiensi latency hiding:
$$\mathcal{E}_{overlap}(c) = \frac{(T_{IO} + T_{comp}(c)) - T_{step}^{overlap}(c)}{\min\left(T_{IO}, \; T_{comp}(c)\right)} \times 100\%$$

Rasio stabilitas tail (Project SLO, lihat §1.5 — bukan konstanta turunan paper [R19]):
$$R_{tail} = \frac{p95}{p50} \le 1{,}35 \quad \text{(Project SLO)}$$

### 2.4 Model Matematis Holistik Tri-Pillar (F1, F5, F16, F18)

Menggabungkan kapasitas DRAM host ($M_{total}$), throughput streaming NVMe ($BW_{eff}$), dan derajat konkurensi core CPU ($c$):

$$B_{tok}(M_{RAM}) = B_{trunk} + (1 - h(M_{RAM})) B_{moe}$$

$$T_{IO}(M_{RAM}, BW_{eff}) = \frac{B_{tok}(M_{RAM})}{BW_{eff}}$$

$$T_{tok}(c, M_{RAM}, BW_{eff}) = \max\left( \frac{B_{tok}(M_{RAM})}{BW_{eff}}, \; T_1 \cdot \left((1 - p) + \frac{p}{c}\right) \right) + \beta(c - 1) + \epsilon_{sync}$$

Implikasi arsitektural:

- **RAM Scaling ($M_{RAM} \uparrow$)**: Menaikkan hit rate $h \to$ menurunkan pembilang $B_{tok} \to$ memangkas waktu transfer I/O ($T_{IO}$).
- **Disk I/O Scaling ($BW_{eff} \uparrow$)**: Mengoptimalkan $BW_{eff}$ via Linux `O_DIRECT`, chunk alignment 4096B, dan tuning QD NVMe $\to$ memangkas penyebut $T_{IO}$.
- **CPU Scaling ($c \uparrow$)**: Menurunkan waktu eksekusi $T_{comp}(c)$ hingga mencapai knee Amdahl ($c^*$).
- **Sinergi Overlap**: Pada sistem dengan RAM besar ($h \ge 50\%$), $T_{IO}$ turun drastis mendekati $T_{comp}(c)$, sehingga optimasi CPU multi-core menjadi penentu krusial latensi total end-to-end.

> **Konsekuensi M11-9 — satu $c^*$ tidak cukup untuk model tiga dimensi.**
> Karena $T_{tok}(c, M_{RAM}, BW_{eff})$ bergantung pada tiga sumbu
> (core, RAM/cache, bandwidth), `dismoen tune` idealnya menghasilkan
> **profile per kondisi hardware**, bukan satu angka $c^*$ global:
>
> ```text
> hardware profile
>   RAM budget        (M_budget, tier 8/16/32/64 GiB)
>   BW_eff            (terukur pada sweep ini)
>   cache hit state   (h hit-rate LRU terukur)
>   C_compute_max     (C_online - C_io)
>   c*                (berlaku HANYA untuk profile ini)
>   r*                (c* / C_compute_max)
>   chunk size        (S_chunk terpakai)
>   QD                (queue depth / in-flight terpakai)
> ```
>
> Artefak `dismoen.hardware.lock` wajib memuat kedelapan field di atas.
> Gate G-M11-1 memverifikasi pasangan $(c^*, r^*)$ **per profile**;
> klaim "$c^*$ mesin A berlaku di mesin B" ditolak spesifikasi.

### 2.5 Protokol Kalibrasi Dua-Rezim (Pemisahan F16 vs F18)

> **Pencegahan Distorsi I/O Masking:**
> Jika curve fitting F16 dievaluasi pada sistem yang sudah menjalankan async double-buffering ($T_{overlap} = \max(T_{IO}, T_{comp})$), lantai latensi I/O disk ($T_{IO}$) akan memotong kurva saat $T_{comp}(c) < T_{IO}$. Kurva yang tampak mendatar akan disalahartikan oleh regresi nonlinear sebagai saturasi fraksi sekuensial Amdahl ($1-p$), padahal komputasi CPU sebenarnya masih mengalami percepatan (_spurious serial fraction hallucination_).

Untuk menjamin kemurnian parameter ilmiah, kalibrasi sistem dipecah menjadi **dua rezim independen**:

1. **Rezim 1 — Kalibrasi F16 (Compute-Isolated / Preloaded RAM)**:
   - Mengukur murni kurva komputasi $T_{comp}(c)$ tanpa hambatan I/O storage.
   - Melakukan fitting nonlinear F16 untuk mengekstrak parameter murni: $T_1$, $p$, koefisien penalti $\beta$, serta titik knee komputasi $c^*$.
   - **Evaluasi Gate G-M11-1**: Kriteria kecocokan galat $e_{T,core} \le 20\%$ dan $M(c^* \to 2c^*) < 10\%$ dievaluasi murni pada rezim ini.
   - Verifier Tool: `tools/bench/bench_core_scaling.py --mode compute-isolated`.

2. **Rezim 2 — Kalibrasi F18 (End-to-End Async Overlap)**:
   - Mengukur waktu sekuensial naif $T_{seq} = T_{IO} + T_{comp}(c) + T_{ovh}(c)$ terhadap waktu tumpang tindih riil $T_{overlap} = \max(T_{IO}, T_{comp}(c)) + T_{ovh}(c) + \epsilon_{sync}$.
   - Memverifikasi bahwa komputasi CPU tersembunyi di balik I/O streaming dengan efisiensi overlap $\mathcal{E}_{overlap}(c^*) \ge 80\%$.
   - **Evaluasi Gate G-M11-2**: Kriteria latency hiding dievaluasi pada rezim end-to-end ini.
   - Verifier Tool: `tests/integration/test_m11_w2_async_io.sh`.

3. **Sintesis Model Holistik**:
   - Parameter terkalibrasi dari Rezim 1 ($T_1, p, \beta$) dan Rezim 2 ($BW_{eff}, \epsilon_{sync}$) disatukan ke dalam formula §2.4 untuk memproyeksikan throughput pada sembarang kapasitas RAM host ($M_{RAM}$) dan bandwidth NVMe ($BW_{eff}$).

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
│  ENGINE-LEVEL ASYNC I/O WORKER│  │   RAM CACHE ALLOCATOR  │ │     WORKER THREADPOOL (c*)    │
│  - Linux O_DIRECT pread       │  │  - Static trunk pinned │ │  - SIMD AVX2/AVX-512 dequant  │
│  - Aligned block streaming    │  │  - Ping-pong buffers   │ │  - Parallel MoE Expert MatMul │
│  - Multi-chunk / io_uring path│  │  - Dynamic LRU cache   │ │  - Chunked GDN recurrence     │
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
2. **Dedicated Engine-Level Asynchronous I/O Worker (Software Pipelining)**:
   - **Mekanisme Primer (Engine-Level Async Overlap)**: Menggunakan thread I/O dedikasi yang menjalankan `O_DIRECT pread()` blocking secara sekuensial pada buffer layer berikutnya _sementara_ CPU mengerjakan layer saat ini. Di level engine, komputasi dan transfer I/O berjalan asinkron bertumpuk (_software pipelining_).
   - **Klarifikasi Queue Depth (QD)**: Pada satu thread dengan `pread()` tunggal, status antrean di level kernel/device secara fisik adalah $\text{QD} = 1$.
   - **Mekanisme Peningkatan Multi-Queue Hardware ($\text{QD} \ge 4$)**:
     Untuk memanfaatkan antrean fisik hardware NVMe dan mencapai bandwidth puncak $BW_{eff}$:
     a) _Parallel Chunk Worker Pool_: Memecah layer menjadi chunk 4096B terpisah yang di-baca paralel oleh beberapa sub-worker I/O (menghasilkan _effective queue depth_ $\text{QD} \ge 4$ secara portabel).
     b) _Kernel-Native Async (`io_uring`)_: Jalur opsional Linux kernel $\ge 5.1$ untuk mengirimkan multiple SQEs (Submission Queue Entries) langsung ke ring buffer kernel tanpa _syscall blocking_.
3. **RAM Budgeting & Dynamic LRU Cache Allocator**:
   - Mengatur partisi memori DRAM host via parameter `--ram-budget-gib <M>` atau autodetect.
   - Menjamin alokasi statis aman untuk trunk weights (~1.5 GB), double buffers (60 MB), dan working state context (KV cache + GDN state).
   - Memaksimalkan alokasi sisa DRAM untuk LRU expert cache ($C_{exp}$), memperbesar hit rate $h$ untuk mereduksi beban I/O NVMe.
4. **On-Device Hardware Prober & Subperintah `dismoen tune`**:
   - **Prinsip Zero-Assumption & Topologi Presisi**: Engine tidak mematok konfigurasi statis untuk satu tipe perangkat tertentu, melainkan menginspeksi lingkungan hardware mesin lokal secara langsung (_runtime dynamic probing_):
     - **CPU Topology via Linux sysfs**: Membaca antarmuka kernel Linux `/sys/devices/system/cpu/cpu[0-9]*/topology/` (`core_id`, `thread_siblings_list`, `physical_package_id`), cache descriptors `/sys/devices/system/cpu/cpu[0-9]*/cache/index[0-9]*/shared_cpu_list`, dan `/sys/devices/system/node/`. Prober mengklasifikasikan physical cores ($C_{phys}$), SMT/Hyperthreading siblings, shared L2/L3 cache domains, dan NUMA nodes. Panggilan `sysconf(_SC_NPROCESSORS_ONLN)` hanya digunakan sebagai fallback minimal bila sysfs tidak terbaca. Dari topologi ini, prober memesan alokasi dedicated I/O worker ($C_{io} \ge 1$) pada core fisik atau SMT sibling terisolasi, menghitung $C_{compute\_max} = \max(1, C_{online} - C_{io})$, serta memetakan worker threads komputasi pada core fisik mandiri sebelum menjadwalkan ke SMT sibling.
     - **RAM Available**: Membaca `sysconf(_SC_AVPHYS_PAGES)` atau `/proc/meminfo`. Mengalokasikan anggaran RAM aktif $M_{budget}$ secara optimal dengan menyisakan batas aman kernel $M_{OS\_reserve} \ge 0{,}5\text{ GiB}$ tanpa memicu swap atau OOM killer.
     - **Disk I/O Alignment & Bandwidth**: Menguji sector alignment (4096B) dan mengukur bandwidth efektif $BW_{eff}$ storage target.
    - **Mode Zero-Config**: Menyediakan flag `--auto` pada `dismoen forward` dan `dismoen decode` yang secara otomatis mengaplikasikan parameter optimal untuk mesin tersebut, serta subperintah `dismoen tune` yang menghasilkan profil lokal `dismoen.hardware.lock`.
    - **Skema output `dismoen tune` (M11-9)**: `dismoen.hardware.lock` wajib memuat satu profile per kondisi kalibrasi — `M_budget`, `BW_eff` terukur, `h` terukur, `C_compute_max`, `c*`, `r*`, `chunk_size`, `QD`. Satu angka $c^*$ tanpa label $(M_{budget}, BW_{eff}, h)$ adalah artefak INVALID.
5. **Automated Core & Resource Scaling Sweeper**:
   - Tool `tools/bench/bench_core_scaling.py`: Melakukan sweep thread $c \in \{1, 2, 4, \dots\} \cap [1, C_{compute\_max}]$, sweep budget RAM aktif ($M_{budget}$), mencatat metrik latensi p50/p95, melakukan curve fitting F16 & F18, serta memvalidasi rasio kepatuhan memori $\mathcal{R}_{RAM} = \text{VmHWM} / M_{budget} \le 0{,}95$.

### 3.1 Kontrak Formal Struktur Buffer & Chunk-Level Ring Buffer

Untuk mencegah ambiguitas operasional dan _latency bubbles_, arsitektur streaming mematuhi 6 kontrak formal:

| Parameter Kontrak          | Spesifikasi Eksak                                                | Keterangan & Justifikasi Rekayasa                                                                                                                                                                                                                                                                                                                                        |
| :------------------------- | :--------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`buffer_capacity`**      | $64\text{ MiB}$ ($67{,}108{,}864\text{ B}$)                      | Menampung $2 \times S_{layer}^{active}$ ($2 \times 26{,}65\text{ MiB} \approx 53{,}3\text{ MiB}$ dinaikkan ke kelipatan 4096B terdekat untuk alokasi 2 layer ping-pong).                                                                                                                                                                                                 |
| **`chunk_size`**           | $S_{chunk} \approx 3{,}33\text{ MiB}$ ($3{,}493{,}888\text{ B}$) | Granularitas per-expert Q3_K (3 matriks: gate, up, down). Layer dengan 8 expert aktif terdiri dari 8 chunk mandiri.                                                                                                                                                                                                                                                      |
| **`alignment`**            | **4096 Bytes**                                                   | Batas alamat memori virtual dan offset berkas wajib ter-align $4\text{ KiB}$ via `posix_memalign` (syarat keras Linux `O_DIRECT`).                                                                                                                                                                                                                                       |
| **`in_flight_io`**         | $N_{in\_flight} \in [2, 4]$ chunk                                | Jumlah permintaan pembacaan `pread` / `io_uring` outstanding yang aktif bersamaan di antrean storage.                                                                                                                                                                                                                                                                    |
| **`tensor_ownership`**     | State Machine Berputar (_Ring Slots_)                            | Slot ring berputar dengan siklus: $\text{EMPTY} \xrightarrow{\text{I/O Dispatch}} \text{IO\_IN\_FLIGHT} \xrightarrow{\text{Completion}} \text{READY} \xrightarrow{\text{Worker Acquire}} \text{COMPUTING} \xrightarrow{\text{Release}} \text{EMPTY}$. Thread I/O memiliki hak tulis eksklusif saat `IO_IN_FLIGHT`; worker thread CPU memiliki hak baca saat `COMPUTING`. |
| **`completion_condition`** | Sinyal Atomik Per-Chunk (_Chunk-Level Pipelining_)               | Penyelesaian I/O diverifikasi per-chunk (`bytes_transferred == chunk_size`). Worker threads **tidak perlu menunggu seluruh 8 expert selesai di-stream**: komputasi pada Expert 1 langsung dimulai seketika Chunk 1 selesai dibaca, sementara thread I/O sedang mengambil Chunk 2!                                                                                        |

### 3.2 Kontrak Determinisme Reduksi Multi-Core (prasyarat G-M11-4, M11-11)

Penjumlahan floating-point IEEE 754 **tidak asosiatif**
($(a+b)+c \ne a+(b+c)$; preseden yang sama didokumentasikan di M8 §6 butir 4–6
untuk FMA, reassociation, dan reduksi SIMD).
Maka reduksi paralel yang urutannya tidak dispesifikasi — misalnya tiap worker
melakukan `atomic add` float ke akumulator bersama, _work-stealing_ dinamis,
atau partisi chunk yang berubah antar run — **dijamin memecah bit-exactness**
walaupun secara matematis ekuivalen.

Gate bit-exact G-M11-4 ($\Delta_{\max} \equiv 0$) **hanya sah** bila implementasi
worker pool mematuhi keempat invariant berikut:

1. **Partisi fixed**: pembagian baris dekuantisasi / blok GEMM antar worker
   adalah fungsi deterministik dari `(thread_id, c)`, identik setiap run
   untuk $c$ yang sama. Tidak ada _dynamic work-stealing_ pada jalur verdict.
2. **Urutan akumulasi fixed**: urutan penjumlahan partial-dot ke hasil akhir
   terdefinisi statis (indeks worker naik), bukan urutan kedatangan
   (_completion order_).
3. **Reduction tree deterministik**: partial hasil worker digabung via tree
   reduksi berurutan tetap (bukan berdasarkan siapa selesai duluan); reduksi
   horizontal SIMD memakai _pairwise tree_ dengan vector width kanonis
   yang di-pin per arsitektur (mengikuti kontrak M8 §6 butir 5).
4. **Larangan reduksi atomik float tak-berurutan**: `atomic add` / `fetch_add`
   floating-point dengan ordering tak-spesifik **DILARANG** pada jalur
   akumulasi logits/state. Sinkronisasi hanya via join/barrier deterministik,
   lalu reduksi sekuensial oleh thread koordinator. Flag `-ffast-math`,
   `-fassociative-math`, `-freciprocal-math` **DILARANG KERAS** (sejajar M8 §6
   butir 3).

> **Fallback bila kontrak di atas tidak dipenuhi**: verdict bit-exact
> dinyatakan TIDAK BERLAKU untuk build tersebut, dan gate diganti
> kriteria toleransi numerik $\Delta_{\max} \le 10^{-3}$
> (sejajar bound drift FMA M8), bukan $\equiv 0$.
> Verifier `test_m11_w5_gates.sh` wajib mendeteksi mode build
> (deterministic vs nondeterministic) dan memilih threshold yang sesuai
> secara eksplisit di laporan — tidak boleh diam-diam melonggarkan.

---

## 4. Quality Gates (M11)

| Gate        | Kriteria Penilaian                                                                                                                                                                                                                                  |                                               Ambang Batas                                                | Verifier Tool                                               |
| :---------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------- |
| **G-M11-1** | **Amdahl Curve Fit & Knee Calibration (Compute-Isolated)**: Kurva F16 cocok terhadap data empiris pada komputasi terisolasi dari I/O disk, monotonik ($\varepsilon = 5\%$), tidak ada regresi ($S_{tok} \ge 1$), titik $c^*$ dan $r^*$ terkalibrasi **per hardware profile** $(M_{budget}, BW_{eff}, h)$ (§2.2, §2.4) |                     $e_{T,core} \le 20\%$, $M(c^* \to 2c^*) < 10\%$, $S_{tok} \ge 1$                      | `tools/bench/bench_core_scaling.py --mode compute-isolated` |
| **G-M11-2** | **Async Double-Buffering Overlap Efficiency (End-to-End)**: Komputasi CPU tersembunyi di balik I/O disk via ping-pong buffer pada pipeline streaming penuh                                                                                          |                                   $\mathcal{E}_{overlap}(c^*) \ge 80\%$                                   | `tests/integration/test_m11_w2_async_io.sh`                 |
| **G-M11-3** | **Tail Latency (Project SLO) & Dynamic RAM Budget Adherence**: Distribusi latensi ekor stabil di bawah multi-core memenuhi **Project SLO** $p95/p50 \le 1{,}35$ (SLO proyek, bukan klaim paper [R19] — lihat §1.5), dan jejak alokasi VmHWM mematuhi plafon dinamis anggaran memori aktif                                                   | $R_{tail} = \frac{p95}{p50} \le 1{,}35$ (Project SLO), $\mathcal{R}_{RAM} = \frac{\text{VmHWM}}{M_{budget}} \le 0{,}95$ | `tools/bench/verify_tail_stability.py`                      |
| **G-M11-4** | **Determinism & Zero Regression**: Output logits pada multi-core $c = c^*$ bit-exact terhadap baseline $c = 1$ **di bawah kontrak determinisme §3.2**, seluruh tes regresi M8–M10 100% hijau                                                                                       |                               $\Delta_{\max} \equiv 0{,}0$ (mode deterministic); fallback $\Delta_{\max} \le 10^{-3}$ bila kontrak §3.2 tidak dipenuhi, zero regression                               | `tests/integration/test_m11_w5_gates.sh`                    |

---

## 5. Rencana Gelombang Kerja (Execution Waves)

- **Gelombang 1 (M11-W1: SIMD Parallel Dequant & MatMul)**:
  - Implementasi worker pool di Mojo / POSIX threads.
  - Paralelisasi dekuantisasi baris Q3_K / Q4_K dan MoE expert GEMM.
- **Gelombang 2 (M11-W2: Async Double-Buffering Storage Pipeline)**:
  - Pembangunan arsitektur ping-pong buffer ter-align 4096 byte.
  - Verifikasi latency hiding dan efisiensi overlap $\mathcal{E}_{overlap} \ge 80\%$ (Gate G-M11-2).
- **Gelombang 3 (M11-W3: Hardware Probing & On-Device Auto-Tuner `dismoen tune`)**:
  - Implementasi dynamic hardware prober berbasis Linux sysfs topology (`cpu*/topology`, `cpu*/cache`, NUMA, RAM available via `/proc/meminfo`).
  - Eksekusi sweep $c \in \{1, 2, 4, \dots\} \cap [1, C_{compute\_max}]$, curve fitting nonlinear F16 ($p, \beta$), penentuan knee $c^*$, rasio aman $r^*$, serta alokasi dinamis anggaran RAM $M_{budget}$ (Gate G-M11-1).
  - Emit artefak `dismoen.hardware.lock` sebagai **hardware profile** lengkap ($M_{budget}$, $BW_{eff}$, $h$, $C_{compute\_max}$, $c^*$, $r^*$, chunk size, QD) — bukan satu $c^*$ global (M11-9, §2.4).
- **Gelombang 4 (M11-W4: Tail Latency Profiling & Dynamic RAM Budgeting)**:
  - Pengujian $N=30$ run di titik $c^*$, pemantauan variansi latensi ekor terhadap **Project SLO** $p95 / p50 \le 1{,}35$ (SLO proyek, bukan turunan paper [R19] — §1.5), dan validasi kepatuhan memori terhadap anggaran aktif $\mathcal{R}_{RAM} = \text{VmHWM} / M_{budget} \le 0{,}95$ (pada tier budget 8/16/32/64 GiB) tanpa kebocoran memori (Gate G-M11-3).
- **Gelombang 5 (M11-W5: Master Certification Gates & Closure)**:
  - Implementasi kontrak determinisme reduksi §3.2 terlebih dahulu, lalu uji bit-exact multi-core vs single-core (Gate G-M11-4; fallback toleransi $\Delta_{\max} \le 10^{-3}$ bila kontrak tidak dipenuhi).
  - Terbitkan laporan scorecard formal di `reports/YYYY-MM-DD/M11-gates-scorecard.md`.

---

## 6. Definisi Selesai (DoD M11)

- [ ] Worker threadpool terintegrasi ke dalam subperintah `dismoen forward` dan `dismoen decode` via parameter `--threads <c>` serta mode otomatis `--auto`.
- [ ] Subperintah `dismoen tune` berfungsi menginspeksi topologi hardware lokal secara dinamis via Linux sysfs (CPU cores, SMT siblings, L3 domains, RAM available) dan mengkalkulasi parameter optimal host tanpa hardcoding absolut.
- [ ] Pipeline asynchronous double-buffering I/O terbukti menyembunyikan komputasi dengan efisiensi overlap $\mathcal{E}_{overlap} \ge 80\%$ (Gate G-M11-2).
- [ ] Kurva kalibrasi core scaling F16 ter-fit dengan galat $e_{T,core} \le 20\%$ dan titik operasi aman $c^*$ serta rasio $r^*$ ter-commit **per hardware profile** $(M_{budget}, BW_{eff}, h)$ di `dismoen.hardware.lock` (Gate G-M11-1).
- [ ] Rasio variabilitas tail latency memenuhi **Project SLO** $p95 / p50 \le 1{,}35$ tanpa degradasi akibat saturasi core, dan alokasi memori mematuhi plafon dinamis $\mathcal{R}_{RAM} = \text{VmHWM} / M_{budget} \le 0{,}95$ (Gate G-M11-3).
- [ ] Paritas numerik multi-core terbukti 100% bit-exact terhadap baseline single-threaded ($\Delta_{\max} = 0$) **di bawah kontrak determinisme §3.2** (fallback $\Delta_{\max} \le 10^{-3}$ bila kontrak tidak dipenuhi) (Gate G-M11-4).
- [ ] Seluruh suite pengujian regresi (`validate-m10`, `validate-m9`, `validate-m8`) dan 13 hook `pre-commit` 100% hijau tanpa supresi (`# noqa`, `#[allow]`).
- [ ] Scorecard formal sertifikasi M11 ter-commit di `reports/YYYY-MM-DD/M11-gates-scorecard.md`.
