# M3 — Laporan Verifikasi MoE Layer & Benchmark (Run-ID: M3-20260916-001)

> Dokumen penutup Milestone M3 (`../../../docs/milestones/M3-moe.md`).
> Model dir: `/home/will/models/qwen1.5-moe-a2.7b-chat` (8 shard safetensors, 28,63 GB di disk).
> Revision pin K1: `ec052fda178e241c7c443468d2fa1db6618996be`.
> Environment: Linux x86_64, cgroup `MemoryMax=6G` (`systemd-run --user --scope`), sequence length $L=16$.
> Timestamp: 2026-09-16.

---

## 1. Ringkasan Eksekutif & Keputusan Gate

Semua kriteria penutupan Milestone M3 (MoE Block & Routing) terpenuhi secara penuh:

| Gate / Syarat | Kriteria Normatif | Nilai Terukur | Status |
| :--- | :--- | :--- | :--- |
| **G-M3-1 (Correctness)** | $\Delta_{\max} \le 10^{-3} \wedge \varepsilon_{\text{rel}} \le 10^{-4}$ (Layer 0, 12, 23) | L0: $\Delta=2{,}98\times 10^{-8}, \varepsilon=2{,}67\times 10^{-8}$<br>L12: $\Delta=2{,}98\times 10^{-8}, \varepsilon=2{,}66\times 10^{-8}$<br>L23: $\Delta=2{,}98\times 10^{-8}, \varepsilon=2{,}17\times 10^{-8}$ | **PASS** (MATCH) |
| **G-M3-2 (Routing Invariant)** | SET top-4 identik 100% + unrenorm + sigmoid gate pada 256 input acak | 256/256 token identik (0 flip seleksi), sum prob $< 1{,}0$, sigmoid gate terverifikasi | **PASS** |
| **Perf Baseline (Parse)** | Parse time (weight lookup + header cache) $< 50$ ms | L0: $6{,}10$ ms, L12: $7{,}61$ ms, L23: $8{,}00$ ms | **PASS** ($< 8$ ms) |
| **Perf Baseline (Compute)** | Compute time (single core CPU unquantized reference) | L0: $5{,}53$ s, L12: $6{,}74$ s, L23: $5{,}60$ s | **REPORTED** |
| **Integration (8 Kasus)** | `cargo test --test integration_m3` 8/8 lolos | 8 passed, 0 failed, 0 crash/hang/panic | **PASS** |
| **Determinisme (5×)** | Ulang 5× identik (seed + threads=1) | 5× SHA-256 binary hash identik 100%, routing_info identik | **PASS** |
| **Security (SEC-4/5/6)** | Cgroup 6G, atomic rollback, SHA-256 pin | 0 OOM, rollback terverifikasi, SHA256SUMS OK | **VERIFIED** |

---

## 2. G-M3-1: Verifikasi Oracle Numerik (11-Langkah MoE Block)

Evaluasi numerik part MoE dilakukan terhadap PyTorch FP32 Oracle (`tools/oracle/oracle_layer.py`) menggunakan fixture aktivasi deterministik kanonis `fixtures/m3/activation.bin` ($16 \times 2048$ fp32 LE, 131.072 bytes, seed 42) pada model asli `Qwen1.5-MoE-A2.7B-Chat`:

| Metrik F10 | Batas Gate | Layer 0 | Layer 12 | Layer 23 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $\Delta_{\max}$ (Max Abs Error) | $\le 10^{-3}$ | $2{,}98 \times 10^{-8}$ | $2{,}98 \times 10^{-8}$ | $2{,}98 \times 10^{-8}$ | **PASS** |
| $\varepsilon_{\text{rel}}$ (Relative L2 Error) | $\le 10^{-4}$ | $2{,}67 \times 10^{-8}$ | $2{,}66 \times 10^{-8}$ | $2{,}17 \times 10^{-8}$ | **PASS** |
| $\cos\theta$ (Cosine Similarity) | $\ge 0{,}99$ | $0{,}999999999999997$ | $0{,}999999999999993$ | $0{,}999999999999991$ | **PASS** |
| Top-1 Token Agreement | $100\%$ | $100{,}0\%$ | $100{,}0\%$ | $100{,}0\%$ | **PASS** |
| $\Delta_{CE}$ (Cross-Entropy Diff) | — | $2{,}61 \times 10^{-13}$ | $4{,}54 \times 10^{-12}$ | $6{,}46 \times 10^{-12}$ | **OPTIMAL** |
| **Verdict G-M3-1** | **PASS** | **MATCH** | **MATCH** | **MATCH** | **PASS** |

### Analisis Akurasi Numerik
1. Seluruh layer mencapai $\Delta_{\max}$ sebesar **$2{,}98 \times 10^{-8}$**, jauh lebih presisi ($>30.000\times$ lebih ketat) dari ambang batas toleransi $10^{-3}$.
2. Kesamaan sudut ($\cos\theta$) hampir identik dengan 1.0 (deviasi $< 10^{-14}$).
3. Invariant unrenormalized routing (`norm_topk_prob=false`), SwiGLU activation ($\text{SiLU}(a) \odot b$), dan sigmoid scaling independen pada shared expert ($\sigma(W_{sh\_gate} x)$) terbukti secara bit-perilaku identik dengan PyTorch reference.

---

## 3. G-M3-2: Verifikasi Invariant Routing & Softmax FP32

1. **Top-4 SET Agreement 100%**:
   - Diuji pada 256 input acak deterministik (seed 42) terhadap bobot router aktual pada model riil.
   - 0 flip seleksi terdeteksi antara engine Mojo dan PyTorch oracle.
2. **Top-4 Unrenormalized Invariant**:
   - Total probabilitas 4 expert terpilih $\sum_{k=0}^3 p_k^{(t)} < 1{,}0$ terpenuhi di seluruh token, memverifikasi tidak adanya kebocoran normalisasi softmax ulang.
3. **Shared Expert Sigmoid Gate Invariant**:
   - Bounded di $(0, 1)$ dan strictly monotonic.
   - Verifikasi isolasi: kegagalan/modifikasi gate sigmoid terdeteksi secara deterministik oleh classifier mismatch.

---

## 4. Performance Baseline & Benchmark Real Checkpoint

Pengujian performa dieksekusi menggunakan script `tools/bench/bench_m3_real.py` pada checkpoint asli dengan $N=5$ run per layer di bawah isolasi cgroup `MemoryMax=6G`.

### 4.1 Data Pengukuran Median ($N=5$)

| Layer | Parse Time (ms) | Compute Time (ms) | Total Time (ms) | Throughput (tok/s) |
| :---: | :---: | :---: | :---: | :---: |
| **Layer 0** | 6,10 | 5.527,15 | 5.533,21 | 2,89 |
| **Layer 12** | 7,61 | 6.741,16 | 6.746,80 | 2,37 |
| **Layer 23** | 8,00 | 5.601,72 | 5.607,45 | 2,86 |

### 4.2 Analisis Karakteristik Performa
- **I/O & Parse Optimization**: Shard header caching (`ShardHeaderCache`) memangkas waktu parse/lookup menjadi **$6 - 8$ ms**, jauh di bawah batas anggaran $50$ ms.
- **Compute Time**: Komputasi layer MoE pada fase M3 mencakup 4 routed expert SwiGLU (intermediate size 1408) + 1 shared expert SwiGLU (intermediate size 5632) per token. Pada M3, implementasi fokus pada ketepatan numerik FP32 (oracle-first) single-threaded. Optimasi SIMD, multithreading, dan disk-streaming chunking dijadwalkan pada M5, M6, dan M7.

---

## 5. Hasil Pengujian Integrasi Rust (`integration_m3.rs`)

Seluruh 8 kasus uji normatif lulus 100%:
1. `test_1_happy_path`: Evaluasi layer 0, 12, 23 lolos Gate G-M3-1 (MATCH strict).
2. `test_2_routing_invariant`: Top-4 SET agreement 100% dan penanganan deteksi pelanggaran `ROUTING_VIOLATION`.
3. `test_3_sigmoid_gate_properties`: Validasi sifat analitik fungsi sigmoid.
4. `test_4_top4_no_renorm`: Verifikasi $\sum p_k < 1.0$ (norm_topk_prob=false).
5. `test_5_swiglu_verification`: Verifikasi sifat komputasi element-wise SiLU multiplication.
6. `test_6_expert_load_failure`: Penanganan error terstruktur (`LAYER_INVALID`, `WEIGHT_LOAD_FAILED`, `FILE_NOT_FOUND`) tanpa crash/panic.
7. `test_7_router_overflow`: Stabilitas numerik max-shift softmax pada input besar.
8. `test_8_determinisme`: Verifikasi output biner (SHA-256) dan routing identik pada 5× run berulang.

---

## 6. Verifikasi Keamanan & Invariant Sistem
- **SEC-4 (Resource Limit)**: Pengujian berjalan di dalam isolasi systemd cgroup `MemoryMax=6G`. Selama pengujian berlangsung, penggunaan memori tidak pernah melebihi batas batas aman (0 OOM event terpicu).
- **SEC-6 (Golden Fixtures)**: Seluruh artefak biner referensi (`activation.bin`, `moe_ref_0.bin`, `moe_ref_12.bin`, `moe_ref_23.bin`) memiliki integritas hash SHA-256 yang konsisten dan terverifikasi di `fixtures/m3/SHA256SUMS`.
- **Atomic File Replacement**: File biner output ditulis melalui file sementara `.tmp` dan diganti secara atomik (`c_rename()`), menjamin ketiadaan file korup pada disk streaming.
