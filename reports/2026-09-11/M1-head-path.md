# M1 — Laporan Verifikasi Head Path & Benchmark (Run-ID: M1-20260911-001)

> Dokumen penutup Milestone M1 (`../../../docs/milestones/M1-head-path.md`).
> Model dir: `~/models/qwen1.5-moe-a2.7b-chat` (8 shard safetensors, 28,63 GB di disk).
> Revision pin K1: `ec052fda178e241c7c443468d2fa1db6618996be`.
> Environment: Linux x86_64, cgroup `MemoryMax=6G` (`systemd-run --user --scope`), single-threaded (`threads=1`).
> Timestamp: 2026-09-11.

---

## 1. Ringkasan Eksekutif & Keputusan Gate

Semua kriteria penutupan Milestone M1 terpenuhi secara penuh:

| Gate / Syarat | Kriteria Normatif | Nilai Terukur | Status |
| :--- | :--- | :--- | :--- |
| **G-M1-1 (M1-A Correctness)** | $\Delta_{max} \le 10^{-3} \wedge \varepsilon_{rel} \le 10^{-4} \wedge \mathbb{A} = 100\%$ | $\Delta_{max}=1{,}79\times 10^{-7}$, $\varepsilon_{rel}=8{,}36\times 10^{-8}$, $\mathbb{A}=100{,}0\%$ | **PASS** (MATCH) |
| **G-M1-2 (M1-B Memory Peak)** | $M_{peak} \le 3{,}5$ GiB (VmHWM authoritative) | $2{,}377$ GiB ($2.552.233.984$ B) | **PASS** |
| **G-M1-2 (Buffer Caps)** | `conv_buf` $\le 64$ MiB, `src_buf` $\le 64$ MiB | `conv_buf` $= 32$ MiB, `src_buf` $= 16$ MiB | **PASS** |
| **G-M1-2 ($M_{KV}$ Invariant)** | $M_{KV} = 0$ (tanpa alokasi KV cache) | $M_{KV} = 0$ | **PASS** |
| **M1-C Benchmark** | Real checkpoint report-only, $N=5$ median | Parse: $1{,}92$ s, Compute: $28{,}07$ s, VmHWM: $2{,}377$ GiB | **REPORTED** |
| **Integration (7 Kasus)** | `cargo test --test integration_m1` 7/7 lolos | 7 passed, 0 failed, exit code deterministik | **PASS** |
| **F1 Calibration** | $e_M \le 5\%$ ($M_{peak}^{meas}$ vs $M_{peak}^{pred}$) | Error: $+0{,}495\%$ | **CALIBRATED** |
| **Security (SEC-4/5/6)** | Cgroup 6G, workdir confinement, SHA-256 pin | 0 OOM, escape ditolak, 7/7 hash OK | **VERIFIED** |

---

## 2. G-M1-1 (M1-A): Verifikasi Oracle Numerik

Evaluasi numerik dilakukan terhadap oracle PyTorch FP32 kanonis (`tools/oracle/oracle_head.py`) menggunakan fixture deterministik `fixtures/m1/` (3 prompt $\times$ 16 token, vocab 512, hidden 64, untied head) dengan `threads=1`:

- **$\Delta_{max}$**: $1{,}7881393432617188 \times 10^{-7}$ (batas $\le 10^{-3}$) → **PASS**
- **$\varepsilon_{rel}$**: $8{,}358788461774378 \times 10^{-8}$ (batas $\le 10^{-4}$) → **PASS**
- **$\cos\theta$**: $0{,}9999999999999917$ (kolinieritas identik)
- **Top-1 Agreement ($\mathbb{A}$)**: $100{,}0\%$ ($48/48$ token argmax cocok) (batas $= 100\%$) → **PASS**
- **$\Delta_{CE}$ (Cross-Entropy Delta)**: $5{,}426749969650041 \times 10^{-10}$
- **Verdict**: `MATCH` / `PASS`

---

## 3. M1-C: Benchmark Checkpoint Asli (Report-Only)

Pengujian dilakukan pada model asli `Qwen1.5-MoE-A2.7B-Chat` (8 shard, 28,63 GB) di bawah cgroup isolation `MemoryMax=6G`.

### 3.1 Data Pengukuran ($N=5$ Run)

| Run | Tipe | Parse Time (ms) | Compute Time (ms) | Total Time (ms) | Throughput (tok/s) | VmHWM (GiB) | I/O Read (GiB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Cold | 2.940,37 | 28.568,76 | 31.509,13 | 1,68 | 2,377 | 0,475 |
| 2 | Warm | 2.410,91 | 28.253,99 | 30.664,90 | 1,70 | 2,377 | 0,128 |
| 3 | Warm | 1.789,08 | 27.935,67 | 29.724,74 | 1,72 | 2,377 | 0,000 |
| 4 | Warm | 1.810,92 | 28.074,59 | 29.885,51 | 1,71 | 2,377 | 0,000 |
| 5 | Warm | 1.921,21 | 28.022,43 | 29.943,64 | 1,71 | 2,377 | 0,000 |
| **Median** | — | **1.921,21** | **28.074,59** | **29.943,64** | **1,71** | **2,377** | **0,000** |

### 3.2 Analisis I/O vs Komputasi

1. **Parse & Weight Loading (I/O Dominant)**:
   - Pada cold run (Run 1), kernel membaca 509,8 MB data dari NVMe SSD dalam 2,94 detik (~173 MB/s unbuffered overhead).
   - Pada warm run (Run 3–5), page cache melayani pembacaan shard sehingga waktu parse stabil di ~1,8–1,9 detik (hanya dibatasi oleh dekompresi chunked BF16 $\to$ FP32 di CPU).
2. **Compute Time**:
   - Komputasi stabil di ~28,0 detik untuk 48 token ($151.936 \times 2.048$ matmul scalar per token tanpa AVX-512/AMX microkernel).
   - Sesuai spesifikasi M1, tidak ada latency gate absolut untuk M1-C; baseline ini menjadi landasan ukur bagi optimasi SIMD & threading pada M5.

---

## 4. Kalibrasi Model Memori F1 & Evaluasi Risiko R5

### 4.1 Kalibrasi F1 ($M_{peak}^{meas}$ vs $M_{peak}^{pred}$)

- **Formula Prediksi F1**:
  $$M_{peak}^{pred} = W_{res} + B_{conv} + B_{src}$$
  $$W_{res} = 2 \times (151.936 \times 2.048 \times 4) + (2.048 \times 4) = 2.489.327.616 \text{ byte } (\approx 2{,}31835 \text{ GiB})$$
  $$B_{conv} = 32 \text{ MiB} = 33.554.432 \text{ byte}$$
  $$B_{src} = 16 \text{ MiB} = 16.777.216 \text{ byte}$$
  $$M_{peak}^{pred} = 2.489.327.616 + 33.554.432 + 16.777.216 = 2.539.659.264 \text{ byte } (\approx 2{,}36524 \text{ GiB})$$
- **Pengukuran Aktual VmHWM**:
  $$M_{peak}^{meas} = 2.552.233.984 \text{ byte } (\approx 2{,}37695 \text{ GiB})$$
- **Overhead Runtime Mojo**:
  $$\Delta_{overhead} = 2.552.233.984 - 2.539.659.264 = 12.574.720 \text{ byte } (\approx 12{,}0 \text{ MiB})$$
- **Galat Kalibrasi ($e_M$)**:
  $$e_M = \frac{|M_{peak}^{meas} - M_{peak}^{pred}|}{M_{peak}^{pred}} = \frac{12.574.720}{2.539.659.264} = 0{,}495\% \le 5\% \quad \text{(LOLOS KALIBRASI)}$$

### 4.2 Evaluasi Risiko R5 (BF16 Resident vs On-the-fly Dequant)

- Pada alokasi FP32 resident, $W_{res} = 2{,}318$ GiB mengonsumsi sekitar 29% dari anggaran fisik 8 GB RAM target.
- Sisa ruang RAM (~5,6 GiB) sangat mencukupi untuk tahapan M2 (Attention) dan M3/M4 (Router & MoE streaming).
- Jika pada M5/M7 ditemukan tekanan memori akibat konkurensi atau batching KV-cache, strategi mitigasi R5 terbukti layak:
  - Menyimpan bobot embedding & `lm_head` dalam format asli BF16 (mengurangi footprint resident menjadi $1{,}159$ GiB, hemat $1{,}16$ GiB RAM).
  - Melakukan konversi BF16 $\to$ FP32 secara on-the-fly per chunk token atau memanfaatkan instruksi dot-product FP32-akumulasi BF16 di CPU.

---

## 5. Hasil Pengujian Integrasi (7 Kasus Normatif)

Rangkaian uji integrasi dijalankan via `cargo test --test integration_m1`:

1. **`test_1_happy_path_g_m1_1`**: Output logits biner diverifikasi terhadap oracle PyTorch menggunakan `dismoen-tools compare`. Status: `MATCH`, Verdict: `PASS`, exit 0.
2. **`test_2_token_validation`**: Token ID negatif dan token ID $\ge V$ ditolak secara deterministik dengan exit code 2 dan error JSON `TOKEN_INVALID`.
3. **`test_3_weight_load_failure`**:
   - (a) Propagasi reader error M0: shard hilang memicu `FILE_NOT_FOUND` (exit 2).
   - (b) Semantik M1: shard ada tapi tensor norm tidak lengkap memicu `WEIGHT_LOAD_FAILED` dengan rincian `missing_tensors: ["model.norm.weight"]` (exit 2).
4. **`test_4_oracle_mismatch`**: Mutasi bit eksponen pada kandidat logits dideteksi oleh gate G-M1-1 dengan status `MISMATCH`, verdict `FAIL`, `fail_category: "dtype-layout"`, exit 1.
5. **`test_5_atomic_write_failure`**: Penulisan ke direktori tanpa izin tulis (`0o555`) digagalkan dengan `OUTPUT_WRITE_FAILED` (exit 2) tanpa meninggalkan sisa file biner parsial atau temporary file di disk.
6. **`test_6_memory_boundary_and_caps`**: Verifikasi kepatuhan anggaran: peak VmHWM $\le 3{,}5$ GiB, `conversion_buffer_bytes` $\le 64$ MiB, `source_buffer_bytes` $\le 64$ MiB.
7. **`test_7_config_contract`**: Eksekusi dengan config tanpa `rms_norm_eps` ditolak dengan `CONFIG_ERROR` pada tahap `config` (exit 2).

Hasil: **7 passed, 0 failed** dalam $0{,}04$ detik.

---

## 6. Verifikasi Keamanan & Confinement (SEC-4, SEC-5, SEC-6)

- **SEC-4 (Cgroup MemoryMax=6G)**: Seluruh 5 run benchmark checkpoint asli berhasil dieksekusi di bawah `systemd-run -p MemoryMax=6G` tanpa ada proses yang terkena OOM killer kernel.
- **SEC-5 (Workdir Confinement & Atomic Rename)**:
  - Pelanggaran direktori kerja melalui path traversal `../` maupun symlink escape berhasil ditolak secara deterministik dengan `OUTPUT_WRITE_FAILED`.
  - Berkas luaran ditulis ke berkas temporer di filesystem yang sama, kemudian dipindahkan via `rename` atomik.
- **SEC-6 (Golden Fixture Hash Pinned)**:
  - Berkas `fixtures/m1/SHA256SUMS` memvalidasi seluruh 7 artefak (`logits_ref.bin`, config, tokens, index, dan 3 shards safetensors).
  - Status `sha256sum -c SHA256SUMS`: 7/7 OK.

---

## 7. Kesimpulan

Milestone M1 (Head Path) resmi **SELESAI (DONE)**. Seluruh target correctness, resource bound, determinisme numerik, dan integrasi telah terbukti secara empiris. Engine siap melangkah ke Milestone M2 (Attention Layer).
