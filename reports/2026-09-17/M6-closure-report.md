# Milestone M6 — Laporan Penutupan & Sertifikasi Gate (M6-W6)

> Dokumen penutup resmi Milestone M6: Quantizer & Compressed Storage (`../../../docs/milestones/M6-quantizer.md`).
> Model directory: `~/models/qwen1.5-moe-a2.7b-chat` (8 shard safetensors, 28,63 GB di disk).
> Run ID: `M6-20260917-001`.
> Tanggal: 2026-09-17.
> Status Milestone: **CLOSED — 100% GREEN (ALL GATES PASSED)**.

---

## 1. Ringkasan Eksekutif & Gate Scorecard (G-M6-1 .. G-M6-3 + G-M6-K)

Seluruh empat gate normatif Milestone M6 telah dievaluasi, diverifikasi secara matematis, dan dinyatakan **PASS**:

| Gate ID    | Definisi Kriteria                     | Batas Toleransi / Syarat                                                                                                                                  | Nilai Terukur                                                                                                  |  Status  |
| :--------- | :------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :------: |
| **G-M6-1** | Relatif error kuantisasi per tensor   | $\max \varepsilon_{rel} \le 10^{-2}$ per tensor bervariansi;<br>jalur absolut Q-domain untuk tensor variansi-nol                                          | $\max \varepsilon_{rel} = 0{,}0070648 \le 10^{-2}$<br>Variansi-nol: `epsilon_rel: null`, $100\%$ lolos absolut | **PASS** |
| **G-M6-2** | Ukuran file logis vs prediksi F11b    | Deviasi $\le 10\%$ vs prediksi $N_q \cdot 4{,}125 / 8 + \text{meta}$ ($\approx 7{,}385\text{ GB}$ logis);<br>Rasio $G=64$ vs $G=128 \approx 1{,}03\times$ | Deviasi ukuran: $2{,}53\% \le 10\%$<br>Rasio $G_{64}/G_{128} = 1{,}0295\times$ ($\approx 1{,}0303\times$)      | **PASS** |
| **G-M6-3** | Kualitas bahasa global (PPL & Argmax) | $\Delta\mathrm{PPL} \le +0{,}5 \wedge \mathbb{A} \ge 95\%$ agregat global atas $N_{pred,total} = 25{,}500$ ($100 \times 255$)                             | $\Delta\mathrm{PPL} = +0{,}2620 \le +0{,}5$<br>$\mathbb{A} = 96{,}82\% \ge 95\%$                               | **PASS** |
| **G-M6-K** | Konformansi SIMD Dequant vs Oracle    | $100\%$ bit-identical vs oracle PyTorch (131.072 elemen seed-42) $\wedge$ penolakan identik nibble reserved `0b1000`                                      | Bit-identical: $100{,}00\%$ ($0$ mismatch)<br>Penolakan `0b1000`: Identik (`M6_ERR_DEQUANT`)                   | **PASS** |

---

## 2. Matriks Integration Tests (IT-M6-1 .. IT-M6-16)

Suite integrasi menyeluruh diotomatisasi pada [`tests/integration/test_m6_quantize.sh`](tests/integration/test_m6_quantize.sh) dan dijalankan via task `pixi run test-m6` (Run-ID `M6-20260917-001`):

| Test ID      | Skenario Pengujian                                                                    | Perilaku yang Diharapkan                                                       | Hasil Observasi                                                    | Verdict  |
| :----------- | :------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------- | :----------------------------------------------------------------- | :------: |
| **IT-M6-1**  | Happy path quantize BF16 $\to$ 4-bit                                                  | Exit 0, output biner valid, F11b lolos, $\max\varepsilon_{rel} \le 10^{-2}$    | Exit 0, JSON output valid, $\max\varepsilon_{rel} = 0{,}007065$    | **PASS** |
| **IT-M6-2**  | Input directory hilang/tidak ada                                                      | Exit 1, error `M6_ERR_INPUT`, stage `input`, RFC 8259 valid                    | Exit 1, JSON error `M6_ERR_INPUT` terverifikasi                    | **PASS** |
| **IT-M6-3**  | Group-size tidak valid ($G=100 \notin \{32,64,128,256\}$)                             | Exit 1, error `M6_ERR_INPUT`, stage `input`                                    | Exit 1, JSON error `M6_ERR_INPUT` ditolak sebelum IO               | **PASS** |
| **IT-M6-4**  | Tensor bobot mengandung nilai NaN                                                     | Exit 2, error `M6_ERR_QUANT`, stage `quantization`                             | Exit 2, scale NaN ditangkap dan proses dihentikan                  | **PASS** |
| **IT-M6-5**  | Dequantization menjumpai reserved nibble `0b1000`                                     | Exit 2, error `M6_ERR_DEQUANT`, stage `dequant` via `--check`                  | Exit 2, reserved nibble `0x8` ditolak instan                       | **PASS** |
| **IT-M6-6**  | Output file terpotong (truncated / invalid framing)                                   | Exit 4, error `M6_ERR_VALIDATION`, stage `validation`                          | Exit 4, framing mismatch terdeteksi oleh validator                 | **PASS** |
| **IT-M6-7**  | PPL measurement & reproducibility lock pins                                           | Exit 0, $\Delta\mathrm{PPL} \le +0{,}5$, $\mathbb{A} \ge 95\%$, pins valid     | Exit 0, $\Delta\mathrm{PPL} = +0{,}2620$, $\mathbb{A} = 96{,}82\%$ | **PASS** |
| **IT-M6-8**  | Determinisme byte-identical lintas 2 run identik                                      | SHA-256 output biner identik secara bitwise                                    | SHA-256 identik (`cca2bd19...`) pada run 1 & 2                     | **PASS** |
| **IT-M6-9**  | Custom group-size 64 execution & size check                                           | Exit 0, rasio ukuran $G_{64}/G_{128} \approx 1{,}03\times$, deviasi $\le 10\%$ | Exit 0, rasio $1{,}0295\times$, deviasi $2{,}53\% \le 10\%$        | **PASS** |
| **IT-M6-10** | Property Q-domain $\|w - \hat{w}^{(32)}\| \le s_g/2$                                  | $100\%$ bobot memenuhi bound FP32 round-half-to-even                           | $100{,}00\%$ lolos (0 violations pada 131.072 elemen)              | **PASS** |
| **IT-M6-11** | Tail group violation ($N \% G \neq 0$)                                                | Exit 1, error `M6_ERR_INPUT`, Opsi A fail-closed                               | Exit 1, tensor berdimensi ganjil ditolak tanpa kompromi            | **PASS** |
| **IT-M6-12** | Determinisme tie-break golden vectors $x = \pm k + 0{,}5$                             | Nilai quant menghasilkan half-to-even integer identik                          | Identik $100\%$ antara oracle PyTorch & Mojo                       | **PASS** |
| **IT-M6-13** | Audit bound Kernel-domain $\|w - \hat{w}_{bf16}\| \le s_g/2 + \|\hat{w}^{(32)}\|/256$ | $100\%$ bobot terdequantisasi memenuhi bound BF16                              | $100{,}00\%$ lolos via TestSuite (`test_dequant.mojo`)             | **PASS** |
| **IT-M6-14** | Jalur absolut tensor variansi-nol ($w_i = c$)                                         | `epsilon_rel: null`, lolos via jalur absolut Q-domain                          | Lolos tanpa pembagian nol dan tanpa epsilon fudge                  | **PASS** |
| **IT-M6-15** | G-M6-K SIMD dequant vs Oracle PyTorch (seed-42)                                       | $100\%$ bit-identical dequantized FP32 values                                  | $100{,}00\%$ bit-identical (131.072 elemen cocok)                  | **PASS** |
| **IT-M6-16** | Konformansi berkas biner file-level Oracle vs SIMD                                    | Dekomposisi payload biner menghasilkan array identik                           | $100\%$ bit-identical pada seluruh payload model                   | **PASS** |

---

## 3. Verifikasi Keamanan (SEC-4, SEC-6, Read-Only, & Atomic Rollback)

1. **SEC-4 (Resource Limits, Parser Caps, & Boundary Hardening)**:
   - **Parser Hardening**: Metadata JSON per-tensor dibatasi oleh batas keras (caps): `max_tensors = 65536`, `max_ndim = 8`, `max_name_len = 512`, dan `meta_len <= 65536`.
   - **Aritmetika Overflow-Safe**: Penghitungan `scales_bytes = num_groups * 2` dan `weights_bytes = ceil(N / 2)` diuji terhadap overflow integer 64-bit sebelum buffer dialokasikan.
   - **Non-Overlapping Payloads**: Offset biner divalidasi strictly monotonically increasing dan tidak pernah tumpang-tindih (non-overlapping) di dalam rentang berkas.
   - **Rejection of Outlier/Wild Inputs**: Validasi fail-closed langsung menghentikan proses jika menemukan $G \notin \{32, 64, 128, 256\}$, skala NaN/INF, atau nibble reserved `0b1000`.

2. **SEC-6 (Cryptographic Manifest & Golden Quant Separation)**:
   - Manifest input model BF16 dan manifest model kuantisasi 4-bit dipisahkan secara struktural.
   - Pin identitas model input, corpus token, dan baseline tersimpan permanen di `tools/fixtures/ppl_golden_pins.json` dengan hash SHA-256 yang divalidasi pada setiap eksekusi runner.

3. **Read-Only Model Directory & Workdir Isolation**:
   - Model directory sumber diperlakukan strictly read-only.
   - Tidak ada operasi modifikasi, pembuatan berkas sementara, atau penulisan apapun di dalam direktori model sumber.

4. **Atomic Rollback & Zero Orphan Files**:
   - Berkas biner target kuantisasi ditulis ke berkas sementara dengan ekstensi `.tmp` yang unik.
   - Penggantian ke nama berkas permanen dilakukan secara atomik melalui syscall `renameat`/POSIX `rename`.
   - Jika terjadi interupsi atau kegagalan pada sembarang stage (input, quant, dequant, validation), handler pembersihan fail-closed menjamin $0$ berkas yatim piatu (orphan files) tersisa di workspace.

---

## 4. Karakteristik Kompresi & Proyeksi Performa

### 4.1. Efisiensi Bit per Bobot ($bpw_{eff}$) & Rasio Kompresi

Dihitung berdasarkan formulasi F11b:
$$bpw_{eff} = 4 + \frac{16}{G}$$

- Untuk $G = 128$ (default):
  $$bpw_{eff} = 4 + \frac{16}{128} = 4{,}125\text{ bit per weight}$$
  $$\text{Rasio Kompresi} = \frac{16}{4{,}125} \approx 3{,}8788\times \approx 3{,}88\times$$
- Untuk $G = 64$ (alternatif presisi):
  $$bpw_{eff} = 4 + \frac{16}{64} = 4{,}250\text{ bit per weight}$$
  $$\text{Rasio Kompresi} = \frac{16}{4{,}250} \approx 3{,}7647\times$$
- Rasio ukuran biner $G_{64}$ terhadap $G_{128}$:
  $$\text{Rasio} = \frac{4{,}250}{4{,}125} \approx 1{,}0303\times \quad (\text{Hasil observasi empiris: } 1{,}0295\times)$$

### 4.2. Ukuran Berkas Logis Model Penuh (Qwen1.5-MoE-A2.7B)

- Bobot model BF16 asli: $\approx 28{,}63\text{ GB}$ ($N_q \approx 14{,}3\text{ miliar elemen}$ total pada parameter model).
- Ukuran biner terkuantisasi 4-bit logis (`stat st_size`): $\approx 7{,}385\text{ GB} \pm 10\%$.
- Deviasi empiris terhadap prediksi F11b: **$2{,}53\%$** (jauh di bawah batas toleransi gate $10\%$).

### 4.3. Proyeksi Bandwidth & Throughput Awal Decode 4-bit (Handoff ke M7)

Pada Milestone M5, decode 16-bit membutuhkan bandwidth streaming bobot per token:
$$B_{tok}^{16-bit} = B_{active,expert}^{16-bit} + B_{dense}^{16-bit} \approx 4{,}13\text{ GB/token}$$

Dengan kuantisasi 4-bit blok-128 ($bpw_{eff} = 4{,}125$):
$$B_{tok}^{4-bit} \approx \frac{4{,}125}{16} \times 4{,}13\text{ GB} \approx 1{,}066\text{ GB/token}$$

Penurunan volume streaming sebesar $\approx 3{,}88\times$ ini secara langsung menurunkan tekanan bandwidth disk NVMe/RAM pada Milestone M7 (`docs/milestones/M7-odirect-lru.md`), memungkinkan engine mencapai target throughput decoding token interaktif melalui integrasi O_DIRECT dan zero-copy custom LRU cache.

---

## 5. Rangkuman Artefak & Golden Identity Pins

Seluruh komponen pendukung telah ter-commit dan terkunci secara kriptografis:

| File Artefak                            | Peran / Fungsi                                                        |  Status Verifikasi   |
| :-------------------------------------- | :-------------------------------------------------------------------- | :------------------: |
| `src/engine/quantize.mojo`              | Engine kuantisasi Mojo (F11a, RNE, group packing, framing)            | Teruji & tervalidasi |
| `src/engine/dequant.mojo`               | Kernel dekuantisasi SIMD Mojo (vectorized FP16 $\to$ FP32 $\to$ BF16) | Teruji & tervalidasi |
| `src/tools/quantize_cli.mojo`           | CLI tool `dismoen quantize` & subcommand handler                         | Teruji & tervalidasi |
| `tools/oracle/oracle_quant.py`          | Reference oracle kuantisasi PyTorch & verifikator Q-domain            | Teruji & tervalidasi |
| `tools/oracle/oracle_ppl.py`            | Runner evaluasi PPL & argmax agreement global                         | Teruji & tervalidasi |
| `tools/fixtures/m6_ppl_corpus.json`     | Korpus teks terstandar 100 dokumen $\times$ 256 token ID              |   Terkunci SHA-256   |
| `tools/fixtures/ppl_golden_pins.json`   | Golden pins reproduksibilitas (manifest model, tokenizer, baseline)   |   Terkunci SHA-256   |
| `tests/integration/test_m6_quantize.sh` | Master test suite IT-M6-1 .. IT-M6-16 & gate certifier                |      100% PASS       |

---

## 6. Keputusan Penutupan Milestone M6

Berdasarkan seluruh hasil pengujian:

1. Seluruh gelombang implementasi M6-W1 (Format, Structs, Validation & Serialization), M6-W2 (Quant Algorithm, Scales & Property Tests), M6-W3 (Quantize CLI, Error Schema & Atomic Rollback), M6-W4 (SIMD Dequant Kernel & Conformance G-M6-K), M6-W5 (Oracle PPL Runner & Language Quality G-M6-3), dan M6-W6 (Master Gates G-M6-1..3 + G-M6-K & Integration Matrix IT-M6-1..16) telah diselesaikan secara lengkap.
2. Seluruh 16 skenario matriks integrasi dinyatakan **PASS** tanpa supresi atau pola `|| true`.
3. Seluruh 4 gate normatif (**G-M6-1**, **G-M6-2**, **G-M6-3**, dan **G-M6-K**) dinyatakan **PASS (GREEN)**.
4. Binary Mojo berhasil dikompilasi bersih (0 warning, 0 error) dan seluruh format mematuhi standar Mojo 1.0.0 serta RFC 8259 JSON.

Dengan ini, **Milestone M6 (Quantizer & Compressed Storage) dinyatakan RESMI DITUTUP (CLOSED)**. Pengembangan sistem disk-streaming MoE engine kini siap melangkah ke konsumen utamanya: **Milestone M7 (O_DIRECT Streaming & Custom LRU Cache)**.
