# M10 — Konsolidasi DISMOEN (Full Qwen 3.6-35B-A3B, Pembersihan Qwen 1.5 & Rebranding)

> Proyek: `disk-streaming-moe-engine`. Fase: **Consolidation & Production Ready**. Index: `../README.md`.
> SSOT Arsitektur: Qwen 3.6-35B-A3B Hybrid (40 layer, 30 GDN + 10 Gated Attention, MoE 256 routed/8 aktif + 1 shared).

| Field             | Nilai                                                                                                         |
| :---------------- | :------------------------------------------------------------------------------------------------------------ |
| **Deliverable**   | Engine terpadu `dismoen` (Qwen 3.6 SSOT, pembersihan total artefak Qwen 1.5)                                  |
| **Komponen**      | CLI Branding (`dismoen`), Config Unifikasi, Forward/Decode Hybrid, Storage Sanitization                       |
| **Prasyarat**     | M9 hijau (Port Qwen 3.6 & Gates G-M9-1..4 lolos) + M8 hijau (GDN)                                             |
| **Next**          | M11 (Multi-Core Scaling, Amdahl Curve F16, Async Double-Buffered I/O)                                         |
| **Gate**          | G-M10-1..G-M10-4                                                                                              |
| **Storage Bebas** | $\approx +33{,}55\text{ GiB}$ (26,68 GiB Safetensors 1.5 + 6,88 GiB quant bin dibersihkan dari storage aktif) |

---

## 1. Latar Belakang & Tujuan Utama

Milestone M10 meresmikan transformasi dari fase riset multi-model (Trial model Qwen 1.5 pada M0–M7 dan Port model Qwen 3.6 pada M9) menuju **satu engine inferensi produksi yang mandiri, bersih, dan optimal**: **`dismoen`** (**DI**sk **S**treaming **MO**e **EN**gine).

### Main Goal (Sasaran Kritis):

1. **Pembersihan Total Bobot Fisik Qwen 1.5 di Disk**:
   - Menghapus bobot legacy Safetensors `~/models/qwen1.5-moe-a2.7b-chat` ($26{,}68\text{ GiB}$) dan custom quant `~/models/qwen1.5-moe-a2.7b-chat-4bit` ($6{,}88\text{ GiB}$).
   - Membebaskan $\approx \mathbf{33{,}55\text{ GiB}}$ kapasitas disk pada partisi host, menaikkan ketersediaan storage dari $\sim 54\text{ GiB}$ menjadi **$\sim 87{,}55\text{ GiB}$**.
   - Menjamin kapasitas storage maksimal dan aman untuk menampung bobot asli Qwen 3.6-35B ($68{,}12\text{ GiB}$) beserta berkas kuantisasi runtime GGUF target ($13{,}5 - 16{,}8\text{ GiB}$), dengan margin operasional $B_{reserved}$ sesuai §4.2 sehingga operasi M10 diizinkan hanya jika safety margin tetap terpenuhi.
2. **Rebranding Resmi Menjadi `dismoen`**:
   - Mengganti nama binary CLI dari `kimo` menjadi `dismoen` (akronim: **DI**sk **S**treaming **MO**e **EN**gine).
   - Mengeliminasi symlink transisi `kimo -> dismoen` dan `tools/kimo-tools` secara tuntas.
   - Memperbarui crate perkakas Rust `tools/kimo-tools` menjadi `tools/dismoen-tools`.
3. **Pembersihan Percabangan Kode Legacy Trial (`trial` codepaths)**:
   - Membuang logika kondisional `if architecture == "trial"` di config parser, scheduler, dan CLI.
   - Menghapus format custom M6 (`quant_model.bin`, 256B header JSON) demi standardisasi penuh ke format industri **GGUF v3** (`.gguf`, Q3_K, Q4_K).
4. **Penyatuan Subperintah CLI (Unifikasi Forward & Decode)**:
   - Menyatukan `forward-port` menjadi perintah standar `dismoen forward`.
   - Mengintegrasikan loop autoregressive decoding hybrid (10 Gated Attention + 30 GDN + KMSS v1 session state) langsung ke dalam `dismoen decode`.
5. **Decoupling Suite Uji Regresi dari Model 26,68 GiB**:
   - Seluruh pengujian unit dan integrasi M0–M9 dialihkan ke fixture sintetis mini Qwen 3.6 port (< 5 MB), sehingga pengujian CI berjalan deterministik dalam hitungan detik tanpa membutuhkan model 26,68 GiB yang dihapus.

---

## 2. Peta Pembersihan & Dekomisioning Qwen 1.5

### 2.1 Berkas Penyimpanan (Storage Sanitization)

> Lokasi root model diatur via variabel lingkungan `${DISMOEN_MODEL_ROOT:-~/models}` atau flag CLI `--model-dir`.
>
> **Precedence (mengikat, bukan implementation-dependent):**
> `--model-dir` > `DISMOEN_MODEL_ROOT` > `~/models`.
> Contoh CLI di milestone ini memakai bentuk portable
> `--model-dir "$DISMOEN_MODEL_ROOT/qwen3.6-35b-a3b"`.

| Lokasi Berkas / Direktori                           | Ukuran Eksak (Bytes / GiB)                             |    Status Tindakan M10     | Justifikasi                                                              |
| :-------------------------------------------------- | :----------------------------------------------------- | :------------------------: | :----------------------------------------------------------------------- |
| `${DISMOEN_MODEL_ROOT}/qwen1.5-moe-a2.7b-chat`      | $28{,}644{,}046{,}163\text{ B}$ ($26{,}68\text{ GiB}$) | **DIHAPUS / DIPENSIUNKAN** | Shard Safetensors 1.5 trial tidak lagi dipakai setelah M9 tersertifikasi |
| `${DISMOEN_MODEL_ROOT}/qwen1.5-moe-a2.7b-chat-4bit` | $7{,}382{,}480{,}468\text{ B}$ ($6{,}88\text{ GiB}$)   | **DIHAPUS / DIPENSIUNKAN** | Format kuantisasi custom M6 usang; digantikan oleh GGUF v3               |
| `${DISMOEN_MODEL_ROOT}/qwen3.6-35b-a3b`             | $73{,}139{,}739{,}806\text{ B}$ ($68{,}12\text{ GiB}$) |  **DIPERTAHANKAN (SSOT)**  | Checkpoint target produksi 26 shard Safetensors BF16                     |

### 2.2 Komponen Kode yang Didepresiasi & Dihapus

```text
KODE LEGACY TRIAL (M0-M7)                    KONSOLIDASI DISMOEN (M10)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│ - parse_model_config (trial 24L)    │ ──> │ - parse_model_config (Universal)     │
│ - parse_model_config_adapter        │     │   Otomatis unwrap text_config        │
│ - Hardcode 60 experts, top-4        │     │   Default Qwen 3.6 hybrid (40L/256E) │
├──────────────────────────────────────┤     ├──────────────────────────────────────┤
│ - cmd_forward (trial homogen)        │ ──> │ - cmd_forward (Universal 40L Hybrid) │
│ - cmd_forward_port                   │     │   (alias cmd_forward_port tetap ada) │
├──────────────────────────────────────┤     ├──────────────────────────────────────┤
│ - cmd_decode (trial MHA 24L)         │ ──> │ - cmd_decode (Hybrid GDN + GatedAttn)│
│   Token output JSON                  │     │   Dukungan KMSS v1 session reuse     │
├──────────────────────────────────────┤     ├──────────────────────────────────────┤
│ - quant_model.bin (Format M6 256B)   │ ──> │ - Standar GGUF v3 (Q3_K / Q4_K)      │
│   src/format/quant_format.mojo       │     │   src/format/gguf.mojo               │
├──────────────────────────────────────┤     ├──────────────────────────────────────┤
│ - models.lock.json (Hash Qwen 1.5)   │ ──> │ - models.lock.json (Hash Qwen 3.6)   │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
```

### 2.3 Dekopling Pengujian Unit & Integrasi

Berkas pengujian berikut yang sebelumnya menyentuh path fisik `~/models/qwen1.5-moe-a2.7b-chat*` diperbarui agar membaca berkas fixture sintetis mini port (`fixtures/m9_port_mini.gguf`, `fixtures/m9_port_weights.safetensors`) atau shard riil Qwen 3.6:

- `tests/unit/test_odirect.mojo`: Target probe dialihkan ke fixture mini GGUF atau shard model 3.6.
- `tests/unit/test_lru_cache.mojo`: Integrasi probe dialihkan ke pembacaan payload terkompresi GGUF.
- `tests/integration/test_m9_w1_config_adapter.sh`: Verifikasi architecture-mismatch dialihkan ke fixture sintetis dummy, bukan path model fisik 26,68 GiB.

---

## 3. Identitas & Spesifikasi Antarmuka `dismoen`

### 3.1 Filosofi Penamaan

`dismoen` dibentuk dari akronim deskriptif arsitektur mesin:
$$\textbf{DI}\text{sk } \textbf{S}\text{treaming } \textbf{MO}\text{e } \textbf{EN}\text{gine} \implies \textbf{dismoen}$$

Binary CLI dipasang sebagai executable mandiri berkinerja tinggi yang ditulis murni dalam bahasa Mojo 1.0.0.

### 3.2 Spesifikasi Perintah CLI `dismoen`

Semua subperintah kini berjalan secara konsisten di bawah executable `dismoen`:

#### 1. Forward Pass Inferensi (`dismoen forward`)

Menjalankan prefill atau eksekusi forward pass pada model target:

```bash
dismoen forward \
  --model-dir "$DISMOEN_MODEL_ROOT/qwen3.6-35b-a3b" \
  --tokens fixtures/m9_port_tokens.json \
  --save-session work/session1.kmss \
  --timing-profile
```

#### 2. Autoregressive Streaming Decode (`dismoen decode`)

Menjalankan decoding autoregresif berbasis kelanjutan session `KMSS v1` (tanpa komputasi ulang token historis — `historical_recompute_tokens == 0`, §3.2):

```bash
dismoen decode \
  --model-dir "$DISMOEN_MODEL_ROOT/qwen3.6-35b-a3b" \
  --session work/session1.kmss \
  --max-tokens 32 \
  --output work/output_tokens.json
```

> **Definisi metric (M10-3).** Decode autoregresif tetap memproses token
> **baru** — yang dilarang dihitung ulang adalah token **historis**.
> Metric gate bernama presisi:
>
> $$historical\_recompute\_tokens \equiv 0$$
>
> = jumlah token prompt/state historis yang dikomputasi ulang saat
> continuation (riwayat prompt + prefix yang sudah ada di KMSS).
> Pemrosesan token baru hasil generasi **diharapkan dan tidak dihitung**
> di metric ini. Nama lama `recompute_tokens` (era M9) bermakna sama
> tetapi mudah disalahbaca sebagai "zero computation"; artefak beku
> M9/M5 tidak ditulis ulang — M9 memakai nama lama sebagai rekam
> historis, M5 memakai kata "recompute" untuk konsep berbeda (baseline
> full-recompute sebagai pembanding ekuivalensi, bukan metric reuse).
> Engine `dismoen` meng-emit field baru `historical_recompute_tokens`;
> verifier M10 menegaskan field baru tersebut.

#### 3. Verifikasi Index & Format Tensor (`dismoen check-index`)

Memeriksa integritas berkas Safetensors / GGUF dan keselarasan padding I/O:

```bash
dismoen check-index "$DISMOEN_MODEL_ROOT/qwen3.6-35b-a3b/model-00001-of-00026.safetensors"
```

#### 4. Recurrent Linear Attention Operator (`dismoen gdn`)

Mengevaluasi kernel chunked scan Gated DeltaNet (WY representation):

```bash
dismoen gdn \
  --tokens fixtures/m8_tokens.json \
  --output work/gdn_state.gdns \
  --chunk-size 512
```

#### 5. Numeric Verification & Gate Evaluation (`dismoen compare`)

Membandingkan paritas numerik kandidat output terhadap oracle reference:

```bash
dismoen compare \
  --reference fixtures/m9_port_logits_naive.bin \
  --candidate work/logits.bin \
  --gate G-M9-1
```

---

## 4. Model Analitis Kapasitas Storage Pasca-Sanitasi

### 4.1 Standar Satuan & Neraca Kapasitas Partisi Host ($S_{host}$)

> **Aturan SSOT Satuan Penyimpanan:**
>
> - $\text{Bytes}$: Integer presisi tunggal (SSOT penghitungan filesystem).
> - $\text{GiB} = \text{Bytes} / 2^{30} = \text{Bytes} / 1{,}073{,}741{,}824$ (standar biner IEC, metrik resmi engine).
> - $\text{GB} = \text{Bytes} / 10^9 = \text{Bytes} / 1{,}000{,}000{,}000$ (standar metrik desimal SI, hanya info tambahan).
>
> Seluruh variabel gate memakai GiB: $S_{avail}$, $S_{model}^{raw}$,
> $S_{quant}$, $S_{headroom}$ dalam GiB. Angka GB (SI) hanya tampil sebagai
> info parentetis dan tidak boleh masuk aritmetika gate.

**Neraca Sebelum Pembersihan (Kondisi Akhir M9):**

- Partisi Host: $S_{used} \approx 387\text{ GiB}$, $S_{avail} \approx 54\text{ GiB}$ (Utilisasi $\approx 88\%$).

**Perhitungan Eksak Berkas Qwen 1.5 yang Dibersihkan:**

$$
\begin{aligned}
S_{safetensors}^{1.5} &= 28{,}644{,}046{,}163\text{ Bytes} \approx 26{,}677\text{ GiB} \quad (28{,}644\text{ GB SI}) \\
S_{quant}^{1.5}       &= 7{,}382{,}480{,}468\text{ Bytes} \approx 6{,}875\text{ GiB} \quad (7{,}382\text{ GB SI}) \\
\Delta S_{freed}      &= 28{,}644{,}046{,}163 + 7{,}382{,}480{,}468 = \mathbf{36{,}026{,}526{,}631\text{ Bytes}} \approx \mathbf{33{,}552\text{ GiB}} \quad (36{,}027\text{ GB SI})
\end{aligned}
$$

Ambang batas kelulusan kuota sanitasi storage (Gate G-M10-2):
$$\Delta S_{freed} \ge 33 \times 2^{30}\text{ Bytes} \quad (= 35{,}433{,}480{,}192\text{ Bytes})$$

Kapasitas ruang bebas baru pasca-sanitasi M10:
$$S_{avail}^{M10} = S_{avail}^{M9} + \Delta S_{freed} \approx 54\text{ GiB} + 33{,}55\text{ GiB} \approx \mathbf{87{,}55\text{ GiB}} \quad (\text{Utilisasi turun ke } \approx 80\%)$$

### 4.2 Alokasi Ruang untuk Qwen 3.6-35B dan Target Kuantisasi

Kondisi fisik penyimpanan host:

- Shard asli Qwen 3.6-35B ($68{,}12\text{ GiB}$) **sudah ada di disk** dan sudah terhitung di dalam $S_{used} = 387\text{ GiB}$.
- Setelah pembersihan artefak Qwen 1.5 ($\Delta S_{freed} = 33{,}552\text{ GiB}$), ruang kosong partisi meningkat menjadi $S_{avail}^{M10} = 87{,}552\text{ GiB}$.
- Berkas baru yang wajib ditulis ke ruang kosong hanyalah berkas kuantisasi runtime GGUF ($S_{quant}^{target} \approx 15{,}20\text{ GiB}$).

**Perhitungan Headroom Penyimpanan yang Benar:**

1. **Skenario Operasi Normal (In-Place Quantization)**:
   Karena bobot asli $68{,}12\text{ GiB}$ sudah tersimpan, pembuatan berkas kuantisasi GGUF hanya mengonsumsi $15{,}20\text{ GiB}$ dari ruang kosong:
   $$S_{headroom}^{real} = S_{avail}^{M10} - S_{quant}^{target} = 87{,}552\text{ GiB} - 15{,}20\text{ GiB} = \mathbf{72{,}352\text{ GiB}} \quad \implies \quad \mathbf{[SANGAT\;LEGA]}$$
   Partisi host menyisakan ruang bebas sebesar $\approx 72{,}35\text{ GiB}$, sangat aman untuk buffer scratch, logs, dan KV cache swap.

2. **Skenario Ekstrem (Worst-Case Duplikasi / Re-Download Full Checkpoint)**:
   Seandainya partisi harus menampung _salinan baru_ checkpoint utuh $68{,}12\text{ GiB}$ dari nol secara bersamaan dengan target GGUF $15{,}20\text{ GiB}$:
   $$S_{headroom}^{worst\_case} = S_{avail}^{M10} - S_{model}^{raw} - S_{quant}^{target} = 87{,}552\text{ GiB} - 68{,}117\text{ GiB} - 15{,}200\text{ GiB} = \mathbf{4{,}236\text{ GiB}} \quad \implies \quad \mathbf{[TEORITIS]}$$
   Angka $4{,}236\text{ GiB}$ adalah **theoretical worst-case headroom** —
   hasil aritmetika atas byte terukur — **bukan jaminan operasional**:
   kuantisasi nyata membutuhkan temporary files, metadata filesystem,
   scratch, page cache, artefak konversi, logs, dan partial output, dan
   `available bytes` filesystem bukan kontrak bahwa seluruh 4,2 GiB dapat
   dipakai satu proses tanpa kondisi lain berubah. Dokumen ini **dilarang**
   mengklaim 4,24 GiB otomatis menjamin `ENOSPC` tidak mungkin.
   Aturan operasional yang mengikat dihitung **sebelum** operasi tulis
   dimulai (contract bug yang dilarang: memakai nilai "after" yang baru
   diketahui setelah operasi):

   $$B_{free\_before} = \text{statvfs}(f\_{bavail} \times f\_{frsize})$$
   $$B_{required} = \text{upper bound estimasi operasi (output + scratch + temp + logs)}$$
   $$B_{free\_expected\_after} = B_{free\_before} - B_{required}$$

   $$\text{ALLOW} \iff B_{free\_expected\_after} \ge B_{reserved} \quad\text{dengan}\quad B_{reserved} = 2\text{ GiB (default proyek, tunable)}$$

   Untuk operasi ber-output diketahui (tulis GGUF $15{,}20\text{ GiB}$),
   pertidaksamaan ini deterministik. Operasi yang dapat membuat temporary
   files berukuran tak-diketahui **dilarang tanpa** upper bound eksplisit
   atau scratch budget yang dideklarasikan — verifier memakai bound
   tersebut sebagai $B_{required}$. Pelanggaran → abort pra-tulis, bukan
   berharap headroom teoretis cukup.

### 4.3 Lapisan Source Zero-Legacy (P1)

Storage bersih saja tidak cukup: simbol legacy yang masih hidup di source
aktif (ditemukan saat audit: cabang `"trial"` di `config.mojo` /
`config_parser.mojo` / `cmd_forward_port.mojo`, path `quant_model.bin` di
`cmd_decode.mojo` / `cmd_quantize.mojo`, kandidat biner `kimo-tools` di
`cmd_compare.mojo`) akan membuat gate lolos semu. Lapisan source G-M10-2:

- **Pola presisi** (dihitung via `rg` quoted/boundary, bukan substring):
  `"trial"`, `quant_model\.bin`, `\.kimo\.bin`, `QuantHeader`,
  `QuantTensorMetadata`, `parse_model_config_adapter`, `kimo-tools` —
  masing-masing $\equiv 0$ di `src/` (kode, komentar, maupun docstring).
- **Bukan pelanggaran**: variabel loop bernama `trial` (tanpa quote),
  `kimo` polos (symlink `kimo` dieliminasi total),
  dan arsip `docs/`/`reports/` historis.
- **Kasus khusus `quant_format.mojo`**: helper matematika FP16 generik
  (`float16_to_u16`, `u16_to_float16`, …) yang dipakai `gguf.mojo`
  **direlokasi ke modul netral** (bukan alasan mempertahankan file);
  file `src/format/quant_format.mojo` beserta simbol kontainer M6-nya
  di-retire. Mempertahankan import "hanya untuk helper" = FAIL.
- **Scope `tests/`/`tools/`**: aturan simbol yang sama; uji mismatch
  `--architecture trial` peninggalan M9 wajib dimigrasi ke kontrak baru
  (flag `--architecture` dihapus → assert unknown-flag, bukan exit-2
  mismatch). Satu-satunya pengecualian adalah data fixture milik verifier
  itu sendiri.
- **Konflik yang harus diselesaikan W1**: `test_m10_w1_rebrand.sh` Stage 4
  saat ini memverifikasi dual-binary (`dismoen-tools` + `kimo-tools`);
  kriteria `kimo-tools ≡ 0` di atas berarti stage tersebut wajib ditulis
  ulang menjadi single-binary `dismoen-tools` — dual-binary tidak boleh
  dipertahankan diam-diam sebagai pengecualian.

### 4.4 Definisi Verifier Byte (P1)

`file size != blocks allocated`, dan delta free-space dapat bergerak karena
aktivitas filesystem lain — maka gate byte memakai ukuran logis yang
deterministik, bukan observasi free-space:

- **$L_{paths}$** = jumlah path legacy yang masih ada
  (`qwen1.5-moe-a2.7b-chat`, `qwen1.5-moe-a2.7b-chat-4bit` di
  `${DISMOEN_MODEL_ROOT}`); syarat $L_{paths} \equiv 0$.
- **$L_{logical}$** = $\sum \text{stat(st\_size)}$ atas path legacy
  (logical bytes, bukan allocated blocks); syarat $L_{logical} \equiv 0$.
- **$\Delta B_{free}$** (selisih free-space sebelum/sesudah sanitasi)
  hanya **cross-check observasional** (ekspektasi $\approx \Delta S_{freed}$,
  toleran terhadap noise filesystem lain) — tidak pernah menjadi kriteria
  lulus/gagal.
- Aturan operasional §4.2 dihitung sebelum operasi tulis:
  $B_{free\_before}$ dari `statvfs`, $B_{required}$ dari upper bound yang
  dideklarasikan, lolos iff $B_{free\_before} - B_{required} \ge B_{reserved}$.

### 4.5 Identitas Lockfile Production-Ready

`lockfile = Qwen3.6` saja bukan penguncian — lock minimal yang membuat
artefak reproducible wajib memuat 7 field:

1. `model_id` (mis. `Qwen/Qwen3.6-35B-A3B`),
2. `architecture` (hybrid 40L: 30 GDN + 10 gated-attention, MoE 256/8+1),
3. `revision` / commit HF yang ter-pin,
4. manifest shard bobot (26 shard, nama + size),
5. `sha256` per shard (placeholder `"pinned"` tidak diterima gate),
6. `tokenizer_revision` + `sha256` (`tokenizer.json`, `tokenizer_config.json`),
7. `config_hash` (`sha256` atas `config.json` efektif yang dibaca engine).

`models.lock.json` wajib dilengkapi hingga 7 field di atas sebagai pekerjaan
wajib W3 (termasuk mengganti placeholder `sha256: "pinned"` bila disalin
dari file port). Engine menolak start saat identitas terukur ≠
identitas terkunci (SEC-1) dengan **manifest equality** penuh — bukan
sekadar perbandingan satu hash:

$$\text{actual\_shard\_set} == \text{locked\_shard\_set} \;\land\; \text{actual\_size} == \text{locked\_size} \;\land\; \text{actual\_sha256} == \text{locked\_sha256}$$

Kasus yang masing-masing wajib FAIL mandiri: shard hilang, shard ekstra,
nama file salah, jumlah shard salah (26 shard adalah bagian identitas
model), dan size mismatch — bahkan bila hash shard yang tersisa cocok.

**Fate `models.lock.port.json` (jaminan SSOT tunggal).**
Target M10 adalah satu production lock identity. Maka:

- `models.lock.json` (7 field di atas) = **satu-satunya lock production**;
  engine production hanya memuat file ini.
- `models.lock.port.json` = **retired setelah M10-W3**: loader live
  (`cmd_forward_port.mojo`, `security_port.mojo`) dimigrasi ke
  `models.lock.json`, lalu file port dihapus dari root repo.
  Tidak ada periode dua-SSOT: setelah W3, referensi "lock" tanpa kualifikasi
  selalu berarti `models.lock.json`.
- Dependensi test M9 yang memuat root file port (`test_m9_w3_loader.sh`)
  dimigrasi/dibekukan bersama W2/W3; referensi M12 ke file port
  (§2.1, W1, DoD) ikut dipindah ke `models.lock.json`.

---

## 5. Quality Gates (Fase Konsolidasi M10)

| Gate        | Kriteria Penilaian                                                                                                                                                                                                                                                                                    |                                                                                                                                                                                                                                                                             Ambang Batas                                                                                                                                                                                                                                                                              | Verifier Tool                   |
| :---------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------ |
| **G-M10-1** | **Rebranding & Toolchain Integrity**: Kompilasi `dismoen` 0 compiler warning, 0 leftover legacy, CLI banner menampilkan nama DISMOEN                                                                                                                                                                  |                                                                                                                                                                                                                                                             Exit code 0, binary executable, symlink valid                                                                                                                                                                                                                                                             | `test_m10_w1_rebrand.sh`        |
| **G-M10-2** | **Storage Sanitization & Zero-Legacy Multi-Layer Verification**: Penghapusan fisik berkas legacy 1.5 di `${DISMOEN_MODEL_ROOT}`, dekopling referensi tes aktif, penguncian lockfile, verifikasi kuota pembebasan byte, **dan zero-legacy source** (tidak ada simbol trial/quant-legacy di kode aktif) | 0 legacy path ($L_{paths} \equiv 0$), 0 test references, lockfile Qwen3.6 lengkap 7 field (§4.5), $L_{logical} \equiv 0$ (§4.4, bukan free-space), $\Delta S \ge 33 \times 2^{30}\text{ B}$, margin operasional $B_{free\_before} - B_{required} \ge B_{reserved} = 2\text{ GiB}$ (§4.2), + lapisan source (§4.3): `"trial"` $\equiv 0$, `quant_model.bin`+`.kimo.bin` $\equiv 0$, `{QuantHeader, QuantTensorMetadata}` $\equiv 0$, `parse_model_config_adapter` $\equiv 0$, `kimo-tools` $\equiv 0$ di `src/` (kode+komentar+docstring), `quant_format.mojo` retired | `test_m10_w2_sanitization.sh`   |
| **G-M10-3** | **Unified Forward Numerical Parity & Decode Continuation**: `dismoen forward` memenuhi paritas numerik terhadap reference logits M9 ($\Delta_{\max} \le 10^{-7}$), `dismoen decode` menjalankan hybrid continuation tanpa menghitung ulang token historis                                             |                                                                                                                                                                                                                          $\Delta_{\max} \le 10^{-7}$, $\text{historical\_recompute\_tokens} = 0$ (§3.2), $\text{gdn\_reused} = \text{true}$                                                                                                                                                                                                                           | `test_m10_w3_forward_decode.sh` |
| **G-M10-4** | **Zero Regression & Code Hygiene**: Seluruh suite tes regresi (`validate-m9`, `validate-m8`) dan 13 hook pre-commit 100% hijau                                                                                                                                                                        |                                                                                                                                                                                                                                                                  100% PASS, 0 `# noqa`, 0 `#[allow]`                                                                                                                                                                                                                                                                  | `test_m10_w5_gates.sh`          |

---

## 6. Rencana Gelombang Kerja (Execution Waves)

Pelaksanaan Milestone M10 dipecah menjadi 5 gelombang kerja berurutan:

### Gelombang 1 (M10-W1): Rebranding Toolchain & Executable `dismoen`

- Konfigurasi `pixi.toml` dan `Makefile` untuk mengompilasi binary `-o dismoen`.
- Perbarui banner usage di `src/main.mojo`.
- Rename / aliaskan crate `tools/kimo-tools` menjadi `tools/dismoen-tools`.
- Verifikasi Gate G-M10-1.

### Gelombang 2 (M10-W2): Storage Sanitization & Test Fixture Decoupling

- Perbarui `test_odirect.mojo` dan `test_lru_cache.mojo` agar membaca fixture sintetis mini atau shard 3.6.
- Perbarui `test_m9_w1_config_adapter.sh` agar bebas dari path fisik `${DISMOEN_MODEL_ROOT}/qwen1.5*`.
- Hapus direktori `${DISMOEN_MODEL_ROOT}/qwen1.5-moe-a2.7b-chat` dan `${DISMOEN_MODEL_ROOT}/qwen1.5-moe-a2.7b-chat-4bit`.
- Perluas `test_m10_w2_sanitization.sh` dengan sweep simbol presisi §4.3 atas `src/` + `tests/` + `tools/` (pola quoted/boundary — substring `trial` pada nama variabel bukan pelanggaran).
- Verifikasi multi-layer Gate G-M10-2: $L_{paths} \equiv 0$, $L_{logical} \equiv 0$ (logical bytes via `stat`, bukan free-space), 0 referensi di `tests/`, lockfile Qwen 3.6, $\Delta B_{free}$ hanya cross-check observasional (§4.4), assert margin $B_{free\_before} - B_{required} \ge B_{reserved} = 2\text{ GiB}$ sebelum operasi tulis (§4.2), dan pembebasan disk $\Delta S_{freed} \ge 33 \times 2^{30}\text{ Bytes}$ ($35{,}433{,}480{,}192\text{ B}$).

### Gelombang 3 (M10-W3): Unifikasi ModelConfig & Perintah `forward`

- Satukan `parse_model_config` dan `parse_model_config_adapter` di `src/cli/config_parser.mojo` (adapter dihapus, bukan di-wrap).
- Hapus enum dan percabangan `architecture == "trial"` (termasuk default `"trial"`, flag `--architecture`, dan cabang `cmd_forward_port.mojo:166`). Jadikan Qwen 3.6 sebagai satu-satunya arsitektur default.
- Satukan implementasi `cmd_forward_port` menjadi `dismoen forward`.
- Ganti kandidat biner `kimo-tools` di `cmd_compare.mojo` menjadi `dismoen-tools` semata.
- Perbarui `models.lock.json` mengunci spesifikasi Qwen 3.6-35B-A3B.
- Lengkapi lockfile hingga 7 field §4.5 (ganti placeholder `sha256: "pinned"` dengan hash terukur; tambah tokenizer revision/hash + config hash); engine assert-on-start dengan manifest equality SEC-1.
- Migrasi loader `models.lock.port.json` (`cmd_forward_port.mojo`, `security_port.mojo`, test M9 terkait, referensi M12) ke `models.lock.json`, lalu retire file port dari root (fate §4.5).

### Gelombang 4 (M10-W4): Unifikasi Hybrid `decode` & Pemensiunan Format M6

- Implementasikan loop autoregresif hybrid penuh pada `cmd_decode.mojo` (mendukung 10 Attention + 30 GDN + session continuation `KMSS v1`).
- Rename field JSON `recompute_tokens` → `historical_recompute_tokens` pada output `dismoen forward`/`dismoen decode` (definisi §3.2; verifier M10 menegaskan field baru; artefak beku M9/M5 tidak diubah).
- Pensiunkan format custom M6 `quant_model.bin` dari codebase aktif (hapus path probe di `cmd_decode.mojo`, `final_dest` di `cmd_quantize.mojo`, dan referensi `.kimo.bin` di `quant_reader.mojo`/`quant_loader.mojo`).
- Retire `src/format/quant_format.mojo`: relokasi helper FP16 generik ke modul netral, hapus simbol kontainer M6 (`QuantHeader`, `QuantTensorMetadata`).
- Tetapkan GGUF v3 sebagai satu-satunya format kuantisasi terkompresi.

### Gelombang 5 (M10-W5): Sertifikasi Quality Gates (G-M10-1..4) & Penutupan M10

- Jalankan master integration test suite `test_m10_w5_gates.sh`.
- Sertifikasi seluruh gate G-M10-1 s.d. G-M10-4.
- Terbitkan scorecard formal penutupan di `reports/YYYY-MM-DD/M10-gates-scorecard.md`.
- Milestone M10 resmi selesai $\to$ Menuju M11 (Core Scaling / Amdahl F16).

---

## 7. Definisi Selesai (DoD M10)

- [ ] Binary utama terkompilasi sebagai `dismoen` dengan 0 leftover legacy dan banner resmi `DISMOEN`.
- [ ] Berkas bobot fisik Qwen 1.5 terhapus dari `${DISMOEN_MODEL_ROOT}` dengan verifikasi multi-layer Gate G-M10-2 ($L_{paths} \equiv 0$, $L_{logical} \equiv 0$ logical-bytes, 0 referensi di tes aktif, $\Delta B_{free}$ hanya observasional, margin tulis $B_{free\_before} - B_{required} \ge 2\text{ GiB}$, dan pembebasan $\ge 33 \times 2^{30}\text{ Bytes}$).
- [ ] Zero-legacy source lolos Gate G-M10-2 lapisan §4.3 (0 simbol trial/quant-legacy/`kimo-tools` di `src/`; `quant_format.mojo` retired; uji mismatch trial M9 dimigrasi; stage dual-binary W1 ditulis ulang single-binary).
- [ ] Pengujian unit `test_odirect.mojo` dan `test_lru_cache.mojo` terbebas dari path Qwen 1.5 dan lulus 100%.
- [ ] Seluruh percabangan `trial` pada `config_parser.mojo` dan `config.mojo` dibersihkan; Qwen 3.6 hybrid menjadi arsitektur default.
- [ ] `dismoen forward` terpadu lolos verifikasi paritas numerik (Unified Forward Numerical Parity) terhadap logits M9 ($\Delta_{\max} \le 10^{-7}$).
- [ ] `dismoen decode` mendukung decoding autoregresif 40-layer hybrid dengan session continuation `KMSS v1` (`historical_recompute_tokens == 0`, §3.2).
- [ ] `models.lock.json` diperbarui mengunci spesifikasi resmi target Qwen 3.6-35B-A3B dengan 7 field identitas §4.5 (tanpa placeholder hash).
- [ ] Seluruh suite pengujian regresi (`validate-m9`, `validate-m8`) dan 13 hook `pre-commit` 100% hijau tanpa suppressions (`# noqa`, `#[allow]`).
- [ ] Laporan scorecard sertifikasi M10 ter-commit di `reports/YYYY-MM-DD/M10-gates-scorecard.md`.
