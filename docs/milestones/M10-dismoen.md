# M10 — Konsolidasi DISMOEN (Full Qwen 3.6-35B-A3B, Pembersihan Qwen 1.5 & Rebranding)

> Proyek: `disk-streaming-moe-engine`. Fase: **Consolidation & Production Ready**. Index: `../README.md`.
> Implementasi dipecah menjadi waves: `../../scratch/wave/m10/README.md` (W1 rebranding → W5 gates).
> SSOT Arsitektur: Qwen 3.6-35B-A3B Hybrid (40 layer, 30 GDN + 10 Gated Attention, MoE 256 routed/8 aktif + 1 shared).

| Field             | Nilai                                                                                             |
| :---------------- | :------------------------------------------------------------------------------------------------ |
| **Deliverable**   | Engine terpadu `dismoen` (Qwen 3.6 SSOT, pembersihan total artefak Qwen 1.5)                      |
| **Komponen**      | CLI Branding (`dismoen`), Config Unifikasi, Forward/Decode Hybrid, Storage Sanitization           |
| **Prasyarat**     | M9 hijau (Port Qwen 3.6 & Gates G-M9-1..4 lolos) + M8 hijau (GDN)                                 |
| **Next**          | M11 (Multi-Core Scaling, Amdahl Curve F16, Async Double-Buffered I/O)                             |
| **Gate**          | G-M10-1..G-M10-4                                                                                  |
| **Storage Bebas** | $\approx +34\text{ GB}$ (27 GB Safetensors 1.5 + 6.9 GB quant bin dibersihkan dari storage aktif) |

---

## 1. Latar Belakang & Tujuan Utama

Milestone M10 meresmikan transformasi dari fase riset multi-model (Trial model Qwen 1.5 pada M0–M7 dan Port model Qwen 3.6 pada M9) menuju **satu engine inferensi produksi yang mandiri, bersih, dan optimal**: **`dismoen`** (**DI**sk **S**treaming **MO**e **EN**gine).

### Main Goal (Sasaran Kritis):

1. **Pembersihan Total Bobot Fisik Qwen 1.5 di Disk**:
   - Menghapus bobot legacy Safetensors `/home/will/models/qwen1.5-moe-a2.7b-chat` ($26{,}68\text{ GB}$) dan custom quant `/home/will/models/qwen1.5-moe-a2.7b-chat-4bit` ($6{,}88\text{ GB}$).
   - Membebaskan $\approx \mathbf{34\text{ GB}}$ kapasitas disk pada partisi host, menaikkan ketersediaan storage dari $\sim 54\text{ GB}$ menjadi **$\sim 88\text{ GB}$**.
   - Menjamin kapasitas storage maksimal dan aman untuk menampung bobot asli Qwen 3.6-35B ($68{,}12\text{ GB}$) beserta berkas kuantisasi runtime GGUF target ($13{,}5 - 16{,}8\text{ GB}$) tanpa risiko `ENOSPC` (_Disk Full_).
2. **Rebranding Resmi Menjadi `dismoen`**:
   - Mengganti nama binary CLI dari `kimo` menjadi `dismoen` (akronim: **DI**sk **S**treaming **MO**e **EN**gine).
   - Menyediakan symlink otomatis `kimo -> dismoen` demi kompatibilitas balik (_backwards compatibility_).
   - Memperbarui crate perkakas Rust `tools/kimo-tools` menjadi `tools/dismoen-tools`.
3. **Pembersihan Percabangan Kode Legacy Trial (`trial` codepaths)**:
   - Membuang logika kondisional `if architecture == "trial"` di config parser, scheduler, dan CLI.
   - Menghapus format custom M6 (`quant_model.bin`, 256B header JSON) demi standardisasi penuh ke format industri **GGUF v3** (`.gguf`, Q3_K, Q4_K).
4. **Penyatuan Subperintah CLI (Unifikasi Forward & Decode)**:
   - Menyatukan `forward-port` menjadi perintah standar `dismoen forward`.
   - Mengintegrasikan loop autoregressive decoding hybrid (10 Gated Attention + 30 GDN + KMSS v1 session state) langsung ke dalam `dismoen decode`.
5. **Decoupling Suite Uji Regresi dari Model 27 GB**:
   - Seluruh pengujian unit dan integrasi M0–M9 dialihkan ke fixture sintetis mini Qwen 3.6 port (< 5 MB), sehingga pengujian CI berjalan deterministik dalam hitungan detik tanpa membutuhkan model 27 GB yang dihapus.

---

## 2. Peta Pembersihan & Dekomisioning Qwen 1.5

### 2.1 Berkas Penyimpanan (Storage Sanitization)

| Lokasi Berkas / Direktori                       |       Ukuran        |    Status Tindakan M10     | Justifikasi                                                              |
| :---------------------------------------------- | :-----------------: | :------------------------: | :----------------------------------------------------------------------- |
| `/home/will/models/qwen1.5-moe-a2.7b-chat`      | $26{,}68\text{ GB}$ | **DIHAPUS / DIPENSIUNKAN** | Shard Safetensors 1.5 trial tidak lagi dipakai setelah M9 tersertifikasi |
| `/home/will/models/qwen1.5-moe-a2.7b-chat-4bit` | $6{,}88\text{ GB}$  | **DIHAPUS / DIPENSIUNKAN** | Format kuantisasi custom M6 usang; digantikan oleh GGUF v3               |
| `/home/will/models/qwen3.6-35b-a3b`             | $68{,}12\text{ GB}$ |  **DIPERTAHANKAN (SSOT)**  | Checkpoint target produksi 26 shard Safetensors BF16                     |

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

Berkas pengujian berikut yang sebelumnya menyentuh path fisik `/home/will/models/qwen1.5-moe-a2.7b-chat*` diperbarui agar membaca berkas fixture sintetis mini port (`fixtures/m9_port_mini.gguf`, `fixtures/m9_port_weights.safetensors`) atau shard riil Qwen 3.6:

- `tests/unit/test_odirect.mojo`: Target probe dialihkan ke fixture mini GGUF atau shard model 3.6.
- `tests/unit/test_lru_cache.mojo`: Integrasi probe dialihkan ke pembacaan payload terkompresi GGUF.
- `tests/integration/test_m9_w1_config_adapter.sh`: Verifikasi architecture-mismatch dialihkan ke fixture sintetis dummy, bukan path model fisik 27 GB.

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
  --model-dir /home/will/models/qwen3.6-35b-a3b \
  --tokens fixtures/m9_port_tokens.json \
  --save-session work/session1.kmss \
  --timing-profile
```

#### 2. Autoregressive Streaming Decode (`dismoen decode`)

Menjalankan decoding autoregresif berbasis kelanjutan session `KMSS v1` (zero-recompute KV cache & GDN recurrent state):

```bash
dismoen decode \
  --model-dir /home/will/models/qwen3.6-35b-a3b \
  --session work/session1.kmss \
  --max-tokens 32 \
  --output work/output_tokens.json
```

#### 3. Verifikasi Index & Format Tensor (`dismoen check-index`)

Memeriksa integritas berkas Safetensors / GGUF dan keselarasan padding I/O:

```bash
dismoen check-index /home/will/models/qwen3.6-35b-a3b/model-00001-of-00026.safetensors
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

### 4.1 Neraca Kapasitas Partisi Host ($S_{host}$)

Sebelum pembersihan (Kondisi Akhir M9):
$$S_{used} = 387\text{ GB}, \quad S_{avail} = 54\text{ GB} \quad (\text{Utilisasi } 88\%)$$

Pembersihan artefak Qwen 1.5:
$$\Delta S_{freed} = S_{safetensors}^{1.5} + S_{quant}^{1.5} = 26{,}68\text{ GB} + 6{,}88\text{ GB} = \mathbf{33{,}56\text{ GB}}$$

Kapasitas baru pasca-sanitasi M10:
$$S_{avail}^{M10} = 54\text{ GB} + 33{,}56\text{ GB} \approx \mathbf{87{,}56\text{ GB}} \quad (\text{Utilisasi turun ke } \approx 80\%)$$

### 4.2 Alokasi Ruang untuk Qwen 3.6-35B dan Target Kuantisasi

Kapasitas $\approx 88\text{ GB}$ yang tersedia menjamin keamanan operasional untuk:

1. **Model Utuh BF16 (26 Shards)**: $68{,}12\text{ GB}$ (sudah ada).
2. **Model Kuantisasi GGUF Q3_K_M Target**: $\approx 15{,}2\text{ GB}$.
3. **Headroom Operasional (Buffer OS, Scratch, Logs)**: $\approx 4{,}2\text{ GB}$.

Total kebutuhan storage terkonsolidasi:
$$S_{total\_req} = 68{,}12\text{ GB} + 15{,}2\text{ GB} = 83{,}32\text{ GB} \le S_{avail}^{M10} + S_{35B}^{current} \quad \implies \quad \mathbf{[AMAN]}$$

---

## 5. Quality Gates (Fase Konsolidasi M10)

| Gate        | Kriteria Penilaian                                                                                                                                                                                                  |                                      Ambang Batas                                       | Verifier Tool                   |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------: | :------------------------------ |
| **G-M10-1** | **Rebranding & Toolchain Integrity**: Kompilasi `dismoen` 0 compiler warning, symlink `kimo` aktif, CLI banner menampilkan nama DISMOEN                                                                             |                      Exit code 0, binary executable, symlink valid                      | `test_m10_w1_rebrand.sh`        |
| **G-M10-2** | **Storage Sanitization & Zero-Legacy**: Seluruh berkas model 1.5 dihapus dari disk, `models.lock.json` terkunci ke 3.6, seluruh tes CI bebas dependensi model 27 GB                                                 |           $\Delta S \ge 33\text{ GB}$, 0 file 1.5 tersisa di active test path           | `test_m10_w2_sanitization.sh`   |
| **G-M10-3** | **Unified Forward Parity & Decode Continuation**: `dismoen forward` bit-exact vs reference logits M9 ($\Delta_{\max} \le 10^{-7}$), `dismoen decode` menjalankan hybrid continuation dengan `recompute_tokens == 0` | $\Delta_{\max} \le 10^{-7}$, $\text{recompute} = 0$, $\text{gdn\_reused} = \text{true}$ | `test_m10_w3_forward_decode.sh` |
| **G-M10-4** | **Zero Regression & Code Hygiene**: Seluruh suite tes regresi (`validate-m9`, `validate-m8`) dan 13 hook pre-commit 100% hijau                                                                                      |                           100% PASS, 0 `# noqa`, 0 `#[allow]`                           | `test_m10_w5_gates.sh`          |

---

## 6. Rencana Gelombang Kerja (Execution Waves)

Pelaksanaan Milestone M10 dipecah menjadi 5 gelombang kerja berurutan:

### Gelombang 1 (M10-W1): Rebranding Toolchain & Executable `dismoen`

- Konfigurasi `pixi.toml` dan `Makefile` untuk mengompilasi binary `-o dismoen && ln -sf dismoen kimo`.
- Perbarui banner usage di `src/main.mojo`.
- Rename / aliaskan crate `tools/kimo-tools` menjadi `tools/dismoen-tools`.
- Verifikasi Gate G-M10-1.

### Gelombang 2 (M10-W2): Storage Sanitization & Test Fixture Decoupling

- Perbarui `test_odirect.mojo` dan `test_lru_cache.mojo` agar membaca fixture sintetis mini atau shard 3.6.
- Perbarui `test_m9_w1_config_adapter.sh` agar bebas dari direktori fisik Qwen 1.5.
- Hapus direktori `/home/will/models/qwen1.5-moe-a2.7b-chat` dan `/home/will/models/qwen1.5-moe-a2.7b-chat-4bit`.
- Verifikasi pembebasan ruang storage ($\ge 33\text{ GB}$) dan Gate G-M10-2.

### Gelombang 3 (M10-W3): Unifikasi ModelConfig & Perintah `forward`

- Satukan `parse_model_config` dan `parse_model_config_adapter` di `src/cli/config_parser.mojo`.
- Hapus enum dan percabangan `architecture == "trial"`. Jadikan Qwen 3.6 sebagai satu-satunya arsitektur default.
- Satukan implementasi `cmd_forward_port` menjadi `dismoen forward`.
- Perbarui `models.lock.json` mengunci spesifikasi Qwen 3.6-35B-A3B.

### Gelombang 4 (M10-W4): Unifikasi Hybrid `decode` & Pemensiunan Format M6

- Implementasikan loop autoregresif hybrid penuh pada `cmd_decode.mojo` (mendukung 10 Attention + 30 GDN + session continuation `KMSS v1`).
- Pensiunkan format custom M6 `quant_model.bin` dari codebase aktif.
- Tetapkan GGUF v3 sebagai satu-satunya format kuantisasi terkompresi.

### Gelombang 5 (M10-W5): Sertifikasi Quality Gates (G-M10-1..4) & Penutupan M10

- Jalankan master integration test suite `test_m10_w5_gates.sh`.
- Sertifikasi seluruh gate G-M10-1 s.d. G-M10-4.
- Terbitkan scorecard formal penutupan di `reports/YYYY-MM-DD/M10-gates-scorecard.md`.
- Milestone M10 resmi selesai $\to$ Menuju M11 (Core Scaling / Amdahl F16).

---

## 7. Definisi Selesai (DoD M10)

- [ ] Binary utama terkompilasi sebagai `dismoen` dengan symlink `kimo` aktif dan banner resmi `DISMOEN`.
- [ ] Berkas bobot fisik Qwen 1.5 ($26{,}68\text{ GB}$ Safetensors + $6{,}88\text{ GB}$ 4-bit bin) terhapus dari host storage, membebaskan $\ge 33\text{ GB}$ disk space.
- [ ] Pengujian unit `test_odirect.mojo` dan `test_lru_cache.mojo` terbebas dari path Qwen 1.5 dan lulus 100%.
- [ ] Seluruh percabangan `trial` pada `config_parser.mojo` dan `config.mojo` dibersihkan; Qwen 3.6 hybrid menjadi arsitektur default.
- [ ] `dismoen forward` terpadu lolos verifikasi paritas numerik bit-exact terhadap logits M9 ($\Delta_{\max} \le 10^{-7}$).
- [ ] `dismoen decode` mendukung decoding autoregresif 40-layer hybrid dengan session continuation `KMSS v1` (`recompute_tokens == 0`).
- [ ] `models.lock.json` diperbarui mengunci spesifikasi resmi target Qwen 3.6-35B-A3B.
- [ ] Seluruh suite pengujian regresi (`validate-m9`, `validate-m8`) dan 13 hook `pre-commit` 100% hijau tanpa suppressions (`# noqa`, `#[allow]`).
- [ ] Laporan scorecard sertifikasi M10 ter-commit di `reports/YYYY-MM-DD/M10-gates-scorecard.md`.
