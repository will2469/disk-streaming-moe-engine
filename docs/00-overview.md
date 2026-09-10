# 00 — Ringkasan & Ruang Lingkup

> Bagian dari `disk-streaming-moe-engine`. Index: `README.md`.
> Sumber asli: `docs/SPEC.md` v0.2 (2026-09-10).

| Field | Nilai |
|---|---|
| Versi | **v0.2 (refined)** |
| Tanggal | 2026-09-10 |
| Proyek | `disk-streaming-moe-engine` — engine inferensi MoE streaming dari nol di Mojo, pola `kimi-k3-in-c` |
| Fase | **Trial:** Qwen1.5-MoE-A2.7B-Chat (M0–M7) → **GDN:** (M8) → **Port:** Qwen3.6-35B-A3B (M9) |
| Target HW | 8 GB RAM, tanpa GPU (CPU), NVMe SSD |
| Dokumen terkait | `README.md` (cara pakai + protokol validasi 5 langkah) · `model_config.json` (ground truth dimensi) |

> **Kontrak dokumen ini:** setiap klaim "benar / beres / lolos" harus bisa direduksi menjadi (1) rumus yang menjelaskan perilakunya, (2) angka threshold, dan (3) metode pengukuran yang bisa diulang siapa pun. Rumus tanpa pengukuran = opini; pengukuran tanpa rumus = tebakan. Keduanya dilarang masuk acceptance criteria.

## 1.1 Deskripsi Proyek

`disk-streaming-moe-engine` adalah engine inferensi MoE yang dibangun dari nol di Mojo dengan pola `kimi-k3-in-c`: checkpoint diperlakukan sebagai **index tensor**, bobot dibaca dari disk hanya saat dibutuhkan (streaming), dan setiap komponen forward pass **divalidasi layer-per-layer terhadap oracle PyTorch fp32** sebelum dianggap benar. **Rust menjadi orchestration/glue**, **Mojo menjadi inference engine**, dan **Python/PyTorch berperan sebagai independent oracle/reference**. Model trial adalah Qwen1.5-MoE-A2.7B-Chat (14,3 B total; 2,7 B activated parameters menurut model card resmi; 28,63 GB BF16 di 3 shard). Untuk kalkulasi streaming dokumen ini, istilah **parameter aktif** dibedakan dari **parameter yang benar-benar di-stream per token**; yang terakhir dihitung bottom-up sebagai `N_stream ≈ 2,0668 B` untuk badan transformer yang disentuh saat decode. [R1][R2] yang sengaja dipilih karena membagi semua konsep teknis dengan target akhir Qwen3.6-35B-A3B — MoE fine-grained, router tanpa renormalisasi, shared expert, streaming per-layer — dengan biaya iterasi yang jauh lebih kecil. Dokumen ini mendefinisikan arsitektur, protokol testing, acceptance criteria per milestone, dan persyaratan security yang mengikat untuk seluruh siklus hidup proyek, dari M0 (reader) sampai M9 (port).

## 1.2 Tujuan & Non-Tujuan

**Tujuan:**

1. Membuktikan kebenaran engine: setiap jalur komputasi MATCH terhadap oracle sesuai metrik F10 (`02-math-models.md` §3.3).
2. Membangun fondasi rekayasa streaming: pread, multi-shard, buffer, dan anggaran memori (F1) yang selalu terukur.
3. Menghasilkan kalibrasi: setiap rumus di `02-math-models.md` punya pasangan angka ukur di laporan benchmark (`03-testing.md` §4.4).
4. Menyiapkan port: M8–M9 menambahkan Gated DeltaNet dan adaptasi ke Qwen3.6-35B-A3B.

**Non-tujuan (untuk fase ini):**

- Tok/s kompetitif — fase trial mengukur **kebenaran** dulu (README §3); gate performa baru masuk di M5+.
- Framework umum / multi-vendor — config diasumsikan milik Qwen1.5-MoE, lalu Qwen3.6.
- Agent orchestration / tool-calling — di luar repo ini (lihat catatan scope di `05-security.md` §6.1).
- Menggantikan llama.cpp untuk pemakaian harian — llama.cpp tetap baseline sanity (C9).

## 1.3 Prinsip Desain (non-negotiable)

1. **Oracle-first.** Tidak ada kode baru yang dianggap benar sebelum MATCH terhadap oracle fp32; urutan 5 langkah README §4 wajib dan tidak boleh dilompati.
2. **Model = index, bukan blob.** Tensor dibaca saat dibutuhkan; tidak ada asumsi "model muat di RAM".
3. **Kejujuran numerik.** Komputasi fp32; noise diukur dengan metrik multi (F10), bukan disembunyikan di balik threshold besar.
4. **Streaming sejak hari pertama.** Peak RAM punya rumus (F1) dan gate; setiap fitur baru wajib melapor VmHWM.
5. **Determinisme penuh.** Seed, fixture, dan jumlah thread untuk verdict numerik dikunci; performa diukur di mode terpisah.
6. **Security berlapis mulai dari parser.** File model dan index adalah *input tidak terpercaya* (`05-security.md`).
7. **Pembagian peran jelas.** Rust mengurus orchestration/tooling; Mojo mengurus inference engine/kernels; Python/PyTorch mengurus oracle/fixture generation. Ketiganya berkomunikasi via artefak file.

## 1.4 Definisi & Konvensi

| Simbol | Arti | Nilai trial (Qwen1.5-MoE) |
|---|---|---|
| $L$ | jumlah layer | 24 |
| $d$ | hidden size | 2048 |
| $H$ / $H_{kv}$ | head attention / head KV | 16 / 16 (MHA) |
| $d_h$ | dimensi per head | 128 |
| $N_e$ / $k$ | jumlah routed expert / top-k | 60 / 4 |
| $d_e$ / $d_{sh}$ | intermediate expert routed / shared | 1408 / 5632 |
| $V$ | vocab | 151.936 (lm_head **untied**) |
| $b$ | bytes per elemen bobot | 2 (BF16) |
| $s$ | panjang sekuens (token) | — |
| $W_{res}$ | bobot resident (embedding + lm_head F32) | ≈ **2,318 GiB** |
| $N_{act}$ | parameter aktif menurut model card | **2,7 B** |
| $N_{stream}$ | parameter badan transformer yang benar-benar di-stream saat decode | **2,0668 B** (turunan F3b) |
| $B_{tok}$ | bytes bobot yang disentuh per token | ≈ **4,134 GB** (decode BF16; turunan `N_stream`) |
| $BW_{SSD}$ / $BW_{RAM}$ | bandwidth NVMe / RAM | ≈ 3 GB/s / ≈ 15 GB/s (GB desimal) |
| $\rho$, $C_{pc}$, $W_{stream}$ | hit-ratio page cache, RAM utk page cache, bagian file yang di-stream | **≈ 0,1154 · 3 GB · 26 GB** (estimasi) |
| $M_{peak}$ | RSS puncak proses (F1) | gate ≤ **5 GiB** (trial) |
| $\varepsilon_{rel}$, $\Delta_{max}$ | error relatif L2, selisih absolut maksimum (F10) | threshold per gate |
| TBM | to be measured — angka final diisi saat pengukuran | — |

Konvensi satuan: **1 GB = 10⁹ byte** untuk ukuran file dan bandwidth; **1 GiB = 2³⁰ byte** untuk anggaran RAM; **1 MiB = 2²⁰ byte** untuk ukuran KV per token. "MATCH" didefinisikan di `03-testing.md` §4.3. Semua rumus diberi ID `F#` dan dirujuk oleh gate di `04-quality.md`. Angka `W_stream`, `BW`, dan `T_comp` yang belum berasal dari pengukuran nyata diberi label *estimasi* dan tidak boleh dipakai sebagai bukti acceptance.

## 1.5 Milestone & Fase

| Milestone | Deliverable | Fase | File spec | Gate |
|---|---|---|---|---|
| M0 | Reader safetensors multi-shard (index → 4.659 tensor) | Trial | `milestones/M0-reader.md` | G-M0-* |
| M1 | Head path: embed → final norm → lm_head | Trial | `milestones/M1-head-path.md` | G-M1-* |
| M2 | Satu layer: attention (rope, qkv bias, MHA) | Trial | `milestones/M2-attention.md` | G-M2-* |
| M3 | Satu layer: MoE (router top-4, shared sigmoid gate) | Trial | `milestones/M3-moe.md` | G-M3-* |
| M4 | Full forward 24 layer, streaming | Trial | `milestones/M4-full-forward.md` | G-M4-* |
| M5 | KV cache + decode incremental | Trial | `milestones/M5-kv-decode.md` | G-M5-* |
| M6 | Quantizer 4-bit buatan sendiri + dequant kernel | Trial | `milestones/M6-quantizer.md` | G-M6-* |
| M7 | O_DIRECT + LRU cache expert (pola kimi-k3-in-c) | Trial | `milestones/M7-odirect-lru.md` | G-M7-* |
| M8 | Gated DeltaNet chunked scan (oracle = naive loop) | GDN | `milestones/M8-gdn.md` | G-M8-* |
| M9 | Port ke Qwen3.6-35B-A3B (GQA, gated attention, vocab 248K) | Port | `milestones/M9-port.md` | G-M9-* |

Detail gate per milestone ada di `04-quality.md` dan diulang di tiap file milestone. DoD per fase ada di `04-quality.md` §5.4.
