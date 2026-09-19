# M4 — Full Forward 40 Layer Streaming (Qwen3.6-35B-A3B)

> Proyek: `disk-streaming-moe-engine`. Fase: **Core Streaming Pipeline** (Puncak Correctness & Memory Boundedness). Index: `../README.md`.
> Model SSOT: `Qwen3.6-35B-A3B` (40 layer hybrid: 30 GDN + 10 Gated Attention, MoE 256 routed / 8 aktif + 1 shared expert, 26 shard Safetensors BF16, 1.045 tensor, vocab 248.320, hidden 2048).

| Field       | Nilai                                                                              |
| ----------- | ---------------------------------------------------------------------------------- |
| Deliverable | Full Forward 40 layer hybrid streaming yang MATCH loose                            |
| Komponen    | CLI (`dismoen forward`), C2 full, C3 streaming pread (3D slice), C7                |
| Prasyarat   | M0–M3 hijau (Reader, Head, Gated Attention, MoE 3D slice streaming tersertifikasi) |
| Next        | `M5-kv-decode.md`                                                                  |
| Gate        | G-M4-1 (Correctness vs PyTorch Oracle), G-M4-2 (Bounded Memory $VmHWM \le 3$ GiB)  |
| Rumus       | F1, F3a, F4, F10, F14 (GDN WY scan)                                                |

---

## 1. Tujuan & Filosofi Arsitektur

Membuktikan bahwa seluruh badan transformer 40 layer hybrid benar secara numerik saat bobot di-stream layer-per-layer dari storage NVMe (`pread` $\to$ komputasi $\to$ discard seketika), tanpa terjadi penumpukan bobot di RAM, dan akumulasi error numerik FP32 tetap berada dalam batas toleransi loose.

### 1.1 Klarifikasi: Hubungan M4, M10, dan Format GGUF

> [!IMPORTANT]
> **Mengapa M4 tidak digantikan/ditarik oleh GGUF di M10?**
>
> 1. **M4 adalah Milestone Streaming Full Forward**: M4 menguji dan membuktikan bahwa _pipeline streaming layer-by-layer_ bekerja benar dari token input, embedding lookup, 40 layer transformer, hingga proyeksi LM Head.
> 2. **Safetensors BF16 adalah SSOT Ground Truth (Oracle Reference)**:
>    - Model fisik yang ada di direktori pengguna adalah `$HOME/models/qwen3.6-35b-a3b` (67 GiB, 26 shard Safetensors BF16 asli).
>    - Format GGUF v3 adalah format kompresi/kuantisasi runtime (Q3_K/Q4_K, $\sim 15\text{ GiB}$) untuk menghemat kapasitas disk pada lingkungan produksi.
>    - Setiap kuantisasi menghasilkan error aproksimasi numerik ($\varepsilon_{rel} \approx 10^{-2}\text{--}10^{-3}$). Untuk sertifikasi numerik presisi tinggi (Gate G-M4-1 loose $\varepsilon_{rel} \le 10^{-4}$ atau strict $10^{-6}$), referensi wajib dihitung terhadap bobot asli tanpa loss (Safetensors BF16).
> 3. **Streaming Bersifat Format-Agnostik**:
>    Loop macro-scheduler per layer (`pread` $\to$ mixer $\to$ MoE 3D slice $\to$ discard) yang kita bangun di M4 adalah fondasi yang sama persis baik saat membaca Safetensors maupun GGUF nantinya.

---

## 2. CLI: `dismoen forward`

Subcommand `forward` menjalankan full forward pass 40 layer hybrid dengan streaming layer weights.

### 2.1 Input

```bash
dismoen forward \
  --model-dir <DIR> \
  --tokens <PATH> \
  --output <PATH> \
  [--workdir <DIR>] \
  [--dump-routing <DIR>] \
  [--threads <N>]
```

- `--model-dir`: Direktori checkpoint (26 shard safetensors + `model.safetensors.index.json` + `config.json`).
- `--tokens`: Path ke `tokens.json` berisi array token IDs `[u32]` — dibatasi `MAX_TOKENS = 1024`, `MAX_TOKENS_FILE_BYTES = 1 MiB` (lihat Batas Input).
- `--output`: Path output logits FP32 binary (`[s, 248320]` f32 LE) — wajib berada di dalam workdir.
- `--workdir`: Direktori kerja untuk temporary files (default: `./work`).
- `--dump-routing`: Direktori output opsional untuk routing dumps Tier-1 (`routing_L<l>.json` per layer: `{"selected_experts": [[8 ID] × s]}`).
- `--threads`: Jumlah thread pekerja (default: 1 untuk eksekusi deterministik).

**Aturan output path (normatif):** `--output` dan `--layer-timing` di-resolve terhadap workdir. Path hasil resolve wajib berada di dalam workdir (path traversal seperti `..` atau symlink escape memicu error `M4_ERR_INPUT`, exit 1). File temporary ditulis secara atomic (`<dest>.tmp.<run-id>`) lalu di-rename ke target.

**Batas Input (normatif):**

1. Ukuran file `tokens.json` $\le 1\text{ MiB}$.
2. Format array integer non-negatif valid tanpa leading zeros.
3. Jumlah token $s \in [1, 1024]$.
4. Setiap token ID $< 248.320$ (kapasitas kosakata `vocab_size` Qwen3.6).

### 2.2 Output JSON

```json
{
  "status": "success",
  "run_id": "M4-20260919-001",
  "model": "qwen3.6-35b-a3b",
  "num_tokens": 16,
  "num_layers": 40,
  "logits_path": "/path/to/logits_mojo.bin",
  "metrics": {
    "walltime_sec": 45.2,
    "vmhwm_bytes": 2684354560,
    "logical_bytes_read": 5033164800,
    "physical_read_bytes": 5242880000,
    "cgroup_peak_bytes": 2800000000,
    "cgroup_oom_kills": 0,
    "phases": {
      "index_load_sec": 0.45,
      "embedding_sec": 0.18,
      "layer_forward_sec": 44.1,
      "final_norm_sec": 0.05,
      "lm_head_sec": 0.38,
      "write_sec": 0.04
    }
  }
}
```

### 2.3 Exit Codes

- `0`: Sukses, logits biner selesai ditulis dan diverifikasi.
- `1`: Error input (token out of vocab, format JSON tidak valid, workdir unescaped/unwritable).
- `2`: Error index/header validation (F15 header checksum/metadata rusak).
- `3`: Error alokasi memori (alokasi buffer gagal / melewati budget).
- `4`: Error I/O (shard corrupt, file tidak ditemukan, pembacaan `pread` gagal).
- `5`: Error komputasi layer forward (terdeteksi NaN, INF, atau overflow numerik).
- `6`: Error output (kegagalan penulisan atomic write atau rename).

---

## 3. Oracle: Full Forward Reference (PyTorch FP32)

Perkakas Oracle `tools/oracle/oracle_full.py` menjalankan full forward reference 40 layer hybrid dengan PyTorch FP32 deterministik:

### 3.1 Input

```bash
python tools/oracle/oracle_full.py \
  --model-dir $HOME/models/qwen3.6-35b-a3b \
  --tokens tools/fixtures/m4_prompt1_tokens.json \
  --output /work/logits_oracle.bin \
  [--dump-routing /work/routing]
```

### 3.2 Proses Eksekusi Oracle

1. **Streaming Shard Loader**: Membaca shard Safetensors on-demand menggunakan `safe_open` per layer, lalu melepaskan memori shard setelah layer selesai untuk mencegah OOM pada host RAM 15 GB.
2. **Embedding Lookup**: Mengambil baris vektor token ID dari `model.language_model.embed_tokens.weight` $\to$ aktivasi awal $[s, 2048]$ FP32.
3. **40 Layer Hybrid Transformer Forward**:
   - **Layer GDN** ($l \% 4 \ne 3$, 30 layer):
     - RMSNorm input (`input_layernorm.weight`).
     - Proyeksi linear attention GDN (WY scan Woodbury) $\to$ mixer output.
     - Residual 1: $x = x + y_{gdn}$.
   - **Layer Gated Attention** ($l \% 4 == 3$, 10 layer):
     - RMSNorm input (`input_layernorm.weight`).
     - Proyeksi QKV (tanpa bias, attention_bias = false).
     - QK-Norm per-head (`q_norm`, `k_norm` $[256]$).
     - Partial RoPE ($0{,}25$ factor = 64 dim rotari, 192 pass-through, $\theta = 10^7$).
     - Causal Masked MHA GQA (16 Q heads, 2 KV heads, head dim 256).
     - Output Sigmoid Gating (dari split proyeksi Q/Gate) $\to$ $o\_proj$.
     - Residual 1: $x = x + y_{attn}$.
   - **MoE Mixer** (seluruh 40 layer):
     - Post-attention RMSNorm (`post_attention_layernorm.weight`).
     - Router Softmax $\to$ Top-8 routed experts (unrenormalized weighting, routing scaling factor 9.6).
     - Evaluasi 8 expert terpilih via SwiGLU (`gate_proj`, `up_proj`, `down_proj`).
     - Evaluasi Shared Expert SwiGLU + Sigmoid gate.
     - Agregasi: $y_{moe} = \sum_{k=1}^8 p_k E_k(x) + \sigma(g_{sh}) E_{sh}(x)$.
     - Residual 2: $x = x + y_{moe}$.
4. **Final RMSNorm & LM Head**:
   - Final RMSNorm (`model.language_model.norm.weight`, $\epsilon = 10^{-6}$).
   - Matmul LM Head (`lm_head.weight` $[248320, 2048]$) $\to$ logits $[s, 248320]$ FP32 LE.
5. **Digest Checksum**: Menghasilkan SHA-256 dari berkas logits biner.

---

## 4. Manajemen Buffer Streaming & Anggaran Memori (RAM $\le 3$ GiB)

### 4.1 Kontrak Kepemilikan Layer (Layer Ownership Contract)

Untuk setiap layer $l \in [0, 39]$:

1. **Pread**: Baca hanya tensor yang dibutuhkan untuk layer $l$ dari shard Safetensors. Untuk MoE, baca hanya **8 active experts** terpilih melalui 3D slice streaming ($8 \times 3 \times (2048 \times 512) \times 4\text{ B} \approx 100\text{ MB}$ per layer), BUKAN seluruh 256 expert ($3{,}2\text{ GB}$).
2. **Forward**: Jalankan Mixer (GDN / Gated Attention) $\to$ Residual 1 $\to$ MoE $\to$ Residual 2.
3. **Discard**: Lepaskan seluruh buffer bobot layer seketika dari memori.
4. **Next**: Melangkah ke layer $l+1$ dengan buffer bersih.

### 4.2 Rincian Anggaran Memori (Memory Budget Breakdown)

Parameter: $s = 16$, $H = 2048$, $V = 248.320$, $E = 256$, $k = 8$, $I_{moe} = 512$, $I_{sh} = 512$.

| Komponen                                         | Ukuran (s=16)                                                     | Lifetime                                 |
| ------------------------------------------------ | ----------------------------------------------------------------- | ---------------------------------------- |
| LM Head resident F32 (atau streamed chunked)     | $248.320 \times 2048 \times 4\text{ B} \approx 2{,}03\text{ GiB}$ | Ekor forward                             |
| `model.language_model.norm.weight` F32           | $2048 \times 4\text{ B} = 8\text{ KiB}$                           | Seluruh forward                          |
| Hidden state $[s, H]$ F32                        | $16 \times 2048 \times 4\text{ B} = 128\text{ KiB}$               | Seluruh forward                          |
| KV Cache (10 layer GatedAttn, 2 KV heads, d=256) | $\approx 64\text{ KiB}$                                           | Seluruh forward                          |
| GDN State (30 layer GDN, $32 \times 32$ state)   | $\approx 120\text{ KiB}$                                          | Seluruh forward                          |
| Logits output $[s, V]$ F32                       | $16 \times 248.320 \times 4\text{ B} \approx 15{,}9\text{ MiB}$   | Ekor forward                             |
| Active Layer Weights (Mixer + 8 Expert Slices)   | $\approx 120\text{--}150\text{ MiB}$                              | 1 layer saja (dibuang)                   |
| Scratch Buffer (Matmul / Attention / SwiGLU)     | $\approx 10\text{--}20\text{ MiB}$                                | 1 layer saja                             |
| **Total Peak Memory Bound ($VmHWM$)**            | **$\approx 2{,}2\text{--}2{,}5\text{ GiB}$**                      | **$\le 3{,}0\text{ GiB}$ (Gate G-M4-2)** |

Puncak memori aman berada di $\approx 2{,}3\text{ GiB}$, memberikan margin lebih dari 12 GiB pada laptop RAM 15 GB dan berada jauh di bawah batas cgroup 6 GiB SEC-4.

---

## 5. Gate Evaluasi & Verifikasi

| Gate   | Kriteria              | Threshold                                                                                                                     | Metode                                                 |
| ------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| G-M4-1 | Correctness Loose     | $\Delta_{max} \le 10^{-2} \wedge \varepsilon_{rel} \le 10^{-4} \wedge \mathbb{A} \ge 99{,}9\% \wedge \cos\theta \ge 0{,}9999$ | Bandingkan logits FP32 vs PyTorch Oracle via compare   |
| G-M4-2 | Memory & Sanity Bound | $VmHWM \le 3{,}0\text{ GiB} \wedge \text{oom\_kills} == 0 \wedge \text{walltime} \le 300\text{s}$                             | Telemetri `/proc/self/status` + cgroup `memory.events` |

### 5.1 Kategori Hard-Fail F10-A

Setiap mismatch struktural memicu status FAIL keras:

1. `router-selection`: Himpunan top-8 expert terpilih berbeda dengan oracle.
2. `rope-style`: Mismatch formula rotari (misal rotari penuh alih-alih partial 0.25).
3. `qk-norm`: Tidak menerapkan rmsnorm per-head pada query/key.
4. `output-gating`: Tidak mengalikan output attention dengan sigmoid gate.
5. `argmax-mismatch`: Terjadi pergeseran greedy top-1 prediction ($\mathbb{A} < 99{,}9\%$).
6. `numeric-order`: Satu-satunya variasi numerik kecil akibat urutan asosiatif pertambahan FP32 yang diizinkan lolos di bawah threshold loose.

---

## 6. Integration Test Suite (`tests/integration/test_m4_real_qwen36.sh`)

Pengujian integrasi otomatis memverifikasi:

1. **IT-M4-1 (Happy Path Full Forward)**: Forward pass pada prompt golden menghasilkan status MATCH dan lolos Gate G-M4-1.
2. **IT-M4-2 (Bounded Memory Gate G-M4-2)**: $VmHWM \le 3{,}0\text{ GiB}$ dan $\text{oom\_kills} = 0$.
3. **IT-M4-3 (Determinisme Eksekusi)**: Run ganda dengan `--threads 1` menghasilkan SHA-256 logits yang identik bit-for-bit.
4. **IT-M4-4 (Error Input Validation)**: Token ID di luar batas kosakata ($\ge 248.320$) ditolak seketika dengan exit 1 (`M4_ERR_INPUT`).
5. **IT-M4-5 (Security Path Containment)**: Output yang mencoba escape workdir ditolak seketika dengan exit 1.
