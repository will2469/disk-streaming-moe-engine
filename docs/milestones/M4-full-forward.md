# M4 — Full Forward 24 Layer, Streaming

> Proyek: `disk-streaming-moe-engine`. Fase: **Trial** (puncak correctness). Index: `../README.md`.
> Implementasi dipecah menjadi waves: `../../scratch/wave/m4/README.md` (W1 forward-cli → W6 gates, catatan kerja gitignored).

| Field       | Nilai                                           |
| ----------- | ----------------------------------------------- |
| Deliverable | Forward 24 layer streaming yang MATCH loose     |
| Komponen    | C1 (`forward`), C2 full, C3 streaming pread, C7 |
| Prasyarat   | M0–M3 hijau                                     |
| Next        | `M5-kv-decode.md`                               |
| Gate        | G-M4-1, G-M4-2                                  |
| Rumus       | F1, F3a, F4, F10                                |

## Tujuan

Membuktikan seluruh badan transformer benar saat bobot di-stream per layer (pread → pakai → buang, RAM tidak menumpuk), dan error akumulasi masih dalam bound loose.

## CLI: `kimo forward`

Subcommand `forward` menjalankan full forward pass 24 layer dengan streaming layer weights.

### Input

```bash
kimo forward \
  --model-dir <DIR> \
  --tokens <PATH> \
  --output <PATH> \
  [--workdir <DIR>] \
  [--dump-routing <DIR>] \
  [--threads <N>]
```

- `--model-dir`: Direktori checkpoint (8 shard safetensors + index.json).
- `--tokens`: Path ke `tokens.json` berisi array token IDs `[u32]` —
  dibatasi `MAX_TOKENS = 1024`, `MAX_TOKENS_FILE_BYTES = 1 MiB` (lihat Batas input).
- `--output`: Path output logits FP32 binary — wajib di dalam workdir (lihat Aturan output path).
- `--workdir`: Direktori kerja untuk temporary files (default: `./work`).
- `--dump-routing`: Direktori output opsional untuk routing dumps Tier-1
  (`routing_L<l>.json` per layer: `{"selected_experts": [[4 ID] × s]}`); dipakai
  verdict F10-A via flag compare `--oracle-routing`/`--cand-routing`.
- `--threads`: Jumlah thread untuk layer forward (default: 1, deterministik untuk verdict).

**Aturan output path (normatif, opsi A — selaras M1):** `--output` dan `--layer-timing`
di-resolve terhadap workdir (nilai `--workdir`, default `./work`); path relatif =
relatif terhadap workdir. Path hasil resolve wajib berada di dalam workdir (cek
setelah normalisasi + resolusi symlink pada komponen parent yang sudah ada;
escape — termasuk `..` dan symlink — = error `M4_ERR_INPUT`, exit 1, sebelum
pekerjaan apa pun dimulai). File tmp atomic dibuat di directory yang sama dengan
target; rename hanya setelah write selesai. `--model-dir`/`--tokens` adalah input
(read-only) dan tidak terikat aturan ini.

**Tata letak workdir (normatif):**

```
<workdir>/
  runs/<run-id>/   ← SATU-SATUNYA area temp; run-id unik per invocation (format M4-YYYYMMDD-NNN)
  <final outputs>  ← hasil commit (--output); BUKAN temp, wajib selamat dari cleanup
```

Aturan cleanup: invocation hanya boleh menghapus objek di bawah `runs/<run-id>`
miliknya sendiri + temp `<dest>.tmp.<run-id>` miliknya. DILARANG menghapus root
workdir, run-id lain, atau final output. Berbagi workdir antar-run (termasuk
paralel) aman bila run-id unik; duplikat run-id = unsupported.

**Batas input (normatif):** embedding menghasilkan `[s, H]` dan scores `[h, s, s]` —
$s$ tak terbatas = amplifikasi memory/time sebelum inferensi normal. Berlaku
`MAX_TOKENS = 1024` (s terbesar yang bound memorinya (§ Streaming Buffer Management)
masih di bawah gate 5 GiB: peak ≈ 4,14 GiB; s lebih besar butuh re-budget, di luar M4)
dan `MAX_TOKENS_FILE_BYTES = 1 MiB` (1024 ID sebagai JSON ≈ 7 KiB; 1 MiB menghentikan
file patologis sebelum parse). Urutan validasi — SEBELUM alokasi proporsional-s:
(1) stat ukuran file ≤ 1 MiB; (2) parse sebagai array-datar-u32 (elemen non-u32 /
nested → `M4_ERR_INPUT`); (3) count 1..1024; (4) tiap ID < 151.936.

### Output JSON

```json
{
  "status": "success",
  "run_id": "M4-20250115-001",
  "model": "qwen1.5-moe-a2.7b-chat",
  "num_tokens": 16,
  "num_layers": 24,
  "logits_path": "/path/to/logits_mojo.bin",
  "metrics": {
    "walltime_sec": 287.5,
    "vmhwm_bytes": 5368709120,
    "logical_bytes_read": 28631568384,
    "physical_read_bytes": 30660512768,
    "cgroup_peak_bytes": 6100000000,
    "cgroup_oom_kills": 0,
    "phases": {
      "index_load_sec": 0.42,
      "embedding_sec": 0.15,
      "layer_forward_sec": 285.8,
      "final_norm_sec": 0.08,
      "lm_head_sec": 0.12,
      "write_sec": 0.03
    }
  }
}
```

### Exit Codes

- `0`: Sukses, logits ditulis.
- `1`: Error input (tokens tidak valid/melebihi batas, model tidak ditemukan, output escape).
- `2`: Error index validation (F15 gagal).
- `3`: Error memory (alokasi gagal di titik mana pun — lihat `M4_ERR_MEMORY`).
- `4`: Error I/O (shard corrupt, read gagal).
- `5`: Error layer forward (overflow, NaN, INF).
- `6`: Error output (gagal atomic write).

### Contoh Invokasi

```bash
# Happy path
kimo forward \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_logits.bin \
  --workdir /work

# Cgroup boundary test
systemd-run --scope -p MemoryMax=6G \
  kimo forward \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_logits.bin \
  --workdir /work
```

## Oracle: Full Forward Reference

Oracle `tools/oracle/oracle_full.py` menjalankan full forward pass reference dengan PyTorch FP32.

### Input

```bash
python tools/oracle/oracle_full.py \
  --model-dir <DIR> \
  --tokens <PATH> \
  --output <PATH> \
  [--dump-routing <DIR>]
```

- `--model-dir`: Direktori checkpoint (sama dengan Mojo).
- `--tokens`: Path ke `tokens.json` (sama dengan Mojo).
- `--output`: Path output logits FP32 binary (row-major).
- `--dump-routing`: Direktori output opsional, format sama dengan Mojo
  (`routing_L<l>.json`: `{"selected_experts": [[4 ID] × s]}`) — sisi oracle
  untuk verdict F10-A Tier-1.

### Process

1. Load model PyTorch dari safetensors (8 shard → merge).
2. Convert semua bobot ke FP32 (reference precision).
3. Embedding lookup → 24 layer forward:
   - RMSNorm (F6).
   - QKV dengan bias (q/k/v saja — lihat § Layer Weight Indexing).
   - RoPE `rotate_half` (F7).
   - Causal MHA.
   - Output projection TANPA bias (checkpoint tidak memiliki `o_proj.bias`).
   - Residual.
   - Router FP32 softmax → Top-4 (catat SET terpilih per layer bila `--dump-routing` dipakai).
   - Routed SwiGLU experts (tanpa bias).
   - Shared expert dengan sigmoid gate (tanpa bias).
   - Residual.
4. Final RMSNorm → lm_head.
5. Output logits FP32 [s, V] (row-major).
6. Compute SHA-256 dari logits.

### Output Format

```
logits_oracle.bin: [num_tokens, vocab_size] f32 row-major
oracle_full.sha256: SHA-256 hex dari logits_oracle.bin
```

### Determinism

Kontrak deterministik oracle (CPU-only, fail-closed):

- `torch.manual_seed(42)`.
- `torch.set_num_threads(1)` dan `torch.set_num_interop_threads(1)` (satu thread
  intra-op dan inter-op — `manual_seed` saja tidak mengunci pool thread).
- `torch.use_deterministic_algorithms(True)` strict (op tanpa implementasi
  deterministik wajib error, bukan fallback diam-diam).
- device `cpu` fixed; golden artifact DILARANG digenerate di GPU (hasil GPU bukan referensi).
- Baris `cudnn.deterministic` dihapus dari kontrak (cudnn tidak ada di path CPU;
  menyebutnya mengaburkan device yang di-gate).

### Verdict Contract

Pipeline presisi normatif (per ADR D3 — komputasi fp32, bobot BF16 di-dequant saat dipakai):

```
FP32 reference (oracle)
     ↓
Mojo internal compute (fp32)
     ↓
FP32 logits (`logits_mojo.bin`)  ←── correctness gate F10 (di sini)
     ↓ (opsional, bukan gate)
BF16 serialization (artifact saja)
```

Rust `compare` membandingkan `logits_mojo.bin` (**FP32**) vs `logits_oracle.bin` (**FP32**)
dengan F10 threshold M4 (loose). Kedua file wajib FP32 — `compare` hanya membaca f32
LE (`read_floats` menolak ukuran non-kelipatan 4), sehingga file BF16 tidak bisa
menjadi kandidat compare.

Rasional: BF16 hanya punya 7-bit fraction eksplisit (error pembulatan relatif per
elemen ≤ 2⁻⁸ ≈ 3,9e-3; round-trip FP32→BF16→FP32 murni pada logits terukur
ε_rel ≈ 1,7e-3 — 17× di atas threshold 1e-4 — dan Δ_max ≈ 0,01–0,12 tergantung
magnitudo). Membandingkan file BF16 langsung terhadap oracle FP32 akan mengukur
noise kuantisasi serialisasi, bukan correctness internal model, dan akan FAIL
bahkan untuk compute yang sempurna. Lihat § Logits File Format untuk toleransi
round-trip BF16 yang terpisah (bukan bagian gate).

## Fixture: M4 Golden Prompt Set

Fixture `tools/fixtures/m4_golden.json` berisi 5 prompt × 16 token untuk gate G-M4-1.

**Aturan fixture (normatif):**

```
m4_golden.json  →  generated artifact  →  SHA-256 committed
```

- Token IDs **hanya** berasal dari `generate_m4.py` + tokenizer asli Qwen — **dilarang
  hand-edit** (contoh inline di § Structure adalah sketsa ilustratif, bukan golden).
- Generator wajib menulis via `json.dump` dan memvalidasi via `json.load` round-trip
  strict (menolak integer leading-zero seperti `0123` — invalid JSON — by construction).
- Yang di-commit: `m4_golden.json` + `m4_golden.sha256`; gate memverifikasi hash
  SEBELUM memakai fixture. Hash mismatch = FAIL (menangkap edit manual apa pun).

### Structure

Sketsa struktur (ID token ilustratif — BUKAN nilai golden; lihat aturan di atas).

```json
{
  "name": "M4 golden set",
  "description": "5 prompts × 16 tokens for full forward streaming gate",
  "prompts": [
    {
      "id": "prompt1",
      "text": "What is the capital of France?",
      "tokens": [
        1234, 5678, 9012, 3456, 7890, 2345, 6789, 123, 4567, 8901, 2345, 6789, 123, 4567, 8901,
        2345
      ]
    },
    {
      "id": "prompt2",
      "text": "Explain quantum computing briefly.",
      "tokens": [
        1357, 2468, 3579, 4680, 5791, 6802, 7913, 8024, 9135, 1246, 2357, 3468, 4579, 5680, 6791,
        7802
      ]
    },
    {
      "id": "prompt3",
      "text": "Write a Python function to sort a list.",
      "tokens": [
        2468, 3579, 4680, 5791, 6802, 7913, 8024, 9135, 1246, 2357, 3468, 4579, 5680, 6791, 7802,
        8913
      ]
    },
    {
      "id": "prompt4",
      "text": "What are the primary colors?",
      "tokens": [
        3579, 4680, 5791, 6802, 7913, 8024, 9135, 1246, 2357, 3468, 4579, 5680, 6791, 7802, 8913,
        9024
      ]
    },
    {
      "id": "prompt5",
      "text": "Summarize the history of the internet.",
      "tokens": [
        4680, 5791, 6802, 7913, 8024, 9135, 1246, 2357, 3468, 4579, 5680, 6791, 7802, 8913, 9024,
        135
      ]
    }
  ]
}
```

### Generation Script

`tools/fixtures/generate_m4.py`:

1. Load full Qwen1.5-MoE tokenizer.
2. Select 5 representative prompts (beragam: factual, technical, code, basic knowledge, summary).
3. Tokenize → truncate ke 16 token.
4. Validate: semua token < 151,936 (vocab size); tepat 16 token per prompt.
5. Tulis via `json.dump`, lalu validasi via `json.load` round-trip strict (gagal =
   generator bug, bukan "perbaiki manual" — lihat aturan fixture di atas).
6. Output `m4_golden.json` + `m4_golden.sha256` + individual `tokens.json` per prompt.

### Golden Artifacts

Untuk setiap prompt:

- `tokens.json`: input token IDs.
- `logits_oracle.bin`: oracle FP32 logits.
- `oracle.sha256`: SHA-256 dari logits.
- `layer_trace.json` (opsional): per-layer intermediate states untuk debugging.

### Regression Protection

- Commit `m4_golden.json` + SHA-256 ke repo.
- Gate G-M4-1 harus PASS dengan fixture ini setiap build.
- Perubahan fixture requires approval dengan rationale.

## Error Handling

### Error Schema

```json
{
  "status": "error",
  "error": {
    "code": "M4_ERR_INDEX_VALIDATION",
    "stage": "index_load",
    "message": "F15 validation failed: shard header checksum mismatch",
    "details": {
      "shard": "model-00001-of-00008.safetensors",
      "expected_checksum": "abc123...",
      "actual_checksum": "def456..."
    }
  }
}
```

### Error Types

| Error Code             | Stage         | Description                              | Exit Code |
| ---------------------- | ------------- | ---------------------------------------- | --------- |
| `M4_ERR_INPUT`         | input         | Input tidak valid: token out of vocab/kosong/melebihi `MAX_TOKENS`/`MAX_TOKENS_FILE_BYTES`, workdir tak writable, atau output escape dari workdir | 1 |
| `M4_ERR_INDEX`         | index_load    | F15 validation gagal                     | 2         |
| `M4_ERR_MEMORY`        | input / embedding / layer_forward / final_norm / output | Alokasi checked gagal di titik mana pun (lihat Situs alokasi) | 3 |
| `M4_ERR_SHARD_IO`      | layer_forward | Shard corrupt / read gagal               | 4         |
| `M4_ERR_LAYER_FORWARD` | layer_forward | NaN/INF/overflow di layer forward        | 5         |
| `M4_ERR_OUTPUT`        | final_norm    | Gagal atomic write logits                | 6         |
| `M4_ERR_COMPARE`       | compare       | Rust compare gagal (mis. file corrupt)   | 7         |

### Stage Failure Behavior

- **Index load**: Batal seluruh forward, cleanup `runs/<run-id>` miliknya, exit 2.
- **Embedding**: Batal, cleanup `runs/<run-id>` miliknya; alokasi gagal → exit 3.
- **Layer L**: Batal pada layer L, cleanup semua buffer miliknya; NaN/INF/overflow → exit 5,
  shard read gagal → exit 4, **alokasi gagal (weights/scratch) → exit 3**.
- **Final norm/lm_head**: Batal, cleanup miliknya; compute gagal → exit 5, write gagal → exit 6,
  **alokasi gagal → exit 3**.
- **Output**: Atomic write rollback jika gagal, exit 6; **alokasi buffer/temp gagal → exit 3**.

**Situs alokasi (semua wajib checked/fallible — kegagalan → exit 3, bukan crash):**
buffer parse tokens, embedding lookup, pread buffer weights per layer, dequant chunk,
scratch attention, scratch MoE, temporaries norm, buffer logits, temp atomic-write.
Batas jujur: exit 3 hanya mencakup kegagalan *checked*; cgroup OOM-kill (SIGKILL kernel)
berada di luar kontrak — pertahanannya adalah Batas input + bound memori § sehingga
run dalam-kontrak tidak pernah menyentuh killer. VmHWM/cgroup adalah backstop
pengukuran, bukan mekanisme error.

### Atomic Rollback

- Invariant (normatif): temp file WAJIB dibuat di **parent directory destination**
  dengan nama unik per run-id (`<dest>.tmp.<run-id>`). `rename(2)` dalam satu
  directory tidak pernah EXDEV — "write temp → rename" tanpa invariant ini tidak aman
  untuk arbitrary path. Di bawah Aturan output path (dest ⊂ workdir), temp pun
  berada di workdir.
- Urutan: write temp lengkap → rename (titik komit tunggal) → hapus temp hanya bila
  rename gagal. Partial output tidak pernah dibiarkan sebagai valid.
- Di luar kontrak: durability lintas power-loss (fsync). Rename menjamin visibilitas
  atomik, bukan ketahanan daya — tidak dijanjikan di M4.

## Streaming Buffer Management

### Layer Ownership Contract

Untuk setiap layer `l` (0..23):

1. **Pread**: Baca bobot layer dari shard → buffer (BF16).
2. **Forward**: Jalankan M2 (attention) + M3 (MoE) → hidden state.
3. **Discard**: Free bobot layer buffer → tidak ada reference.
4. **Next**: Lanjut ke layer `l+1` dengan fresh buffer.

### Memory Budget Breakdown

Gate: $s = 16$, $H = 2048$, MHA 16 head ($d_h = 128$), 60 routed top-4
($I = 1408$), shared ($I_{sh} = 5632$). Komputasi fp32 per ADR D3
(scratch 4 B/elemen); bobot di disk/buffer BF16 (2 B/elemen).

| Component                        | Size (s=16)                  | Lifetime        |
| -------------------------------- | ---------------------------- | --------------- |
| Embedding + lm_head resident F32 | 2 × 151936 × 2048 × 4 B = 2,318 GiB | Seluruh forward |
| `model.norm.weight` F32          | 2048 × 4 B = 8 KiB           | Seluruh forward |
| Hidden state [s, H] F32          | 16 × 2048 × 4 B = 128 KiB    | Seluruh forward |
| Logits [s, V] F32                | 16 × 151936 × 4 B = 9,27 MiB | Ekor forward    |
| Per-layer weights (BF16)         | 1.141.121.024 B ≈ 1,063 GiB  | 1 layer saja    |
| Dequant scratch (chunked ≤64 MiB, strategi M1) | ≤ 64 MiB bound | 1 layer |
| Attention scratch F32 (QKV/scores/out, § alokasi) | 802.816 B = 784 KiB | 1 layer, fase attn |
| MoE scratch F32 (router/dispatch/SwiGLU/combine/shared, § alokasi) | 1.880.320 B ≈ 1,79 MiB | 1 layer, fase MoE |
| I/O buffers                      | 1 MiB                        | 1 layer         |
| **Total peak bound**             | **≈ 3,46 GiB**               | < 5 GiB gate (margin ≈ 1,5 GiB) |

Peak $= W_{res} + W_{layer} + \max(\text{attn}, \text{moe}) + \text{dequant} + \text{logits} + \text{io}$.
Cross-check: $24 \times W_{layer} +$ resident BF16 embed/head $+ \gamma = 28.631.568.384$ B
tepat sama dengan ukuran file — akuntansi bobot terbukti ke byte, bukan ordo.

### Buffer Lifecycle

```
Layer l:
  [weights] ← pread (BF16, ≈1.063 GiB)
  [hidden] ← input (128 KiB F32)
  [attn_scratch] ← M2 (norm + QKV + scores + out, 784 KiB)
  free/reuse [attn_scratch]  ← sebelum/saat fase MoE (kewajiban reuse, § alokasi)
  [moe_scratch] ← M3 (router + dispatch + SwiGLU + combine + shared, 1.79 MiB)
  [hidden] ← hidden + attn_out + moe_out (in-place)
  free [weights]  ← critical untuk streaming
  free [moe_scratch] ← cleanup
Layer l+1:
  repeat...
```

Tanpa KV cache: K/V dihitung ulang tiap layer dan dibuang bersama
`attn_scratch` (belum M5). Satu-satunya state persistent: embedding, lm_head,
`model.norm.weight`, hidden, logits ekor.

### No Cross-Layer Accumulation

- Tidak ada gradient accumulation (inference only).
- Tidak ada layer caching (belum M5).
- Tidak ada expert weight caching (belum M7).
- Satu-satunya persistent state: embedding, lm_head, hidden.

### Failure on Memory Leak

- Jika VmHWM > 5 GiB di gate G-M4-2 → FAIL.
- Investigasi: buffer tidak freed, double allocation, fragmentasi.
- Fix sebelum gate PASS.

## Workflow Diagram

```mermaid
flowchart TD
    A[Start: kimo forward] --> B[Load tokens.json + validasi path output ⊂ workdir]
    B --> C{Validate tokens?}
    C -->|No| ERR1[Error: M4_ERR_INPUT, exit 1]
    C -->|Yes| D[Read 8 shard HEADERS (tanpa body)]
    D --> E{F15 valid?}
    E -->|No| ERR2[Error: M4_ERR_INDEX, exit 2]
    E -->|Yes| F[Merge index]
    F --> G[Load embedding + lm_head F32 resident]
    G --> H{Memory OK?}
    H -->|No| ERR3[Error: M4_ERR_MEMORY, exit 3]
    H -->|Yes| I[Embedding lookup]
    I --> L[Initialize layer l=0]

    L --> M{All layers done?}
    M -->|Yes| W[Final RMSNorm]
    M -->|No| N[Pread layer l weights]
    N --> O{Shard IO OK?}
    O -->|No| ERR4[Error: M4_ERR_SHARD_IO, exit 4]
    O -->|Yes| P[Forward M2 Attention]
    P --> Q[Forward M3 MoE]
    Q --> R{Forward OK?}
    R -->|No| ERR5[Error: M4_ERR_LAYER_FORWARD, exit 5]
    R -->|Yes| S[Discard layer buffers]
    S --> T[Increment l = l + 1]
    T --> M

    W --> X[lm_head projection]
    X --> Y[Write logits_mojo.bin temp]
    Y --> Z{Write OK?}
    Z -->|No| ERR6[Error: M4_ERR_OUTPUT, exit 6]
    Z -->|Yes| AA[Rename atomik]
    AA --> AB[Rust compare F10]
    AB --> AC{MATCH loose?}
    AC -->|No| FAIL[FAIL: category logged]
    AC -->|Yes| AD[Success: metrics logged]
    AD --> AE[End]

    ERR1 --> END1[Cleanup runs/<run-id> miliknya]
    ERR2 --> END2[Cleanup runs/<run-id> miliknya]
    ERR3 --> END3[Cleanup runs/<run-id> miliknya]
    ERR4 --> END4[Cleanup runs/<run-id> miliknya]
    ERR5 --> END5[Cleanup runs/<run-id> miliknya]
    ERR6 --> END6[Cleanup runs/<run-id> miliknya]
    FAIL --> ENDFAIL[Cleanup runs/<run-id> miliknya]
    END1 --> ZE[End]
    END2 --> ZE
    END3 --> ZE
    END4 --> ZE
    END5 --> ZE
    END6 --> ZE
    ENDFAIL --> ZE
```

## Alur

1. Baca 8 header shard saja (nol byte body tensor) → validasi F15 → merge index.
2. Embedding + lm_head resident F32 (2,318 GiB).
3. `tokens.json` → embedding → untuk `l=0..23`: pread bobot layer, forward (attn M2 + MoE M3), buang buffer.
4. Final norm → lm_head → `logits_mojo.bin` FP32 (atomic).
5. Rust `compare` → verdict F10 + kategori FAIL.
6. Log waktu/fase, logical/physical bytes, VmHWM + memory.events untuk kalibrasi.

Bytes (logical): $B_{fwd}(s) \approx W_{file} = 28{,}63$ GB (prefill streaming, tiap layer dibaca sekali untuk semua $s$ token).

**Invariant I/O:** hingga merge index selesai, nol byte body tensor tersentuh — body
pertama kali mengalir saat pread layer. Terbukti via `logical_bytes_read` fase index
≈ jumlah byte header saja (lihat § Performance Baseline untuk definisi logical vs physical).

Propagasi error: bila tiap layer ≤ δ, bound kasar $\varepsilon_{full} \lesssim 24\delta$ — penunjuk arah saja; yang di-gate hasil ukur.

## Gate

| Gate   | Kriteria              | Threshold                                                                                                                    | Metode            |
| ------ | --------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| G-M4-1 | MATCH loose           | $\Delta_{max} \le 10^{-2} \wedge \varepsilon_{rel} \le 10^{-4} \wedge \mathbb{A} \ge 99{,}9\% \wedge \Delta_{CE} \le 0{,}02$ | 5 prompt × 16 tok |
| G-M4-2 | memori & waktu sanity | $M_{peak} \le 5$ GiB (VmHWM) $\wedge$ `oom_kill` $= 0$; selesai ≤ 5 mnt (NVMe) | VmHWM + memory.events + timer (memory.peak observability) |

Loose diizinkan hanya di M4 (akumulasi urutan penjumlahan fp32). Kategori `numeric-order` (beda kecil merata) wajar; kategori `router-selection`/`rope-style`/`bias-placement` tetap FAIL keras.

### Definisi $\mathbb{A}$ (normatif)

$$\mathbb{A} = 100 \times \frac{1}{n}\sum_{t=1}^{n} \mathbb{1}\left[\arg\max \hat{x}_t = \arg\max x_t\right]$$

yaitu persentase baris token yang argmax-nya identik (F10b `02-math-models.md` §3.3
dalam konvensi persen, sama seperti yang dihitung Rust `compare`).
Konsekuensi diskrit pada golden set ($n = 5 \times 16 = 80$): $80/80 = 100\%$ PASS,
$79/80 = 98{,}75\%$ FAIL — threshold $99{,}9\%$ di sini **efektif berarti 100\%**
(satu flip argmax = FAIL). Angka $99{,}9\%$ dipertahankan agar threshold tidak
bergantung pada $n$; ia baru non-trivial pada $n \ge 1000$.
Metrik ini dipertahankan (tidak diganti Δ/CE) karena ia satu-satunya yang menjaga
perilaku user-visible — token pilihan greedy — yang bisa flip saat margin top-2
$< \Delta_{max}$ walau Δ dan CE lolos.

### Tiga verdict F10: A / N / S (normatif)

"MATCH loose" adalah tiga verdict berurutan, bukan satu agregat. Urutan evaluasi
wajib **A → N → S** dengan short-circuit: verdict awal yang FAIL menghentikan
penilaian (kategori hard-fail tidak pernah "tertutup" metrik agregat —
operasionalisasi dari aturan keras `03-testing.md` §4.3).

| Verdict | Nama | Input | Kriteria PASS | Sifat |
|---|---|---|---|---|
| F10-A | architecture | logits FP32 + routing dumps (bila ada) + pita Δ | Bebas kategori hard-fail (tabel di bawah) | short-circuit pertama |
| F10-N | numeric | logits FP32 vs FP32 | 4 threshold loose G-M4-1 | hanya dinilai bila A PASS |
| F10-S | serialization | round-trip FP32→BF16→FP32 | toleransi § Logits File Format (bukan F10) | hanya artifact, bukan gate |

**Kategori hard-fail F10-A** (FAIL verdict A + keseluruhan, apapun nilai Δ/ε):

| Kategori | Sinyal bukti | Mekanisme bukti |
|---|---|---|
| `router-selection` | SET top-4 per token per layer ≠ oracle | Tier-1 definitif: kesetaraan SET pada routing dumps (§ Fixture/CLI `--dump-routing`); compare flag `--oracle-routing`/`--cand-routing` (order-insensitive). Tanpa dump: Tier-2 screening via pita Δ + argmax (dugaan saja → wajib re-run dengan dump) |
| `rope-style` | $0{,}05 \le \Delta_{max} \le 0{,}25$ stabil lintas prompt | pita Δ (M2) + diagnosis `rotate_half` vs interleaved |
| `bias-placement` | $0{,}25 < \Delta_{max} \le 1{,}0$ acak | pita Δ (M2) + audit 72 bias q/k/v |
| `dtype-layout` alias `tensor-mapping` | $\Delta_{max} > 1{,}0 \vee \cos\theta < 0{,}90$ | pita Δ + audit stride/transpos/mapping |
| `argmax-mismatch` | $\mathbb{A} < 99{,}9\%$ | hitung langsung (bagian gate N, diklasifikasikan di sini agar terlihat) |
| `numeric-order` | di bawah semua pita di atas | satu-satunya kategori yang boleh lolos via threshold loose |

Pita Δ dihitung Rust `compare` untuk gate G-M4-1/G-M5-1 (bukan hanya M2/M3) —
mekanisme pembuktian ini executable, bukan prosa.

## Testing

- O: full forward tiap build.
- I: end-to-end 5 prompt, exit code, schema JSON.
- B: prefill N=5 + 2 warm-up; catat $e_T$ awal untuk F4/F5.
- Tanpa KV cache: hasilkan $n$ token = ulang prefill × $n$ → motivasi M5.

## Integration Tests

### Test Matrix

| Test ID  | Scenario                         | Expected                      | Priority |
| -------- | -------------------------------- | ----------------------------- | -------- |
| IT-M4-1  | Happy path: 5 prompt × 16 token  | Exit 0, F10 PASS loose        | HIGH     |
| IT-M4-2  | Missing shard file               | Exit 4, error M4_ERR_SHARD_IO | HIGH     |
| IT-M4-3  | Corrupt shard header (F15 fail)  | Exit 2, error M4_ERR_INDEX    | HIGH     |
| IT-M4-4  | Invalid tokens (out of vocab)    | Exit 1, error M4_ERR_INPUT    | HIGH     |
| IT-M4-5  | Empty tokens array               | Exit 1, error M4_ERR_INPUT    | HIGH     |
| IT-M4-6  | Cgroup memory.max=6G boundary    | Exit 0, cgroup.peak tercatat, oom_kill == 0, VmHWM ≤ 5 GiB | HIGH     |
| IT-M4-7  | Deterministic output (threads=1) | SHA-256 match di 2 run        | MEDIUM   |
| IT-M4-8  | Workdir not writable             | Exit 1, error M4_ERR_INPUT    | MEDIUM   |
| IT-M4-9  | Model dir not readable           | Exit 1, error M4_ERR_INPUT    | MEDIUM   |
| IT-M4-10 | Layer buffer release test        | VmHWM ≤ 5 GiB, oom_kill == 0, no leak | MEDIUM   |
| IT-M4-11 | Output escape (`..`/absolut di luar workdir, termasuk via symlink) | Exit 1, error M4_ERR_INPUT, tidak ada file tertulis | HIGH |
| IT-M4-12 | Tokens melebihi batas (count > 1024 atau file > 1 MiB) | Exit 1, error M4_ERR_INPUT, tidak ada output, sebelum alokasi besar | HIGH |
| IT-M4-13 | Isolasi cleanup (2 run berbagi workdir, run-id beda) | Kedua output selamat, tidak ada orphan tmp milik run lain | MEDIUM |

### Test Automation

`tests/integration/test_m4_full_forward.sh`:

```bash
#!/bin/bash
set -e

# Setup
MODEL_DIR="/tmp/test_model"
WORKDIR="/tmp/test_work"
FIXTURE_DIR="tools/fixtures"

# IT-M4-1: Happy path
for i in {1..5}; do
  TOKENS="$FIXTURE_DIR/m4_prompt${i}_tokens.json"
  OUTPUT="$WORKDIR/prompt${i}_logits.bin"
  kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output "$OUTPUT" --workdir "$WORKDIR"
  # Compare with oracle (FP32 vs FP32, gate G-M4-1; verdict A→N→S)
  # Bila kedua sisi memakai --dump-routing: Tier-1 architecture proof per layer
  # (atribusi layer tepat untuk debugging — SET dibandingkan order-insensitive):
  # for l in $(seq 0 23); do
  #   kimo-tools compare --ref "$FIXTURE_DIR/m4_prompt${i}_oracle.bin" --cand "$OUTPUT" \
  #     --gate G-M4-1 --dim 151936 \
  #     --oracle-routing "$FIXTURE_DIR/m4_prompt${i}_routing/routing_L${l}.json" \
  #     --cand-routing "$WORKDIR/routing/routing_L${l}.json" || exit 1
  # done
  kimo-tools compare --ref "$FIXTURE_DIR/m4_prompt${i}_oracle.bin" --cand "$OUTPUT" --gate G-M4-1 --dim 151936
done

# IT-M4-2: Missing shard
mv "$MODEL_DIR/model-00002-of-00008.safetensors" "$MODEL_DIR/model-00002-of-00008.safetensors.bak"
kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output "$OUTPUT" --workdir "$WORKDIR" || true
# Expect exit 4

# IT-M4-3: Corrupt shard header
# (modify header checksum)
kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output "$OUTPUT" --workdir "$WORKDIR" || true
# Expect exit 2

# IT-M4-6: Cgroup boundary
systemd-run --scope -p MemoryMax=6G \
  kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output "$OUTPUT" --workdir "$WORKDIR"
# Expect exit 0, VmHWM ≤ 5 GiB, oom_kill == 0

# IT-M4-7: Deterministic
OUTPUT1="$WORKDIR/prompt1_run1.bin"
OUTPUT2="$WORKDIR/prompt1_run2.bin"
kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output "$OUTPUT1" --workdir "$WORKDIR" --threads 1
kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output "$OUTPUT2" --workdir "$WORKDIR" --threads 1
SHA1=$(sha256sum "$OUTPUT1" | cut -d' ' -f1)
SHA2=$(sha256sum "$OUTPUT2" | cut -d' ' -f1)
[ "$SHA1" = "$SHA2" ] || exit 1

# IT-M4-11: Output escape ditolak (aturan output path, SEC-5)
if kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output "$WORKDIR/../escape.bin" --workdir "$WORKDIR"; then
  echo "FAIL: escape via .. diterima"; exit 1
fi
if kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output /tmp/escape.bin --workdir "$WORKDIR"; then
  echo "FAIL: absolut di luar workdir diterima"; exit 1
fi
[ ! -e "$WORKDIR/../escape.bin" ] && [ ! -e /tmp/escape.bin ] || exit 1

# IT-M4-12: Tokens melebihi batas (sebelum alokasi besar)
python3 -c "import json; json.dump(list(range(1025)), open('$WORKDIR/big_tokens.json','w'))"
if kimo forward --model-dir "$MODEL_DIR" --tokens "$WORKDIR/big_tokens.json" --output "$WORKDIR/big.bin" --workdir "$WORKDIR"; then
  echo "FAIL: 1025 token diterima"; exit 1
fi
[ ! -e "$WORKDIR/big.bin" ] || exit 1

# IT-M4-13: Isolasi cleanup workdir bersama
kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output "$WORKDIR/iso_a.bin" --workdir "$WORKDIR"
kimo forward --model-dir "$MODEL_DIR" --tokens "$TOKENS" --output "$WORKDIR/iso_b.bin" --workdir "$WORKDIR"
[ -f "$WORKDIR/iso_a.bin" ] && [ -f "$WORKDIR/iso_b.bin" ] || exit 1
[ -z "$(find "$WORKDIR/runs" -mindepth 1 2>/dev/null)" ] || { echo "FAIL: orphan temp tersisa"; exit 1; }
```

### Regression Golden Outputs

- Commit `m4_prompt{1..5}_oracle.bin` + SHA-256 ke repo.
- Setiap build: jalankan IT-M4-1 → bandingkan dengan golden.
- Jika mismatch: investigasi, fix, re-commit golden dengan rationale.

### Negative Path Coverage

- Error codes 1-6 semua teruji.
- Error JSON schema valid di semua failure paths.
- Cleanup `runs/<run-id>` miliknya verified setiap error (tidak ada orphan files).

## Layer Loop Specification

### Loop Structure

```python
# Pseudocode for layer loop
hidden = embedding(tokens)  # [s, H]
for l in range(24):
    # Load layer weights
    weights = load_layer_weights(l)  # pread from shard

    # Attention (M2)
    attn_output = attention(hidden, weights)  # [s, H]

    # MoE (M3)
    moe_output = moe(hidden, weights)  # [s, H]

    # Residual
    hidden = hidden + attn_output + moe_output  # [s, H]

    # Discard weights
    free(weights)
```

### Layer Weight Indexing

Qwen1.5-MoE-A2.7B tensor naming convention (diverifikasi terhadap
`model.safetensors.index.json` live — 4.659 tensor):

- Attention weights: `model.layers.{l}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight` (4×[H,H])
- Attention biases: `model.layers.{l}.self_attn.{q_proj,k_proj,v_proj}.bias` (3×[H]) — q/k/v SAJA
- MoE router: `model.layers.{l}.mlp.gate.weight` ([E,H]) + `model.layers.{l}.mlp.shared_expert_gate.weight` ([1,H])
- Routed experts: `model.layers.{l}.mlp.experts.{e}.w1.weight`, `w2.weight`, `w3.weight` (e=0..59)
- Shared expert: `model.layers.{l}.mlp.shared_expert.{w1,w2,w3}.weight`
- Layer norms: `model.layers.{l}.input_layernorm.weight`, `post_attention_layernorm.weight`

**Kontrak bias eksplisit (normatif):** index trial memuat tepat **72 tensor bias**
= `q/k/v_proj.bias × 24 layer`. Yang TIDAK ada di checkpoint dan karenanya
DILARANG di-require atau ditambahkan oleh oracle/engine:

- `o_proj.bias` (0 tensor) — o_proj tanpa bias;
- bias MLP/expert/shared/gate dalam bentuk apa pun (0 tensor) — SwiGLU tanpa bias.

Rasional: `config.json` upstream tidak menulis field `attention_bias` sama sekali,
sehingga config bukan authority — index yang menentukan (jebakan §2.3, property
P-2). Dua kegagalan yang dicegah: (a) melewatkan q/k/v bias → correctness bug
langsung dengan signature `bias-placement` (Δ ~1e-1..1); (b) me-require
`o_proj.bias` → `WEIGHT_LOAD_FAILED` palsu pada checkpoint valid. Checksum
silang: $24 \times W_{layer} +$ resident $= 28.631.568.384$ B
(`metadata.total_size` index) tepat ke byte dengan komposisi di atas.

Shard distribution (dari index.json ter-pin `ec052fda…` — layer MENYEBRANG batas shard,
jangan asumsikan 1 layer = 1 file):

- Shard 1–7: masing-masing ~3,5 layer (≈578–681 tensor); layer 2, 6, 9, 13, 16, 20, 23 terbagi ke 2 file
- Shard 8: sisa layer 23 (5 tensor) + `lm_head.weight` + `model.embed_tokens.weight` (7 tensor total)
- `model.norm.weight` di shard 1; total 4.659 tensor, 28.631.568.384 B

### State Persistence

- `hidden`: [s, H] F32 tensor, persistent across layers (fp32 per ADR D3,
  konsisten dengan aktivasi fp32 M1/M2/M3 — bukan BF16).
- `hidden` initialized at embedding lookup.
- `hidden` updated each layer with residual connection.
- `hidden` passed to final norm after layer 23.

### Per-Layer Buffer Allocation

Gate $s = 16$. Bobot BF16 (buffer pread), scratch F32 (komputasi D3).

**Bobot per layer (total 1.141.121.024 B ≈ 1,063 GiB):**

| Buffer           | Shape            | Type | Bytes        | Lifetime     |
| ---------------- | ---------------- | ---- | ------------ | ------------ |
| `weights_attn`   | [4×H×H] + [3×H] bias (q/k/v saja, tanpa o-bias) | BF16 | 32,0 MiB + 12 KiB | Layer l only |
| `weights_router` | [E×H] + [1×H] shared gate | BF16 | 240 KiB + 4 KiB | Layer l only |
| `weights_moe`    | [60×3×I×H]       | BF16 | 990,0 MiB    | Layer l only |
| `weights_shared` | [3×I_sh×H]       | BF16 | 66,0 MiB     | Layer l only |
| `weights_norm`   | [2×H]            | BF16 | 8 KiB        | Layer l only |

**Scratch attention, fase M2 (total 802.816 B = 784 KiB):**

| Buffer          | Shape              | Type | Bytes   | Catatan                              |
| --------------- | ------------------ | ---- | ------- | ------------------------------------ |
| `norm_hidden`   | [s, H]             | F32  | 128 KiB | output RMSNorm input                 |
| `Q`, `K`, `V`   | 3 × [s, H]         | F32  | 384 KiB | QKV + bias; K/V dibuang (belum M5)   |
| `scores`        | [h, s, s] = [16,16,16] | F32 | 16 KiB | softmax **in-place** (probs reuse) |
| `attn_out`, `o_out` | 2 × [s, H]     | F32  | 256 KiB | concat head + o_proj (tanpa bias)          |

**Scratch MoE, fase M3 (total 1.880.320 B ≈ 1,79 MiB):**

| Buffer              | Shape              | Type | Bytes    | Catatan                                |
| ------------------- | ------------------ | ---- | -------- | -------------------------------------- |
| `norm_hidden_moe`   | [s, H]             | F32  | 128 KiB  | reuse buffer `norm_hidden`             |
| `router_logits`     | [s, E] = [16,60]   | F32  | 3,8 KiB  | softmax in-place (probs reuse)         |
| `topk_idx`, `topk_w`| [s,4] + [s,4]     | I32+F32 | 0,5 KiB | indices + bobot top-4               |
| `dispatch`          | [s, H]             | F32  | 128 KiB  | gather sekuensial per expert; bound = semua token → 1 expert |
| `expert_gate/up/act`| 3 × [s, I]        | F32  | 264 KiB  | SwiGLU 1 expert dalam satu waktu       |
| `expert_down`       | [s, H]             | F32  | 128 KiB  | output down-projection                 |
| `moe_combine`       | [s, H]             | F32  | 128 KiB  | akumulasi combine top-4                |
| `shared_gate/up/act`| 3 × [s, I_sh]     | F32  | 1056 KiB | SwiGLU shared expert                   |

Shared-down mengakumulasi langsung ke `moe_combine` (0 buffer tambahan).
Koreksi terhadap versi lama: `4 × 1408 × 2 B = 11 KB` salah — hilang faktor
$s = 16$ (benar: $4 \times 16 \times 1408 \times 2 = 180.224$ B $= 176$ KiB
bahkan dalam BF16) dan scratch komputasi adalah F32, bukan BF16.

**Kewajiban implementasi (bound di atas batal bila dilanggar):**

- RoPE `rotate_half` in-place; residual add in-place ke `hidden`.
- Transpos QKV/head hanya via strided view (0 buffer transpose eksplisit).
- Loop expert sekuensial (satu SwiGLU expert live dalam satu waktu);
  paralelisme antar-expert wajib re-budget $\times$ faktor paralel.
- Akumulasi matmul FP32 di register/tiling — tidak ada akumulator tensor penuh
  di luar buffer output yang terdaftar.
- Scratch fase attn di-free/reuse sebelum high-water fase MoE (tanpa reuse pun
  selisihnya hanya +784 KiB — bound gate tidak terpengaruh material).

### Residual Connection Order

Per architecture §2.4:

1. Input RMSNorm: `norm_hidden = RMSNorm(hidden)`
2. Attention: `attn_out = Attention(norm_hidden)`
3. MoE input RMSNorm: `norm_hidden_moe = RMSNorm(hidden)`
4. MoE: `moe_out = MoE(norm_hidden_moe)`
5. Residual: `hidden = hidden + attn_out + moe_out`

### Index Mapping

Index.json mapping untuk layer weights:

```json
{
  "model.layers.0.self_attn.q_proj.weight": {
    "shape": [2048, 2048],
    "dtype": "F16",
    "data_offsets": [0, 8388608],
    "file": "model-00001-of-00008.safetensors"
  },
  ...
}
```

- `data_offsets`: [BEGIN, END) relatif terhadap awal byte buffer (`data_base = 8 + header_len`, bukan absolut file); koordinat file = `data_base + offset`.
- `file`: shard identifier (1/2/3).
- Mojo implementation: parse index.json → per-layer offset map → pread exact byte range.

## Forward CLI Output Examples

### Success Output

```json
{
  "status": "success",
  "run_id": "M4-20250115-001",
  "model": "qwen1.5-moe-a2.7b-chat",
  "num_tokens": 16,
  "num_layers": 24,
  "logits_path": "/work/prompt1_logits.bin",
  "metrics": {
    "walltime_sec": 287.5,
    "vmhwm_bytes": 5368709120,
    "logical_bytes_read": 28631568384,
    "physical_read_bytes": 30660512768,
    "cgroup_peak_bytes": 6100000000,
    "cgroup_oom_kills": 0,
    "phases": {
      "index_load_sec": 0.42,
      "embedding_sec": 0.15,
      "layer_forward_sec": 285.8,
      "final_norm_sec": 0.08,
      "lm_head_sec": 0.12,
      "write_sec": 0.03
    }
  }
}
```

### Error Output (Invalid Tokens)

```json
{
  "status": "error",
  "error": {
    "code": "M4_ERR_INPUT",
    "stage": "input",
    "message": "Token ID 200000 out of vocabulary range (max: 151935)",
    "details": {
      "token_id": 200000,
      "vocab_size": 151936,
      "tokens_path": "/data/prompt1_tokens.json"
    }
  }
}
```

### Error Output (Memory Exceeded, contoh: alokasi weights layer 12 gagal)

```json
{
  "status": "error",
  "error": {
    "code": "M4_ERR_MEMORY",
    "stage": "layer_forward",
    "message": "Memory allocation failed: layer weight buffer",
    "details": {
      "layer": 12,
      "requested_bytes": 1141121024,
      "cgroup_limit_bytes": 6442450944,
      "vmhwm_bytes": 5100000000
    }
  }
}
```

### Error Output (Shard IO)

```json
{
  "status": "error",
  "error": {
    "code": "M4_ERR_SHARD_IO",
    "stage": "layer_forward",
    "message": "Failed to read shard: I/O error",
    "details": {
      "layer": 12,
      "shard": "model-00002-of-00008.safetensors",
      "offset": 1234567890,
      "size": 8388608,
      "errno": 5
    }
  }
}
```

## Logits File Format

### Binary Format

`logits_mojo.bin`: [num_tokens, vocab_size] FP32 row-major (f32 little-endian).

- **Endianness**: Little-endian (x86 default).
- **Data type**: FP32 (float32, 4 bytes per element) — sama dengan oracle dan M1/M5/M9.
- **Layout**: Row-major (row-major C order).
- **Shape**: [s, V] dengan s = num_tokens, V = 151,936.
- **Size**: s × V × 4 bytes.

### Example (s=16, V=151936)

- Total size: 16 × 151,936 × 4 = 9,723,904 bytes (~9.27 MB).
- Row 0: logits untuk token pertama [V] FP32.
- Row 1: logits untuk token kedua [V] FP32.
- ...
- Row 15: logits untuk token keenambelas [V] FP32.

### Validation

- File size harus tepat: s × V × 4 bytes.
- Semua nilai harus finite (tidak ada NaN/INF).
- Byte order valid (endianness check).
- SHA-256 untuk regression.

### Python Reader Example

```python
import numpy as np

def read_logits(path: str, num_tokens: int, vocab_size: int = 151936):
    logits = np.fromfile(path, dtype=np.float32)
    return logits.reshape(num_tokens, vocab_size)
```

### Optional BF16 Artifact (bukan gate)

Bila distribusi membutuhkan logits BF16 (hemat 2× ukuran file), serialisasi BF16
ditulis sebagai **file terpisah** (mis. `--emit-bf16 <path>` atau konversi
post-hoc dari `logits_mojo.bin` FP32) — tidak pernah menggantikan output FP32
untuk gate F10.

Uji serialisasi BF16 terpisah (FP32 → BF16 → FP32 round-trip, bukan vs oracle):

- $\varepsilon_{rel} \le 5 \times 10^{-3}$ (≈3× headroom di atas noise kuantisasi
  murni terukur ≈1,7e-3; threshold F10 1e-4 TIDAK berlaku di sini).
- $\Delta_{max} \le 0{,}15$ (scale-dependent: step BF16 pada |x| ≤ 16 adalah
  0,0625; bound ini mengasumsikan |logits| ≤ ~30 — catat max |logits| di laporan).
- $\mathbb{A}$ (argmax agreement round-trip) $\ge 99{,}9\%$.
- Decode BF16 normatif (bit-exact, terverifikasi vs `torch.bfloat16`): BF16 **bukan**
  IEEE FP16 — membaca byte BF16 sebagai `np.float16` salah secara encoding
  (terukur: 1.0 terdecode 1.875, error hingga ~1e10). Pola yang benar adalah
  zero-extend mantissa ke FP32:

```python
import numpy as np

def read_bf16_logits(path: str, num_tokens: int, vocab_size: int = 151936):
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) == num_tokens * vocab_size * 2
    u16 = np.frombuffer(data, dtype="<u2")          # BF16 bits LE
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(num_tokens, vocab_size)
```

  Prinsip bit: `[sign | exponent | 7-bit fraction]` → `[sign | exponent |
  7-bit fraction + 16-bit zero padding]`. Untuk *encode* FP32→BF16 gunakan
  `torch.bfloat16` / `ml_dtypes.bfloat16` (round-to-nearest-even), bukan truncasi
  manual. Prosedur ini di luar Rust `compare` saat ini (hanya membaca f32);
  gunakan skrip Python di atas hingga ada subcommand khusus.

## Layer-wise Timing Breakdown

### Timing Schema

Optional per-layer timing untuk debugging layer bottleneck:

```json
{
  "run_id": "M4-20250115-001",
  "layer_timing": [
    {
      "layer": 0,
      "pread_sec": 0.05,
      "attention_sec": 0.12,
      "moe_sec": 0.08,
      "total_sec": 0.25
    },
    {
      "layer": 1,
      "pread_sec": 0.04,
      "attention_sec": 0.11,
      "moe_sec": 0.09,
      "total_sec": 0.24
    },
    ...
    {
      "layer": 23,
      "pread_sec": 0.05,
      "attention_sec": 0.13,
      "moe_sec": 0.08,
      "total_sec": 0.26
    }
  ],
  "total_layer_forward_sec": 285.8
}
```

### Metrics per Layer

- `pread_sec`: waktu pread bobot layer dari shard.
- `attention_sec`: waktu M2 attention compute.
- `moe_sec`: waktu M3 MoE compute.
- `total_sec`: `pread_sec + attention_sec + moe_sec`.

### Debugging Use Cases

- Identifikasi layer dengan pread lambat (shard imbalance).
- Identifikasi layer dengan compute bottleneck (attention vs MoE).
- Verifikasi streaming: `total_sec` ≈ constant per layer (tidak ada cache effect).
- Correlate dengan shard latency pattern untuk M7 O_DIRECT tuning.

### Optional Flag

```bash
kimo forward \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_logits.bin \
  --workdir /work \
  --layer-timing /work/prompt1_layer_timing.json
```

## Security

- SEC-4: lolos `memory.max=6G` + RLIMIT_FSIZE.
- SEC-5: output hanya workdir — ditegakkan oleh Aturan output path
  (containment check + `M4_ERR_INPUT`/exit 1 untuk escape), bukan imbauan.
- R5: bila $W_{res}$ F32 menyempitkan workspace → keputusan M4+: embedding/lm_head tetap BF16 di disk + dequant on-the-fly.

## Performance Baseline

### Target

- **Prefill time**: ≤ 5 menit untuk 5 prompt × 16 token (16 token per prompt, 24 layer).
- **Memory peak**: ≤ 5 GiB (VmHWM/RSS) ∧ `oom_kill == 0` di `memory.events`.
- **Logical bytes read**: ≈ 28,63 GB per prompt ($W_{file}$ + header, cache-independent).
- **Physical read bytes**: observability saja (termasuk readahead/page-cache — bukan gate).
- **Bandwidth effective**: `logical_bytes_read / layer_forward_sec` ≥ 10 GB/s (sustained, measured per ref-perf).

### Measurement Protocol

1. **Warm-up**: 2 run dummy (kosong) untuk heat cache filesystem.
2. **Cold runs**: N=5 run dengan `sync; echo 3 > /proc/sys/vm/drop_caches` antar run.
3. **Governor**: catat `cpupower frequency-info -g` (harus `performance` atau `schedutil`).
4. **Metrics per run**:
   - Walltime: `time kimo forward ...`
   - VmHWM — GATE (RSS, cache-independent): `/proc/<pid>/status` → `VmHWM`
   - cgroup OOM — GATE: `<cgroup>/memory.events` → `oom_kill == 0` di semua run
   - cgroup peak — observability: `<cgroup>/memory.peak` dibaca setelah run
   - Logical bytes — GATE-able: engine-counter Σ panjang `pread` → `logical_bytes_read`
   - Physical bytes — observability: `/proc/<pid>/io` → `read_bytes`
   - Per-phase timing: index_load, embedding, layer_forward, final_norm, lm_head, write.

**Keputusan metrik memori (normatif):** gate = VmHWM ≤ 5 GiB ∧ `oom_kill == 0`.
`memory.peak` mencakup page cache yang ter-charge ke cgroup — di bawah streaming
pread 28 GB ia wajar melampaui VmHWM (dan `max`/throttling di `memory.events`
adalah reclaim normal, bukan kegagalan). Meng-gate `memory.peak` ≤ 5 GiB akan
FAIL spuriously; ia dicatat sebagai observability. Bound memori § dihitung atas
alokasi anonim (= RSS), sehingga VmHWM adalah cermin yang tepat. M7 (O_DIRECT,
bypass cache) meninjau ulang keputusan ini.
5. **Statistik**: report p50/p95 (headroom per [R19] Tail at Scale 2013, ε=5%).

### Expected Values (NVMe, entry-tier)

| Metric                  | Target  | Unit |
| ----------------------- | ------- | ---- |
| Walltime (p50)          | ≤ 300   | s    |
| Walltime (p95)          | ≤ 330   | s    |
| VmHWM (p50)             | ≤ 5     | GiB  |
| cgroup oom_kill         | == 0    | semua run |
| Logical bytes read (per prompt) | ≈ 28,63 | GB |
| BW effective (p50, logical/layer_forward) | ≥ 10 | GB/s |

### F4/F5 Calibration

Catat initial $e_T$ (compute intensity per token) untuk proyeksi F4/F5 di M5:

- $e_T = \frac{\text{FLOPs per token}}{\text{bytes per token}}$
- FLOPs per token ≈ 2 × parameter × s (prefill) atau 2 × parameter (decode per token).
- $I_{prefill} = e_T \cdot s$ (compute intensity untuk prefill).
- Gunakan untuk menentukan memory-bound vs compute-bound di M5 decode.

### Failure Criteria

- FAIL jika:
  - p95 walltime > 5 menit (330 s).
  - p95 VmHWM > 5 GiB.
  - `oom_kill` > 0 di run mana pun (cgroup 6G tersentuh killer — kegagalan keras,
    investigasi bound/caps sebelum gate PASS).
  - BW effective (logical) < 10 GB/s (investigasi: throttling, misconfiguration).
- Investigasi dan fix sebelum gate PASS.

## DoD

### Gate Requirements

- [ ] G-M4-1 MATCH loose PASS (Δ_max ≤ 1e-2, ε_rel ≤ 1e-4, A ≥ 99.9%, Δ_CE ≤ 0.02)
- [ ] Verdict A→N→S dieksekusi berurutan dengan short-circuit (hard-fail tak termask agregat)
- [ ] Routing dumps (`--dump-routing` kedua sisi) untuk Tier-1 `router-selection` proof per layer
- [ ] G-M4-2 memory/time sanity PASS (M_peak ≤ 5 GiB, walltime ≤ 5 menit)

### CLI Implementation

- [ ] `kimo forward` subcommand terimplementasi dengan semua argumen
- [ ] Input validation: tokens JSON format + `MAX_TOKENS`/`MAX_TOKENS_FILE_BYTES` sebelum alokasi, model dir existence, workdir writability, output containment
- [ ] Exit codes: 0 (success), 1-6 (error per stage), semuanya teruji
- [ ] Output JSON dengan run_id, metrics, phase breakdown tercommit schema

### Oracle & Fixture

- [ ] `tools/oracle/oracle_full.py` menghasilkan reference logits FP32 untuk 5 prompt
- [ ] Oracle deterministik (seed=42, threads intra+inter=1, deterministic-algorithms strict, CPU-only, FP32)
- [ ] `tools/fixtures/m4_golden.json` tercommit dengan 5 prompt × 16 token
- [ ] SHA-256 logits oracle tercommit untuk regression protection
- [ ] Generation script `tools/fixtures/generate_m4.py` teruji

### Error Handling

- [ ] Error schema JSON terimplementasi untuk semua 7 error types
- [ ] Stage failure handling: index_load, embedding, layer_forward, final_norm, output
- [ ] Atomic rollback: temp se-directory destination → rename → cleanup temp miliknya jika gagal
- [ ] Cleanup `runs/<run-id>` miliknya sebelum exit (tidak ada orphan; final output selamat)

### Streaming Buffer Management

- [ ] Layer ownership contract: pread → forward → discard per layer
- [ ] Memory budget breakdown teruji (total peak bound ≈3,46 GiB < 5 GiB, § Streaming Buffer Management)
- [ ] No cross-layer accumulation (tidak ada gradient, caching, expert weight cache)
- [ ] Buffer lifecycle: weights freed setelah layer forward
- [ ] VmHWM termonitor + `memory.events` oom_kill == 0; VmHWM > 5 GiB → FAIL di G-M4-2

### Layer Loop Implementation

- [ ] Loop 0..23 terimplementasi dengan indexing layer weights
- [ ] Per-layer pread dari shard yang benar (shard 1/2/3 distribution)
- [ ] M2 (attention) dan M3 (MoE) composition per layer
- [ ] Hidden state persistence antar layer (embedding → layer 0 → ... → layer 23 → final norm)
- [ ] Residual connections terimplementasi (pre-RMSNorm, post-attention, post-MoE)

### Numerical Correctness

- [ ] F10 verdict PASS loose untuk semua 5 prompt fixture
- [ ] Kategori FAIL: router-selection, rope-style, bias-placement tetap hard FAIL
- [ ] Kategori FAIL: numeric-order wajar (loose threshold)
- [ ] Rust `compare` menghasilkan kategori FAIL yang akurat

### Integration Tests

- [ ] Happy path: 5 prompt × 16 token → logits → compare → PASS
- [ ] Missing shard: error M4_ERR_SHARD_IO, exit 4
- [ ] Corrupt shard header: error M4_ERR_INDEX, exit 2
- [ ] Invalid tokens (out of vocab): error M4_ERR_INPUT, exit 1
- [ ] Oversize tokens (>1024 / >1 MiB): error M4_ERR_INPUT, exit 1, tanpa output (IT-M4-12)
- [ ] Output escape: error M4_ERR_INPUT, exit 1 (IT-M4-11)
- [ ] Isolasi cleanup workdir bersama: tanpa orphan, output selamat (IT-M4-13)
- [ ] Alokasi gagal di titik mana pun: error M4_ERR_MEMORY, exit 3 (checked; SIGKILL kernel di luar kontrak)
- [ ] Cgroup memory.max=6G boundary: PASS tanpa OOM
- [ ] Deterministic output: threads=1 → logits sama di 2 run (SHA-256 match)

### Security Tests

- [ ] SEC-4: cgroup memory.max=6G terpenuhi (VmHWM ≤ 5 GiB ∧ oom_kill == 0; memory.peak observability)
- [ ] SEC-4: RLIMIT_FSIZE terpenuhi (tidak ada file size overflow)
- [ ] SEC-5: output hanya ke workdir (Aturan output path; escape → M4_ERR_INPUT/exit 1, IT-M4-11)
- [ ] Model directory read-only setelah validation (tidak ada modifikasi)
- [ ] Output atomic: tidak ada partial logits valid jika gagal

### Performance Baseline

- [ ] N=5 cold runs + 2 warm-up terimplementasi
- [ ] p50/p95 walltime tercatat (≤ 300 s p50, ≤ 330 s p95)
- [ ] p50/p95 VmHWM tercatat (≤ 5 GiB) + oom_kill == 0 semua run
- [ ] Logical bytes read per prompt tercatat (≈ 28,63 GB); physical + cgroup.peak observability
- [ ] BW effective ≥ 10 GB/s (logical/layer_forward, sustained, measured per ref-perf)
- [ ] Governor tercatat (performance/schedutil)
- [ ] F4/F5 calibration $e_T$ tercatat untuk M5 projection
- [ ] Failure criteria teruji (p95 walltime, VmHWM, oom_kill, BW effective logical)

### Reporting & Artifacts

- [ ] Laporan prefill tercommit (p50/p95 walltime, VmHWM, oom_kill, logical/physical bytes, BW)
- [ ] Run ID tercatat per run (format: M4-YYYYMMDD-NNN)
- [ ] Log per-phase timing tercatat (index_load, embedding, layer_forward, final_norm, lm_head, write)
- [ ] Keputusan resident F32 vs BF16 tercatat di M4 notes
- [ ] Wave note terlink ke `../../scratch/wave/m4/README.md`

## Wave Note

Lihat implementasi notes di: <ref_file file="../../scratch/wave/m4/README.md" />
