# M8 — Gated DeltaNet Chunked Scan (Oracle = Naive Loop)

> Proyek: `disk-streaming-moe-engine`. Fase: **GDN**. Index: `../README.md`.
> Implementasi dipecah menjadi waves: `../../scratch/wave/m8/README.md` (W1 naive-oracle → W6 gates, catatan kerja gitignored).

| Field       | Nilai                                                                 |
| ----------- | --------------------------------------------------------------------- |
| Deliverable | Implementasi Mojo chunked scan Gated DeltaNet yang MATCH naive oracle |
| Komponen    | C2 ekstensi (GDN kernels), C7                                         |
| Prasyarat   | M7 hijau (trial stabil)                                               |
| Next        | `M9-port.md`                                                          |
| Gate        | G-M8-1..G-M8-3                                                        |
| Rumus       | F14, F10, kontras F2                                                  |

## Tujuan

Menyiapkan linear attention untuk port Qwen3.6 (30/40 layer GDN): state ukuran tetap, bukan tumbuh dengan $s$ seperti KV.

## Rumus (F14) [R9]

$$S_t = \gamma_t\, S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top,\quad S\in\mathbb{R}^{d_k\times d_v} \tag{F14}$$

- Tanpa gate: $\gamma_t=1$.
- Oracle = **loop rekuren naive** Python fp32 (bukan model hybrid publik — R6).
- Implementasi Mojo = chunked scan (paralel per blok, representasi WY) wajib MATCH strict vs naive.
- Sifat kunci: $|S|$ konstan terhadap $s$ → kontras F2 ($M_{KV}$ tumbuh linear).

## Gate

| Gate   | Kriteria                | Threshold                           | Metode              |
| ------ | ----------------------- | ----------------------------------- | ------------------- |
| G-M8-1 | chunked == naive oracle | $\Delta_{max} \le 10^{-3}$ (strict) | 100 sekuens acak    |
| G-M8-2 | state fixed-size vs $s$ | $M_{state}=H·d_k·d_v·b$ konstan     | sampler s∈{1K..32K} |
| G-M8-3 | throughput sanity       | chunked ≥ 2× naive                  | timer               |

## CLI Contract

### Command: `kimo gdn`

```bash
kimo gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 \
  --dk 128 \
  --dv 128 \
  --chunk-size 512 \
  --workdir ./work \
  --threads 1 \
  --seed 42
```

### Arguments

| Argument       | Type | Default  | Description                                          |
| -------------- | ---- | -------- | ---------------------------------------------------- |
| `--model-dir`  | path | required | Direktori model dengan safetensors shard             |
| `--tokens`     | path | required | Path ke file JSON dengan input token IDs             |
| `--output`     | path | required | Path output untuk state final (binary)               |
| `--layers`     | int  | 30       | Jumlah layer GDN (untuk port M9: 30)                 |
| `--dk`         | int  | 128      | Dimensi key $d_k$ (trial/port: sesuai config)        |
| `--dv`         | int  | 128      | Dimensi value $d_v$ (trial/port: sesuai config)      |
| `--chunk-size` | int  | 512      | Jumlah token per chunk untuk chunked scan            |
| `--workdir`    | path | `./work` | Direktori kerja untuk temporary files                |
| `--threads`    | int  | 1        | Jumlah thread (default 1 untuk determinisme verdict) |
| `--seed`       | int  | 42       | Random seed untuk initialization (jika random init)  |

### Input JSON

```json
{
  "tokens": [12345, 67890, 23456, 78901],
  "seq_len": 4
}
```

### Output JSON

```json
{
  "status": "success",
  "run_id": "M8-20250115-001",
  "model": "qwen1.5-moe-a2.7b-chat",
  "layers": 30,
  "dk": 128,
  "dv": 128,
  "chunk_size": 512,
  "seq_len": 4,
  "state_path": "/work/prompt1_state.bin",
  "state_shape": [30, 128, 128],
  "state_dtype": "float32",
  "metrics": {
    "walltime_sec": 0.15,
    "naive_time_sec": 0.32,
    "speedup": 2.13,
    "vmhwm_bytes": 1073741824,
    "peak_state_bytes": 589824
  }
}
```

### Exit Codes

- `0`: Sukses, state ditulis.
- `1`: Error input (tokens tidak valid, model tidak ditemukan).
- `2`: Error config ($d_k, d_v$, layers tidak valid).
- `3`: Error memory (melebihi cgroup 6G, alloc gagal).
- `4`: Error I/O (shard corrupt, read gagal).
- `5`: Error GDN forward (NaN/INF/overflow di state evolution).
- `6`: Error output (gagal atomic write state).
- `7`: Error chunk size (tidak habis membagi seq_len, atau chunk_size <= 0).

### Contoh Invokasi

```bash
# Happy path: 4 token, 30 layers, chunk size 512
kimo gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 \
  --dk 128 \
  --dv 128 \
  --chunk-size 512

# Cgroup boundary test
systemd-run --scope -p MemoryMax=6G \
  kimo gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 \
  --dk 128 \
  --dv 128 \
  --chunk-size 512

# Long sequence (8K tokens) untuk bukti state fixed-size
kimo gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt_long_tokens.json \
  --output /work/prompt_long_state.bin \
  --layers 30 \
  --dk 128 \
  --dv 128 \
  --chunk-size 512
```

### Additional CLI Output Examples

#### Example 1: Error - Invalid Config

```json
{
  "status": "error",
  "error_code": 2,
  "error_type": "CONFIG_INVALID",
  "message": "Invalid config: dk must be positive (got: -1)"
}
```

#### Example 2: Error - Memory Allocation Failure

```json
{
  "status": "error",
  "error_code": 3,
  "error_type": "MEMORY_ALLOC_FAILURE",
  "message": "Memory allocation failed: requested 10737418240 bytes, cgroup limit 6G"
}
```

#### Example 3: Error - NaN/INF in State

```json
{
  "status": "error",
  "error_code": 5,
  "error_type": "GDN_FORWARD_ERROR",
  "message": "GDN forward error: NaN/INF detected at chunk 2, token range [1024, 1536)"
}
```

#### Example 4: Success with Performance Details

```json
{
  "status": "success",
  "run_id": "M8-20250115-001",
  "model": "qwen1.5-moe-a2.7b-chat",
  "layers": 30,
  "dk": 128,
  "dv": 128,
  "chunk_size": 512,
  "seq_len": 1024,
  "state_path": "/work/prompt1_state.bin",
  "state_shape": [30, 128, 128],
  "state_dtype": "float32",
  "metrics": {
    "walltime_sec": 12.3,
    "naive_time_sec": 32.5,
    "speedup": 2.64,
    "vmhwm_bytes": 1073741824,
    "peak_state_bytes": 1966080,
    "phases": {
      "load_weights_sec": 0.42,
      "init_state_sec": 0.01,
      "chunked_scan_sec": 11.8,
      "serialize_state_sec": 0.05,
      "write_output_sec": 0.02
    }
  }
}
```

#### Example 5: Success with State Continuation

```json
{
  "status": "success",
  "run_id": "M8-20250115-002",
  "model": "qwen1.5-moe-a2.7b-chat",
  "layers": 30,
  "dk": 128,
  "dv": 128,
  "chunk_size": 512,
  "seq_len": 512,
  "state_input_path": "/work/seq1_state.bin",
  "state_path": "/work/seq2_state.bin",
  "state_shape": [30, 128, 128],
  "state_dtype": "float32",
  "metrics": {
    "walltime_sec": 6.1,
    "naive_time_sec": 16.2,
    "speedup": 2.66,
    "vmhwm_bytes": 1073741824,
    "peak_state_bytes": 1966080
  }
}
```

## Workflow Diagram

```mermaid
flowchart TD
    A[Input: tokens.json] --> B[Validate Input]
    B --> C{Valid?}
    C -->|No| D[Error: Exit 1]
    C -->|Yes| E[Load Model Weights]
    E --> F[Initialize State S0 = 0]
    F --> G[Split into Chunks]
    G --> H{Chunk Processing Loop}
    H --> I[Compute WY Coefficients]
    I --> J[Apply WY Update to State]
    J --> K{NaN/INF Check}
    K -->|Yes| L[Error: Exit 5]
    K -->|No| M{More Chunks?}
    M -->|Yes| H
    M -->|No| N[Serialize State to Binary]
    N --> O[Atomic Write to Output]
    O --> P{Write Success?}
    P -->|No| Q[Error: Exit 6]
    P -->|Yes| R[Output JSON + State Path]
    R --> S[Success: Exit 0]

    subgraph Oracle Naive Loop
        T[Input: tokens.json] --> U[Validate Input]
        U --> V[Initialize State S0 = 0]
        V --> W[Token Loop t=0..seq_len-1]
        W --> X[Compute kt, vt, γt, βt]
        X --> Y[Apply F14: St = γt·St-1·(I-βt·kt·ktᵀ) + βt·vt·ktᵀ]
        Y --> Z{More Tokens?}
        Z -->|Yes| W
        Z -->|No| AA[Serialize State]
        AA --> AB[Compare with Chunked]
    end

    style R fill:#90EE90
    style D fill:#FFB6C1
    style L fill:#FFB6C1
    style Q fill:#FFB6C1
    style AB fill:#87CEEB
```

### Workflow Steps

1. **Input Validation**: Cek tokens JSON format, seq_len > 0.
2. **Model Loading**: Load weights dari safetensors (validasi F15).
3. **State Initialization**: $S_0 = \mathbf{0}$ (zero init untuk semua layer).
4. **Chunk Splitting**: Bagi tokens menjadi chunks dengan size `chunk_size`.
5. **Chunk Processing**:
   - Compute WY coefficients untuk chunk.
   - Apply WY update ke state dalam satu operasi chunked.
   - Cek NaN/INF setelah setiap chunk.
6. **State Serialization**: Tulis state ke binary raw float32 (row-major).
7. **Atomic Write**: Write ke temp file lalu rename.
8. **Oracle Comparison**: Bandingkan state chunked vs naive loop dengan F10 metrics.

### Parallel Execution Points

- **Within chunk**: Operasi WY dapat diparalelkan dengan SIMD/vectorized.
- **Between chunks**: Barrier synchronization untuk state consistency.

## Oracle: Naive Loop Reference

Oracle `tools/oracle/oracle_gdn.py` menjalankan loop rekuren naive Python FP32 untuk Gated DeltaNet.

### Input

```bash
python tools/oracle/oracle_gdn.py \
  --tokens /data/prompt1_tokens.json \
  --layers 30 \
  --dk 128 \
  --dv 128 \
  --output /work/prompt1_state_naive.bin \
  --seed 42
```

### Algoritma Naive

```python
# Pseudocode untuk naive loop (FP32)
S = zeros(dk, dv)  # state initial
for t in range(seq_len):
    kt = compute_k(t)
    vt = compute_v(t)
    gamma_t = compute_gamma(t)
    beta_t = compute_beta(t)

    # F14: Delta rule
    S = gamma_t * S @ (I - beta_t * outer(kt, kt)) + beta_t * outer(vt, kt)

save_state(S, output_path)
```

### Output Binary Format

State disimpan sebagai binary raw float32 (row-major):

- Shape: `[layers, dk, dv]`
- Total bytes: `layers * dk * dv * 4`
- Contoh untuk 30 layers, dk=128, dv=128: `30 * 128 * 128 * 4 = 1,966,080 bytes`

### Compare Contract

Rust `compare` tool membaca dua state binary (Mojo chunked vs Oracle naive):

```bash
kimo compare \
  --reference /work/prompt1_state_naive.bin \
  --candidate /work/prompt1_state.bin \
  --tolerance 1e-3 \
  --output /work/compare_report.json
```

### Compare Output JSON

```json
{
  "status": "MATCH",
  "run_id": "M8-20250115-001",
  "reference_path": "/work/prompt1_state_naive.bin",
  "candidate_path": "/work/prompt1_state.bin",
  "metrics": {
    "delta_max": 8.2e-4,
    "epsilon_rel": 3.1e-5,
    "cos_theta": 0.9999998,
    "agreement": 100.0,
    "delta_ce": 0.0
  },
  "verdict": "PASS",
  "threshold": "delta_max <= 1e-3"
}
```

### Compare Exit Codes

- `0`: MATCH (semua threshold terpenuhi)
- `1`: FAIL (threshold tidak terpenuhi)
- `2`: ERROR (shape mismatch, file tidak ditemukan)

## Testing

- O: strict F10 vs naive, tiap build.
- B: scaling $s$ 1K..32K (bukti konstan) + speedup.
- R6: bila model hybrid kecil publik muncul → opsional, bukan prasyarat.

## M8-Specific Fixture

### Synthetic Mini GDN Config

Untuk testing CI tanpa download 28 GB, gunakan config synthetic:

| Parameter       | Value                |
| --------------- | -------------------- |
| Layers          | 2 (mini untuk cepat) |
| $d_k$           | 32                   |
| $d_v$           | 32                   |
| Vocab           | 512                  |
| Sequence length | 16 (untuk prefill)   |
| Chunk size      | 8                    |
| Seed            | 42 (deterministik)   |

### Fixture Path

- Tokens: `fixtures/m8_tokens.json`
- Weights GDN: `fixtures/m8_gdn_weights.safetensors` (synthetic, seed 42)
- Oracle state: `fixtures/m8_state_naive.bin` (precomputed)

### Fixture Tokens JSON

```json
{
  "tokens": [1, 23, 45, 67, 89, 101, 123, 145, 167, 189, 201, 223, 245, 267, 289, 311],
  "seq_len": 16
}
```

### Fixture Generation

```bash
# Generate synthetic weights
python tools/oracle/generate_gdn_fixture.py \
  --layers 2 \
  --dk 32 \
  --dv 32 \
  --vocab 512 \
  --seed 42 \
  --output fixtures/m8_gdn_weights.safetensors

# Generate oracle state (naive loop)
python tools/oracle/oracle_gdn.py \
  --tokens fixtures/m8_tokens.json \
  --layers 2 \
  --dk 32 \
  --dv 32 \
  --weights fixtures/m8_gdn_weights.safetensors \
  --output fixtures/m8_state_naive.bin \
  --seed 42
```

### Expected Memory Usage

- State size: `2 * 32 * 32 * 4 = 8,192 bytes` (FP32)
- Total memory footprint: < 10 MB (cocok untuk CI 8 GB)

### Integration dengan CI

```bash
# Test fixture synthetic (tanpa model asli)
make validate-m8
```

Target: G-M8-1 (chunked == naive) lulus dengan fixture synthetic sebelum testing dengan model 28 GB.

## Integration Tests

### Integration dengan M7 (O_DIRECT + LRU)

**Test objective**: Verifikasi GDN state processing berjalan dengan O_DIRECT reader dan LRU cache expert.

**Test setup**:

- Model trial Qwen1.5-MoE dengan safetensors 8 shards.
- O_DIRECT reader aktif (M7).
- LRU cache expert aktif (M7).
- GDN layer diproses dengan chunked scan.

**Test command**:

```bash
# Full pipeline: M7 reader + M8 GDN
kimo gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 --dk 128 --dv 128 \
  --use-odirect \
  --lru-capacity 100
```

**Verification**:

- G-M8-1 lulus (chunked == naive).
- O_DIRECT alignment verified (M7 gates).
- LRU hit rate measured (M7 gates).
- Memory peak ≤ 6G (SEC-4).

### Integration dengan M9 (Port Qwen3.6)

**Test objective**: Verifikasi 30 GDN layers + 10 Gated Attention layers berjalan dalam pipeline M9.

**Test setup**:

- Model port Qwen3.6-35B-A3B (checkpoint resmi [R4][R5]).
- 10 Gated Attention layers (M9).
- 30 GDN layers (M8).
- GQA 16Q/2KV (M9).

**Test command**:

```bash
# Full M9 pipeline: 10 GatedAttn + 30 GDN
kimo forward-port \
  --model-dir /models/qwen3.6-35b \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_logits.bin \
  --architecture qwen3.6
```

**Verification**:

- G-M9-1 lulus (oracle layer-by-layer).
- G-M9-2 lulus (full forward + memory peak ≤ 7.5 GiB).
- G-M9-3 lulus (decode streaming ≥ 0.5 tok/s).
- G-M9-4 lulus (KV GQA sesuai rumus F2).
- M8 gates (G-M8-1..3) lulus sebagai sub-component.

### State Continuation Test

**Test objective**: Verifikasi state dapat di-load dan dilanjutkan untuk continuation.

**Test setup**:

- Prefill sequence 1 → save state.
- Load state dari sequence 1 → process sequence 2.
- Bandingkan dengan single-pass sequence 1+2.

**Test command**:

```bash
# Part 1: Prefill sequence 1
kimo gdn \
  --tokens /data/seq1_tokens.json \
  --output /work/seq1_state.bin \
  --layers 30 --dk 128 --dv 128

# Part 2: Continue from seq1
kimo gdn \
  --tokens /data/seq2_tokens.json \
  --state-input /work/seq1_state.bin \
  --output /work/seq2_state.bin \
  --layers 30 --dk 128 --dv 128

# Part 3: Single-pass baseline
kimo gdn \
  --tokens /data/seq1_seq2_combined.json \
  --output /work/combined_state.bin \
  --layers 30 --dk 128 --dv 128

# Compare
kimo compare \
  --reference /work/combined_state.bin \
  --candidate /work/seq2_state.bin \
  --tolerance 1e-3
```

**Verification**:

- Continuation state == single-pass state (Δ_max ≤ 1e-3).
- State reset berfungsi untuk independent sequences.

### Long-Sequence Stability Test

**Test objective**: Verifikasi numerical stability untuk $s$ ∈ {1K, 2K, 4K, 8K, 16K, 32K}.

**Test command**:

```bash
for s in 1024 2048 4096 8192 16384 32768; do
  kimo gdn \
    --tokens /data/seq_${s}_tokens.json \
    --output /work/seq_${s}_state.bin \
    --layers 30 --dk 128 --dv 128

  python tools/oracle/oracle_gdn.py \
    --tokens /data/seq_${s}_tokens.json \
    --output /work/seq_${s}_state_naive.bin \
    --layers 30 --dk 128 --dv 128

  kimo compare \
    --reference /work/seq_${s}_state_naive.bin \
    --candidate /work/seq_${s}_state.bin \
    --tolerance 1e-3
done
```

**Verification**:

- G-M8-2 lulus (state size konstan terhadap $s$).
- Δ_max ≤ 1e-3 untuk semua $s$.
- Tidak ada NaN/INF untuk $s$ besar.

## Performance Baseline

### Protocol

Mengikuti `docs/03-testing.md` §4.4:

1. **Environment terkunci**:
   - AC/plug-in stabil, CPU governor `performance`.
   - Aplikasi lain ditutup.
   - Kondisi dicatat di header laporan.

2. **Cold read**:

   ```bash
   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```

   Sebelum run cold.

3. **Warm-up**:
   - 2× warm-up (tidak dihitung).
   - Lalu N run terukur: N=10 untuk GDN (lightweight vs M4/M5).

4. **Sampling**:
   - RSS via VmHWM + poller 100 ms.
   - Bytes I/O via `/proc/<pid>/io`.
   - Waktu per fase dari log engine.

5. **Output**:
   - CSV per run + laporan markdown (p50/p95, min/max).
   - Dengan **run-id**.
   - Disimpan `reports/YYYY-MM-DD/`.

### Run ID Format

```
M8-YYYYMMDD-NNN
```

Contoh: `M8-20250115-001`

### Performance Metrics

| Metric             | Description                | Target                 |
| ------------------ | -------------------------- | ---------------------- |
| `walltime_sec`     | Total wall clock time      | TBM (diukur)           |
| `naive_time_sec`   | Naive loop time (baseline) | TBM (diukur)           |
| `speedup`          | `naive_time / walltime`    | ≥ 2× (G-M8-3)          |
| `vmhwm_bytes`      | Peak memory (VmHWM)        | ≤ 6G (SEC-4)           |
| `peak_state_bytes` | Peak state memory          | `layers * dk * dv * 4` |
| `tokens_per_sec`   | Throughput                 | TBM (diukur)           |

### Expected Performance (Trial, Entry-Tier NVMe)

**Estimasi kasar** (TBM, perlu diukur):

| Config            | seq_len | chunk_size | Naive time | Chunked time | Speedup |
| ----------------- | ------- | ---------- | ---------- | ------------ | ------- |
| Mini (2×32×32)    | 16      | 8          | ~0.5 ms    | ~0.2 ms      | ~2.5×   |
| Port (30×128×128) | 1K      | 512        | ~32 ms     | ~12 ms       | ~2.7×   |
| Port (30×128×128) | 8K      | 512        | ~256 ms    | ~96 ms       | ~2.7×   |

Catatan: Angka di atas adalah estimasi kasar. Angka aktual harus diukur dan dilaporkan dengan run-id.

### p50/p95 Reporting

Laporan performance harus menyertakan:

```markdown
## Performance Report: M8-20250115-001

| Metric         | p50        | p95        | min        | max        |
| -------------- | ---------- | ---------- | ---------- | ---------- |
| walltime_sec   | 12.3       | 13.1       | 11.8       | 14.2       |
| naive_time_sec | 32.5       | 34.2       | 31.0       | 36.8       |
| speedup        | 2.64       | 2.61       | 2.63       | 2.59       |
| vmhwm_bytes    | 1073741824 | 1073741824 | 1073741824 | 1073741824 |

Environment:

- CPU governor: performance
- C_max: terdeteksi run-time (tanpa angka absolut di spec)
- c: 1 (verdict), c* (performance)
- Device: NVMe entry-tier
```

### Governor Logging

Wajib log governor sebelum run:

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Jika governor bukan `performance`, angka performance tidak valid untuk baseline.

### Chunk Size Sweep

Untuk G-M8-3, sweep chunk size untuk menemukan optimal:

```bash
for cs in 64 128 256 512 1024; do
  for i in {1..10}; do
    kimo gdn \
      --tokens /data/prompt1_tokens.json \
      --output /work/prompt1_state.bin \
      --layers 30 --dk 128 --dv 128 \
      --chunk-size $cs \
      --run-id M8-20250115-chunk${cs}-run${i}
  done
done
```

Analisis p50/p95 untuk setiap chunk size, pilih optimal untuk default.

## Per-Token Timing Breakdown

### Naive Loop Timing

Naive loop memproses satu token per iterasi:

```
Time per token (naive) = T_compute_k + T_compute_v + T_compute_gamma + T_compute_beta + T_delta_rule
```

Breakdown estimasi (TBM, diukur):

| Component           | Description              | Estimated Time (microseconds) |
| ------------------- | ------------------------ | ----------------------------- |
| T_compute_k         | Compute key projection   | ~5 μs                         |
| T_compute_v         | Compute value projection | ~5 μs                         |
| T_compute_gamma     | Compute gate γ_t         | ~2 μs                         |
| T_compute_beta      | Compute gate β_t         | ~2 μs                         |
| T_delta_rule        | Apply F14 matrix update  | ~15 μs                        |
| **Total per token** |                          | **~29 μs**                    |

Total untuk 1K tokens: ~29 ms (consistent dengan naive_time_sec ~32.5 ms di contoh).

### Chunked Scan Timing

Chunked scan memproses chunk tokens secara paralel:

```
Time per chunk (chunked) = T_wy_coeff + T_wy_update + T_sync
```

Breakdown estimasi (TBM, diukur):

| Component                        | Description                       | Estimated Time (microseconds) |
| -------------------------------- | --------------------------------- | ----------------------------- |
| T_wy_coeff                       | Compute WY coefficients for chunk | ~40 μs                        |
| T_wy_update                      | Apply WY update to state          | ~80 μs                        |
| T_sync                           | Barrier synchronization           | ~10 μs                        |
| **Total per chunk (512 tokens)** |                                   | **~130 μs**                   |

Per-token equivalent: ~0.25 μs/token (dengan chunk size 512).

Total untuk 1K tokens (2 chunks): ~260 μs (consistent dengan chunked_time_sec ~12.3 ms? Wait, ini tidak match — estimasi perlu revisi setelah pengukuran aktual).

### Speedup Analysis

Theoretical speedup untuk chunk size C:

```
Speedup = (seq_len * T_naive_per_token) / (num_chunks * T_chunked_per_chunk)
```

Dengan estimasi di atas:

- seq_len = 1024
- T_naive_per_token = 29 μs
- num_chunks = 2 (chunk_size = 512)
- T_chunked_per_chunk = 130 μs

Speedup = (1024 _ 29) / (2 _ 130) = 29,696 / 260 ≈ 114×

Ini adalah speedup teoretis maksimal. Speedup aktual mungkin lebih rendah karena:

- Overhead chunking
- Barrier synchronization
- Memory bandwidth bound (bukan compute-bound)

Speedup aktual yang diukur di contoh: ~2.64× (bukan 114×), menunjukkan implementasi terikat memory bandwidth, bukan compute.

### Per-Phase Timing

Laporan output JSON menyertakan breakdown per fase:

```json
"phases": {
  "load_weights_sec": 0.42,
  "init_state_sec": 0.01,
  "chunked_scan_sec": 11.8,
  "serialize_state_sec": 0.05,
  "write_output_sec": 0.02
}
```

**Interpretasi**:

- `chunked_scan_sec` adalah fase utama yang dioptimasi (89% dari total waktu di contoh).
- `load_weights_sec` bisa di-cache untuk repeated runs (menggunakan M7 LRU).
- `serialize_state_sec` dan `write_output_sec` adalah I/O-bound, tidak signifikan untuk GDN.

### Timing Profile Collection

Untuk mengumpulkan timing profile:

```bash
# Enable detailed timing
kimo gdn \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 --dk 128 --dv 128 \
  --timing-profile
```

Output tambahan:

```json
{
  "timing_profile": {
    "per_chunk_times_ms": [5.8, 6.0, 5.9, 5.7],  // Waktu per chunk
    "per_token_times_us": [28, 29, 30, 28, 29, ...],  // Waktu per token (naive baseline)
    "wy_coeff_time_ms": 40.2,
    "wy_update_time_ms": 80.5,
    "sync_time_ms": 10.1
  }
}
```

### Bottleneck Analysis

Berdasarkan timing breakdown:

1. **Jika T_wy_coeff dominan**: Compute-bound → optimasi WY coefficient computation (SIMD, vectorization).
2. **Jika T_wy_update dominan**: Memory-bound → optimasi memory access pattern, cache blocking.
3. **Jika T_sync dominan**: Synchronization overhead → pertimbangkan chunk size lebih besar atau reduce barrier frequency.

Target M8: Speedup ≥ 2× (G-M8-3). Jika speedup < 2×, analisis bottleneck dan optimasi sesuai.

## State Lifecycle

### Initialization

State $S$ diinisialisasi sebelum memproses token pertama:

$$S_0 = \mathbf{0} \in \mathbb{R}^{d_k \times d_v}$$

Aturan:

- Zero initialization wajib untuk semua layer.
- Tidak ada random initialization (deterministik).
- Seed parameter hanya untuk reproducibility test, bukan untuk init state.

### Reset Between Sequences

Untuk setiap sequence baru (independent prompt), state harus di-reset ke $S_0 = \mathbf{0}$:

```python
# Pseudocode untuk reset
def reset_state():
    S = zeros(dk, dv)  # Reset ke zero
```

Kasus reset:

- Prefill baru untuk prompt berbeda → reset.
- Decode baru setelah prefill → jangan reset (gunakan state dari prefill).
- Continuation (kontinuing dari sequence sebelumnya) → jangan reset (gunakan state tersimpan).

### Persist (Serialization)

State dapat diserialisasi untuk continuation atau debugging:

**Format**: Binary raw float32 (row-major)

- Shape: `[layers, dk, dv]`
- Total bytes: `layers * dk * dv * 4`

**Serialization**:

```bash
# Save state
kimo gdn \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 --dk 128 --dv 128
```

**Deserialization** (untuk continuation):

```bash
# Load state dan continue
kimo gdn \
  --tokens /data/prompt2_tokens.json \
  --state-input /work/prompt1_state.bin \
  --output /work/prompt2_state.bin \
  --layers 30 --dk 128 --dv 128
```

Catatan: CLI awal mungkin belum support `--state-input`. Tambahkan di M8 jika continuation diperlukan untuk M9.

### State Serialization Format (Detailed)

**Binary Header** (optional, untuk future-proofing):

```c
struct StateHeader {
    uint32_t magic;      // 0x47444E53 ("GDNS" in hex)
    uint32_t version;    // 1
    uint32_t layers;     // Jumlah layer GDN
    uint32_t dk;         // Dimensi key
    uint32_t dv;         // Dimensi value
    uint32_t dtype;      // 1 = float32
    uint64_t state_bytes; // Total bytes data state
};
```

**Data Layout**:

Setelah header (jika ada), data state disimpan secara row-major:

```
[L=0, dk=0, dv=0] [L=0, dk=0, dv=1] ... [L=0, dk=0, dv=dv-1]
[L=0, dk=1, dv=0] [L=0, dk=1, dv=1] ... [L=0, dk=1, dv=dv-1]
...
[L=0, dk=dk-1, dv=0] ... [L=0, dk=dk-1, dv=dv-1]
[L=1, dk=0, dv=0] ... [L=1, dk=dk-1, dv=dv-1]
...
[L=layers-1, dk=dk-1, dv=dv-1]
```

**Offset Formula**:

Untuk mengakses elemen `S[l][i][j]`:

```c
size_t offset = l * dk * dv + i * dv + j;
float value = data[offset];
```

**Endianness**: Little-endian (standard x86/ARM).

**Padding**: Tidak ada padding antar elemen (packed).

**Checksum** (optional, untuk SEC-6):

Setelah data state, tambahkan SHA-256 checksum:

```
[state_data] [sha256_hash (32 bytes)]
```

Hash mencakup header + data state (tidak termasuk hash itu sendiri).

**Serialization Example** (C-like pseudocode):

```c
void serialize_state(FILE* fp, float* state, int layers, int dk, int dv) {
    // Write header (optional)
    StateHeader hdr = {
        .magic = 0x47444E53,
        .version = 1,
        .layers = layers,
        .dk = dk,
        .dv = dv,
        .dtype = 1,
        .state_bytes = layers * dk * dv * 4
    };
    fwrite(&hdr, sizeof(StateHeader), 1, fp);

    // Write state data
    fwrite(state, sizeof(float), layers * dk * dv, fp);

    // Write checksum (optional)
    uint8_t hash[32];
    compute_sha256(fp, hash);  // Compute hash of written data
    fwrite(hash, 1, 32, fp);
}
```

**Deserialization Example**:

```c
void deserialize_state(FILE* fp, float* state, int* layers, int* dk, int* dv) {
    // Read header
    StateHeader hdr;
    fread(&hdr, sizeof(StateHeader), 1, fp);

    // Validate magic and version
    if (hdr.magic != 0x47444E53 || hdr.version != 1) {
        error("Invalid state file format");
    }

    // Validate dimensions match expected
    if (hdr.layers != *layers || hdr.dk != *dk || hdr.dv != *dv) {
        error("State dimensions mismatch");
    }

    // Read state data
    fread(state, sizeof(float), hdr.layers * hdr.dk * hdr.dv, fp);

    // Verify checksum (optional)
    uint8_t expected_hash[32];
    fread(expected_hash, 1, 32, fp);
    uint8_t computed_hash[32];
    compute_sha256(fp, computed_hash);
    if (memcmp(expected_hash, computed_hash, 32) != 0) {
        error("State checksum mismatch");
    }
}
```

**Versioning**:

Jika format berubah di masa depan:

- Increment `version` di header.
- Backward compatibility: reader harus support version 1..N.
- Migration: tool untuk convert dari version lama ke baru.

### Memory Layout

**Layout**: Row-major (C-style)

- Struktur: `S[layer][row][col]` dengan `row` ∈ $[0, d_k)$, `col` ∈ $[0, d_v)$
- Stride: `stride_row = dv`, `stride_layer = dk * dv`

**Contoh access** (C-like pseudocode):

```c
// Access S[l][i][j]
float value = state[l * dk * dv + i * dv + j];
```

### dtype

**State dtype**: FP32 (float32)

- Alasan: Numerical stability untuk delta rule (F14).
- Oracle: FP32.
- Mojo implementation: Chunked scan internal juga FP32 untuk MATCH strict.
- Quantization state: Out of scope M8 (pertimbangkan di phase quant M6+).

### Memory Growth with Context Length

**Invariant**: State size konstan terhadap $s$.

$$M_{state} = L_{gdn} \cdot d_k \cdot d_v \cdot 4 \text{ bytes}$$

Contoh untuk port M9:

- $L_{gdn} = 30$ layers
- $d_k = 128$
- $d_v = 128$
- $M_{state} = 30 \times 128 \times 128 \times 4 = 1,966,080$ bytes ≈ **1.88 MiB**

Bandingkan dengan KV cache (F2) yang tumbuh linear dengan $s$:

- Trial MHA @ 8K ctx: 0.75 GiB
- GDN state @ 8K ctx: 1.88 MiB (konstan)

### Numerical Stability (Long Sequences)

Untuk $s$ besar (1K..32K), delta rule (F14) dapat mengalami:

- Akumulasi error dari produk matriks $S_{t-1}(I - \beta_t k_t k_t^\top)$
- Underflow/overflow jika $\beta_t$ ekstrem

**Mitigasi**:

- Oracle dan Mojo menggunakan FP32 (bukan BF16) untuk state.
- Clamp $\beta_t$ ke range wajar (mis. $[0, 1]$).
- Optional: Periodic renormalization jika norm state tumbuh terlalu besar (opsional, bukan wajib di M8).

Test G-M8-2 (scaling $s$ 1K..32K) dimaksudkan untuk mendeteksi instability ini.

## Chunked Scan Semantics

### Chunk Size

**Default**: 512 tokens per chunk

**Range**: Valid chunk size ∈ $[8, 4096]$ (power of 2 disarankan untuk efisiensi)

**Constraint**: `chunk_size` harus habis membagi `seq_len` atau implementasi harus handle remainder chunk secara eksplisit.

### WY Representation (Woodbury Identity)

Chunked scan menggunakan representasi Woodbury Identity untuk paralelisasi:

Untuk chunk dengan $m$ token $(t, t+1, \dots, t+m-1)$, state evolution dapat dinyatakan sebagai:

$$S_{t+m} = \gamma_{t:m}\, S_t \left(I - \sum_{i=t}^{t+m-1} \beta_i \gamma_{i+1:m}^{-1} k_i k_i^\top\right) + \sum_{i=t}^{t+m-1} \beta_i \gamma_{i+1:m}^{-1} v_i k_i^\top$$

dengan $\gamma_{i+1:m}^{-1} = \prod_{j=i+1}^{t+m-1} \gamma_j^{-1}$ (kumulatif gate inverse).

### Parallel Chunk Processing

**Algorithm sketch**:

```python
# Pseudocode untuk chunked scan
def chunked_scan(tokens, chunk_size):
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    S = zeros(dk, dv)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, seq_len)
        chunk_tokens = tokens[start:end]

        # Compute chunk-specific WY coefficients
        Wy_coeff = compute_wy_coefficients(chunk_tokens)

        # Update state dalam satu operasi chunked
        S = apply_wy_update(S, Wy_coeff)

    return S
```

### Synchronization Boundaries

**Chunk boundary**: State $S$ hanya perlu disinkronisasi di akhir setiap chunk.

**Within chunk**: Operasi paralel (SIMD/vectorized) di dalam perhitungan WY untuk satu chunk.

**Between chunks**: Barrier synchronization untuk memastikan $S$ dari chunk sebelumnya siap sebelum chunk berikutnya.

### Remainder Handling

Jika `seq_len` tidak habis dibagi `chunk_size`:

**Option 1**: Truncate ke kelipatan terdekat (tidak disarankan — kehilangan data).
**Option 2**: Process remainder dengan naive loop (rekomendasi).
**Option 3**: Adaptive chunk size untuk chunk terakhir (kompleks, opsional).

**Implementation recommendation**: Gunakan naive loop untuk remainder chunk untuk menjaga MATCH strict dengan oracle.

### Associativity Assumptions

Delta rule (F14) adalah **operasi asosiatif** dalam arti:

$$(S_{t} \circ k_t) \circ k_{t+1} = S_{t} \circ (k_t \circ k_{t+1})$$

dimana $\circ$ adalah operator delta rule. Ini memungkinkan chunked scan.

**Verification**: Wajib test untuk memastikan chunked scan == naive loop untuk berbagai chunk size (G-M8-1).

### Chunk Size Tuning

**Default**: 512 tokens.

**Trade-off**:

- Chunk size kecil: lebih banyak overhead barrier/sync.
- Chunk size besar: lebih sedikit paralelisme, memory pressure tinggi untuk intermediate WY coefficients.

**Benchmark**: Di M8, ukur throughput untuk chunk size ∈ {64, 128, 256, 512, 1024} dan pilih optimal. Default 512 adalah titik awal.

## Error Handling

### Error Schema

Semua error mengembalikan JSON dengan field `status: "error"` dan `error_code`:

```json
{
  "status": "error",
  "error_code": 2,
  "error_type": "CONFIG_INVALID",
  "message": "Invalid config: dk must be positive"
}
```

### Error Categories

| Error Code | Type                 | Scenario                                                               | Handling                              |
| ---------- | -------------------- | ---------------------------------------------------------------------- | ------------------------------------- |
| 1          | INPUT_INVALID        | Tokens file tidak ditemukan, format JSON salah, tokens kosong          | Validasi input sebelum alloc          |
| 2          | CONFIG_INVALID       | $d_k \le 0$, $d_v \le 0$, layers $\le 0$, chunk_size $\le 0$           | Tolak sebelum alloc                   |
| 3          | MEMORY_ALLOC_FAILURE | Melebihi cgroup 6G, alloc gagal                                        | Cek VmHWM, cleanup partial alloc      |
| 4          | IO_ERROR             | Shard corrupt, read gagal, permission denied                           | Clean error message, exit 4           |
| 5          | GDN_FORWARD_ERROR    | NaN/INF/overflow di state evolution                                    | Log token index, exit 5               |
| 6          | OUTPUT_ERROR         | Gagal atomic write state binary                                        | Retry atau cleanup temp file, exit 6  |
| 7          | CHUNK_SIZE_ERROR     | chunk_size tidak habis membagi seq_len (jika remainder tidak didukung) | Suggest chunk_size yang valid, exit 7 |

### Allocation Failure Handling

**Pre-alloc validation**:

```python
# Validasi sebelum alloc
state_bytes = layers * dk * dv * 4
if state_bytes > MAX_STATE_BYTES:
    raise Error(CONFIG_INVALID, f"State size {state_bytes} exceeds limit {MAX_STATE_BYTES}")
```

**Post-alloc check**:

```python
# Cek setelah alloc
VmHWM = get_vmhwm()
if VmHWM > 6 * 1024 * 1024 * 1024:  # 6 GB
    cleanup_state(S)
    raise Error(MEMORY_ALLOC_FAILURE, f"Exceeded cgroup: {VmHWM} bytes")
```

### NaN/INF Detection

**Detection**: Setelah setiap chunk, cek state untuk NaN/INF:

```python
if torch.isnan(S).any() or torch.isinf(S).any():
    raise Error(GDN_FORWARD_ERROR, f"NaN/INF detected at chunk {chunk_idx}")
```

**Debugging**: Log chunk index, token range, dan $\beta_t$ values untuk root cause.

### Output Atomic Write

**Pattern**: Write ke temp file lalu rename atomic:

```python
temp_path = output_path + ".tmp"
write_state(temp_path, S)
os.rename(temp_path, output_path)  # Atomic
```

**Failure**: Jika rename gagal, cleanup temp file dan return error 6.

### Specific Error Messages

- **Error 1**: `"Input file not found: {path}"` atau `"Invalid token JSON: {error}"`
- **Error 2**: `"Invalid config: dk={dk} must be positive"` atau `"chunk_size={cs} must divide seq_len={sl}"`
- **Error 3**: `"Memory allocation failed: requested {bytes} bytes, cgroup limit 6G"`
- **Error 4**: `"I/O error reading shard {shard}: {errno}"`
- **Error 5**: `"GDN forward error: NaN/INF at chunk {chunk_idx}, token range [{start}, {end})"`
- **Error 6**: `"Output error: failed atomic write to {path}"`
- **Error 7**: `"Chunk size error: {chunk_size} does not divide seq_len {seq_len}. Valid options: {valid_sizes}"`

## Security / Quality

### SEC-4: Resource Guard (Cgroup)

**Cgroup enforcement**: Run `kimo gdn` di bawah cgroup `memory.max=6G`:

```bash
systemd-run --scope -p MemoryMax=6G \
  kimo gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 --dk 128 --dv 128
```

**Pre-alloc validation**:

- Cek state size sebelum alloc: `state_bytes = layers * dk * dv * 4`
- Tolak jika `state_bytes > MAX_STATE_BYTES` (default: 100 MB)
- Cek VmHWM setelah alloc, cleanup jika melebihi 6G

**RLIMIT_FSIZE**: Aktif untuk mencegah write tak terbatas ke disk.

### SEC-5: File Hygiene

**Model directory**: Read-only (0444/0555) saat engine jalan.

**Output directory**: Hanya write ke workdir, tidak pernah ke model dir atau home.

**Atomic write**: State binary ditulis dengan pattern temp + rename (lihat Error Handling §6).

**Golden artifacts**: Oracle state binary hash-protected (SHA-256) untuk regresi senyap.

### SEC-6: Silent Regression Detection

**Golden hash**: State binary dari oracle di-hash (SHA-256) dan disimpan di `fixtures/m8_state_naive.bin.sha256`.

**Regression check**: Setiap commit yang mengubah GDN implementation harus:

1. Re-generate oracle state dengan seed tetap.
2. Compare hash dengan golden hash.
3. Jika beda → jelaskan di commit message (mis. "fix numerical stability").
4. Jika tidak ada penjelasan → blocking review.

**CI integration**:

```bash
# Check regression
sha256sum fixtures/m8_state_naive.bin > fixtures/m8_state_naive.bin.sha256
git diff fixtures/m8_state_naive.bin.sha256
```

### Fuzzing

**Corpus**: 20+ mutasi untuk input tokens dan config:

- Tokens kosong, tokens dengan nilai negatif, tokens dengan nilai > vocab_size
- $d_k = 0$, $d_k = 10^9$ (overflow), $d_v$ serupa
- chunk_size = 0, chunk_size > seq_len
- layers = 0, layers = 10^6

**Expected behavior**: 0 crash, 0 hang, 0 OOM. Semua harus return clean error code (1-7).

**Fuzz command**:

```bash
cargo fuzz run gdn_fuzzer fixtures/fuzz_corpus/
```

### Allocation Bounds

**Config validation**:

- $d_k \in [1, 4096]$ (max reasonable dimensi)
- $d_v \in [1, 4096]$
- layers $\in [1, 100]$ (realistic untuk M9: 30)
- chunk_size $\in [8, 4096]$

**State size limit**:

```python
MAX_STATE_BYTES = 100 * 1024 * 1024  # 100 MB
state_bytes = layers * dk * dv * 4
if state_bytes > MAX_STATE_BYTES:
    raise Error(CONFIG_INVALID, f"State size {state_bytes} exceeds limit {MAX_STATE_BYTES}")
```

### Determinisme

**Seed tetap**: Oracle dan Mojo menggunakan seed tetap (default 42) untuk determinisme.

**Threads = 1**: Verdict numerik hanya sah pada `--threads 1`. Performance benchmark gunakan `--threads nproc` terpisah.

**Reproducibility test**: Ulang 5× dengan seed tetap → hasil identik (state bytes sama).

### Code Hygiene

**Rust**: `cargo clippy` bersih, tidak ada `unwrap` di data path, semua error lewat `Result`.

**Python**: `ruff` bersih, noMcCabe C901 violations.

**Mojo**: `mojo format` bersih, complexity ≤ 15 per fungsi (review manual saat skill `mojo-1-0` aktif).

### Catatan Deviasi vs Paper [R9]

**Wajib commit**: Di `docs/milestones/M8-gdn.md` atau file terpisah, catat:

- Perbedaan bentuk F14 vs paper asli (Yang et al.)
- Alasan deviasi (mis. simplifikasi untuk implementasi)
- Trade-off numerical vs computational

**Contoh format**:

```markdown
## Deviation Notes vs Yang et al. [R9]

1. Paper menggunakan $\gamma_t$ dan $\beta_t$ yang dipelajari; implementasi M8 menggunakan $\gamma_t=1$ (tanpa gate) untuk simplifikasi awal.
2. Paper menggunakan quantization state; M8 menggunakan FP32 untuk numerical stability.
```

## Deviation Notes Template vs Yang et al. [R9]

### Structure untuk Mencatat Deviasi

Gunakan template berikut untuk mencatat perbedaan implementasi M8 vs paper asli (Yang et al.):

```markdown
## Deviation Notes vs Yang et al. [R9]

### Deviation 1: Gate Simplification

**Paper**: $\gamma_t$ dan $\beta_t$ adalah parameter yang dipelajari (learned gates).

**Implementation M8**: $\gamma_t = 1$ (tanpa gate), $\beta_t$ di-set ke nilai konstan atau computed sederhana.

**Alasan**: Simplifikasi untuk MVP M8. Gate yang dipelajari memerlukan training loop yang kompleks dan out of scope untuk initial implementation.

**Trade-off**:

- Pro: Implementasi lebih sederhana, lebih cepat untuk mengimplementasikan.
- Kontra: Mengurangi ekspresivitas model, mungkin menurunkan performa di task tertentu.
- Mitigasi: Gate learned dapat ditambahkan di phase later (M8.2 atau separate milestone).

### Deviation 2: State Quantization

**Paper**: State $S$ dapat di-quantize ke FP16 atau BF16 untuk efisiensi memory.

**Implementation M8**: State menggunakan FP32 untuk numerical stability.

**Alasan**: Delta rule (F14) sensitif terhadap akumulasi error. Quantization dapat menyebabkan drift numerical signifikan untuk long sequences.

**Trade-off**:

- Pro: Numerical stability lebih baik,MATCH strict dengan oracle lebih mudah.
- Kontra: Memory usage lebih tinggi (4 bytes per element vs 2 bytes).
- Mitigasi: Quantization dapat ditambahkan di phase quant (M6+) setelah baseline FP32 stabil.

### Deviation 3: Chunk Size Strategy

**Paper**: Tidak spesifik mengenai chunking (asumsi sequential scan).

**Implementation M8**: Chunked scan dengan chunk size 512 (tunable).

**Alasan**: Chunked scan memungkinkan paralelisasi dan speedup. Sequential scan terlalu lambat untuk CPU-bound implementation.

**Trade-off**:

- Pro: Speedup 2-3× (G-M8-3), cocok untuk CPU target.
- Kontra: Menambah kompleksitas implementasi (WY representation, synchronization).
- Mitigasi: Remainder handling dan thorough testing memastikan correctness.

### Deviation 4: [Tambahkan deviasi lain jika ditemukan]

**Paper**: [Deskripsi paper]

**Implementation M8**: [Deskripsi implementasi]

**Alasan**: [Alasan deviasi]

**Trade-off**:

- Pro: [Keuntungan]
- Kontra: [Kerugian]
- Mitigasi: [Strategi mitigasi]
```

### Commit Deviation Notes

Simpan deviation notes di salah satu lokasi:

1. Inline di `docs/milestones/M8-gdn.md` (section ini).
2. File terpisah `docs/milestones/M8-deviations.md`.
3. Appendix di `docs/appendices/M8-deviations.md`.

Wajib commit di git sebelum M8 hijau.

## Long-Sequence Stability Analysis

### Numerical Stability Concerns

Untuk sequence panjang ($s$ > 1K), delta rule (F14) dapat mengalami masalah:

1. **Akumulasi error**: Produk matriks $S_{t-1}(I - \beta_t k_t k_t^\top)$ berulang kali dapat menyebabkan error accumulation.
2. **Underflow/overflow**: Jika $\beta_t$ ekstrem (sangat kecil atau sangat besar), komputasi dapat underflow/overflow.
3. **Drift**: State dapat drift dari nilai "benar" karena pembulatan FP32.

### Mitigation Strategies

#### Strategy 1: FP32 State (Default)

- State disimpan dalam FP32 (bukan BF16/FP16).
- Oracle dan Mojo sama-sama FP32 untuk MATCH strict.
- Trade-off: Memory lebih tinggi, tetapi stability lebih baik.

#### Strategy 2: Clamp Beta Values

- Clamp $\beta_t$ ke range wajar, mis. $[0.01, 0.99]$.
- Mencegah extreme values yang menyebabkan underflow/overflow.

```python
beta_t = torch.clamp(beta_t, min=0.01, max=0.99)
```

#### Strategy 3: Periodic Renormalization (Optional)

Jika norm state tumbuh terlalu besar:

- Renormalisasi state secara periodik (mis. setiap 1K tokens).
- Bagi state dengan norm atau lakukan scaling.

```python
if t % 1000 == 0:
    norm = torch.norm(S)
    if norm > THRESHOLD:
        S = S / norm
```

Catatan: Ini mengubah semantik F14. Gunakan hanya jika drift terdeteksi di testing.

#### Strategy 4: Higher Precision for Intermediate (Optional)

Untuk komputasi intermediate dalam chunked scan:

- Gunakan FP64 untuk WY coefficient computation.
- State tetap FP32.
- Trade-off: Komputasi lebih lambat, tetapi accuracy lebih tinggi.

### Testing for Stability

#### Test G-M8-2: Scaling $s$ 1K..32K

Run test untuk berbagai sequence length:

```bash
for s in 1024 2048 4096 8192 16384 32768; do
  kimo gdn \
    --tokens /data/seq_${s}_tokens.json \
    --output /work/seq_${s}_state.bin \
    --layers 30 --dk 128 --dv 128

  python tools/oracle/oracle_gdn.py \
    --tokens /data/seq_${s}_tokens.json \
    --output /work/seq_${s}_state_naive.bin \
    --layers 30 --dk 128 --dv 128

  kimo compare \
    --reference /work/seq_${s}_state_naive.bin \
    --candidate /work/seq_${s}_state.bin \
    --tolerance 1e-3
done
```

**Expected outcome**:

- Δ_max ≤ 1e-3 untuk semua $s$.
- Tidak ada NaN/INF.
- Tidak ada monotonic increase dalam Δ_max sebagai fungsi $s$.

#### Norm Monitoring

Monitor norm state sebagai fungsi $t$:

```python
norms = []
for t in range(seq_len):
    S = update_state(S, kt, vt, gamma_t, beta_t)
    norms.append(torch.norm(S).item())

# Plot norms vs t
# Cek apakah ada exponential growth
```

Jika norm tumbuh eksponensial → indikasi instability.

### Expected Stability Behavior

Untuk implementasi yang benar:

- Norm state harus bounded (tidak tumbuh tanpa batas).
- Δ_max harus stabil sebagai fungsi $s$ (tidak meningkat drastis).
- Tidak ada NaN/INF untuk $s$ hingga 32K.

Jika instability terdeteksi:

1. Review implementasi F14 (cek order operasi).
2. Tambahkan clamp untuk $\beta_t$.
3. Pertimbangkan periodic renormalization.
4. Cek untuk overflow di intermediate computation.

## DoD

### Implementation-Specific Items

- [ ] CLI `kimo gdn` implementasi lengkap dengan semua flags dan exit codes
- [ ] Oracle `tools/oracle/oracle_gdn.py` implementasi naive loop FP32
- [ ] Rust `compare` tool untuk state binary comparison dengan F10 metrics
- [ ] M8 fixture synthetic (`fixtures/m8_tokens.json`, `fixtures/m8_gdn_weights.safetensors`, `fixtures/m8_state_naive.bin`)
- [ ] Mojo chunked scan kernel dengan WY representation
- [ ] State lifecycle: zero init, reset antar sequence, persist binary
- [ ] Memory layout row-major dengan shape `[layers, dk, dv]`
- [ ] Chunked scan dengan chunk size 512 (tunable via CLI)
- [ ] Remainder handling untuk seq_len tidak habis membagi chunk_size
- [ ] Error handling dengan 7 error codes dan JSON output
- [ ] Cgroup enforcement di bawah 6G (SEC-4)
- [ ] Atomic write untuk state output (SEC-5)
- [ ] Golden hash untuk regresi senyap (SEC-6)
- [ ] Fuzz corpus 20+ mutasi dengan 0 crash/hang/OOM
- [ ] Determinisme test (5× ulang, hasil identik)
- [ ] Code hygiene: `cargo clippy`, `ruff`, `mojo format` bersih
- [ ] Workflow diagram Mermaid ditambahkan dan divalidasi sintaksnya
- [ ] Integration tests (M7, M9, continuation, long-sequence) ditulis dan lulus
- [ ] Performance baseline protocol diikuti dengan p50/p95 reporting
- [ ] State serialization format dengan header dan checksum diimplementasikan
- [ ] Per-token timing breakdown tersedia untuk bottleneck analysis
- [ ] Deviation notes vs paper [R9] di-commit sebelum M8 hijau
- [ ] Long-sequence stability analysis dilakukan untuk $s$ ∈ {1K..32K}

### Quality Gates

- [ ] G-M8-1 hijau: chunked == naive oracle dengan $\Delta_{max} \le 10^{-3}$ (100 sekuens acak)
- [ ] G-M8-2 hijau: state fixed-size vs $s$ (sampler 1K..32K, bukti konstan)
- [ ] G-M8-3 hijau: throughput sanity chunked ≥ 2× naive
- [ ] Catatan deviasi vs Yang et al. [R9] ter-commit
- [ ] Fase GDN hijau → `../04-quality.md` §5.4

### Integration Readiness

- [ ] M8 lulus CI dengan fixture synthetic (tanpa model 28 GB)
- [ ] M8 lulus testing dengan model asli (validasi numerik akhir)
- [ ] M8 siap untuk integrasi M9 (30 GDN layers + 10 Gated Attention layers)
