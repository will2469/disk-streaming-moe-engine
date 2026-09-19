# M8 — DeltaNet/GDN Recurrence Kernel Baseline (Chunked Scan vs Naive Loop)

> Proyek: `disk-streaming-moe-engine`. Fase: **GDN**. Index: `../README.md`.

| Field       | Nilai                                                                                                                |
| ----------- | -------------------------------------------------------------------------------------------------------------------- |
| Deliverable | Implementasi Mojo chunked scan DeltaNet/GDN recurrence kernel baseline yang ekuivalen secara numerik vs naive oracle |
| Komponen    | C2 ekstensi (GDN kernels), C7                                                                                        |
| Prasyarat   | M7 hijau (trial stabil)                                                                                              |
| Next        | `M9-port.md`                                                                                                         |
| Gate        | G-M8-1..G-M8-3                                                                                                       |
| Rumus       | F14, F10, kontras F2                                                                                                 |

## Tujuan

Menyiapkan dan memverifikasi **engine kernel recurrence DeltaNet/GDN baseline** untuk porting ke model Qwen3.6: representasi chunked WY scan, batas memori runtime $O(1)$ (state konstan, bukan tumbuh dengan $s$ seperti KV cache), dan penanganan remainder boundary. Proyeksi bobot checkpoint learned gate full forward model diintegrasikan pada milestone M9.

## Rumus (F14) [R9]

$$S_t = \gamma_t\, S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top,\quad S\in\mathbb{R}^{d_v\times d_k} \tag{F14}$$

### Canonical State Layout Contract ($S \in \mathbb{R}^{d_v \times d_k}$)

1. **Mathematical State**:
   - $k_t \in \mathbb{R}^{d_k}$ (vektor key kolom $[d_k \times 1]$).
   - $v_t \in \mathbb{R}^{d_v}$ (vektor value kolom $[d_v \times 1]$).
   - $S_t \in \mathbb{R}^{d_v \times d_k}$ (matriks state: baris $= d_v$, kolom $= d_k$).
   - Suku update 1: $(I - \beta_t k_t k_t^\top) \in \mathbb{R}^{d_k \times d_k}$, di mana $I$ matriks identitas $d_k \times d_k$. Perkalian kanan $S_{t-1}(I - \beta_t k_t k_t^\top)$ adalah $[d_v \times d_k] \times [d_k \times d_k] = [d_v \times d_k]$.
   - Suku update 2: $v_t k_t^\top \in \mathbb{R}^{d_v \times d_k}$ (outer product $[d_v \times 1] \times [1 \times d_k]$).
   - Output retrieval: $y_t = S_t q_t \in \mathbb{R}^{d_v}$ untuk query $q_t \in \mathbb{R}^{d_k}$ ($[d_v \times d_k] \times [d_k \times 1] = [d_v \times 1]$).
2. **Physical Storage**:
   - `S[layer][row=dv][col=dk]` dalam memori contiguous row-major FP32.
   - Offset: `l * dv * dk + i * dk + j` dengan row $i \in [0, d_v)$ dan col $j \in [0, d_k)$.
   - Stride: `stride_row = dk`, `stride_layer = dv * dk`.
3. **Penyebab Bug Asimetris & Penolakan $[d_k, d_v]$**:
   - Orientasi $[d_k \times d_v]$ bertabrakan dengan aljabar linier: $[d_k \times d_v] \times [d_k \times d_k]$ tidak terdefinisi kecuali $d_v == d_k$.
   - Fixture simetris ($d_k = d_v = 128$) berisiko menyembunyikan pelanggaran kontrak ini.
   - Gate G-M8-1 mewajibkan probe asimetris ($d_k \ne d_v$, mis. $d_k=32, d_v=48$) untuk memastikan implementasi menaati layout kanonis.
4. **Karakteristik & Batasan Scope (M8 Recurrence Kernel vs M9 Full Model)**:
   - **M8 = DeltaNet/GDN Recurrence Kernel Baseline**: Menguji ekuivalensi matematis antara representasi chunked WY scan vs naive loop rekuren serial atas input stream $(k_t, v_t, \beta_t, \gamma_t)$ yang diberikan. Pada baseline fixtures/pengujian M8, $\gamma_t = 1$ (decay konstan/un-gated) dan $\beta_t$ berupa skalar konstan atau formulasi terhitung sederhana.
   - **M9 = Full Forward Model Checkpoint Integration**: Parameter proyeksi gate dari bobot checkpoint resmi ($\beta_t = \text{sigmoid}(x_t W_\beta)$ dan $\gamma_t = \text{sigmoid}(x_t W_\gamma)$) diuji secara penuh pada milestone M9 saat forward hybrid 40-layer diintegrasikan. M9 mengonsumsi kernel recurrence M8 tanpa mengubah kontrak matematika recurrence kernel.
   - Oracle = **loop rekuren naive** Python fp32 (bukan model hybrid publik — R6).
   - Implementasi Mojo = chunked scan (paralel per blok, representasi WY) wajib ekuivalen secara numerik (numerical equivalence $\Delta_{max} \le 10^{-3}$) terhadap naive loop.
   - Sifat kunci: $|S|$ konstan terhadap $s$ → kontras F2 ($M_{KV}$ tumbuh linear).

### Reproducible Floating-Point Execution Contract (Numerical Equivalence)

Klaim perbandingan antara Mojo chunked scan dan naive loop Python adalah **ekuivalensi numerik terikat toleransi (numerical equivalence)** dengan threshold $\Delta_{max} \le 10^{-3}$ pada metrik F10, **bukan bitwise equality ("strict MATCH")**.

Perbedaan operasional floating-point:
Secara aljabar murni, representasi Woodbury (WY) ekuivalen dengan akumulasi serial token-by-token. Namun, penjumlahan dan perkalian IEEE 754 **tidak asosiatif**:
$$(a + b) + c \ne a + (b + c), \qquad a \cdot b + c \ne \mathrm{fma}(a, b, c)$$
Akumulasi serial token-by-token pada oracle vs blok GEMM + tree reduction pada kernel chunked Mojo mengeksekusi urutan round-off floating-point yang berbeda. Untuk memastikan reprodusibilitas hasil across compiler, flags, dan perangkat keras, kontrak floating-point FP32 dikunci sebagai berikut:

1. **Rounding Mode**:
   - IEEE 754-2008 single-precision (binary32 / FP32).
   - Rounding mode default: Round-to-Nearest, ties to Even (`FE_TONEAREST` / `roundTiesToEven`).
   - Perlakuan subnormal: Flush-to-Zero (FTZ) dan Denormals-Are-Zero (DAZ) diizinkan pada CPU registers. State didesain tetap dalam rentang normal melalui inisialisasi nol dan aktivasi input terikat (bounded input activations) dari definisi arsitektur model/fixture (bukan clamping ad-hoc saat runtime).
2. **FMA (Fused Multiply-Add)**:
   - **Allowed in Mojo**: FMA (`fma(a, b, c)` atau fused contraction `-ffp-contract=on`) **diizinkan** pada kernel chunked Mojo untuk memaksimalkan efisiensi hardware SIMD (AVX2/AVX-512/NEON).
   - **Oracle Python**: Menggunakan evaluasi ekspresi PyTorch/NumPy standar (unfused mul + add / double rounding).
   - **Bound Drift**: Perbedaan roundoff akibat FMA berorde $\le 0{,}5 \text{ ULP} \approx 6 \times 10^{-8}$ per operasi; akumulasi drift pada chunk $m=512$ terbukti secara analitis dan empiris $\le 5 \times 10^{-5}$, jauh di dalam margin $\Delta_{max} \le 10^{-3}$.
3. **Reassociation & Fast-Math**:
   - Flag optimizer agresif/tak aman (`-ffast-math`, `-fassociative-math`, `-freciprocal-math`) **DILARANG KERAS** pada kompilasi Mojo.
   - Reasosiasi aljabar hanya sah pada level algoritma matematika WY chunking, bukan melalui transformasi sembarang oleh compiler passes. Urutan ekspresi dalam fungsi kernel harus deterministik.
4. **Reduction Order**:
   - **Inter-chunk chain (strictly sequential)**: Rantai evolusi antar-chunk ($S_{c} \to S_{c+1}$) wajib sekuensial linear. Tidak boleh ada parallel tree reduction antar-chunk yang mengubah kausalitas urutan waktu $t$.
   - **Intra-chunk accumulation**: Reduksi matriks blok WY mengikuti urutan loop kontraksi $d_k$ yang terdefinisi.
5. **SIMD Reduction Order**:
   - Reduksi horizontal pada vector register (mis. horizontal add pada dot product) wajib menggunakan deterministic pairwise tree reduction dengan vector width kanonis yang di-pin per arsitektur (misal 8 elemen pada float32x8 AVX2).
   - Unrolling loop dalam kernel harus bernilai konstan eksplisit, tidak bergantung pada heuristik compiler dinamis.
6. **Parallel Reduction & Single-Thread Invariant**:
   - Reduksi paralel dinamis antar-thread OS dilarang pada jalur validasi numerik.
   - Verdict Gate G-M8-1 **hanya sah dievaluasi pada `--threads 1`** (deterministic single-thread execution, invariant I-8 `mojo-1-0`). Multi-threading (`--threads > 1`) dievaluasi terpisah khusus untuk throughput sanity (G-M8-3).

## Gate

| Gate   | Kriteria                                    | Threshold                                                                                                                         | Metode                                                                      |
| ------ | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| G-M8-1 | ekuivalensi numerik chunked vs naive oracle | $\Delta_{max} \le 10^{-3}$ (numerical equivalence)                                                                                | 100 sekuens acak (wajib termasuk config asimetris $d_k \ne d_v$), threads=1 |
| G-M8-2 | peak memory $O(1)$ konstan vs $s$           | $\|\Delta \text{PeakVmHWM}/\Delta s\| \approx 0$ ($\text{VmHWM}_{32K} - \text{VmHWM}_{1K} \le 10\text{ MB}$), $M_{state}$ konstan | sampler $s \in \{1\text{K}..32\text{K}\}$, chunk_size 512, scratchpad reuse |
| G-M8-3 | core speedup (apples-to-apples)             | $\text{speedup\_core} = T_{\text{naive\_scan}} / T_{\text{chunked\_scan}} \ge 2{,}0\times$                                        | timer in-memory kernel scan-only (tanpa disk/file I/O)                      |

## CLI Contract

### Command: `dismoen gdn`

```bash
dismoen gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 \
  --dk 128 \
  --dv 128 \
  --chunk-size 512 \
  --workdir ./work \
  --threads 1
```

### Arguments

| Argument       | Type | Default  | Description                                                                                           |
| -------------- | ---- | -------- | ----------------------------------------------------------------------------------------------------- |
| `--model-dir`  | path | required | Direktori model dengan safetensors shard                                                              |
| `--tokens`     | path | required | Path ke file JSON dengan input token IDs                                                              |
| `--output`     | path | required | Path output untuk state final (binary)                                                                |
| `--layers`     | int  | 30       | Jumlah layer GDN (untuk port M9: 30)                                                                  |
| `--dk`         | int  | 128      | Dimensi key $d_k$ (trial/port: sesuai config)                                                         |
| `--dv`         | int  | 128      | Dimensi value $d_v$ (trial/port: sesuai config)                                                       |
| `--chunk-size` | int  | 512      | Jumlah token per chunk untuk chunked scan (valid: [8, 4096]; remainder diproses via partial WY chunk) |
| `--workdir`    | path | `./work` | Direktori kerja untuk temporary files                                                                 |
| `--threads`    | int  | 1        | Jumlah thread (default 1 untuk determinisme verdict)                                                  |

> [!NOTE] Determinisme Inferensi & Scope Random Seed
> Runtime CLI `dismoen gdn` tidak memerlukan argumen `--seed` karena seluruh proses forward inferensi bersifat deterministik murni: state selalu diinisialisasi nol ($S_0 = 0$), token IDs berasal dari input deterministik, dan bobot dibaca langsung dari file safetensors/fixture. Argumen `--seed` hanya berlaku pada skrip offline generator fixture (`generate_m8_fixtures.py --seed 42`) dan fuzzer pengujian.

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
    "chunked_scan_sec": 0.05,
    "naive_scan_sec": 0.12,
    "speedup_core": 2.4,
    "walltime_sec": 0.15,
    "tokens_per_sec": 26.7,
    "core_tokens_per_sec": 80.0,
    "vmhwm_bytes": 1073741824,
    "peak_state_bytes": 1966080
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
- `7`: Error chunk size (chunk_size <= 0 atau di luar rentang valid [8, 4096]).

### Contoh Invokasi

```bash
# Happy path: 4 token, 30 layers, chunk size 512
dismoen gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 \
  --dk 128 \
  --dv 128 \
  --chunk-size 512

# Cgroup boundary test
systemd-run --scope -p MemoryMax=6G \
  dismoen gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 \
  --dk 128 \
  --dv 128 \
  --chunk-size 512

# Long sequence (8K tokens) untuk bukti state fixed-size
dismoen gdn \
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
    "chunked_scan_sec": 11.8,
    "naive_scan_sec": 32.5,
    "speedup_core": 2.75,
    "walltime_sec": 12.3,
    "tokens_per_sec": 83.3,
    "core_tokens_per_sec": 86.8,
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
    "chunked_scan_sec": 4.12,
    "naive_scan_sec": 10.95,
    "speedup_core": 2.66,
    "walltime_sec": 6.1,
    "tokens_per_sec": 83.9,
    "core_tokens_per_sec": 124.3,
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
6. **State Serialization**: Tulis state ke format framed binary GDNS v1 kanonis (header 128B + payload FP32 [layers, dv, dk] + trailing SHA-256 32B).
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
# State shape: [dv, dk] (kanonis F14: row = dv, col = dk)
S = zeros(dv, dk)  # state initial [dv, dk]
for t in range(seq_len):
    kt = compute_k(t)          # [dk]
    vt = compute_v(t)          # [dv]
    gamma_t = compute_gamma(t)  # skalar
    beta_t = compute_beta(t)    # skalar

    # F14: Delta rule
    # outer(kt, kt): [dk, dk], I: [dk, dk]
    # S @ (I - beta_t * outer(kt, kt)): [dv, dk] @ [dk, dk] -> [dv, dk]
    # outer(vt, kt): [dv, dk]
    S = gamma_t * S @ (I - beta_t * outer(kt, kt)) + beta_t * outer(vt, kt)

save_state(S, output_path)
```

### Output Binary Format (Framed GDNS v1)

State disimpan dalam format biner framed kanonis **GDNS v1 (normatif)**:

- **Header (128 bytes)**: Magic bytes `{'G','D','N','S'}` (`47 44 4E 53`), Version `1`, `architecture_id=1` (`ARCH_QWEN_GDN`), `dtype=1` (FP32), `layers`, `dv`, `dk`, `state_bytes`, `model_manifest_hash[32]`, dan padding 56B.
- **Payload State (`layers * dv * dk * 4` bytes)**: Tensor state FP32 row-major dengan shape `[layers, dv, dk]` (kanonis F14; row = dv, col = dk).
- **Trailing Checksum (32 bytes, normatif, SEC-6)**: Raw binary digest SHA-256 dari konkatenasi byte header dan payload state: $\text{digest} = \text{SHA-256}(\text{header\_bytes} \mathbin{\Vert} \text{state\_bytes})$, dihitung secara incremental selama stream write/read tanpa seek pass tambahan.
- **Ukuran File Total di Disk**: $128 + (L_{\text{gdn}} \cdot d_v \cdot d_k \cdot 4) + 32 = 160 + (L_{\text{gdn}} \cdot d_v \cdot d_k \cdot 4)$ bytes.
- Contoh untuk 30 layers, dv=128, dk=128: $128 + 1.966.080 + 32 = 1.966.240$ bytes (payload state in-memory tetap 1.966.080 bytes).

### Compare Contract

Rust `compare` tool membaca dua state binary (Mojo chunked vs Oracle naive):

```bash
dismoen compare \
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

- O: numerical equivalence F10 vs naive ($\Delta_{max} \le 10^{-3}$, Kontrak FP32), tiap build.
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
- Layout-order probe: 1 config asimetris tambahan WAJIB (mis. $d_k=32, d_v=48$) — order layout yang salah lolos tak terdeteksi bila selalu $d_k==d_v$.

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
dismoen gdn \
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
dismoen forward-port \
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

### State Continuation & Chunk Boundary Stress Test (normatif)

**Test objective**: Verifikasi bahwa representasi WY chunked scan dan pemrosesan partial remainder chunk mematuhi hukum komposisi state:
$$\text{state}(seq_1 \mathbin{\Vert} seq_2) \equiv \text{continuation}(\text{state}(seq_1), seq_2)$$
di bawah variasi ukuran chunk $C$ dan pemotongan boundary token asimetris/tidak rata (_boundary stress_).

**Latar Belakang Arsitektural**:
Banyak implementasi chunked scan tampak benar pada sekuens utuh atau pemotongan chunk yang rapi ($s_1 \bmod C = 0$), namun mengalami deviasi numerik atau bug index saat sekuens dipotong tepat di perbatasan chunk:

- Kasus _clean boundary_: $s_1 = 512, s_2 = 512$ ($s_1 \bmod 512 = 0$).
- Kasus _pre-boundary / under-cut_: $s_1 = 511, s_2 = 513$. Pada single-pass combined (1024 token), terdapat tepat 2 full chunk berukuran 512. Namun pada continuation, $seq_1$ membentuk 1 partial chunk berukuran 511. Lalu $seq_2$ (dimulai dari token global 511) membentuk 1 full chunk 512 (token 511..1022) dan 1 remainder chunk berukuran 1 token (token 1023).
- Kasus _post-boundary / over-cut_: $s_1 = 513, s_2 = 511$.
- Kasus _arbitrary / prime-length cut_: $s_1 = 337, s_2 = 687$.

**Protokol Matriks Pengujian**:

Uji continuation wajib dieksekusi melintasi seluruh kombinasi grid berikut:

1. **Chunk Sizes**: $C \in \{64, 128, 256, 512, 1024\}$.
2. **Boundary Split Scenarios ($s_1 + s_2 = 1024$)**:
   - Split Simetris Rapi: $(s_1=512, s_2=512)$
   - Split Under-Cut (-1 token): $(s_1=511, s_2=513)$
   - Split Over-Cut (+1 token): $(s_1=513, s_2=511)$
   - Split Asimetris Arbitrer: $(s_1=337, s_2=687)$

**Test Script (Multi-Chunk & Boundary Matrix)**:

```bash
for C in 64 128 256 512 1024; do
  for SPLIT in "512 512" "511 513" "513 511" "337 687"; do
    set -- $SPLIT
    S1=$1
    S2=$2

    # 1. Jalankan Prefill seq1 -> simpan GDNS v1 state
    dismoen gdn \
      --tokens /data/tokens_${S1}.json \
      --output /work/seq1_C${C}_S${S1}.bin \
      --layers 30 --dk 128 --dv 128 --chunk-size ${C}

    # 2. Jalankan Continuation seq2 dari seq1 state
    dismoen gdn \
      --tokens /data/tokens_${S2}.json \
      --state-input /work/seq1_C${C}_S${S1}.bin \
      --output /work/seq2_cont_C${C}_S${S2}.bin \
      --layers 30 --dk 128 --dv 128 --chunk-size ${C}

    # 3. Jalankan Single-pass baseline (S1 + S2 gabungan)
    dismoen gdn \
      --tokens /data/tokens_combined_1024.json \
      --output /work/combined_C${C}.bin \
      --layers 30 --dk 128 --dv 128 --chunk-size ${C}

    # 4. Verifikasi ekuivalensi numerik
    dismoen-tools compare \
      --reference /work/combined_C${C}.bin \
      --candidate /work/seq2_cont_C${C}_S${S2}.bin \
      --gate G-M8-1
  done
done
```

**Kriteria Penerimaan (Acceptance Criteria)**:

- Pada **seluruh** pasangan $(C, s_1, s_2)$, selisih maksimum wajib memenuhi Gate G-M8-1:
  $$\Delta_{\max} \le 10^{-3},\quad \epsilon_{rel} \le 10^{-4}$$
- Tidak ada crash/assert failure pada boundary remainder handling ($m_{rem} < C$).
- State reset berfungsi sempurna untuk independent sequences (menghasilkan state identik dengan fresh zero-init).

### Long-Sequence Stability Test

**Test objective**: Verifikasi numerical stability untuk $s$ ∈ {1K, 2K, 4K, 8K, 16K, 32K}.

**Test command**:

```bash
for s in 1024 2048 4096 8192 16384 32768; do
  dismoen gdn \
    --tokens /data/seq_${s}_tokens.json \
    --output /work/seq_${s}_state.bin \
    --layers 30 --dk 128 --dv 128

  python tools/oracle/oracle_gdn.py \
    --tokens /data/seq_${s}_tokens.json \
    --output /work/seq_${s}_state_naive.bin \
    --layers 30 --dk 128 --dv 128

  dismoen compare \
    --reference /work/seq_${s}_state_naive.bin \
    --candidate /work/seq_${s}_state.bin \
    --tolerance 1e-3
done
```

**Verification**:

- G-M8-2 lulus (peak memory runtime $O(1)$ konstan dengan slope $\approx 0$ dan state size konstan terhadap $s$).
- Δ_max ≤ 1e-3 untuk semua $s$.
- Tidak ada NaN/INF untuk $s$ besar.

## Performance Baseline (normatif)

### Protocol (normatif)

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

### Run ID Format (normatif)

```
M8-YYYYMMDD-NNN
```

Contoh: `M8-20250115-001`

### Performance Metrics (normatif)

| Metric                | Description                                                    | Target                                                   |
| --------------------- | -------------------------------------------------------------- | -------------------------------------------------------- |
| `chunked_scan_sec`    | Waktu kernel chunked scan Mojo (in-memory compute)             | TBM (diukur)                                             |
| `naive_scan_sec`      | Waktu baseline naive recurrence loop (in-memory compute)       | TBM (diukur)                                             |
| `speedup_core`        | Core speedup: `naive_scan_sec / chunked_scan_sec`              | $\ge 2{,}0\times$ (G-M8-3, apples-to-apples scan-only)   |
| `walltime_sec`        | Total wall clock time proses CLI (load + scan + write)         | TBM (diukur, diagnostik end-to-end)                      |
| `tokens_per_sec`      | Throughput end-to-end CLI: $\text{seq\_len} / \text{walltime}$ | TBM (diukur, diagnostik)                                 |
| `core_tokens_per_sec` | Throughput kernel murni: $\text{seq\_len} / \text{chunked}$    | TBM (diukur)                                             |
| `vmhwm_bytes`         | Peak memory process (VmHWM)                                    | $\le 6\text{G}$ (SEC-4), slope vs $s \approx 0$ (G-M8-2) |
| `peak_state_bytes`    | Ukuran tensor state kanonis                                    | $L_{\text{gdn}} \cdot d_v \cdot d_k \cdot 4$             |

### Estimasi Awal & Amplop Acuan (informatif)

Tabel berikut adalah **amplop acuan kasar non-normatif** untuk orientasi hardware kelas entry (NVMe/DDR4), bukan kriteria penerimaan (acceptance criteria):

| Config            | seq_len | chunk_size | Naive scan time (est) | Chunked scan time (est) | Speedup core (est) |
| ----------------- | ------- | ---------- | --------------------- | ----------------------- | ------------------ |
| Mini (2×32×32)    | 16      | 8          | ~0.5 ms               | ~0.2 ms                 | ~2.5×              |
| Port (30×128×128) | 1K      | 512        | ~32 ms                | ~12 ms                  | ~2.7×              |
| Port (30×128×128) | 8K      | 512        | ~256 ms               | ~96 ms                  | ~2.7×              |

Catatan: Angka di atas adalah estimasi kasar komputasi in-memory kernel murni (tanpa disk I/O model loading dan write state). Angka aktual harus diukur dan dilaporkan dengan run-id.

### p50/p95 Reporting (normatif)

Laporan performance harus menyertakan:

```markdown
## Performance Report: M8-20250115-001

| Metric           | p50        | p95        | min        | max        |
| ---------------- | ---------- | ---------- | ---------- | ---------- |
| chunked_scan_sec | 11.8       | 12.5       | 11.2       | 13.5       |
| naive_scan_sec   | 32.5       | 34.2       | 31.0       | 36.8       |
| speedup_core     | 2.75       | 2.74       | 2.77       | 2.73       |
| walltime_sec     | 12.3       | 13.1       | 11.8       | 14.2       |
| tokens_per_sec   | 83.3       | 78.2       | 86.8       | 72.1       |
| vmhwm_bytes      | 1073741824 | 1073741824 | 1073741824 | 1073741824 |

Environment:

- CPU governor: performance
- C_max: terdeteksi run-time (tanpa angka absolut di spec)
- c: 1 (verdict), c\* (performance)
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
    dismoen gdn \
      --tokens /data/prompt1_tokens.json \
      --output /work/prompt1_state.bin \
      --layers 30 --dk 128 --dv 128 \
      --chunk-size $cs \
      --run-id M8-20250115-chunk${cs}-run${i}
  done
done
```

Analisis p50/p95 untuk setiap chunk size, pilih optimal untuk default.

## Analisis Bottleneck & Catatan Eksplorasi (informatif)

> [!NOTE]
> **Status Non-Normatif (Exploratory Benchmark Note)**:
> Seluruh rincian timing breakdown mikro-detik (~29 μs/token, ~130 μs/chunk) dan speedup teoretis 114× di bawah ini adalah catatan eksplorasi analitis (_back-of-the-envelope_) untuk mengidentifikasi bottleneck arsitektural (memory-bound vs compute-bound) pada lingkungan uji tertentu.
>
> Angka-angka estimasi ini **BUKAN** kriteria penerimaan (_acceptance criteria_) atau bagian dari kontrak milestone. Kontrak normatif milestone M8 **hanya** mengikat terpenuhinya rasio komparatif $\text{speedup\_core} \ge 2{,}0\times$ (Gate G-M8-3) dan slope memori (Gate G-M8-2) yang diukur secara resmi via protokol § Performance Baseline (normatif).

### Observasi Naive Loop Timing (Eksplorasi)

Naive loop memproses satu token per iterasi:

```
Time per token (naive) = T_compute_k + T_compute_v + T_compute_gamma + T_compute_beta + T_delta_rule
```

Breakdown estimasi awal pada mesin uji single-thread:

| Component           | Description              | Estimated Time (microseconds) |
| ------------------- | ------------------------ | ----------------------------- |
| T_compute_k         | Compute key projection   | ~5 μs                         |
| T_compute_v         | Compute value projection | ~5 μs                         |
| T_compute_gamma     | Compute gate γ_t         | ~2 μs                         |
| T_compute_beta      | Compute gate β_t         | ~2 μs                         |
| T_delta_rule        | Apply F14 matrix update  | ~15 μs                        |
| **Total per token** |                          | **~29 μs**                    |

Total untuk 1K tokens: ~29 ms (mendekati observasi in-memory ~32.5 ms).

### Observasi Chunked Scan Timing (Eksplorasi)

Chunked scan memproses $C$ token dalam satu blok menggunakan kernel representasi WY:

```
Time per chunk (chunked) = T_wy_coeff + T_wy_update + T_sync
```

Breakdown estimasi komponen komputasi murni:

| Component                        | Description                       | Estimated Time (microseconds) |
| -------------------------------- | --------------------------------- | ----------------------------- |
| T_wy_coeff                       | Compute WY coefficients for chunk | ~40 μs                        |
| T_wy_update                      | Apply WY update to state          | ~80 μs                        |
| T_sync                           | Barrier synchronization           | ~10 μs                        |
| **Total per chunk (512 tokens)** |                                   | **~130 μs**                   |

Per-token compute equivalent: ~0.25 μs/token (pada chunk size 512).

Total estimasi komputasi aritmetika murni untuk 1K tokens (2 chunks): ~260 μs.

### Analisis Diskrepansi: Teoretis Compute FLOPS vs Realitas Memory Bandwidth (Roofline)

Secara teoretis, jika kernel hanya dibatasi oleh clock CPU dan throughput instruksi ALU/SIMD:
$$\text{Theoretical Peak Speedup} = \frac{1024 \times 29\,\mu\text{s}}{2 \times 130\,\mu\text{s}} = \frac{29.696}{260} \approx 114\times$$

Namun, pada eksekusi aktual CPU host, waktu terukur adalah ~11–12 ms, menghasilkan speedup core aktual ~2.6×–2.8× (memenuhi target G-M8-3 $\ge 2{,}0\times$). Diskrepansi antara 114× (teoretis compute) dan 2.7× (aktual) dijelaskan oleh **Roofline Model**:

1. **Memory-Bandwidth Bound**: Setiap chunk memerlukan pembacaan dan pembaruan state matriks $S \in \mathbb{R}^{d_v \times d_k}$ across 30 layer ($30 \times 128 \times 128 \times 4\text{ B} \approx 1{,}97\text{ MB}$ per pass).
2. **Saturasi Bus DDR4/DDR5**: Akses baca-tulis acak ke RAM host tersaturasi pada bandwidth efektif single-thread CPU (~15–25 GB/s), sehingga kernel berada di regime memory-bound horizontal pada kurva Roofline.
3. **Kesimpulan**: Mengoptimalkan instruksi floating-point lebih jauh tidak akan menghasilkan percepatan signifikan. Optimasi di M8 harus difokuskan pada cache blocking, cache line alignment, dan minimasi traffic memory bus antar-chunk.

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
dismoen gdn \
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

Target M8: speedup_core ≥ 2.0× (G-M8-3). Jika speedup_core < 2×, analisis bottleneck dan optimasi sesuai.

## State Lifecycle

### Initialization

State $S$ diinisialisasi sebelum memproses token pertama:

$$S_0 = \mathbf{0} \in \mathbb{R}^{d_v \times d_k}$$

Aturan:

- Zero initialization wajib untuk semua layer.
- Tidak ada random initialization (deterministik).
- Seed parameter hanya untuk reproducibility test, bukan untuk init state.

### Reset Between Sequences

Untuk setiap sequence baru (independent prompt), state harus di-reset ke $S_0 = \mathbf{0}$:

```python
# Pseudocode untuk reset
def reset_state():
    S = zeros(dv, dk)  # Reset ke zero [dv, dk]
```

Kasus reset:

- Prefill baru untuk prompt berbeda → reset.
- Decode baru setelah prefill → jangan reset (gunakan state dari prefill).
- Continuation (kontinuing dari sequence sebelumnya) → jangan reset (gunakan state tersimpan).

### Persist (Serialization) (normatif)

State diserialisasi untuk continuation konteks, caching, atau verifikasi regresi:

**Format**: Framed Binary GDNS v1 (normatif, mandatory)

- Header: 32 bytes (`StateHeader`)
- Payload State: `layers * dv * dk * 4` bytes (row-major FP32, shape `[layers, dv, dk]`)
- Trailing Checksum: 32 bytes (SHA-256 binary digest)
- Total file size: $64 + (\text{layers} \cdot d_v \cdot d_k \cdot 4)$ bytes

**Serialization**:

```bash
# Save state
dismoen gdn \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 --dk 128 --dv 128
```

**Deserialization** (untuk continuation):

```bash
# Load state dan continue
dismoen gdn \
  --tokens /data/prompt2_tokens.json \
  --state-input /work/prompt1_state.bin \
  --output /work/prompt2_state.bin \
  --layers 30 --dk 128 --dv 128
```

Catatan: `--state-input` dan `--state-output` adalah parameter normatif M8 menggunakan format framed GDNS v1, memastikan kontinuitas state rekuren lintas chunk dan handoff deterministik ke M9.

### State Serialization Format: Framed GDNS v1 (normatif)

Untuk menjamin integritas data, ketiadaan silent corruption, dan kompatibilitas handoff ke milestone M9, format biner state GDN mengikat header dan checksum trailing secara **wajib (mandatory)**.

**Binary Header (128 bytes, normatif)**:

Untuk mencegah ambiguitas endianness integer dan mencegah pemakaian state lintas model yang tidak kompatibel secara semantik, header GDNS v1 menetapkan kontrak level byte eksplisit:

```c
struct StateHeader {
    uint8_t magic[4];                // Byte sequence: {'G', 'D', 'N', 'S'} -> 0x47, 0x44, 0x4E, 0x53
    uint32_t version;                // Format version = 1 (little-endian)
    uint32_t architecture_id;        // Model architecture ID (1 = ARCH_QWEN_GDN)
    uint32_t dtype;                  // 1 = IEEE 754 float32
    uint32_t layers;                 // GDN layer count (e.g. 30)
    uint32_t dv;                     // Value dimension / baris state S (e.g. 128)
    uint32_t dk;                     // Key dimension / kolom state S (e.g. 128)
    uint32_t reserved1;              // 8-byte alignment padding (set to 0)
    uint64_t state_bytes;            // Total bytes data state payload (layers * dv * dk * 4)
    uint8_t model_manifest_hash[32]; // SHA-256 hash dari models.lock.json (SEC-1), [0;32] jika synthetic
    uint8_t reserved2[56];           // Extension padding / reserved (set to 0)
};
```

**Kontrak Byte-Level Header**:

- **Magic**: Tepat 4 byte array `[0x47, 0x44, 0x4E, 0x53]` (`47 44 4E 53`). Komparasi dilakukan per-byte (`hdr.magic[0]=='G' && ...`), bukan interpretasi integer CPU, guna memastikan determinisme lintas platform (x86_64, aarch64).
- **Architecture ID**: `architecture_id = 1` menandakan arsitektur Qwen-Hybrid GDN DeltaNet. State file yang dibaca oleh arsitektur berbeda wajib ditolak (`MODEL_CONFIG_MISMATCH`).
- **Model Manifest Hash (32 bytes, SEC-1)**: Mengikat state secara kriptografis ke identitas checkpoint (`models.lock.json`). Mencegah penggunaan state dari Model A pada Model B meskipun konfigurasi shape ($30 \times 128 \times 128$) identik. Untuk fixture sintetis lokal, nilai diisi 32 byte nol `[0; 32]`.
- **Ukuran Struct Header**: Tepat 128 bytes ($2^7$, selaras dengan cache-line 64B dan batas perataan 8B/64B/128B).

**Data Layout**:

Tepat setelah header 128-byte, data state disimpan secara packed row-major:

```
[L=0, dv=0, dk=0] [L=0, dv=0, dk=1] ... [L=0, dv=0, dk=dk-1]
[L=0, dv=1, dk=0] [L=0, dv=1, dk=1] ... [L=0, dv=1, dk=dk-1]
...
[L=0, dv=dv-1, dk=0] ... [L=0, dv=dv-1, dk=dk-1]
[L=1, dv=0, dk=0] ... [L=1, dv=dv-1, dk=dk-1]
...
[L=layers-1, dv=dv-1, dk=dk-1]
```

**Offset Formula**:

Untuk mengakses elemen `S[l][i][j]` (`i` = baris dv, `j` = kolom dk) dalam payload data:

```c
size_t offset = l * dv * dk + i * dk + j;
float value = data[offset];
```

**Endianness**: Little-endian (standard x86/ARM).

**Padding**: Tidak ada padding antar elemen tensor state (packed).

**Trailing Checksum (32 bytes, normatif, SEC-6)**:

Setelah data state, wajib disertakan 32 bytes raw binary SHA-256 digest:

```
[StateHeader (128B)] [StateData (state_bytes)] [SHA-256 Digest (32B)]
```

Kontrak matematis digest didefinisikan secara eksplisit dan deterministik sebagai hash atas konkatenasi byte header dan byte payload state:

$$\text{digest} = \text{SHA-256}(\text{header\_bytes} \mathbin{\Vert} \text{state\_bytes})$$

di mana:

- $\text{header\_bytes}$: Tepat 128 byte biner dari struct `StateHeader` (little-endian).
- $\text{state\_bytes}$: Tepat $L_{\text{gdn}} \cdot d_v \cdot d_k \cdot 4$ byte data tensor floating-point (IEEE 754 float32, little-endian) dalam urutan baris (_row-major_).
- Total rentang byte yang di-hash adalah tepat $128 + \text{state\_bytes}$ byte pertama berkas.
- Hashing dihitung secara bertahap (_incremental hashing_ via `sha256_update`) saat proses streaming I/O (write atau read), tanpa ketergantungan pada posisi pointer `FILE*` yang ambigu atau operasi seek pass tambahan (_zero redundant I/O_).

**Serialization Implementation (Incremental Hashing)**:

```c
void serialize_state(FILE* fp, const float* state, uint32_t layers, uint32_t dv, uint32_t dk, const uint8_t manifest_hash[32]) {
    uint64_t state_bytes = (uint64_t)layers * dv * dk * sizeof(float);

    // 1. Siapkan header kanonis GDNS v1 (128 bytes)
    StateHeader hdr = {
        .magic = {'G', 'D', 'N', 'S'}, // Byte sequence: 47 44 4E 53
        .version = 1,
        .architecture_id = 1,          // 1 = ARCH_QWEN_GDN
        .dtype = 1,                    // 1 = float32
        .layers = layers,
        .dv = dv,
        .dk = dk,
        .reserved1 = 0,
        .state_bytes = state_bytes,
        .reserved2 = {0}
    };
    if (manifest_hash != NULL) {
        memcpy(hdr.model_manifest_hash, manifest_hash, 32);
    } else {
        memset(hdr.model_manifest_hash, 0, 32);
    }

    // 2. Inisialisasi incremental hasher
    SHA256_CTX sha_ctx;
    sha256_init(&sha_ctx);

    // 3. Tulis header (128 bytes) & update hasher
    if (fwrite(&hdr, sizeof(StateHeader), 1, fp) != 1) {
        error("Failed writing state header");
    }
    sha256_update(&sha_ctx, &hdr, sizeof(StateHeader));

    // 4. Tulis payload state tensor [layers, dv, dk] & update hasher
    size_t num_elements = (size_t)layers * dv * dk;
    if (fwrite(state, sizeof(float), num_elements, fp) != num_elements) {
        error("Failed writing state payload");
    }
    sha256_update(&sha_ctx, state, state_bytes);

    // 5. Finalisasi digest: digest = SHA-256(header_bytes || state_bytes)
    uint8_t digest[32];
    sha256_final(&sha_ctx, digest);

    // 6. Tulis 32-byte trailing checksum
    if (fwrite(digest, 1, 32, fp) != 32) {
        error("Failed writing state checksum");
    }
}
```

**Deserialization & Integrity Verification (Incremental Hashing)**:

```c
void deserialize_state(FILE* fp, float* state, uint32_t expected_layers, uint32_t expected_dv, uint32_t expected_dk, const uint8_t expected_manifest_hash[32]) {
    // 1. Inisialisasi incremental hasher
    SHA256_CTX sha_ctx;
    sha256_init(&sha_ctx);

    // 2. Baca dan validasi header (128 bytes)
    StateHeader hdr;
    if (fread(&hdr, sizeof(StateHeader), 1, fp) != 1) {
        error("Failed reading state header");
    }
    // Verifikasi magic byte-by-byte
    if (hdr.magic[0] != 'G' || hdr.magic[1] != 'D' || hdr.magic[2] != 'N' || hdr.magic[3] != 'S') {
        error("Invalid state magic bytes: LAYOUT_MISMATCH");
    }
    if (hdr.version != 1) {
        error("Invalid state version: LAYOUT_MISMATCH");
    }
    if (hdr.architecture_id != 1) {
        error("Model architecture mismatch: MODEL_CONFIG_MISMATCH");
    }
    if (hdr.dtype != 1) {
        error("Unsupported dtype: expected 1 (FP32)");
    }
    if (hdr.layers != expected_layers || hdr.dv != expected_dv || hdr.dk != expected_dk) {
        error("State dimensions mismatch: LAYOUT_MISMATCH");
    }
    uint64_t expected_bytes = (uint64_t)expected_layers * expected_dv * expected_dk * sizeof(float);
    if (hdr.state_bytes != expected_bytes) {
        error("Corrupted state_bytes field in header: LAYOUT_MISMATCH");
    }

    // Validasi model identity: tolak jika manifest hash berbeda
    if (expected_manifest_hash != NULL) {
        static const uint8_t zero_hash[32] = {0};
        if (memcmp(hdr.model_manifest_hash, zero_hash, 32) != 0 &&
            memcmp(hdr.model_manifest_hash, expected_manifest_hash, 32) != 0) {
            error("State model_manifest_hash mismatch: MODEL_CONFIG_MISMATCH");
        }
    }

    // Update hasher dengan header bytes (128B)
    sha256_update(&sha_ctx, &hdr, sizeof(StateHeader));

    // 3. Baca payload state tensor & update hasher
    size_t num_elements = (size_t)expected_layers * expected_dv * expected_dk;
    if (fread(state, sizeof(float), num_elements, fp) != num_elements) {
        error("Truncated state payload: LAYOUT_MISMATCH");
    }
    sha256_update(&sha_ctx, state, hdr.state_bytes);

    // 4. Baca trailing checksum 32-byte
    uint8_t expected_digest[32];
    if (fread(expected_digest, 1, 32, fp) != 32) {
        error("Missing or truncated trailing checksum: LAYOUT_MISMATCH");
    }

    // 5. Finalisasi digest yang dihitung dan bandingkan
    uint8_t computed_digest[32];
    sha256_final(&sha_ctx, computed_digest);

    if (memcmp(expected_digest, computed_digest, 32) != 0) {
        error("State checksum mismatch: corrupt file (CORRUPT_STATE_CHECKSUM)");
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

- Struktur: `S[layer][row][col]` dengan `row` ∈ $[0, d_v)$, `col` ∈ $[0, d_k)$
- Stride: `stride_row = dk`, `stride_layer = dv * dk`

**Contoh access** (C-like pseudocode):

```c
// Access S[l][i][j] (layer l, row i in [0, dv), col j in [0, dk))
float value = state[l * dv * dk + i * dk + j];
```

### dtype

**State dtype**: FP32 (float32)

- Alasan: Numerical stability untuk delta rule (F14).
- Oracle: FP32.
- Mojo implementation: Chunked scan internal juga FP32 sesuai Kontrak Floating-Point untuk numerical equivalence.
- Quantization state: Out of scope M8 (pertimbangkan di phase quant M6+).

### Memory Growth with Context Length

**Invariant**: State size konstan terhadap $s$.

$$M_{state} = L_{gdn} \cdot d_v \cdot d_k \cdot 4 \text{ bytes}$$

Contoh untuk port M9:

- $L_{gdn} = 30$ layers
- $d_v = 128$ (baris)
- $d_k = 128$ (kolom)
- $M_{state} = 30 \times 128 \times 128 \times 4 = 1,966,080$ bytes ≈ **1.88 MiB**

Bandingkan dengan KV cache (F2) yang tumbuh linear dengan $s$:

- Trial MHA @ 8K ctx: 0.75 GiB
- GDN state @ 8K ctx: 1.88 MiB (konstan)

### Numerical Stability (Long Sequences)

Untuk $s$ besar (1K..32K), delta rule (F14) dapat mengalami tantangan numerik:

- Akumulasi error dari rantai produk matriks $S_{t-1}(I - \beta_t k_t k_t^\top)$
- Underflow/overflow jika input bernilai ekstrem

**Prinsip Ground-Truth & Stabilitas Baseline**:

- **FP32 Canonical State**: Oracle dan Mojo sama-sama menggunakan FP32 murni (bukan BF16) untuk akumulasi state agar mempertahankan presisi numerik.
- **$\beta$ Policy Intrinsik Model**: Rentang nilai $\beta_t$ ditentukan secara deterministik oleh arsitektur model / bobot checkpoint (misal aktivasi Sigmoid $\sigma(x W_\beta) \in (0, 1)$). Runtime engine **DILARANG meng-clamp $\beta_t$ secara sembarang** (mis. $[0.01, 0.99]$) sebagai "mitigasi stabilitas", karena pembatasan nilai buatan mengubah nilai eigen matriks transisi $(I - \beta_t k_t k_t^\top)$ dan merusak recurrence F14 yang sebenarnya.
- **Larangan Modifikasi Semantik F14**: Mitigasi heuristik seperti clamping $\beta$ atau periodic renormalization **STRICTLY OUT OF BASELINE PATH**. Baseline kernel M8 wajib mengevaluasi recurrence F14 murni.

Test G-M8-2 (scaling $s$ 1K..32K) mengevaluasi kestabilan alami F14 raw FP32 ini tanpa modifikasi heuristik.

## Chunked Scan Semantics

### Chunk Size

**Default**: 512 tokens per chunk

**Range**: Valid chunk size ∈ $[8, 4096]$ (power of 2 disarankan untuk efisiensi)

**Constraint**: `chunk_size` harus berada dalam rentang valid $[8, 4096]$ (exit 7 jika di luar rentang). Jika `seq_len` tidak habis dibagi `chunk_size` ($N \bmod C \ne 0$), sisa token diproses secara native sebagai partial WY chunk berukuran $m_{rem} < C$, bukan kondisi error.

### WY Representation (Woodbury Identity) [R9]

Chunked scan menggunakan representasi Woodbury Identity (Bischof-Van Loan / Yang et al. [R9]) untuk memparalelkan perkalian matriks transisi rank-1 berurutan dalam satu chunk:

Untuk chunk $A$ sepanjang $m$ token $(t, t+1, \dots, t+m-1)$, pembaruan state diekspresikan sebagai operator affine:

$$\text{ChunkOp}_A(S_t) = S_t \mathbf{M}_A + \mathbf{B}_A \tag{F14-chunk}$$

dengan:

- **Matriks Transisi Kumulatif ($\mathbf{M}_A \in \mathbb{R}^{d_k \times d_k}$)**:
  $$\mathbf{M}_A = \Gamma_A \left(I_{d_k} - K_A^\top W_A\right)$$
  di mana $K_A \in \mathbb{R}^{m \times d_k}$ adalah matriks baris key $k_i^\top$, $\Gamma_A = \prod_{i=t}^{t+m-1} \gamma_i$, dan $W_A \in \mathbb{R}^{m \times d_k}$ diperoleh melalui representasi WY kompak:
  $$W_A = \left(D_A^{-1} + L_A\right)^{-1} K_A$$
  dengan $D_A = \text{diag}(\beta_t, \dots, \beta_{t+m-1})$ dan $L_A \in \mathbb{R}^{m \times m}$ adalah bagian _strictly lower triangular_ dari $K_A K_A^\top$ (terbobot decay $\gamma$).
- **Matriks Bias Kumulatif ($\mathbf{B}_A \in \mathbb{R}^{d_v \times d_k}$)**:
  $$\mathbf{B}_A = V_A^\top \tilde{P}_A K_A$$
  di mana $V_A \in \mathbb{R}^{m \times d_v}$ adalah matriks value, dan $\tilde{P}_A \in \mathbb{R}^{m \times m}$ adalah matriks bobot segitiga bawah yang mengintegrasikan decay antartoken dan interaksi delta.

Formula ini invarian terhadap panjang chunk $m$ dan berlaku untuk sembarang $m \in [1, C]$ (termasuk partial remainder chunk $m_{rem} < C$).

### Parallel Chunk Processing

**Algorithm sketch**:

```python
# Pseudocode untuk chunked scan dengan native partial WY remainder
def chunked_scan(tokens, chunk_size):
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    S = zeros(dv, dk)  # [dv, dk] kanonis F14 (row=dv, col=dk)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, seq_len)
        chunk_tokens = tokens[start:end]  # panjang m: C untuk full chunk, m_rem < C untuk partial chunk

        # Compute chunk-specific WY coefficients untuk ukuran m (mendukung m <= chunk_size)
        Wy_coeff = compute_wy_coefficients(chunk_tokens)

        # Update state dalam satu operasi chunked
        S = apply_wy_update(S, Wy_coeff)

    return S
```

### Synchronization Boundaries

**Chunk boundary**: State $S$ hanya perlu disinkronisasi di akhir setiap chunk.

**Within chunk**: Operasi paralel (SIMD/vectorized) di dalam perhitungan WY untuk satu chunk.

**Between chunks**: Barrier synchronization untuk memastikan $S$ dari chunk sebelumnya siap sebelum chunk berikutnya.

### Remainder Handling (Native Partial WY Chunk)

Jika `seq_len` tidak habis dibagi `chunk_size` ($N \bmod C \ne 0$):

- **Partisi Token**:
  - $N_{full} = \lfloor N / C \rfloor$ chunk penuh berukuran $C$.
  - $m_{rem} = N \bmod C$ token sisa (remainder).
- **Keputusan Kontrak (Option B — Native Partial WY Chunk)**:
  - **Bukan Error**: Kondisi $N \bmod C \ne 0$ adalah pola input normal dan **TIDAK** memicu Exit 7.
  - **Tanpa Truncate**: Seluruh token diproses penuh tanpa kehilangan konteks.
  - **Tanpa Naive Fallback**: Remainder **TIDAK** dialihkan ke loop rekuren naive Python/Mojo. Tujuannya adalah menjaga satu jalur data path kernel terpadu di engine Mojo serta menguji kebenaran representasi WY pada chunk berukuran arbitrari.
  - **Native Partial WY**: Chunk terakhir dengan panjang $m_{rem} < C$ dieksekusi langsung oleh kernel WY yang sama:
    $$S_{t+m_{rem}} = \text{ChunkOp}_{rem}(S_t) = S_t \mathbf{M}_{rem} + \mathbf{B}_{rem}$$
    dengan $(\mathbf{M}_{rem}, \mathbf{B}_{rem})$ dikomputasi langsung dari sub-matriks $K_{rem}, V_{rem}$ berukuran $m_{rem} \times d_k$ dan $m_{rem} \times d_v$.
  - Kernel WY di Mojo menerima parameter panjang aktif $m \in [1, C]$ sehingga loop SIMD memproses remainder dengan masking/bounded iteration tanpa overhead branching atau fallback sekunder.

### Chunk Operator Semantics & Affine Composition Law

Paralelisasi chunked scan pada Gated DeltaNet (F14) tidak bertumpu pada klaim asosiatif informal tingkat vektor, melainkan pada **struktur aljabar monoid operator affine** pada ruang state matriks $\mathcal{S} = \mathbb{R}^{d_v \times d_k}$.

#### 1. Formulasi Operator Affine Satu Langkah

Setiap langkah token tunggal $t$ dengan parameter input $(k_t, v_t, \beta_t, \gamma_t)$ menginduksi pemetaan affine $\mathcal{T}_t: \mathcal{S} \to \mathcal{S}$:

$$\mathcal{T}_t(S) = S M_t + B_t$$

dengan:

- $M_t = \gamma_t (I_{d_k} - \beta_t k_t k_t^\top) \in \mathbb{R}^{d_k \times d_k}$ (matriks transisi multiplikatif kanan)
- $B_t = \beta_t v_t k_t^\top \in \mathbb{R}^{d_v \times d_k}$ (matriks injeksi bias state)

#### 2. Operator Chunk Segmen Kontigu

Untuk sembarang irisan token kontigu $A = [t_1, t_2)$, aplikasi sekuensial langkah $\mathcal{T}_t$ menghasilkan operator affine komposit:

$$\text{ChunkOp}_A(S) = S \cdot \mathbf{M}_A + \mathbf{B}_A$$

di mana pasangan $(\mathbf{M}_A, \mathbf{B}_A) \in \mathbb{R}^{d_k \times d_k} \times \mathbb{R}^{d_v \times d_k}$ memenuhi:

- $\mathbf{B}_A = \text{ChunkOp}_A(\mathbf{0}_{d_v \times d_k})$ (keadaan state yang dihasilkan murni dari stimulus masukan internal segmen $A$).
- $\mathbf{M}_A = \prod_{t=t_1}^{t_2-1} M_t = \prod_{t=t_1}^{t_2-1} \gamma_t (I_{d_k} - \beta_t k_t k_t^\top)$ (perkalian matriks transisi terurut waktu dari kiri ke kanan).

#### 3. Hukum Komposisi (Monoid Pasangan $(\mathbf{M}, \mathbf{B})$)

Diberikan dua segmen kontigu berdampingan $A = [t_1, t_2)$ dan $B = [t_2, t_3)$, operator gabungan untuk $A \cup B = [t_1, t_3)$ diperoleh melalui komposisi fungsi:

$$\text{ChunkOp}_{A \cup B}(S) = (\text{ChunkOp}_B \circ \text{ChunkOp}_A)(S) = \text{ChunkOp}_B(\text{ChunkOp}_A(S))$$

Substitusi langsung menghasilkan:
$$\text{ChunkOp}_B(S \mathbf{M}_A + \mathbf{B}_A) = (S \mathbf{M}_A + \mathbf{B}_A) \mathbf{M}_B + \mathbf{B}_B = S (\mathbf{M}_A \mathbf{M}_B) + (\mathbf{B}_A \mathbf{M}_B + \mathbf{B}_B)$$

Oleh karena itu, hukum komposisi pasangan $(\mathbf{M}, \mathbf{B})$ membentuk operasi semigroup biner $\star$:

$$(\mathbf{M}_{A \cup B}, \mathbf{B}_{A \cup B}) = (\mathbf{M}_A, \mathbf{B}_A) \star (\mathbf{M}_B, \mathbf{B}_B) \coloneqq \left(\mathbf{M}_A \mathbf{M}_B, \; \mathbf{B}_A \mathbf{M}_B + \mathbf{B}_B\right)$$

#### 4. Asosiatif Aljabar & Elemen Identitas

Operasi $\star$ bersifat asosiatif murni untuk sembarang tiga segmen berurutan $A, B, C$:
$$\left((\mathbf{M}_A, \mathbf{B}_A) \star (\mathbf{M}_B, \mathbf{B}_B)\right) \star (\mathbf{M}_C, \mathbf{B}_C) = (\mathbf{M}_A, \mathbf{B}_A) \star \left((\mathbf{M}_B, \mathbf{B}_B) \star (\mathbf{M}_C, \mathbf{B}_C)\right)$$
keduanya menghasilkan pasangan identik:
$$\left(\mathbf{M}_A \mathbf{M}_B \mathbf{M}_C, \; \mathbf{B}_A \mathbf{M}_B \mathbf{M}_C + \mathbf{B}_B \mathbf{M}_C + \mathbf{B}_C\right)$$
Elemen identitas dari monoid ini adalah $(\mathbf{I}_{d_k}, \mathbf{0}_{d_v \times d_k})$.

Keberadaan monoid ini membuktikan secara analitis bahwa evaluasi chunked scan secara blok $m$ (maupun hierarkis) memiliki ekuivalensi matematis eksak terhadap akumulasi sekuensial token-by-token.

#### 5. Kontrak Uji Komposisi Terverifikasi (Executable Invariants)

Sebagai ganti pengujian verbal, kebenaran implementasi M8 diverifikasi melalui predikat matematis eksak yang diuji di test suite:

1. **Segment Continuation Invariant (Oracle Naive)**:
   Untuk sembarang sekuens token $T[a:b]$ dan titik belah sembarang $m \in (a, b)$:
   $$\text{naive}(T[a:b], S_0) \equiv \text{naive}\left(T[m:b], \; \text{naive}(T[a:m], S_0)\right)$$
   (Evaluasi berantai pada naive loop wajib identik dengan evaluasi bentang penuh tanpa drift).

2. **Operator Composition Invariant**:
   Untuk dua sub-chunk bersebelahan $T[a:m]$ dan $T[m:b]$:
   $$\text{ExtractOp}(T[a:b]) \approx \text{ExtractOp}(T[a:m]) \star \text{ExtractOp}(T[m:b])$$
   di mana:
   $$\|\mathbf{M}_{[a, b)} - \mathbf{M}_{[a, m)} \mathbf{M}_{[m, b)}\|_{\max} \le 10^{-5}$$
   $$\|\mathbf{B}_{[a, b)} - (\mathbf{B}_{[a, m)} \mathbf{M}_{[m, b)} + \mathbf{B}_{[m, b)})\|_{\max} \le 10^{-5}$$

3. **WY Block Operator Equivalence**:
   Operator affine blok $(\mathbf{M}^{WY}_A, \mathbf{B}^{WY}_A)$ dari kernel Mojo WY wajib ekuivalen terhadap operator sequential naive yang diekstrak dari loop oracle:
   $$\|\mathbf{M}^{WY}_A - \mathbf{M}^{\text{naive}}_A\|_{\max} \le 5 \times 10^{-4}$$
   $$\|\mathbf{B}^{WY}_A - \mathbf{B}^{\text{naive}}_A\|_{\max} \le 5 \times 10^{-4}$$

4. **Inter-Chunk State Evolution (Chain Invariant)**:
   Evolusi state antar-chunk berurutan $C_0, C_1, \dots, C_{P-1}$ dilakukan sekuensial tanpa kehilangan konteks:
   $$S_{j+1} = S_j \mathbf{M}_{C_j} + \mathbf{B}_{C_j}$$

#### 6. Batasan Realitas Floating-Point (Numerical Equivalence)

Asosiatif aljabar matematis pada $\star$ **tidak berarti bitwise identical pada aritmetika floating-point IEEE 754**. Evaluasi blok GEMM + tree reduction pada kernel Mojo mengeksekusi urutan penjumlahan berbeda dibandingkan akumulasi serial token-by-token. Target Gate G-M8-1 dirumuskan sebagai **numerical equivalence** ($\Delta_{max} \le 10^{-3}$), dengan jaminan determinisme melalui `--threads 1` dan larangan `-ffast-math`.

**Verification**: Wajib menyertakan unit test `test_gdn_composition_law` (menguji predikat 1 & 2 di atas) serta benchmark G-M8-1 untuk menjamin integritas kernel chunked scan.

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

| Error Code | Type                 | Scenario                                                      | Handling                                    |
| ---------- | -------------------- | ------------------------------------------------------------- | ------------------------------------------- |
| 1          | INPUT_INVALID        | Tokens file tidak ditemukan, format JSON salah, tokens kosong | Validasi input sebelum alloc                |
| 2          | CONFIG_INVALID       | $d_k \le 0$, $d_v \le 0$, layers $\le 0$                      | Tolak sebelum alloc                         |
| 3          | MEMORY_ALLOC_FAILURE | OOM heap allocator failure (heap alloc returns null/fails)    | Tangani kegagalan heap alloc, return exit 3 |
| 4          | IO_ERROR             | Shard corrupt, read gagal, permission denied                  | Clean error message, exit 4                 |
| 5          | GDN_FORWARD_ERROR    | NaN/INF/overflow di state evolution                           | Log token index, exit 5                     |
| 6          | OUTPUT_ERROR         | Gagal atomic write state binary                               | Retry atau cleanup temp file, exit 6        |
| 7          | CHUNK_SIZE_ERROR     | `chunk_size` $\le 0$ atau di luar rentang valid $[8, 4096]$   | Tolak sebelum alloc, exit 7                 |

### Allocation Failure Handling

**Pre-alloc checked arithmetic validation**:

Sebelum mengalokasikan memory di heap, runtime wajib melakukan validasi checked arithmetic untuk mencegah integer overflow (misal jika $d_k=10^9$ dan $\text{layers}=10^6$ yang akan me-wrap integer 32/64-bit unchecked menjadi nilai alokasi kecil dan memicu heap memory corruption):

```python
# Checked arithmetic pre-alloc guard (Python reference)
MAX_STATE_BYTES = 100 * 1024 * 1024  # 100 MB hard ceiling

def validate_and_compute_state_bytes(layers: int, dv: int, dk: int) -> int:
    if layers <= 0 or dv <= 0 or dk <= 0:
        raise Error(CONFIG_INVALID, f"Invalid dimensions: layers={layers}, dv={dv}, dk={dk}")

    # Checked arithmetic: cegah perkalian melampaui batas sebelum overflow
    # Pada Rust/Mojo: layers.checked_mul(dv).and_then(|x| x.checked_mul(dk)).and_then(|x| x.checked_mul(4))
    try:
        dim_prod = layers * dv
        if dim_prod > MAX_STATE_BYTES // (dk * 4):
            raise OverflowError()
        state_bytes = dim_prod * dk * 4
    except OverflowError:
        raise Error(CONFIG_INVALID, f"State allocation size overflow or exceeds {MAX_STATE_BYTES} bytes")

    if state_bytes > MAX_STATE_BYTES:
        raise Error(CONFIG_INVALID, f"State size {state_bytes} exceeds limit {MAX_STATE_BYTES}")
    return state_bytes
```

**Boundary Enforcements & Telemetry Invariant**:

1. **Hard OS Boundary (Primary Enforcement)**: Cgroup limit `MemoryMax=6G` (Linux cgroups v2 `memory.max`). Jika konsumsi memori proses melampaui 6 GB, kernel OOM-killer akan menembak proses secara sinkron (`SIGKILL`). Kode aplikasi tidak dapat mengandalkan inspeksi VmHWM in-process untuk "mencegah" OOM cgroup.
2. **Software Guard (Preflight)**: Validasi checked arithmetic di atas memastikan alokasi state $\le 100$ MB dan alokasi workspace terhitung secara aman sebelum heap requested.
3. **Observability Only (Post-Run Telemetry)**: Metrik `VmHWM` (Peak Resident Set Size dari `/proc/self/status`) direkam pada output JSON `metrics.vmhwm_bytes` murni untuk telemetri profiling dan gate G-M8-2 slope testing, BUKAN sebagai runtime safety guard aktif.

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
- **Error 2**: `"Invalid config: dk={dk} must be positive, layers={layers} must be positive"` atau `"Model/config mismatch: state file architecture_id={id} or manifest_hash={hash} does not match model"`
- **Error 3**: `"Memory allocation failed: requested {bytes} bytes, cgroup limit 6G"`
- **Error 4**: `"I/O error reading shard {shard}: {errno}"`
- **Error 5**: `"GDN forward error: NaN/INF at chunk {chunk_idx}, token range [{start}, {end})"`
- **Error 6**: `"Output error: failed atomic write to {path}"`
- **Error 7**: `"Chunk size error: chunk_size={chunk_size} out of valid range [8, 4096]"`

## Security / Quality

### SEC-4: Resource Guard (Cgroup & Checked Alloc)

**Cgroup enforcement (Hard OS Boundary)**: Run `dismoen gdn` di bawah cgroup `MemoryMax=6G`:

```bash
systemd-run --scope -p MemoryMax=6G \
  dismoen gdn \
  --model-dir /models/qwen-moe \
  --tokens /data/prompt1_tokens.json \
  --output /work/prompt1_state.bin \
  --layers 30 --dk 128 --dv 128
```

**Pre-alloc Checked Arithmetic Guard**:

- Wajib menggunakan checked integer multiplication (`checked_mul` di Rust/Mojo) sebelum alokasi heap untuk menangkal integer overflow pada input fuzzing ekstrem ($d_k = 10^9, \text{layers} = 10^6$).
- Tolak sebelum alokasi jika perkalian overflow atau `state_bytes > MAX_STATE_BYTES` (100 MB) dengan error code 2 (`CONFIG_INVALID`).
- Alokasi heap yang gagal ditangani dengan error code 3 (`MEMORY_ALLOC_FAILURE`).

**Observability vs Guard Distinction**:

- `VmHWM` adalah metrik observabilitas paska-run (telemetri via `/proc/self/status`), **bukan** runtime safety guard in-process (karena kernel cgroup membunuh proses secara instan via `SIGKILL` saat menyentuh `MemoryMax`).
- Evaluasi kepatuhan batas memori dilakukan melalui telemetri `metrics.vmhwm_bytes` pada laporan JSON.

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
- chunk_size = 0, chunk_size = -1, chunk_size = 7, chunk_size = 4097 (out of range [8, 4096])
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

**State size limit & Checked Arithmetic**:

```python
MAX_STATE_BYTES = 100 * 1024 * 1024  # 100 MB

# Checked integer multiplication: tolak overflow sebelum alokasi heap
def compute_state_bytes_safe(layers: int, dv: int, dk: int) -> int:
    # Rust/Mojo: layers.checked_mul(dv)?.checked_mul(dk)?.checked_mul(4)
    if layers > 100 or dv > 4096 or dk > 4096 or layers <= 0 or dv <= 0 or dk <= 0:
        raise Error(CONFIG_INVALID, "Config parameters exceed allowable range")
    state_bytes = layers * dv * dk * 4
    if state_bytes > MAX_STATE_BYTES:
        raise Error(CONFIG_INVALID, f"State size {state_bytes} exceeds limit {MAX_STATE_BYTES}")
    return state_bytes
```

### Determinisme

**Test Fixture Seed vs Runtime Engine**: Test fixture generation offline menggunakan seed tetap (default 42). Runtime forward `dismoen gdn` bersifat deterministik murni dari weights dan token inputs tanpa RNG state internal.

**Threads = 1**: Verdict numerik hanya sah pada `--threads 1`. Performance benchmark gunakan `--threads nproc` terpisah.

**Reproducibility test**: Ulang 5× pada fixture input yang sama → hasil identik secara bit (state bytes dan checksum SHA-256 identik).

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

1. Paper menggunakan $\gamma_t = \sigma(x_t W_\gamma)$ dan $\beta_t = \sigma(x_t W_\beta)$ yang diproyeksikan dari matriks bobot checkpoint; M8 menguji **recurrence kernel baseline** dengan input stream $(k, v, \beta, \gamma)$ terisolasi ($\gamma_t=1$ baseline, $\beta_t$ scalar) sebelum integrasi full forward hybrid 40-layer di M9.
2. Paper menggunakan quantization state; M8 menggunakan FP32 untuk numerical stability.
```

## Deviation Notes vs Yang et al. [R9]

### Deviation 1: Recurrence Kernel Baseline vs Full Model Gate Projections

**Paper**: $\gamma_t = \sigma(x_t W_\gamma)$ dan $\beta_t = \sigma(x_t W_\beta)$ adalah proyeksi gate dinamis yang dihitung dari aktivasi input token menggunakan matriks bobot checkpoint yang dipelajari (_learned gates_).

**Implementation M8**: M8 bertindak secara spesifik sebagai **DeltaNet/GDN recurrence kernel baseline**. Untuk pengujian matematis kernel, verifikasi ekuivalensi numerik, dan benchmark throughput scan, kernel diuji menggunakan $\gamma_t = 1$ (decay konstan/un-gated) dan $\beta_t$ berupa skalar konstan/terhitung langsung. Matriks proyeksi linier bobot checkpoint ($W_\beta, W_\gamma, W_k, W_v$) secara formal diintegrasikan pada milestone **M9 (Full Hybrid Model Port)**.

**Alasan**: Memisahkan verifikasi kebenaran engine kernel recurrence (WY chunked representation, remainder handling, stabilitas FP32, dan $O(1)$ peak memory) dari layer proyeksi bobot transformer 40-layer. Hal ini menjamin bahwa jika terjadi regresi numerik, akar masalah dapat diisolasi dengan tegas antara scan recurrence kernel vs weight projection error.

**Trade-off**:

- Pro: Isolasi murni terhadap kernel aljabar linier WY scan; pengujian unit & integrasi dapat berjalan cepat di CI tanpa memuat checkpoint 70 GB.
- Kontra: M8 belum memvalidasi aktivasi dinamis full checkpoint (ini menjadi tanggung jawab M9).
- Mitigasi: M9 mewarisi kernel M8 secara langsung dan memverifikasi gate learned via checkpoint resmi Qwen3.6.

### Deviation 2: State Quantization

**Paper**: State $S$ dapat di-quantize ke FP16 atau BF16 untuk efisiensi memory.

**Implementation M8**: State menggunakan FP32 untuk numerical stability.

**Alasan**: Delta rule (F14) sensitif terhadap akumulasi error. Quantization dapat menyebabkan drift numerical signifikan untuk long sequences.

**Trade-off**:

- Pro: Numerical stability lebih baik, verifikasi numerical equivalence dengan oracle lebih terjamin.
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

````

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

### Mitigation Strategies & Boundary Rules

#### Strategy 1: FP32 Canonical State (Normative Baseline)

- State wajib disimpan dan diakumulasikan dalam FP32 (bukan BF16/FP16).
- Oracle Python dan Mojo chunked kernel sama-sama menggunakan representasi FP32 untuk menjamin kesetaraan numerik penuh ($\Delta_{max} \le 10^{-3}$).
- Trade-off: Footprint state 4 byte per elemen, namun memberikan stabilitas dinamik yang kokoh sepanjang sekuens 32K token.

#### Non-Baseline Alterations: Banned in M8 Baseline Path

Beberapa teknik heuristik yang sering dibahas dalam eksplorasi numerik secara fundamental **mengubah semantik aljabar F14**:

1. **Clamping $\beta_t$ (STRICTLY OUT OF BASELINE PATH)**:
   - Membatasi $\beta_t$ secara artifisial (mis. $[0.01, 0.99]$) mengubah nilai eigen dari matriks transisi rank-1 update $(I - \beta_t k_t k_t^\top)$.
   - Kebijakan $\beta$ adalah **bagian dari spesifikasi arsitektur model** (misalnya fungsi aktivasi sigmoid terikat $\sigma(x W_\beta) \in (0, 1)$ pada checkpoint resmi Qwen), **BUKAN** patch numerik ad-hoc yang boleh disisipkan oleh engine di runtime.
   - Jika oracle dan kernel Mojo sama-sama meng-clamp $\beta$, pengujian akan lulus secara semu pada target aljabar yang salah. Oleh karena itu, runtime baseline M8 **DILARANG** melakukan clamping $\beta$.

2. **Periodic Renormalization (STRICTLY OUT OF BASELINE PATH)**:
   - Membagi matriks state $S_t$ dengan norm-nya setiap $K$ langkah (mis. $S = S / \|S\|$) mengubah semantik rekurensi linier F14 secara permanen dan merusak kesetaraan terhadap oracle.
   - Teknik ini dilarang keras pada jalur baseline M8.

#### Strategy 2: Intermediate Register Precision (Allowed)

Untuk komputasi intermediate dalam blok WY kernel Mojo:
- Akumulasi dot product dan inversion triangular matriks kecil $W_A$ diizinkan menggunakan SIMD FMA register atau register FP64 jika diperlukan untuk meminimalisasi round-off drift intra-chunk.
- State $S$ yang dioper antar-chunk tetap dalam format FP32 kanonis.

### Testing for Stability

#### Test G-M8-2: Scaling $s$ 1K..32K (Peak Memory & Stability)

Run test untuk berbagai sequence length dengan chunk size konstan $C=512$:

```bash
for s in 1024 2048 4096 8192 16384 32768; do
  dismoen gdn \
    --tokens /data/seq_${s}_tokens.json \
    --output /work/seq_${s}_state.bin \
    --layers 30 --dk 128 --dv 128 \
    --chunk-size 512 > /work/seq_${s}_report.json

  python tools/oracle/oracle_gdn.py \
    --tokens /data/seq_${s}_tokens.json \
    --output /work/seq_${s}_state_naive.bin \
    --layers 30 --dk 128 --dv 128

  dismoen compare \
    --reference /work/seq_${s}_state_naive.bin \
    --candidate /work/seq_${s}_state.bin \
    --tolerance 1e-3
done

# Verifikasi Slope Peak Memory Runtime (Gate G-M8-2)
python -c '
import json
vmhwm = {}
for s in [1024, 2048, 4096, 8192, 16384, 32768]:
    with open(f"/work/seq_{s}_report.json") as f:
        data = json.load(f)
        vmhwm[s] = data["metrics"]["vmhwm_bytes"]

delta_bytes = vmhwm[32768] - vmhwm[1024]
# Kenaikan VmHWM dari 1K ke 32K (faktor 32x panjang sekuens) wajib <= 10 MB (toleransi heap arena allocator)
print(f"VmHWM 1K: {vmhwm[1024]/1e6:.2f} MB, 32K: {vmhwm[32768]/1e6:.2f} MB, delta: {delta_bytes/1e6:.2f} MB")
assert delta_bytes <= 10 * 1024 * 1024, f"FAIL G-M8-2: Runtime memory scaled with s! Delta={delta_bytes} bytes"
print("PASS G-M8-2: Peak runtime memory is O(1) bounded with slope near zero.")
'
```

**Expected outcome (Gate G-M8-2)**:

1. **Numerical Stability**:
   - $\Delta_{max} \le 10^{-3}$ untuk semua $s \in \{1\text{K}..32\text{K}\}$.
   - Tidak ada NaN/INF.
   - Tidak ada monotonic increase dalam $\Delta_{max}$ sebagai fungsi $s$.
2. **Peak Memory Invariant ($O(1)$ vs $s$)**:
   - Runtime peak memory ($\text{VmHWM}$) tidak tumbuh proporsional terhadap $s$: $|\text{VmHWM}(32\text{K}) - \text{VmHWM}(1\text{K})| \le 10\text{ MB}$.
   - Membuktikan bahwa intermediate workspace koefisien WY ($K_A, W_A, V_A, \tilde{P}_A$) menggunakan scratchpad yang di-reuse antar-chunk, bukan menimbun tensor aktivasi di memori.
   - Ukuran tensor state kanonis tetap konstan: $M_{state} = L_{\text{gdn}} \cdot d_v \cdot d_k \cdot 4 = 1.966.080$ bytes (~1.97 MB untuk konfigurasi port 30 layer).

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

1. Review implementasi F14 (verifikasi urutan evaluasi kontraksi dan pairwise tree reduction order).
2. Periksa dynamic range intermediate activations ($K_A, W_A, V_A$).
3. Periksa penanganan subnormal/FTZ/DAZ pada CPU SIMD registers.
4. Cek potensi loss-of-precision atau overflow pada akumulasi intermediate dot product FP32 vs SIMD register.

## DoD

### Implementation-Specific Items

- [x] CLI `dismoen gdn` implementasi lengkap dengan semua flags dan exit codes
- [x] Oracle `tools/oracle/oracle_gdn.py` implementasi naive loop FP32
- [x] Rust `compare` tool untuk state binary comparison dengan F10 metrics
- [x] M8 fixture synthetic (`fixtures/m8_tokens.json`, `fixtures/m8_gdn_weights.safetensors`, `fixtures/m8_state_naive.bin`)
- [x] Mojo chunked scan kernel dengan WY representation
- [x] State lifecycle: zero init, reset antar sequence, persist binary
- [x] Memory layout row-major dengan shape `[layers, dv, dk]` (kanonis F14)
- [x] Chunked scan dengan chunk size 512 (tunable via CLI)
- [x] Remainder handling via native partial WY chunk berukuran $m_{rem} < \text{chunk\_size}$ (tanpa naive fallback)
- [x] Error handling dengan 7 error codes dan JSON output
- [x] Cgroup enforcement di bawah 6G (SEC-4)
- [x] Atomic write untuk state output (SEC-5)
- [x] Golden hash untuk regresi senyap (SEC-6)
- [x] Fuzz corpus 20+ mutasi dengan 0 crash/hang/OOM
- [x] Determinisme test (5× ulang, hasil identik)
- [x] Code hygiene: `cargo clippy`, `ruff`, `mojo format` bersih
- [x] Workflow diagram Mermaid ditambahkan dan divalidasi sintaksnya
- [x] Integration tests (M7, M9, continuation, long-sequence) ditulis dan lulus
- [x] Unit test hukum komposisi operator chunk lulus (`test_gdn_composition_law`: segment continuation dan chunk operator equivalence)
- [x] Performance baseline protocol diikuti dengan p50/p95 reporting
- [x] State serialization format kanonis GDNS v1 mandatory (header 128B dengan magic bytes [47,44,4E,53], arch_id, manifest_hash, dims + row-major FP32 [layers, dv, dk] + trailing SHA-256 32B) diimplementasikan dan divalidasi pada reader & writer
- [x] Per-token timing breakdown tersedia untuk bottleneck analysis
- [x] Deviation notes vs paper [R9] di-commit sebelum M8 hijau
- [x] Long-sequence stability analysis dilakukan untuk $s$ ∈ {1K..32K}

### Quality Gates

- [x] G-M8-1 hijau: ekuivalensi numerik chunked vs naive oracle dengan $\Delta_{max} \le 10^{-3}$ (100 sekuens acak, termasuk $d_k \ne d_v$, threads=1, sesuai Kontrak FP32)
- [x] G-M8-2 hijau: peak memory $O(1)$ konstan vs $s$ ($|\Delta \text{PeakVmHWM}/\Delta s| \approx 0$, $\text{VmHWM}_{32K} - \text{VmHWM}_{1K} \le 10\text{ MB}$, sampler 1K..32K, scratchpad reuse)
- [x] G-M8-3 hijau: core speedup $\text{speedup\_core} = T_{\text{naive\_scan}} / T_{\text{chunked\_scan}} \ge 2\times$ (apples-to-apples in-memory scan-only)
- [x] Catatan deviasi vs Yang et al. [R9] ter-commit
- [x] Fase GDN hijau → `../04-quality.md` §5.4

### Integration Readiness

- [x] M8 lulus CI dengan fixture synthetic (tanpa model 28 GB)
- [x] M8 lulus testing dengan model asli (validasi numerik akhir)
- [x] M8 siap untuk integrasi M9 (30 GDN layers + 10 Gated Attention layers)

## Hasil Sertifikasi Nyata (Qwen 3.6-35B-A3B)

Pengujian integrasi penuh Milestone M8 pada model riil Qwen 3.6-35B-A3B (40 layer: 30 GDN linear attention + 10 full attention) dieksekusi via `tests/integration/test_m8_real_qwen36.sh` (`pixi run test-m8-real`):

| Gate / Metrik | Target Kualifikasi | Nilai Terukur (Qwen 3.6) | Status |
| :--- | :--- | :--- | :--- |
| **Audit GDN Layers** | 30 Layer GDN $\times$ 9 tensor = 270 tensor | 270/270 tensor terverifikasi di index | **PASS** |
| **G-M8-1** | Ekuivalensi numerik chunked vs naive oracle | $\Delta_{\max} = 4.62 \times 10^{-7} \le 10^{-3}$, $\varepsilon_{rel} = 1.12 \times 10^{-6} \le 10^{-4}$ | **PASS** |
| **G-M8-2** | Memory $O(1)$ state kanonis GDNS v1 | State konstan 1,966,240 B (~1.97 MB), $\Delta \text{VmHWM}_{32K-1K} = 2.12\text{ MB} \le 10\text{ MB}$ | **PASS** |
| **G-M8-3** | Core speedup chunked scan vs naive loop | $\text{speedup\_core} = 2.50\times \ge 2.0\times$ ($p50$) | **PASS** |
| **GDNS v1 Framing** | Magic header 128B + trailing SHA-256 (32B) | SHA-256 header+payload cocok bit-for-bit | **PASS** |
| **SEC-4** | Batas pagu memori sistem | $\text{VmHWM} = 0.1005\text{ GB} \le 6.0\text{ GB}$ | **PASS** |
| **SEC-5** | Model dir read-only isolation | PASS (chmod a-w model dir, zero mutations) | **PASS** |
| **M9 Readiness** | Skedul 40-layer hybrid (10 siklus $\times$ [3 GDN + 1 Attn]) | Konsisten 100% dengan bobot riil Qwen 3.6 | **PASS** |
````
