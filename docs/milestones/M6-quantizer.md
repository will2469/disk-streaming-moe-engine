# M6 — Quantizer 4-bit Buatan Sendiri + Dequant Kernel

> Proyek: `disk-streaming-moe-engine`. Fase: **Trial (rekayasa)**. Index: `../README.md`.
> Implementasi dipecah menjadi waves: `../../scratch/wave/m6/README.md` (W1 quantize-cli → W6 gates, catatan kerja gitignored).

| Field       | Nilai                                                         |
| ----------- | ------------------------------------------------------------- |
| Deliverable | Format quant 4-bit sendiri + kernel dequant yang terkalibrasi |
| Komponen    | C5 quantizer, C2 dequant di kernel, C7                        |
| Prasyarat   | M5 hijau                                                      |
| Next        | `M7-odirect-lru.md`                                           |
| Gate        | G-M6-1..G-M6-3                                                |
| Rumus       | F11, F12                                                      |

## Tujuan

"Menemukan kembali GGUF": kendali penuh atas format quant untuk menekan $B_{tok}$ (F3b) tanpa merusak kualitas. Bukan memakai GGUF/llama.cpp quant (ADR D6).

## CLI: `kimo quantize`

Subcommand `quantize` mengkonversi model BF16 ke format quant 4-bit buatan sendiri.

### Input

```bash
kimo quantize \
  --input-dir <DIR> \
  --output-dir <DIR> \
  [--group-size <N>] \
  [--workdir <DIR>]
```

- `--input-dir`: Direktori checkpoint BF16 (8 shard safetensors + index.json).
- `--output-dir`: Direktori output untuk file quant 4-bit.
- `--group-size`: Ukuran grup quant (default: 128).
- `--workdir`: Direktori kerja untuk temporary files (default: `./work`).

### Output JSON

```json
{
  "status": "success",
  "run_id": "M6-20250115-001",
  "model": "qwen1.5-moe-a2.7b-chat",
  "input_format": "BF16",
  "output_format": "4-bit per-group",
  "group_size": 128,
  "num_tensors": 4659,
  "metrics": {
    "walltime_sec": 245.3,
    "input_bytes": 30660512768,
    "output_bytes": 7934542592,
    "compression_ratio": 3.86,
    "avg_epsilon_rel": 0.0087,
    "max_epsilon_rel": 0.0095
  }
}
```

### Exit Codes

- `0`: Sukses, file quant ditulis.
- `1`: Error input (input-dir tidak ada, index tidak valid).
- `2`: Error quantization (tensor corrupt, quantization fail).
- `3`: Error output (gagal atomic write).
- `4`: Error validation (output tidak lolos F11b validation).

### Contoh Invokasi

```bash
# Happy path: quantize BF16 → 4-bit
kimo quantize \
  --input-dir /models/qwen-moe-bf16 \
  --output-dir /models/qwen-moe-4bit \
  --group-size 128

# Custom group size
kimo quantize \
  --input-dir /models/qwen-moe-bf16 \
  --output-dir /models/qwen-moe-4bit \
  --group-size 64
```

## Oracle: Quantization Roundtrip

Oracle `tools/oracle/oracle_quant.py` menjalankan quantization roundtrip BF16 → 4-bit → BF16 untuk verifikasi G-M6-1.

### Input

```bash
python tools/oracle/oracle_quant.py \
  --input-dir <DIR> \
  --group-size <N> \
  --output-report <PATH>
```

- `--input-dir`: Direktori checkpoint BF16 (sama dengan Mojo).
- `--group-size`: Ukuran grup quant (default: 128).
- `--output-report`: Path output report JSON (per-tensor epsilon_rel).

### Process

1. Load model BF16 dari safetensors (8 shard → merge).
2. Per tensor:
   - Split ke grup G (default: 128).
   - Compute scale tersimpan $s_g=\mathrm{ceil}_{\mathrm{F16}}(\max|w_j|/7)$ per grup; grup nol memakai $s_g=1$, $q=0$.
   - Quantize: $q_j = \text{clip}(\text{round}(w_j/s_g), -7, 7)$.
   - Dequantize ke FP32: $\hat{w}^{(32)}_j = \mathrm{fp32}(s_g)q_j$.
   - Compute $\varepsilon_{rel} = \sqrt{\text{MSE}} / \sqrt{\text{var}(w)}$.
   - Verify property FP32: $|\mathrm{fp32}(w_j)-\hat{w}^{(32)}_j| \le s_g/2$.
3. Collect semua per-tensor $\varepsilon_{rel}$.
4. Compute $\max \varepsilon_{rel}$ (untuk G-M6-1).

### Output Format

```json
{
  "run_id": "M6-20250115-001",
  "model": "qwen1.5-moe-a2.7b-chat",
  "group_size": 128,
  "num_tensors": 4659,
  "results": {
    "max_epsilon_rel": 0.0095,
    "avg_epsilon_rel": 0.0087,
    "min_epsilon_rel": 0.0072,
    "tensors": [
      {
        "name": "model.layers.0.self_attn.q_proj.weight",
        "epsilon_rel": 0.0089,
        "property_ok": true
      },
      ...
    ]
  }
}
```

### Determinism

- `torch.manual_seed(42)` untuk deterministik (jika diperlukan).
- Quantization deterministic untuk input BF16 yang sama.

### Verdict Contract

G-M6-1 PASS jika $\max \varepsilon_{rel} \le 10^{-2}$ untuk semua tensor.

## Fixture: M6 PPL Corpus

Fixture `tools/fixtures/m6_ppl_corpus.json` berisi corpus 100×256 untuk gate G-M6-3 (PPL measurement).

### Structure

```json
{
  "name": "M6 PPL corpus",
  "description": "Corpus 100×256 for PPL measurement (G-M6-3)",
  "corpus": [
    {
      "id": "doc1",
      "text": "The quick brown fox jumps over the lazy dog. This is a sample text for perplexity measurement..."
    },
    {
      "id": "doc2",
      "text": "Quantization reduces model size while maintaining quality. This document tests the impact..."
    },
    ...
  ]
}
```

### Corpus Generation

`tools/fixtures/generate_m6_ppl.py`:

1. Load representative text corpus (Wikipedia, books, code).
2. Select 100 documents dengan panjang ~256 token.
3. Tokenize → verify length ≈ 256 (±10 token).
4. Validate: semua token < 151,936 (vocab size).
5. Output JSON dengan 100 documents.

### PPL Measurement Process

1. Load model BF16 → quantize → dequant (roundtrip).
2. Load model BF16 original (baseline).
3. Per document:
   - Tokenize → [s] token IDs.
   - Compute PPL dengan model quant: $\mathrm{PPL}_{quant} = \exp(-1/N_{pred} \sum \ln p_{quant})$.
   - Compute PPL dengan model BF16: $\mathrm{PPL}_{bf16} = \exp(-1/N_{pred} \sum \ln p_{bf16})$.
4. Compute $\Delta\mathrm{PPL} = \mathrm{PPL}_{quant} - \mathrm{PPL}_{bf16}$.
5. Compute argmax agreement $\mathbb{A}$ (percentage of matching argmax).

### Golden Artifacts

Untuk PPL gate:

- `m6_ppl_corpus.json`: 100 documents.
- `ppl_bf16_baseline.json`: PPL BF16 per document.
- `ppl_quant_baseline.json`: PPL quant per document.
- `delta_ppl.json`: $\Delta\mathrm{PPL}$ per document.

### Regression Protection

- Commit `m6_ppl_corpus.json` + baseline PPL ke repo.
- Gate G-M6-3 harus PASS dengan corpus ini setiap build.
- Perubahan corpus requires approval dengan rationale.

## Error Handling

### Error Schema

```json
{
  "status": "error",
  "error": {
    "code": "M6_ERR_QUANT",
    "stage": "quantization",
    "message": "Quantization failed for tensor: NaN in scale computation",
    "details": {
      "tensor_name": "model.layers.0.self_attn.q_proj.weight",
      "group_id": 42,
      "scale": NaN
    }
  }
}
```

### Error Types

| Error Code          | Stage        | Description                            | Exit Code |
| ------------------- | ------------ | -------------------------------------- | --------- |
| `M6_ERR_INPUT`      | input        | Input-dir tidak ada, index tidak valid | 1         |
| `M6_ERR_QUANT`      | quantization | Quantization fail (NaN/INF/overflow)   | 2         |
| `M6_ERR_DEQUANT`    | dequant      | Dequantization fail (invalid data)     | 2         |
| `M6_ERR_OUTPUT`     | output       | Gagal atomic write                     | 3         |
| `M6_ERR_VALIDATION` | validation   | Output tidak lolos F11b validation     | 4         |

### Stage Failure Behavior

- **Input validation**: Batal seluruh quantization, cleanup, exit 1.
- **Quantization (tensor t)**: Batal pada tensor t, cleanup temporary, exit 2.
- **Dequantization**: Batal jika data invalid, cleanup, exit 2.
- **Output**: Atomic write rollback jika gagal, exit 3.
- **Validation**: Output ditulis tapi ditandai invalid, exit 4.

### Atomic Rollback

- Quant file: write ke temp → rename atomik → hapus temp jika gagal.
- Partial quant tidak pernah dibiarkan sebagai valid.
- Workdir: bersihkan temporary files sebelum exit.

## Quantization File Format

### Overall Structure

File quant 4-bit menggunakan format custom (bukan GGUF/llama.cpp):

```
quant_model.bin:
  [header 256 bytes]
  [tensor 0 metadata]
  [tensor 0 quantized data]
  [tensor 1 metadata]
  [tensor 1 quantized data]
  ...
  [tensor N metadata]
  [tensor N quantized data]
```

### Header Format

Header JSON + padding to 256 bytes:

```json
{
  "version": 1,
  "model": "qwen1.5-moe-a2.7b-chat",
  "quantization": {
    "format": "4-bit per-group",
    "group_size": 128,
    "scale_dtype": "FP16"
  },
  "num_tensors": 4659,
  "total_bytes": 7934542592
}
```

### Per-Tensor Metadata

Setiap tensor memiliki metadata:

```json
{
  "name": "model.layers.0.self_attn.q_proj.weight",
  "shape": [2048, 2048],
  "dtype": "BF16",
  "quantized_dtype": "4-bit",
  "group_size": 128,
  "num_groups": 32768,
  "scale_offset": 0,
  "data_offset": 512
}
```

### Quantized Data Layout

Per tensor:

```
[scales: num_groups × 2 bytes FP16]
[quantized_weights: num_elements × 0.5 bytes 4-bit]
```

- **Scales**: FP16 (2 bytes per group), row-major.
- **Quantized weights**: 4-bit per weight (2 weights per byte), packed.

### 4-bit Packing

4-bit values (-8..7) packed 2 per byte:

```
Byte b: [w1 (4 bits)][w0 (4 bits)]
```

- w0: bits 0-3 (first weight in group).
- w1: bits 4-7 (second weight in group).
- Endianness: little-endian (standard x86).

### Scale Computation

Per group G:

```python
s_g = ceil_to_fp16(max(|w_j|) / 7)  # FP16 scale; zero group => s_g=1, q=0
```

- Scale disimpan sebagai FP16 (2 bytes).
- Scale digunakan untuk dequantization: $\hat{w}_j = s_g \cdot q_j$.

### Validation

- Header JSON valid.
- Tensor name matches original BF16 model.
- Shape matches original BF16 model.
- group_size = 128 (default).
- Total file size = 256 + sum(tensor_metadata + tensor_data).
- Semua scales finite (tidak ada NaN/INF).
- Semua quantized values dalam range -8..7.

### Example: Single Tensor

Tensor shape [2048, 2048] = 4,194,304 elements:

- num_groups = 4,194,304 / 128 = 32,768 groups.
- scales: 32,768 × 2 bytes = 65,536 bytes.
- quantized_weights: 4,194,304 × 0.5 bytes = 2,097,152 bytes.
- total: 65,536 + 2,097,152 = 2,162,688 bytes.

Original BF16: 4,194,304 × 2 bytes = 8,388,608 bytes.
Compression ratio: 8,388,608 / 2,162,688 ≈ 3.88×.

## Workflow Diagram

```mermaid
flowchart TD
    A[Start: kimo quantize] --> B[Load BF16 model]
    B --> C{Input valid?}
    C -->|No| ERR1[Error: M6_ERR_INPUT, exit 1]
    C -->|Yes| D[Parse index.json]
    D --> E[Loop tensor 0..N]
    E --> F{All tensors done?}
    F -->|Yes| Z[Write header]
    F -->|No| G[Load tensor BF16]
    G --> H[Split to groups]
    H --> I[Compute scales]
    I --> J{Quantization OK?}
    J -->|No| ERR2[Error: M6_ERR_QUANT, exit 2]
    J -->|Yes| K[Quantize to 4-bit]
    K --> L[Write metadata]
    L --> M[Write quantized data]
    M --> N[Increment tensor]
    N --> E
    Z --> AA{Write OK?}
    AA -->|No| ERR3[Error: M6_ERR_OUTPUT, exit 3]
    AA -->|Yes| AB[Validate output]
    AB --> AC{Validation OK?}
    AC -->|No| ERR4[Error: M6_ERR_VALIDATION, exit 4]
    AC -->|Yes| AD[Success: metrics logged]
    AD --> AE[End]

    ERR1 --> END1[Cleanup workdir]
    ERR2 --> END2[Cleanup workdir]
    ERR3 --> END3[Cleanup workdir]
    ERR4 --> END4[Cleanup workdir]
    END1 --> ZE[End]
    END2 --> ZE
    END3 --> ZE
    END4 --> ZE
```

## Rumus (F11, grup G=128, skala fp16)

$$a_g=\max|w_j|,\qquad s_g\ge a_g/7$$

Pilih $s_g$ sebagai nilai FP16 finite terkecil yang memenuhi ketaksamaan itu; grup nol memakai $s_g=1$, $q=0$.

$$q_j=\mathrm{clip}(\mathrm{round}(w_j/s_g),-7,7),\qquad \hat w_j=s_gq_j \tag{F11a}$$
$$\mathrm{MSE},\ \varepsilon_{rel},\ \text{bytes}≈N·bpw_{eff}/8 \tag{F11b}$$

$bpw_{eff}=4+16/128=\mathbf{4{,}125}$ (+metadata). $N_{total}=14{,}32$ B → file ≈ **7,384 GB**.

Property FP32: $|w_j-\hat w_j| \le s_g/2$. Pembulatan scale FP16 selalu ke atas agar nilai maksimum tidak tersaturasi.

Kualitas F12: $\mathrm{PPL}=\exp(-1/N_{pred}\sum_{t\in\mathcal P}\ln p)$, $\Delta\mathrm{PPL}=\mathrm{PPL}_{quant}-\mathrm{PPL}_{bf16}$.

Dampak ke F5: $B_{tok}^{4bit}≈2{,}0668\text{B}×4{,}125/8≈\mathbf{1{,}066}$ GB; $W_{stream,quant}≈7{,}06$ GB → $\rho_B≈\rho_C≈0{,}4248$. Pada $BW_{RAM}=15$ GB/s dan $BW_{SSD}=3$ GB/s, $BW_{eff}≈4{,}544$ GB/s → forecast serial ≈0,285 s/token ≈3,51 tok/s (forecast, bukan acceptance).

## Gate

| Gate   | Kriteria                    | Threshold                                                  | Metode         |
| ------ | --------------------------- | ---------------------------------------------------------- | -------------- |
| G-M6-1 | error per tensor (grup 128) | $\varepsilon_{rel} \le 10^{-2}$                            | semua tensor   |
| G-M6-2 | ukuran file                 | \|pred-meas\|/meas ≤ 10%                                   | `du` vs F11b   |
| G-M6-3 | kualitas end-to-end         | $\Delta\mathrm{PPL} \le +0{,}5 \wedge \mathbb{A} \ge 95\%$ | corpus 100×256 |

## Testing

- P: quant roundtrip, property $s_g/2$.
- O: ΔPPL + argmax agreement vs BF16.
- B: ukuran + tok/s awal 4-bit.
- Oracle tetap fp32; quant tidak boleh jadi ground truth (D4).

## Integration Tests

### Test Matrix

| Test ID  | Scenario                                    | Expected                        | Priority |
| -------- | ------------------------------------------- | ------------------------------- | -------- | --------- | ------ |
| IT-M6-1  | Happy path: quantize BF16 → 4-bit           | Exit 0, G-M6-1 PASS             | HIGH     |
| IT-M6-2  | Input-dir tidak ada                         | Exit 1, error M6_ERR_INPUT      | HIGH     |
| IT-M6-3  | Invalid group-size (not power of 2)         | Exit 1, error M6_ERR_INPUT      | HIGH     |
| IT-M6-4  | Quantization fail (NaN in scale)            | Exit 2, error M6_ERR_QUANT      | HIGH     |
| IT-M6-5  | Dequantization fail (invalid data)          | Exit 2, error M6_ERR_DEQUANT    | HIGH     |
| IT-M6-6  | Output validation fail (file size mismatch) | Exit 4, error M6_ERR_VALIDATION | HIGH     |
| IT-M6-7  | PPL measurement                             | Exit 0, G-M6-3 PASS             | HIGH     |
| IT-M6-8  | Deterministic output                        | SHA-256 match di 2 run          | MEDIUM   |
| IT-M6-9  | Custom group-size (64)                      | Exit 0, file size ±10%          | MEDIUM   |
| IT-M6-10 | Property test FP32                          | fp32(w) - ŵ^(32)                | ≤ s_g/2  | 100% pass | MEDIUM |

### Test Automation

`tests/integration/test_m6_quantize.sh`:

```bash
#!/bin/bash
set -e

# Setup
INPUT_DIR="/tmp/test_model_bf16"
OUTPUT_DIR="/tmp/test_model_4bit"
WORKDIR="/tmp/test_work"

# IT-M6-1: Happy path
kimo quantize \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --group-size 128 \
  --workdir "$WORKDIR"
# Expect exit 0

# IT-M6-2: Input-dir tidak ada
kimo quantize \
  --input-dir "/nonexistent" \
  --output-dir "$OUTPUT_DIR" \
  --workdir "$WORKDIR" || true
# Expect exit 1

# IT-M6-3: Invalid group-size
kimo quantize \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --group-size 100 \
  --workdir "$WORKDIR" || true
# Expect exit 1

# IT-M6-7: PPL measurement
python tools/oracle/oracle_ppl.py \
  --model-bf16 "$INPUT_DIR" \
  --model-quant "$OUTPUT_DIR" \
  --corpus "tools/fixtures/m6_ppl_corpus.json" \
  --output-report "$WORKDIR/ppl_report.json"
# Expect ΔPPL ≤ +0.5, argmax agreement ≥ 95%

# IT-M6-8: Deterministic
OUTPUT1="$OUTPUT_DIR/quant_run1.bin"
OUTPUT2="$OUTPUT_DIR/quant_run2.bin"
kimo quantize \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --group-size 128 \
  --workdir "$WORKDIR"
SHA1=$(sha256sum "$OUTPUT_DIR/quant_model.bin" | cut -d' ' -f1)
kimo quantize \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --group-size 128 \
  --workdir "$WORKDIR"
SHA2=$(sha256sum "$OUTPUT_DIR/quant_model.bin" | cut -d' ' -f1)
[ "$SHA1" = "$SHA2" ] || exit 1
```

### Regression Golden Outputs

- Commit `m6_ppl_corpus.json` + baseline PPL ke repo.
- Setiap build: jalankan IT-M6-7 → bandingkan dengan golden.
- Jika mismatch: investigasi, fix, re-commit golden dengan rationale.

### Negative Path Coverage

- Error codes 1-4 semua teruji.
- Error JSON schema valid di semua failure paths.
- Cleanup workdir verified setiap error (tidak ada orphan files).

## Dequant Kernel Specification

### Kernel Function

Dequant kernel mengkonversi 4-bit weights + FP16 scales → BF16 output untuk layer forward.

### Input

Per tensor:

- `scales`: [num_groups] FP16 (2 bytes per group).
- `quantized_weights`: [num_elements] 4-bit packed (2 weights per byte).
- `group_size`: 128 (default).

### Output

- `dequantized_weights`: [num_elements] BF16 (2 bytes per element).

### Algorithm

```python
# Pseudocode for dequant kernel
def dequant_kernel(scales, quantized_weights, group_size):
    num_elements = len(quantized_weights) * 2  # 2 weights per byte
    num_groups = num_elements // group_size
    dequantized = zeros(num_elements, dtype=BF16)

    for g in range(num_groups):
        scale = scales[g]  # FP16
        start = g * group_size
        end = start + group_size

        for i in range(start, end):
            # Unpack 4-bit value
            byte_idx = i // 2
            bit_offset = (i % 2) * 4
            if bit_offset == 0:
                q = quantized_weights[byte_idx] & 0x0F  # lower 4 bits
            else:
                q = (quantized_weights[byte_idx] >> 4) & 0x0F  # upper 4 bits

            # Convert signed 4-bit to int
            if q >= 8:
                q = q - 16  # signed: -8..7

            # Dequantize
            dequantized[i] = scale * q  # BF16

    return dequantized
```

### Vectorization

Kernel harus vectorized untuk efisiensi:

- **SIMD**: Pack 4-bit weights → 8-bit chunks → unpack → multiply scales.
- **Batch**: Process multiple groups per iteration.
- **Cache-friendly**: Access scales and quantized weights sequentially.

### Memory Layout

Input layout:

```
scales: [G] FP16 contiguous
quantized_weights: [N/2] bytes contiguous
```

Output layout:

```
dequantized_weights: [N] BF16 contiguous
```

### Validation

- Semua scales finite (tidak ada NaN/INF).
- Semua quantized values dalam range -8..7 (setelah unpack).
- Output BF16 finite (tidak ada overflow).
- Dequantization harus reversible: dequant(quant(w)) ≈ w (dengan error ≤ F11 threshold).

### Performance Considerations

- Dequantization adalah bottleneck di M6 (setiap layer decode).
- Target: dequantization overhead ≤ 10% dari total decode time.
- Optimization: SIMD vectorization, batch processing, cache blocking.

### Integration with Layer Forward

Dequant kernel di-integrasikan ke layer forward (M2/M3):

1. Pread quantized weights dari disk.
2. Dequant ke BF16 buffer (on-the-fly).
3. Gunakan BF16 buffer untuk layer forward.
4. Discard BF16 buffer setelah layer forward (streaming).

## Quantize CLI Output Examples

### Success Output

```json
{
  "status": "success",
  "run_id": "M6-20250115-001",
  "model": "qwen1.5-moe-a2.7b-chat",
  "input_format": "BF16",
  "output_format": "4-bit per-group",
  "group_size": 128,
  "num_tensors": 4659,
  "metrics": {
    "walltime_sec": 245.3,
    "input_bytes": 30660512768,
    "output_bytes": 7934542592,
    "compression_ratio": 3.86,
    "avg_epsilon_rel": 0.0087,
    "max_epsilon_rel": 0.0095,
    "min_epsilon_rel": 0.0072
  }
}
```

### Error Output (Input Invalid)

```json
{
  "status": "error",
  "error": {
    "code": "M6_ERR_INPUT",
    "stage": "input",
    "message": "Input directory does not exist",
    "details": {
      "input_dir": "/nonexistent"
    }
  }
}
```

### Error Output (Quantization Fail)

```json
{
  "status": "error",
  "error": {
    "code": "M6_ERR_QUANT",
    "stage": "quantization",
    "message": "Quantization failed for tensor: NaN in scale computation",
    "details": {
      "tensor_name": "model.layers.0.self_attn.q_proj.weight",
      "group_id": 42,
      "scale": NaN
    }
  }
}
```

### Error Output (Validation Fail)

```json
{
  "status": "error",
  "error": {
    "code": "M6_ERR_VALIDATION",
    "stage": "validation",
    "message": "Output file size mismatch: expected 7934542592, got 7934542593",
    "details": {
      "expected_bytes": 7934542592,
      "actual_bytes": 7934542593,
      "diff_bytes": 1
    }
  }
}
```

## Per-Tensor Error Breakdown

### Error Report Schema

Optional per-tensor error breakdown untuk debugging quantization quality:

```json
{
  "run_id": "M6-20250115-001",
  "model": "qwen1.5-moe-a2.7b-chat",
  "group_size": 128,
  "summary": {
    "num_tensors": 4659,
    "max_epsilon_rel": 0.0095,
    "avg_epsilon_rel": 0.0087,
    "min_epsilon_rel": 0.0072,
    "num_high_error": 23
  },
  "tensors": [
    {
      "name": "model.layers.0.self_attn.q_proj.weight",
      "shape": [2048, 2048],
      "epsilon_rel": 0.0089,
      "mse": 0.000156,
      "property_ok": true,
      "max_abs_error": 0.0042,
      "num_groups": 32768
    },
    {
      "name": "model.layers.12.mlp.experts.5.w1.weight",
      "shape": [1408, 2048],
      "epsilon_rel": 0.0095,
      "mse": 0.000198,
      "property_ok": true,
      "max_abs_error": 0.0051,
      "num_groups": 22528
    },
    ...
  ]
}
```

### Metrics per Tensor

- `name`: tensor name (matches BF16 model).
- `shape`: tensor shape.
- `epsilon_rel`: relative error $\varepsilon_{rel} = \sqrt{\text{MSE}} / \sqrt{\text{var}(w)}$.
- `mse`: mean squared error.
- `property_ok`: property test $|w - \hat{w}| \le s_g/2$ pass/fail.
- `max_abs_error`: maximum absolute error in tensor.
- `num_groups`: number of quantization groups.

### Debugging Use Cases

- Identifikasi tensor dengan epsilon_rel tinggi (threshold: > 0.01).
- Identifikasi tensor yang gagal property test (property_ok = false).
- Correlate high error dengan tensor layer (attention vs MoE).
- Investigasi tensor dengan outlier scales (scale terlalu besar/kecil).

### High Error Threshold

Tensors dengan epsilon_rel > 0.01 flagged sebagai "high error":

- Cek jika weight distribution extreme (heavy tails).
- Cek jika scale computation menyebabkan overflow/underflow.
- Pertimbangkan custom group-size untuk problematic tensors.

### Optional Flag

```bash
kimo quantize \
  --input-dir /models/qwen-moe-bf16 \
  --output-dir /models/qwen-moe-4bit \
  --group-size 128 \
  --error-report /work/per_tensor_error.json
```

## Security

- SEC-4: dequant alloc tervalidasi; tolak grup/skala liar.
- SEC-6: golden quant ter-versioning terpisah dari BF16.

## DoD

### Gate Requirements

- [ ] G-M6-1 quantization error ≤ 1e-2 (max epsilon_rel per tensor)
- [ ] G-M6-2 file size ±10% vs prediction (7,384 GB)
- [ ] G-M6-3 PPL quality: ΔPPL ≤ +0.5, argmax agreement ≥ 95%

### CLI Implementation

- [ ] `kimo quantize` subcommand terimplementasi dengan semua argumen
- [ ] Input validation: input-dir existence, index validity, group-size validity
- [ ] Exit codes: 0 (success), 1-4 (error per stage), semuanya teruji
- [ ] Output JSON dengan run_id, metrics, per-tensor epsilon_rel tercommit schema

### Oracle & Fixture

- [ ] `tools/oracle/oracle_quant.py` menghasilkan quantization roundtrip report
- [ ] Oracle deterministik (group-size=128, F11a property verified)
- [ ] `tools/fixtures/m6_ppl_corpus.json` tercommit dengan 100 documents
- [ ] PPL baseline BF16 dan quant tercommit untuk regression protection
- [ ] Generation script `tools/fixtures/generate_m6_ppl.py` teruji
- [ ] Per-tensor error breakdown report terimplementasi (optional flag)

### Error Handling

- [ ] Error schema JSON terimplementasi untuk semua 5 error types
- [ ] Stage failure handling: input, quantization, dequantization, output, validation
- [ ] Atomic rollback: temp file → rename atomik → cleanup jika gagal
- [ ] Cleanup temporary files sebelum exit (tidak ada workdir pollution)

### Quantization File Format

- [ ] Header JSON format terdokumentasi (version, model, quantization, num_tensors, total_bytes)
- [ ] Per-tensor metadata format terdokumentasi (name, shape, dtype, group_size, offsets)
- [ ] Quantized data layout terdokumentasi (scales FP16 + 4-bit weights packed)
- [ ] 4-bit packing specification (2 weights per byte, little-endian)
- [ ] Scale computation specification (max|w|/7 per group)
- [ ] Validation checks: header, tensor name, shape, group_size, file size, finite scales

### Quantization Algorithm

- [ ] F11a algorithm terimplementasi: s_g = max|w|/7, q = clip(round(w/s_g), -8, 7), ŵ = s_g q
- [ ] Group splitting: 128 weights per group
- [ ] Scale computation per group (FP16)
- [ ] Quantization: BF16 → 4-bit clip(-7, 7)
- [ ] Dequantization: 4-bit × scale → BF16
- [ ] Property verification FP32: |fp32(w) - ŵ^(32)| ≤ s_g/2 untuk semua weights

### Dequant Kernel

- [ ] Dequant kernel terimplementasi (scales FP16 + 4-bit weights → BF16 output)
- [ ] Kernel vectorized untuk efisiensi
- [ ] Dequant alloc tervalidasi (SEC-4)
- [ ] Tolak grup/skala liar (SEC-4)

### Numerical Correctness

- [ ] G-M6-1 quantization error ≤ 1e-2 (max epsilon_rel per tensor)
- [ ] Property test FP32: |fp32(w) - ŵ^(32)| ≤ s_g/2 untuk semua weights
- [ ] Roundtrip error: BF16 → 4-bit → BF16 ε_rel ≤ 1e-2

### PPL Quality

- [ ] G-M6-3 PPL measurement terimplementasi (corpus 100×256)
- [ ] ΔPPL ≤ +0.5 (PPL_quant - PPL_bf16)
- [ ] Argmax agreement ≥ 95% (percentage of matching argmax)
- [ ] PPL baseline BF16 dan quant tercommit

### Integration Tests

- [ ] Happy path: quantize BF16 → 4-bit → validate → PASS
- [ ] Input-dir tidak ada: error M6_ERR_INPUT, exit 1
- [ ] Invalid group-size: error M6_ERR_INPUT, exit 1
- [ ] Quantization fail (NaN): error M6_ERR_QUANT, exit 2
- [ ] Dequantization fail: error M6_ERR_DEQUANT, exit 2
- [ ] Output validation fail: error M6_ERR_VALIDATION, exit 4
- [ ] Deterministic output: same input → same output (SHA-256 match)

### Security Tests

- [ ] SEC-4: dequant alloc tervalidasi (tidak ada OOM)
- [ ] SEC-4: tolak grup/skala liar (group-size > 256, scale INF/NAN)
- [ ] SEC-6: golden quant ter-versioning terpisah dari BF16
- [ ] Model directory read-only saat quantization
- [ ] Output atomic: tidak ada partial quant valid jika gagal

### Performance Baseline

- [ ] Quantization walltime tercatat
- [ ] File size tercatat (≈7,38 GB ±10%)
- [ ] Compression ratio tercatat (≈3.88×)
- [ ] Per-tensor epsilon_rel tercatat (max, avg, min)
- [ ] Tok/s awal 4-bit tercatat (decode dengan quant weights)

### Reporting & Artifacts

- [ ] Laporan quantization tercommit (walltime, file size, compression ratio, epsilon_rel)
- [ ] Run ID tercatat per run (format: M6-YYYYMMDD-NNN)
- [ ] Quantization file format terdokumentasi (grup, skala, layout)
- [ ] PPL corpus + baseline tercommit
- [ ] Golden quant ter-versioning terpisah dari BF16 (SEC-6)

## Wave Note

Lihat implementasi notes di: <ref_file file="../../scratch/wave/m6/README.md" />
