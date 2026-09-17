# Spesifikasi Format File Kuantisasi 4-bit (`quant_model.bin`)

> Milestone M6 — Gelombang 1 (`m6-w1-format.md`).
> Standar format custom disk-streaming-moe-engine (ADR D6 — "Menemukan kembali GGUF").
> Status: **FROZEN (DIBEKUKAN)**.
>
> **Amendemen A1 (audit M6, disetujui pemilik):** nibble `0b1000` adalah
> reserved/invalid — encoder dilarang memancarkan, decoder wajib menolak
> (sebelumnya "valid tapi tak dipancarkan"); $G \in \{32,64,128,256\}$
> (satu-satunya definisi, menggantikan "pangkat dua"); $N \% G == 0$ wajib
> (opsi A — formula `ceil` di bawah berimpit dengan pembagian eksak untuk
> input konform). SSOT: `../milestones/M6-quantizer.md` § SSOT Kontrak M6.

---

## 1. Struktur Berkas (File Structure)

Berkas biner kuantisasi (`quant_model.bin` atau shard per-model) mengadopsi struktur byte-exact yang deterministik:

```text
+-------------------------------------------------------------+
| Header Berkas (Tepat 256 byte, JSON terisi padding)         |
+-------------------------------------------------------------+
| Blok Tensor 0:                                              |
|   - Panjang Metadata (4 byte, u32 LE)                       |
|   - JSON Metadata Tensor 0                                  |
|   - Skala FP16 (num_groups × 2 byte)                        |
|   - Bobot Terkuantisasi 4-bit (num_elements × 0.5 byte)     |
+-------------------------------------------------------------+
| Blok Tensor 1:                                              |
|   - Panjang Metadata (4 byte, u32 LE)                       |
|   - JSON Metadata Tensor 1                                  |
|   - Skala FP16 (num_groups × 2 byte)                        |
|   - Bobot Terkuantisasi 4-bit (num_elements × 0.5 byte)     |
+-------------------------------------------------------------+
| ...                                                         |
+-------------------------------------------------------------+
| Blok Tensor N-1                                             |
+-------------------------------------------------------------+
```

Total ukuran file:
$$\text{Total File Size} = 256 + \sum_{i=0}^{N-1} \Big( 4 + \text{len}(\text{meta\_json}_i) + \text{payload\_size}_i \Big)$$
dengan:
$$\text{payload\_size}_i = (G_{num, i} \times 2) + \left\lceil \frac{E_{num, i}}{2} \right\rceil$$

---

## 2. Header Berkas (File Header — 256 Byte)

Header berada pada offset byte $0$ hingga $255$. Header ditulis dalam format UTF-8 JSON yang dipadatkan dengan karakter spasi (`0x20`) hingga mencapai panjang eksak **256 byte**.

### Skema JSON Header

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

### Kolom Normatif Header

| Field                      | Tipe      | Deskripsi                                                     |
| :------------------------- | :-------- | :------------------------------------------------------------ |
| `version`                  | `integer` | Versi spesifikasi format (harus bernilai `1`).                |
| `model`                    | `string`  | Identitas arsitektur model target.                            |
| `quantization.format`      | `string`  | Format kuantisasi (harus `"4-bit per-group"`).                |
| `quantization.group_size`  | `integer` | Ukuran grup kuantisasi: wajib ∈ $\{32, 64, 128, 256\}$ (default $128$; himpunan izin SSOT M6). |
| `quantization.scale_dtype` | `string`  | Tipe data skala FP16 IEEE 754 (harus `"FP16"`).               |
| `num_tensors`              | `integer` | Jumlah total tensor yang tersimpan di dalam berkas.           |
| `total_bytes`              | `integer` | Ukuran total berkas dalam byte untuk validasi integritas.     |

---

## 3. Metadata Per-Tensor (Per-Tensor Metadata)

Setiap blok tensor diawali dengan panjang metadata berupa `u32` little-endian (4 byte), diikuti oleh string UTF-8 JSON metadata:

### Skema JSON Metadata

```json
{
  "name": "model.layers.0.self_attn.q_proj.weight",
  "shape": [2048, 2048],
  "dtype": "BF16",
  "quantized_dtype": "4-bit",
  "group_size": 128,
  "num_groups": 32768,
  "scale_offset": 0,
  "data_offset": 65536
}
```

### Kolom Normatif Metadata

| Field             | Tipe         | Deskripsi                                                                          |
| :---------------- | :----------- | :--------------------------------------------------------------------------------- |
| `name`            | `string`     | Nama tensor sesuai penamaan state dict safetensors asli.                           |
| `shape`           | `array[int]` | Dimensi tensor asli (misal `[2048, 2048]`).                                        |
| `dtype`           | `string`     | Tipe data asli tensor sebelum kuantisasi (biasanya `"BF16"`).                      |
| `quantized_dtype` | `string`     | Tipe data kuantisasi (harus `"4-bit"`).                                            |
| `group_size`      | `integer`    | Ukuran elemen per grup ($G = 128$).                                                |
| `num_groups`      | `integer`    | Jumlah grup: $G_{num} = \lceil \text{num\_elements} / G \rceil$.                   |
| `scale_offset`    | `integer`    | Offset byte awal skala relatif terhadap payload tensor ($0$).                      |
| `data_offset`     | `integer`    | Offset byte awal bobot 4-bit relatif terhadap payload tensor ($G_{num} \times 2$). |

---

## 4. Layout Data Kuantisasi (Quantized Data Layout)

Data biner per tensor disusun secara kontigu tanpa padding celah:

```text
+-------------------------------------------------------------+
| Skala Grup FP16 (num_groups × 2 byte, little-endian)        |
+-------------------------------------------------------------+
| Bobot 4-bit Ter-pack (ceil(num_elements / 2) byte)          |
+-------------------------------------------------------------+
```

1. **Skala FP16 (Scales)**:
   - Disimpan dalam format IEEE 754 half-precision float (1 sign bit, 5 exponent bits, 10 mantissa bits, 2 byte per grup).
   - Panjang buffer skala: $G_{num} \times 2$ byte.
2. **Bobot 4-bit Ter-pack (Packed Weights)**:
   - Setiap byte mengemas tepat 2 nilai bobot 4-bit simetris $w \in [-7, 7]$; nibble `0x8` reserved/invalid (decoder wajib menolak).
   - Panjang buffer bobot: $\lceil N / 2 \rceil$ byte.

---

## 5. Spesifikasi Packing 4-bit Little-Endian (Bitwise Packing)

Dua bobot 4-bit berurutan $(w_0, w_1)$ dikemas ke dalam satu byte $B$ dengan skema Little-Endian:

```text
Bit:    7   6   5   4   3   2   1   0
      +---------------+---------------+
Byte: |      w1       |      w0       |
      +---------------+---------------+
        (High Nibble)   (Low Nibble)
```

### Aturan Packing (Encoding)

Untuk nilai integer bertanda $w_0, w_1 \in [-7, 7]$ (nilai $-8$ dilarang di sisi encoder):
$$b = \Big( (w_1 \ \& \ \text{0x0F}) \ll 4 \Big) \ \Big| \ (w_0 \ \& \ \text{0x0F})$$

### Aturan Unpacking (Decoding)

Dari byte unsigned $b \in [0, 255]$:
$$w_0^{raw} = b \ \& \ \text{0x0F}, \qquad w_0 = \begin{cases} w_0^{raw} - 16 & \text{jika } w_0^{raw} \ge 8 \\ w_0^{raw} & \text{lainnya} \end{cases}$$
$$w_1^{raw} = (b \gg 4) \ \& \ \text{0x0F}, \qquad w_1 = \begin{cases} w_1^{raw} - 16 & \text{jika } w_1^{raw} \ge 8 \\ w_1^{raw} & \text{lainnya} \end{cases}$$

Kedua nilai hasil decoding terjamin berada pada rentang tertutup $[-7, 7]$; nibble `0x8` tidak pernah dihasilkan decoder yang konform (wajib ditolak sebagai reserved/invalid).

---

## 6. Formula Kuantisasi Skala F11a

Untuk setiap grup ke-$g$ yang berisi elemen $\{w_j\}_{j \in G}$:

$$a_g = \max_{j \in G} |w_j|$$

1. **Perhitungan Skala $s_g$**:
   - Jika $a_g = 0$ (grup bernilai nol murni): $s_g = 1{,}0$ dan seluruh kode kuantisasi $q_j = 0$.
   - Jika $a_g > 0$: $s_g$ dipilih sebagai nilai FP16 terkecil yang memenuhi $s_g \ge a_g / 7$ (pembulatan ke atas / ceil FP16 untuk mencegah saturasi).
2. **Kuantisasi Simetris**:
   $$m_j = \mathrm{round}(w_j / s_g)$$
   $$q_j = \mathrm{clamp}(m_j, -7, 7)$$
    _(Catatan Amendemen A1: nibble $-8$ (`0x8`) adalah reserved/invalid —
    encoder dilarang memancarkan, decoder wajib menolak; bukan "valid tapi
    tak dipancarkan".)_
3. **Dequantisasi Rekonstruksi**:
   $$\hat{w}_j = s_g \cdot q_j$$
   Property FP32: $|w_j - \hat{w}_j| \le s_g / 2$.

---

## 7. Perhitungan Manual: Contoh Tensor `[2048, 2048]`

Sebagai verifikasi kebenaran formula pada tensor proyeksi $W \in \mathbb{R}^{2048 \times 2048}$:

1. **Jumlah Elemen**:
   $$N = 2048 \times 2048 = 4.194.304\text{ elemen}$$
2. **Jumlah Grup ($G = 128$)**:
   $$G_{num} = \frac{4.194.304}{128} = 32.768\text{ grup}$$
3. **Ukuran Skala FP16**:
   $$B_{scales} = 32.768 \times 2\text{ byte} = 65.536\text{ byte} = 64\text{ KiB}$$
4. **Ukuran Bobot Ter-pack 4-bit**:
   $$B_{weights} = \frac{4.194.304}{2}\text{ byte} = 2.097.152\text{ byte} = 2048\text{ KiB} = 2\text{ MiB}$$
5. **Total Payload Data Tensor**:
   $$B_{total\_payload} = B_{scales} + B_{weights} = 65.536 + 2.097.152 = \mathbf{2.162.688\text{ byte}}$$
6. **Perbandingan Terhadap Bobot Asli BF16**:
   $$B_{bf16} = 4.194.304 \times 2\text{ byte} = 8.388.608\text{ byte} = 8\text{ MiB}$$
   $$\text{Rasio Kompresi} = \frac{8.388.608}{2.162.688} = \frac{8}{2{,}0625} \approx \mathbf{3{,}8788\times} \approx \mathbf{3{,}88\times}$$

---

## 8. Aturan Validasi Integritas Format

Sebuah berkas `.bin` kuantisasi sah jika dan hanya jika:

1. **Header**:
   - Berukuran tepat 256 byte.
   - Mengandung field wajib `version == 1`, `quantization.group_size == 128`, `quantization.scale_dtype == "FP16"`.
2. **Metadata**:
   - `name` non-empty dan sesuai model safetensors.
   - `shape` berdimensi positif dan hasil kali dimensi cocok dengan $N$.
   - $G_{num} = \lceil N / G \rceil$.
   - `scale_offset == 0` dan `data_offset == G_{num} * 2`.
3. **Payload**:
   - Seluruh nilai skala FP16 bertanda positif dan finite ($\text{NaN}$ dan $\pm\infty$ ditolak).
   - Seluruh nilai kuantisasi ter-pack berada pada rentang $[-7, 7]$; nibble `0x8` wajib ditolak.
   - $N \% G == 0$ (tail group ditolak); `group_size` ∈ $\{32, 64, 128, 256\}$.
4. **Ukuran Berkas**:
   - Ukuran fisik berkas sama persis dengan `total_bytes` di header.
