# M7 — O_DIRECT + LRU Cache Expert (Pola kimi-k3-in-c)

> Proyek: `disk-streaming-moe-engine`. Fase: **Trial (rekayasa, puncak performa trial)**. Index: `../README.md`.

| Field       | Nilai                                                                                 |
| ----------- | ------------------------------------------------------------------------------------- |
| Deliverable | Reader O_DIRECT + LRU cache expert dengan model terkalibrasi                          |
| Komponen    | C6 io_direct + LRU, C3 evolusi pread, C7 benchmark                                    |
| Prasyarat   | M5, M6 hijau (butuh KV + quant untuk $B_{tok}^{4bit}$)                                |
| Next        | `M8-gdn.md`                                                                           |
| Gate        | G-M7-1..G-M7-7 (correctness-first: G-M7-6/7 wajib PASS sebelum verdict G-M7-1..5 sah) |
| Rumus       | F13, F5 (koreksi F9), F3b-quant, F16, F17                                             |

## Tujuan

Buktikan kebenaran dulu (M0–M6), optimasi I/O kemudian (D7): bypass page cache untuk kontrol penuh + cache expert panas di RAM, mengikuti pola `kimi-k3-in-c`.

## Arsitektur Empat Lapis (normatif)

```
┌────────────────────┐
│ Decode scheduler   │
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│ Expert cache       │
│ quantized only     │
│ LRU + pin budget   │
└─────────┬──────────┘
          │ miss
┌─────────▼──────────┐
│ Direct I/O reader  │
│ aligned requests   │
│ queue depth Q      │
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│ Quant file         │
│ record framing     │
└────────────────────┘
```

Invarian (dilanggar = bug, bukan toleransi):

1. Setiap physical read alignment-valid (vs `dio_alignment`).
2. Setiap cache entry quantized (dequant per use; BF16 tak di-cache).
3. Satu `(layer, expert)` paling banyak satu in-flight load (single-flight).
4. `sum(pinned) ≤ pin_budget < capacity` (§ Pin Budget Invariant).
5. QD = outstanding I/O aktual (terverifikasi counter, bukan klaim).
6. Fallback buffered hanya untuk platform unsupported (fase probe).
7. Error format/alignment pasca-probe tak pernah fallback diam-diam.

## O_DIRECT Reader Specification

### O_DIRECT Basics

O_DIRECT bypasses page cache untuk I/O langsung ke storage (bypass kernel buffer). Ini memberikan kontrol penuh atas buffer alignment dan memory management.

### Alignment Requirements (Triple Alignment)

O_DIRECT requires triple alignment (buffer, offset, length) — tetapi modulusnya
**bukan** konstanta universal, melainkan `dio_alignment` hasil discovery
(§ DIO Alignment Discovery):

- **Buffer alignment**: alamat buffer kelipatan `dio_alignment`.
- **Offset alignment**: file offset kelipatan `dio_alignment`.
- **Length alignment**: read length kelipatan `dio_alignment`.

Misaligned O_DIRECT → `EINVAL` atau fallback ke buffered I/O (tergantung filesystem).

### DIO Alignment Discovery (normatif)

Jangan asumsikan 4 KB. `dio_alignment` = requirement aktual path model, ditemukan
saat startup per model path:

1. Kandidat menaik: `[512, 4096]` — pilih yang terkecil lolos.
2. Untuk tiap $A$: `aligned_alloc(A)` buffer $A$ byte; `open(path, O_RDONLY|O_DIRECT)`; `pread` $A$ byte di offset 0.
   - Sukses ($n == A$) → `dio_alignment = A`; selesai.
   - `EINVAL` → tutup fd, coba $A$ berikutnya.
   - Error lain (`EIO`/`EACCES`) → fail sesuai tabel error.
3. Semua kandidat `EINVAL` → O_DIRECT unsupported di path ini → fallback buffered + warning (kebijakan existing).
4. `dio_alignment` + hasil probe wajib di-log per run (§ Filesystem Logging).

Aturan relasi (fail-fast `M7_ERR_ODIRECT_ALIGNMENT` saat init bila dilanggar):

- `io_block_size >= dio_alignment` dan `io_block_size % dio_alignment == 0`.
- `--block-size` adalah granularitas I/O yang diminta, bukan hukum alignment storage.

### Format-vs-Platform Failure Policy (normatif)

Fallback diam-diam memalsukan gate hijau. Maka penyebab `EINVAL` wajib dibedakan berdasarkan fase — probe-lah yang memisahkan kemampuan platform dari bug format:

1. **Fase probe** (`EINVAL` saat discovery): platform/path tidak mendukung O_DIRECT → buffered fallback **diizinkan** (satu-satunya kasus fallback yang sah) + warning.
2. **Kontrak Aligned Physical Span + Staging Buffer (Format M6 v1)**:
   Format M6 v1 secara eksplisit mendefinisikan layout unaligned (`256B header` + `4B meta_len` + `JSON variable length` + `payload`). Maka offset payload di dalam file (`payload_offset`) tidak diasumsikan kelipatan `dio_alignment`. Reader M7 **DILARANG KERAS** menolak berkas M6 v1 yang unaligned. Sebaliknya, reader M7 mengimplementasikan **translasi physical span selaras + logical slice**:
   - Hitung batas fisik yang selaras dengan $A = \text{dio\_alignment}$:
     $$\text{phys\_start} = \lfloor \text{offset} / A \rfloor \times A$$
     $$\text{phys\_end} = \lceil (\text{offset} + \text{length}) / A \rceil \times A$$
     $$\text{phys\_len} = \text{phys\_end} - \text{phys\_start}$$
   - Alokasi buffer staging selaras dengan ukuran $\text{phys\_len}$ (otomatis kelipatan $A$).
   - Baca fisik via `pread_o_direct_span(fd, staging_buf, phys_start, phys_len, A)`.
   - Ekstrak slice logis: `staging_buf[offset - phys_start : offset - phys_start + length]`.
   - Pelanggaran `M7_ERR_FORMAT_ALIGNMENT` hanya terjadi jika kalkulasi span fisik melanggar batas berkas tanpa penanganan EOF yang sah, atau jika format baru (misal M6 v2 aligned container) yang mengklaim direct zero-copy alignment gagal memenuhi kelipatan `dio_alignment`.
3. **Runtime pasca-probe** (`EINVAL` pada read physical span): berarti ada kegagalan internal kernel I/O atau hardware alignment fault → hard fail `M7_ERR_FORMAT_ALIGNMENT`. **Fallback dilarang.**
4. Intinya: fallback hanya untuk kapabilitas platform, **tidak pernah** untuk bug format internal kita sendiri.

### Block Size

- `--block-size`: granularitas I/O yang diminta (512/4096/8192, default 4096).
- 4 KB adalah default yang baik untuk NVMe modern, bukan requirement universal.
- Engine mengalokasi buffer dengan `alloc_size = round_up(size, dio_alignment)` dan membaca kelipatan `dio_alignment`.

### Short Read Handling

O_DIRECT `pread` bisa mengembalikan fewer bytes than requested (short read).
Bahaya spesifik O_DIRECT: bila $n$ bukan kelipatan `dio_alignment`, cursor
`offset + total_read` / `length - total_read` menjadi misaligned — read lanjutan
TIDAK BOLEH diterbitkan (akan `EINVAL`, dan itu bukan bug format). Maka loop
naif dilarang; yang normatif adalah baca berbasis span selaras:

```python
# Pseudocode: span-based read loop & logical slice (O_DIRECT-safe)
# A = dio_alignment hasil discovery (konteks reader, bukan parameter I/O)

def pread_o_direct_span(fd, buffer, phys_start, phys_len, A, max_span_retries=3):
    """Membaca span fisik yang DIJAMIN selaras (phys_start % A == 0, phys_len % A == 0)."""
    total_read = 0
    span_retries = 0
    while total_read < phys_len:
        n = pread(fd, buffer[total_read:], phys_start + total_read, phys_len - total_read)
        if n == 0:
            break  # EOF
        if n < 0:
            return error  # EAGAIN/EINTR -> retry; EIO -> fail (ENOSPC mustahil di read path)
        if n % A != 0:
            # UNEXPECTED short read: sisa tak representable via O_DIRECT.
            # Dilarang menerbitkan read lanjutan misaligned; ulangi SPAN penuh
            # terbatas, lalu fail eksplisit (bukan fallback diam-diam).
            if span_retries_exhausted(max_span_retries):
                return fail(M7_ERR_ODIRECT_SHORT_READ)
            total_read = 0  # ulangi span penuh yang selaras
            span_retries += 1
            continue
        total_read += n  # sisa tetap selaras -> aman lanjut
    return total_read

def read_logical_payload_m6(fd, logical_offset, logical_length, A):
    """Membaca payload M6 v1 (unaligned) via aligned staging buffer."""
    phys_start = (logical_offset // A) * A
    phys_end = ((logical_offset + logical_length + A - 1) // A) * A
    phys_len = phys_end - phys_start

    # Alokasi staging buffer yang selaras: alloc_size wajib kelipatan A
    alloc_size = ((phys_len + A - 1) // A) * A
    staging_buf = aligned_alloc(A, alloc_size)

    bytes_read = pread_o_direct_span(fd, staging_buf, phys_start, phys_len, A)
    if bytes_read < (logical_offset - phys_start + logical_length):
        return fail(M7_ERR_UNEXPECTED_EOF)

    slice_offset = logical_offset - phys_start
    return staging_buf[slice_offset : slice_offset + logical_length]
```

Aturan: hanya remainder yang tetap selaras boleh dilanjutkan; remainder tak
selaras → retry span penuh terbatas → fail eksplisit. Tidak ada read misaligned
yang diterbitkan dalam keadaan apa pun.

### Error Handling

| Error    | Description                                             | Handling                                                                          |
| -------- | ------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `EINVAL` | Misaligned buffer/offset/length                         | Pra-probe: fallback; pasca-probe: hard fail (§ Format-vs-Platform Failure Policy) |
| `EAGAIN` | Non-blocking I/O would block                            | Retry                                                                             |
| `EINTR`  | Interrupted by signal                                   | Retry                                                                             |
| `ENOSPC` | No space left on device (write path: workdir/artifacts) | Fail (disk full)                                                                  |
| `EIO`    | I/O error (disk failure)                                | Fail                                                                              |

### Buffer Management

- **Buffer allocation**:
  ```c
  size_t alloc_size = ((logical_size + dio_alignment - 1) / dio_alignment) * dio_alignment;
  void* buf = aligned_alloc(dio_alignment, alloc_size);
  ```
  `alloc_size` **wajib** di-round-up ke kelipatan `dio_alignment` sebelum memanggil `aligned_alloc`, guna mencegah `EINVAL` pada implementasi `aligned_alloc` standar C11/POSIX yang memandatkan parameter `size` harus kelipatan `alignment`.
- **Buffer lifetime**: allocate before read → free after read (atau reuse pool).
- **No double allocation** untuk satu read operation.

### Reader Concurrency Contract (normatif, opsi B)

`pread` berurutan tidak menghasilkan QD > 1 — QD hanya bermakna dengan concurrency nyata. Maka reader M7 wajib API dua-fase:

```python
tok = submit_read(fd, buffer, offset, length)  # non-blocking, outstanding += 1
...
done = completion()  # reap yang selesai; short-read loop berlaku per unit baca
```

- Invarian: `outstanding <= queue_depth` setiap saat; `queue_depth = 1` berdegenerasi menjadi perilaku sinkron.
- Backend wajib benar-benar $N$ outstanding: worker-pool threads dengan blocking `pread`, atau `io_uring`. Backend sekuensial sinkron **dilarang** mengklaim QD > 1.
- Concurrency decode datang dari fan-out: submit sekaligus untuk semua expert yang miss dalam satu layer (pola F17b) + prefetch layer berikut selagi forward berjalan; `completion()` wajib sebelum buffer dipakai.
- Instrumentasi wajib: counter `max_outstanding_observed` per run; sweep QD hanya sah bila counter mencapai $q$ — bila tidak, run tersebut INVALID untuk $q$ itu (melaporkan BW atas QD fiktif dilarang).

### Integration with Layer Streaming

Per layer (M2/M3):

1. Pre-allocate aligned buffer untuk layer weights.
2. Miss → `submit_read` O_DIRECT ke buffer (boleh batch dengan miss lain selama `outstanding <= QD`).
3. `completion()` → insert ke cache → dequant (M6) jika quant weights.
4. Forward layer (dapat overlap dengan submit prefetch layer berikut).
5. Free buffer → next layer.

### O_DIRECT Flag

```c
// C-style (Mojo equivalent)
int fd = open(path, O_RDONLY | O_DIRECT);
```

Jika O_DIRECT tidak didukung oleh filesystem, fallback ke buffered I/O dengan warning.

## LRU Cache Specification

### Cache Purpose

LRU (Least Recently Used) cache menyimpan expert weights yang sering diakses di RAM untuk mengurangi disk I/O (expert miss pattern F17b).

### Cache Capacity

- Default capacity: dari config (MB).
- Dynamic: bisa disesuaikan berdasarkan available RAM.
- Constraint: capacity ≤ available RAM after resident weights + KV cache.

### Memory Budget (normatif)

`capacity ≤ available RAM` terlalu longgar — runtime juga menampung KV cache,
buffer I/O, buffer dequant, metadata, dan headroom. Kontrak init:

$$\text{resident} + \text{kv} + \text{io\_buffers} + \text{dequant} + \text{runtime\_headroom} \le \text{memory\_limit}$$

- Cache capacity (termasuk pinned bytes) adalah satu suku di dalamnya, bukan keseluruhan RAM.
- Pelanggaran saat init → fail-fast `M7_ERR_LRU_ALLOC` (alokasi mustahil dalam budget), bukan evict.
- Komponen + angka aktual wajib di-log per run (atribusi OOM/degradasi).

### Cache Policy

**Eviction policy**: LRU (Least Recently Used)

- Setiap access: update timestamp.
- Cache full: evict least recently used entry.
- Expert weights entry: one expert per entry (not per layer).

### Expert Pinning

Untuk workload dengan routing CV tinggi (expert panas), pin expert panas di LRU:

- **Hot expert**: expert dengan high access frequency (dari F9 diagnostik).
- **Pinning**: never evict hot expert dari cache.
- **Capacity allocation**: reserve subset of cache for pinned experts.
- Pinned set = prefill set (§ Selective Prefill Policy).

### Pin Budget Invariant (normatif)

Tanpa batas, pinned bisa mengunci seluruh cache (penuh + semua pinned + miss baru
= tanpa victim) dan LRU berhenti menjadi LRU. Maka:

- `pin_budget = 25% × cache_capacity` (default; reservation statis).
- Invarian: `sum(pinned_entry_bytes) ≤ pin_budget < cache_capacity` — selalu ada ≥75% ruang evictable, victim untuk miss unpinned selalu ada.
- Admission: pin yang melanggar budget → pin **ditolak** (expert tetap unpinned, LRU normal) + warning; dilarang meng-evict pinned lain untuk memberi ruang.
- Prefill terikat invariant ini juga: prefill bytes ≤ `pin_budget` (lebih ketat dari ≤ capacity).
- Pelanggaran invariant saat runtime = bug (`M7_ERR_LRU_CORRUPT` + clear); bukan kondisi normal.

### Cache Entry Structure

```python
class CacheEntry:
    key: (layer_id, expert_id)  # expert identifier
    data: bytes                  # quantized expert weights (M6 payload slice)
    timestamp: int               # last access time
    access_count: int            # access frequency
    pinned: bool                # hot expert flag
    # digest: RESERVED untuk checksum per-entry M6 v2 (wajib absent di v1)
```

### Cache Content Policy (normatif)

Satu policy untuk M7: cache menyimpan **quantized bytes** ($\approx 4{,}125$ bpw),
dequant M6 jalan **per use** setiap hit maupun miss:

- Hit: retrieve quantized → dequant → forward. BF16 hasil dequant adalah buffer sementara yang di-discard (tidak di-cache).
- Miss: read → insert quantized → dequant → forward.
- Alasan: tujuan M7 adalah mengurangi disk I/O, bukan mengubah memory model. Cache BF16 ($\approx 16$ bpw, ~4× footprint) mengubah model memori F5 dan dequant accounting — itu eksperimen terpisah dengan gate sendiri, bukan default diam-diam.
- Capacity dihitung dalam quantized bytes; entry yang bukan quantized payload M6 valid → tolak (corrupt).

### Concurrency & Single-Flight (normatif)

QD > 1 tanpa semantik concurrency = race: miss ganda membaca disk dua kali,
access-vs-evict (update timestamp entry yang sedang di-evict), dan keputusan
victim basi. Kontrak minimal:

**State machine per entry** (transisi atomik/CAS):

```
Absent --(miss, CAS menang)--> Loading --(fetch ok)--> Resident
   ^                              |
   |                        (fetch gagal: kembali + error ke penunggu)
   +----------- Evicting <--------+
  (free selesai)  ^  (hanya dari Resident, tak pernah dari pinned/Loading)
```

- **Single-flight per `(layer_id, expert_id)`**: miss pada `Absent` → CAS ke `Loading`; pemenang SATU-SATUNYA yang fetch disk; yang kalah menunggu entry yang sama (tanpa read kedua).
- **Fetch gagal** → kembali `Absent` + error diteruskan ke semua penunggu (tanpa entry setengah-jadi).
- **Access saat `Evicting`** → diperlakukan sebagai miss (resolve ulang), bukan update timestamp mayat.
- **Eviction**: pilih victim + CAS `Resident → Evicting` atomik; CAS gagal (race) → scan ulang, bukan overwrite.
- **Map & field**: mutasi map terserialisasi; `timestamp`/`access_count` atomik terhadap keputusan eviksi. Mekanisme (mutex/shard/atomic-CAS) pilihan implementasi selama jaminan di atas terpenuhi dan diuji.
- Invarian § Pin Budget tetap berlaku di semua transisi (pinned tak pernah masuk `Evicting`).

### Cache Hit/Miss Tracking

- **Hit**: expert found in cache → retrieve from RAM.
- **Miss**: expert not in cache → read from disk → insert into cache.
- **Hit rate**: $HR = \text{hits} / (\text{hits} + \text{misses})$.

### Cache Statistics

Per run:

- Total hits / misses (request).
- Hit rate (HR, diagnostik).
- `hit_bytes` / `miss_bytes` / `disk_bytes` / `ram_bytes` (wajib, § Rumus F13).
- Eviction count.
- Pinned expert count.

### Cache Warm-up

- Init: selective prefill pinned/hot experts (§ Selective Prefill); sisanya cold.
- Warm-up: after several decode runs, cache fills on-demand with requested experts.
- Steady state: HR stabilizes → cache effective.

### Selective Prefill Policy (normatif, opsi B)

"Prefill all layer weights" mustahil (model ~7,9 GB vs cache 512 MB) dan
bertentangan dengan tujuan LRU. Maka:

- Prefill = **hanya pinned/hot experts** dari diagnostik F9 (§ Expert Pinning), dalam quantized bytes.
- Batas: total prefill bytes ≤ `pin_budget` (25% capacity, § Pin Budget Invariant), else init fail-fast `M7_ERR_LRU_ALLOC` (alokasi mustahil; evict-normal hanya untuk miss runtime, bukan prefill pinned).
- Prefill set + bytes wajib di-log per run (atribusi HR/reproducibility).
- Opsi A (no-prefill penuh) ditolak: tanpa pinned set, HR awal tak teratribusi; opsi prefill-all ditolak: melebihi capacity.

### Cache Integration with O_DIRECT Reader

Per layer decode:

1. Check if expert weights in cache (hit).
2. If hit: retrieve quantized from RAM → dequant (per use) → forward; buffer BF16 di-discard.
3. If miss: `submit_read` O_DIRECT (batch antar-miss, § Reader Concurrency Contract) → `completion()` → insert quantized into cache → dequant → forward.
4. Update timestamp/access count.
5. If cache full → evict LRU entry.

### F9 Routing Diagnostics Integration

Use F9 (load-balance diagnostik) dari M3 untuk:

- Identify hot experts (high frequency $f_i$).
- Compute routing CV (coefficient of variation).
- Pin hot experts if CV high (imbalance routing).
- Adjust ρ effective in F5 model.

## M7-Specific Fixture: I/O Pattern Benchmark

Fixture `tools/fixtures/m7_io_patterns.json` berisi workload untuk benchmark dua pola I/O (F17).

### Structure

```json
{
  "name": "M7 I/O pattern benchmark",
  "description": "Workload for sequential trunk and expert-miss I/O patterns",
  "seed": 42,
  "patterns": [
    {
      "id": "trunk_sequential",
      "name": "Sequential large blocks (trunk pattern F17a)",
      "block_size": 4194304,
      "block_count": 100,
      "pattern": "sequential",
      "queue_depth": 1,
      "offsets": [0, 4194304, "... (100 offset)"],
      "offsets_sha256": "<hex>",
      "description": "Read 4 MB blocks sequentially (QD1) for G-M7-1"
    },
    {
      "id": "expert_miss",
      "name": "Expert-size blocks jumping (LRU-miss pattern F17b)",
      "block_size": 10485760,
      "block_count": 100,
      "pattern": "random_jump",
      "queue_depth": 16,
      "offsets": [10485760, 52428800, "... (100 offset, seed-42)"],
      "offsets_sha256": "<hex>",
      "description": "Read 10 MB blocks jumping between offsets (QD sweep) for G-M7-5"
    }
  ]
}
```

### Determinisme Fixture (normatif, P1)

Random tanpa seed = benchmark tak sebanding. Maka:

- Offset dibangkitkan **sekali** dengan `seed = 42`, **disimpan** di fixture beserta `offsets_sha256`; run memvalidasi sha sebelum mengukur (mismatch = INVALID).
- Setiap offset kelipatan 4096 (valid di semua `dio_alignment` kandidat); region `[off, off+block)` pairwise non-overlapping dan di dalam file.
- **Workload identik lintas QD**: sweep $q \in \{1,2,4,8,16\}$ memakai request set yang **sama persis** — yang bervariasi hanya concurrency, bukan urutan/isi request. Sequence baru per QD dilarang (perbandingan kehilangan makna).

### Trunk Sequential Pattern (F17a)

- **Purpose**: Measure $BW_{seq}$ (sequential large blocks, QD1, storage-cold).
- **Block size**: 4 MB (≈ layer weight size).
- **Pattern**: Sequential read (offset 0 → 4 MB → 8 MB → ...).
- **Queue depth**: 1 (QD1).
- **Cold**: storage-cold (§ Cold vs Warm I/O) before run.
- **Run**: 5 runs cold, take median.

### Expert-Miss Pattern (F17b)

- **Purpose**: Measure $BW_{exp}(q)$ (expert-size blocks jumping, QD=q).
- **Block size**: 10 MB (≈ expert weight size).
- **Pattern**: Random jump between offsets (simulate LRU miss) — request set seed-42 yang sama untuk semua $q$ (§ Determinisme Fixture).
- **Queue depth**: Sweep $q \in \{1, 2, 4, 8, 16\}$.
- **Cold**: storage-cold (§ Cold vs Warm I/O) before run.
- **Warm**: 10 runs warm after cold.
- **Run**: 5 runs cold + 10 runs warm per QD, take median.

### Generation Script

`tools/fixtures/generate_m7_io_patterns.py`:

1. Generate sequential offset list for trunk pattern.
2. Generate random offset list (seed 42) for expert-miss pattern.
3. Validate offsets: kelipatan 4096, dalam range file size, pairwise non-overlapping.
4. Output JSON dengan pattern configurations + `offsets` + `offsets_sha256` (satu workload untuk semua QD).

### Benchmark Execution

`tools/benchmark/benchmark_io_patterns.sh`:

```bash
#!/bin/bash
set -e

# Trunk sequential (F17a)
./io_benchmark \
  --pattern sequential \
  --block-size 4194304 \
  --block-count 100 \
  --queue-depth 1 \
  --file /models/qwen-moe-4bit/quant_model.bin \
  --offsets-fixture fixtures/m7_io_patterns.json \
  --output /work/trunk_seq_benchmark.json

# Expert-miss (F17b) - sweep QD
for q in 1 2 4 8 16; do
  ./io_benchmark \
    --pattern random_jump \
    --block-size 10485760 \
    --block-count 100 \
    --queue-depth $q \
    --file /models/qwen-moe-4bit/quant_model.bin \
    --offsets-fixture fixtures/m7_io_patterns.json \
    --output "/work/expert_miss_qd${q}_benchmark.json"
done
```

### Metrics Output

Per pattern:

- $BW_{seq}$ (GB/s) untuk trunk sequential.
- $BW_{exp}(q)$ (GB/s) untuk expert-miss per QD.
- $R_{io}(q) = BW_{exp}(q) / BW_{seq}$ (dilaporkan; bukan gate — expert-miss QD tinggi bisa mendekati/melampaui sequential QD1 tergantung storage/controller/workload).
- `max_outstanding_observed` per QD (syarat validitas sweep).
- $q^*$ (knee, normatif): atas $q$ terurut menaik dengan pendahulu $p$,
  $\mathrm{marginal}(q) = (BW(q)-BW(p))/BW(p)$;
  $q^*$ = $q$ pertama dengan $\mathrm{marginal}(q) < 0{,}10$
  DAN dua titik berikut tanpa rebound besar
  ($\max BW(r_i) - BW(q^*) \le 0{,}10 \cdot BW(q^*)$; pelanggaran = anomali → investigasi, $q^*$ INVALID).
- $D_{sus}$ (normatif): $D_{sus} = (BW_{burst} - BW_{sustained}) / BW_{burst}$ dalam SATU pass $\ge W_{file}$;
  $BW_{burst}$ = throughput 10% byte pertama, $BW_{sustained}$ = throughput 25% byte final.

## Error Handling

### Error Schema

```json
{
  "status": "error",
  "error": {
    "code": "M7_ERR_ODIRECT_ALIGNMENT",
    "stage": "io_direct",
    "message": "O_DIRECT failed: buffer not aligned to block size",
    "details": {
      "buffer_address": "0x7f1234567890",
      "block_size": 4096,
      "required_alignment": 4096
    }
  }
}
```

### Error Types

| Error Code                  | Stage     | Description                                                                  | Handling                                                                                                          |
| --------------------------- | --------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `M7_ERR_ODIRECT_ALIGNMENT`  | io_direct | Buffer/offset/length misaligned                                              | Pra-probe: fallback; pasca-probe: hard fail `M7_ERR_FORMAT_ALIGNMENT`                                             |
| `M7_ERR_FORMAT_ALIGNMENT`   | io_direct | Span fisik I/O atau struktur kontainer melanggar `dio_alignment` pasca-probe | Hard fail, fallback DILARANG                                                                                      |
| `M7_ERR_ODIRECT_SHORT_READ` | io_direct | Short read (incomplete read)                                                 | Remainder selaras → lanjut loop; remainder tak selaras → retry span penuh terbatas → fail (tanpa read misaligned) |
| `M7_ERR_ODIRECT_ENOSPC`     | io_direct | No space left on device (write path: workdir/artifacts)                      | Fail (disk full)                                                                                                  |
| `M7_ERR_ODIRECT_EIO`        | io_direct | I/O error (disk failure)                                                     | Fail                                                                                                              |
| `M7_ERR_LRU_NO_VICTIM`      | lru_cache | Eviksi tak menemukan victim (semua pinned = invariant jebol)                 | Hard fail (evict-normal hanya bila victim ada; tanpa victim tidak ada operasi normal)                             |
| `M7_ERR_LRU_ALLOC`          | lru_cache | Failed to allocate cache entry                                               | Fail (OOM)                                                                                                        |
| `M7_ERR_LRU_CORRUPT`        | lru_cache | Cache data corruption                                                        | Fail + clear cache                                                                                                |

### O_DIRECT Error Handling

- **EINVAL (misaligned) pra-probe**: Log warning → fallback to buffered I/O → continue.
- **EINVAL (misaligned) pasca-probe**: bug format internal → hard fail `M7_ERR_FORMAT_ALIGNMENT`, fallback dilarang (§ Format-vs-Platform Failure Policy).
- **EAGAIN (non-blocking)**: Retry (loop).
- **EINTR (interrupted)**: Retry (loop).
- **ENOSPC (disk full, write path)**: Fail immediately → error code M7_ERR_ODIRECT_ENOSPC.
- **EIO (disk failure)**: Fail immediately → error code M7_ERR_ODIRECT_EIO.

### LRU Cache Error Handling

- **Capacity exceeded**: Normal operation → evict LRU entry (not error; victim dijamin ada oleh § Pin Budget Invariant).
- **Alloc fail**: OOM → error code M7_ERR_LRU_ALLOC → fail.
- **Data corruption**: terdeteksi via revalidasi struktural entry pada setiap access (framing/bounds/q-domain recheck) atau manifest file mismatch → error code M7_ERR_LRU_CORRUPT → clear cache → fail. Tanpa digest tersimpan, "hash(data) vs dirinya sendiri" BUKAN deteksi — maka: (a) v1 mengandalkan revalidasi + manifest per-file (dari download manifest); (b) bit-rot yang lolos struktur tak terdeteksi di v1; (c) checksum per-entry adalah open item M6 v2 (field `digest` reserved, wajib absent di v1).

### Stage Failure Behavior

- **O_DIRECT setup**: Batal layer forward, cleanup, exit error.
- **Layer read (O_DIRECT fail)**: Retry short read, fail on persistent error.
- **LRU cache miss**: Normal operation → read from disk.
- **LRU cache alloc fail**: Batal layer forward, cleanup, exit error.

### Atomic Rollback

- Cache: clear jika corruption detected.
- O_DIRECT buffer: free jika error.
- Workdir: bersihkan temporary files sebelum exit.

## O_DIRECT Configuration Specification

### Configuration Parameters

```bash
dismoen decode \
  --model-dir /models/qwen-moe-4bit \
  --prompt "Test" \
  --max-tokens 64 \
  --context-size 2048 \
  --o-direct \
  --block-size 4096 \
  --queue-depth 16 \
  --cache-capacity 512 \
  --offsets-fixture fixtures/m7_io_patterns.json
```

### Parameters

- `--o-direct`: Enable O_DIRECT I/O (default: disabled, use buffered I/O).
- `--block-size`: Granularitas I/O yang diminta (default: 4096 bytes = 4 KB); wajib memenuhi relasi terhadap `dio_alignment` (§ DIO Alignment Discovery).
- `--queue-depth`: Queue depth untuk async I/O (default: 16, sweep {1,2,4,8,16} for G-M7-5). Wajib didukung backend concurrency nyata (§ Reader Concurrency Contract); backend sinkron dilarang mengklaim QD > 1.
- `--cache-capacity`: LRU cache capacity in MB (default: 512 MB, from config).
- `--offsets-fixture`: Path ke fixture JSON pola I/O precomputed (`fixtures/m7_io_patterns.json`). Wajib dimuat untuk benchmark guna menjamin determinisme request set antar-run dan lintas sweep queue depth (memvalidasi `offsets_sha256` sebelum eksekusi).

### Block Size Options

- 512 bytes: granularitas minimum (sektor disk).
- 4096 bytes (4 KB): default; cocok untuk NVMe modern.
- 8192 bytes (8 KB): larger blocks untuk sequential I/O.

Nilai di atas adalah granularitas yang diminta, bukan alignment storage: semuanya tetap tunduk pada `io_block_size >= dio_alignment` dan habis dibagi `dio_alignment`.

Default: 4096 bytes (4 KB) untuk optimal NVMe performance.

### Queue Depth Sweep

Queue depth $q$ (number of outstanding I/O requests):

- **Trunk sequential**: QD1 (single request, no overlap).
- **Expert-miss**: Sweep $q \in \{1, 2, 4, 8, 16\}$ untuk mencari $q^*$.
- **Validitas sweep**: tiap run wajib mencatat `max_outstanding_observed`; $q$ yang tidak tercapai = run INVALID untuk $q$ tersebut.

Default: 16 (balanced untuk expert-miss pattern).

### Cache Capacity Options

- 256 MB: minimal untuk hot experts.
- 512 MB: default (cukup untuk ~10-20 hot experts).
- 1024 MB: larger cache untuk more experts.

Constraint: cache capacity ≤ available RAM after resident weights + KV cache.

### Readahead Policy

- **O_DIRECT path**: readahead N/A (bypass page cache).
- **Buffered path**: `posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)` → double window.
- **Config**: log jalur mana yang diukur (O_DIRECT vs buffered).

### Filesystem Logging

Per run:

- Filesystem type (ext4, xfs, etc.).
- Mount options (noatime, nobarrier, etc.).
- Block size (statvfs).
- `dio_alignment` hasil discovery + hasil read probe + hasil layout scan.
- Alignment verified (terhadap `dio_alignment`, bukan konstanta).

### Thermal Logging

Per run (per ref-storage skill):

- SSD temperature (via smartctl or /sys/class/thermal).
- Power consumption (jika tersedia).
- Run duration (must be ≥ 30s untuk sustained measurement).

### Configuration Validation

- Block size: valid values {512, 4096, 8192} + relasi `io_block_size >= dio_alignment`, habis dibagi (fail-fast `M7_ERR_ODIRECT_ALIGNMENT`).
- Queue depth: valid values {1, 2, 4, 8, 16} (QD > 1 wajib backend concurrency nyata).
- Cache capacity: ≤ available RAM (detection run-time).
- O_DIRECT: check filesystem support (**open + aligned read probe**, bukan open saja; § DIO Alignment Discovery).

## Workflow Diagram

```mermaid
flowchart TD
    A[Start: decode with O_DIRECT+LRU] --> B[Init O_DIRECT reader]
    B --> C{O_DIRECT supported?}
    C -->|No| ERR1[Error: fallback buffered, log warning]
    C -->|Yes| D[Init LRU cache]
    D --> E[Selective prefill: pinned/hot experts ≤ capacity (§ Selective Prefill)]
    E --> G[Decode loop t=1..N]
    G --> H{All tokens done?}
    H -->|Yes| Z[Success: throughput measured]
    H -->|No| I[Sample token t-1]
    I --> J[Layer loop 0..23]
    J --> K{Expert in cache?}
    K -->|Yes| L[Retrieve from RAM LRU]
    K -->|No| M[Pread O_DIRECT from disk]
    M --> N{O_DIRECT OK?}
    N -->|No| ERR2[Error: retry or fail]
    N -->|Yes| O[Insert into LRU cache]
    O --> P[Dequant quant weights]
    P --> Q[Forward layer]
    Q --> R{Layer done?}
    R -->|No| J
    R -->|Yes| S[Increment t]
    S --> G

    ERR1 --> WARN[Continue with buffered I/O]
    WARN --> G
    ERR2 --> FAIL[Error cleanup]
    FAIL --> END[End]
    Z --> END
```

## Rumus (F13)

$$HR_{req}=\text{hits}/(\text{hits}+\text{misses}),\quad \rho_B=S_{RAM}/(S_{RAM}+S_{disk}),\quad BW_{eff}=\left(\rho_B/BW_{RAM}+(1-\rho_B)/BW_{disk}^{O\_DIRECT}\right)^{-1}$$

$\rho_B$ adalah rasio **byte-level**: $S_{RAM}$ = byte disajikan dari RAM, $S_{disk}$ = byte dibaca dari disk; $e_T$ G-M7-2 dihitung terhadap $\rho_B$ ini. $HR_{req}$ (rasio request) hanya diagnostik — sama dengan $\rho_B$ hanya jika seluruh blok sama besar, sehingga dilarang dipakai sebagai prediktor rasio byte.

Enam field log wajib per run: `cache_hit_requests`, `cache_miss_requests`, `hit_bytes` ($S_{RAM}$), `miss_bytes`, `disk_bytes` ($S_{disk}$), `ram_bytes`.

Koreksi via F9: bila $CV$ routing tinggi (expert panas), pin expert panas di LRU/page cache → $ρ$ efektif naik. Baseline F9 dari M3 dipakai di sini.

## Gate

| Gate   | Kriteria                                   | Threshold                                                                                                                                     | Metode                                                           |
| ------ | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| G-M7-1 | bandwidth cold sequential                  | $BW_{seq}\ge$ 2,5 GB/s (pola trunk F17a)                                                                                                      | storage-cold, 5 run                                              |
| G-M7-2 | model cache F13                            | $e_T \le 30\%$ dengan rho_B byte-level terukur                                                                                                | cache-cold vs cache-warm, 30 run                                 |
| G-M7-3 | decode 4-bit throughput                    | ≥ 2 tok/s di $c^*$, protokol § Decode Gate Protocol                                                                                           | `../03-testing.md` §4.4                                          |
| G-M7-4 | kurva core + I/O (rasio)                   | $BW_{eff}$ independen $c$ (slope≈0, bukti memory-bound) + HR stabil ±5pp lintas $c$ + $e_T\le30\%$ dgn F5+F16                                 | sweep core cache-warm, 10 run/level + 30 run di $c^*$            |
| G-M7-5 | pola I/O storage (dua angka, bukan brosur) | $BW_{seq}$ + $BW_{exp}(q)$ + $q^*$ + $R_{io}$ dilaporkan + $D_{sus}\le30\%$ + `dio_alignment` probe-verified + fs/readahead/suhu logged (F17) | §4.4 dua pola storage-cold/cache-warm + sustained $\ge W_{file}$ |
| G-M7-6 | Direct-I/O correctness                     | buffer/offset/length selaras + probe sukses + tanpa silent fallback                                                                           | probe + layout scan + IT-M7-2/3/11                               |
| G-M7-7 | LRU correctness                            | hit/miss + eviksi + pin invariant + single-flight + stats deterministik                                                                       | IT-M7-6/7/8/13/14/15 + stats                                     |

Cold: tiga kondisi § Cold vs Warm I/O (storage-cold / storage-warm / cache-warm). Warm-up 2× tidak dihitung.

## Testing

- B: 5 run storage-cold (BW) + 30 run cache-warm vs cache-cold (rho_B, HR, $e_T$).
- B-core (F16): ulang sweep $c$ di atas O*DIRECT+LRU; buktikan $T*{IO}$ datar vs $c$ dan $T_{comp}(c)$ ikut F16; bila $BW_{eff}$ naik ikut $c$ berarti masih compute-bound → investigasi, bukan klaim memory-bound.
- F: error path O_DIRECT (alignment fase-split, short read, EIO) → clean error.
- Log: rho*B + HR + enam field byte (§ Rumus F13), VmHWM, waktu/fase, $C*{max}, c, r$.

### Decode Gate Protocol (normatif, G-M7-3)

Tanpa ini 2 tok/s tak reproducible:

- $c^*$ = operating point dari artefak kalibrasi F16 ter-commit (`c*`, `r*` final); run dengan $c \ne c^*$ → INVALID.
- Prompt: fixture fixed `fixtures/m4/m4_prompt1_tokens.json`.
- Warmup: 2 runs not counted; state cache-warm steady.
- Hitung: $N_{gen} = 64$ token; throughput $= (N_{gen}-1)/(t_{last} - t_{first})$ — latensi first-token + prefill excluded, decode saja.
- Fallback buffered selama run → INVALID (karakterisasi buffered terpisah, bukan gate ini).

## Integration Tests

### Test Matrix

| Test ID  | Scenario                                    | Expected                                               | Priority |
| -------- | ------------------------------------------- | ------------------------------------------------------ | -------- |
| IT-M7-1  | Happy path: O_DIRECT + LRU → decode 4-bit   | Exit 0, ≥ 2 tok/s protokol § Decode Gate Protocol      | HIGH     |
| IT-M7-2  | O_DIRECT not supported                      | Fallback to buffered I/O, log warning                  | HIGH     |
| IT-M7-3  | Layout format langgar alignment pasca-probe | Hard fail M7_ERR_FORMAT_ALIGNMENT, tanpa fallback      | HIGH     |
| IT-M7-4  | O_DIRECT short read selaras                 | Loop until complete, continue                          | HIGH     |
| IT-M7-5  | ENOSPC on write path (workdir/artifacts)    | Exit error M7_ERR_ODIRECT_ENOSPC                       | HIGH     |
| IT-M7-6  | LRU cache capacity exceeded                 | Evict LRU entry (normal)                               | HIGH     |
| IT-M7-7  | LRU cache alloc fail (OOM)                  | Exit error M7_ERR_LRU_ALLOC                            | HIGH     |
| IT-M7-8  | LRU cache corruption (fault injection)      | Revalidasi deteksi → clear + M7_ERR_LRU_CORRUPT        | HIGH     |
| IT-M7-9  | I/O pattern benchmark: trunk sequential     | BW_seq ≥ 2,5 GB/s                                      | HIGH     |
| IT-M7-10 | I/O pattern benchmark: expert-miss          | BW_exp(q) + R_io(q) reported (tanpa threshold)         | HIGH     |
| IT-M7-16 | Fixture determinism                         | seed-42 offsets sama lintas QD + aligned + non-overlap | MEDIUM   |
| IT-M7-11 | EINVAL pasca-probe (fault injection)        | Hard fail, fallback dilarang                           | HIGH     |
| IT-M7-12 | Short remainder tak selaras                 | Retry span terbatas → fail, tanpa read misaligned      | HIGH     |
| IT-M7-13 | Selective prefill bound                     | prefill bytes > pin_budget → init fail; set logged     | HIGH     |
| IT-M7-14 | Pin budget enforcement                      | over-budget pin ditolak; victim selalu ada             | HIGH     |
| IT-M7-15 | Single-flight miss ganda                    | N thread × 1 key miss → tepat 1 disk read              | HIGH     |

### Test Automation

`tests/integration/test_m7_odirect_lru.sh`:

```bash
#!/bin/bash
set -e

# Setup
MODEL_DIR="/models/qwen-moe-4bit"
WORKDIR="/tmp/test_work"

# IT-M7-1: Happy path
dismoen decode \
  --model-dir "$MODEL_DIR" \
  --prompt "Test" \
  --max-tokens 64 \
  --context-size 2048 \
  --o-direct \
  --block-size 4096 \
  --queue-depth 16 \
  --cache-capacity 512 \
  --workdir "$WORKDIR"
# Expect exit 0, throughput ≥ 2 tok/s

# IT-M7-9: I/O pattern benchmark
./io_benchmark \
  --pattern sequential \
  --block-size 4194304 \
  --block-count 100 \
  --queue-depth 1 \
  --file "$MODEL_DIR/quant_model.bin" \
  --output "$WORKDIR/trunk_seq_benchmark.json"
# Expect BW_seq ≥ 2,5 GB/s

# IT-M7-10: I/O pattern benchmark expert-miss
for q in 1 2 4 8 16; do
  ./io_benchmark \
    --pattern random_jump \
    --block-size 10485760 \
    --block-count 100 \
    --queue-depth $q \
    --file "$MODEL_DIR/quant_model.bin" \
    --output "$WORKDIR/expert_miss_qd${q}_benchmark.json"
done
# Expect BW_exp(q) + R_io(q) reported (tanpa threshold; lihat § Metrics Output)
```

### Negative Path Coverage

- Error codes M7*ERR*\* semua teruji.
- Error JSON schema valid di semua failure paths.
- Cleanup buffer/cache verified setiap error (tidak ada memory leak).

## Performance Baseline

### Cold vs Warm I/O

`drop_caches` BUKAN LRU-cold: O_DIRECT bypass page cache, sehingga yang
membedakan run untuk M7 adalah state cache aplikasi. Tiga kondisi normatif:

- **storage-cold**: `drop_caches` + proses fresh + LRU prefill-only. Untuk G-M7-1 (BW_seq).
- **storage-warm**: tanpa `drop_caches`, proses fresh + LRU prefill-only (storage hangat, app-cache cold).
- **cache-warm**: proses sama, LRU terpopulasi (sesudah warm-up; 2 run pertama tidak dihitung). Untuk G-M7-2 (HR/e_T).

Untuk run O_DIRECT, sumbu pembeda adalah state app-cache; `drop_caches`
dipertahankan untuk keseragaman protokol + run pembanding buffered-fallback.

**Cold I/O** (G-M7-1):

- Definition: storage-cold (lihat di atas).
- Target: $BW_{seq} \ge 2,5$ GB/s (trunk sequential pattern).
- Method: 5 runs cold, take median.
- Purpose: Measure raw storage bandwidth without cache.

**Warm I/O** (G-M7-2):

- Definition: cache-warm (LRU terpopulasi; 2 run pertama tidak dihitung).
- Target: rho_B byte-level + HR terukur (stabil).
- Method: 30 runs cache-warm (vs cache-cold sebagai kontras), compute e_T.
- Purpose: Measure LRU cache effectiveness.

### Sustained vs Burst

**Burst vs Sustained** (per ref-storage skill):

- **Burst**: SSD SLC cache → high initial bandwidth; diukur = throughput 10% byte pertama dalam pass yang sama.
- **Sustained**: After SLC exhausted → lower bandwidth; diukur = throughput 25% byte final dalam pass yang sama (run ≥ $W_{file}$).
- Target: $D_{sus} = (BW_{burst}-BW_{sustained})/BW_{burst} \le 30\%$.
- Method: One pass ≥ $W_{file}$ (≥ 28,63 GB), not 1-second run.
- Logging: SSD temperature, power consumption.

### Thermal Monitoring

Per run (per ref-storage skill):

- **SSD temperature**: via smartctl or `/sys/class/thermal`.
- **Power consumption**: if available (PCIe power stats).
- **Run duration**: must be ≥ 30s for sustained measurement.
- **Throttling detection**: if temp > threshold → degrade performance 10-20%.

### Measurement Protocol

1. **Environment**: AC/plug-in stable, governor `performance`.
2. **Storage-cold run**: `sync && echo 3 > /proc/sys/vm/drop_caches` + proses fresh + LRU prefill-only.
3. **Warm-up**: 2 runs not counted (menuju cache-warm).
4. **Measured runs**: N=5 storage-cold (BW), N=30 cache-warm vs cache-cold (rho_B, HR, e_T).
5. **Metrics**:
   - BW_seq (GB/s) for trunk sequential.
   - BW_exp(q) (GB/s) for expert-miss per QD.
   - rho_B byte-level + HR + byte counters (§ Rumus F13) for LRU cache.
   - e_T (prediction error) for F13 model.
   - Temperature (°C) for SSD.
   - Power (W) if available.
6. **Statistics**: p50/p95, median (per ref-perf skill).

### Expected Values (NVMe, entry-tier)

| Metric        | Target     | Unit  |
| ------------- | ---------- | ----- |
| BW_seq (p50)  | ≥ 2,5      | GB/s  |
| BW_seq (p95)  | ≥ 2,0      | GB/s  |
| BW_exp(q) p50 | measured   | GB/s  |
| R_io(q)       | dilaporkan | ratio |
| D_sus         | ≤ 30       | %     |
| Temperature   | < 70       | °C    |

### Failure Criteria

- FAIL jika:
  - p95 BW_seq < 2,0 GB/s (below minimum).
  - D_sus > 30% (excessive degradation).
  - Temperature > 70°C (thermal throttling).
- Investigasi dan fix sebelum gate PASS.

## CLI: Reader Configuration

CLI parameters for O_DIRECT reader + LRU cache are already integrated into `dismoen decode` (from M5). Additional reader-specific parameters:

```bash
dismoen decode \
  --model-dir /models/qwen-moe-4bit \
  --prompt "Test" \
  --max-tokens 64 \
  --context-size 2048 \
  --o-direct \
  --block-size 4096 \
  --queue-depth 16 \
  --cache-capacity 512 \
  [--readahead-policy <POLICY>]
```

- `--o-direct`: Enable O_DIRECT I/O (default: disabled).
- `--block-size`: Requested I/O granularity (default: 4096); wajib memenuhi relasi terhadap `dio_alignment`.
- `--queue-depth`: Queue depth for async I/O (default: 16).
- `--cache-capacity`: LRU cache capacity in MB (default: 512).
- `--readahead-policy`: SEQUENTIAL (buffered) or NONE (O_DIRECT).

## Reader CLI Output Examples

### Success Output (O_DIRECT + LRU)

```json
{
  "status": "success",
  "run_id": "M7-20250115-001",
  "model": "qwen1.5-moe-a2.7b-chat",
  "io_mode": "O_DIRECT",
  "cache_config": {
    "enabled": true,
    "capacity_mb": 512,
    "entries": 15
  },
  "metrics": {
    "prefill_time_sec": 42.3,
    "decode_time_sec": 58.7,
    "total_time_sec": 101.0,
    "tokens_per_sec": 0.634,
    "vmhwm_bytes": 5368709120,
    "bytes_read_prefill": 7934542592,
    "bytes_read_decode": 264577024,
    "cache": {
      "hits": 245,
      "misses": 120,
      "hit_rate": 0.671
    }
  }
}
```

### Error Output (O_DIRECT Not Supported)

```json
{
  "status": "warning",
  "message": "O_DIRECT not supported by filesystem, falling back to buffered I/O",
  "io_mode": "buffered",
  "details": {
    "filesystem": "ext4",
    "mount_opts": "noatime"
  }
}
```

### Error Output (O_DIRECT Alignment Fail)

```json
{
  "status": "error",
  "error": {
    "code": "M7_ERR_ODIRECT_ALIGNMENT",
    "stage": "io_direct",
    "message": "O_DIRECT failed: buffer not aligned to block size",
    "details": {
      "buffer_address": "0x7f1234567890",
      "block_size": 4096,
      "required_alignment": 4096
    }
  }
}
```

## LRU Cache Statistics

### Statistics Schema

Optional LRU cache statistics output for debugging cache effectiveness:

```json
{
  "run_id": "M7-20250115-001",
  "cache_config": {
    "capacity_mb": 512,
    "max_entries": 60
  },
  "statistics": {
    "total_accesses": 365,
    "hits": 245,
    "misses": 120,
    "hit_rate": 0.671,
    "evictions": 8,
    "pinned_entries": 5
  },
  "hot_experts": [
    {
      "layer_id": 0,
      "expert_id": 3,
      "access_count": 42,
      "frequency": 0.115
    },
    {
      "layer_id": 12,
      "expert_id": 27,
      "access_count": 38,
      "frequency": 0.104
    }
  ]
}
```

### Metrics

- **total_accesses**: Total cache accesses (hits + misses).
- **hits**: Cache hits (retrieved from RAM).
- **misses**: Cache misses (read from disk).
- **hit_rate**: HR = hits / (hits + misses) (diagnostik; dilarang sebagai prediktor rasio byte).
- **hit_bytes / miss_bytes / disk_bytes / ram_bytes**: byte counters wajib (§ Rumus F13).
- **evictions**: Number of LRU evictions.
- **pinned_entries**: Number of pinned hot experts.
- **hot_experts**: List of top accessed experts (layer_id, expert_id, access_count, frequency).

### Debugging Use Cases

- Identify hot experts → pin in cache.
- Monitor hit rate stability across runs.
- Detect cache thrashing (high eviction count).
- Correlate hit rate with decode throughput.

### Optional Flag

```bash
dismoen decode \
  --model-dir /models/qwen-moe-4bit \
  --prompt "Test" \
  --max-tokens 64 \
  --context-size 2048 \
  --o-direct \
  --cache-stats /work/lru_cache_stats.json
```

## Security

- SEC-4: buffer O_DIRECT aligned + bounded; LRU capacity dari config, bukan file.
- SEC-5: cache hanya di workdir/RAM, tidak menulis ke model dir.

## DoD

### Gate Requirements

- [ ] G-M7-1 bandwidth cold sequential ≥ 2,5 GB/s (trunk pattern F17a)
- [ ] G-M7-2 cache model F13 with e_T ≤ 30% (rho_B byte-level measured)
- [ ] G-M7-3 decode 4-bit throughput ≥ 2 tok/s at c\* (protokol § Decode Gate Protocol)
- [ ] G-M7-4 core + I/O curve: BW_eff independent of c (slope≈0), HR stable ±5pp, e_T ≤ 30% with F5+F16
- [ ] G-M7-5 I/O storage patterns: BW_seq + BW_exp(q) + q\* + R_io reported, D_sus ≤ 30%, dio_alignment probe-verified, fs/readahead/temp logged
- [ ] G-M7-6 Direct-I/O correctness: probe + layout scan + tanpa silent fallback
- [ ] G-M7-7 LRU correctness: hit/miss + eviksi + pin invariant + single-flight + stats deterministik
- [ ] Correctness-first: verdict G-M7-1..5 tidak sah bila G-M7-6/7 FAIL

### O_DIRECT Reader Implementation

- [ ] O_DIRECT reader terimplementasi dengan triple alignment terhadap `dio_alignment` hasil discovery (buffer, offset, length)
- [ ] DIO discovery + aligned read probe terimplementasi (open saja tidak cukup); relasi block-size vs alignment divalidasi
- [ ] Staging buffer & physical span reader: translasi payload unaligned M6 v1 ke aligned physical pread span + logical slice (pelanggaran fisik = hard fail `M7_ERR_FORMAT_ALIGNMENT`)
- [ ] Short read loop terimplementasi (remainder selaras lanjut; remainder tak selaras → retry span → fail; short read selaras bukan error)
- [ ] Error handling: EINVAL pra-probe → fallback, EINVAL pasca-probe → hard fail, EAGAIN/EINTR → retry, EIO → fail, ENOSPC write-path → fail
- [ ] Buffer management: aligned_alloc → submit/completion → free per layer
- [ ] Integration with layer streaming: submit O_DIRECT → completion → dequant → forward → free buffer (overlap prefetch diizinkan selama outstanding ≤ QD)

### LRU Cache Implementation

- [ ] LRU cache terimplementasi dengan capacity dari config
- [ ] State machine entry (Absent/Loading/Resident/Evicting) + single-flight per key + aturan access-vs-evict
- [ ] Eviction policy: LRU (least recently used)
- [ ] Expert pinning: hot experts (high frequency from F9) → never evict
- [ ] Cache entry structure: (layer_id, expert_id) key, quantized-bytes data, timestamp, access_count, pinned flag
- [ ] Cache content policy: quantized-only + dequant per use (buffer BF16 di-discard; BF16-cache = eksperimen terpisah)
- [ ] Hit/miss tracking: hits, misses, HR (diagnostik), byte counters, eviction count, pinned count
- [ ] Cache warm-up: selective prefill pinned → on-demand fill → steady state
- [ ] Selective prefill bound: prefill bytes ≤ pin_budget else init fail; set + bytes logged
- [ ] Pin budget: sum(pinned) ≤ 25% capacity; over-budget pin ditolak; victim selalu ada
- [ ] Integration with O_DIRECT reader: hit → retrieve from RAM, miss → O_DIRECT read → insert
- [ ] F9 integration: identify hot experts, compute CV, pin if CV high, adjust ρ effective

### M7 I/O Pattern Fixture

- [ ] `tools/fixtures/m7_io_patterns.json` tercommit (seed-42 offsets tersimpan + sha; satu workload lintas QD)
- [ ] Generation script `tools/fixtures/generate_m7_io_patterns.py` teruji
- [ ] Benchmark script `tools/benchmark/benchmark_io_patterns.sh` terimplementasi
- [ ] Trunk sequential: 4 MB blocks, QD1, 5 runs cold, BW_seq measured
- [ ] Expert-miss: 10 MB blocks, random jump, sweep QD 1..16, BW_exp(q) measured
- [ ] Metrics: BW_seq, BW_exp(q), R_io(q), q\*, D_sus

### Error Handling

- [ ] Error schema JSON terimplementasi untuk semua 8 error types
- [ ] O_DIRECT error handling: EINVAL pra-probe → fallback, pasca-probe → hard fail, EAGAIN/EINTR → retry, EIO → fail, ENOSPC write-path → fail
- [ ] LRU error handling: capacity exceeded → evict (normal), alloc fail → fail, corruption (revalidasi/manifest) → clear + fail
- [ ] Stage failure behavior: O_DIRECT setup fail → cleanup, layer read fail → retry/persistent fail, LRU alloc fail → cleanup
- [ ] Atomic rollback: cache clear if corrupt, O_DIRECT buffer free if error, workdir cleanup

### O_DIRECT Configuration

- [ ] CLI parameters terimplementasi: --o-direct, --block-size, --queue-depth, --cache-capacity
- [ ] Block size validation: valid values {512, 4096, 8192}, default 4096
- [ ] Queue depth sweep: q ∈ {1, 2, 4, 8, 16}, default 16, dengan verifikasi max-outstanding per q
- [ ] Cache capacity validation: ≤ available RAM (detection run-time)
- [ ] O_DIRECT support check: open + aligned read probe (§ DIO Alignment Discovery), fallback if unsupported
- [ ] Readahead policy: O_DIRECT N/A, buffered POSIX_FADV_SEQUENTIAL
- [ ] Filesystem logging: fs type, mount opts, block size, alignment verified
- [ ] Thermal logging: SSD temp, power, run duration ≥ 30s (sustained)

### Performance Baseline

- [ ] BW_seq ≥ 2,5 GB/s (trunk sequential, cold)
- [ ] BW_exp(q) measured per QD, R_io(q) = BW_exp(q)/BW_seq dilaporkan (measurement, bukan gate; threshold hanya bila target hardware-specific terkalibrasi)
- [ ] q\* operasional (§ Metrics Output): marginal < 10% + dua titik berikut tanpa rebound besar
- [ ] D_sus ≤ 30% via formula § Metrics Output (burst 10% awal vs sustained 25% final, satu pass ≥ W_file)
- [ ] Decode 4-bit throughput ≥ 2 tok/s at c\* (protokol § Decode Gate Protocol)
- [ ] rho_B byte-level measured (cache-warm), e_T ≤ 30% for F13 model
- [ ] BW_eff independent of c (slope≈0, proof memory-bound)
- [ ] HR stable ±5pp across c (proof cache effectiveness)

### Core Scaling (F16)

- [ ] F16 sweep terimplementasi di atas O_DIRECT+LRU
- [ ] T_IO flat vs c (proof memory-bound, G-M7-4)
- [ ] T_comp(c) follows F16 model
- [ ] e_T,core ≤ 30% with F5+F16 calibration
- [ ] Non-regression: S_tok(c) ≥ 1 (never slower than c=1)
- [ ] c*, r* final committed as Trial operating point

### Integration Tests

- [ ] Happy path: O_DIRECT + LRU → decode 4-bit → throughput ≥ 2 tok/s (protokol § Decode Gate Protocol)
- [ ] O_DIRECT probe fail: fallback buffered I/O + warning (satu-satunya fallback sah)
- [ ] Format-vs-platform: layout misaligned pasca-probe → hard fail `M7_ERR_FORMAT_ALIGNMENT`, fallback dilarang
- [ ] O_DIRECT short read test: remainder selaras → loop until complete; remainder tak selaras → retry span terbatas → fail
- [ ] LRU capacity test: capacity exceeded → evict LRU entry
- [ ] LRU alloc fail test: OOM → error M7_ERR_LRU_ALLOC
- [ ] Cache corruption test: fault injection → revalidasi deteksi → clear cache → error M7_ERR_LRU_CORRUPT
- [ ] I/O pattern benchmark: trunk sequential + expert-miss → BW_seq + BW_exp(q) measured
- [ ] QD validity: max outstanding tercapai per q, run tanpa itu = INVALID
- [ ] Sustained measurement: run ≥ 30s → D_sus ≤ 30%

### Security Tests

- [ ] SEC-4: buffer O_DIRECT aligned + bounded (verified alignment)
- [ ] SEC-4: LRU capacity dari config (not from file size)
- [ ] SEC-5: cache hanya di workdir/RAM (tidak menulis ke model dir)
- [ ] Model directory read-only saat engine jalan
- [ ] Output atomic: tidak ada partial output valid jika gagal

### Reporting & Artifacts

- [ ] Laporan F13 tercommit (rho_B byte-level, e_T, BW_eff with measured rho_B)
- [ ] Laporan F17 tercommit (BW_seq, BW_exp(q), q\*, R_io, D_sus)
- [ ] Laporan F16 tercommit (c*, r* final sebagai Trial operating point)
- [ ] Run ID tercatat per run (format: M7-YYYYMMDD-NNN)
- [ ] Log O_DIRECT config: block size, queue depth (minta + efektif), cache capacity
- [ ] Log filesystem: fs type, mount opts, block size, alignment
- [ ] Log thermal: SSD temp, power, run duration
- [ ] Fase Trial dinyatakan hijau (G-M0..G-M7) + retro jebakan → 04-quality.md §5.4

## Wave Note

Lihat implementasi notes di: <ref_file file="../../scratch/wave/m7/README.md" />
