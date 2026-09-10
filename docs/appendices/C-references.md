# Lampiran C — Referensi (Grouped)

> Bagian dari `disk-streaming-moe-engine`. Index: `../README.md`.
> Semua fakta eksternal yang menentukan dimensi/arsitektur harus dapat ditelusuri ke sumber di bawah. **Untuk reproduksibilitas keamanan, revision/commit checkpoint tetap wajib dipin di `models.lock.json`; tautan model saja tidak menggantikan pin tersebut.**
> Nomor R stabil dan dirujuk lintas dokumen — dilarang menomori ulang; penambahan memakai nomor lanjutan.

## Peta grup → pemakai

| Grup | Isi | Dipakai oleh |
|---|---|---|
| A. Ground truth model (R1–R5) | model card + `transformers` config/modeling | M0–M4, M9; skill spec-compliance |
| B. Format & parsing (R7–R8) | safetensors spec + metadata parsing | F15, M0; skill `mojo-1-0` (I-7) |
| C. Transformer / MoE / paralelisme (R9–R10, R14–R16, R18, R20) | delta rule, load-balance, Amdahl/Gustafson/multicore, prefill-vs-decode, host-bound | F4, F9, F14, F16; M3, M5, M8 |
| D. Model performa & tail (R17, R19, R21) | roofline, tail-at-scale, STREAM | F4/F5, §4.4, G-M5-6 |
| E. Storage / SSD I/O (R22–R27) | O_DIRECT, pread, NVMe queues, readahead, thermal, burst-vs-sustained | F17, M7 |
| F. Pola engineering & baseline (R6, R11–R13) | pola kimi-k3-in-c, quant/GGUF, MTP, tooling boundary | Arsitektur, M6–M7, B1 |

## A. Ground truth model

- **[R1] Qwen1.5-MoE-A2.7B-Chat — Hugging Face model card**
  https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat
  Menyatakan 14,3 B parameter total dan **2,7 B activated parameters**.

- **[R2] Hugging Face Transformers — `Qwen2MoeConfig`**
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_moe/configuration_qwen2_moe.py
  Ground truth implementasi/config: `hidden_size=2048`, `num_hidden_layers=24`, `num_experts=60`, `num_experts_per_tok=4`, `moe_intermediate_size=1408`, `shared_expert_intermediate_size=5632`, `norm_topk_prob=false`, `qkv_bias=true`, `tie_word_embeddings=false`.

- **[R3] Hugging Face Transformers — `Qwen2Moe` modeling**
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
  Ground truth operasi: router softmax fp32, optional top-k renorm (nonaktif pada config trial), shared-expert gate dengan sigmoid, dan load-balance loss yang merujuk Switch Transformer.

- **[R4] Qwen3.6-35B-A3B — Hugging Face model card**
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B
  Menyatakan 35 B total / **3 B activated**, 40 layer, layout `10 × (3 × (Gated DeltaNet → MoE) → (Gated Attention → MoE))`, 16Q/2KV pada gated attention, 256 expert, 8 routed + 1 shared, inter 512, vocab 248320 padded, dan context native 262144.

- **[R5] Qwen — halaman resmi Qwen3.6-35B-A3B**
  https://qwen.ai/blog?id=qwen3.6-35b-a3b
  Sumber resmi dari Qwen yang ditautkan dari model card.

## B. Format & parsing

- **[R7] Safetensors — format specification / repository**
  https://github.com/safetensors/safetensors
  Spesifikasi format: 8-byte little-endian header length, JSON header, `data_offsets`, byte buffer; implementasi saat ini menolak header > **100,000,000 bytes**.

- **[R8] Hugging Face — Safetensors metadata parsing**
  https://huggingface.co/docs/safetensors/metadata_parsing
  Contoh parsing 8-byte header length, JSON header, `data_offsets`, serta parsing index sharded.

## C. Transformer / MoE / paralelisme — what we use

- **[R9] Yang et al. — *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*** → F14
  https://arxiv.org/abs/2406.06484
  Rujukan DeltaNet / delta-rule dan chunked sequence-length parallelization (F14).

- **[R10] Fedus, Zoph & Shazeer — *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*** → F9
  https://arxiv.org/abs/2101.03961
  Rujukan bentuk load-balancing loss yang dipakai F9.

- **[R14] Amdahl, G.M. — *Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities*** → F16a (bentuk $S(c)$, knee)
  AFIPS Conference Proceedings, vol. 30, 1967, pp. 483–485.
  Teks asli (ETH scan): https://safari.ethz.ch/digitaltechnik/spring2020/lib/exe/fetch.php?media=amdahl1967.pdf

- **[R15] Gustafson, J.L. — *Reevaluating Amdahl's Law*** → F16 (decode problem-tetap ⇒ Amdahl, bukan scaled)
  Communications of the ACM, 31(5), 1988, pp. 532–533. DOI: https://doi.org/10.1145/42411.42415

- **[R16] Hill, M.D. & Marty, M.R. — *Amdahl's Law in the Multicore Era*** → F16d (operasi di $c^*$, rasio $r^*$)
  IEEE Computer, 41(7), 2008, pp. 33–38.
  PDF: https://research.cs.wisc.edu/multifacet/papers/ieeecomputer08_amdahl_multicore.pdf

- **[R18] *LLM Inference Unveiled: Survey and Roofline Model Insights* (2024)** → F4 (prefill compute-bound, decode memory-bound)
  https://arxiv.org/abs/2402.16363

- **[R20] *TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition* (preprint 2026, pendukung)** → F16b (suku $\beta$)
  https://arxiv.org/abs/2603.12465
  Decode MoE persisten host-bound; single-thread CPU orde-satu. Status pendukung, bukan normatif.

## D. Model performa & tail

- **[R17] Williams, S., Waterman, A. & Patterson, D. — *Roofline: An Insightful Visual Performance Model for Multicore Architectures*** → F4
  Communications of the ACM, 52(4), 2009, pp. 65–76. DOI: https://doi.org/10.1145/1498765.1498785

- **[R19] Dean, J. & Barroso, L.A. — *The Tail at Scale*** → §4.4 (p50/p95, headroom, noise $\varepsilon$)
  Communications of the ACM, 56(2), 2013, pp. 74–80. DOI: https://doi.org/10.1145/2408776.2408794

- **[R21] McCalpin, J.D. — STREAM: *Sustainable Memory Bandwidth in High Performance Computers*** → G-M5-6
  Technical report, University of Virginia, 1991–2007 (continually updated). https://www.cs.virginia.edu/stream/
  Standar industri untuk bandwidth RAM sustainable (Copy/Scale/Add/Triad); aturan array ≥ 4× LLC. Teoritis DDR4-3200 = 3200 MT/s × 8 B = 25,6 GB/s per kanal — yang di-gate adalah hasil ukur, bukan angka ini.

## E. Storage / SSD I/O

- **[R22] Linux manual — `open(2)` O_DIRECT + `read(2)` alignment**
  https://man7.org/linux/man-pages/man2/open.2.html · https://man7.org/linux/man-pages/man2/read.2.html
  O_DIRECT meminimalkan efek cache; ada batasan alignment (buffer, offset, panjang) yang bervariasi per filesystem/kernel — misaligned bisa `EINVAL` atau fallback buffered; sejak Linux 6.1 bisa di-query via `statx(2)` `STATX_DIOALIGN`.

- **[R23] Linux manual — `pread(2)`**
  https://man7.org/linux/man-pages/man2/pread.2.html
  `pread` membaca di offset tanpa mengubah file offset; short read (lebih sedikit dari yang diminta) bukan error — wajib di-loop.

- **[R24] NVM Express — arsitektur antrean**
  https://nvmexpress.org/faq-items/what-makes-nvme-architecture-so-efficient
  NVMe mendukung hingga 65.535 I/O queue × 65.535 command per queue, dipetakan ke core CPU — dasar pengujian QD/prefetch depth $q$ di F17.

- **[R25] Linux manual — `readahead(2)` + `posix_fadvise(2)`**
  https://man7.org/linux/man-pages/man2/readahead.2.html · https://man7.org/linux/man-pages/man2/posix_fadvise.2.html
  Readahead hanya mengisi page cache (jalur buffered); `POSIX_FADV_SEQUENTIAL` menggandakan window, `POSIX_FADV_RANDOM` mematikannya — dasar pencatatan jalur buffered vs O_DIRECT di G-M7-5.

- **[R26] Zhang et al. — *Power, Energy and Thermal Considerations in SSD-Based I/O Acceleration***
  HotStorage'14, USENIX. https://www.usenix.org/system/files/conference/hotstorage14/hotstorage14-paper-zhang.pdf
  SSD panas memicu power throttling dengan degradasi performa belasan persen — dasar pencatatan suhu + larangan run 1-detik sebagai bukti.

- **[R27] Grupp et al. — *The Harey Tortoise: Managing Heterogeneous Write Performance in SSDs***
  ATC'13, USENIX. https://www.usenix.org/system/files/conference/atc13/atc13-grupp.pdf
  Perangkat flash punya performa burst (kecepatan SLC) vs sustained yang berbeda sistematis (heterogeneous write performance) — dasar pemisahan $BW_{burst}$ vs $BW_{sustained}$ dan batas $D_{sus}\le30\%$.

## F. Pola engineering & baseline (informatif, bukan ground truth)

- **[R6] FareedKhan-dev / `kimi-k3-in-c`**
  https://github.com/FareedKhan-dev/kimi-k3-in-c
  Referensi pola inference CPU, streaming checkpoint, memory-budget-as-a-dial, incremental decode, dan engineering oracle/test.

- **[R11] ggml / llama.cpp — supported quantization types**
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h
  Referensi dukungan tipe quant seperti Q3_K/Q4_K dan tipe GGML saat ini; bukan ground truth untuk quantizer proyek.

- **[R12] llama.cpp PR #22673 — MTP support**
  https://github.com/ggml-org/llama.cpp/pull/22673
  PR #22673 berstatus merged (16 May 2026); dipakai hanya sebagai referensi eksperimen MTP pasca-M9, bukan oracle.

- **[R13] `kimi-k3-in-c` — Python tooling / reference boundary**
  https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/pyproject.toml
  Repository mendeskripsikan paket Python sebagai tooling untuk fixture generation, reference-vs-C conformance, dan cache replay; tokenizer juga dipisahkan dari engine native. [R6]
