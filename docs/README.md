# Spec — `disk-streaming-moe-engine`

Kontrak: setiap klaim "benar / lolos" = (1) rumus + (2) threshold + (3) metode ukur yang bisa diulang. Rumus tanpa ukur = opini; ukur tanpa rumus = tebakan.

## Peta Dokumen

### Common (lintas milestone)

- `00-overview.md` — ringkasan, tujuan/non-tujuan, prinsip, definisi, tabel milestone
- `01-architecture.md` — diagram, komponen C1–C9, boundary Rust/Mojo/Python, ground truth config, alur streaming, anggaran memori, ADR, delta port
- `02-math-models.md` — rumus F1–F17
- `03-testing.md` — strategi oracle-first, matriks U/P/O/I/B/F/R, verdict MATCH, benchmark & kalibrasi, fixtures, CI
- `04-quality.md` — definisi gate, gate per milestone, quality umum, DoD fase
- `05-security.md` — scope, STRIDE-lite T1–T7, kontrol K1–K6, SEC-1..SEC-6
- `06-risks.md` — R1–R8 + mitigasi

### Milestone

**Trial inti (correctness):**

- `milestones/M0-reader.md` — reader multi-shard → G-M0-\*
- `milestones/M1-head-path.md` — embed → norm → lm_head → G-M1-\*
- `milestones/M2-attention.md` — attention 1 layer → G-M2-\*
- `milestones/M3-moe.md` — MoE 1 layer (paling kritis) → G-M3-\*
- `milestones/M4-full-forward.md` — full 24 layer streaming → G-M4-\*

**Trial rekayasa (performa):**

- `milestones/M5-kv-decode.md` — KV + decode → G-M5-\*
- `milestones/M6-quantizer.md` — quant 4-bit sendiri → G-M6-\*
- `milestones/M7-odirect-lru.md` — O_DIRECT + LRU → G-M7-\*

**Frontier:**

- `milestones/M8-gdn.md` — Gated DeltaNet → G-M8-\*
- `milestones/M9-port.md` — port Qwen3.6-35B-A3B → G-M9-\*

**Konsolidasi & Produksi:**

- `milestones/M10-dismoen.md` — konsolidasi DISMOEN, unifikasi Qwen 3.6, pembersihan Qwen 1.5 → G-M10-\*
- `milestones/M11-core-scaling.md` — optimasi tri-pilar (multicore CPU F16, disk I/O streaming F17/F18, skalabilitas RAM F1-F3) → G-M11-\*
- `milestones/M12-chat-cli.md` — terminal chat CLI interaktif & native server OpenAI-compatible (`dismoen chat` & `dismoen serve`) → G-M12-\*

### Lampiran

- `appendices/A-calculations.md` — contoh hitung trial
- `appendices/B-playbook.md` — playbook mismatch
- `appendices/B1-tooling-contract.md` — kontrak `tools/` Rust/Python
- `appendices/C-references.md` — R1–R30
- `appendices/D-core-scaling.md` — bukti safe core F16 (Amdahl → Roofline → Tail)

## Urutan Baca per Peran

- Baru gabung: `00-overview.md` → `01-architecture.md` → `milestones/M0-reader.md` → M1..M4 berurutan.
- Implementor kernel: `02-math-models.md` §3.2–§3.3 + `03-testing.md` §4.3 + milestone terkait (M2/M3/M8).
- Performa: `02-math-models.md` §3.1/§3.5/§3.7 + `03-testing.md` §4.4 + M5/M6/M7 + `appendices/D-core-scaling.md`.
- Reviewer gate: `04-quality.md` + file milestone terkait + laporan run-id.
- Security: `05-security.md` + M0 (parser) + M4 (cgroup/atomic).

## Aturan Main

1. Milestone berikutnya tidak mulai sebelum gate saat ini hijau semua (`04-quality.md` §5.1).
2. Oracle PyTorch fp32 satu-satunya ground truth; llama.cpp hanya sanity.
3. `router-selection` FAIL = root-cause, bukan naikkan threshold.
4. Setiap milestone: gate hijau + laporan run-id + kalibrasi + update docs bila perilaku berubah.
5. Angka estimasi (TBM, placeholder $T_{comp}$) tidak boleh dipakai sebagai bukti acceptance.
