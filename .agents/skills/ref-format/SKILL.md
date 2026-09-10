---
name: ref-format
description: "Guards safetensors parsing facts for disk-streaming-moe-engine against the format spec (F15). Auto-triggers when writing or reviewing header parsers, index merging, dtype/shape validation, O_DIRECT readers, or fuzz corpora. Keywords: safetensors, header JSON, data_offsets, F15, check-index, fuzz, O_DIRECT alignment."
compatibility: "Requires bash, git; pairs with mojo-1-0 (I-7)"
metadata:
  version: "1.0.0"
  author: "will2469"
  license: "Apache-2.0"
  citations:
    - "Grup B, docs/appendices/C-references.md (R7–R8)"
---

# Skill: ref-format

> **Core Thesis:** File model adalah input tidak terpercaya. Setiap byte yang dibaca harus lolos predikat F15 dulu; alokasi dari angka header mentah adalah bug security, bukan optimasi.

## Workflow

**Step 1 — Format dulu, data kemudian.** File = `[len: u64][header JSON][data]` [R7]; contoh parsing + index sharded [R8].

**Step 2 — Tegakkan F15 sebelum 1 byte data dibaca:** konversi dulu (`data_base = 8 + header_len`; file = `data_base + offset`), lalu `0 ≤ BEGIN ≤ END ∧ data_base + END ≤ filesize`, dtype himpunan eksak + F15c, nama unik intra-header (F15) dan unik global saat merge (aturan MERGE), buffer penuh tanpa lubang; `header_len ≤ 100 MB`, tensor ≤ 100.000. SHA/revision = SEC-1, bukan F15 — jangan campur. Payload (offset ≥ `data_base`) dilarang dibaca: kontrak I/O M0.

**Step 3 — I/O aman.** Triple alignment O_DIRECT (Grup E, [R22]); short-`pread` di-loop ([R23]); buffer aligned + bounded; 20+ mutasi fuzz → 0 crash/hang/OOM (SEC-2).

## Anti-pattern (DILARANG)

- `malloc(header_len)` / alokasi dari angka file sebelum validasi `filesize`.
- Parser JSON rekursif; dtype tak dikenal di-default diam-diam.
- Mencampur O_DIRECT + buffered I/O pada byte region yang overlap ([R22]).

## Checklist

- [ ] G-M0-1..G-M0-3 hijau; fuzz corpus 20+ mutasi.
- [ ] Tidak ada `unwrap`/`panic` di parser (§5.3); tidak ada alloc dari angka mentah.
- [ ] Fakta format baru menunjuk R7/R8 spesifik.
