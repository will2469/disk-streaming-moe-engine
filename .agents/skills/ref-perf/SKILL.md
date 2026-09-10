---
name: ref-perf
description: "Backs performance-model claims for disk-streaming-moe-engine (Roofline, tail latency, STREAM). Auto-triggers when writing F4/F5 formulas, bandwidth floors, benchmark protocols, p50/p95 reports, or G-M5-6 work. Keywords: roofline, bandwidth, STREAM, tail latency, p95, memory-bound, G-M5-6."
compatibility: "Requires bash, git; pairs with ref-parallel"
metadata:
  version: "1.0.0"
  author: "will2469"
  license: "Apache-2.0"
  citations:
    - "Grup D, docs/appendices/C-references.md (R17, R19, R21)"
---

# Skill: ref-perf

> **Core Thesis:** Klaim performa = model (Roofline) + angka ukur (STREAM-like) + statistik jujur (p50/p95). Angka brosur dan run 1-detik bukan bukti.

## Peta klaim → paper → rumus

| Klaim | Paper | Dipakai |
|---|---|---|
| Compute- vs memory-bound via $I$ | [R17] Roofline 2009 | F4, $I_{decode}\approx1$ |
| Headroom, p50/p95, toleransi noise | [R19] Tail at Scale 2013 | §4.4, $\varepsilon=5\%$, $1-r^*$ |
| Bandwidth sustainable + aturan 4× LLC | [R21] STREAM | G-M5-6 ($BW_{RAM}\ge10$ GB/s) |

## Anti-pattern (DILARANG)

- Memakai bandwidth teoritis (mis. 25,6 GB/s/kanal) sebagai $BW_{RAM}$ di F5 — yang sah hasil ukur.
- Mean tanpa p50/p95; run pendek sebagai bukti sustained.
- Utilisasi 100% sebagai target (melanggar headroom [R19]).

## Checklist

- [ ] $BW_{RAM}$ dari metode §4.4 butir 6 (array ≥4× LLC, median, governor tercatat).
- [ ] Laporan ada p50/p95 + run-id; proyeksi dilabeli proyeksi (§5.1).
