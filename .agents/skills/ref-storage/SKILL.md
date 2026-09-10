---
name: ref-storage
description: "Backs storage I/O claims for disk-streaming-moe-engine (O_DIRECT, NVMe queues, sustained vs burst). Auto-triggers when writing F17 formulas, O_DIRECT readers, LRU/prefetch tuning, disk benchmarks, or G-M7-5 work. Keywords: O_DIRECT, pread, NVMe, queue depth, readahead, sustained, thermal, G-M7-5."
compatibility: "Requires bash, git; pairs with mojo-1-0 and ref-format"
metadata:
  version: "1.0.0"
  author: "will2469"
  license: "Apache-2.0"
  citations:
    - "Grup E, docs/appendices/C-references.md (R22–R27)"
---

# Skill: ref-storage

> **Core Thesis:** "NVMe" saja bukan angka. Dua pola (trunk sequential vs expert-miss) diukur terpisah; burst bukan sustained; suhu dicatat.

## Peta klaim → sumber → rumus

| Klaim | Sumber | Dipakai |
|---|---|---|
| Alignment triple + `EINVAL`/fallback | [R22] `open(2)`/`read(2)` | F17, reader O_DIRECT |
| Short-`pread` di-loop (bukan error) | [R23] `pread(2)` | F17, I/O loop |
| 64K queue × 64K depth per core | [R24] NVM Express | F17, sweep prefetch $q$/$q^*$ |
| Readahead hanya jalur buffered | [R25] `readahead(2)`/`posix_fadvise(2)` | G-M7-5 (catat jalur) |
| Thermal throttle belasan persen | [R26] HotStorage'14 | G-M7-5 (suhu logged, anti run-pendek) |
| Burst vs sustained sistematis | [R27] ATC'13 Harey Tortoise | F17d, $D_{sus}\le30\%$ |

## Anti-pattern (DILARANG)

- Satu angka $BW_{SSD}$ untuk trunk + expert-miss; angka brosur sebagai $BW_{eff}$.
- $BW_{eff}$ naik ikut $c$ lalu diklaim memory-bound (itu compute-bound — investigasi, G-M7-4).
- Mencampur O_DIRECT + buffered pada region overlap.

## Checklist

- [ ] $BW_{seq}$ + $BW_{exp}(q)$ + $q^*$ + $D_{sus}$ terukur cold/warm + sustained $\ge W_{file}$.
- [ ] fs, mount opts, alignment, $q$, suhu, daya logged (§4.4 butir 7).
