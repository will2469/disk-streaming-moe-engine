---
name: ref-core-scaling
description: "Guards CPU core-scaling claims for disk-streaming-moe-engine (Amdahl curve F16, safe ratio, knee). Auto-triggers when writing core sweeps, fitting p/beta, choosing operating thread counts, or working G-M5-5/G-M7-4. Keywords: core scaling, Amdahl, knee, safe ratio, operating point, F16, c-star."
compatibility: "Requires bash, git; pairs with ref-parallel and ref-perf"
metadata:
  version: "1.0.0"
  author: "will2469"
  license: "Apache-2.0"
  citations:
    - "docs/appendices/D-core-scaling.md"
    - "Grup C, docs/appendices/C-references.md (R14–R16, R20)"
---

# Skill: ref-core-scaling

> **Core Thesis:** Skala core di-gate sebagai kurva rasio ($r$, knee, $e_{T,core}$) — bukan titik tercepat, bukan angka absolut. Titik operasi = $c^*$, bukan $C_{max}$.

## Workflow

**Step 1 — Sweep rasio.** $c \in \{1,2,4,\dots\} \cap [1,C_{max}]$ (terdeteksi run-time); tiap level 10 run + 2 warm-up, 30 run di $c^*$; verdict numerik tetap `threads=1` terpisah (§4.4 butir 4).

**Step 2 — Fit F16.** $T_{tok}(c) = T_{IO} + T_1/S(c) + \beta(c-1)$, $S(c)$ Amdahl [R14]; lapor $p, \beta, S_{tok}, \eta, M(c{\to}2c)$, knee $c^*$ (gain <10%), $r^* = c^*/C_{max}$.

**Step 3 — Gate.** Monotonik ($\times1{,}05$ noise), $S_{tok}\ge1$, $e_{T,core}\le20\%$ (G-M5-5); di M7 tambah BW-datar + HR stabil (G-M7-4). Detail bukti: `D-core-scaling.md`.

## Anti-pattern (DILARANG)

- Klaim linear / gaspol $C_{max}$ (melanggar [R16]: sisakan $1-r^*$ untuk OS).
- Angka core absolut di spec/docs (hook menolak).
- Gustafson untuk decode 1-token; [R20] sebagai dasar tunggal.

## Checklist

- [ ] $p, \beta, c^*, r^*, e_{T,core}$ ter-commit; proyeksi dilabeli proyeksi (§5.1).
- [ ] Sweep + fit terdokumentasi di laporan run-id (bukan cuma angka tok/s akhir).
