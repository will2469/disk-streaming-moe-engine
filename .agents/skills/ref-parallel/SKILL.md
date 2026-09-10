---
name: ref-parallel
description: "Backs parallelism and performance claims for disk-streaming-moe-engine with papers (Amdahl, Roofline, MoE theory). Auto-triggers when writing F4/F5/F9/F14/F16 formulas, core-scaling curves, prefill-vs-decode analysis, load-balance diagnostics, or DeltaNet work. Keywords: Amdahl, roofline, speedup, load-balance, delta rule, knee, memory-bound."
compatibility: "Requires bash, git; pairs with mojo-1-0"
metadata:
  version: "1.0.0"
  author: "will2469"
  license: "Apache-2.0"
  citations:
    - "Grup C, docs/appendices/C-references.md (R9–R10, R14–R16, R18, R20)"
    - "docs/appendices/D-core-scaling.md"
---

# Skill: ref-parallel

> **Core Thesis:** Setiap klaim paralelisme/skala memetakan ke paper → rumus → gate. Klaim tanpa paper = opini; angka device tanpa model = anekdot 1-device (§5.1).

## Peta klaim → paper → rumus (what we use)

| Klaim | Paper | Dipakai |
|---|---|---|
| Load-balance MoE | [R10] Switch Transformer | F9, koreksi $\rho$ M7 |
| Delta rule chunked | [R9] Yang et al. | F14, M8 (oracle = naive loop) |
| Speedup + knee | [R14] Amdahl 1967 | F16a, $c^*$ gain <10% |
| Decode problem-tetap ⇒ Amdahl | [R15] Gustafson 1988 | F16 (bukan scaled-speedup) |
| Operasi di $c^*$, rasio $r^*$ | [R16] Hill & Marty 2008 | F16d, headroom OS |
| Prefill compute / decode memory | [R18] LLM Inference Unveiled | F4, uji BW-datar G-M7-4 |
| Overhead dispatch $\beta$ | [R20] TaxBreak (pendukung) | F16b — bukan normatif |

## Anti-pattern (DILARANG)

- Klaim speedup linear sampai $C_{max}$; Gustafson untuk decode 1-token.
- Angka core absolut di spec (pakai $r$, $c^*$); kesimpulan lintas device dari 1 device.
- [R20] sebagai satu-satunya dasar gate (kalah vs [R14]–[R19] bila konflik).

## Checklist

- [ ] Rumus baru menunjuk paper + gate (F16/F17 butuh keduanya).
- [ ] Kurva monotonik + knee + $e_{T,core}$ dilaporkan, bukan sekadar titik tercepat.
- [ ] Detail bukti: `D-core-scaling.md`.
