---
name: ref-patterns
description: "Guards informative engineering references for disk-streaming-moe-engine (kimi-k3-in-c patterns, quant baselines). Auto-triggers when citing kimi-k3-in-c, comparing against llama.cpp/GGUF, scoping MTP experiments, or drawing tooling boundaries. Keywords: kimi-k3-in-c, GGUF, llama.cpp, MTP, baseline, oracle boundary."
compatibility: "Requires bash, git"
metadata:
  version: "1.0.0"
  author: "will2469"
  license: "Apache-2.0"
  citations:
    - "Grup F, docs/appendices/C-references.md (R6, R11–R13)"
---

# Skill: ref-patterns

> **Core Thesis:** Grup F itu pola dan pembanding — informatif, bukan ground truth. Satu-satunya oracle adalah PyTorch fp32 (two-tier: Candle pun wajib MATCH vs PyTorch dulu).

## Batas pakai

| Sumber | Boleh | Dilarang |
|---|---|---|
| [R6]/[R13] kimi-k3-in-c | Pola streaming, LRU, oracle/test engineering | Dijadikan bukti correctness gate |
| [R11] GGUF/llama.cpp | Baseline sanity kelayakan (C9) | Dijadikan oracle (D4) atau ground truth quantizer |
| [R12] MTP PR | Referensi eksperimen pasca-M9 | Masuk roadmap inti (D8: out of scope) |

## Anti-pattern (DILARANG)

- "llama.cpp benar" sebagai argumen debug (R8 di 06-risks: duplikasi bug orang lain).
- Quantizer proyek meniru GGUF tanpa lewat F11/G-M6-1..3 sendiri (D6).
- Menggeser Python/oracle boundary tanpa ADR (D9).

## Checklist

- [ ] Sitasi F selalu berlabel pola/baseline/eksperimen — tidak pernah "ground truth".
- [ ] Perbandingan numerik vs baseline memakai F10 + kategori FAIL, bukan kesan.
