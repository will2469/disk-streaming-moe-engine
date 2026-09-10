---
name: ref-ground-truth
description: "Guards checkpoint/config facts for disk-streaming-moe-engine against authoritative sources (HF model cards, transformers code). Auto-triggers when adding or changing model dimensions, expert/router config, vocab, bias handling, checkpoint pins, or M9 port deltas. Keywords: model card, Qwen2MoeConfig, weight_map, attention_bias, norm_topk_prob, models.lock.json."
compatibility: "Requires bash, git; no toolchain needed"
metadata:
  version: "1.0.0"
  author: "will2469"
  license: "Apache-2.0"
  citations:
    - "Grup A, docs/appendices/C-references.md (R1–R5)"
---

# Skill: ref-ground-truth

> **Core Thesis:** Setiap angka dimensi/arsitektur harus bisa ditelusuri ke kartu model resmi atau kode `transformers` — tidak pernah ke ingatan, `config.json` mentah, atau postingan blog. Dua jebakan permanen (§2.3) ada karena seseorang percaya config mentah.

## Workflow

**Step 1 — Ambil dari sumber, bukan dari kepala.** Dimensi trial (hidden 2048, 24 layer, 60 expert top-4 inter 1408, shared 5632 sigmoid, MHA 16×128, vocab 151936 untied) → [R1][R2]. Perilaku router/gate → [R3]. Delta port → [R4][R5].

**Step 2 — Verifikasi silang config-vs-index.** `config.json` tidak menulis `attention_bias` padahal checkpoint punya 72 tensor bias; gate shared = sigmoid bukan softmax. Keduanya invariant test permanen (G-M0 property, G-M3-2).

**Step 3 — Pin revision.** Tautan model ≠ pin. Setiap shard: revision HF + SHA-256 di `models.lock.json`; mismatch → tolak start (SEC-1).

## Anti-pattern (DILARANG)

- Menyalin dimensi dari memori/chat tanpa membuka [R1]–[R5].
- Mengikuti `config.json` mentah tanpa cek `model.safetensors.index.json`.
- Menambah fakta arsitektur M9 dari rumor; yang belum terukur = TBM.

## Checklist

- [ ] Setiap angka baru menunjuk R1–R5 spesifik (bukan "docs Qwen").
- [ ] Jebakan §2.3 tetap hijau (property P-2, G-M3-2).
- [ ] `models.lock.json` terisi revision + hash bila menyentuh checkpoint.
