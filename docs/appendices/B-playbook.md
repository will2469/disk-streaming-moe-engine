# Lampiran B — Playbook Debugging Mismatch (Ringkas)

> Bagian dari `disk-streaming-moe-engine`. Index: `../README.md`.
> Verdict: `../03-testing.md` §4.3. Rumus: `../02-math-models.md`.

Urutan pengecekan saat verdict FAIL:

1. Baca kategori FAIL dari `compare.py` →
2. `router-selection`: softmax fp32 + tanpa renorm + sigmoid shared gate (F8) →
3. `rope-style`: rotate_half vs interleaved, invariant isometri F7 →
4. `bias-placement`: 72 tensor QKV bias, untied lm_head →
5. `numeric-order`: beda kecil merata = wajar fp32, naikkan konteks ke loose (M4 saja) →
6. `dtype-layout`: transpos/stride salah pada bobot yang di-stream.

Kategori (2)–(4) biasanya menghasilkan $\Delta_{max} \sim 10^{-1}..1$; kategori (5) $\sim 10^{-3}$ atau kurang.

Aturan keras:

- `router-selection` tidak pernah diselesaikan dengan menaikkan threshold — wajib root-cause.
- Menaikkan threshold untuk "menyelesaikan" FAIL adalah pelanggaran spec.
- Setiap FAIL wajib dikategorikan sebelum boleh diperbaiki.

Rujukan milestone:

- M2 → `../milestones/M2-attention.md` (rope, bias)
- M3 → `../milestones/M3-moe.md` (router, shared gate)
- M4 → `../milestones/M4-full-forward.md` (loose hanya di sini)
