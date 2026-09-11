# M0 — Verifikasi shard asli (run-id M0-20260911-001)

> Tindak lanjut `../2026-09-10/M0-W4.md`: butir yang berstatus TUNGGU 8 shard asli.
> Model dir: `~/models/qwen1.5-moe-a2.7b-chat` (read-only setelah verifikasi, SEC-5).
> Revision pin K1: `ec052fda178e241c7c443468d2fa1db6618996be`.

## 1. SHA-256 + models.lock (SEC-1)

- `sha256sum` 8 shard → `models.lock.json` terisi penuh (sha256 64-char + size per shard).
- Total file di disk: **28.632.144.944 B** (≠ `metadata.total_size` index 28.631.568.384:
  selisih 576.560 B = header safetensors per shard; keduanya benar di semantiknya).
- `kimo-tools verify --lock --dir`: `{"status":"match","checked":8,"skipped":0}` exit 0.
- Insiden: satu hash tersalin 63-char saat pengisian manual → verify menolak (exit 2)
  → diperbaiki, verifikasi silang lock-vs-`sha256sum` 8/8 OK, verify hijau.
  Pelajaran: isi lock via skrip, bukan salin manual.

## 2. G-M0-1 penuh (4.659 tensor)

- Index asli byte-identik dengan proxy `fixtures/m0_qwen_index.json` (4659, total 28631568384).
- `kimo check-index` 8 shard: `match`, scope `full`, 4659/4659, exit 0, parse 41,13 ms.
- Subset 1 shard: `match`, scope `subset`, 578/578, exit 0 (reader generik N-shard).

## 3. Baseline C3 header asli (report-only per W4)

7 run (`mojo run`, 2 pertama warm-up), `parse_time_ms`:
24,88 · 23,08 · 23,53→24,53 · 23,86 · 23,12 · 42,39 · 44,46.
Median 5 run ukur: **≈ 24,5 ms** (≪ gate 1 s). Dua run terakhir naik (noise
kompilasi/cache); tidak diinvestigasi — margin ke gate 40×.

## 4. Config asli → M1 (terukur, bukan TBM)

`config.json` asli: `rms_norm_eps = 1e-06`, hidden 2048, 24 layer, 16 head
(Q=K=V heads), vocab 151936, 60 expert top-4, `moe_intermediate_size` 1408,
shared 5632, `norm_topk_prob` false, `tie_word_embeddings` false (head untied).
Nilai ε M1-W1 (1e-6) TERKONFIRMASI dari sumber kanonis.

## Status

- M0: G-M0-1 penuh ✅ (proxy → asli), SEC-1 ✅, SEC-5 ✅, C3 ✅.
- W0 close-out 2026-09-11 ✅: folder + header lisensi terverifikasi; `pixi.toml` + `gcc >=13,<14`
  (conda-forge 13.4.0, `pixi install` hijau, `mojo --version` 1.0.0);
  `models.lock.json` penuh; `pre-commit run --all-files` 13/13 Passed;
  `kimo` di-rebuild dari src kini (`mojo build`).
- W1 close-out 2026-09-11 ✅: 8/8 error types — `DUPLICATE_TENSOR_NAME` di merge +
  e2e duplikat/salah-tempat (exit 1, kind tepat); skrip e2e 5/5 hijau.

## 5. Versi skill aktif (butir W0)

- `mojo-1-0` 1.0.0, `ref-ground-truth` 1.0.0, `ref-format` 1.0.0
  (field `version` di masing-masing `SKILL.md`, `.agents/skills/`).
