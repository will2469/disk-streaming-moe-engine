# 05 — Spesifikasi Security

> Bagian dari `disk-streaming-moe-engine`. Index: `README.md`.

## 6.1 Ruang Lingkup Security

Engine ini **offline**: tidak ada panggilan jaringan di runtime, sehingga permukaan serang terbatas pada file model + index + header JSON + token input + toolchain build. Runtime orchestration dilakukan Rust→Mojo, dengan Python/PyTorch berperan di tahap offline (oracle/fixture). Data pengguna tidak keluar mesin, dan output (logits/bins) adalah artefak lokal. Catatan batas: agent orchestration / tool-calling **bukan** bagian repo ini; jika nanti ditambahkan, wajib spec sandbox terpisah (pola minimum: allowlist perintah, namespace tanpa jaringan, timeout, confirmation gate) sebelum satu baris pun ditulis.

Relevansi per milestone:

- M0 → SEC-1..SEC-3 paling kritis (parser adalah perimeter).
- M1–M4 → SEC-4 (cgroup), SEC-5 (atomic write), SEC-6 (golden hash).
- M5–M9 → SEC-4 tetap (KV/LRU/quant/GDN/port tidak boleh jebolkan batas RAM/disk).

## 6.2 Model Ancaman (STRIDE-lite)

| ID | Ancaman | Vektor | Dampak | Kontrol |
|---|---|---|---|---|
| T1 | Rantai pasok model | shard ditrombak saat unduh / sumber palsu | output dimanipulasi; exploit parser | SEC-1 (SHA-256 + pin revision HF), K5 |
| T2 | Parser exploit | header JSON liar, offset negatif, dtype tak dikenal, file truncation | OOB read, panic, OOM | SEC-2/SEC-3 (F15, batas ukuran, fuzz) |
| T3 | Resource exhaustion | file raksasa, alloc liar, ctx tak masuk akal | OOM kill sesi, disk penuh | SEC-4 (cgroup, RLIMIT, cek pra-alloc) |
| T4 | Penulisan tak sengaja | engine/tools menimpa model dir atau home | checkpoint rusak | SEC-5 (model read-only, atomic write ke workdir) |
| T5 | Rantai pasok toolchain | install script dipipe ke bash | kompromi build | K6 (pin versi pixi.toml, review sebelum install) |
| T6 | Output salah yang "terlihat benar" | regresi senyap lolos review | kepercayaan engine palsu | SEC-6 (golden hash + oracle wajib) — beririsan dengan `04-quality.md` §5.3 |
| T7 | (Future) agent tooling | tool shell berbahaya via model | kompromi sistem | di luar scope — lihat §6.1 |

## 6.3 Kontrol (K1–K6)

- **K1 — Supply chain model**: unduhan via `hf download` dengan **pin revision** (commit HF), lalu SHA-256 setiap shard dicatat di `models.lock.json` dan diverifikasi otomatis sebelum run pertama; mismatch → engine menolak start (F15 + hash).
- **K2 — Parser hardening**: semua invarian F15 ditegakkan sebelum satu byte data dibaca: batas `header_len` ≤ **100 MB**, jumlah tensor ≤ 100.000, dtype himpunan eksak {BF16, F32, F16, F64} + konsistensi F15c, nama unik intra-header (F15) dan unik global saat merge (aturan MERGE), tidak ada rekursi di parser JSON; alokasi memori hanya setelah ukuran tervalidasi terhadap `filesize` (tidak ada alloc dari angka header mentah). [R7][R8]
- **K3 — Fuzz & negative tests**: corpus 20+ mutasi (header length liar, offset overflow/negatif, BEGIN>END, lubang/overlap buffer, dtype asing, layout mismatch, truncation di tengah data, duplikat kunci JSON / nama lintas shard, JSON rusak) dijalankan pada setiap perubahan parser; kegagalan = crash/hang/OOM apa pun, bukan hanya exit code.
- **K4 — Resource guard**: run validasi/exec di bawah cgroup `memory.max=6G` (trial) dan `RLIMIT_FSIZE`; cek ruang disk ≥ 1,5× ukuran model sebelum unduhan; setiap alloc disanitasi terhadap batas turunan config (bukan angka dari file).
- **K5 — File perms & workspace**: direktori model dibuat read-only (0444/0555) setelah verifikasi; engine hanya menulis ke workdir output; semua tulisan biner atomic (tmp + rename) agar tidak ada bins setengah jadi yang memicu false-MATCH.
- **K6 — Toolchain**: versi pixi & paket Mojo di-pin di `pixi.toml`; install script direview/sebelum dieksekusi; baseline llama.cpp hanya dari release resmi + checksum, dan tidak pernah dipakai sebagai oracle (D4).

## 6.4 Security Acceptance Criteria

| ID | Kriteria | Threshold | Metode |
|---|---|---|---|
| SEC-1 | Integritas model | SHA-256 semua shard == `models.lock.json`; tamper 1 byte → tolak start | 100%; test tamper |
| SEC-2 | Parser robustness | 20+ mutasi korup → 0 crash, 0 hang, 0 OOM; semua clean error | fuzz F |
| SEC-3 | Predikat F15 ditegakkan | tensor ilegal tidak pernah dibaca; alloc hanya dari ukuran tervalidasi | unit + fuzz |
| SEC-4 | Batas resource | forward lolos di bawah `memory.max=6G`; RLIMIT_FSIZE aktif; cek disk 1,5× | cgroup test |
| SEC-5 | Higienis file | model dir read-only saat engine jalan; output hanya workdir; atomic rename | CI + audit manual |
| SEC-6 | Regresi senyap | golden bins hash: 0 perubahan tak terjelaskan | level R tiap commit |
