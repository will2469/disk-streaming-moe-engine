# 03 — Spesifikasi Testing

> Bagian dari `disk-streaming-moe-engine`. Index: `README.md`.

## 4.1 Strategi: Oracle-First

Ground truth proyek ini adalah **PyTorch fp32** (`tools/oracle/oracle_head.py`, `tools/oracle/oracle_layer.py`), bukan llama.cpp — output quant Q4/Q3 tidak akan pernah byte-identical dengan fp32, jadi llama.cpp hanya baseline kelayakan (C9). Setiap kemampuan engine wajib lolos ekivalensi sebelum milestone berikutnya dimulai; **Rust `compare` menjadi satu-satunya verifier/pengumpul verdict**, sementara PyTorch hanya menghasilkan reference artifacts; urutan 5 langkah di README §4 bersifat wajib dan tidak boleh dilompati karena setiap langkah mengisolasi satu class bug (reader → head → attn → moe → full). Perbedaan numerik harus dipahami, bukan ditoleransi: setiap FAIL wajib dikategorikan (§4.3) sebelum boleh diperbaiki, dan menaikkan threshold untuk "menyelesaikan" FAIL adalah pelanggaran spec.

## 4.2 Matriks Pengujian

| Level | Scope | Tooling | Frekuensi | Kriteria lolos |
|---|---|---|---|---|
| U — Unit | F10 metrics, tokenizer roundtrip, merge index 3 shard, config loader | **cargo test + pytest (oracle only)** | tiap commit | 100% pass; coverage util ≥ 85% |
| P — Property | softmax stabil (max shift), RoPE isometri (F7), quant roundtrip (F11), predikat F15 pada shape acak, config-vs-index (jebakan `01-architecture.md` §2.3) | hypothesis | nightly | semua invariant lolos |
| O — Oracle equivalence | 5 langkah README §4 (head, attn, moe, full) | **PyTorch oracle + Rust compare + Mojo binary** | tiap build | verdict MATCH (§4.3) |
| I — Integration | forward end-to-end 5 prompt, exit code CLI, schema report JSON | **Rust CLI + Mojo + fixture** | tiap build | 100% sesuai kontrak C1/C7 |
| B — Benchmark | waktu prefill, VmHWM, bytes dibaca, (M5+) tok/s & BW | harness §4.4 | per milestone + on-demand | laporan + kalibrasi ≤ gate |
| F — Fuzz/negative | 20+ mutasi file korup (header liar, offset negatif, BEGIN>END, lubang/overlap, dtype asing, layout mismatch, truncation, duplikat kunci/nama, JSON rusak) | corpus mutasi | tiap perubahan parser | 0 crash / 0 hang / 0 OOM; semua clean error (SEC-2) |
| R — Regression | golden bins + SHA-256, fixture synthetic | **cargo test + pytest oracle generation (sesuai kebutuhan)** | tiap commit | hash stabil; perubahan tanpa alasan = blocking |

Pemakaian per milestone:

- M0 → U, P (F15, config-vs-index), F, R (shape-fidelity)
- M1–M4 → O (head/attn/moe/full), I, B (prefill), R
- M5 → O (KV vs recompute), B (decode 30 run)
- M6 → P (quant roundtrip), O (ΔPPL), B
- M7 → B (cold/warm), F (O_DIRECT error path)
- M8 → O (strict vs naive), B (chunked ≥2× naive)
- M9 → O, I, B full

## 4.3 Metrik Ekivalensi & Aturan Verdict

Semua metrik dari F10 (`02-math-models.md` §3.3). `compare.py` wajib melapor kelima metrik + kategori FAIL; verdict mengikuti tabel:

| Konteks | Verdict **MATCH** jika | Catatan |
|---|---|---|
| M1 head path (logits) | $\Delta_{max} \le 10^{-3} \wedge \varepsilon_{rel} \le 10^{-4} \wedge \mathbb{A} = 100\%$ | strict |
| M2 attn layer (per part) | $\Delta_{max} \le 10^{-3} \wedge \varepsilon_{rel} \le 10^{-4}$ | layer 0, 12, 23; L=16 |
| M3 moe layer | $\Delta_{max} \le 10^{-3} \wedge \varepsilon_{rel} \le 10^{-4} \wedge$ SET top-4 identik 100% | seleksi bukan opsi |
| M4 full forward | $\Delta_{max} \le 10^{-2} \wedge \varepsilon_{rel} \le 10^{-4} \wedge \mathbb{A} \ge 99{,}9\% \wedge \Delta_{CE} \le 0{,}02$ | loose (akumulasi urutan penjumlahan) |
| M5 KV decode | sama dengan M4, dibandingkan recompute | |
| M8 GDN | $\Delta_{max} \le 10^{-3}$ | strict vs naive loop |
| M9 port 35B | sama dengan M4 (angka final TBM) | |

Kategori FAIL (playbook debugging, README §4): `router-selection` (SET beda), `rope-style` (pola ~1e-1 stabil), `bias-placement` (pola ~1e-1..1 acak), `numeric-order` (semua beda sedikit, pola sama → wajar fp32), `dtype-layout` (transpos/stride salah). Aturan keras: kategori `router-selection` tidak pernah diselesaikan dengan menaikkan threshold — wajib root-cause.

Detail playbook: `appendices/B-playbook.md`.

## 4.4 Protokol Benchmark & Kalibrasi

1. Lingkungan terkunci: AC/plug-in stabil, CPU governor `performance`, aplikasi lain ditutup; kondisi dicatat di header laporan.
2. Cold read: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` sebelum run cold.
3. Warm-up 2× (tidak dihitung), lalu N run terukur: N=5 untuk prefill (M4), N=30 untuk decode (M5+); temperature 0 / greedy, seed tetap, prompt dari golden set.
4. Sweep core F16 (M5+): $c$ kelipatan 2 hingga $C_{max}$ (terdeteksi run-time, tidak dipatok di spec); tiap level 10 run + 2 warm-up, lalu 30 run di $c^*$; catat $C_{max}, c, r=c/C_{max}$, governor, `OMP_NUM_THREADS=c`. Verdict numerik tetap `threads=1` terpisah dari sweep performa.
5. Sampling: RSS via VmHWM + poller 100 ms; bytes I/O via `/proc/<pid>/io`; waktu per fase dari log engine.
6. Ukur $BW_{RAM}$ ala STREAM [R21] (gate G-M5-6): kernel Copy single-thread, array ≥ 4× total LLC (atau ≥ 1M elemen), 10 repetisi, ambil median read-equiv GB/s; governor `performance`; $C_{max},c$ dan kondisi daya dicatat. Array harus jauh lebih besar dari cache agar yang terukur bandwidth RAM, bukan cache.
7. Ukur pola I/O storage F17 (gate G-M7-5): dua pola terpisah — (a) sequential blok ≈4 MB QD1 cold (`drop_caches`), (b) blok seukuran expert melompat antar offset pada QD $q\in\{1,2,4,\dots\}$; tiap pola 5 run cold + 10 run warm; satu pass sustained $\ge W_{file}$ untuk $D_{sus}$; catat filesystem, opsi mount, triple alignment O_DIRECT, $q$, suhu SSD, dan daya. Run 1-detik tidak sah sebagai bukti.
8. Output: CSV per run + laporan markdown (p50/p95, min/max) dengan **run-id**; disimpan `reports/YYYY-MM-DD/`.

Kalibrasi rumus (wajib di setiap gate performa):

$$e_{KV} = \left|\frac{M_{KV}^{pred} - M_{KV}^{meas}}{M_{KV}^{meas}}\right| \le 0{,}05 \qquad e_{T} = \left|\frac{T^{pred} - T^{meas}}{T^{meas}}\right| \le 0{,}30\ (\text{trial}) \to 0{,}20\ (\text{M9}) \qquad e_{T,core} = \left|\frac{T^{pred}(c) - T^{meas}(c)}{T^{meas}(c)}\right| \le 0{,}20\ \text{(kurva F16)}$$

Kegagalan kalibrasi tidak memblokir milestone, tetapi wajib memicu pembaruan konstanta (ρ, BW) di `02-math-models.md` dan catatan di laporan — prediksi boleh meleset, dokumen tidak boleh bohong.

## 4.5 Fixtures & Golden Set

- **Synthetic mini checkpoint**: config varian kecil (mis. L=2, H=2, 8 expert, $d_e$=64, vocab=512) dengan bobot acak seed 42; oracle membaca config yang sama → seluruh level U/P/O/I jalan **tanpa unduhan 28,6 GB** (prasyarat CI di mesin 8 GB).
- **Shape-fidelity test**: index asli 4.659 tensor divalidasi strukturnya tanpa download (mendukung G-M0-1).
- **Golden set teks**: 50 prompt (commit ke repo), `tokens.json` precomputed; golden bins + SHA-256 disimpan untuk level R.
- **Corpus PPL**: 100 prompt × 256 token untuk ΔPPL (G-M6-3).

## 4.6 CI & Determinisme

- `cargo test` — Rust tooling/unit tests; `pytest` — oracle/fixture tests; `make validate` — orkestrasi seluruh level U/P/I + O dengan fixture synthetic (tanpa model asli); `make validate-full` — O dengan model 28,6 GB.
- Verdict numerik hanya sah pada mode **threads=1**; mode performa (threads=nproc) dijalankan terpisah dan tidak pernah dipakai untuk verdict.
- Seed, versi toolchain (pixi.toml), dan schema report di-versioning; `mojo format` + `ruff` wajib bersih.
