# 04 — Quality & Acceptance Criteria

> Bagian dari `disk-streaming-moe-engine`. Index: `README.md`.

## 5.1 Definisi Gate & Definition of Done

**Gate** adalah kondisi binary pass/fail yang diukur (bukan kesan). Milestone berikutnya tidak boleh dimulai sebelum gate milestone saat ini hijau semua. **DoD per milestone** = semua gate hijau + artefak ter-commit: laporan benchmark dengan run-id (`03-testing.md` §4.4), hasil kalibrasi, dan pembaruan README/spec bila perilaku berubah. Angka TBM wajib terisi sebelum gate dinyatakan hijau.

**Prinsip generalisasi 1-device (teori di spec, angka di laporan).** Spec ini hanya mengikat teori (rumus F#), metode ukur (`03-testing.md` §4.4), dan gate rasio/kalibrasi/monotonisitas. Angka absolut device (GB/s, tok/s, suhu, nama fs/SSD) hanya sah sebagai *fakta laporan* device yang diuji, bukan kesimpulan umum. Proyeksi ke device lain hanya lewat model terkalibrasi (F5/F13/F16/F17) beserta batas error-nya ($e_T$, $e_{T,core}$), dan wajib dilabeli proyeksi — bukan hasil ukur. Klaim dukungan kelas device / varian produk di luar scope spec.

Ringkasan gate per milestone ada di §5.2. Detail eksekusi per milestone ada di `milestones/M*.md`.

## 5.2 Gate per Milestone

**Trial inti (M0–M4):**

| Gate | Milestone | Kriteria | Rumus / Threshold | Metode |
|---|---|---|---|---|
| G-M0-1 | Reader | metadata 4.659 tensor == index (nama → shard, dtype, shape) | 100% match | `check-index` vs `weight_map` |
| G-M0-2 | Reader | predikat validitas F15 | 100% tensor lolos | unit U + P |
| G-M0-3 | Reader | file korup → clean error, tanpa crash/hang | 20/20 mutasi lolos | fuzz F (SEC-2) |
| G-M1-1 | Head path | MATCH strict logits | $\Delta_{max} \le 10^{-3}$; $\mathbb{A} = 100\%$ | 3 prompt × 16 token |
| G-M1-2 | Head path | anggaran memori | $M_{peak} \le 3{,}5$ GiB (F1) | VmHWM |
| G-M2-1 | Attn layer | MATCH strict (part attn) | $\Delta_{max} \le 10^{-3}$ | layer 0, 12, 23; L=16 |
| G-M3-1 | MoE layer | MATCH strict (part moe) | $\Delta_{max} \le 10^{-3}$ | layer 0, 12, 23 |
| G-M3-2 | MoE layer | invariant F8: SET top-4 identik & sigmoid shared gate | 100% dari 256 input acak | oracle + engine |
| G-M4-1 | Full forward | MATCH loose | $\Delta_{max} \le 10^{-2} \wedge \mathbb{A} \ge 99{,}9\% \wedge \Delta_{CE} \le 0{,}02$ | 5 prompt × 16 tok |
| G-M4-2 | Full forward | memori & waktu sanity | $M_{peak} \le 5$ GiB; selesai ≤ 5 mnt (NVMe) | VmHWM + timer |

**Rekayasa trial (M5–M7):**

| Gate | Milestone | Kriteria | Rumus / Threshold | Metode |
|---|---|---|---|---|
| G-M5-1 | KV cache | decode incremental == recompute | verdict loose (§4.3) | 64 token @ ctx 2K |
| G-M5-2 | KV cache | prediksi F2 vs ukur | $e_{KV} \le 5\%$ | log engine + sampler |
| G-M5-3 | KV cache | memori @4K ctx | $M_{peak} \le 5$ GiB | VmHWM |
| G-M5-4 | Decode | kalibrasi waktu F5 | $e_T \le 30\%$ | 30 run (`03-testing.md` §4.4) |
| G-M5-5 | Decode | kurva skala core (rasio) | monotonik + $S_{tok}\ge1$ + $e_{T,core}\le20\%$ + $c^*,r^*$ dilaporkan (F16) | sweep $c\in\{1,2,4,\dots\}\cap[1,C_{max}]$ |
| G-M5-6 | Decode | floor bandwidth RAM (gate minimal) | $BW_{RAM}\ge10$ GB/s single-thread Copy read-equiv (F5) | STREAM-like §4.4, median 10 run |
| G-M6-1 | Quantizer | error per tensor (grup 128) | $\varepsilon_{rel} \le 10^{-2}$ (F11) | semua tensor |
| G-M6-2 | Quantizer | ukuran file | $\lvert pred - meas \rvert / meas \le 10\%$ | `du` vs F11b |
| G-M6-3 | Quantizer | kualitas end-to-end | $\Delta\mathrm{PPL} \le +0{,}5 \wedge \mathbb{A} \ge 95\%$ (F12) | corpus 100×256 |
| G-M7-1 | O_DIRECT+LRU | bandwidth cold sequential | $BW_{seq}\ge$ 2,5 GB/s (F17a) | drop_caches, 5 run |
| G-M7-2 | O_DIRECT+LRU | model cache F13 | $e_T \le 30\%$ dengan HR terukur | 30 run warm/cold |
| G-M7-3 | Decode 4-bit | throughput engineering | ≥ 2 tok/s (forecast F5: $B_{tok}^{4bit}$ ≈ **1,066 GB**) | §4.4 |
| G-M7-4 | O_DIRECT+LRU | kurva core + I/O | $BW_{eff}$ independen $c$ + HR stabil ±5pp + $e_T\le30\%$ dgn F5+F16 di $c^*$ | sweep core warm/cold |
| G-M7-5 | O_DIRECT+LRU | pola I/O terkarakterisasi | $BW_{seq}$ + $BW_{exp}(q)$ + $q^*$ + $D_{sus}\le30\%$ + alignment verified (F17) | §4.4 dua pola, cold/warm, sustained |

**Frontier (M8–M9) — angka final TBM:**

| Gate | Milestone | Kriteria | Rumus / Threshold | Metode |
|---|---|---|---|---|
| G-M8-1 | GDN | chunked scan == naive oracle | $\Delta_{max} \le 10^{-3}$ (F14, strict) | 100 sekuens acak |
| G-M8-2 | GDN | state fixed-size terhadap $s$ | $M_{state} = H \cdot d_k \cdot d_v \cdot b$ konstan | sampler s∈{1K..32K} |
| G-M8-3 | GDN | throughput | chunked ≥ 2× naive (sanity) | timer |
| G-M9-1 | Port 35B | oracle layer-by-layer (proxy hybrid kecil) | sama M2/M3 | TBM saat checkpoint ada |
| G-M9-2 | Port 35B | full forward | sama M4 + $M_{peak} \le 7{,}5$ GiB | VmHWM |
| G-M9-3 | Port 35B | decode streaming | ≥ 0,5 tok/s cold (F3/F5) | §4.4 |
| G-M9-4 | Port 35B | KV GQA sesuai rumus | $e_{KV} \le 5\%$ (F2, $L_{att} \approx L/4$) | log + sampler |

Catatan scope angka absolut: threshold seperti $BW_{seq}\ge2{,}5$ GB/s, $\ge2$ tok/s, $BW_{RAM}\ge10$ GB/s, $\ge0{,}5$ tok/s adalah *amplop acuan kelas entry* (8 GB/DDR4/NVMe), bukan spesifikasi device tertentu. Lolos = memenuhi amplop **pada device yang diuji** + model terkalibrasi. Kesimpulan lintas device dari 1 device uji tidak sah — lihat prinsip §5.1.

## 5.3 Quality Gates Umum (lintas milestone)

1. **Determinisme**: verdict numerik reproducible — ulang 5× hasil identik (seed + threads lock).
2. **Hygiene kode**: tidak ada `panic` di data path (semua error lewat Result/errno-style); tidak ada `unwrap` di parser; `mojo format` & `ruff` bersih; complexity ≤ 15 per fungsi (Rust: clippy cognitive via `clippy.toml`; Python: ruff McCabe C901 via `ruff.toml`; Mojo: belum ada tool → review manual saat skill `mojo-1-0` aktif).
3. **Bukti ter-commit**: setiap milestone melampirkan laporan run-id + kalibrasi; klaim tanpa run-id ditolak review.
4. **Regresi senyap**: perubahan golden hash tanpa justifikasi = blocking (juga SEC-6).
5. **Dokumen hidup**: README/spec diperbarui di milestone yang mengubah perilaku atau konstanta.
6. **Batas ranah produk**: yang di-gate hanya kepatuhan protokol + kelengkapan catatan + kalibrasi model — bukan nilainya. Butir ranah produk berstatus *record-only* (wajib logged lengkap di laporan, tanpa pass/fail atas angkanya):
   1. bandwidth RAM (di-gate sebagai floor G-M5-6; angka tipikal hanya acuan),
   2. pola I/O storage (di-gate sebagai kurva F17/G-M7-5; angka brosur dilarang),
   3. OS/runtime: governor, proses background, swap, cgroup, scheduler/afinitas, THP, RLIMIT,
   4. CPU microarch: ISA/SIMD, hierarki cache, SMT, allocator (yang di-gate kurva rasio F16, bukan angka spesifik),
   5. workload di luar golden set: prompt/konteks/batch/sampling lain → proyeksi via model, bukan verdict,
   6. daya/thermal umum: kondisi daya + suhu dicatat (yang di-gate $D_{sus}$, bukan threshold suhu),
   7. toolchain selain yang di-pin di `pixi.toml` (+ `mojo format` & `ruff` bersih).
   Aman dengan 1 device: gate correctness (MATCH, F15, fuzz, determinisme), gate kalibrasi per-device ($e_{KV}$, $e_T$, $e_{T,core}$), dan gate rasio/monotonisitas (F16, F17, HR).

## 5.4 Definition of Done per Fase

- **Fase Trial** = G-M0..G-M7 hijau semua + laporan kalibrasi (F1, F2, F5, F13, F16, F17) + retro jebakan (diperbarui dari `01-architecture.md` §2.3).
- **Fase GDN** = G-M8 hijau + catatan deviasi bentuk F14 terhadap paper.
- **Fase Port** = G-M9 hijau + tabel delta `01-architecture.md` §2.7 terisi angka nyata (TBM → measured).
