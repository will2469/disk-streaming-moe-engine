# Lampiran D — Bukti Safe Core (Fondasi F16)

> Bagian dari `disk-streaming-moe-engine`. Index: `../README.md`.
> Fondasi rumus `../02-math-models.md` §3.7 (F16) dan gate G-M5-5 / G-M7-4.
> Prinsip: tiap pilihan di F16 harus punya pasangan paper. Klaim tanpa paper = opini.

## D.1 Peta klaim → paper → implikasi

| # | Klaim di F16 | Paper / sumber | Implikasi ke gate |
|---|---|---|---|
| D1 | Speedup mengikuti Amdahl $S(c)=1/((1-p)+p/c)$, diminishing returns, ada knee | [R14] Amdahl 1967 | Bentuk F16a; definisi $c^*$ sebagai knee $M<10\%$; fit $p$ dari kurva ukur |
| D2 | Untuk problem ukuran tetap (1 token decode), pakai Amdahl; Gustafson hanya untuk problem yang ikut membesar | [R15] Gustafson 1988 | Decode (batch-1, $n=1$) di-gate dengan Amdahl; pandangan scaled-speedup tidak dipakai untuk menjustifikasi core tak terbatas di decode |
| D3 | Di chip multicore, speedup dibatasi sumber daya + fraksi sekuensial; core sekuensial yang cepat tetap penting | [R16] Hill & Marty 2008 | Titik operasi $c^*$ bukan $C_{max}$; sisakan $1-r^*$ untuk OS; lapor $r^*=c^*/C_{max}$ sebagai rasio, bukan angka absolut |
| D4 | Kernel diklasifikasikan compute-bound vs memory-bound via operational intensity $I$; $P=\min(P_{peak}, I\cdot BW)$ | [R17] Williams et al. 2009 (Roofline) | Fondasi F4; $I_{decode}\approx1$ → decode di sisi memory-bound → $T_{IO}$ diasumsikan independen $c$ (diuji di G-M7-4, bukan diasumsikan buta) |
| D5 | Pada LLM: prefill mayoritas compute-bound, decode semua memory-bound; quant menaikkan $I$ dan membantu decode | [R18] LLM Inference Unveiled 2024 | Prefill vs decode diperlakukan beda di sweep; uji "BW datar vs $c$" di G-M7-4 sebagai bukti memory-bound; quant (M6) dibaca sebagai pengurang bytes (F3b), bukan penambah FLOPs |
| D6 | Variabilitas (shared resources, daemon, queueing) memanjangkan tail; utilisasi penuh tanpa headroom memperburuk p95 | [R19] Dean & Barroso 2013 | Wajib lapor p50/p95 + sisakan headroom ($1-r^*$); monotonisitas pakai toleransi noise $\varepsilon=5\%$; tidak menargetkan utilisasi 100\% |
| D7 | Decode MoE persisten host-bound; single-thread CPU adalah parameter orde-satu (pendukung, preprint 2026) | [R20] TaxBreak 2026 | Suku overhead $T_{ovh}(c)=\beta(c-1)$ di F16b; fit $\beta$; bila $\beta$ besar → investigasi dispatch/threading sebelum tambah core |

## D.2 Cara membaca tiap paper untuk F16

**[R14] Amdahl (1967).** Hukum: percepat fraksi $p$ sebesar $S$ → speedup keseluruhan $1/((1-p)+p/S)$. Saat $S\to\infty$, speedup $\to 1/(1-p)$. Dipakai apa adanya sebagai $S(c)$ di F16a. Konsekuensi yang di-gate: kurva pasti cekung; knee pasti ada; klaim "linear sampai $C_{max}$" ditolak review.

**[R15] Gustafson (1988).** Koreksi: bila ukuran problem ikut membesar dengan resources, speedup terukur bisa tampak linear (scaled speedup $S=(1-p)+pN$). Decode kita problem tetap (1 token, bobot tetap $B_{tok}$), jadi rezim yang benar = Amdahl. Prefill konteks-besar boleh dibahas dengan kacamata scaled, tapi gate M5/M7 memakai Amdahl.

**[R16] Hill & Marty (2008).** Menambah model biaya hardware ke Amdahl untuk chip multicore (symmetric/asymmetric, BCE). Hasil kunci: speedup bagus butuh $p$ sangat dekat 1; core sekuensial yang cepat tetap berharga. Implikasi: (a) jangan habiskan semua budget di banyak core lambat — titik $c^*$ bisa $< C_{max}$; (b) $C_{max}$ hanya penyebut rasio, tidak dipatok di spec.

**[R17] Roofline (2009).** $P = \min(P_{peak}, I\cdot BW)$. Fondasi F4. Ridge point memisahkan dua rezim. $I_{decode}\approx 2N_{stream}/B_{tok}=1$ menaruh decode di sisi memory-bound; maka $T_{IO}=B_{tok}/BW_{eff}$ dominan dan independen $c$. G-M7-4 menguji independensi ini: bila $BW_{eff}$ terukur naik ikut $c$, asumsi gugur → investigasi (masih compute-bound / salah ukur), bukan klaim lolos.

**[R18] LLM Inference Unveiled (2024).** Survei + analisis Roofline per layer (contoh LLaMA-2-7B di A6000): prefill compute-bound, decode memory-bound; quant menaikkan arithmetic intensity sehingga membantu decode. Mendukung pemisahan prefill (N=5) vs decode (N=30 + sweep) di protokol §4.4 dan pembacaan M6 sebagai "kurangi bytes".

**[R19] Tail at Scale (2013).** Penyebab variabilitas: shared resources (CPU, disk), background daemon, queueing, power-state. Teknik tail-tolerant + jangan kejar utilisasi 100\%. Implikasi: (a) sisakan $1-r^*$ untuk OS/daemon; (b) lapor p50/p95 bukan mean saja; (c) monotonisitas pakai $\varepsilon=5\%$ karena noise run-to-run itu normal, bukan FAIL otomatis.

**[R20] TaxBreak (preprint 2026, pendukung).** Dekomposisi overhead inferensi: decode MoE persisten host-bound; dispatch CPU per kernel menumpuk di decode (10× lebih banyak launch vs prefill pada contoh kecil); single-thread CPU orde-satu untuk kasus host-bound. Dipakai hanya untuk menjustifikasi suku $\beta(c-1)$ dan anjuran ukur dispatch — bukan sebagai satu-satunya dasar gate.

## D.3 Batasan yang diakui

- F16 memakai Amdahl + overhead linear — model orde-satu. NUMA, frekuensi boost/thermal, SMT, dan afinitas thread tidak dimodelkan; ditangkap sebagai noise + catatan laporan, bukan rumus.
- [R20] adalah preprint baru — statusnya pendukung, bukan normatif. Bila konflik dengan [R14]–[R19], yang lama dan peer-review menang.
- Tidak ada angka core absolut di spec. Semua threshold F16 dalam rasio ($r$, $M$, $e_{T,core}$) agar berlaku di laptop kecil maupun workstation.
