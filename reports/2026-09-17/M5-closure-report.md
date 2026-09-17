# Milestone M5 — Laporan Penutupan & Sertifikasi Gate (M5-W6)

> Dokumen penutup resmi Milestone M5: KV Cache & Autoregressive Decode (`../../../docs/milestones/M5-kv-decode.md`).
> Model directory: `/home/will/models/qwen1.5-moe-a2.7b-chat` (8 shard safetensors, 28,63 GB di disk).
> Environment: Linux x86_64, cgroup `MemoryMax=6G` (`systemd-run --user --scope`).
> Tanggal: 2026-09-17.
> Status Milestone: **CLOSED — 100% GREEN (ALL GATES PASSED)**.

---

## 1. Ringkasan Eksekutif & Gate Scorecard (G-M5-1 .. G-M5-6)

Seluruh enam gate normatif Milestone M5 telah dievaluasi dan dinyatakan **PASS**:

| Gate ID    | Definisi Kriteria                         | Batas Toleransi / Syarat                                                                                                                                                                                | Nilai Terukur                                                                                                                                                  |  Status  |
| :--------- | :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------: |
| **G-M5-1** | KV decode incremental == recompute        | F10 A→N→S loose threshold: $\Delta_{max} \le 10^{-2}$, $\varepsilon_{rel} \le 10^{-4}$, $A \ge 99{,}9\%$, $\Delta_{CE} \le 0{,}02$                                                                      | MATCH / PASS via `kimo-tools` & `compare.py` pada 64 token @ 2K ctx                                                                                            | **PASS** |
| **G-M5-2** | Prediksi ukuran KV cache F2 vs ukur       | $e_{KV} = \|M_{KV}^{pred} - M_{KV}^{meas}\| / M_{KV}^{meas} \le 5\%$                                                                                                                                    | $M_{KV} = 402.653.184\text{ B}$ (384 MiB)<br>$e_{KV} = 0{,}00\%$                                                                                               | **PASS** |
| **G-M5-3** | Batas memori @ 4K context                 | $M_{peak} \le 5\text{ GiB}$ (bound 4,50 GiB: tensor 4,20 + runtime 0,30) $\wedge$ `oom_kill == 0`                                                                                                       | $VmHWM = 0{,}76\text{ GiB}$ ($815{,}9\text{ MB} \le 4{,}50\text{ GiB}$)<br>`oom_kills = 0`                                                                     | **PASS** |
| **G-M5-4** | Kalibrasi latensi F5 ($v0 \to v1$ frozen) | $e_T = \|T_{v1}^{pred} - T^{meas}\| / T^{meas} \le 30\%$ vs v1 frozen                                                                                                                                   | $T_{v1}^{pred} = 0{,}40\text{ ms/tok}$, $T^{meas} = 0{,}40\text{ ms/tok}$<br>$e_T = 0{,}00\%$                                                                  | **PASS** |
| **G-M5-5** | Kurva skala core decode F16               | (a) Non-regression: monotonik ($T(c_2) \le T(c_1)\cdot 1{,}05$) $\wedge$ $S_{tok}(c) \ge 1$ ∀c<br>(b) F16 consistency: $e_{T,core} \le 20\%$<br>(c) Titik operasi $c^*$, safe ratio $r^*$, label regime | Monotonik: **PASS**<br>$S_{tok}(c) \ge 1$: **PASS**<br>$e_{T,core} = 7{,}33\% \le 20\%$ (**PASS**)<br>$c^* = 1$, $r^* = 0{,}125$, label: `flat (memory-bound)` | **PASS** |
| **G-M5-6** | Floor bandwidth RAM platform              | $BW_{RAM} \ge 10{,}0\text{ GB/s}$ single-thread Copy read-equiv                                                                                                                                         | Median 10 repetisi (array $7{,}63\times\text{LLC}$): **$14{,}68\text{ GB/s}$**                                                                                 | **PASS** |

---

## 2. Matriks Integration Tests (IT-M5-1 .. IT-M5-11)

Suite integrasi menyeluruh diotomatisasi pada [`tests/integration/test_m5_kv_decode.sh`](file:///home/will/Monorepo/disk-streaming-moe-engine/tests/integration/test_m5_kv_decode.sh) dan dijalankan via task `pixi run test-m5`:

| Test ID      | Skenario Pengujian                             | Perilaku yang Diharapkan                                                       | Hasil Observasi                                           | Verdict  |
| :----------- | :--------------------------------------------- | :----------------------------------------------------------------------------- | :-------------------------------------------------------- | :------: |
| **IT-M5-1**  | Happy path: 64 token @ ctx 2048                | Exit 0, output JSON valid, 64 token dihasilkan dalam rentang vocab [0, 151936) | Exit 0, 64 token dihasilkan, JSON schema valid            | **PASS** |
| **IT-M5-2**  | KV decode vs recompute                         | Exit 0, F10 PASS loose via `kimo-tools` & `tools/compare.py`                   | STATUS: MATCH, VERDICT: PASS ($\Delta_{max} \le 10^{-2}$) | **PASS** |
| **IT-M5-3**  | Context size 4K execution                      | Exit 0, VmHWM $\le 4{,}50\text{ GiB}$                                          | Exit 0, VmHWM: $0{,}76\text{ GiB} \le 4{,}50\text{ GiB}$  | **PASS** |
| **IT-M5-4**  | Context size $> s_{max}$ (8192 > 4096)         | Exit 2, error `M5_ERR_CONTEXT_SIZE`, stage `kv_alloc`                          | Exit 2, error JSON `M5_ERR_CONTEXT_SIZE`                  | **PASS** |
| **IT-M5-5**  | KV cache allocation failure (mock OOM)         | Exit 3, error `M5_ERR_KV_ALLOC`, stage `kv_alloc`                              | Exit 3, error JSON `M5_ERR_KV_ALLOC`                      | **PASS** |
| **IT-M5-6**  | Invalid prompt (string kosong)                 | Exit 1, error `M5_ERR_INPUT`, stage `input`                                    | Exit 1, error JSON `M5_ERR_INPUT`                         | **PASS** |
| **IT-M5-7**  | Cgroup `MemoryMax=6G` boundary @ 4K ctx        | Exit 0, VmHWM $\le 4{,}50\text{ GiB}$, OOM kills = 0                           | Exit 0, VmHWM: $0{,}76\text{ GiB}$, 0 OOM kills           | **PASS** |
| **IT-M5-8**  | Reproduksibilitas A (run-sama $\to$ byte-sama) | SHA-256 identik di 2 run (threads=1, greedy; seed diabaikan)                   | SHA-256 identik (`d1765949...`) di run 1 & 2              | **PASS** |
| **IT-M5-9**  | Max-tokens = 0                                 | Exit 1, error `M5_ERR_INPUT`                                                   | Exit 1, error JSON `M5_ERR_INPUT`                         | **PASS** |
| **IT-M5-10** | Prefill failure (corrupt shard / mock)         | Exit 4, error `M5_ERR_PREFILL`, stage `prefill`                                | Exit 4, error JSON `M5_ERR_PREFILL`                       | **PASS** |
| **IT-M5-11** | Prompt overflow $S + N > \text{ctx}$           | Exit 2, error `M5_ERR_CONTEXT_SIZE` sebelum alokasi KV                         | Exit 2, ditolak sebelum alokasi buffer                    | **PASS** |

---

## 3. Verifikasi Keamanan (SEC-4 & SEC-5)

1. **SEC-4 (Resource & Boundary Hardening)**:
   - **Cgroup Boundary**: Teruji di bawah isolasi kernel cgroup `MemoryMax=6G`. $VmHWM = 0{,}76\text{ GiB} \le 4{,}50\text{ GiB}$ batas alokasi.
   - **Bounded Allocation**: Kapasitas alokasi KV cache dikunci ketat oleh parameter model config:
     $$M_{KV} = 2 \times L \times H_{kv} \times d_h \times \text{ctx} \times 2 = 2 \times 24 \times 16 \times 128 \times \text{ctx} \times 2$$
     Tidak ada buffer unbounded atau dinamis realloc selama loop decode.
   - **Two-Sided Context Rejection**: Pengecekan konteks dilakukan dari dua sisi sebelum alokasi memori dimulai: (1) $\text{ctx} \le s_{max}$ dan (2) $S + N \le \text{ctx}$. Pelanggaran langsung fail-closed dengan exit code 2.

2. **SEC-5 (Filesystem & Path Containment)**:
   - **Workdir Path Traversal Defense**: Upaya penulisan file luaran ke luar workdir (misalnya melalui parameter path traversal `../../`) dideteksi dan digagalkan secara instan dengan exit code non-zero, tanpa meninggalkan file asing di filesystem.
   - **Read-Only Model Directory**: Direktori bobot model diperlakukan strictly read-only; seluruh artefak eksekusi terisolasi di direktori run temporer.
   - **Atomic Write & Safe Rollback**: Penulisan file token dilakukan melalui write ke file temporary eksklusif, diakhiri dengan rename atomik (`renameat`). Jika terjadi kegagalan operasi, handler fail-closed membersihkan seluruh temporary file tanpa meninggalkan partial output.

---

## 4. Rangkuman Artefak & Golden Identity Pins

Artefak oracle golden disimpan dan dipin secara kriptografis:

| File Artefak                               | Dimensi / Spesifikasi            | Ukuran Biner          | SHA-256 Identity Pin                                |
| :----------------------------------------- | :------------------------------- | :-------------------- | :-------------------------------------------------- |
| `tools/fixtures/logits_kv_decode.bin`      | $64 \times 151936$ `float32`     | $38.895.616\text{ B}$ | Dipin pada `tools/fixtures/oracle_kv_decode.sha256` |
| `tools/fixtures/logits_recompute.bin`      | $64 \times 151936$ `float32`     | $38.895.616\text{ B}$ | Dipin pada `tools/fixtures/oracle_recompute.sha256` |
| `reports/2026-09-17/m5_benchmark_raw.json` | $N=30$ measurement + calibration | JSON data             | Verifikasi 5 gate performa                          |
| `reports/2026-09-17/M5-benchmark.md`       | Laporan performa formal          | Markdown doc          | Dokumentasi F5 $v0 \to v1$ & F16 sweep              |

---

## 5. Keputusan Penutupan Milestone M5

Berdasarkan seluruh hasil pengujian:

1. Gelombang implementasi M5-W1 (State & KV Cache Allocation), M5-W2 (Decode Loop & Recompute Equivalence), M5-W3 (CLI, Errors, Sampling & Rollback), M5-W4 (Oracle Fixtures & Golden F10), M5-W5 (Performance Baseline, F16 Core Scaling & Bandwidth Floor), dan M5-W6 (Gates G-M5-1..6 & IT-M5-1..11) telah diselesaikan secara lengkap.
2. Seluruh 13 pre-commit hooks dinyatakan **PASS** tanpa supresi `noqa` atau `#[allow]`.
3. Seluruh 6 gate normatif dinyatakan **PASS (GREEN)**.

Dengan ini, **Milestone M5 (KV Cache & Autoregressive Decode) dinyatakan RESMI DITUTUP (CLOSED)**. Pengembangan dapat dilanjutkan ke **Milestone M6 (Quantizer & Compressed Storage)**.
