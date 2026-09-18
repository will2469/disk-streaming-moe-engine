# M3 — Laporan Baseline Distribusi Routing F9 & Rekomendasi Pinning M7

> Dokumen analisis load-balance & baseline distribusi routing Milestone M3 (`scratch/wave/m3/m3-w5-f9.md`, `docs/milestones/M3-moe.md` § F9 Diagnostics).
> Model: `~/models/qwen1.5-moe-a2.7b-chat` (8 shard safetensors, revision pin `ec052fda178e241c7c443468d2fa1db6618996be`).
> Tool: `tools/bench/f9_diagnostics.py` (256 random inputs, seed 42, hidden_dim 2048, fp32).
> Timestamp: 2026-09-16.

---

## 1. Ringkasan Eksekutif

Pengukuran diagnostik F9 dilakukan terhadap 256 input acak (256 token) untuk mengkarakterisasi skew routing pada model Qwen1.5-MoE-A2.7B-Chat (60 routed experts, top-4 tanpa renormalisasi, shared expert dengan sigmoid gate terpisah).

Hasil utama:
- **Load-balance loss ($\mathcal{L}_{lb}$)** berada di kisaran $1{,}0003 - 1{,}0014$ (sangat dekat dengan lower bound teoritis $1{,}0000$ untuk routing seimbang).
- **Coefficient of Variation ($CV = \sigma_f / \mu_f$)** berada di rentang $0{,}2598 - 0{,}3878$, mengindikasikan bahwa distribusi pemilihan expert memiliki dispersi moderat dengan subset expert terpilih hingga $3\times$ lebih sering daripada expert dingin.
- **Integritas Invariant 100%**:
  - Tepat 4 expert terpilih per token ($K=4$).
  - Seluruh expert ID berada dalam interval valid $[0, 59]$.
  - $\sum_{k=0}^3 p_k^{(t)} \le 1{,}0$ terpenuhi pada semua token (top-4 unrenormalized).
  - Total frekuensi $\sum_{i=0}^{59} f_i = 1{,}000000$.
  - Kesamaan SET expert antara Mojo engine (`dismoen`) dan PyTorch Oracle mencapai $100\%$ tanpa ada perbedaan seleksi (0 flip).

---

## 2. Definisi Matematis F9

Sesuai spesifikasi `docs/milestones/M3-moe.md` § F9 Diagnostics dan `docs/02-math-models.md`:

$$f_i = \frac{1}{K \cdot T}\sum_{t=1}^T \mathbb{1}[i \in \mathcal{A}_t], \qquad P_i = \frac{1}{T}\sum_{t=1}^T p_i^{(t)}$$

$$\mathcal{L}_{lb} = N_e \sum_{i=1}^{N_e} f_i P_i, \qquad CV = \frac{\sigma_f}{\mu_f}$$

Keterangan:
- $N_e = 60$: jumlah total routed experts.
- $K = 4$: jumlah expert terpilih per token.
- $T = 256$: jumlah sample input.
- $\mathcal{A}_t$: himpunan indeks $K$ expert terpilih pada token ke-$t$.
- $f_i$: frekuensi seleksi relatif expert $i$ terhadap total seleksi ($K \cdot T = 1.024$).
- $P_i$: probabilitas rata-rata router untuk expert $i$ dari distribusi softmax penuh.
- $\mu_f = 1 / N_e \approx 0{,}016667$: frekuensi rata-rata ideal seragam.

---

## 3. Data Pengukuran Diagnostik Empiris ($T=256$, Seed=42)

Pengujian dievaluasi pada 3 layer representatif: Layer 0 (awal), Layer 12 (tengah), dan Layer 23 (akhir).

| Metrik | Layer 0 | Layer 12 | Layer 23 | Keterangan |
| :--- | :---: | :---: | :---: | :--- |
| **Load-Balance Loss ($\mathcal{L}_{lb}$)** | **$1{,}001355$** | **$1{,}000315$** | **$1{,}000324$** | Ideal = 1.0, lower is better |
| **Coefficient of Variation ($CV$)** | **$0{,}387763$** | **$0{,}259817$** | **$0{,}372714$** | Dispersi frekuensi |
| **Mean Frequency ($\mu_f$)** | $0{,}016667$ | $0{,}016667$ | $0{,}016667$ | $1/60$ |
| **Std Dev Frequency ($\sigma_f$)** | $0{,}006463$ | $0{,}004330$ | $0{,}006212$ | Standar deviasi $f_i$ |
| **Max Selection Count** | 31 ($f_{59} = 3{,}03\%$) | 27 ($f_{0} = 2{,}64\%$) | 32 ($f_{7} = 3{,}13\%$) | Hot expert teratas |
| **Min Selection Count** | 4 ($f_{5} = 0{,}39\%$) | 10 ($f_{20} = 0{,}98\%$) | 8 ($f_{5} = 0{,}78\%$) | Cold expert terendah |
| **Top/Bottom Ratio** | $7{,}75\times$ | $2{,}70\times$ | $4{,}00\times$ | Skew beban aktivasi |

### 3.1 Ranking Hot Experts (Kandidat Pinning Cache M7)

Top 8 expert dengan frekuensi seleksi tertinggi per layer:

- **Layer 0**: `[59, 36, 3, 10, 17, 28, 34, 38]`
  - Expert 59: 31 hit ($f = 3{,}03\%$, $P = 0{,}016844$)
  - Expert 36: 29 hit ($f = 2{,}83\%$, $P = 0{,}016802$)
  - Expert 3: 27 hit ($f = 2{,}64\%$, $P = 0{,}016776$)
  - Expert 10: 26 hit ($f = 2{,}54\%$, $P = 0{,}016763$)

- **Layer 12**: `[0, 16, 43, 2, 17, 19, 3, 6]`
  - Expert 0: 27 hit ($f = 2{,}64\%$, $P = 0{,}016690$)
  - Expert 16: 27 hit ($f = 2{,}64\%$, $P = 0{,}016662$)
  - Expert 43: 27 hit ($f = 2{,}64\%$, $P = 0{,}016730$)
  - Expert 2: 24 hit ($f = 2{,}34\%$, $P = 0{,}016669$)

- **Layer 23**: `[7, 41, 15, 30, 17, 19, 29, 22]`
  - Expert 7: 32 hit ($f = 3{,}13\%$, $P = 0{,}016731$)
  - Expert 41: 31 hit ($f = 3{,}03\%$, $P = 0{,}016766$)
  - Expert 15: 30 hit ($f = 2{,}93\%$, $P = 0{,}016727$)
  - Expert 30: 29 hit ($f = 2{,}83\%$, $P = 0{,}016689$)

*Observasi Konsistensi Antar-Layer*: Expert 17 konsisten masuk ke jajaran top 8 di seluruh layer (L0: rank 5, L12: rank 5, L23: rank 5), menandakan adanya basis aktivasi intrinsik yang lebih sering memicu expert tersebut.

### 3.2 Ranking Cold Experts

Bottom 8 expert dengan frekuensi seleksi terendah per layer:

- **Layer 0**: `[1, 4, 5, 6, 33, 37, 44, 45]` (Expert 5 paling jarang terpilih: 4 hit, $f = 0{,}39\%$).
- **Layer 12**: `[10, 20, 22, 47, 48, 49, 58, 59]` (Expert 20: 10 hit, $f = 0{,}98\%$).
- **Layer 23**: `[5, 6, 18, 21, 38, 39, 49, 58]` (Expert 5, 6, 18: 8 hit, $f = 0{,}78\%$).

---

## 4. Implikasi Desain untuk Milestone M7 (Prefetch & Disk Streaming)

Hasil baseline F9 memberikan panduan penting untuk perancangan storage streaming dan memori caching di Milestone M7:

### 4.1 Koreksi Model Memori & IO Bandwidth Formula F5
Pada spesifikasi awal `docs/02-math-models.md` Formula F5, I/O bandwidth MoE sering memodelkan probabilitas akses seragam $\rho = K / N_e = 4 / 60 \approx 0{,}0667$.
Dengan adanya routing skew empiris:
- Frekuensi aktual expert bervariasi antara $0{,}0039$ hingga $0{,}0313$.
- Formula bandwidth efektif dikoreksi menjadi:
  $$\rho_{\text{eff}} = \sum_{i=1}^{N_e} f_i \cdot \rho_i$$
- Jika expert dipetakan ke shard atau blok disk terpisah, estimasi I/O demand per token harus memperhitungkan clustering hot experts untuk menghindari antrean I/O bottleneck pada shard tertentu.

### 4.2 Kebijakan Cache Pinning M7
1. **Tiered Residency**:
   - Shared expert wajib 100% resident di RAM (selalu aktif di setiap token).
   - Hot experts per layer (misalnya top 4 per layer: 4 × 3 tensor per layer) dapat di-pin dalam page cache atau O_DIRECT prefetch buffer jika memori mencukupi.
2. **LRU Eviction Priority**:
   - Cold experts (`f_i < 0{,}01`) langsung di-evict segera setelah forward pass token selesai untuk menjaga peak memory tetap berada di batas cgroup $M_{\text{gate}} \le 6\text{ GiB}$.
   - Hot experts diberikan bobot retention lebih tinggi dalam kebijakan LRU/ARC cache M7.

---

## 5. Metadata File Hasil Analisis
- Data mentah JSON: `reports/2026-09-16/f9_routing_distribution.json`
- Script generator: `tools/bench/f9_diagnostics.py`
- Verifikasi reproduktifitas:
  ```bash
  uv run python tools/bench/f9_diagnostics.py --samples 256 --seed 42 --model-dir ~/models/qwen1.5-moe-a2.7b-chat
  ```
