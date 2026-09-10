# 06 — Risiko & Pertanyaan Terbuka

> Bagian dari `disk-streaming-moe-engine`. Index: `README.md`.

| ID | Risiko | Dampak | Mitigasi |
|---|---|---|---|
| R1 | API Mojo berubah cepat antar versi | build rusak | pin versi pixi.toml; README §5 sudah memetakan titik penyesuaian; fallback C (D1) |
| R2 | Flip seleksi router akibat numerik | FAIL M3 berulang | root-cause wajib; threshold tidak boleh dinaikkan (`03-testing.md` §4.3) |
| R3 | Ruang disk: 28,6 GB model + golden bins + reports | unduhan/CI gagal | cek ≥ 1,5× (SEC-4); golden bins disimpan ter-pisah |
| R4 | Noise fp32 (urutan penjumlahan) membingungkan verdict | waktu terbuang | kategori FAIL `03-testing.md` §4.3 + playbook README §4; multi-metrik F10 |
| R5 | $W_{res}$ F32 **2,318 GiB** menyempitkan workspace di 8 GB | gate $M_{peak}$ | opsi: embedding/lm_head tetap BF16 di disk + dequant on-the-fly (keputusan M4+) |
| R6 | GDN tanpa model hybrid kecil publik sebagai oracle | M8 macet | oracle = naive loop dari kode referensi paper; model kecil opsional bila muncul |
| R7 | Implementasi/config Qwen3.6-35B-A3B yang dipakai lokal berbeda dari checkpoint/ref yang dipin | asumsi M9 salah | pin revision/checkpoint; fakta arsitektur di `01-architecture.md` §2.7 berasal dari model card resmi; angka performa tetap TBM hingga diukur |
| R8 | llama.cpp (quant) dianggap "benar" saat debugging | salah duplikasi bug orang lain | kebijakan `03-testing.md` §4.1: oracle fp32 satu-satunya ground truth |

Keterkaitan milestone:

- R2 → `milestones/M3-moe.md` (G-M3-2)
- R4 → `milestones/M4-full-forward.md` (verdict loose)
- R5 → `milestones/M1-head-path.md`, `milestones/M4-full-forward.md` (opsi BF16 resident)
- R6 → `milestones/M8-gdn.md`
- R7 → `milestones/M9-port.md`
