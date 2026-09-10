# Lampiran B.1 — Kontrak Implementasi Tooling

> Bagian dari `disk-streaming-moe-engine`. Index: `../README.md`.
> Boundary: `../01-architecture.md` §2.2.1.

Target repository:

```text
tools/
├── kimo-tools/                 # Rust
│   ├── src/main.rs
│   ├── check_index.rs
│   ├── compare.rs
│   ├── benchmark.rs
│   └── report.rs
└── oracle/                     # Python + PyTorch
    ├── oracle_head.py
    ├── oracle_layer.py
    ├── oracle_full.py
    └── fixtures.py
```

Alur validasi normatif:

```text
Python/PyTorch
  └── generate reference *.bin
             │
             ▼
        Rust kimo-tools
          ├── launch Mojo
          ├── collect output
          ├── compare F10
          └── emit verdict/report
             │
             ▼
        PASS / FAIL gate
```

Hot path decode berjalan di Mojo/Rust; Python/PyTorch berjalan di tahap persiapan/verifikasi offline. Pembagian ini bersifat arsitektural agar setiap bagian mudah diuji.

- Rust = CLI/orchestration, index/metadata, compare, benchmark, report.
- Mojo = tensor loading, streaming, KV, quant/dequant, attention, MoE, GDN, forward/decode.
- Python/PyTorch = oracle + fixture/golden generation di tahap offline.
