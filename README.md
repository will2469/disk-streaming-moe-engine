# disk-streaming-moe-engine

Engine inferensi MoE streaming dari nol di **Mojo 1.0.0** (CPU, RAM 8 GB, NVMe SSD),
pola `kimi-k3-in-c`: checkpoint = index tensor, bobot di-stream dari disk per layer,
setiap komponen divalidasi layer-per-layer terhadap oracle PyTorch fp32.

- **Trial:** Qwen1.5-MoE-A2.7B-Chat (M0–M7) → **GDN** (M8) → **Port** Qwen3.6-35B-A3B (M9).
- Status: implementasi M0 berjalan (wave notes di `scratch/wave/`, catatan kerja gitignored).

## Prasyarat

| Tool        | Cara                                           | Catatan                                     |
| ----------- | ---------------------------------------------- | ------------------------------------------- |
| pixi        | [pixi.prefix.dev](https://pixi.prefix.dev)     | env Mojo + task runner                      |
| Mojo 1.0.0  | `pixi install` (ter-pin di `pixi.toml`)        | nightly dilarang untuk engine               |
| Rust stable | rustup                                         | `cargo fmt` + clippy cognit ≤15             |
| uv/uvx      | [docs.astral.sh/uv](https://docs.astral.sh/uv) | Python oracle/tooling (bukan Python sistem) |
| gcc         | sistem                                         | linker untuk kompilasi Mojo                 |

## Mulai cepat

```bash
pixi install && pixi run check        # Mojo 1.0.0?
pixi run test-m0                       # parser safetensors + F15 (11 tests)
pixi run fmt                           # mojo format src tests
cargo test --manifest-path tools/dismoen-tools/Cargo.toml
uv tool install pre-commit && pre-commit install
```

Commit selalu via wrapper (semantik lint-staged — format → stage → valid → done):

```bash
git add <file> && scripts/commit.sh "pesan"
```

Model (28,6 GB, di luar repo): lihat `scratch/download-shards.md` — metadata dulu,
shard nyicil; `~/models/`, revision ter-pin di `models.lock.json`.

## Layout

```text
src/                 # engine Mojo (safetensors.mojo, model.mojo, main.mojo)
tools/dismoen-tools/ # orchestration Rust (CLI, compare, benchmark, report)
tools/oracle/        # oracle PyTorch fp32 (Tier-1 normatif, bukan runtime)
tools/fixtures/      # generator fixture synthetic + golden
tests/               # TestSuite Mojo (*.mojo) + integrasi Rust
fixtures/            # fixture ter-commit (kecil; tanpa bobot asli)
reports/             # laporan benchmark run-id (ter-commit per gate)
docs/                # spec (index: docs/README.md)
scratch/             # catatan kerja agent, gitignored (waves, download)
```

## Kontrak (ringkas)

- Oracle-first: tanpa MATCH oracle tidak ada kode benar. Falsifikasi via kategori FAIL (§4.3).
- Teori di spec, angka device di laporan (prinsip 1-device, §5.1).
- Python/Candle: oracle & fixture offline; tidak ada di hot path decode.
- Detail penuh: [`docs/README.md`](docs/README.md).

## Lisensi

Apache-2.0 — lihat [LICENSE](LICENSE).
