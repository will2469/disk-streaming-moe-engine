# Delta Mojo 1.0.0 yang Relevan Proyek

> Disarikan dari changelog resmi https://mojolang.org/releases/v1.0.0 (+ sub-halaman v1.0.0b1/b2).
> Aturan: bila ragu, changelog menang atas ingatan. Docs nightly bisa berbeda.

## 1. Stability policy (baru di 1.0)

- Stdlib API yang ditandai **stable** tidak akan dihapus/diubah merusak kompatibilitas sumber.
- Set awal sengaja kecil: trait `Deinitable, Movable, Copyable, ImplicitlyCopyable` (utuh); `Array` (dulu `InlineArray`), `List`, `Span`; `String`; `Bool`, `Optional`.
- Selama 1.x perubahan seharusnya aditif (ala C++). Breaking change tetap mungkin tapi dikelola hati-hati + fix-it compiler.

## 2. Breaking yang treating kita langsung (semua ada fix-it / deprecated alias)

| Dulu | 1.0.0 | Catatan |
|---|---|---|
| `fn` | `def` (semantik non-raising seperti `fn` lama) | `fn` = hard parse error |
| bare-assign `x = 1` | `var x = 1` | compile error; predeclare + tipe bila assign di branch |
| `alias` / `@parameter if|for` | `comptime` / `comptime if|for` | `comptime assert` hanya di badan fungsi |
| `let`, `borrowed`, `read`, `inout`, `owned` | `var`, `imm` (default, jarang ditulis), `mut`, `var` (konvensi arg), `out self` di `__init__` | `var`/`ref` hard keyword, tidak bisa jadi identifier |
| `@value` | `@fieldwise_init` + conformance eksplisit | |
| `UnsafePointer` / `Mut/ImmUnsafePointer` | `Pointer` / `MutPointer` / `ImmPointer` (non-null by design) | `UnsafePointer` masih kompilasi + warning; sisanya hard error |
| `alloc[T](n)` / `p.free()` / `p[i]` / `p.load/store` | `alloc(Layout[T](count=n))` / `dealloc(allocation^)` / `p[unsafe_offset=i]` / `p.unsafe_load/store` | `alloc` kembalikan `Allocation` linear (wajib dispose tiap path) |
| `NDBuffer` | `TileTensor` | GPU-side; kita CPU-only, sekadar tahu |
| import `from memory/algorithm/sys/os/pathlib import` | prefix `std.` penuh | `std` reserved sebagai identifier modul |
| `mojo test` | `mojo run` + `TestSuite` | subcommand dihapus |
| `.mojopkg` | `.mojoc` (`mojo precompile` ganti `mojo package`) | relevan saat packaging nanti |

## 3. Testing 1.0 (pola baku)

- `from std.testing import assert_equal, assert_almost_equal, assert_true, assert_raises, TestSuite`.
- Test = fungsi `test_*` tanpa arg, return `None`, gagal = raise. Float → `assert_almost_equal(..., atol|rtol)`.
- Runner: `TestSuite.discover_tests[__functions_in_module()]().run()` di `main()`; eksekusi `mojo run <file>` (atau `-I src`).
- Filter CLI: `--skip`, `--only`, `--skip-all`; programatik `suite.skip[name]()` (+ transfer `^` di `.run()`).

## 4. CLI yang dipakai proyek

- `mojo format <sources...>` (tidak ada `--check`; opsi: `-l/--line-length` default 80, `-q`). CI yang diff untuk mode check.
- `mojo run [-I src] <file> [--only t]`, `mojo build`, `mojo --version`.
- Contoh testing resmi: `mojo/examples/testing` di tag `mojo/v1.0.0`.

## 5. Channel & versi (anti-mismatch)

- Stable: `mojo` = **1.0.0**, `max` = **26.5** (penomoran beda, itu normal).
- Nightly: `mojo` 1.1.0.dev / `max` 26.6.0.dev — dilarang untuk engine.
- pixi stable: channel `https://conda.modular.com/max/` + `conda-forge`, lalu `pixi add mojo`. Butuh linker C (`gcc`) di OS.
- Compiler + toolchain open source Apache-2.0 dirilis bertahap 2026 (stdlib sudah; compiler dijadwalkan) — pin pixi tetap wajib (K6).

## 6. Batas Mojo vs MAX (1.0)

- API akselerator pindah ke paket `max`; `layout` ikut MAX (kecuali bagian yang tetap di mojo — cek docs versi pin).
- GPU programming pindah ke docs MAX. Proyek ini CPU-only → abaikan seluruhnya (I-4).
