---
name: mojo-1-0
description: "Writes Mojo 1.0.0 engine code for disk-streaming-moe-engine: CPU-only inference, oracle-first, deterministic fp32. Auto-triggers when writing, editing, or reviewing *.mojo files, adding Mojo TestSuite tests, debugging Mojo compiler errors, or pinning the Mojo toolchain via pixi. Keywords: mojo, def, TestSuite, pixi, struct, SIMD, pread, mojo format."
compatibility: "Requires Mojo 1.0.0 stable via pixi, bash, git; pairs with official mojo-syntax skill"
metadata:
  version: "1.0.0"
  author: "will2469"
  license: "Apache-2.0"
  citations:
    - "Mojo v1.0.0 changelog — https://mojolang.org/releases/v1.0.0"
    - "Mojo Manual 1.0.0 — https://docs.modular.com/mojo/manual/"
    - "modular/skills (mojo-syntax) — https://github.com/modular/skills"
---

# Skill: mojo-1-0

# Mojo 1.0.0 Engine Skill (disk-streaming-moe-engine)

> **Core Thesis:** Mojo pra-1.0 berubah tiap rilis sehingga pengetahuan bawaan model pasti basi; Mojo 1.0.0 stabil tetapi memutus sintaks lama secara mekanis. Skill ini memaksa agen menulis Mojo 1.0.0 idiomatis untuk engine inferensi CPU — deterministik, tanpa Python di hot path, tanpa API GPU/MAX — dan memverifikasi via `TestSuite` + oracle Rust, bukan via kesan.

---

## 1. Kapan skill ini AKTIF (trigger)

- Menulis / mengedit / me-review file `*.mojo` apa pun di repo ini.
- Menambah / memperbaiki test Mojo (`test_*.mojo`, `TestSuite`).
- Debugging error compiler Mojo atau migrasi sintaks lama.
- Menyiapkan / mem-pin toolchain Mojo via pixi.
- Trigger phrases: `"tulis kernel mojo"`, `"mojo test"`, `"mojo error"`, `"pixi mojo"`, `"struct mojo"`, `"SIMD"`.

```
 WRITE *.mojo ──► FORMAT ──► TEST (TestSuite) ──► ORACLE-COMPARE (Rust) ──► GATE
      ▲               │               │                      │
      └───────────────┴── fix-it ─────┴── FAIL kategorikan ───┘
```

---

## 2. Prasyarat (jangan mulai tanpa ini)

1. Toolchain **Mojo 1.0.0 stable** via pixi, di-pin di `pixi.toml` (K6). Nightly (1.1.0.dev) **dilarang** untuk kode engine.
2. Skill resmi **`mojo-syntax`** terinstal (`npx skills add modular/skills --skill mojo-syntax`) — itu lapisan koreksi sintaks; skill ini lapisan *proyek*, tidak menduplikasinya.
3. Docs acuan = versi **1.0.0** (`docs.modular.com` pemilih versi 1.0.0 / `mojolang.org/releases/v1.0.0`). Docs nightly bisa berbeda — jangan dikutip untuk kode engine.

---

## 3. Invariant proyek (non-negotiable)

| # | Invariant | Alasan |
|---|---|---|
| I-1 | `def` untuk semua fungsi; `var` untuk setiap deklarasi baru; import std berprefix `std.` | 1.0.0: `fn` = parse error, bare-assign = compile error, bare std = tidak resolve |
| I-2 | Fungsi yang bisa raise wajib `raises`; data path tanpa `panic`/`unwrap` (error → stderr terstruktur) | Kontrak C1 + quality §5.3 |
| I-3 | Komputasi fp32; perbandingan float pakai `assert_almost_equal` (tol) — bukan `assert_equal` | Verdict F10 butuh toleransi, bukan equality bit |
| I-4 | **CPU-only**: dilarang API paket `max`, `DeviceContext` GPU, `layout` akselerator | GPU out of scope (§1.2, ADR) |
| I-5 | **Tanpa Python interop di engine** (`std.python` dilarang di `src/`); Python hanya di oracle/fixtures | Boundary runtime (§2.2.1) |
| I-6 | Test = fungsi `test_*` + `TestSuite.discover_tests[__functions_in_module()]().run()` via `mojo run`; `mojo test` **tidak ada** di 1.0 | Testing 1.0.0 resmi |
| I-7 | Buffer I/O sejajar block size (`Layout`/`alloc` + `dealloc`), loop short-`pread`, tanpa alokasi dari angka file mentah | F15 + SEC-2/SEC-3 |
| I-8 | Verdict numerik hanya sah `threads=1`; sweep performa terpisah | Determinisme §4.6 |

---

## 4. Workflow eksekusi

**Step 1 — Tulis minimal.** Satu kernel / satu file per langkah. `comptime assert` hanya di dalam badan fungsi. Parameter struct selalu `Self.X`.

**Step 2 — Format.** `mojo format <file>` (tidak ada `--check` di 1.0; CI yang diff). Pre-commit hook memformat otomatis.

**Step 3 — Test lokal.** `mojo run -I src <test_file>.mojo [--only <nama>]`. Filter `--skip` untuk isolasi; `skip()` programatik hanya untuk test broken yang tercatat.

**Step 4 — Bandingkan ke oracle.** Emit `.bin` → Rust `compare` → verdict F10 + kategori FAIL (§4.3). FAIL `router-selection` = root-cause, tidak pernah naikkan threshold.

**Step 5 — Gate.** Hijau + laporan run-id ter-commit, atau tidak dihitung (§5.1).

---

## 5. Anti-pattern (DILARANG)

- Sintaks pra-1.0: `fn`, `let`, bare-assign, `alias`, `@parameter if/for`, `inout`/`owned`/`borrowed`, `UnsafePointer`, `from memory import`, `mojo test`. (Detail tabel: skill resmi `mojo-syntax`.)
- `List[Int](1,2,3)` (pakai literal `[1,2,3]`); `s[0]` / slice String (pakai `s[byte=i]`); `lst[-1]`; konversi numerik implisit.
- API GPU/MAX, `async def`, `@__parameter` pada closure, `match` statement.
- Menebak versi docs: bila ragu antara nightly vs 1.0.0 → buka `references/mojo-1-0-deltas.md` atau changelog resmi, jangan mengarang flag/API.

---

## 6. Checklist verifikasi

- [ ] `pixi.toml` pin `mojo` 1.0.0 (bukan nightly); `mojo --version` cocok.
- [ ] `mojo format` bersih; `mojo run` test hijau; verdict F10 MATCH via Rust compare.
- [ ] Fungsi kecil (complexity ≤ 15): belum ada linter Mojo → review manual, pecah fungsi bila bercabang banyak.
- [ ] Tidak ada `fn|let|alias|@parameter|UnsafePointer|mojo test|std.python` di `src/` (grep).
- [ ] Tidak ada API `max`/GPU di `src/` (grep).
- [ ] SKILL.md ini < 500 baris; detail tebal ada di `references/`.

---

## 7. References

- `references/mojo-1-0-deltas.md` — delta 1.0.0 yang relevan proyek (stability scope, breaking→fix-it, testing, CLI, channel).
- Changelog resmi: https://mojolang.org/releases/v1.0.0
- Manual 1.0.0: https://docs.modular.com/mojo/manual/ · Testing: https://docs.modular.com/mojo/tools/testing/ · CLI: https://docs.modular.com/mojo/cli/
- Stdlib: https://docs.modular.com/mojo/std/ · Cheat sheet: rilis 1.0.0 · Contoh testing: https://github.com/modular/modular/tree/mojo/v1.0.0/mojo/examples/testing
- Skill resmi: https://github.com/modular/skills (`mojo-syntax`, `new-modular-project`)
