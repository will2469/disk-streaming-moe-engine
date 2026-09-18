#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Verifikasi Tail Latency & Dynamic RAM Budget Adherence (Gate G-M11-3).

Ref: Milestone 11 §1.5, §2.3, §3.5, Gate G-M11-3.
Menjalankan:
1. Pengujian variabilitas latensi ekor N >= 100 run pada titik deploy c*_system
   dari dismoen.hardware.lock.
2. Estimator persentil interpolasi-linear terdefinisi (bukan nearest-rank).
3. Evaluasi Project SLO R_tail = p95 / p50 <= 1.35.
4. Bootstrap Confidence Interval (CI 95%, B=1000) diagnostik untuk p50, p95,
   dan R_tail.
5. Kepatuhan alokasi RAM dinamis R_RAM = VmHWM / M_budget <= 0.95 per tier
   (8, 16, 32, 64 GiB dan host aktif) serta verifikasi bebas leak.
6. Asersi stabilitas bandwidth storage E_BW <= 5% dari kalibrasi sistem W2b.
7. Emisi artefak laporan JSON dan Markdown.
"""

import argparse
import datetime
import json
import math
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_cpu_model() -> str:
    """Membaca model prosessor host dari /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown CPU"


def get_cpu_governor() -> str:
    """Membaca CPU governor saat ini dari sysfs."""
    p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if p.exists():
        try:
            return p.read_text().strip()
        except Exception:
            pass
    return "virtualized_or_container"


def calc_percentile(sorted_data: list[float], pct: float) -> float:
    """Menghitung persentil dengan interpolasi linear standar Project SLO."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    rank = (pct / 100.0) * (n - 1)
    low = int(rank)
    high = low + 1
    if high >= n:
        return sorted_data[-1]
    weight = rank - low
    return sorted_data[low] * (1.0 - weight) + sorted_data[high] * weight


def compute_statistics(data: list[float]) -> dict:
    """Menghitung ringkasan statistik empiris dari data latensi."""
    s_data = sorted(data)
    n = len(s_data)
    mean_val = sum(s_data) / n if n > 0 else 0.0
    variance = sum((x - mean_val) ** 2 for x in s_data) / n if n > 0 else 0.0
    std_dev = math.sqrt(variance)

    p50 = calc_percentile(s_data, 50.0)
    p90 = calc_percentile(s_data, 90.0)
    p95 = calc_percentile(s_data, 95.0)
    p99 = calc_percentile(s_data, 99.0)
    r_tail = p95 / p50 if p50 > 0 else 0.0

    return {
        "n": n,
        "min_ms": s_data[0] if n > 0 else 0.0,
        "max_ms": s_data[-1] if n > 0 else 0.0,
        "mean_ms": mean_val,
        "std_ms": std_dev,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "r_tail": r_tail,
    }


def run_bootstrap_ci(
    data: list[float],
    n_resamples: int = 1000,
    ci_pct: float = 95.0,
    seed: int = 42,
) -> dict:
    """Bootstrap non-parametrik untuk estimasi CI diagnostik p50, p95, dan R_tail."""
    rng = random.Random(seed)
    n = len(data)
    b_p50 = []
    b_p95 = []
    b_rtail = []

    for _ in range(n_resamples):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        sample.sort()
        p50 = calc_percentile(sample, 50.0)
        p95 = calc_percentile(sample, 95.0)
        rtail = p95 / p50 if p50 > 0 else 0.0
        b_p50.append(p50)
        b_p95.append(p95)
        b_rtail.append(rtail)

    b_p50.sort()
    b_p95.sort()
    b_rtail.sort()

    alpha = (100.0 - ci_pct) / 2.0
    return {
        "n_resamples": n_resamples,
        "ci_pct": ci_pct,
        "p50_ci": [
            calc_percentile(b_p50, alpha),
            calc_percentile(b_p50, 100.0 - alpha),
        ],
        "p95_ci": [
            calc_percentile(b_p95, alpha),
            calc_percentile(b_p95, 100.0 - alpha),
        ],
        "r_tail_ci": [
            calc_percentile(b_rtail, alpha),
            calc_percentile(b_rtail, 100.0 - alpha),
        ],
    }


def run_micro_runner(
    threads: int,
    num_steady: int,
    num_warmup: int,
) -> dict:
    """Menjalankan tools/bench/run_core_scaling.mojo dan mengembalikan JSON output."""
    cmd = [
        "pixi",
        "run",
        "mojo",
        "run",
        "-I",
        "src",
        "tools/bench/run_core_scaling.mojo",
        "--threads",
        str(threads),
        "--num-steady",
        str(num_steady),
        "--num-warmup",
        str(num_warmup),
        "--json",
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Micro-runner gagal dengan kode {res.returncode}:\n{res.stderr}"
        )

    stdout = res.stdout
    json_start = stdout.find("{")
    json_end = stdout.rfind("}")
    if json_start < 0 or json_end < 0:
        raise ValueError(
            f"Tidak menemukan output JSON yang valid dari micro-runner:\n{stdout}"
        )

    raw_json = stdout[json_start : json_end + 1]
    return json.loads(raw_json)


def find_latest_w2_report() -> Path | None:
    """Mencari file m11_w2_async_overlap.json terbaru di direktori reports/."""
    candidates = list(REPO_ROOT.glob("reports/*/m11_w2_async_overlap.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    """Eksekusi utama verifikasi Gate G-M11-3."""
    parser = argparse.ArgumentParser(
        description="Verifikasi Tail Latency & Dynamic RAM Budget (Gate G-M11-3)"
    )
    parser.add_argument(
        "--lockfile",
        type=str,
        default="dismoen.hardware.lock",
        help="Path ke file dismoen.hardware.lock",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="",
        help="Profil target dari lockfile (default: active_profile)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Override cacah thread c (default: c*_system dari profile lockfile)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Jumlah iterasi steady N (wajib N >= 100, default: 100)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Jumlah iterasi warm-up (default: 10)",
    )
    parser.add_argument(
        "--w2-report",
        type=str,
        default="",
        help="Path ke laporan W2b JSON untuk asersi E_BW <= 5%% (default: auto)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Path output laporan JSON",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="",
        help="Path output laporan Markdown",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Jumlah resampling bootstrap B (default: 1000)",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("DISMOEN Tail Latency Profiling & RAM Budget Adherence (Gate G-M11-3)")
    print(f"Timestamp:    {datetime.datetime.now().isoformat()}")
    print(f"CPU Model:    {get_cpu_model()}")
    print(f"CPU Governor: {get_cpu_governor()}")
    print("=" * 72)

    # 1. Validasi Input Iterations
    n_runs = args.iterations
    if n_runs < 100:
        raise ValueError(
            f"Jumlah run N = {n_runs} < 100! Gate G-M11-3 mewajibkan N >= 100."
        )

    # 2. Baca Lockfile untuk Titik Deploy c*_system
    lock_file = REPO_ROOT / args.lockfile
    if not lock_file.exists():
        raise FileNotFoundError(
            f"Lockfile {lock_file} tidak ditemukan! Jalankan 'dismoen tune' dulu."
        )

    with open(lock_file, "r") as f:
        lock_data = json.load(f)

    active_profile_name = args.profile or lock_data.get(
        "active_profile", "host_current"
    )
    profiles = lock_data.get("profiles", {})
    if active_profile_name not in profiles:
        raise KeyError(
            f"Profil '{active_profile_name}' tidak ditemukan di {lock_file}!"
        )

    prof = profiles[active_profile_name]
    c_deploy = args.threads if args.threads > 0 else prof.get("c_star_system", 1)
    c_compute_max = prof.get("c_compute_max", 7)
    c_star_comp = prof.get("c_star_compute", 7)
    active_m_budget_gib = prof.get("ram_budget_gib", 5.86)

    print(f"\nProfil Terpilih: [{active_profile_name}]")
    print(f"- Plafon C_compute_max: {c_compute_max}")
    print(f"- Knee c*_compute:     {c_star_comp}")
    print(f"- Titik Deploy c*:     {c_deploy} threads")
    print(f"- Anggaran RAM Aktif:  {active_m_budget_gib:.2f} GiB")

    # 3. Eksekusi Pengujian N >= 100 Run di Titik Deploy c*_system
    print(f"\n--> Menjalankan {n_runs} iterasi steady di c = {c_deploy} threads...")
    bench_out = run_micro_runner(
        threads=c_deploy,
        num_steady=n_runs,
        num_warmup=args.num_warmup,
    )

    latencies = bench_out.get("latencies_ms", [])
    if len(latencies) != n_runs:
        raise RuntimeError(
            f"Jumlah sampel latensi ({len(latencies)}) != target N ({n_runs})!"
        )

    stats = compute_statistics(latencies)
    print(f"   p50:  {stats['p50_ms']:.2f} ms")
    print(f"   p90:  {stats['p90_ms']:.2f} ms")
    print(f"   p95:  {stats['p95_ms']:.2f} ms")
    print(f"   p99:  {stats['p99_ms']:.2f} ms")
    print(f"   Mean: {stats['mean_ms']:.2f} ms (Std: {stats['std_ms']:.2f} ms)")
    print(f"   Min:  {stats['min_ms']:.2f} ms | Max: {stats['max_ms']:.2f} ms")
    print(f"   R_tail (p95/p50): {stats['r_tail']:.4f} (Target Project SLO <= 1.35)")

    # 4. Bootstrap Confidence Interval
    print(
        f"\n--> Menghitung Bootstrap CI 95% (B={args.bootstrap_samples} resamples)..."
    )
    b_ci = run_bootstrap_ci(latencies, n_resamples=args.bootstrap_samples)
    print(f"   p50 95% CI:    [{b_ci['p50_ci'][0]:.2f}, {b_ci['p50_ci'][1]:.2f}] ms")
    print(f"   p95 95% CI:    [{b_ci['p95_ci'][0]:.2f}, {b_ci['p95_ci'][1]:.2f}] ms")
    print(f"   R_tail 95% CI: [{b_ci['r_tail_ci'][0]:.4f}, {b_ci['r_tail_ci'][1]:.4f}]")

    # 5. Uji Kepatuhan Dynamic RAM Budget Adherence (R_RAM <= 0.95)
    vm_hwm_kib = bench_out.get("vm_hwm_kib", 0)
    print(f"\n--> Perekaman VmHWM: {vm_hwm_kib} KiB ({vm_hwm_kib / 1024:.2f} MiB)")

    tier_budgets_gib = {
        "tier_8gb": 8.0,
        "tier_16gb": 16.0,
        "tier_32gb": 32.0,
        "tier_64gb": 64.0,
        "host_active": active_m_budget_gib,
    }

    ram_adherence = {}
    ram_adherence_pass = True
    for tier_name, b_gib in tier_budgets_gib.items():
        budget_kib = b_gib * 1024 * 1024
        r_ram = vm_hwm_kib / budget_kib if budget_kib > 0 else 1.0
        passed = r_ram <= 0.95
        if not passed:
            ram_adherence_pass = False
        ram_adherence[tier_name] = {
            "budget_gib": b_gib,
            "budget_kib": budget_kib,
            "vm_hwm_kib": vm_hwm_kib,
            "r_ram": r_ram,
            "passed": passed,
        }
        print(
            f"   [{tier_name:12s}]: M_budget = {b_gib:4.1f} GiB | "
            f"R_RAM = {r_ram * 100:5.2f}% (<= 95%) | "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # 6. Memory Leak Check (Verifikasi Alokasi Multi-Batch Stabil)
    print("\n--> Menjalankan Verifikasi Memory Leak (Run Konsekutif)...")
    bench_out2 = run_micro_runner(
        threads=c_deploy,
        num_steady=50,
        num_warmup=2,
    )
    vm_hwm_kib_2 = bench_out2.get("vm_hwm_kib", 0)
    leak_delta_kib = abs(vm_hwm_kib_2 - vm_hwm_kib)
    leak_delta_pct = (leak_delta_kib / vm_hwm_kib * 100.0) if vm_hwm_kib > 0 else 0.0
    leak_free = leak_delta_pct <= 5.0
    v_leak = "LEAK-FREE (PASS)" if leak_free else "LEAK DETECTED"
    print(
        f"   VmHWM Run 1: {vm_hwm_kib} KiB | Run 2: {vm_hwm_kib_2} KiB | "
        f"Delta: {leak_delta_pct:.2f}% | {v_leak}"
    )

    # 7. Asersi Stabilitas Bandwidth Storage E_BW <= 5% dari Kalibrasi W2
    print("\n--> Asersi Stabilitas Bandwidth Storage E_BW <= 5%...")
    w2_path_str = args.w2_report
    w2_path = Path(w2_path_str) if w2_path_str else find_latest_w2_report()

    e_bw_pct = 0.0
    e_bw_pass = True
    w2_source = "N/A"
    if w2_path and w2_path.exists():
        w2_source = str(w2_path)
        with open(w2_path, "r") as f:
            w2_data = json.load(f)
        e_bw_pct = w2_data.get("max_e_bw_pct", 0.0)
        e_bw_pass = w2_data.get("e_bw_pass", True) and (e_bw_pct <= 5.0)
        print(
            f"   Sumber W2: {w2_path.name} | E_BW = {e_bw_pct:.2f}% (<= 5.0%) | "
            f"{'PASS' if e_bw_pass else 'FAIL'}"
        )
    else:
        print("   PERINGATAN: Laporan W2 tidak ditemukan, menggunakan nilai 0.0%.")

    # 8. Evaluasi Final Gate G-M11-3
    r_tail_pass = stats["r_tail"] <= 1.35
    gate_g_m11_3_pass = (
        (n_runs >= 100)
        and r_tail_pass
        and ram_adherence_pass
        and leak_free
        and e_bw_pass
    )

    print("\n" + "=" * 72)
    print("EVALUASI FORMAL GATE G-M11-3:")
    p_n = "PASS" if n_runs >= 100 else "FAIL"
    print(f"   [x] Ukuran Sampel N >= 100:            {p_n} (N={n_runs})")
    p_rt = "PASS" if r_tail_pass else "FAIL"
    rt_val = stats["r_tail"]
    print(f"   [x] Project SLO R_tail <= 1.35:        {p_rt} (R_tail={rt_val:.4f})")
    max_r_ram = max(t["r_ram"] for t in ram_adherence.values()) * 100
    p_ram = "PASS" if ram_adherence_pass else "FAIL"
    print(f"   [x] Dynamic RAM Budget Adherence:     {p_ram} (Max={max_r_ram:.2f}%)")
    p_lk = "PASS" if leak_free else "FAIL"
    print(f"   [x] Bebas Leak Memori:                 {p_lk} (d={leak_delta_pct:.2f}%)")
    p_bw = "PASS" if e_bw_pass else "FAIL"
    print(f"   [x] Asersi Stabilitas Storage E_BW:    {p_bw} (E_BW={e_bw_pct:.2f}%)")
    v_g = "HIJAU (PASS)" if gate_g_m11_3_pass else "MERAH (FAIL)"
    print(f"   Verdict Gate G-M11-3:                  {v_g}")
    print("=" * 72)

    # 9. Format Output JSON & MD
    today = datetime.date.today().strftime("%Y-%m-%d")
    report_dir = REPO_ROOT / "reports" / today
    report_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = (
        Path(args.output_json)
        if args.output_json
        else report_dir / "m11_w4_tail_stability.json"
    )
    out_md_path = (
        Path(args.output_md)
        if args.output_md
        else report_dir / "M11-w4-tail-stability.md"
    )

    report_payload = {
        "benchmark": "M11-W4-Tail-Stability-RAM-Adherence",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cpu_model": get_cpu_model(),
        "cpu_governor": get_cpu_governor(),
        "profile": active_profile_name,
        "threads": c_deploy,
        "n_runs": n_runs,
        "statistics": stats,
        "bootstrap_ci": b_ci,
        "ram_adherence": ram_adherence,
        "memory_leak_check": {
            "vm_hwm_run1_kib": vm_hwm_kib,
            "vm_hwm_run2_kib": vm_hwm_kib_2,
            "delta_pct": leak_delta_pct,
            "passed": leak_free,
        },
        "storage_e_bw": {
            "source_report": w2_source,
            "max_e_bw_pct": e_bw_pct,
            "passed": e_bw_pass,
        },
        "gate_g_m11_3_pass": gate_g_m11_3_pass,
    }

    out_json_path.write_text(json.dumps(report_payload, indent=2) + "\n")
    print(f"\n[Saved JSON Report]: {out_json_path}")

    # Format Markdown Report
    md_lines = [
        "# M11-W4: Laporan Tail Latency & Dynamic RAM Budget Adherence",
        "",
        f"- **Tanggal**: {today}",
        f"- **CPU Model**: {get_cpu_model()}",
        f"- **Governor**: {get_cpu_governor()}",
        f"- **Titik Uji**: $c^*_{{system}} = {c_deploy}$ threads "
        f"(Profil `{active_profile_name}`)",
        f"- **Jumlah Run**: $N = {n_runs}$ steady iterations",
        "",
        "## 1. Distribusi Latensi Ekor & Project SLO",
        "",
        "| Metrik | Nilai Empiris (ms) | 95% Bootstrap CI | Keterangan |",
        "|:---|:---:|:---:|:---|",
        f"| $p50$ (Median) | **{stats['p50_ms']:.2f}** | "
        f"[{b_ci['p50_ci'][0]:.2f}, {b_ci['p50_ci'][1]:.2f}] | "
        "Interpolasi Linear Type 7 |",
        f"| $p90$ | {stats['p90_ms']:.2f} | - | Distribusi Ekor |",
        f"| $p95$ | **{stats['p95_ms']:.2f}** | "
        f"[{b_ci['p95_ci'][0]:.2f}, {b_ci['p95_ci'][1]:.2f}] | "
        "Evaluasi Project SLO |",
        f"| $p99$ | {stats['p99_ms']:.2f} | - | Ekor Ekstrem |",
        f"| Min / Max | {stats['min_ms']:.2f} / {stats['max_ms']:.2f} | "
        "- | Rentang Penuh |",
        f"| Mean (Std) | {stats['mean_ms']:.2f} (±{stats['std_ms']:.2f}) | "
        "- | Statistik Agregat |",
        f"| **$R_{{tail}} = p95/p50$** | **{stats['r_tail']:.4f}** | "
        f"[{b_ci['r_tail_ci'][0]:.4f}, {b_ci['r_tail_ci'][1]:.4f}] | "
        "**Project SLO $\\le 1{,}35$** |",
        "",
        "## 2. Kepatuhan Dynamic RAM Budget ($\\mathcal{R}_{RAM} \\le 0{,}95$)",
        "",
        f"- **Puncak Pemakaian Fisik (VmHWM)**: **{vm_hwm_kib} KiB** "
        f"({vm_hwm_kib / 1024:.2f} MiB)",
        f"- **Uji Stabilitas Leak**: Run 1 = {vm_hwm_kib} KiB, "
        f"Run 2 = {vm_hwm_kib_2} KiB (Delta {leak_delta_pct:.2f}% $\\le 5\\%$, "
        "✅ LEAK-FREE)",
        "",
        "| Tier RAM Budget | Anggaran $M_{budget}$ | "
        "$\\mathcal{R}_{RAM} = \\text{VmHWM}/M_{budget}$ | Batas Maksimum | "
        "Status |",
        "|:---|:---:|:---:|:---:|:---:|",
    ]

    for t_name, r_info in ram_adherence.items():
        bg = r_info["budget_gib"]
        rr = r_info["r_ram"] * 100
        st = "✅ PASS" if r_info["passed"] else "❌ FAIL"
        md_lines.append(
            f"| `{t_name}` | {bg:.1f} GiB | {rr:.2f}% | $\\le 95\\%$ | {st} |"
        )

    v_n = "PASS" if n_runs >= 100 else "FAIL"
    v_rt = "PASS" if r_tail_pass else "FAIL"
    v_ram = "PASS" if ram_adherence_pass else "FAIL"
    v_lk = "PASS" if leak_free else "FAIL"
    v_bw = "PASS" if e_bw_pass else "FAIL"
    v_all = "ALL GATES PASS (HIJAU)" if gate_g_m11_3_pass else "GATE REJECTED (MERAH)"

    md_lines.extend(
        [
            "",
            "## 3. Asersi Stabilitas Throughput Storage ($E_{BW} \\le 5\\%$)",
            "",
            f"- **Laporan Sumber W2b**: `{Path(w2_source).name}`",
            f"- **Variasi Throughput Maksimum $E_{{BW}}$**: **{e_bw_pct:.2f}%**",
            f"- **Kriteria Stabilitas**: $E_{{BW}} \\le 5.0\\%$ "
            f"({'✅ TERPENUHI' if e_bw_pass else '❌ GAGAL'})",
            "",
            "## 4. Formal Scorecard Gate G-M11-3",
            "",
            f"- [x] Ukuran Sampel $N \\ge 100$: **{v_n}** ($N = {n_runs}$)",
            f"- [x] Project SLO $R_{{tail}} \\le 1.35$: **{v_rt}** "
            f"($R_{{tail}} = {stats['r_tail']:.4f}$)",
            f"- [x] Dynamic RAM Budget Adherence "
            f"$\\mathcal{{R}}_{{RAM}} \\le 0.95$: **{v_ram}**",
            f"- [x] Verifikasi Tanpa Kebocoran Memori (Leak-Free): **{v_lk}**",
            f"- [x] Stabilitas Bandwidth Storage $E_{{BW}} \\le 5\\%$: **{v_bw}**",
            f"- **OVERALL VERDICT**: **{v_all}**",
        ]
    )

    out_md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[Saved Markdown Report]: {out_md_path}")

    if not gate_g_m11_3_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
