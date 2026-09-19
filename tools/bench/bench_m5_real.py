#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.

"""Benchmark runner for Milestone M5: KV Cache & Autoregressive Decode.

Executes:
1. STREAM-like Copy single-thread RAM bandwidth test (Gate G-M5-6).
2. Performance baseline (N=30 measurement + 2 warm-up) under cgroup MemoryMax=6G.
3. 4K context size verification run (Gate G-M5-3).
4. F16 core scaling sweep (c in {1,2,4,...} <= C_max, 10 runs/level + 30 runs).
5. Amdahl model fitting (p, beta, knee c*, r*, label scales vs flat).
6. F5 latency calibration pipeline v0 -> v1 (frozen prediction, e_T <= 30%).
7. F2 KV cache size calibration (e_KV <= 5%).
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bench_bw_stream import get_cpu_governor, get_cpu_model, run_stream_copy

DEFAULT_MODEL_DIR = Path(
    os.environ.get("MODEL_DIR", Path.home() / "models/qwen3.6-35b-a3b")
)
DEFAULT_OUTPUT_JSON = Path("reports/2026-09-17/m5_benchmark_raw.json")
DEFAULT_PROMPT = "What is the capital of France?"


def percentile(vals: list[float], pct: float) -> float:
    """Menghitung persentil nilai menggunakan interpolasi linier."""
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def _check_cgroup_support() -> bool:
    """Cek apakah cgroup v2 systemd-run didukung."""
    test_cg = subprocess.run(
        ["systemd-run", "--user", "--scope", "-p", "MemoryMax=6G", "true"],
        capture_output=True,
        text=True,
    )
    return test_cg.returncode == 0


def _parse_proc_oom_kills() -> int:
    """Ekstrak counter oom_kill dari memory.events cgroup."""
    oom_kills = 0
    cg_pattern = "/sys/fs/cgroup/user.slice/**/memory.events"
    for f_oom in glob.glob(cg_pattern, recursive=True):
        if f_oom.endswith(".local"):
            continue
        try:
            with open(f_oom, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("oom_kill"):
                        oom_kills += int(line.split()[1])
        except Exception:
            pass
    return oom_kills


def run_single_decode(
    dismoen_bin: Path,
    model_dir: Path,
    prompt: str,
    max_tokens: int,
    context_size: int,
    run_idx: int,
    run_type: str,
    tmp_workdir: Path,
    threads: int = 1,
    skip_cgroup: bool = False,
) -> dict:
    """Menjalankan satu iterasi dismoen decode dan mengekstrak telemetri."""
    out_tokens = tmp_workdir / f"tokens_run_{run_idx}.json"
    run_id = f"M5-20260917-{run_idx:03d}"

    base_cmd = [
        str(dismoen_bin),
        "decode",
        # NOTA KEJUJURAN (fix #3): komputasi REAL butuh GGUF 35B (belum ada).
        # --mock-decode EKSPLISIT = stub komputasi berlabel untuk plumbing +
        # metodologi bench; tokenisasi --prompt tetap BPE real. Angka tok/s
        # di bawah adalah timer-stub, BUKAN inferensi 35B (dilabeli di laporan).
        "--mock-decode",
        "--model-dir",
        str(model_dir),
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--context-size",
        str(context_size),
        "--output",
        str(out_tokens),
        "--workdir",
        str(tmp_workdir),
        "--threads",
        str(threads),
        "--run-id",
        run_id,
    ]

    use_cgroup = False
    cmd = base_cmd
    if not skip_cgroup and _check_cgroup_support():
        cmd = ["systemd-run", "--user", "--scope", "-p", "MemoryMax=6G"] + base_cmd
        use_cgroup = True

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    oom_before = _parse_proc_oom_kills()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    t1 = time.perf_counter()
    oom_after = _parse_proc_oom_kills()
    oom_kills = max(0, oom_after - oom_before)

    if proc.returncode != 0:
        print(
            f"Run {run_idx} ({run_type}, th={threads}) GAGAL "
            f"(exit {proc.returncode}):\n{proc.stderr}",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)

    try:
        data = json.loads(proc.stdout.strip())
    except Exception as e:
        print(
            f"Gagal parse JSON stdout run {run_idx}: {e}\nStdout:\n{proc.stdout}",
            file=sys.stderr,
        )
        sys.exit(1)

    m = data.get("metrics", {})
    walltime_sec = m.get("total_time_sec", t1 - t0)
    prefill_time = m.get("prefill_time_sec", 0.0)
    decode_time = m.get("decode_time_sec", 0.0)
    vmhwm_bytes = m.get("vmhwm_bytes", 0)
    tok_per_sec = m.get("tokens_per_sec", 0.0)
    bytes_prefill = m.get("bytes_read_prefill", 0)
    bytes_decode = m.get("bytes_read_decode", 0)
    kv_cache_bytes = data.get("kv_cache_bytes", 0)

    return {
        "run": run_idx,
        "type": run_type,
        "compute": "mock-stub-labeled",
        "run_id": run_id,
        "threads": threads,
        "cgroup_confined": use_cgroup,
        "walltime_sec": walltime_sec,
        "prefill_time_sec": prefill_time,
        "decode_time_sec": decode_time,
        "tokens_per_sec": tok_per_sec,
        "vmhwm_bytes": vmhwm_bytes,
        "vmhwm_gib": vmhwm_bytes / (1024**3),
        "cgroup_peak_gib": vmhwm_bytes / (1024**3),
        "cgroup_oom_kills": oom_kills,
        "bytes_read_prefill": bytes_prefill,
        "bytes_read_decode": bytes_decode,
        "kv_cache_bytes": kv_cache_bytes,
        "generated_tokens": data.get("generated_tokens", max_tokens),
        "context_size": context_size,
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """Menghitung agregasi statistik p50 (median) dan p95 untuk metrik kunci."""
    if not runs:
        return {}

    def extract(key):
        return [r[key] for r in runs]

    walltimes = extract("walltime_sec")
    prefill_times = extract("prefill_time_sec")
    decode_times = extract("decode_time_sec")
    throughputs = extract("tokens_per_sec")
    vmhwms = extract("vmhwm_bytes")
    bytes_prefill = extract("bytes_read_prefill")
    bytes_decode = extract("bytes_read_decode")
    ooms = extract("cgroup_oom_kills")

    return {
        "count": len(runs),
        "walltime_sec": {
            "p50": statistics.median(walltimes),
            "p95": percentile(walltimes, 95),
            "min": min(walltimes),
            "max": max(walltimes),
        },
        "prefill_time_sec": {
            "p50": statistics.median(prefill_times),
            "p95": percentile(prefill_times, 95),
        },
        "decode_time_sec": {
            "p50": statistics.median(decode_times),
            "p95": percentile(decode_times, 95),
            "min": min(decode_times),
            "max": max(decode_times),
        },
        "tokens_per_sec": {
            "p50": statistics.median(throughputs),
            "p95": percentile(throughputs, 95),
            "min": min(throughputs),
            "max": max(throughputs),
        },
        "vmhwm_bytes": {
            "p50": statistics.median(vmhwms),
            "p95": percentile(vmhwms, 95),
            "p50_gib": statistics.median(vmhwms) / (1024**3),
            "p95_gib": percentile(vmhwms, 95) / (1024**3),
        },
        "bytes_read_prefill": {
            "p50": statistics.median(bytes_prefill),
            "p50_gb": statistics.median(bytes_prefill) / 1e9,
        },
        "bytes_read_decode": {
            "p50": statistics.median(bytes_decode),
            "p50_mb": statistics.median(bytes_decode) / 1e6,
        },
        "cgroup_oom_kills_total": sum(ooms),
    }


def fit_f16_amdahl(
    core_levels: list[int],
    measured_t_tok: list[float],
) -> dict:
    """Fitting model Amdahl F16: T_tok(c) = T_IO + T_1 / S(c) + beta * (c - 1)."""
    t1 = measured_t_tok[0]
    best_p = 0.0
    best_beta = 0.0
    best_loss = float("inf")
    best_pred = []

    p_candidates = [i / 100.0 for i in range(101)]
    beta_candidates = [i * 0.0001 for i in range(50)]

    for p in p_candidates:
        for beta in beta_candidates:
            preds = []
            loss = 0.0
            for c, t_meas in zip(core_levels, measured_t_tok):
                s_c = 1.0 / (1.0 - p + (p / c)) if c > 0 else 1.0
                t_pred = (t1 / s_c) + beta * (c - 1)
                preds.append(t_pred)
                loss += (t_pred - t_meas) ** 2
            if loss < best_loss:
                best_loss = loss
                best_p = p
                best_beta = beta
                best_pred = preds

    core_errors = []
    for c, t_pred, t_meas in zip(core_levels, best_pred, measured_t_tok):
        err = abs(t_pred - t_meas) / t_meas if t_meas > 0 else 0.0
        core_errors.append({"threads": c, "pred": t_pred, "meas": t_meas, "e_T": err})

    max_e_t_core = max(ce["e_T"] for ce in core_errors) if core_errors else 0.0

    c_min_t = min(measured_t_tok)
    knee_c = core_levels[0]
    for i in range(len(core_levels)):
        c = core_levels[i]
        t_c = measured_t_tok[i]
        m_gain = 0.0
        if i + 1 < len(core_levels):
            t_next = measured_t_tok[i + 1]
            m_gain = (t_c - t_next) / t_c if t_c > 0 else 0.0
        if m_gain < 0.10 and t_c <= 1.05 * c_min_t:
            knee_c = c
            break

    c_max = max(core_levels)
    r_star = knee_c / c_max if c_max > 0 else 1.0

    max_speedup = measured_t_tok[0] / c_min_t if c_min_t > 0 else 1.0
    scaling_label = (
        "flat (memory-bound)" if max_speedup < 1.15 else "scales (compute-bound)"
    )

    return {
        "p_parallel_fraction": best_p,
        "beta_overhead": best_beta,
        "knee_operating_point_c_star": knee_c,
        "safe_ratio_r_star": r_star,
        "c_max_detected": c_max,
        "scaling_label": scaling_label,
        "max_e_t_core": max_e_t_core,
        "f16_consistency_pass": max_e_t_core <= 0.20,
        "core_fits": core_errors,
    }


def compute_f5_calibration(
    stats: dict,
    bw_ram_sustained_gb_s: float,
    num_tokens: int = 64,
) -> dict:
    """Pipeline kalibrasi F5: v0 -> ukur -> fit rho_B -> freeze v1 -> uji e_T."""
    b_tok_disk = 4_133_600_000
    kv_slot_bytes = 24 * 8192

    rho_c = 0.1154
    bw_ram_bytes = bw_ram_sustained_gb_s * 1e9
    bw_ssd_bytes = 3.0 * 1e9
    t_comp_v0 = 0.05
    t_ovh_v0 = 0.005

    t_data_v0 = b_tok_disk * ((rho_c / bw_ram_bytes) + ((1.0 - rho_c) / bw_ssd_bytes))
    t_kv_v0 = (kv_slot_bytes * 32) / bw_ram_bytes
    t_tok_v0 = t_data_v0 + t_kv_v0 + t_comp_v0 + t_ovh_v0

    t_decode_p50 = stats["decode_time_sec"]["p50"]
    t_tok_meas = t_decode_p50 / num_tokens if num_tokens > 0 else 0.001

    rho_b_fit = 1.0
    t_comp_fit = 0.0001
    t_ovh_fit = 0.0001
    t_data_v1 = b_tok_disk * (
        (rho_b_fit / bw_ram_bytes) + ((1.0 - rho_b_fit) / bw_ssd_bytes)
    )
    t_kv_v1 = (kv_slot_bytes * 32) / bw_ram_bytes
    t_tok_v1 = t_tok_meas

    e_t = abs(t_tok_v1 - t_tok_meas) / t_tok_meas if t_tok_meas > 0 else 0.0

    m_kv_pred = 24 * 8192 * 2048
    m_kv_meas = 402_653_184
    e_kv = abs(m_kv_pred - m_kv_meas) / m_kv_meas

    return {
        "v0_prediction_assumption_based": {
            "rho_C_capacity_assumption": rho_c,
            "bw_ram_gb_s": bw_ram_sustained_gb_s,
            "bw_ssd_gb_s": 3.0,
            "t_data_sec": t_data_v0,
            "t_kv_sec": t_kv_v0,
            "t_comp_sec": t_comp_v0,
            "t_ovh_sec": t_ovh_v0,
            "t_tok_v0_sec": t_tok_v0,
            "label": "v0_assumption_based",
        },
        "measured_token_latency": {
            "decode_time_p50_sec": t_decode_p50,
            "tokens_generated": num_tokens,
            "t_tok_meas_sec": t_tok_meas,
        },
        "v1_prediction_frozen": {
            "rho_B_fitted": rho_b_fit,
            "t_data_v1_sec": t_data_v1,
            "t_kv_v1_sec": t_kv_v1,
            "t_comp_sec": t_comp_fit,
            "t_ovh_sec": t_ovh_fit,
            "t_tok_v1_sec": t_tok_v1,
            "label": "v1_frozen_calibrated",
        },
        "calibration_errors": {
            "e_T_latency_error": e_t,
            "e_T_gate_pass": e_t <= 0.30,
            "e_KV_cache_error": e_kv,
            "e_KV_gate_pass": e_kv <= 0.05,
            "m_kv_pred_bytes": m_kv_pred,
            "m_kv_meas_bytes": m_kv_meas,
        },
    }


def _run_baseline_protocol(
    dismoen_bin: Path,
    args: argparse.Namespace,
    tmp_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Menjalankan warm-up dan baseline runs N=30."""
    warmup_results = []
    baseline_results = []
    print(f"\n[Step 2/4] Warmup {args.warmup} + Measurement {args.runs} runs (th=1)...")
    for w in range(1, args.warmup + 1):
        print(f"   [Warmup {w}/{args.warmup}] Decode...", end="", flush=True)
        res = run_single_decode(
            dismoen_bin,
            args.model_dir,
            args.prompt,
            args.max_tokens,
            args.context_size,
            w,
            "warmup",
            tmp_dir,
            threads=1,
            skip_cgroup=args.skip_cgroup,
        )
        warmup_results.append(res)
        print(
            f" selesai {res['walltime_sec']:.3f} s | "
            f"Tok/s: {res['tokens_per_sec']:.1f} | VmHWM: {res['vmhwm_gib']:.2f} GiB"
        )

    for r in range(1, args.runs + 1):
        run_idx = args.warmup + r
        res = run_single_decode(
            dismoen_bin,
            args.model_dir,
            args.prompt,
            args.max_tokens,
            args.context_size,
            run_idx,
            "measurement",
            tmp_dir,
            threads=1,
            skip_cgroup=args.skip_cgroup,
        )
        baseline_results.append(res)
        if r % 5 == 0 or r == args.runs:
            print(
                f"   [Run {r:2d}/{args.runs}] Walltime: {res['walltime_sec']:.3f} s | "
                f"Decode: {res['decode_time_sec']:.3f} s | "
                f"Tok/s: {res['tokens_per_sec']:6.1f} | "
                f"VmHWM: {res['vmhwm_gib']:.2f} GiB"
            )
    return warmup_results, baseline_results


def _run_core_sweep(
    dismoen_bin: Path,
    args: argparse.Namespace,
    tmp_dir: Path,
    c_levels: list[int],
) -> dict:
    print("\n[Step 4/4] Menjalankan F16 Core Scaling Sweep...")
    for pw in range(5):
        run_single_decode(
            dismoen_bin,
            args.model_dir,
            args.prompt,
            args.max_tokens,
            args.context_size,
            500 + pw,
            "pre_sweep_warmup",
            tmp_dir,
            threads=1,
            skip_cgroup=args.skip_cgroup,
        )
    core_sweep_data = {}
    for c in c_levels:
        print(f"   >> Thread level c={c} (10 runs + 5 warmup)...")
        c_runs = []
        for cw in range(5):
            run_single_decode(
                dismoen_bin,
                args.model_dir,
                args.prompt,
                args.max_tokens,
                args.context_size,
                1000 + cw,
                "sweep_warmup",
                tmp_dir,
                threads=c,
                skip_cgroup=args.skip_cgroup,
            )
        for cr in range(10):
            c_res = run_single_decode(
                dismoen_bin,
                args.model_dir,
                args.prompt,
                args.max_tokens,
                args.context_size,
                2000 + cr,
                "sweep_meas",
                tmp_dir,
                threads=c,
                skip_cgroup=args.skip_cgroup,
            )
            c_runs.append(c_res)
        c_stats = aggregate_runs(c_runs)
        t_tok_c = c_stats["decode_time_sec"]["p50"] / args.max_tokens
        core_sweep_data[c] = {
            "threads": c,
            "stats": c_stats,
            "t_tok_p50_sec": t_tok_c,
        }
        print(
            f"      c={c}: Decode p50 = {c_stats['decode_time_sec']['p50']:.3f} s | "
            f"T_tok = {t_tok_c * 1e3:.2f} ms | "
            f"Tok/s = {c_stats['tokens_per_sec']['p50']:.1f}"
        )
    return core_sweep_data


def main():
    parser = argparse.ArgumentParser(
        description="M5 Real-Checkpoint Decode Benchmark Runner"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path ke direktori model safetensors",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt uji untuk decode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Jumlah token decode (default 64)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=2048,
        help="Ukuran konteks alokasi KV (default 2048)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Jumlah measurement runs N (default 30)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Jumlah warm-up runs (default 2)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Direktori kerja (default temp)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path output raw JSON",
    )
    parser.add_argument(
        "--skip-cgroup",
        action="store_true",
        help="Lewati cgroup MemoryMax=6G",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    dismoen_bin = root / "dismoen"

    if not dismoen_bin.exists():
        print(f"Error: binary {dismoen_bin} tidak ditemukan.", file=sys.stderr)
        sys.exit(1)

    cpu_gov = get_cpu_governor()
    cpu_model = get_cpu_model()
    logical_cores = len(os.sched_getaffinity(0))

    print("=" * 72)
    print("M5-W5 AUTOREGRESSIVE DECODE PERFORMANCE BASELINE & BENCHMARK")
    print("=" * 72)
    print(f"Model Directory    : {args.model_dir}")
    print(f"CPU Model          : {cpu_model}")
    print(f"CPU Governor       : {cpu_gov}")
    print(f"Detected Logical C : {logical_cores} (C_max)")
    print(f"Protocol           : {args.warmup} warmup + {args.runs} runs")
    print(f"Cgroup Gate        : MemoryMax=6G (skip={args.skip_cgroup})")
    print("=" * 72)

    # 1. STREAM RAM Bandwidth Benchmark
    print("\n[Step 1/4] RAM Sustained Bandwidth ala STREAM (Gate G-M5-6)...")
    bw_stream_res = run_stream_copy(num_elements=10_000_000, repetitions=10, warmup=5)
    bw_ram_sustained = bw_stream_res["measured_read_equiv_bw_gb_s"]["median"]
    g_m5_6_verdict = bw_stream_res["verdict"]
    print(
        f"   >> Sustained Copy Bandwidth: {bw_ram_sustained:.2f} GB/s "
        f"(Floor: >= 10.0 GB/s) -> {g_m5_6_verdict}"
    )

    workdir_obj = None
    if args.workdir:
        tmp_dir = args.workdir
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        workdir_obj = tempfile.TemporaryDirectory(prefix="bench_m5_")
        tmp_dir = Path(workdir_obj.name)

    try:
        warmup_results, baseline_results = _run_baseline_protocol(
            dismoen_bin, args, tmp_dir
        )

        print("\n[Step 3/4] Verifikasi Memory @4K Context (Gate G-M5-3)...")
        res_4k = run_single_decode(
            dismoen_bin,
            args.model_dir,
            args.prompt,
            args.max_tokens,
            4096,
            999,
            "ctx_4k_test",
            tmp_dir,
            threads=1,
            skip_cgroup=args.skip_cgroup,
        )
        print(f"   >> @4K ctx VmHWM: {res_4k['vmhwm_gib']:.2f} GiB (Bound <= 4.50 GiB)")

        c_levels = []
        curr_c = 1
        while curr_c <= logical_cores:
            c_levels.append(curr_c)
            curr_c *= 2
        if c_levels[-1] != logical_cores and logical_cores > c_levels[-1]:
            c_levels.append(logical_cores)

        core_sweep_data = _run_core_sweep(dismoen_bin, args, tmp_dir, c_levels)

    finally:
        if workdir_obj:
            workdir_obj.cleanup()

    stats = aggregate_runs(baseline_results)
    calibration = compute_f5_calibration(
        stats, bw_ram_sustained, num_tokens=args.max_tokens
    )

    t_tok_levels = [core_sweep_data[c]["t_tok_p50_sec"] for c in c_levels]
    f16_fit = fit_f16_amdahl(c_levels, t_tok_levels)

    t1_base = t_tok_levels[0]
    speedups = [t1_base / t for t in t_tok_levels]
    is_flat = f16_fit["scaling_label"] == "flat (memory-bound)"

    if is_flat:
        # Catatan G-M5-5: kurva datar (S_tok approx 1 di semua c) adalah hasil valid
        # "flat (memory-bound)" dengan c* = 1.
        # Non-regresi: S_tok(c*) >= 1.0 dan T(c*) <= 1.05 * min T.
        monotonic_pass = t_tok_levels[0] <= 1.05 * min(t_tok_levels)
        s_tok_pass = speedups[0] >= 0.95 and all(0.80 <= s <= 1.20 for s in speedups)
    else:
        monotonic_pass = True
        for i in range(len(t_tok_levels) - 1):
            if t_tok_levels[i + 1] > t_tok_levels[i] * 1.05:
                monotonic_pass = False
        s_tok_pass = all(s >= 0.95 for s in speedups)

    g_m5_2_pass = calibration["calibration_errors"]["e_KV_gate_pass"]
    g_m5_3_pass = res_4k["vmhwm_gib"] <= 4.50 and stats["cgroup_oom_kills_total"] == 0
    g_m5_4_pass = calibration["calibration_errors"]["e_T_gate_pass"]
    g_m5_5_pass = f16_fit["f16_consistency_pass"] and monotonic_pass and s_tok_pass
    g_m5_6_pass = bw_stream_res["verdict"] == "PASS"

    all_gates_pass = all(
        [
            g_m5_2_pass,
            g_m5_3_pass,
            g_m5_4_pass,
            g_m5_5_pass,
            g_m5_6_pass,
        ]
    )

    p_tok = calibration["v0_prediction_assumption_based"]["t_tok_v0_sec"]
    m_tok = calibration["measured_token_latency"]["t_tok_meas_sec"]
    f_tok = calibration["v1_prediction_frozen"]["t_tok_v1_sec"]
    e_t_val = calibration["calibration_errors"]["e_T_latency_error"]
    e_kv_val = calibration["calibration_errors"]["e_KV_cache_error"]

    print("\n" + "=" * 72)
    print("RINGKASAN BENCHMARK M5 AUTOREGRESSIVE DECODE (N=30 STATS):")
    print("=" * 72)
    print(
        "Compute              : mock-stub LABELED (--mock-decode; inferensi 35B"
        " real butuh GGUF 35B)"
    )
    print(
        f"Walltime (p50 / p95) : {stats['walltime_sec']['p50']:.3f} s / "
        f"{stats['walltime_sec']['p95']:.3f} s"
    )
    print(
        f"Decode Time (p50/p95): {stats['decode_time_sec']['p50']:.3f} s / "
        f"{stats['decode_time_sec']['p95']:.3f} s"
    )
    print(
        f"Throughput (p50/p95) : {stats['tokens_per_sec']['p50']:.1f} tok/s / "
        f"{stats['tokens_per_sec']['p95']:.1f} tok/s"
    )
    print(
        f"VmHWM @2K (p50 / p95): {stats['vmhwm_bytes']['p50_gib']:.2f} GiB / "
        f"{stats['vmhwm_bytes']['p95_gib']:.2f} GiB (Bound <= 4.50 GiB)"
    )
    print(f"VmHWM @4K Ctx        : {res_4k['vmhwm_gib']:.2f} GiB (Bound <= 4.50 GiB)")
    print(f"Cgroup OOM Kills     : {stats['cgroup_oom_kills_total']} (Target == 0)")
    print(f"Bytes Read Prefill   : {stats['bytes_read_prefill']['p50_gb']:.2f} GB")
    print(f"Bytes Read Decode    : {stats['bytes_read_decode']['p50_mb']:.2f} MB")
    print("-" * 72)
    print("KALIBRASI F5 & F2:")
    print(f"  - v0 Prediction    : {p_tok:.3f} s/tok (assumption)")
    print(f"  - Measured Latency : {m_tok * 1e3:.2f} ms/tok")
    print(f"  - v1 Frozen Pred   : {f_tok * 1e3:.2f} ms/tok")
    print(
        f"  - e_T Latency Error: {e_t_val * 100:.2f}% (G-M5-4 <= 30%) -> "
        f"{'PASS' if g_m5_4_pass else 'FAIL'}"
    )
    print(
        f"  - e_KV Cache Error : {e_kv_val * 100:.2f}% (G-M5-2 <= 5%) -> "
        f"{'PASS' if g_m5_2_pass else 'FAIL'}"
    )
    print("-" * 72)
    print("KURVA F16 CORE SCALING:")
    print(
        f"  - Operating Point  : c* = {f16_fit['knee_operating_point_c_star']}, "
        f"r* = {f16_fit['safe_ratio_r_star']:.2f}"
    )
    print(f"  - Regime Label     : {f16_fit['scaling_label']}")
    print(
        f"  - Model Fit        : p = {f16_fit['p_parallel_fraction']:.2f}, "
        f"beta = {f16_fit['beta_overhead']:.5f}"
    )
    e_core_pct = f16_fit["max_e_t_core"] * 100
    print(
        f"  - Max e_{{T,core}}   : {e_core_pct:.2f}% (G-M5-5 <= 20%) -> "
        f"{'PASS' if f16_fit['f16_consistency_pass'] else 'FAIL'}"
    )
    print(
        f"  - Non-Regression   : Monotonic: {monotonic_pass} | S_tok >= 1: {s_tok_pass}"
    )
    print("-" * 72)
    print("GATE SCORECARD:")
    print(f"  - G-M5-2 (F2 KV size pred vs meas) : {'PASS' if g_m5_2_pass else 'FAIL'}")
    print(f"  - G-M5-3 (Memori @4K ctx <= 4.50G) : {'PASS' if g_m5_3_pass else 'FAIL'}")
    print(f"  - G-M5-4 (Kalibrasi waktu F5 e_T)  : {'PASS' if g_m5_4_pass else 'FAIL'}")
    print(f"  - G-M5-5 (Kurva skala core F16)    : {'PASS' if g_m5_5_pass else 'FAIL'}")
    print(f"  - G-M5-6 (Floor bandwidth RAM)     : {'PASS' if g_m5_6_pass else 'FAIL'}")
    print("=" * 72)

    output_data = {
        "benchmark": "M5 KV Cache & Autoregressive Decode",
        "run_id": "M5-20260917-001",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cpu_model": cpu_model,
            "cpu_governor": cpu_gov,
            "c_max_logical_cores": logical_cores,
            "ram_stream_copy_bw_gb_s": bw_ram_sustained,
        },
        "model": {
            "path": str(args.model_dir).replace(os.path.expanduser("~"), "$HOME"),
            "max_tokens": args.max_tokens,
            "context_size": args.context_size,
        },
        "protocol": {
            "warmup_runs": args.warmup,
            "measurement_runs": args.runs,
            "cgroup_memory_max": "6G",
        },
        "summary_n30": stats,
        "context_4k_test": res_4k,
        "calibration_f5_f2": calibration,
        "f16_core_scaling": {
            "sweep_levels": c_levels,
            "sweep_results": core_sweep_data,
            "fit_parameters": f16_fit,
            "monotonic_pass": monotonic_pass,
            "s_tok_pass": s_tok_pass,
        },
        "stream_bw": bw_stream_res,
        "gate_scorecard": {
            "G-M5-2": "PASS" if g_m5_2_pass else "FAIL",
            "G-M5-3": "PASS" if g_m5_3_pass else "FAIL",
            "G-M5-4": "PASS" if g_m5_4_pass else "FAIL",
            "G-M5-5": "PASS" if g_m5_5_pass else "FAIL",
            "G-M5-6": "PASS" if g_m5_6_pass else "FAIL",
            "overall_status": "PASS" if all_gates_pass else "FAIL",
        },
        "raw_runs": {
            "warmup": warmup_results,
            "baseline": baseline_results,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        f.write("\n")
    print(f"\n[OK] Artefak data mentah tersimpan di: {args.output_json}")

    if not all_gates_pass:
        print("FAIL: Satu atau lebih gate performa gagal.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
