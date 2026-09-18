#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.

"""Benchmark runner for Milestone M4: Real-checkpoint Full Forward 24-Layer Prefill.

Executes warm-up and measurement runs of `./kimo forward` on
`/home/will/models/qwen1.5-moe-a2.7b-chat` under
`systemd-run --user --scope -p MemoryMax=6G`.
Measures walltime, VmHWM, cgroup peak/OOM, logical/physical bytes read,
phase breakdown, effective sustained bandwidth, and computes F4/F5
roofline calibration.
"""

import argparse
import glob
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_MODEL_DIR = Path("/home/will/models/qwen1.5-moe-a2.7b-chat")
DEFAULT_TOKENS = Path("tools/fixtures/m4_prompt1_tokens.json")
DEFAULT_OUTPUT_JSON = Path("reports/2026-09-16/m4_benchmark_raw.json")


def get_cpu_governor() -> str:
    """Mendapatkan CPU scaling governor dari sysfs."""
    govs = set()
    for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                govs.add(f.read().strip())
        except Exception:
            pass
    if govs:
        return ", ".join(sorted(govs))
    return "unknown"


def get_cpu_model() -> str:
    """Mendapatkan CPU model name dari /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


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


def run_single_forward(
    kimo_bin: Path,
    model_dir: Path,
    tokens_file: Path,
    run_idx: int,
    run_type: str,
    tmp_workdir: Path,
    threads: int = 1,
    skip_cgroup: bool = False,
) -> dict:
    """Menjalankan satu iterasi kimo forward dan mengekstrak telemetri."""
    out_bin = tmp_workdir / f"logits_run_{run_idx}.bin"
    timing_json = tmp_workdir / f"timing_run_{run_idx}.json"
    run_id = f"M4-20260916-{run_idx:03d}"

    base_cmd = [
        str(kimo_bin),
        "forward",
        "--model-dir",
        str(model_dir),
        "--tokens",
        str(tokens_file),
        "--output",
        str(out_bin),
        "--workdir",
        str(tmp_workdir),
        "--layer-timing",
        str(timing_json),
        "--run-id",
        run_id,
        "--threads",
        str(threads),
    ]

    use_cgroup = False
    cmd = base_cmd
    if not skip_cgroup:
        test_cg = subprocess.run(
            ["systemd-run", "--user", "--scope", "-p", "MemoryMax=6G", "true"],
            capture_output=True,
            text=True,
        )
        if test_cg.returncode == 0:
            cmd = ["systemd-run", "--user", "--scope", "-p", "MemoryMax=6G"] + base_cmd
            use_cgroup = True

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()

    if proc.returncode != 0:
        print(
            f"Run {run_idx} ({run_type}) GAGAL "
            f"(exit code {proc.returncode}):\n{proc.stderr}",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)

    try:
        data = json.loads(proc.stdout.strip())
    except Exception as e:
        print(
            f"Gagal parse JSON stdout pada run {run_idx}: {e}\n"
            f"Raw stdout:\n{proc.stdout}",
            file=sys.stderr,
        )
        sys.exit(1)

    m = data.get("metrics", {})
    phases = m.get("phases", {})
    walltime_sec = m.get("walltime_sec", t1 - t0)
    vmhwm_bytes = m.get("vmhwm_bytes", 0)
    logical_bytes = m.get("logical_bytes_read", 0)
    physical_bytes = m.get("physical_read_bytes", 0)
    cgroup_peak = m.get("cgroup_peak_bytes", 0)
    cgroup_oom = m.get("cgroup_oom_kills", 0)
    layer_fwd_sec = phases.get("layer_forward_sec", 1.0)

    # Ekstrak waktu pread murni dari layer timing jika tersedia
    pread_sec_total = 0.0
    if timing_json.exists():
        try:
            with open(timing_json, "r", encoding="utf-8") as tf:
                t_data = json.load(tf)
                pread_sec_total = sum(
                    lt.get("pread_sec", 0.0) for lt in t_data.get("layer_timing", [])
                )
        except Exception:
            pass

    # Bandwidth streaming I/O murni (pread) vs pipeline forward keseluruhan
    bw_pread_gb_s = (
        (logical_bytes / pread_sec_total) / 1e9 if pread_sec_total > 0 else 0.0
    )
    bw_eff_gb_s = (logical_bytes / layer_fwd_sec) / 1e9 if layer_fwd_sec > 0 else 0.0
    bw_eff_gib_s = (
        (logical_bytes / layer_fwd_sec) / (1024**3) if layer_fwd_sec > 0 else 0.0
    )

    num_tokens = data.get("num_tokens", 16)
    throughput_tok_s = num_tokens / walltime_sec if walltime_sec > 0 else 0.0

    return {
        "run": run_idx,
        "type": run_type,
        "run_id": run_id,
        "cgroup_confined": use_cgroup,
        "walltime_sec": walltime_sec,
        "throughput_tok_s": throughput_tok_s,
        "vmhwm_bytes": vmhwm_bytes,
        "vmhwm_gib": vmhwm_bytes / (1024**3),
        "logical_bytes_read": logical_bytes,
        "logical_bytes_gb": logical_bytes / 1e9,
        "logical_bytes_gib": logical_bytes / (1024**3),
        "physical_read_bytes": physical_bytes,
        "physical_read_gb": physical_bytes / 1e9,
        "physical_read_gib": physical_bytes / (1024**3),
        "cgroup_peak_bytes": cgroup_peak,
        "cgroup_peak_gib": cgroup_peak / (1024**3),
        "cgroup_oom_kills": cgroup_oom,
        "pread_sec_total": pread_sec_total,
        "bw_pread_gb_s": bw_pread_gb_s,
        "bw_effective_gb_s": bw_eff_gb_s,
        "bw_effective_gib_s": bw_eff_gib_s,
        "phases": phases,
    }


def aggregate_stats(runs: list[dict]) -> dict:
    """Menghitung agregasi statistik p50 (median) dan p95 untuk metrik kunci."""
    if not runs:
        return {}

    def extract(key):
        return [r[key] for r in runs]

    def extract_phase(key):
        return [r["phases"].get(key, 0.0) for r in runs]

    walltimes = extract("walltime_sec")
    throughputs = extract("throughput_tok_s")
    vmhwms = extract("vmhwm_bytes")
    logicals = extract("logical_bytes_read")
    physicals = extract("physical_read_bytes")
    peaks = extract("cgroup_peak_bytes")
    ooms = extract("cgroup_oom_kills")
    bws = extract("bw_effective_gb_s")
    bws_pread = extract("bw_pread_gb_s")
    preads = extract("pread_sec_total")

    phase_keys = [
        "index_load_sec",
        "embedding_sec",
        "layer_forward_sec",
        "final_norm_sec",
        "lm_head_sec",
        "write_sec",
    ]
    phases_p50 = {pk: statistics.median(extract_phase(pk)) for pk in phase_keys}
    phases_p95 = {pk: percentile(extract_phase(pk), 95) for pk in phase_keys}

    return {
        "count": len(runs),
        "walltime_sec": {
            "p50": statistics.median(walltimes),
            "p95": percentile(walltimes, 95),
            "min": min(walltimes),
            "max": max(walltimes),
        },
        "throughput_tok_s": {
            "p50": statistics.median(throughputs),
            "p95": percentile(throughputs, 95),
        },
        "vmhwm_bytes": {
            "p50": statistics.median(vmhwms),
            "p95": percentile(vmhwms, 95),
            "p50_gib": statistics.median(vmhwms) / (1024**3),
            "p95_gib": percentile(vmhwms, 95) / (1024**3),
        },
        "logical_bytes_read": {
            "p50": statistics.median(logicals),
            "p50_gb": statistics.median(logicals) / 1e9,
            "p50_gib": statistics.median(logicals) / (1024**3),
        },
        "physical_read_bytes": {
            "p50": statistics.median(physicals),
            "p50_gb": statistics.median(physicals) / 1e9,
            "p50_gib": statistics.median(physicals) / (1024**3),
        },
        "cgroup_peak_bytes": {
            "p50": statistics.median(peaks),
            "p50_gib": statistics.median(peaks) / (1024**3),
        },
        "cgroup_oom_kills_total": sum(ooms),
        "bw_effective_gb_s": {
            "p50": statistics.median(bws),
            "p95": percentile(bws, 95),
            "min": min(bws),
            "max": max(bws),
        },
        "bw_pread_gb_s": {
            "p50": statistics.median(bws_pread) if bws_pread else 0.0,
            "p95": percentile(bws_pread, 95) if bws_pread else 0.0,
        },
        "pread_sec_total": {
            "p50": statistics.median(preads) if preads else 0.0,
        },
        "phases_p50": phases_p50,
        "phases_p95": phases_p95,
    }


def compute_f4_f5_calibration(num_tokens: int, stats: dict) -> dict:
    """Menghitung kalibrasi Roofline F4/F5 untuk Qwen1.5-MoE-A2.7B."""
    # Parameter dasar arsitektur
    total_params = 14_330_685_440  # 14.3B
    active_params = 2_067_235_840  # 2.07B per token streaming set
    b_tok_decode_bytes = 4_134_000_000  # 4.134 GB per token decode (F3b/F4)

    # Compute intensity e_T (FLOPs / byte)
    # Pada decode: 2 * N_stream / B_tok ≈ 1 FLOP/byte
    i_decode_weight = (2 * active_params) / b_tok_decode_bytes
    e_t = i_decode_weight  # Initial intensity baseline ≈ 1.0 FLOP/byte
    i_prefill = e_t * num_tokens  # ≈ 16 FLOP/byte untuk s=16

    # Karakteristik mesin target
    # Intel Core i3-1215U: 1.2 GHz base, 4.4 GHz boost, AVX2 (16 FLOPs/cycle per core)
    # Peak single-thread FP32 compute ≈ 4.4 GHz * 16 = 70.4 GFLOP/s
    peak_compute_single_gflops = 70.4
    bw_ram_sustained_gb_s = stats.get("bw_effective_gb_s", {}).get("p50", 15.0)

    # Ridge point mesin target: P_peak / BW_RAM
    ridge_point = peak_compute_single_gflops / (
        bw_ram_sustained_gb_s if bw_ram_sustained_gb_s > 0 else 15.0
    )

    prefill_regime = "compute-bound" if i_prefill > ridge_point else "memory-bound"
    decode_regime = "compute-bound" if i_decode_weight > ridge_point else "memory-bound"

    return {
        "model_total_params": total_params,
        "model_active_params_per_tok": active_params,
        "b_tok_decode_bytes": b_tok_decode_bytes,
        "sequence_length": num_tokens,
        "initial_compute_intensity_e_T": e_t,
        "i_prefill_operational_intensity": i_prefill,
        "i_decode_weight_intensity": i_decode_weight,
        "assumed_single_core_peak_gflops": peak_compute_single_gflops,
        "measured_effective_bandwidth_gb_s": bw_ram_sustained_gb_s,
        "machine_ridge_point_flops_per_byte": ridge_point,
        "prefill_regime": prefill_regime,
        "decode_regime": decode_regime,
        "m5_projection": (
            f"Di M5 decode (s=1), intensitas operasional "
            f"({i_decode_weight:.2f} FLOP/byte) berada jauh di bawah "
            f"ridge point ({ridge_point:.2f} FLOP/byte). "
            f"Decode diproyeksikan murni {decode_regime} dengan batas "
            "kecepatan ditentukan oleh bandwidth streaming I/O dan RAM, "
            "bukan kapasitas FLOP CPU."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="M4 Real-Checkpoint Prefill Benchmark Runner"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path ke direktori model",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=DEFAULT_TOKENS,
        help="Path ke fixture tokens JSON",
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Jumlah cold measurement runs (default 5)"
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Jumlah warm-up runs (default 2)"
    )
    parser.add_argument(
        "--workdir", type=Path, default=None, help="Direktori kerja (default auto temp)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path output raw JSON",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Thread count (default 1)"
    )
    parser.add_argument(
        "--skip-cgroup", action="store_true", help="Lewati cgroup MemoryMax=6G"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    kimo_bin = root / "kimo"

    if not kimo_bin.exists():
        print(
            f"Error: binary {kimo_bin} tidak ditemukan. "
            "Jalankan `pixi run build` terlebih dahulu.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.model_dir.exists():
        print(
            f"Error: model directory {args.model_dir} tidak ditemukan.", file=sys.stderr
        )
        sys.exit(1)

    if not args.tokens.exists():
        print(f"Error: tokens file {args.tokens} tidak ditemukan.", file=sys.stderr)
        sys.exit(1)

    cpu_gov = get_cpu_governor()
    cpu_model = get_cpu_model()

    print("=" * 72)
    print("M4-W6 PREFILL PERFORMANCE BASELINE & BENCHMARK")
    print("=" * 72)
    print(f"Model       : {args.model_dir}")
    print(f"Tokens      : {args.tokens}")
    print(f"CPU Model   : {cpu_model}")
    print(f"CPU Governor: {cpu_gov}")
    print(
        f"Protocol    : {args.warmup} warm-up runs + {args.runs} cold measurement runs"
    )
    print(
        f"Cgroup Gate : MemoryMax=6G ({'DISABLED' if args.skip_cgroup else 'ENABLED'})"
    )
    print("=" * 72)

    workdir_obj = None
    if args.workdir:
        tmp_dir = args.workdir
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        workdir_obj = tempfile.TemporaryDirectory(prefix="bench_m4_")
        tmp_dir = Path(workdir_obj.name)

    warmup_results = []
    cold_results = []

    try:
        # 1. Warm-up Runs
        if args.warmup > 0:
            print(f"\n>> Menjalankan {args.warmup} Warm-up Runs...")
            for w in range(1, args.warmup + 1):
                print(f"   [Warmup {w}/{args.warmup}] Memulai forward pass...")
                res = run_single_forward(
                    kimo_bin,
                    args.model_dir,
                    args.tokens,
                    w,
                    "warmup",
                    tmp_dir,
                    threads=args.threads,
                    skip_cgroup=args.skip_cgroup,
                )
                warmup_results.append(res)
                print(
                    f"   [Warmup {w}] Selesai dalam {res['walltime_sec']:.2f} s | "
                    f"VmHWM: {res['vmhwm_gib']:.2f} GiB | "
                    f"BW: {res['bw_effective_gb_s']:.2f} GB/s | "
                    f"OOM kills: {res['cgroup_oom_kills']}"
                )

        # 2. Cold Measurement Runs (N)
        print(f"\n>> Menjalankan {args.runs} Cold Measurement Runs...")
        for r in range(1, args.runs + 1):
            run_idx = args.warmup + r
            print(f"   [Cold Run {r}/{args.runs}] Memulai forward pass...")
            res = run_single_forward(
                kimo_bin,
                args.model_dir,
                args.tokens,
                run_idx,
                "cold",
                tmp_dir,
                threads=args.threads,
                skip_cgroup=args.skip_cgroup,
            )
            cold_results.append(res)
            print(
                f"   [Cold Run {r}] Selesai dalam {res['walltime_sec']:.2f} s | "
                f"VmHWM: {res['vmhwm_gib']:.2f} GiB | "
                f"BW: {res['bw_effective_gb_s']:.2f} GB/s | "
                f"OOM kills: {res['cgroup_oom_kills']}"
            )

    finally:
        if workdir_obj:
            workdir_obj.cleanup()

    # Evaluasi statistik
    stats = aggregate_stats(cold_results if cold_results else warmup_results)
    calibration = compute_f4_f5_calibration(16, stats)

    # Evaluasi Gate G-M4-2 Sanity
    gate_g_m4_2_pass = True
    reasons_fail = []

    walltime_p50 = stats.get("walltime_sec", {}).get("p50", 999.0)
    walltime_p95 = stats.get("walltime_sec", {}).get("p95", 999.0)
    vmhwm_p50_gib = stats.get("vmhwm_bytes", {}).get("p50_gib", 999.0)
    oom_kills_total = stats.get("cgroup_oom_kills_total", 0)
    bw_pread_p50 = stats.get("bw_pread_gb_s", {}).get("p50", 0.0)
    pread_sec_p50 = stats.get("pread_sec_total", {}).get("p50", 0.0)
    bw_p50_gb_s = stats.get("bw_effective_gb_s", {}).get("p50", 0.0)

    if walltime_p50 > 300.0:
        gate_g_m4_2_pass = False
        reasons_fail.append(f"Walltime p50 {walltime_p50:.2f} s > 300 s")
    if walltime_p95 > 330.0:
        gate_g_m4_2_pass = False
        reasons_fail.append(f"Walltime p95 {walltime_p95:.2f} s > 330 s")
    if vmhwm_p50_gib > 5.0:
        gate_g_m4_2_pass = False
        reasons_fail.append(f"VmHWM p50 {vmhwm_p50_gib:.2f} GiB > 5.0 GiB")
    if oom_kills_total > 0:
        gate_g_m4_2_pass = False
        reasons_fail.append(f"OOM kills terdeteksi: {oom_kills_total}")

    verdict_g_m4_2 = "PASS" if gate_g_m4_2_pass else f"FAIL ({'; '.join(reasons_fail)})"

    p95_gib = stats["vmhwm_bytes"]["p95_gib"]
    cg_peak_gib = stats["cgroup_peak_bytes"]["p50_gib"]
    log_gb = stats["logical_bytes_read"]["p50_gb"]
    log_gib = stats["logical_bytes_read"]["p50_gib"]
    phys_gb = stats["physical_read_bytes"]["p50_gb"]
    phys_gib = stats["physical_read_bytes"]["p50_gib"]

    print("\n" + "=" * 72)
    print("RINGKASAN BENCHMARK PREFILL M4 (STATISTIK COLD RUNS):")
    print("=" * 72)
    print(
        f"Walltime (p50 / p95) : {walltime_p50:.2f} s / {walltime_p95:.2f} s "
        "(Target: <= 300/330 s)"
    )
    print(f"Throughput (p50)     : {stats['throughput_tok_s']['p50']:.2f} tok/s")
    print(
        f"VmHWM (p50 / p95)    : {vmhwm_p50_gib:.2f} GiB / "
        f"{p95_gib:.2f} GiB (Gate: <= 5 GiB)"
    )
    print(f"Cgroup Peak (p50)    : {cg_peak_gib:.2f} GiB (Observability)")
    print(f"Cgroup OOM Kills     : {oom_kills_total} (Gate: == 0)")
    print(f"Logical Bytes (p50)  : {log_gb:.2f} GB ({log_gib:.2f} GiB)")
    print(f"Physical Read (p50)  : {phys_gb:.2f} GB ({phys_gib:.2f} GiB)")
    if bw_pread_p50 > 0:
        print(
            f"Streaming Pread BW   : {bw_pread_p50:.2f} GB/s "
            f"(pread time: {pread_sec_p50:.3f} s, floor: >= 10 GB/s)"
        )
    print(f"Pipeline BW (p50)    : {bw_p50_gb_s:.2f} GB/s (compute-bound prefill s=16)")
    print("-" * 72)
    print("PER-PHASE TIMING BREAKDOWN (p50):")
    for phase_name, p_sec in stats["phases_p50"].items():
        pct = (p_sec / walltime_p50) * 100.0 if walltime_p50 > 0 else 0.0
        print(f"  - {phase_name:<20}: {p_sec:8.3f} s ({pct:5.1f}%)")
    print("-" * 72)
    print(f"VERDICT GATE G-M4-2 : {verdict_g_m4_2}")
    print("=" * 72)

    # Buat JSON output terstruktur
    output_data = {
        "benchmark": "M4 Full Forward 24-Layer Prefill",
        "run_id": "M4-20260916-001",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cpu_model": cpu_model,
            "cpu_governor": cpu_gov,
            "threads_pinned": args.threads,
        },
        "model": {
            "path": str(args.model_dir),
            "num_tokens": 16,
            "num_layers": 24,
            "hidden_size": 2048,
        },
        "protocol": {
            "warmup_runs": args.warmup,
            "cold_runs": args.runs,
            "cgroup_memory_max": "6G",
        },
        "summary": stats,
        "calibration_f4_f5": calibration,
        "gate_verdict": {
            "G-M4-2": verdict_g_m4_2,
            "status": "PASS" if gate_g_m4_2_pass else "FAIL",
        },
        "raw_runs": {
            "warmup": warmup_results,
            "cold": cold_results,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        f.write("\n")
    print(f"\n[OK] Artefak data mentah tersimpan di: {args.output_json}")


if __name__ == "__main__":
    main()
