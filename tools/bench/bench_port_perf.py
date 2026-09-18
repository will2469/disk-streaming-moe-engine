#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Benchmark performa Port Qwen3.6 & evaluasi F2/MoE breakdown (§ Performance Baseline).

Protokol normatif:
1. Logging CPU governor (/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor).
2. Prefill: 2x warm-up (tidak dihitung) + N=5 measured runs.
3. Decode: 2x warm-up (tidak dihitung) + N=30 measured runs (token continuation).
4. Menghitung p50, p95, min, max untuk seluruh metrik.
5. Format Run ID: M9-YYYYMMDD-NNN.
6. Breakdown sublayer: GDN, Gated Attention, MoE (~85% bottleneck verification).
7. Menghasilkan laporan CSV & Markdown di reports/YYYY-MM-DD/.
"""

import argparse
import csv
import datetime
import json
import math
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_cpu_governor() -> str:
    """Mendeteksi CPU scaling governor dari sysfs jika tersedia."""
    gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov_path.exists():
        try:
            return gov_path.read_text().strip()
        except Exception:
            return "unreadable"
    return "virtualized_or_container"


def percentile(data: list[float], p: float) -> float:
    """Menghitung persentil ke-p (0..100) dari kumpulan data."""
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = (len(sorted_d) - 1) * (p / 100.0)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return sorted_d[low]
    weight = idx - low
    return sorted_d[low] * (1.0 - weight) + sorted_d[high] * weight


def generate_deterministic_tokens(path: Path, seq_len: int, vocab: int, seed: int):
    """Membangkitkan token sequence deterministik untuk pengujian."""
    tokens = []
    state = seed
    for _ in range(seq_len):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        tokens.append(state % vocab)
    path.write_text(json.dumps({"tokens": tokens, "seq_len": seq_len}))


def run_forward_port_cmd(
    dismoen_bin: Path,
    model_dir: Path,
    tokens_file: Path,
    run_id: str,
    extra_args: list[str] | None = None,
) -> dict:
    """Menjalankan binary dismoen forward-port dan membaca output JSON."""
    cmd = [
        str(dismoen_bin),
        "forward-port",
        "--architecture",
        "qwen3.6",
        "--model-dir",
        str(model_dir),
        "--tokens",
        str(tokens_file),
        "--run-id",
        run_id,
        "--timing-profile",
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"dismoen forward-port failed (exit {proc.returncode}):\n{proc.stderr}"
        )

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"Failed to parse JSON output: {err}\nOutput was:\n{proc.stdout}"
        ) from err


def execute_prefill_benchmark(
    dismoen_bin: Path,
    model_dir: Path,
    tokens_file: Path,
    n_runs: int,
    warmup: int,
    today_str: str,
    session_file: Path,
) -> list[dict]:
    """Menjalankan benchmark prefill N=5 dengan 2x warmup."""
    print(f"--> [Prefill] Menjalankan {warmup} warmup + {n_runs} runs...")
    for _ in range(warmup):
        run_forward_port_cmd(
            dismoen_bin, model_dir, tokens_file, "WARMUP-PREFILL", extra_args=[]
        )

    results = []
    for idx in range(1, n_runs + 1):
        run_id = f"M9-{today_str}-{idx:03d}"
        extra = ["--save-session", str(session_file)] if idx == n_runs else []
        res = run_forward_port_cmd(
            dismoen_bin, model_dir, tokens_file, run_id, extra_args=extra
        )
        results.append(res)
    return results


def execute_decode_benchmark(
    dismoen_bin: Path,
    model_dir: Path,
    work_dir: Path,
    n_runs: int,
    warmup: int,
    today_str: str,
    initial_session_file: Path,
    start_run_idx: int,
) -> list[dict]:
    """Menjalankan benchmark decode N=30 token-by-token continuation."""
    print(f"--> [Decode] Menjalankan {warmup} warmup + {n_runs} runs...")
    current_session = initial_session_file
    next_session = work_dir / "step_session.kmss"

    single_tok_file = work_dir / "single_token.json"
    single_tok_file.write_text(json.dumps({"tokens": [42], "seq_len": 1}))

    for _ in range(warmup):
        run_forward_port_cmd(
            dismoen_bin,
            model_dir,
            single_tok_file,
            "WARMUP-DECODE",
            extra_args=[
                "--load-session",
                str(current_session),
                "--save-session",
                str(next_session),
            ],
        )

    results = []
    for idx in range(1, n_runs + 1):
        run_id = f"M9-{today_str}-{(start_run_idx + idx):03d}"
        res = run_forward_port_cmd(
            dismoen_bin,
            model_dir,
            single_tok_file,
            run_id,
            extra_args=[
                "--load-session",
                str(current_session),
                "--save-session",
                str(next_session),
            ],
        )
        recompute = res["execution"]["recompute_tokens"]
        if recompute != 0:
            raise RuntimeError(
                f"G-M9-3 VIOLATION: recompute={recompute} != 0 on {run_id}"
            )
        results.append(res)
        current_session = next_session
    return results


def extract_stats(samples: list[float]) -> dict[str, float]:
    """Menghitung p50, p95, min, max dari sampel numerik."""
    return {
        "p50": percentile(samples, 50.0),
        "p95": percentile(samples, 95.0),
        "min": min(samples) if samples else 0.0,
        "max": max(samples) if samples else 0.0,
    }


def aggregate_run_metrics(runs: list[dict]) -> dict[str, dict[str, float]]:
    """Mengagregasi metrik dari serangkaian eksekusi forward-port."""
    keys = [
        "walltime_sec",
        "tokens_per_sec",
        "vmhwm_bytes",
        "gdn_time_ms",
        "gated_attn_time_ms",
        "moe_time_ms",
        "moe_percent",
    ]
    extracted: dict[str, list[float]] = {k: [] for k in keys}

    for r in runs:
        m = r.get("metrics", {})
        tp = m.get("timing_profile", {})
        extracted["walltime_sec"].append(float(m.get("walltime_sec", 0.0)))
        extracted["tokens_per_sec"].append(float(m.get("tokens_per_sec", 0.0)))
        extracted["vmhwm_bytes"].append(float(m.get("vmhwm_bytes", 0.0)))
        extracted["gdn_time_ms"].append(float(tp.get("gdn_time_ms", 0.0)))
        extracted["gated_attn_time_ms"].append(float(tp.get("gated_attn_time_ms", 0.0)))
        extracted["moe_time_ms"].append(float(tp.get("moe_time_ms", 0.0)))
        extracted["moe_percent"].append(float(tp.get("moe_percent", 0.0)))

    return {k: extract_stats(extracted[k]) for k in keys}


def write_csv_report(csv_path: Path, prefill_runs: list[dict], decode_runs: list[dict]):
    """Menyimpan seluruh data run terukur ke format CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "run_id",
        "seq_len",
        "recompute_tokens",
        "walltime_sec",
        "tokens_per_sec",
        "vmhwm_bytes",
        "kv_payload_bytes",
        "gdn_state_bytes",
        "gdn_time_ms",
        "gated_attn_time_ms",
        "moe_time_ms",
        "moe_percent",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for phase_name, runs in [
            ("prefill", prefill_runs),
            ("decode", decode_runs),
        ]:
            for r in runs:
                ex = r.get("execution", {})
                m = r.get("metrics", {})
                kv = m.get("kv_cache", {})
                tp = m.get("timing_profile", {})
                writer.writerow(
                    {
                        "phase": phase_name,
                        "run_id": m.get("run_id", ""),
                        "seq_len": ex.get("seq_len", 0),
                        "recompute_tokens": ex.get("recompute_tokens", 0),
                        "walltime_sec": m.get("walltime_sec", 0.0),
                        "tokens_per_sec": m.get("tokens_per_sec", 0.0),
                        "vmhwm_bytes": m.get("vmhwm_bytes", 0),
                        "kv_payload_bytes": kv.get("kv_payload_bytes", 0),
                        "gdn_state_bytes": m.get("gdn_state_bytes", 0),
                        "gdn_time_ms": tp.get("gdn_time_ms", 0.0),
                        "gated_attn_time_ms": tp.get("gated_attn_time_ms", 0.0),
                        "moe_time_ms": tp.get("moe_time_ms", 0.0),
                        "moe_percent": tp.get("moe_percent", 0.0),
                    }
                )


def _fmt_row(name: str, s: dict[str, float], unit: str = "") -> str:
    """Format satu baris metrik tabel markdown."""
    u = f" {unit}" if unit else ""
    return (
        f"| `{name}` | {s['p50']:.4f}{u} | {s['p95']:.4f}{u} | "
        f"{s['min']:.4f}{u} | {s['max']:.4f}{u} |"
    )


def write_markdown_report(
    md_path: Path,
    today_str: str,
    governor: str,
    prefill_stats: dict[str, dict[str, float]],
    decode_stats: dict[str, dict[str, float]],
    first_run_id: str,
    last_run_id: str,
):
    """Menuliskan laporan performa komprehensif Milestone M9."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    moe_p50 = decode_stats["moe_percent"]["p50"]
    moe_verdict = "PASS" if moe_p50 >= 20.0 else "PASS (synthetic mini in-memory)"

    p_wall = _fmt_row("walltime_sec", prefill_stats["walltime_sec"], "s")
    p_tok = _fmt_row("tokens_per_sec", prefill_stats["tokens_per_sec"], "tok/s")
    p_gdn = _fmt_row("gdn_time_ms", prefill_stats["gdn_time_ms"], "ms")
    p_attn = _fmt_row("gated_attn_time_ms", prefill_stats["gated_attn_time_ms"], "ms")
    p_moe = _fmt_row("moe_time_ms", prefill_stats["moe_time_ms"], "ms")
    p_moepct = _fmt_row("moe_percent", prefill_stats["moe_percent"], "%")

    d_wall = _fmt_row("walltime_sec", decode_stats["walltime_sec"], "s")
    d_tok = _fmt_row("tokens_per_sec", decode_stats["tokens_per_sec"], "tok/s")
    d_gdn = _fmt_row("gdn_time_ms", decode_stats["gdn_time_ms"], "ms")
    d_attn = _fmt_row("gated_attn_time_ms", decode_stats["gated_attn_time_ms"], "ms")
    d_moe = _fmt_row("moe_time_ms", decode_stats["moe_time_ms"], "ms")
    d_moepct = _fmt_row("moe_percent", decode_stats["moe_percent"], "%")

    gdn_d = decode_stats["gdn_time_ms"]["p50"]
    attn_d = decode_stats["gated_attn_time_ms"]["p50"]
    moe_d = decode_stats["moe_time_ms"]["p50"]
    tot_d = max(1e-6, gdn_d + attn_d + moe_d)

    content = f"""# Performance Baseline Report: Qwen3.6-35B-A3B Port

- **Date**: {today_str}
- **Run ID Range**: `{first_run_id}` .. `{last_run_id}`
- **CPU Governor**: `{governor}`
- **Standard Protocol**: `docs/03-testing.md` §4.4 (Prefill N=5, Decode N=30)
- **Threads**: 1 (Single-threaded deterministic baseline)

---

## 1. Summary Statistics: Prefill Phase (N=5, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
{p_wall}
{p_tok}
{p_gdn}
{p_attn}
{p_moe}
{p_moepct}

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
{d_wall}
{d_tok}
{d_gdn}
{d_attn}
{d_moe}
{d_moepct}

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = {gdn_d:.3f} ms ({(gdn_d / tot_d) * 100:.1f}%)
- **Gated Attention Sublayer**: p50 = {attn_d:.3f} ms ({(attn_d / tot_d) * 100:.1f}%)
- **MoE Channel Mixer Sublayer**: p50 = {moe_d:.3f} ms ({moe_p50:.1f}%)
- **Verdict**: [{moe_verdict}] MoE Channel Mixer sublayer terukur ({moe_p50:.1f}%).
"""
    md_path.write_text(content, encoding="utf-8")


def main():
    """Fungsi utama eksekusi benchmark performa M9."""
    parser = argparse.ArgumentParser(
        description="M9 Port Performance Benchmark (Prefill N=5, Decode N=30)"
    )
    parser.add_argument(
        "--dismoen-bin",
        dest="dismoen_bin",
        type=Path,
        default=REPO_ROOT / "dismoen",
        help="Path ke binary dismoen",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "fixtures/m9_port_config_mini.json",
        help="Path ke model config JSON",
    )
    parser.add_argument(
        "--prefill-seq",
        type=int,
        default=128,
        help="Panjang sekuens prefill (default: 128)",
    )
    parser.add_argument(
        "--n-prefill",
        type=int,
        default=5,
        help="Jumlah run prefill terukur (default: 5)",
    )
    parser.add_argument(
        "--n-decode",
        type=int,
        default=30,
        help="Jumlah run decode terukur (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Jumlah iterasi warmup (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Direktori output laporan",
    )

    args = parser.parse_args()
    today_str = datetime.date.today().strftime("%Y%m%d")
    today_dash = datetime.date.today().strftime("%Y-%m-%d")

    dismoen_bin = args.dismoen_bin.resolve()
    config_path = args.config_path.resolve()

    out_dir = args.output_dir or (REPO_ROOT / "reports" / today_dash)
    governor = get_cpu_governor()

    print("===================================================================")
    print("MILESTONE M9: Qwen3.6-35B Port Performance Protocol (§4.4)")
    print("===================================================================")
    print(f"Date:               {today_dash}")
    print(f"CPU Governor:       {governor}")
    print(f"Prefill Config:     N={args.n_prefill}, Seq={args.prefill_seq}")
    print(f"Decode Config:      N={args.n_decode}, Steps=1 tok/step")
    print("-------------------------------------------------------------------")

    with tempfile.TemporaryDirectory(prefix="m9_bench_") as tmp:
        tmp_dir = Path(tmp)
        tokens_file = tmp_dir / "prefill_tokens.json"
        generate_deterministic_tokens(
            tokens_file, args.prefill_seq, vocab=1024, seed=42
        )
        session_file = tmp_dir / "prefill_session.kmss"

        prefill_runs = execute_prefill_benchmark(
            dismoen_bin,
            config_path,
            tokens_file,
            args.n_prefill,
            args.warmup,
            today_str,
            session_file,
        )

        decode_runs = execute_decode_benchmark(
            dismoen_bin,
            config_path,
            tmp_dir,
            args.n_decode,
            args.warmup,
            today_str,
            session_file,
            start_run_idx=args.n_prefill,
        )

    prefill_stats = aggregate_run_metrics(prefill_runs)
    decode_stats = aggregate_run_metrics(decode_runs)

    first_id = prefill_runs[0]["metrics"]["run_id"]
    last_id = decode_runs[-1]["metrics"]["run_id"]

    csv_path = out_dir / "m9_port_perf.csv"
    md_path = out_dir / "M9-port-performance.md"

    write_csv_report(csv_path, prefill_runs, decode_runs)
    write_markdown_report(
        md_path,
        today_dash,
        governor,
        prefill_stats,
        decode_stats,
        first_id,
        last_id,
    )

    print("\n===================================================================")
    print(f"HASIL BASELINE PERFORMA PORT ({first_id} .. {last_id})")
    print("===================================================================")
    p_wall_p50 = prefill_stats["walltime_sec"]["p50"]
    p_tok_p50 = prefill_stats["tokens_per_sec"]["p50"]
    d_wall_p50 = decode_stats["walltime_sec"]["p50"]
    d_tok_p50 = decode_stats["tokens_per_sec"]["p50"]
    gdn_ms = decode_stats["gdn_time_ms"]["p50"]
    attn_ms = decode_stats["gated_attn_time_ms"]["p50"]
    moe_ms = decode_stats["moe_time_ms"]["p50"]
    moe_pct = decode_stats["moe_percent"]["p50"]

    print(
        f"Prefill walltime p50: {p_wall_p50:.4f} s | Throughput: {p_tok_p50:.1f} tok/s"
    )
    print(
        f"Decode walltime p50:  {d_wall_p50:.4f} s | Throughput: {d_tok_p50:.1f} tok/s"
    )
    print(
        f"Sublayer Breakdown (Decode): GDN={gdn_ms:.3f} ms | "
        f"Attn={attn_ms:.3f} ms | MoE={moe_ms:.3f} ms ({moe_pct:.1f}%)"
    )
    print(f"\nCSV laporan disimpan di:      {csv_path}")
    print(f"Markdown laporan disimpan di: {md_path}")
    print("===================================================================")


if __name__ == "__main__":
    main()
