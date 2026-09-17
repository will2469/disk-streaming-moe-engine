#!/bin/bash
# ==============================================================================
# benchmark_io_patterns.sh — Benchmark Runner Dua Pola I/O O_DIRECT (F17)
#
# Mengukur:
# 1. BW_seq (Trunk Sequential 4 MB, QD1, median)
# 2. BW_exp(q) (Expert-Miss 10 MB, sweep QD 1..16, median)
# 3. R_io(q) = BW_exp(q) / BW_seq
# 4. q* (operational knee anti-rebound)
# 5. D_sus (degradasi burst 10% vs sustained 25%)
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_FILE="${1:-$HOME/models/qwen1.5-moe-a2.7b-chat-4bit/quant_model.bin}"
OUTPUT_JSON="${2:-m7_io_benchmark_summary.json}"
FIXTURE_PATH="tools/fixtures/m7_io_patterns.json"
QUICK_MODE="${QUICK_MODE:-0}"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model binary tidak ditemukan di $MODEL_FILE"
    exit 1
fi

if [ ! -f "$FIXTURE_PATH" ]; then
    echo ">> Membangkitkan fixture I/O deterministik m7_io_patterns.json..."
    python3 tools/fixtures/generate_m7_io_patterns.py "$MODEL_FILE" "$FIXTURE_PATH"
fi

# 1. Pastikan io_benchmark terkompilasi
if [ ! -f "./io_benchmark" ]; then
    echo ">> Mengompilasi io_benchmark binary..."
    pixi run bash -c 'PATH="/usr/bin:$PATH" mojo build -I src tools/bench/io_benchmark.mojo -o io_benchmark'
fi

WORKDIR=$(mktemp -d /tmp/m7_bench_io_XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT

echo "======================================================================"
echo "M7 Benchmark I/O Patterns (F17a Trunk Sequential & F17b Expert-Miss)"
echo "Target Model: $MODEL_FILE"
echo "Fixture:      $FIXTURE_PATH"
echo "======================================================================"

# ----------------------------------------------------------------------
# Helper Python untuk agregasi median dan perhitungan metrik
# ----------------------------------------------------------------------
CALC_SCRIPT="$WORKDIR/calc_metrics.py"
cat << 'EOF' > "$CALC_SCRIPT"
import sys, json, statistics

mode = sys.argv[1]

if mode == "median":
    values = [float(x) for x in sys.argv[2:] if x]
    if not values:
        print("0.0")
    else:
        print(f"{statistics.median(values):.6f}")

elif mode == "evaluate_all":
    trunk_file = sys.argv[2]
    qd_results_file = sys.argv[3]
    out_file = sys.argv[4]

    with open(trunk_file) as f:
        trunk_runs = json.load(f)
    with open(qd_results_file) as f:
        qd_runs = json.load(f)

    bw_seq = statistics.median([r["bandwidth_gb_s"] for r in trunk_runs])
    d_sus_seq = statistics.median([r["d_sus"] for r in trunk_runs])
    dio_align = trunk_runs[0].get("dio_alignment", 512)

    bw_exp_by_qd = {}
    r_io_by_qd = {}
    max_obs_by_qd = {}

    qd_levels = [1, 2, 4, 8, 16]
    for q in qd_levels:
        q_str = str(q)
        if q_str in qd_runs:
            runs = qd_runs[q_str]
            med_bw = statistics.median([r["bandwidth_gb_s"] for r in runs])
            bw_exp_by_qd[q_str] = round(med_bw, 6)
            r_io_by_qd[q_str] = round(med_bw / bw_seq if bw_seq > 0 else 0.0, 6)
            max_obs_by_qd[q_str] = max([r["max_outstanding_observed"] for r in runs])

    # Hitung q* (operational knee anti-rebound):
    # q* = q pertama dengan marginal(q) < 0.10 DAN 2 titik berikut tanpa rebound > 10%
    q_star = qd_levels[-1]
    for idx in range(1, len(qd_levels)):
        q = qd_levels[idx]
        p = qd_levels[idx - 1]
        bw_q = bw_exp_by_qd.get(str(q), 0.0)
        bw_p = bw_exp_by_qd.get(str(p), 0.0)
        if bw_p > 0:
            marginal = (bw_q - bw_p) / bw_p
            if marginal < 0.10:
                # Cek anti-rebound pada titik-titik berikutnya
                rebound_detected = False
                for r_idx in range(idx + 1, min(idx + 3, len(qd_levels))):
                    rq = qd_levels[r_idx]
                    bw_r = bw_exp_by_qd.get(str(rq), 0.0)
                    if bw_r - bw_q > 0.10 * bw_q:
                        rebound_detected = True
                        break
                if not rebound_detected:
                    q_star = q
                    break

    summary = {
        "status": "success",
        "model_file": trunk_runs[0]["file"],
        "dio_alignment": dio_align,
        "metrics": {
            "bw_seq_gb_s": round(bw_seq, 6),
            "bw_exp_by_qd": bw_exp_by_qd,
            "r_io_by_qd": r_io_by_qd,
            "max_outstanding_observed_by_qd": max_obs_by_qd,
            "q_star": q_star,
            "d_sus": round(d_sus_seq, 6)
        }
    }

    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
EOF

# ----------------------------------------------------------------------
# 1. Trunk Sequential Pattern (F17a)
# ----------------------------------------------------------------------
echo ">> [1/2] Mengukur Trunk Sequential Pattern (4 MB, QD1)..."
TRUNK_RUNS_FILE="$WORKDIR/trunk_runs.json"
echo "[" > "$TRUNK_RUNS_FILE"

N_TRUNK_RUNS=5
N_BLOCKS_TRUNK=100
if [ "$QUICK_MODE" = "1" ]; then
    N_TRUNK_RUNS=2
    N_BLOCKS_TRUNK=20
fi

for run in $(seq 1 $N_TRUNK_RUNS); do
    RUN_OUT="$WORKDIR/trunk_run_${run}.json"
    ./io_benchmark \
      --pattern sequential \
      --block-size 4194304 \
      --block-count $N_BLOCKS_TRUNK \
      --queue-depth 1 \
      --file "$MODEL_FILE" \
      --offsets-fixture "$FIXTURE_PATH" \
      --output "$RUN_OUT" > /dev/null

    if [ "$run" -gt 1 ]; then echo "," >> "$TRUNK_RUNS_FILE"; fi
    cat "$RUN_OUT" >> "$TRUNK_RUNS_FILE"
    BW=$(python3 -c "import json; print(json.load(open('$RUN_OUT'))['bandwidth_gb_s'])")
    echo "   Run $run: $BW GB/s"
done
echo "]" >> "$TRUNK_RUNS_FILE"

# ----------------------------------------------------------------------
# 2. Expert-Miss Pattern Sweep QD (F17b)
# ----------------------------------------------------------------------
echo ">> [2/2] Mengukur Expert-Miss Pattern (10 MB, Sweep QD {1, 2, 4, 8, 16})..."
QD_RUNS_FILE="$WORKDIR/qd_runs.json"
echo "{" > "$QD_RUNS_FILE"

QD_LEVELS=(1 2 4 8 16)
FIRST_QD=1

N_EXPERT_RUNS=5
N_BLOCKS_EXPERT=100
if [ "$QUICK_MODE" = "1" ]; then
    N_EXPERT_RUNS=2
    N_BLOCKS_EXPERT=15
    QD_LEVELS=(1 4 16)
fi

for q in "${QD_LEVELS[@]}"; do
    echo "   >> Sweep QD = $q"
    if [ "$FIRST_QD" -eq 0 ]; then echo "," >> "$QD_RUNS_FILE"; fi
    FIRST_QD=0
    echo "\"$q\": [" >> "$QD_RUNS_FILE"

    for run in $(seq 1 $N_EXPERT_RUNS); do
        RUN_OUT="$WORKDIR/expert_qd${q}_run_${run}.json"
        ./io_benchmark \
          --pattern random_jump \
          --block-size 10485760 \
          --block-count $N_BLOCKS_EXPERT \
          --queue-depth "$q" \
          --file "$MODEL_FILE" \
          --offsets-fixture "$FIXTURE_PATH" \
          --output "$RUN_OUT" > /dev/null

        if [ "$run" -gt 1 ]; then echo "," >> "$QD_RUNS_FILE"; fi
        cat "$RUN_OUT" >> "$QD_RUNS_FILE"
        BW=$(python3 -c "import json; print(json.load(open('$RUN_OUT'))['bandwidth_gb_s'])")
        OBS=$(python3 -c "import json; print(json.load(open('$RUN_OUT'))['max_outstanding_observed'])")
        echo "      Run $run (QD $q): $BW GB/s (max_obs: $OBS)"
    done
    echo "]" >> "$QD_RUNS_FILE"
done
echo "}" >> "$QD_RUNS_FILE"

# ----------------------------------------------------------------------
# 3. Hitung Metrik Lengkap & Output Summary
# ----------------------------------------------------------------------
echo "======================================================================"
echo "HASIL RINGKASAN BENCHMARK I/O M7 (F17)"
echo "======================================================================"
python3 "$CALC_SCRIPT" evaluate_all "$TRUNK_RUNS_FILE" "$QD_RUNS_FILE" "$OUTPUT_JSON"

echo "Hasil tersimpan di $OUTPUT_JSON"
