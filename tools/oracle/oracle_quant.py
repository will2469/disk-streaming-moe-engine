#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle Quantization Roundtrip & Error Evaluation (M6-W5).

Menjalankan quantization roundtrip BF16 -> 4-bit -> FP32 untuk verifikasi G-M6-1:
1. Validasi group-size dalam {32, 64, 128, 256}.
2. Penegakan tail group N % G == 0 (M6_ERR_INPUT, exit 1).
3. Evaluasi kuantisasi F11a per tensor dengan tie-break RNE eksplisit fp32.
4. Verifikasi properti Q-domain (|w - w_hat^(32)| <= s_g / 2).
5. Perhitungan epsilon_rel & penanganan tensor variansi-nol via jalur absolut.
6. Pelaporan JSON RFC 8259 terstruktur (max/avg/min epsilon_rel).
"""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from tools.quant.quant_algo import (
    compute_quant_metrics,
    dequantize_tensor_q32,
    quantize_tensor_f11a,
    verify_qdomain_property,
)
from tools.quant.quant_format import (
    QUANT_ALLOWED_GROUP_SIZES,
    QUANT_DEFAULT_GROUP_SIZE,
)

torch.manual_seed(42)


def fail(
    error_code: str,
    stage: str,
    message: str,
    details: dict[str, Any] | None = None,
    exit_code: int = 1,
) -> None:
    """Mengeluarkan error payload strict RFC 8259 ke stderr dan exit."""
    payload = {
        "status": "error",
        "error": {
            "code": error_code,
            "stage": stage,
            "message": message,
            "details": details or {},
        },
    }
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.exit(exit_code)


def discover_safetensors(model_dir: str) -> list[tuple[str, str]]:
    """Menemukan semua tensor dari index.json atau safetensors tunggal.

    Mengembalikan list of (tensor_name, shard_file_path).
    """
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        fail(
            "M6_ERR_INPUT",
            "input",
            f"input directory does not exist or is not a directory: {model_dir}",
            exit_code=1,
        )

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            weight_map = idx_data.get("weight_map", {})
        except Exception as e:
            fail(
                "M6_ERR_INPUT",
                "input",
                f"failed to parse index file {index_path}: {e}",
                exit_code=1,
            )

        tensors = []
        for name, rel_shard in weight_map.items():
            shard_path = os.path.join(model_dir, rel_shard)
            if not os.path.exists(shard_path):
                fail(
                    "M6_ERR_INPUT",
                    "input",
                    f"shard file referenced in index does not exist: {shard_path}",
                    exit_code=1,
                )
            tensors.append((name, shard_path))
        return sorted(tensors, key=lambda x: x[0])

    st_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if not st_files:
        fail(
            "M6_ERR_INPUT",
            "input",
            f"no safetensors files found in {model_dir}",
            exit_code=1,
        )

    st_files.sort()
    tensors = []
    for sf in st_files:
        full_path = os.path.join(model_dir, sf)
        try:
            with safe_open(full_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensors.append((name, full_path))
        except Exception as e:
            fail(
                "M6_ERR_INPUT",
                "input",
                f"failed reading keys from {full_path}: {e}",
                exit_code=1,
            )

    return sorted(tensors, key=lambda x: x[0])


def quantize_and_evaluate_tensor(
    name: str,
    raw_t: torch.Tensor,
    group_size: int,
) -> dict[str, Any]:
    """Kuantisasi satu tensor, evaluasi properti Q-domain, dan hitung metrik."""
    shape = list(raw_t.shape)
    t_f32 = raw_t.float().contiguous().view(-1)
    w_np = t_f32.numpy()
    w_list = [float(x) for x in w_np]
    n_elem = len(w_list)

    if n_elem % group_size != 0:
        fail(
            "M6_ERR_INPUT",
            "input",
            f"tail group on {name}: N={n_elem} % G={group_size} != 0",
            details={
                "tensor_name": name,
                "num_elements": n_elem,
                "group_size": group_size,
            },
            exit_code=1,
        )

    if not np.isfinite(w_np).all():
        fail(
            "M6_ERR_QUANT",
            "quantization",
            f"non-finite values encountered in tensor {name}",
            details={"tensor_name": name},
            exit_code=2,
        )

    try:
        scales, q_weights, _packed = quantize_tensor_f11a(w_list, group_size)
        q32_weights = dequantize_tensor_q32(scales, q_weights, group_size)
    except Exception as e:
        fail(
            "M6_ERR_QUANT",
            "quantization",
            f"quantization failure for tensor {name}: {e}",
            details={"tensor_name": name},
            exit_code=2,
        )

    property_ok, max_abs_err = verify_qdomain_property(
        w_list, q32_weights, scales, group_size
    )
    m = compute_quant_metrics(w_list, q32_weights)

    is_zero_var = m["zero_variance"]
    eps_rel = m["epsilon_rel"]
    mse_val = m["mse"]

    record: dict[str, Any] = {
        "name": name,
        "shape": shape,
        "epsilon_rel": eps_rel,
        "zero_variance": is_zero_var,
        "mse": round(mse_val, 8),
        "property_ok": property_ok,
        "max_abs_error": round(max_abs_err, 8),
        "num_groups": len(scales),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle Quantization Roundtrip & Error Evaluator (M6-W5)"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input model directory containing BF16 safetensors",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=QUANT_DEFAULT_GROUP_SIZE,
        help="Quantization group size G in {32, 64, 128, 256} (default: 128)",
    )
    parser.add_argument(
        "--output-report",
        required=True,
        help="Path to output JSON report",
    )
    args = parser.parse_args()

    group_size = args.group_size
    if group_size not in QUANT_ALLOWED_GROUP_SIZES:
        allowed = sorted(QUANT_ALLOWED_GROUP_SIZES)
        fail(
            "M6_ERR_INPUT",
            "input",
            f"group-size must be in {allowed}, got {group_size}",
            exit_code=1,
        )

    tensor_entries = discover_safetensors(args.input_dir)
    if not tensor_entries:
        fail("M6_ERR_INPUT", "input", "no tensors discovered", exit_code=1)

    shard_handles: dict[str, Any] = {}
    tensor_results: list[dict[str, Any]] = []

    sum_epsilon_rel = 0.0
    count_variance = 0
    max_epsilon_rel = 0.0
    min_epsilon_rel = float("inf")
    num_high_error = 0

    for name, shard_path in tensor_entries:
        if shard_path not in shard_handles:
            try:
                shard_handles[shard_path] = safe_open(
                    shard_path, framework="pt", device="cpu"
                )
            except Exception as e:
                fail(
                    "M6_ERR_INPUT",
                    "input",
                    f"failed to open shard {shard_path}: {e}",
                    exit_code=1,
                )

        shard = shard_handles[shard_path]
        try:
            raw_t = shard.get_tensor(name)
        except Exception as e:
            fail(
                "M6_ERR_INPUT",
                "input",
                f"failed to read tensor {name} from {shard_path}: {e}",
                exit_code=1,
            )

        record = quantize_and_evaluate_tensor(name, raw_t, group_size)
        eps_rel = record["epsilon_rel"]
        if not record["zero_variance"] and eps_rel is not None:
            if eps_rel > max_epsilon_rel:
                max_epsilon_rel = eps_rel
            if eps_rel < min_epsilon_rel:
                min_epsilon_rel = eps_rel
            sum_epsilon_rel += eps_rel
            count_variance += 1
            if eps_rel > 0.01:
                num_high_error += 1

        tensor_results.append(record)

    avg_epsilon_rel = sum_epsilon_rel / count_variance if count_variance > 0 else 0.0
    if min_epsilon_rel == float("inf"):
        min_epsilon_rel = 0.0

    report_payload = {
        "run_id": "M6-QUANT-ORACLE",
        "model": "qwen1.5-moe-a2.7b-chat",
        "group_size": group_size,
        "num_tensors": len(tensor_results),
        "results": {
            "max_epsilon_rel": round(max_epsilon_rel, 6),
            "avg_epsilon_rel": round(avg_epsilon_rel, 6),
            "min_epsilon_rel": round(min_epsilon_rel, 6),
            "tensors": tensor_results,
        },
        "summary": {
            "num_tensors": len(tensor_results),
            "max_epsilon_rel": round(max_epsilon_rel, 6),
            "avg_epsilon_rel": round(avg_epsilon_rel, 6),
            "min_epsilon_rel": round(min_epsilon_rel, 6),
            "num_high_error": num_high_error,
        },
    }

    out_dir = os.path.dirname(os.path.abspath(args.output_report))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
            f.write("\n")
    except Exception as e:
        fail(
            "M6_ERR_OUTPUT",
            "output",
            f"failed to write output report to {args.output_report}: {e}",
            exit_code=1,
        )

    print(
        json.dumps(
            {
                "status": "success",
                "run_id": "M6-QUANT-ORACLE",
                "num_tensors": len(tensor_results),
                "max_epsilon_rel": round(max_epsilon_rel, 6),
                "avg_epsilon_rel": round(avg_epsilon_rel, 6),
                "num_high_error": num_high_error,
            }
        )
    )


if __name__ == "__main__":
    main()
