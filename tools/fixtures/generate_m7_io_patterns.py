#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generator untuk M7 I/O Pattern Benchmark Fixture (m7_io_patterns.json).

Membangkitkan dua pola I/O normatif deterministik (F17):
1. Trunk Sequential (F17a): 100 blok x 4 MB, QD1, offset sekuensial selaras 4096.
2. Expert-Miss (F17b): 100 blok x 10 MB, random jump seed 42,
   pairwise non-overlapping.
"""

import hashlib
import json
import os
import random
import sys

DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/models/qwen1.5-moe-a2.7b-chat-4bit/quant_model.bin"
)
OUTPUT_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "m7_io_patterns.json")

# Ukuran blok standar
TRUNK_BLOCK_SIZE = 4 * 1024 * 1024  # 4 MB
EXPERT_BLOCK_SIZE = 10 * 1024 * 1024  # 10 MB
BLOCK_COUNT = 100
ALIGNMENT = 4096


def get_file_size(path: str) -> int:
    if os.path.exists(path):
        return os.path.getsize(path)
    # Default fallback size jika model belum di-download (7.382 GB)
    return 7382480468


def compute_offsets_sha256(offsets: list) -> str:
    canonical = json.dumps(offsets, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_trunk_sequential_offsets(file_size: int) -> list:
    offsets = []
    for i in range(BLOCK_COUNT):
        off = i * TRUNK_BLOCK_SIZE
        assert off % ALIGNMENT == 0, f"Offset {off} harus kelipatan {ALIGNMENT}"
        assert (
            off + TRUNK_BLOCK_SIZE <= file_size
        ), f"Offset {off} melebihi ukuran file {file_size}"
        offsets.append(off)
    return offsets


def generate_expert_miss_offsets(file_size: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    # Hitung jumlah slot 10 MB yang muat di file
    total_slots = file_size // EXPERT_BLOCK_SIZE
    assert (
        total_slots >= BLOCK_COUNT
    ), f"File size {file_size} tidak cukup untuk {BLOCK_COUNT} slot non-overlapping"

    # Pilih 100 slot unik yang pairwise non-overlapping
    selected_slots = rng.sample(range(total_slots), BLOCK_COUNT)

    offsets = []
    intervals = []
    for slot in selected_slots:
        # Align ke 4096
        raw_off = slot * EXPERT_BLOCK_SIZE
        aligned_off = (raw_off // ALIGNMENT) * ALIGNMENT
        assert (
            aligned_off + EXPERT_BLOCK_SIZE <= file_size
        ), f"Offset {aligned_off} melebihi ukuran file {file_size}"
        offsets.append(aligned_off)
        intervals.append((aligned_off, aligned_off + EXPERT_BLOCK_SIZE))

    # Verifikasi pairwise non-overlapping
    intervals.sort(key=lambda x: x[0])
    for i in range(len(intervals) - 1):
        assert (
            intervals[i][1] <= intervals[i + 1][0]
        ), f"Overlap terdeteksi antara {intervals[i]} dan {intervals[i+1]}"

    return offsets


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FIXTURE_PATH

    file_size = get_file_size(model_path)
    print(f"Target model: {model_path} ({file_size:,} bytes)")

    # 1. Generate trunk sequential
    trunk_offsets = generate_trunk_sequential_offsets(file_size)
    trunk_sha = compute_offsets_sha256(trunk_offsets)

    # 2. Generate expert-miss
    expert_offsets = generate_expert_miss_offsets(file_size, seed=42)
    expert_sha = compute_offsets_sha256(expert_offsets)

    fixture_data = {
        "name": "M7 I/O pattern benchmark",
        "description": "Workload for sequential trunk and expert-miss I/O patterns",
        "seed": 42,
        "model_file_size": file_size,
        "patterns": [
            {
                "id": "trunk_sequential",
                "name": "Sequential large blocks (trunk pattern F17a)",
                "block_size": TRUNK_BLOCK_SIZE,
                "block_count": BLOCK_COUNT,
                "pattern": "sequential",
                "queue_depth": 1,
                "offsets": trunk_offsets,
                "offsets_sha256": trunk_sha,
                "description": "Read 4 MB blocks sequentially (QD1) for G-M7-1",
            },
            {
                "id": "expert_miss",
                "name": "Expert-size blocks jumping (LRU-miss pattern F17b)",
                "block_size": EXPERT_BLOCK_SIZE,
                "block_count": BLOCK_COUNT,
                "pattern": "random_jump",
                "queue_depth": 16,
                "offsets": expert_offsets,
                "offsets_sha256": expert_sha,
                "description": (
                    "Read 10 MB blocks jumping between offsets (QD sweep) for" " G-M7-5"
                ),
            },
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fixture_data, f, indent=2)

    print(f"Fixture berhasil disimpan ke {output_path}")
    print(f"  Trunk sequential: {len(trunk_offsets)} offsets, SHA: {trunk_sha}")
    print(f"  Expert-miss:      {len(expert_offsets)} offsets, SHA: {expert_sha}")


if __name__ == "__main__":
    main()
