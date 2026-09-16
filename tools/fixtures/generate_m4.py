#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generator untuk dataset M4 Golden Prompts Set.

Memproduksi tools/fixtures/m4_golden.json dan individual tokens JSON
menggunakan tokenizer asli Qwen1.5-MoE (fail-closed, strictly generated).
"""

import argparse
import hashlib
import json
import os
import sys

from tokenizers import Tokenizer

VOCAB_SIZE_GATE = 151936

REPRESENTATIVE_PROMPTS = [
    {
        "id": "prompt1",
        "category": "factual",
        "text": (
            "What is the capital of France and what are the most famous"
            " historical landmarks in Paris?"
        ),
    },
    {
        "id": "prompt2",
        "category": "technical",
        "text": (
            "Explain quantum computing briefly, including qubits,"
            " superposition, and quantum entanglement in modern physics"
            " systems."
        ),
    },
    {
        "id": "prompt3",
        "category": "code",
        "text": (
            "Write a Python function to sort a list of numbers using the"
            " quicksort algorithm efficiently."
        ),
    },
    {
        "id": "prompt4",
        "category": "basic_knowledge",
        "text": (
            "What are the primary colors and how do they combine to form"
            " secondary colors in painting?"
        ),
    },
    {
        "id": "prompt5",
        "category": "summary",
        "text": (
            "Summarize the history of the internet from the early ARPANET"
            " experiments to the modern World Wide Web."
        ),
    },
]


def fail(msg: str) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate M4 Golden Prompt Set and Token Fixtures"
    )
    parser.add_argument(
        "--model-dir",
        default="/home/will/models/qwen1.5-moe-a2.7b-chat",
        help="Path to model directory containing tokenizer.json",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "."),
        help="Directory to write golden fixtures",
    )
    args = parser.parse_args()

    tok_path = os.path.join(args.model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        fail(f"tokenizer.json not found in model-dir: {tok_path}")

    try:
        tokenizer = Tokenizer.from_file(tok_path)
    except Exception as e:
        fail(f"failed to initialize Tokenizer from {tok_path}: {e}")

    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab > VOCAB_SIZE_GATE:
        fail(
            f"tokenizer vocab size {actual_vocab} exceeds gate limit {VOCAB_SIZE_GATE}"
        )

    golden_prompts = []
    for p in REPRESENTATIVE_PROMPTS:
        pid = p["id"]
        text = p["text"]
        enc = tokenizer.encode(text)
        token_ids = enc.ids
        if len(token_ids) < 16:
            fail(f"prompt {pid} produced only {len(token_ids)} tokens (< 16)")

        truncated = token_ids[:16]
        for tid in truncated:
            if not isinstance(tid, int) or tid < 0 or tid >= VOCAB_SIZE_GATE:
                fail(f"prompt {pid} contains invalid token ID: {tid}")

        golden_prompts.append(
            {
                "id": pid,
                "category": p["category"],
                "text": text,
                "tokens": truncated,
            }
        )

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    golden_payload = {
        "name": "M4 golden set",
        "description": "5 prompts × 16 tokens for full forward streaming gate",
        "vocab_size": VOCAB_SIZE_GATE,
        "prompts": golden_prompts,
    }

    # 1. Tulis m4_golden.json via json.dump
    golden_file = os.path.join(out_dir, "m4_golden.json")
    with open(golden_file, "w", encoding="utf-8") as f:
        json.dump(golden_payload, f, indent=2)
        f.write("\n")

    # 2. Validasi strict round-trip json.load
    with open(golden_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if loaded.get("name") != "M4 golden set" or len(loaded.get("prompts", [])) != 5:
        fail("m4_golden.json failed round-trip structural validation")

    for p in loaded["prompts"]:
        if len(p["tokens"]) != 16:
            fail(
                f"round-trip validation failed: {p['id']} tokens len !="
                f" 16 ({len(p['tokens'])})"
            )
        for tid in p["tokens"]:
            if not isinstance(tid, int) or tid < 0 or tid >= VOCAB_SIZE_GATE:
                fail(
                    f"round-trip validation failed: token out of range {tid} in"
                    f" {p['id']}"
                )

    # 3. Hitung SHA-256 m4_golden.json
    with open(golden_file, "rb") as f:
        golden_bytes = f.read()
    golden_hash = hashlib.sha256(golden_bytes).hexdigest()
    sha_file = os.path.join(out_dir, "m4_golden.sha256")
    with open(sha_file, "w", encoding="utf-8") as f:
        f.write(f"{golden_hash}  m4_golden.json\n")

    # 4. Tulis file tokens individual m4_prompt{1..5}_tokens.json (flat u32 array)
    for i, p in enumerate(loaded["prompts"], start=1):
        ind_file = os.path.join(out_dir, f"m4_prompt{i}_tokens.json")
        with open(ind_file, "w", encoding="utf-8") as f:
            json.dump(p["tokens"], f, indent=2)
            f.write("\n")

        # Validasi ulang flat JSON
        with open(ind_file, "r", encoding="utf-8") as f:
            ind_loaded = json.load(f)
        if ind_loaded != p["tokens"]:
            fail(f"individual token file {ind_file} mismatch with golden")

    print(
        f"SUCCESS: Generated m4_golden.json ({golden_hash}) and 5 prompt token"
        f" files in {out_dir}"
    )


if __name__ == "__main__":
    main()
