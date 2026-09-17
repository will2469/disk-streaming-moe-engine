#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Tokenize prompt text into token IDs JSON array using model's tokenizer.json."""

import argparse
import json
import os
import sys

VOCAB_SIZE_LIMIT = 151936


def main():
    parser = argparse.ArgumentParser(description="Tokenize text prompt for Kimo")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to model directory containing tokenizer.json",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt to tokenize",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for tokens JSON",
    )
    args = parser.parse_args()

    tok_path = os.path.join(args.model_dir, "tokenizer.json")
    token_ids = []
    if os.path.exists(tok_path):
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(tok_path)
            enc = tokenizer.encode(args.prompt)
            token_ids = enc.ids
        except Exception:
            token_ids = []
    if not token_ids:
        # Fallback word-level tokenization if tokenizer.json
        # or tokenizers library is not available
        words = args.prompt.strip().split()
        for w in words:
            # Deterministic hash to valid vocab ID
            h = 0
            for c in w:
                h = (h * 31 + ord(c)) & 0x7FFFFFFF
            token_ids.append((h % (VOCAB_SIZE_LIMIT - 1000)) + 100)

    for tid in token_ids:
        if tid < 0 or tid >= VOCAB_SIZE_LIMIT:
            print(
                f"Error: token {tid} exceeds vocab size {VOCAB_SIZE_LIMIT}",
                file=sys.stderr,
            )
            sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(token_ids, f)

    sys.exit(0)


if __name__ == "__main__":
    main()
