#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Tokenizer CLI & Fast Daemon helper for Dismoen Engine (M12-W2b).

Provides:
1. --encode "<text>" : Outputs JSON list of token IDs
2. --decode "<id or json-list>" : Decodes token ID(s) to UTF-8 text
3. --daemon : Persistent mode reading stdin and outputting to stdout for low-latency IPC
"""

import argparse
import json
import os
import sys

DEFAULT_VOCAB_SIZE = 248320


def load_vocab_size(model_dir: str) -> int:
    """Loads vocab_size from config.json if present, otherwise returns default."""
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "vocab_size" in data:
                    return int(data["vocab_size"])
                if "text_config" in data and "vocab_size" in data["text_config"]:
                    return int(data["text_config"]["vocab_size"])
        except Exception:
            pass
    return DEFAULT_VOCAB_SIZE


def fallback_encode(text: str, vocab_size: int) -> list:
    """Simple deterministic hashing encoder for mock/synthetic tests."""
    words = text.strip().split()
    token_ids = []
    for w in words:
        h = 0
        for c in w:
            h = (h * 31 + ord(c)) & 0x7FFFFFFF
        token_ids.append((h % (vocab_size - 1000)) + 100)
    return token_ids


def resolve_model_dir(cli_model_dir: str) -> str:
    """Precedence: --model-dir > DISMOEN_MODEL_ROOT > $HOME/models/qwen3.6-35b-a3b."""
    if cli_model_dir:
        return cli_model_dir

    model_root = os.environ.get("DISMOEN_MODEL_ROOT", "")
    if model_root and os.path.isdir(model_root):
        candidate = os.path.join(model_root, "qwen3.6-35b-a3b")
        return candidate if os.path.isdir(candidate) else model_root

    home = os.environ.get("HOME", "")
    if home:
        candidate = os.path.join(home, "models", "qwen3.6-35b-a3b")
        if os.path.isdir(candidate):
            return candidate
        cand_root = os.path.join(home, "models")
        if os.path.isdir(cand_root):
            return cand_root

    return ""


def init_tokenizer(model_dir: str):
    """Initializes tokenizers.Tokenizer if tokenizer.json exists."""
    tok_path = os.path.join(model_dir, "tokenizer.json") if model_dir else ""
    if tok_path and os.path.exists(tok_path):
        try:
            from tokenizers import Tokenizer

            return Tokenizer.from_file(tok_path)
        except Exception:
            return None
    return None


def encode_text(text: str, tokenizer, vocab_size: int) -> list:
    """Encodes text to token IDs using tokenizer or fallback."""
    if not text:
        return []
    if tokenizer is not None:
        try:
            return tokenizer.encode(text).ids
        except Exception:
            pass
    return fallback_encode(text, vocab_size)


def decode_token(tok_id: int, tokenizer) -> str:
    """Decodes single token ID preserving whitespace."""
    if tokenizer is not None:
        try:
            return tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:
            pass
    return f" tok_{tok_id} "


def handle_daemon(tokenizer, vocab_size: int):
    """Runs interactive line-by-line daemon."""
    while True:
        try:
            line = sys.stdin.readline()
        except (KeyboardInterrupt, EOFError):
            break
        if not line:
            break
        line = line.rstrip("\r\n")
        if not line:
            continue

        if line.startswith("Q") or line == "/exit":
            break
        elif line.startswith("E "):
            ids = encode_text(line[2:], tokenizer, vocab_size)
            sys.stdout.write(json.dumps(ids) + "\n")
            sys.stdout.flush()
        elif line.startswith("D "):
            try:
                tid = int(line[2:].strip())
                dec_text = decode_token(tid, tokenizer)
                sys.stdout.write(json.dumps(dec_text) + "\n")
                sys.stdout.flush()
            except Exception:
                sys.stdout.write(json.dumps("") + "\n")
                sys.stdout.flush()
        else:
            sys.stdout.write(json.dumps([]) + "\n")
            sys.stdout.flush()


def handle_encode(text_to_encode: str, tokenizer, vocab_size: int, output_path: str):
    """Handles one-shot encode command."""
    ids = encode_text(text_to_encode, tokenizer, vocab_size)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ids, f)
    else:
        print(json.dumps(ids))


def handle_decode(decode_arg: str, tokenizer):
    """Handles one-shot decode command."""
    try:
        val = json.loads(decode_arg)
        if isinstance(val, list):
            if tokenizer is not None:
                print(tokenizer.decode(val, skip_special_tokens=False))
            else:
                print(" ".join([decode_token(x, tokenizer) for x in val]))
            return
        elif isinstance(val, int):
            print(decode_token(val, tokenizer))
            return
    except Exception:
        pass

    try:
        tid = int(decode_arg)
        print(decode_token(tid, tokenizer))
    except Exception:
        print("")


def main():
    parser = argparse.ArgumentParser(description="Dismoen tokenizer & detokenizer CLI")
    parser.add_argument("--model-dir", default="", help="Path to model directory")
    parser.add_argument("--encode", default="", help="Text string to encode")
    parser.add_argument("--decode", default="", help="Token ID or JSON array to decode")
    parser.add_argument("--daemon", action="store_true", help="Run daemon")
    parser.add_argument("--prompt", default="", help="Alias for --encode")
    parser.add_argument("--output", default="", help="Output file path for encode")

    args = parser.parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    vocab_size = load_vocab_size(model_dir) if model_dir else DEFAULT_VOCAB_SIZE
    tokenizer = init_tokenizer(model_dir)

    if args.daemon:
        handle_daemon(tokenizer, vocab_size)
        sys.exit(0)

    text_to_encode = args.encode or args.prompt
    if text_to_encode:
        handle_encode(text_to_encode, tokenizer, vocab_size, args.output)
        sys.exit(0)

    if args.decode:
        handle_decode(args.decode, tokenizer)
        sys.exit(0)

    print("No action specified. Use --encode, --decode, or --daemon.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
