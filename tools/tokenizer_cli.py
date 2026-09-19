#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Tokenizer CLI & Fast Daemon helper for Dismoen Engine (M12-W2b, fix #3).

Kontrak non-negotiable (fix #3): TIDAK ADA fallback hash/sintetis.
Bila tokenizer.json tidak ada atau pustaka `tokenizers` gagal memuatnya,
proses GAGAL fail-closed (exit != 0 + pesan ke stderr). ID yang keluar
selalu ID BPE sebenarnya dari pustaka HF — nilai asli, bukan mock.

Protokol daemon (satu fork per sesi CLI, dipakai klien Mojo):
- Server mencetak satu baris `READY` saat siap.
- Klien mengirim satu baris JSON per permintaan:
    {"op": "encode", "text": "<teks mentah>"}
    {"op": "decode", "ids": [1, 2, 3]}
- Server menjawab satu baris JSON:
    {"ok": true, "ids": [...]} / {"ok": true, "text": "..."}
    {"ok": false, "error": "..."}
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


def init_tokenizer_or_fail(model_dir: str):
    """Initializes tokenizers.Tokenizer or FAILS CLOSED (no hash fallback)."""
    tok_path = os.path.join(model_dir, "tokenizer.json") if model_dir else ""
    if not tok_path or not os.path.exists(tok_path):
        print(
            f"TOKENIZER_NOT_FOUND: no tokenizer.json in model dir: {model_dir!r} "
            "(hash fallback dilarang per fix #3)",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(tok_path)
    except Exception as exc:
        print(
            f"TOKENIZER_LOAD_FAILED: cannot load {tok_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)


def encode_text(text: str, tokenizer) -> list:
    """Encodes text to REAL BPE token IDs (raises on failure, no fallback)."""
    return tokenizer.encode(text).ids


def decode_token_ids(ids: list, tokenizer) -> str:
    """Decodes token IDs to text via the REAL tokenizer (raises on failure)."""
    return tokenizer.decode(ids, skip_special_tokens=False)


def handle_daemon(tokenizer):
    """Runs persistent daemon: READY handshake, one JSON line per request."""
    sys.stdout.write(json.dumps({"ok": True, "status": "READY"}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            sys.stdout.write(
                json.dumps({"ok": False, "error": f"bad JSON: {exc}"}) + "\n"
            )
            sys.stdout.flush()
            continue
        op = req.get("op", "")
        try:
            if op == "encode":
                ids = encode_text(req.get("text", ""), tokenizer)
                sys.stdout.write(json.dumps({"ok": True, "ids": ids}) + "\n")
            elif op == "decode":
                ids = req.get("ids", [])
                text = decode_token_ids([int(x) for x in ids], tokenizer)
                sys.stdout.write(json.dumps({"ok": True, "text": text}) + "\n")
            elif op == "quit":
                break
            else:
                sys.stdout.write(
                    json.dumps({"ok": False, "error": f"unknown op: {op}"}) + "\n"
                )
        except Exception as exc:
            sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
        sys.stdout.flush()


def handle_encode(text_to_encode: str, tokenizer, output_path: str):
    """Handles one-shot encode command (REAL tokenizer only)."""
    ids = encode_text(text_to_encode, tokenizer)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ids, f)
    else:
        print(json.dumps(ids))


def handle_decode(decode_arg: str, tokenizer):
    """Handles one-shot decode command (REAL tokenizer only).

    Output ditulis TANPA newline tambahan (data murni untuk klien Mojo).
    """
    try:
        val = json.loads(decode_arg)
        if isinstance(val, list):
            sys.stdout.write(decode_token_ids([int(x) for x in val], tokenizer))
            return
        elif isinstance(val, int):
            sys.stdout.write(decode_token_ids([val], tokenizer))
            return
    except Exception:
        pass

    try:
        tid = int(decode_arg)
        sys.stdout.write(decode_token_ids([tid], tokenizer))
    except Exception as exc:
        print(f"TOKENIZER_DECODE_FAILED: {exc}", file=sys.stderr)
        sys.exit(2)


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
    tokenizer = init_tokenizer_or_fail(model_dir)

    if args.daemon:
        handle_daemon(tokenizer)
        sys.exit(0)

    text_to_encode = args.encode or args.prompt
    if text_to_encode:
        handle_encode(text_to_encode, tokenizer, args.output)
        sys.exit(0)

    if args.decode:
        handle_decode(args.decode, tokenizer)
        sys.exit(0)

    print("No action specified. Use --encode, --decode, or --daemon.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
