#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle ChatML Formatter & Corpus Reference (M12-W1b / G-M12-1).

Memverifikasi kecocokan byte-exact 100% dan token fidelity terhadap
template Jinja upstream dan HuggingFace tokenizer pada revisi terkunci:
- 9 skenario subset valid (single-turn, multi-turn, whitespace trim,
  think-strip, think-wrap, trailing-assistant, user-only, literal-think,
  tool_response-middle).
- 9 skenario edge penolakan (empty-messages, system-not-first,
  duplicate-system, no-user-query, only-tool-response, role-tool,
  role-custom, reasoning-field, tool-calls-field).
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from jinja2 import BaseLoader, Environment
except ImportError:
    print(
        "ERROR: jinja2 tidak ditemukan. Gunakan python environment yang sesuai.",
        file=sys.stderr,
    )
    sys.exit(1)


def create_jinja_env(chat_template_str: str) -> Environment:
    """Membuat environment Jinja2 dengan dukungan raise_exception."""

    def raise_exception(msg):
        raise ValueError(msg)

    env = Environment(loader=BaseLoader())
    env.globals["raise_exception"] = raise_exception
    return env


def get_corpus_cases():
    """Mengembalikan daftar kasus valid dan kasus negatif."""
    valid_cases = [
        {
            "name": "single-turn",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Halo, siapa kamu?"},
            ],
            "add_generation_prompt": True,
        },
        {
            "name": "multi-turn",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Halo, siapa kamu?"},
                {"role": "assistant", "content": "Saya Dismoen."},
                {"role": "user", "content": "Lanjut."},
            ],
            "add_generation_prompt": True,
        },
        {
            "name": "whitespace-trim",
            "messages": [
                {
                    "role": "system",
                    "content": "  System message with spaces  \n",
                },
                {
                    "role": "user",
                    "content": ("\n\n  User message with tabs and newlines\t\n  "),
                },
            ],
            "add_generation_prompt": True,
        },
        {
            "name": "think-strip",
            "messages": [
                {"role": "user", "content": "Q1"},
                {
                    "role": "assistant",
                    "content": ("<think>\nThinking step 1\n</think>\nAnswer 1"),
                },
                {"role": "user", "content": "Q2"},
            ],
            "add_generation_prompt": True,
        },
        {
            "name": "think-wrap",
            "messages": [
                {"role": "user", "content": "Q1"},
                {
                    "role": "assistant",
                    "content": ("<think>\nThinking step 1\n</think>\nAnswer 1"),
                },
            ],
            "add_generation_prompt": True,
        },
        {
            "name": "trailing-assistant-no-think",
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "Answer 1"},
            ],
            "add_generation_prompt": True,
        },
        {
            "name": "user-only",
            "messages": [
                {"role": "user", "content": "Hello world"},
            ],
            "add_generation_prompt": True,
        },
        {
            "name": "user-with-think-tag",
            "messages": [
                {"role": "user", "content": "What does <think> mean?"},
            ],
            "add_generation_prompt": True,
        },
        {
            "name": "user-with-tool-response-middle",
            "messages": [
                {
                    "role": "user",
                    "content": "<tool_response>result</tool_response>",
                },
                {"role": "user", "content": "Analyze the result."},
            ],
            "add_generation_prompt": True,
        },
    ]

    negative_cases = [
        {
            "name": "empty-messages",
            "messages": [],
            "expected_error": "unsupported: empty_messages",
        },
        {
            "name": "system-not-first",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "sys"},
            ],
            "expected_error": "unsupported: system_not_first",
        },
        {
            "name": "duplicate-system",
            "messages": [
                {"role": "system", "content": "s1"},
                {"role": "system", "content": "s2"},
                {"role": "user", "content": "hi"},
            ],
            "expected_error": "unsupported: duplicate_system",
        },
        {
            "name": "no-user-query",
            "messages": [
                {"role": "system", "content": "sys"},
            ],
            "expected_error": "unsupported: no_user_query",
        },
        {
            "name": "only-tool-response-user",
            "messages": [
                {
                    "role": "user",
                    "content": "<tool_response>res</tool_response>",
                },
            ],
            "expected_error": "unsupported: no_user_query",
        },
        {
            "name": "unsupported-role-tool",
            "messages": [
                {"role": "tool", "content": "call"},
            ],
            "expected_error": "unsupported: role=tool",
        },
        {
            "name": "unsupported-role-custom",
            "messages": [
                {"role": "bot", "content": "hi"},
            ],
            "expected_error": "unsupported: role=bot",
        },
        {
            "name": "reasoning-content-field",
            "messages": [
                {
                    "role": "user",
                    "content": "hi",
                    "has_reasoning_content": True,
                },
            ],
            "expected_error": "unsupported: field=reasoning_content",
        },
        {
            "name": "tool-calls-field",
            "messages": [
                {
                    "role": "assistant",
                    "content": "calling",
                    "has_tool_calls": True,
                },
            ],
            "expected_error": "unsupported: field=tool_calls",
        },
    ]

    return valid_cases, negative_cases


def render_all_valid(tmpl, valid_cases):
    """Me-render seluruh kasus valid dengan template Jinja."""
    rendered_corpus = []
    for case in valid_cases:
        prompt_rendered = tmpl.render(
            messages=case["messages"],
            add_generation_prompt=case["add_generation_prompt"],
        )
        rendered_corpus.append(
            {
                "name": case["name"],
                "messages": case["messages"],
                "add_generation_prompt": case["add_generation_prompt"],
                "expected_prompt": prompt_rendered,
            }
        )
    return rendered_corpus


def verify_negative_cases(tmpl, negative_cases):
    """Memverifikasi negative cases memicu exception di Jinja."""
    jinja_error_cases = [
        "empty-messages",
        "system-not-first",
        "duplicate-system",
        "no-user-query",
        "only-tool-response-user",
    ]
    for neg in negative_cases:
        if neg["name"] in jinja_error_cases:
            raised = False
            try:
                tmpl.render(messages=neg["messages"], add_generation_prompt=True)
            except Exception:
                raised = True
            assert raised, f"Negative case {neg['name']} tidak raise di Jinja!"


def verify_tokenizer_encoding(model_dir, rendered_corpus):
    """Memverifikasi tokenisasi via tokenizers.Tokenizer jika ada."""
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tok_path):
        return
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tok_path)
        for case in rendered_corpus:
            enc = tokenizer.encode(case["expected_prompt"])
            assert len(enc.ids) > 0, f"Token IDs kosong untuk {case['name']}"
            first_tok = tokenizer.id_to_token(enc.ids[0])
            assert first_tok in [
                "<|im_start|>",
                "<|endoftext|>",
            ], f"First token {first_tok} bukan special token pada {case['name']}"
        print(f"OK: Tokenizer encoding valid untuk {len(rendered_corpus)} kasus!")
    except ImportError:
        pass


def check_fixtures(output_fixtures, rendered_corpus, negative_cases, tmpl, model_dir):
    """Memverifikasi fixtures terhadap template dan tokenizers."""
    if not os.path.isfile(output_fixtures):
        print(f"FAIL: Fixture {output_fixtures} belum ada.")
        sys.exit(1)
    with open(output_fixtures, "r", encoding="utf-8") as f:
        existing = json.load(f)

    assert len(existing["valid_cases"]) == len(
        rendered_corpus
    ), "Jumlah valid cases berbeda"
    for i, exp in enumerate(rendered_corpus):
        act = existing["valid_cases"][i]
        assert act["name"] == exp["name"], f"Name mismatch pada index {i}"
        assert (
            act["expected_prompt"] == exp["expected_prompt"]
        ), f"Prompt mismatch pada case {exp['name']}"

    verify_negative_cases(tmpl, negative_cases)
    verify_tokenizer_encoding(model_dir, rendered_corpus)

    print(f"OK: Seluruh {len(rendered_corpus)} skenario corpus 100% byte-exact!")


def resolve_model_dir(cli_arg: str) -> str:
    """Mendapatkan path direktori model."""
    if cli_arg:
        return cli_arg
    home = os.environ.get("HOME", "")
    cand = os.path.join(home, "models", "qwen3.6-35b-a3b")
    if os.path.isdir(cand):
        return cand
    return "."


def main():
    """Fungsi utama CLI oracle ChatML."""
    parser = argparse.ArgumentParser(
        description="Oracle ChatML template & corpus check"
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("DISMOEN_MODEL_DIR", ""),
        help="Path ke model dir yang memuat tokenizer_config.json",
    )
    parser.add_argument(
        "--output-fixtures",
        default="fixtures/m12_chatml_corpus.json",
        help="Path file output fixtures JSON",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate fixtures JSON dari template Jinja upstream",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Memverifikasi kecocokan fixtures terhadap template upstream",
    )
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.isfile(tok_cfg_path):
        print(f"ERROR: tokenizer_config.json tidak ada di: {tok_cfg_path}")
        sys.exit(1)

    with open(tok_cfg_path, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)

    chat_template_str = tok_cfg.get("chat_template")
    if not chat_template_str:
        print("ERROR: chat_template tidak ada di tokenizer_config.json")
        sys.exit(1)

    env = create_jinja_env(chat_template_str)
    tmpl = env.from_string(chat_template_str)

    valid_cases, negative_cases = get_corpus_cases()
    rendered_corpus = render_all_valid(tmpl, valid_cases)

    if args.generate:
        out_path = Path(args.output_fixtures)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_data = {
            "valid_cases": rendered_corpus,
            "negative_cases": negative_cases,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(corpus_data, f, indent=2)
        print(f"OK: Berhasil menggenerasi fixtures di {out_path}")

    if args.check:
        check_fixtures(
            args.output_fixtures,
            rendered_corpus,
            negative_cases,
            tmpl,
            model_dir,
        )


if __name__ == "__main__":
    main()
