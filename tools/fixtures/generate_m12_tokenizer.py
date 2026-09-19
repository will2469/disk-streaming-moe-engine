#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generator fixture tokenizer BPE REAL kecil untuk M12 (fix #3).

Melatih BPE mungil via HuggingFace `tokenizers` (algoritma REAL yang sama
dengan model produksi) lalu menulis 4 berkas yang konsisten silang:
- fixtures/m12_tokenizer/tokenizer.json (BPE + 3 special tokens ChatML)
- fixtures/m12_tokenizer/tokenizer_config.json (added_tokens_decoder + eos)
- fixtures/m12_tokenizer/generation_config.json (eos_token_id)
- fixtures/m12_tokenizer/config.json (dimensi mini GGUF + vocab fixture)

BUKAN mock: engine + daemon memuat fixture ini dengan pustaka HF yang sama
persis seperti tokenizer 12.8MB model real; ID yang keluar adalah ID BPE
sebenarnya untuk fixture ini. Konsistensi silang memenuhi kontrak
resolve_special_tokens (M12-W1a): added_tokens <-> decoder <-> eos policy.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "fixtures", "m12_tokenizer")

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

CORPUS = [
    "Halo Dismoen dari pengujian otomatis!",
    "Jelaskan cara kerja MoE dengan singkat.",
    "The quick brown fox jumps over the lazy dog.",
    "Disk streaming memuat bobot per layer dari NVMe.",
    "Quantizer 4-bit menekan ukuran model ke gigabytes.",
    "Ketik pesan Anda, atau gunakan perintah internal.",
    "Turn 1 sukses dan Turn 2 juga sukses.",
    "Pertanyaan ringkas satu arah untuk mode one-shot.",
    "Para tetua menelaah naskah kuno di bawah langit senja.",
    "abc def ghi jkl mno pqr stu vwx yz 0123456789",
]


def main() -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    os.makedirs(OUT_DIR, exist_ok=True)

    tok = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=1024, special_tokens=SPECIALS, show_progress=False)
    tok.train_from_iterator(CORPUS, trainer=trainer)

    tok_path = os.path.join(OUT_DIR, "tokenizer.json")
    tok.save(tok_path)

    data = json.load(open(tok_path, encoding="utf-8"))
    added = {a["content"]: a["id"] for a in data.get("added_tokens", [])}
    im_start = added["<|im_start|>"]
    im_end = added["<|im_end|>"]
    endoftext = added["<|endoftext|>"]
    vocab_size = len(data["model"]["vocab"])
    print(f"trained BPE vocab={vocab_size} specials={added}")

    decoder_map = {
        str(im_start): {"content": "<|im_start|>"},
        str(im_end): {"content": "<|im_end|>"},
        str(endoftext): {"content": "<|endoftext|>"},
    }
    tok_cfg = {
        "added_tokens_decoder": decoder_map,
        "eos_token": "<|im_end|>",
        "tokenizer_class": "Qwen2Tokenizer",
    }
    with open(
        os.path.join(OUT_DIR, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(tok_cfg, f, indent=2)
        f.write("\n")

    gen_cfg = {"bos_token_id": im_start, "eos_token_id": [im_end, endoftext]}
    with open(
        os.path.join(OUT_DIR, "generation_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(gen_cfg, f, indent=2)
        f.write("\n")

    # Dimensi komputasi mini (selaras GGUF fixtures/m9_port_mini.gguf).
    # vocab_size TETAP 1024 (syarat validator mini port): tokenizer fixture
    # hanya memakai sub-vocab awal (<vocab tokenizer); baris sisa adalah
    # bobot real yang ter-stream tapi tak terpilih argmax pada umumnya.
    # embedding_lookup tetap jujur (tid < 1024, tabel 1024 baris).
    cfg = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": False,
            "full_attention_interval": 4,
            "head_dim": 32,
            "hidden_size": 128,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 64,
            "num_attention_heads": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "shared_expert_intermediate_size": 64,
            "vocab_size": 1024,
        },
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    # Bukti kewarasan: round-trip HF langsung.
    enc = tok.encode("Halo Dismoen!")
    dec = tok.decode(enc.ids)
    assert dec == "Halo Dismoen!", f"round-trip rusak: {dec!r}"
    print(f"round-trip OK: 'Halo Dismoen!' -> {enc.ids}")

    # Bukti determinisme: encode dua kali identik.
    assert (
        tok.encode("Jelaskan cara kerja MoE").ids
        == tok.encode("Jelaskan cara kerja MoE").ids
    )
    print("determinisme OK")


if __name__ == "__main__":
    main()
