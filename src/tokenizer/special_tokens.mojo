# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Runtime Special-Token Resolver & Lockfile Verification (M12-W1a).

Menegakkan kontrak §2.1 & M12-1:
- Resolusi runtime <|im_start|>, <|im_end|>, <|endoftext|> dari metadata tokenizer.
- DILARANG KERAS konstanta numerik hardcoded untuk token ID di mana pun.
- Validasi silang tokenizer.json:added_tokens vs tokenizer_config.json:added_tokens_decoder.
- Validasi stop policy {im_end, endoftext} ⊆ generation_config.json:eos_token_id.
- Validasi range ID < vocab_size (TOKEN_INVALID).
- Verifikasi SHA-256 dan revision terhadap models.lock.json (CONFIG_MISMATCH).
"""

from format.file_io import read_small_file
from format.scanner import Scanner
from format.sha256 import sha256
from std.collections import List


def sha256_hex(data: List[UInt8]) -> String:
    """Menghitung digest SHA-256 dalam representasi 64-karakter lowercase hex.
    """
    var digest = sha256(data)
    var hex_bytes = List[UInt8]()
    hex_bytes.reserve(64)
    for i in range(len(digest)):
        var b = Int(digest[i])
        var hi = (b >> 4) & 0x0F
        var lo = b & 0x0F
        hex_bytes.append(UInt8(48 + hi if hi < 10 else 87 + hi))
        hex_bytes.append(UInt8(48 + lo if lo < 10 else 87 + lo))
    return String(from_utf8_lossy=Span(hex_bytes))


@fieldwise_init
struct SpecialTokenResolver(Copyable, Movable):
    """Resolver hasil resolusi runtime untuk special token ChatML."""

    var im_start_id: Int
    var im_end_id: Int
    var endoftext_id: Int
    var eos_token_ids: List[Int]
    var eos_token_str: String

    def is_stop_token(self, token_id: Int) -> Bool:
        """Memeriksa apakah token ID termasuk dalam himpunan EOS stop tokens."""
        for i in range(len(self.eos_token_ids)):
            if self.eos_token_ids[i] == token_id:
                return True
        return False


def verify_tokenizer_lockfile(model_dir: String, lock_path: String) raises:
    """Memverifikasi SHA-256 dan revision tokenizer terhadap models.lock.json.
    """
    var raw_lock = read_small_file(lock_path)
    var sc = Scanner(raw_lock^, lock_path, True)
    sc.skip_ws()
    sc.expect(123)  # '{'

    var locked_rev = String("")
    var locked_tok_json_sha = String("")
    var locked_tok_cfg_sha = String("")

    while not sc.eof():
        sc.skip_ws()
        if sc.peek() == 125:  # '}'
            sc.pos += 1
            break
        var key = sc.parse_string()
        sc.skip_ws()
        sc.expect(58)  # ':'
        sc.skip_ws()

        if key == "tokenizer_revision":
            locked_rev = sc.parse_string()
        elif key == "tokenizer_sha256":
            sc.expect(123)  # '{'
            while True:
                sc.skip_ws()
                if sc.peek() == 125:  # '}'
                    sc.pos += 1
                    break
                var f_name = sc.parse_string()
                sc.skip_ws()
                sc.expect(58)  # ':'
                sc.skip_ws()
                var f_hash = sc.parse_string()
                if f_name == "tokenizer.json":
                    locked_tok_json_sha = f_hash
                elif f_name == "tokenizer_config.json":
                    locked_tok_cfg_sha = f_hash
                sc.skip_ws()
                if sc.peek() == 44:  # ','
                    sc.pos += 1
        else:
            sc.skip_value()
        sc.skip_ws()
        if sc.peek() == 44:  # ','
            sc.pos += 1

    if locked_rev == "":
        raise Error("CONFIG_MISMATCH: missing tokenizer_revision in lockfile")
    if locked_tok_json_sha == "" or locked_tok_cfg_sha == "":
        raise Error("CONFIG_MISMATCH: missing tokenizer_sha256 in lockfile")

    var tok_json_path = String(model_dir, "/tokenizer.json")
    var tok_cfg_path = String(model_dir, "/tokenizer_config.json")

    var raw_tj = read_small_file(tok_json_path)
    var actual_tj_sha = sha256_hex(raw_tj)
    if actual_tj_sha != locked_tok_json_sha:
        raise Error(
            "CONFIG_MISMATCH: tokenizer.json sha256 mismatch (actual="
            + actual_tj_sha
            + " vs locked="
            + locked_tok_json_sha
            + ")"
        )

    var raw_tc = read_small_file(tok_cfg_path)
    var actual_tc_sha = sha256_hex(raw_tc)
    if actual_tc_sha != locked_tok_cfg_sha:
        raise Error(
            "CONFIG_MISMATCH: tokenizer_config.json sha256 mismatch (actual="
            + actual_tc_sha
            + " vs locked="
            + locked_tok_cfg_sha
            + ")"
        )


def resolve_special_tokens(
    model_dir: String,
    vocab_size: Int = 248320,
    lock_path: String = "",
    verify_lock: Bool = False,
) raises -> SpecialTokenResolver:
    """Melakukan resolusi runtime ID token khusus ChatML dari metadata model.

    Invarian:
    1. Ekstraksi ID <|im_start|>, <|im_end|>, <|endoftext|> dari tokenizer.json:added_tokens.
    2. Validasi silang pemetaan terhadap tokenizer_config.json:added_tokens_decoder.
    3. Validasi eos_token merujuk ke token khusus yang sah.
    4. Validasi stop policy {im_end, endoftext} ⊆ generation_config.json:eos_token_id.
    5. Validasi rentang ID < vocab_size (TOKEN_INVALID).
    6. Verifikasi SHA-256 terhadap lockfile bila verify_lock == True.
    """
    var tok_path = String(model_dir, "/tokenizer.json")
    var cfg_path = String(model_dir, "/tokenizer_config.json")
    var gen_path = String(model_dir, "/generation_config.json")

    # 1. Parse tokenizer.json -> added_tokens
    var raw_tok: List[UInt8]
    try:
        raw_tok = read_small_file(tok_path)
    except e:
        raise Error("CONFIG_MISMATCH: cannot open tokenizer.json: " + String(e))

    var sc_tok = Scanner(raw_tok^, tok_path, True)
    sc_tok.skip_ws()
    sc_tok.expect(123)  # '{'

    var im_start_id = -1
    var im_end_id = -1
    var endoftext_id = -1
    var found_im_start = False
    var found_im_end = False
    var found_endoftext = False

    while not sc_tok.eof():
        sc_tok.skip_ws()
        if sc_tok.peek() == 125:  # '}'
            sc_tok.pos += 1
            break
        var key = sc_tok.parse_string()
        sc_tok.skip_ws()
        sc_tok.expect(58)  # ':'
        sc_tok.skip_ws()

        if key == "added_tokens":
            sc_tok.expect(91)  # '['
            while True:
                sc_tok.skip_ws()
                if sc_tok.peek() == 93:  # ']'
                    sc_tok.pos += 1
                    break
                sc_tok.expect(123)  # '{'
                var token_id = -1
                var token_content = String("")
                while True:
                    sc_tok.skip_ws()
                    if sc_tok.peek() == 125:  # '}'
                        sc_tok.pos += 1
                        break
                    var item_key = sc_tok.parse_string()
                    sc_tok.skip_ws()
                    sc_tok.expect(58)  # ':'
                    sc_tok.skip_ws()
                    if item_key == "id":
                        token_id = sc_tok.parse_uint()
                    elif item_key == "content":
                        token_content = sc_tok.parse_string()
                    else:
                        sc_tok.skip_value()
                    sc_tok.skip_ws()
                    if sc_tok.peek() == 44:  # ','
                        sc_tok.pos += 1

                if token_content == "<|im_start|>":
                    im_start_id = token_id
                    found_im_start = True
                elif token_content == "<|im_end|>":
                    im_end_id = token_id
                    found_im_end = True
                elif token_content == "<|endoftext|>":
                    endoftext_id = token_id
                    found_endoftext = True

                sc_tok.skip_ws()
                if sc_tok.peek() == 44:  # ','
                    sc_tok.pos += 1
            # added_tokens selesai dipindai
            break
        else:
            sc_tok.skip_value()
        sc_tok.skip_ws()
        if sc_tok.peek() == 44:  # ','
            sc_tok.pos += 1

    if not (found_im_start and found_im_end and found_endoftext):
        raise Error(
            "CONFIG_MISMATCH: tokenizer.json missing required special tokens"
            " (<|im_start|>, <|im_end|>, <|endoftext|>)"
        )

    # 2. Parse tokenizer_config.json -> added_tokens_decoder & eos_token
    var raw_cfg: List[UInt8]
    try:
        raw_cfg = read_small_file(cfg_path)
    except e:
        raise Error(
            "CONFIG_MISMATCH: cannot open tokenizer_config.json: " + String(e)
        )

    var sc_cfg = Scanner(raw_cfg^, cfg_path, True)
    sc_cfg.skip_ws()
    sc_cfg.expect(123)  # '{'

    var found_dec_im_start = False
    var found_dec_im_end = False
    var found_dec_endoftext = False
    var eos_token_str = String("")

    while not sc_cfg.eof():
        sc_cfg.skip_ws()
        if sc_cfg.peek() == 125:  # '}'
            sc_cfg.pos += 1
            break
        var key = sc_cfg.parse_string()
        sc_cfg.skip_ws()
        sc_cfg.expect(58)  # ':'
        sc_cfg.skip_ws()

        if key == "added_tokens_decoder":
            sc_cfg.expect(123)  # '{'
            while True:
                sc_cfg.skip_ws()
                if sc_cfg.peek() == 125:  # '}'
                    sc_cfg.pos += 1
                    break
                var id_str = sc_cfg.parse_string()
                var dec_id = Int(id_str)
                sc_cfg.skip_ws()
                sc_cfg.expect(58)  # ':'
                sc_cfg.skip_ws()
                sc_cfg.expect(123)  # '{'
                var dec_content = String("")
                while True:
                    sc_cfg.skip_ws()
                    if sc_cfg.peek() == 125:  # '}'
                        sc_cfg.pos += 1
                        break
                    var d_key = sc_cfg.parse_string()
                    sc_cfg.skip_ws()
                    sc_cfg.expect(58)  # ':'
                    sc_cfg.skip_ws()
                    if d_key == "content":
                        dec_content = sc_cfg.parse_string()
                    else:
                        sc_cfg.skip_value()
                    sc_cfg.skip_ws()
                    if sc_cfg.peek() == 44:  # ','
                        sc_cfg.pos += 1

                if dec_content == "<|im_start|>":
                    if dec_id != im_start_id:
                        raise Error(
                            "CONFIG_MISMATCH: id mismatch for <|im_start|>"
                            " between tokenizer.json ("
                            + String(im_start_id)
                            + ") and tokenizer_config.json ("
                            + String(dec_id)
                            + ")"
                        )
                    found_dec_im_start = True
                elif dec_content == "<|im_end|>":
                    if dec_id != im_end_id:
                        raise Error(
                            "CONFIG_MISMATCH: id mismatch for <|im_end|>"
                            " between tokenizer.json ("
                            + String(im_end_id)
                            + ") and tokenizer_config.json ("
                            + String(dec_id)
                            + ")"
                        )
                    found_dec_im_end = True
                elif dec_content == "<|endoftext|>":
                    if dec_id != endoftext_id:
                        raise Error(
                            "CONFIG_MISMATCH: id mismatch for <|endoftext|>"
                            " between tokenizer.json ("
                            + String(endoftext_id)
                            + ") and tokenizer_config.json ("
                            + String(dec_id)
                            + ")"
                        )
                    found_dec_endoftext = True

                sc_cfg.skip_ws()
                if sc_cfg.peek() == 44:  # ','
                    sc_cfg.pos += 1
        elif key == "eos_token":
            if sc_cfg.peek() == 34:  # '"'
                eos_token_str = sc_cfg.parse_string()
            elif sc_cfg.peek() == 123:  # '{'
                sc_cfg.pos += 1
                while True:
                    sc_cfg.skip_ws()
                    if sc_cfg.peek() == 125:  # '}'
                        sc_cfg.pos += 1
                        break
                    var e_key = sc_cfg.parse_string()
                    sc_cfg.skip_ws()
                    sc_cfg.expect(58)
                    sc_cfg.skip_ws()
                    if e_key == "content":
                        eos_token_str = sc_cfg.parse_string()
                    else:
                        sc_cfg.skip_value()
                    sc_cfg.skip_ws()
                    if sc_cfg.peek() == 44:
                        sc_cfg.pos += 1
            else:
                sc_cfg.skip_value()
        else:
            sc_cfg.skip_value()
        sc_cfg.skip_ws()
        if sc_cfg.peek() == 44:  # ','
            sc_cfg.pos += 1

    if not (found_dec_im_start and found_dec_im_end and found_dec_endoftext):
        raise Error(
            "CONFIG_MISMATCH: tokenizer_config.json added_tokens_decoder is"
            " missing required special tokens"
        )
    if not (
        eos_token_str == "<|im_start|>"
        or eos_token_str == "<|im_end|>"
        or eos_token_str == "<|endoftext|>"
    ):
        raise Error(
            "CONFIG_MISMATCH: eos_token in tokenizer_config.json ('"
            + eos_token_str
            + "') does not refer to a valid special token (<|im_start|>,"
            " <|im_end|>, <|endoftext|>)"
        )

    # 3. Parse generation_config.json -> eos_token_id
    var raw_gen: List[UInt8]
    try:
        raw_gen = read_small_file(gen_path)
    except e:
        raise Error(
            "CONFIG_MISMATCH: cannot open generation_config.json: " + String(e)
        )

    var sc_gen = Scanner(raw_gen^, gen_path, True)
    sc_gen.skip_ws()
    sc_gen.expect(123)  # '{'

    var eos_token_ids = List[Int]()

    while not sc_gen.eof():
        sc_gen.skip_ws()
        if sc_gen.peek() == 125:  # '}'
            sc_gen.pos += 1
            break
        var key = sc_gen.parse_string()
        sc_gen.skip_ws()
        sc_gen.expect(58)  # ':'
        sc_gen.skip_ws()

        if key == "eos_token_id":
            if sc_gen.peek() == 91:  # '['
                eos_token_ids = sc_gen.parse_int_array()
            elif sc_gen.peek() >= 48 and sc_gen.peek() <= 57:
                eos_token_ids.append(sc_gen.parse_uint())
            else:
                sc_gen.skip_value()
        else:
            sc_gen.skip_value()
        sc_gen.skip_ws()
        if sc_gen.peek() == 44:  # ','
            sc_gen.pos += 1

    var has_im_end = False
    var has_endoftext = False
    for i in range(len(eos_token_ids)):
        if eos_token_ids[i] == im_end_id:
            has_im_end = True
        if eos_token_ids[i] == endoftext_id:
            has_endoftext = True

    if not (has_im_end and has_endoftext):
        raise Error(
            "CONFIG_MISMATCH: generation_config.json eos_token_id does not"
            " contain required stop tokens {<|im_end|>, <|endoftext|>}"
        )

    # 4. Validasi rentang ID terhadap vocab_size (TOKEN_INVALID)
    var effective_vocab_size = vocab_size
    var model_cfg_path = String(model_dir, "/config.json")
    try:
        var raw_mcfg = read_small_file(model_cfg_path)
        var sc_mcfg = Scanner(raw_mcfg^, model_cfg_path, True)
        sc_mcfg.skip_ws()
        if sc_mcfg.peek() == 123:
            sc_mcfg.pos += 1
            while not sc_mcfg.eof():
                sc_mcfg.skip_ws()
                if sc_mcfg.peek() == 125:
                    break
                var k = sc_mcfg.parse_string()
                sc_mcfg.skip_ws()
                sc_mcfg.expect(58)
                sc_mcfg.skip_ws()
                if k == "vocab_size":
                    var vs = sc_mcfg.parse_uint()
                    if vs > 0:
                        effective_vocab_size = vs
                    break
                else:
                    sc_mcfg.skip_value()
                sc_mcfg.skip_ws()
                if sc_mcfg.peek() == 44:
                    sc_mcfg.pos += 1
    except:
        pass

    if im_start_id < 0 or im_start_id >= effective_vocab_size:
        raise Error(
            "TOKEN_INVALID: <|im_start|> id "
            + String(im_start_id)
            + " outside vocab_size range [0, "
            + String(effective_vocab_size)
            + ")"
        )
    if im_end_id < 0 or im_end_id >= effective_vocab_size:
        raise Error(
            "TOKEN_INVALID: <|im_end|> id "
            + String(im_end_id)
            + " outside vocab_size range [0, "
            + String(effective_vocab_size)
            + ")"
        )
    if endoftext_id < 0 or endoftext_id >= effective_vocab_size:
        raise Error(
            "TOKEN_INVALID: <|endoftext|> id "
            + String(endoftext_id)
            + " outside vocab_size range [0, "
            + String(effective_vocab_size)
            + ")"
        )

    # 5. Verifikasi lockfile bila diminta
    if verify_lock:
        var actual_lock_path = lock_path
        if actual_lock_path == "":
            actual_lock_path = "models.lock.json"
        verify_tokenizer_lockfile(model_dir, actual_lock_path)

    return SpecialTokenResolver(
        im_start_id=im_start_id,
        im_end_id=im_end_id,
        endoftext_id=endoftext_id,
        eos_token_ids=eos_token_ids^,
        eos_token_str=eos_token_str,
    )
