# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M12-W1a: Special-Token Resolver & Lockfile Pinning.

Memverifikasi pemenuhan butir 1-4 & 6 §2.1 M12-chat-cli.md serta M12-1:
- Resolusi runtime special token ChatML dari metadata model.
- STRICT PROHIBITION hardcoded token ID constants.
- 4 jalur kegagalan: CONFIG_MISMATCH (mapping mismatch, invalid eos_token,
  stop-policy violation) dan TOKEN_INVALID (range >= vocab_size).
- Uji negatif geser-ID (shifted-ID mismatch).
- Verifikasi lockfile SHA-256 dan revision terhadap models.lock.json.
"""

from std.collections import List
from std.ffi import external_call
from std.os import getenv, unlink
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from tokenizer.special_tokens import (
    SpecialTokenResolver,
    resolve_special_tokens,
    verify_tokenizer_lockfile,
    sha256_hex,
)


def c_mkdir(path: String) -> Int:
    """Membuat direktori via libc mkdir."""
    var p = path.as_bytes()
    var p_z = List[UInt8]()
    p_z.reserve(len(p) + 1)
    for i in range(len(p)):
        p_z.append(p[i])
    p_z.append(0)
    return Int(external_call["mkdir", Int32](p_z.unsafe_ptr(), 0o777))


def write_file(path: String, content: String) raises:
    """Menulis konten string ke file."""
    var f = open(path, "w")
    f.write(content)
    f.close()


def setup_synthetic_fixture(
    base_dir: String,
    im_start_id: Int,
    im_end_id: Int,
    eot_id: Int,
    dec_im_start_id: Int,
    dec_im_end_id: Int,
    dec_eot_id: Int,
    eos_token_ref: String,
    gen_eos_ids: List[Int],
    vocab_size: Int,
) raises:
    """Menulis set fixture metadata sintetis untuk pengujian resolver."""
    _ = c_mkdir(base_dir)

    # 1. tokenizer.json
    var tj = String(
        '{\n  "added_tokens": [\n    {\n      "id": ',
        String(im_start_id),
        ',\n      "content": "<|im_start|>",\n      "single_word": false,\n'
        '      "lstrip": false,\n      "rstrip": false,\n      "normalized":'
        ' false,\n      "special": true\n    },\n    {\n      "id": ',
        String(im_end_id),
        ',\n      "content": "<|im_end|>",\n      "single_word": false,\n'
        '      "lstrip": false,\n      "rstrip": false,\n      "normalized":'
        ' false,\n      "special": true\n    },\n    {\n      "id": ',
        String(eot_id),
        ',\n      "content": "<|endoftext|>",\n      "single_word": false,\n'
        '      "lstrip": false,\n      "rstrip": false,\n      "normalized":'
        ' false,\n      "special": true\n    }\n  ]\n}\n',
    )
    write_file(String(base_dir, "/tokenizer.json"), tj)

    # 2. tokenizer_config.json
    var tc = String(
        '{\n  "added_tokens_decoder": {\n    "',
        String(dec_im_start_id),
        '": {\n      "content": "<|im_start|>",\n      "lstrip": false,\n'
        '      "normalized": false,\n      "rstrip": false,\n     '
        ' "single_word": false,\n      "special": true\n    },\n    "',
        String(dec_im_end_id),
        '": {\n      "content": "<|im_end|>",\n      "lstrip": false,\n'
        '      "normalized": false,\n      "rstrip": false,\n     '
        ' "single_word": false,\n      "special": true\n    },\n    "',
        String(dec_eot_id),
        '": {\n      "content": "<|endoftext|>",\n      "lstrip": false,\n'
        '      "normalized": false,\n      "rstrip": false,\n     '
        ' "single_word": false,\n      "special": true\n    }\n  },\n '
        ' "eos_token": "',
        eos_token_ref,
        '"\n}\n',
    )
    write_file(String(base_dir, "/tokenizer_config.json"), tc)

    # 3. generation_config.json
    var gen_list_str = String("[")
    for i in range(len(gen_eos_ids)):
        if i > 0:
            gen_list_str = String(gen_list_str, ", ")
        gen_list_str = String(gen_list_str, String(gen_eos_ids[i]))
    gen_list_str = String(gen_list_str, "]")

    var gc = String('{\n  "eos_token_id": ', gen_list_str, "\n}\n")
    write_file(String(base_dir, "/generation_config.json"), gc)

    # 4. config.json
    var mc = String('{\n  "vocab_size": ', String(vocab_size), "\n}\n")
    write_file(String(base_dir, "/config.json"), mc)


def test_real_model_resolution() raises:
    """Menguji resolusi special token pada model aktual bila direktori tersedia."""
    var model_dir = getenv("DISMOEN_MODEL_DIR")
    if model_dir == "":
        var model_root = getenv("DISMOEN_MODEL_ROOT")
        if model_root != "":
            model_dir = String(model_root, "/qwen3.6-35b-a3b")
        else:
            var home = getenv("HOME")
            if home != "":
                model_dir = String(home, "/models/qwen3.6-35b-a3b")

    if model_dir == "":
        return

    # Cek apakah direktori model benar-benar memuat tokenizer.json
    try:
        var r = open(String(model_dir, "/tokenizer.json"), "r")
        r.close()
    except:
        # Model tidak ada di path default/env, lewati tes model nyata
        return

    # Sesuai M12-1: TIDAK BOLEH hardcode angka token ID. Seluruh pengujian relasional.
    var res = resolve_special_tokens(
        model_dir,
        vocab_size=248320,
        lock_path="models.lock.json",
        verify_lock=True,
    )

    assert_true(res.im_start_id > 0)
    assert_true(res.im_end_id > 0)
    assert_true(res.endoftext_id > 0)

    # Identitas token harus unik satu sama lain
    assert_true(res.im_start_id != res.im_end_id)
    assert_true(res.im_end_id != res.endoftext_id)
    assert_true(res.im_start_id != res.endoftext_id)

    # Batas rentang kosakata
    assert_true(res.im_start_id < 248320)
    assert_true(res.im_end_id < 248320)
    assert_true(res.endoftext_id < 248320)

    # Stop policy: im_end dan endoftext wajib merupakan stop token
    assert_true(res.is_stop_token(res.im_end_id))
    assert_true(res.is_stop_token(res.endoftext_id))

    # im_start BUKAN stop token
    assert_false(res.is_stop_token(res.im_start_id))

    # Himpunan stop token minimal memuat 2 ID (im_end dan endoftext)
    assert_true(len(res.eos_token_ids) >= 2)

    # eos_token_str harus merujuk ke token yang sah
    assert_true(
        res.eos_token_str == "<|im_end|>"
        or res.eos_token_str == "<|endoftext|>"
    )


def test_synthetic_happy_path() raises:
    """Menguji resolusi sukses pada lingkungan metadata sintetis terkontrol."""
    var fx_dir = "/tmp/dismoen_test_tok_happy"
    var base_id = 700
    var id_start = base_id + 1
    var id_end = base_id + 2
    var id_eot = base_id + 3
    var vocab_sz = base_id + 100

    var gen_eos = List[Int]()
    gen_eos.append(id_end)
    gen_eos.append(id_eot)

    setup_synthetic_fixture(
        fx_dir,
        im_start_id=id_start,
        im_end_id=id_end,
        eot_id=id_eot,
        dec_im_start_id=id_start,
        dec_im_end_id=id_end,
        dec_eot_id=id_eot,
        eos_token_ref="<|im_end|>",
        gen_eos_ids=gen_eos,
        vocab_size=vocab_sz,
    )

    var res = resolve_special_tokens(
        fx_dir, vocab_size=vocab_sz, verify_lock=False
    )

    assert_equal(res.im_start_id, id_start)
    assert_equal(res.im_end_id, id_end)
    assert_equal(res.endoftext_id, id_eot)
    assert_true(res.is_stop_token(id_end))
    assert_true(res.is_stop_token(id_eot))
    assert_false(res.is_stop_token(id_start))
    assert_equal(res.eos_token_str, "<|im_end|>")


def test_path_1_mapping_mismatch() raises:
    """Jalur 1: Mismatch pemetaan token ID antara tokenizer.json dan tokenizer_config.json."""
    var fx_dir = "/tmp/dismoen_test_tok_err_map"
    var base_id = 800
    var id_start = base_id + 1
    var id_end = base_id + 2
    var id_eot = base_id + 3
    var vocab_sz = base_id + 100

    var gen_eos = List[Int]()
    gen_eos.append(id_end)
    gen_eos.append(id_eot)

    # Sengaja dec_im_start_id berbeda dari id_start
    setup_synthetic_fixture(
        fx_dir,
        im_start_id=id_start,
        im_end_id=id_end,
        eot_id=id_eot,
        dec_im_start_id=id_start + 50,
        dec_im_end_id=id_end,
        dec_eot_id=id_eot,
        eos_token_ref="<|im_end|>",
        gen_eos_ids=gen_eos,
        vocab_size=vocab_sz,
    )

    var caught = False
    try:
        _ = resolve_special_tokens(
            fx_dir, vocab_size=vocab_sz, verify_lock=False
        )
    except e:
        var s = String(e)
        if s.find("CONFIG_MISMATCH") >= 0:
            caught = True
    assert_true(caught)


def test_path_2_invalid_eos_token_ref() raises:
    """Jalur 2: eos_token pada tokenizer_config.json bukan token khusus yang sah."""
    var fx_dir = "/tmp/dismoen_test_tok_err_eos"
    var base_id = 900
    var id_start = base_id + 1
    var id_end = base_id + 2
    var id_eot = base_id + 3
    var vocab_sz = base_id + 100

    var gen_eos = List[Int]()
    gen_eos.append(id_end)
    gen_eos.append(id_eot)

    # eos_token_ref invalid
    setup_synthetic_fixture(
        fx_dir,
        im_start_id=id_start,
        im_end_id=id_end,
        eot_id=id_eot,
        dec_im_start_id=id_start,
        dec_im_end_id=id_end,
        dec_eot_id=id_eot,
        eos_token_ref="<|invalid_token|>",
        gen_eos_ids=gen_eos,
        vocab_size=vocab_sz,
    )

    var caught = False
    try:
        _ = resolve_special_tokens(
            fx_dir, vocab_size=vocab_sz, verify_lock=False
        )
    except e:
        var s = String(e)
        if s.find("CONFIG_MISMATCH") >= 0:
            caught = True
    assert_true(caught)


def test_path_3_stop_policy_subset_violation() raises:
    """Jalur 3: generation_config.json:eos_token_id tidak memuat {im_end, endoftext}."""
    var fx_dir = "/tmp/dismoen_test_tok_err_stop"
    var base_id = 1100
    var id_start = base_id + 1
    var id_end = base_id + 2
    var id_eot = base_id + 3
    var vocab_sz = base_id + 100

    # Hanya memuat id_end, sengaja tidak memuat id_eot
    var gen_eos = List[Int]()
    gen_eos.append(id_end)

    setup_synthetic_fixture(
        fx_dir,
        im_start_id=id_start,
        im_end_id=id_end,
        eot_id=id_eot,
        dec_im_start_id=id_start,
        dec_im_end_id=id_end,
        dec_eot_id=id_eot,
        eos_token_ref="<|im_end|>",
        gen_eos_ids=gen_eos,
        vocab_size=vocab_sz,
    )

    var caught = False
    try:
        _ = resolve_special_tokens(
            fx_dir, vocab_size=vocab_sz, verify_lock=False
        )
    except e:
        var s = String(e)
        if s.find("CONFIG_MISMATCH") >= 0:
            caught = True
    assert_true(caught)


def test_path_4_range_violation_token_invalid() raises:
    """Jalur 4: Salah satu token ID >= vocab_size (pelanggaran rentang -> TOKEN_INVALID)."""
    var fx_dir = "/tmp/dismoen_test_tok_err_range"
    var base_id = 1200
    var id_start = base_id + 1
    var id_end = base_id + 2
    var id_eot = base_id + 3
    # Sengaja vocab_sz lebih kecil dari token ID
    var vocab_sz = base_id

    var gen_eos = List[Int]()
    gen_eos.append(id_end)
    gen_eos.append(id_eot)

    setup_synthetic_fixture(
        fx_dir,
        im_start_id=id_start,
        im_end_id=id_end,
        eot_id=id_eot,
        dec_im_start_id=id_start,
        dec_im_end_id=id_end,
        dec_eot_id=id_eot,
        eos_token_ref="<|im_end|>",
        gen_eos_ids=gen_eos,
        vocab_size=vocab_sz,
    )

    var caught = False
    try:
        _ = resolve_special_tokens(
            fx_dir, vocab_size=vocab_sz, verify_lock=False
        )
    except e:
        var s = String(e)
        if s.find("TOKEN_INVALID") >= 0:
            caught = True
    assert_true(caught)


def test_shifted_id_negative_test() raises:
    """Uji negatif: ID pada tokenizer.json digeser tanpa update tokenizer_config.json."""
    var fx_dir = "/tmp/dismoen_test_tok_shifted"
    var base_id = 1300
    var id_start = base_id + 1
    var id_end = base_id + 2
    var id_eot = base_id + 3
    var vocab_sz = base_id + 100

    var gen_eos = List[Int]()
    gen_eos.append(id_end + 10)
    gen_eos.append(id_eot + 10)

    # tokenizer.json memakai shifted ID, sedangkan decoder memakai unshifted
    setup_synthetic_fixture(
        fx_dir,
        im_start_id=id_start + 10,
        im_end_id=id_end + 10,
        eot_id=id_eot + 10,
        dec_im_start_id=id_start,
        dec_im_end_id=id_end,
        dec_eot_id=id_eot,
        eos_token_ref="<|im_end|>",
        gen_eos_ids=gen_eos,
        vocab_size=vocab_sz,
    )

    var caught = False
    try:
        _ = resolve_special_tokens(
            fx_dir, vocab_size=vocab_sz, verify_lock=False
        )
    except e:
        var s = String(e)
        if s.find("CONFIG_MISMATCH") >= 0:
            caught = True
    assert_true(caught)


def test_lockfile_tamper_detection() raises:
    """Verifikasi lockfile menolak berkas tokenizer yang hash SHA-256-nya tidak cocok."""
    var fx_dir = "/tmp/dismoen_test_tok_lock_tamper"
    var base_id = 1400
    var id_start = base_id + 1
    var id_end = base_id + 2
    var id_eot = base_id + 3
    var vocab_sz = base_id + 100

    var gen_eos = List[Int]()
    gen_eos.append(id_end)
    gen_eos.append(id_eot)

    setup_synthetic_fixture(
        fx_dir,
        im_start_id=id_start,
        im_end_id=id_end,
        eot_id=id_eot,
        dec_im_start_id=id_start,
        dec_im_end_id=id_end,
        dec_eot_id=id_eot,
        eos_token_ref="<|im_end|>",
        gen_eos_ids=gen_eos,
        vocab_size=vocab_sz,
    )

    # Buat lockfile dengan SHA-256 palsu / tampered
    var fake_lock_path = "/tmp/dismoen_test_tampered.lock.json"
    var fake_lock_content = (
        '{\n  "tokenizer_revision": "dummy_rev",\n  "tokenizer_sha256": {\n   '
        ' "tokenizer.json":'
        ' "0000000000000000000000000000000000000000000000000000000000000000",\n'
        '    "tokenizer_config.json":'
        ' "1111111111111111111111111111111111111111111111111111111111111111"\n'
        '  }\n}\n'
    )
    write_file(fake_lock_path, fake_lock_content)

    var caught = False
    try:
        verify_tokenizer_lockfile(fx_dir, fake_lock_path)
    except e:
        var s = String(e)
        if s.find("CONFIG_MISMATCH") >= 0:
            caught = True
    assert_true(caught)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_real_model_resolution]()
    suite.test[test_synthetic_happy_path]()
    suite.test[test_path_1_mapping_mismatch]()
    suite.test[test_path_2_invalid_eos_token_ref]()
    suite.test[test_path_3_stop_policy_subset_violation]()
    suite.test[test_path_4_range_violation_token_invalid]()
    suite.test[test_shifted_id_negative_test]()
    suite.test[test_lockfile_tamper_detection]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
