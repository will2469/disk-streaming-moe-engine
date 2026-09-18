# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M12-W1b: ChatML Template Formatter, Rejection, & Streaming Detokenizer.

Memverifikasi Gate G-M12-1:
- Replikasi 5 cabang normatif rendering prompt ChatML (subset text-only).
- 100% byte-exact match terhadap template Jinja upstream.
- Penolakan terstruktur seluruh input di luar subset (400/422 + unsupported: ...).
- Streaming detokenizer UTF-8 boundary multi-byte aman tanpa korupsi byte.
- ZERO hardcoded token ID constants (M12-1).
"""

from std.collections import List
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from tokenizer.chatml import (
    ChatMessage,
    extract_assistant_think,
    extract_last_query_index,
    render_chatml,
)
from tokenizer.detokenizer import StreamingDetokenizer
from tokenizer.special_tokens import verify_tokenizer_lockfile


def test_chatml_single_turn() raises:
    """Verifikasi rendering satu giliran (system + user + generation suffix)."""
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("system", "You are a helpful assistant."))
    msgs.append(ChatMessage("user", "Halo, siapa kamu?"))

    var act = render_chatml(msgs, add_generation_prompt=True)
    var exp = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        + "<|im_start|>user\nHalo, siapa kamu?<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n"
    )
    assert_equal(act, exp)


def test_chatml_multi_turn() raises:
    """Verifikasi rendering multi-turn dengan riwayat assistant lampau."""
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("system", "You are a helpful assistant."))
    msgs.append(ChatMessage("user", "Halo, siapa kamu?"))
    msgs.append(ChatMessage("assistant", "Saya Dismoen."))
    msgs.append(ChatMessage("user", "Lanjut."))

    var act = render_chatml(msgs, add_generation_prompt=True)
    var exp = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        + "<|im_start|>user\nHalo, siapa kamu?<|im_end|>\n"
        + "<|im_start|>assistant\nSaya Dismoen.<|im_end|>\n"
        + "<|im_start|>user\nLanjut.<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n"
    )
    assert_equal(act, exp)


def test_chatml_whitespace_trim() raises:
    """Verifikasi perlakuan whitespace-trimming pada seluruh role pesan."""
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("system", "  System message with spaces  \n"))
    msgs.append(
        ChatMessage("user", "\n\n  User message with tabs and newlines\t\n  ")
    )

    var act = render_chatml(msgs, add_generation_prompt=True)
    var exp = (
        "<|im_start|>system\nSystem message with spaces<|im_end|>\n"
        + "<|im_start|>user\nUser message with tabs and newlines<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n"
    )
    assert_equal(act, exp)


def test_chatml_think_strip_before_last_query() raises:
    """Verifikasi blok think pada assistant lampau dibuang persis upstream."""
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("user", "Q1"))
    msgs.append(
        ChatMessage("assistant", "<think>\nThinking step 1\n</think>\nAnswer 1")
    )
    msgs.append(ChatMessage("user", "Q2"))

    var act = render_chatml(msgs, add_generation_prompt=True)
    var exp = (
        "<|im_start|>user\nQ1<|im_end|>\n"
        + "<|im_start|>assistant\nAnswer 1<|im_end|>\n"
        + "<|im_start|>user\nQ2<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n"
    )
    assert_equal(act, exp)


def test_chatml_think_wrap_trailing_assistant() raises:
    """Verifikasi trailing assistant dengan tag think dibungkus utuh."""
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("user", "Q1"))
    msgs.append(
        ChatMessage("assistant", "<think>\nThinking step 1\n</think>\nAnswer 1")
    )

    var act = render_chatml(msgs, add_generation_prompt=True)
    var exp = (
        "<|im_start|>user\nQ1<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\nThinking step"
        " 1\n</think>\n\nAnswer 1<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n"
    )
    assert_equal(act, exp)


def test_chatml_trailing_assistant_no_think() raises:
    """Verifikasi trailing assistant tanpa tag think dibungkus quirk think-kosong.
    """
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("user", "Q1"))
    msgs.append(ChatMessage("assistant", "Answer 1"))

    var act = render_chatml(msgs, add_generation_prompt=True)
    var exp = (
        "<|im_start|>user\nQ1<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer 1<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n"
    )
    assert_equal(act, exp)


def test_chatml_user_only_and_literal_think() raises:
    """Verifikasi user tanpa system serta tag think literal pada pesan user."""
    var msgs1 = List[ChatMessage]()
    msgs1.append(ChatMessage("user", "Hello world"))

    var act1 = render_chatml(msgs1, add_generation_prompt=True)
    var exp1 = (
        "<|im_start|>user\nHello world<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n"
    )
    assert_equal(act1, exp1)

    var msgs2 = List[ChatMessage]()
    msgs2.append(ChatMessage("user", "What does <think> mean?"))

    var act2 = render_chatml(msgs2, add_generation_prompt=True)
    var exp2 = (
        "<|im_start|>user\nWhat does <think> mean?<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n"
    )
    assert_equal(act2, exp2)


def test_chatml_tool_response_scanning() raises:
    """Verifikasi pesan berbungkus tool_response dilewati pemindai last query.
    """
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage("user", "<tool_response>result</tool_response>"))
    msgs.append(ChatMessage("user", "Analyze the result."))

    var act = render_chatml(msgs, add_generation_prompt=True)
    var exp = (
        "<|im_start|>user\n<tool_response>result</tool_response><|im_end|>\n"
        + "<|im_start|>user\nAnalyze the result.<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n"
    )
    assert_equal(act, exp)


def test_chatml_negative_rejections() raises:
    """Verifikasi 9 skenario input non-subset ditolak dengan format error terstruktur.
    """
    # 1. empty_messages
    var empty_msgs = List[ChatMessage]()
    var err1_caught = False
    try:
        _ = render_chatml(empty_msgs)
    except e:
        if String(e).find("unsupported: empty_messages") >= 0:
            err1_caught = True
    assert_true(err1_caught)

    # 2. system_not_first
    var sys_middle = List[ChatMessage]()
    sys_middle.append(ChatMessage("user", "hi"))
    sys_middle.append(ChatMessage("system", "sys"))
    var err2_caught = False
    try:
        _ = render_chatml(sys_middle)
    except e:
        if String(e).find("unsupported: system_not_first") >= 0:
            err2_caught = True
    assert_true(err2_caught)

    # 3. duplicate_system
    var sys_dup = List[ChatMessage]()
    sys_dup.append(ChatMessage("system", "s1"))
    sys_dup.append(ChatMessage("system", "s2"))
    sys_dup.append(ChatMessage("user", "hi"))
    var err3_caught = False
    try:
        _ = render_chatml(sys_dup)
    except e:
        if String(e).find("unsupported: duplicate_system") >= 0:
            err3_caught = True
    assert_true(err3_caught)

    # 4. no_user_query (hanya system)
    var no_user = List[ChatMessage]()
    no_user.append(ChatMessage("system", "sys"))
    var err4_caught = False
    try:
        _ = render_chatml(no_user)
    except e:
        if String(e).find("unsupported: no_user_query") >= 0:
            err4_caught = True
    assert_true(err4_caught)

    # 5. only-tool-response user query (tidak ada query manusia)
    var only_tool_resp = List[ChatMessage]()
    only_tool_resp.append(
        ChatMessage("user", "<tool_response>res</tool_response>")
    )
    var err5_caught = False
    try:
        _ = render_chatml(only_tool_resp)
    except e:
        if String(e).find("unsupported: no_user_query") >= 0:
            err5_caught = True
    assert_true(err5_caught)

    # 6. unsupported role tool
    var role_tool = List[ChatMessage]()
    role_tool.append(ChatMessage("tool", "call_res"))
    var err6_caught = False
    try:
        _ = render_chatml(role_tool)
    except e:
        if String(e).find("unsupported: role=tool") >= 0:
            err6_caught = True
    assert_true(err6_caught)

    # 7. unsupported custom role
    var role_bot = List[ChatMessage]()
    role_bot.append(ChatMessage("bot", "hello"))
    var err7_caught = False
    try:
        _ = render_chatml(role_bot)
    except e:
        if String(e).find("unsupported: role=bot") >= 0:
            err7_caught = True
    assert_true(err7_caught)

    # 8. reasoning_content field
    var field_reasoning = List[ChatMessage]()
    field_reasoning.append(
        ChatMessage(
            "user", "hi", reasoning_content="think", has_reasoning_content=True
        )
    )
    var err8_caught = False
    try:
        _ = render_chatml(field_reasoning)
    except e:
        if String(e).find("unsupported: field=reasoning_content") >= 0:
            err8_caught = True
    assert_true(err8_caught)

    # 9. tool_calls field
    var field_tools = List[ChatMessage]()
    field_tools.append(
        ChatMessage(
            "assistant",
            "calling",
            reasoning_content="",
            has_reasoning_content=False,
            has_tool_calls=True,
        )
    )
    var err9_caught = False
    try:
        _ = render_chatml(field_tools)
    except e:
        if String(e).find("unsupported: field=tool_calls") >= 0:
            err9_caught = True
    assert_true(err9_caught)


def test_streaming_detokenizer_ascii() raises:
    """Verifikasi decoding streaming ASCII polos tanpa buffering yang tertunda.
    """
    var detok = StreamingDetokenizer()
    var c1 = detok.feed_string("Hello ")
    assert_equal(c1, "Hello ")
    var c2 = detok.feed_string("world!")
    assert_equal(c2, "world!")
    assert_equal(detok.flush(), "")


def test_streaming_detokenizer_utf8_multibyte_split() raises:
    """Verifikasi penanganan potongan boundary UTF-8 multi-byte (2, 3, 4 bytes).
    """
    var detok = StreamingDetokenizer()

    # 1. Uji 2-byte: 'é' (0xC3, 0xA9) terbelah menjadi [0xC3] lalu [0xA9]
    var part1 = List[UInt8]()
    part1.append(0x63)  # 'c'
    part1.append(0x61)  # 'a'
    part1.append(0x66)  # 'f'
    part1.append(0xC3)  # Byte pertama dari 'é' (belum lengkap)
    var out1 = detok.feed_bytes(part1)
    assert_equal(out1, "caf")  # 'é' harus tertahan

    var part2 = List[UInt8]()
    part2.append(0xA9)  # Byte kedua dari 'é'
    part2.append(0x20)  # spasi
    var out2 = detok.feed_bytes(part2)
    assert_equal(out2, "é ")  # Sekarang 'é' dipancarkan utuh

    # 2. Uji 3-byte CJK: '你' (0xE4, 0xBD, 0xA0) terbelah 1 byte per feed
    var b_cjk1 = List[UInt8]()
    b_cjk1.append(0xE4)
    assert_equal(detok.feed_bytes(b_cjk1), "")

    var b_cjk2 = List[UInt8]()
    b_cjk2.append(0xBD)
    assert_equal(detok.feed_bytes(b_cjk2), "")

    var b_cjk3 = List[UInt8]()
    b_cjk3.append(0xA0)
    b_cjk3.append(0x21)  # '!'
    assert_equal(detok.feed_bytes(b_cjk3), "你!")

    # 3. Uji 4-byte Emoji: '😊' (0xF0, 0x9F, 0x98, 0x8A) terbelah 2 byte + 2 byte
    var b_em1 = List[UInt8]()
    b_em1.append(0xF0)
    b_em1.append(0x9F)
    assert_equal(detok.feed_bytes(b_em1), "")

    var b_em2 = List[UInt8]()
    b_em2.append(0x98)
    b_em2.append(0x8A)
    assert_equal(detok.feed_bytes(b_em2), "😊")


def test_streaming_detokenizer_flush_and_reset() raises:
    """Verifikasi fungsi flush dan reset pada sisa buffer tak lengkap."""
    var detok = StreamingDetokenizer()
    var b_inc = List[UInt8]()
    b_inc.append(0xF0)
    b_inc.append(0x9F)
    _ = detok.feed_bytes(b_inc)

    # Flush sisa byte
    var flushed = detok.flush()
    assert_true(flushed.byte_length() > 0)
    assert_equal(len(detok.buffer), 0)

    # Reset
    var b_dummy = List[UInt8]()
    b_dummy.append(0xC3)
    _ = detok.feed_bytes(b_dummy)
    detok.reset()
    assert_equal(len(detok.buffer), 0)


from std.ffi import external_call


def c_mkdir(path: String) -> Int:
    """Membuat direktori via libc mkdir."""
    var p = path.as_bytes()
    var p_z = List[UInt8]()
    p_z.reserve(len(p) + 1)
    for i in range(len(p)):
        p_z.append(p[i])
    p_z.append(0)
    return Int(external_call["mkdir", Int32](p_z.unsafe_ptr(), 0o777))


def test_tamper_id_and_token_resolution() raises:
    """Uji tamper-ID: memastikan resolver menolak perubahan ID dan lockfile."""
    var test_dir = "/tmp/dismoen_test_tamper_w1b_dir"
    _ = c_mkdir(test_dir)
    var f1 = open(String(test_dir, "/tokenizer.json"), "w")
    f1.write("{}")
    f1.close()
    var f2 = open(String(test_dir, "/tokenizer_config.json"), "w")
    f2.write("{}")
    f2.close()

    var fake_lock_path = "/tmp/dismoen_test_tamper_w1b.lock.json"
    var fake_lock_content = (
        '{\n  "tokenizer_revision": "dummy_rev",\n  "tokenizer_sha256": {\n'
        '    "tokenizer.json":'
        ' "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",\n'
        '    "tokenizer_config.json":'
        ' "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"\n'
        "  }\n}\n"
    )
    var f = open(fake_lock_path, "w")
    f.write(fake_lock_content)
    f.close()

    var caught = False
    try:
        verify_tokenizer_lockfile(test_dir, fake_lock_path)
    except e:
        if String(e).find("CONFIG_MISMATCH") >= 0:
            caught = True
    assert_true(caught)


def add_tests_to_suite(mut suite: TestSuite):
    suite.test[test_chatml_single_turn]()
    suite.test[test_chatml_multi_turn]()
    suite.test[test_chatml_whitespace_trim]()
    suite.test[test_chatml_think_strip_before_last_query]()
    suite.test[test_chatml_think_wrap_trailing_assistant]()
    suite.test[test_chatml_trailing_assistant_no_think]()
    suite.test[test_chatml_user_only_and_literal_think]()
    suite.test[test_chatml_tool_response_scanning]()
    suite.test[test_chatml_negative_rejections]()
    suite.test[test_streaming_detokenizer_ascii]()
    suite.test[test_streaming_detokenizer_utf8_multibyte_split]()
    suite.test[test_streaming_detokenizer_flush_and_reset]()
    suite.test[test_tamper_id_and_token_resolution]()


def main() raises:
    var suite = TestSuite()
    add_tests_to_suite(suite)
    suite^.run()
