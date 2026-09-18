# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M12-W2b: Interactive Terminal REPL dismoen chat."""

from cli.cmd_chat import (
    clean_input_line,
    posix_read,
    resolve_chat_model_dir,
    sigint_handler,
)
from std.collections import List
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv, setenv
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)
from tokenizer.chatml import (
    ChatMessage,
    render_chatml,
)


def test_clean_input_line() raises:
    """Memverifikasi pembersihan string input dari whitespace, tab, dan null byte.
    """
    assert_equal(clean_input_line(""), "")
    assert_equal(clean_input_line("   "), "")
    assert_equal(clean_input_line("  /clear  \n"), "/clear")
    assert_equal(clean_input_line("\t/history\t\r"), "/history")
    assert_equal(clean_input_line("Halo dunia\n"), "Halo dunia")


def test_model_dir_precedence_explicit_flag() raises:
    """Memverifikasi bahwa flag --model-dir memiliki prioritas tertinggi."""
    var resolved = resolve_chat_model_dir("/custom/model/path")
    assert_equal(resolved, "/custom/model/path")


def test_model_dir_precedence_env_root() raises:
    """Memverifikasi fallback ke variabel lingkungan DISMOEN_MODEL_ROOT."""
    _ = setenv("DISMOEN_MODEL_ROOT", "/env/model/root", True)
    var resolved = resolve_chat_model_dir("")
    assert_equal(resolved, "/env/model/root")


def test_model_dir_precedence_default_home() raises:
    """Memverifikasi fallback ke direktori models default di HOME."""
    _ = setenv("DISMOEN_MODEL_ROOT", "", True)
    var home = getenv("HOME")
    var resolved = resolve_chat_model_dir("")
    assert_true(resolved.byte_length() > 0)
    if home.byte_length() > 0:
        assert_true(resolved.startswith(home))


def test_slash_command_dispatch() raises:
    """Memverifikasi identifikasi perintah slash interaktif."""
    var cmd1 = clean_input_line("/clear\n")
    assert_true(cmd1 == "/clear")

    var cmd2 = clean_input_line("  /history  ")
    assert_true(cmd2 == "/history")

    var cmd3 = clean_input_line("/exit")
    assert_true(cmd3 == "/exit" or cmd3 == "/quit")

    var cmd4 = clean_input_line("/help")
    assert_true(cmd4 == "/help")


def test_chatml_session_context_turn_flow() raises:
    """Memverifikasi akumulasi histori percakapan dan rendering subset ChatML.
    """
    var messages = List[ChatMessage]()
    messages.append(ChatMessage("system", "You are a helpful assistant."))
    assert_equal(len(messages), 1)

    # Turn 1
    messages.append(ChatMessage("user", "Pertanyaan 1"))
    var p1 = render_chatml(messages, add_generation_prompt=True)
    assert_true(
        p1.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        >= 0
    )
    assert_true(p1.find("<|im_start|>user\nPertanyaan 1<|im_end|>\n") >= 0)
    assert_true(p1.endswith("<|im_start|>assistant\n<think>\n"))

    # Turn 1 Assistant response
    messages.append(
        ChatMessage("assistant", "<think>\nReasoning 1\n</think>\nJawaban 1")
    )
    assert_equal(len(messages), 3)

    # Turn 2
    messages.append(ChatMessage("user", "Pertanyaan 2"))
    var p2 = render_chatml(messages, add_generation_prompt=True)
    # Think tag dari turn sebelumnya wajib di-strip pada histori kanonis
    assert_true(p2.find("<|im_start|>assistant\nJawaban 1<|im_end|>\n") >= 0)
    assert_true(p2.find("<|im_start|>user\nPertanyaan 2<|im_end|>\n") >= 0)


def test_self_pipe_signal_mechanism() raises:
    """Memverifikasi propagasi sinyal melalui self-pipe non-blocking."""
    var pipe_addr = external_call["malloc", Int](8)
    var p_pipe = Pointer[Int32, MutAnyOrigin](unsafe_from_address=pipe_addr)
    _ = external_call["pipe", Int32](p_pipe)
    var r_fd = p_pipe[unsafe_offset=0]
    var w_fd = p_pipe[unsafe_offset=1]

    _ = external_call["dup2", Int32](w_fd, Int32(88))
    _ = external_call["close", Int32](w_fd)
    _ = external_call["fcntl", Int32](r_fd, Int32(4), Int32(2048))  # O_NONBLOCK

    # Pasang handler
    _ = external_call["signal", Int](Int32(2), sigint_handler)

    # Verifikasi pipe kosong sebelum sinyal
    var buf_addr = external_call["malloc", Int](16)
    var n0 = posix_read(r_fd, buf_addr, 1)
    assert_equal(n0, -1)

    # Picu SIGINT
    _ = external_call["raise", Int32](Int32(2))

    # Verifikasi byte sinyal terbaca dari pipe
    var n1 = posix_read(r_fd, buf_addr, 1)
    assert_equal(n1, 1)

    # Bersihkan
    _ = external_call["close", Int32](r_fd)
    _ = external_call["close", Int32](Int32(88))
    external_call["free", NoneType](pipe_addr)
    external_call["free", NoneType](buf_addr)


def main() raises:
    print("Menjalankan Test Suite REPL dismoen chat (M12-W2b)...")
    var suite = TestSuite()
    suite.test[test_clean_input_line]()
    suite.test[test_model_dir_precedence_explicit_flag]()
    suite.test[test_model_dir_precedence_env_root]()
    suite.test[test_model_dir_precedence_default_home]()
    suite.test[test_slash_command_dispatch]()
    suite.test[test_chatml_session_context_turn_flow]()
    suite.test[test_self_pipe_signal_mechanism]()
    suite^.run()
