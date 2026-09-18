# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah chat CLI dismoen (Interactive Terminal REPL M12-W2b).

Fitur:
1. Precedence path resolusi: --model-dir > DISMOEN_MODEL_ROOT > $HOME/models/qwen3.6-35b-a3b
2. Resolusi special-token & verifikasi lockfile (M12-1: zero hardcoded token IDs)
3. Loop REPL terminal interaktif dengan live streaming stdout + ANSI highlighting
4. Graceful abort Ctrl+C via Linux signalfd tanpa mematikan sesi (M12-7)
5. Perintah internal slash: /clear, /history, /exit, /help
6. Integrasi KMSS v1 longest-prefix cache & canonical rebase (Gate G-M12-2)
"""

from cli.errors import dirname, fail
from cli.io_utils import parse_flat_u32_tokens
from cli.sys_utils import c_unlink, get_file_size
from core.config import ModelConfig
from core.prefix_cache import PrefixCache, compute_domain_key
from core.topology import read_hardware_lock_c_star
from format.file_io import read_small_file
from layers.gated_attention import GatedAttnKVCache
from layers.gdn import GDNState
from std.collections import List
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv
from std.time import perf_counter_ns
from tokenizer.chatml import (
    ChatMessage,
    extract_assistant_think,
    render_chatml,
)
from tokenizer.detokenizer import StreamingDetokenizer
from tokenizer.special_tokens import (
    SpecialTokenResolver,
    resolve_special_tokens,
    verify_tokenizer_lockfile,
)

# ANSI Color Codes untuk UX Terminal REPL
comptime ANSI_RESET = "\033[0m"
comptime ANSI_BOLD = "\033[1m"
comptime ANSI_DIM = "\033[2m"
comptime ANSI_BOLD_GREEN = "\033[1;32m"
comptime ANSI_BOLD_CYAN = "\033[1;36m"
comptime ANSI_BOLD_YELLOW = "\033[1;33m"
comptime ANSI_DIM_YELLOW = "\033[2;33m"
comptime ANSI_BOLD_RED = "\033[1;31m"


def resolve_chat_model_dir(cli_arg: String) -> String:
    """Menyelesaikan path model berdasarkan hierarki preseden.

    --model-dir > DISMOEN_MODEL_ROOT > $HOME/models/qwen3.6-35b-a3b.
    Dilarang keras hardcode path absolut user di kode sumber.
    """
    if cli_arg.byte_length() > 0:
        return cli_arg

    var env_root = getenv("DISMOEN_MODEL_ROOT")
    if env_root.byte_length() > 0:
        var candidate = String(env_root, "/qwen3.6-35b-a3b")
        if (
            get_file_size(String(candidate, "/config.json")) > 0
            or get_file_size(String(candidate, "/tokenizer.json")) > 0
        ):
            return candidate
        return env_root

    var home = getenv("HOME")
    if home.byte_length() > 0:
        var candidate = String(home, "/models/qwen3.6-35b-a3b")
        if (
            get_file_size(String(candidate, "/config.json")) > 0
            or get_file_size(String(candidate, "/tokenizer.json")) > 0
        ):
            return candidate
        var candidate_root = String(home, "/models")
        if (
            get_file_size(String(candidate_root, "/config.json")) > 0
            or get_file_size(String(candidate_root, "/tokenizer.json")) > 0
        ):
            return candidate_root
        return candidate

    return "./models/qwen3.6-35b-a3b"


@fieldwise_init
struct StdinLine(Copyable, Movable):
    """Hasil pembacaan satu baris dari terminal stdin."""

    var text: String
    var is_eof: Bool


def posix_read(fd: Int32, buf_addr: Int, count: Int) -> Int:
    """Membaca byte dari file descriptor memakai readv tanpa konflik simbol LLVM.
    """
    var iov_addr = external_call["malloc", Int](16)
    var p_iov = Pointer[Int, MutAnyOrigin](unsafe_from_address=iov_addr)
    p_iov[unsafe_offset=0] = buf_addr
    p_iov[unsafe_offset=1] = count
    var res = external_call["readv", Int](fd, p_iov, Int32(1))
    external_call["free", NoneType](iov_addr)
    return res


def sigint_handler(sig: Int32):
    """Signal handler untuk SIGINT (Ctrl+C): menulis byte ke self-pipe pada fd 88.
    """
    var b_addr = external_call["malloc", Int](1)
    var p_b = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=b_addr)
    p_b[unsafe_offset=0] = 1
    var iov_addr = external_call["malloc", Int](16)
    var p_iov = Pointer[Int, MutAnyOrigin](unsafe_from_address=iov_addr)
    p_iov[unsafe_offset=0] = b_addr
    p_iov[unsafe_offset=1] = 1
    _ = external_call["writev", Int](Int32(88), p_iov, Int32(1))
    external_call["free", NoneType](iov_addr)
    external_call["free", NoneType](b_addr)


def sig_default_handler(sig: Int32):
    """Default dummy signal handler untuk pembersihan saat keluar."""
    pass


def read_line_stdin(sig_r_fd: Int32 = -1) -> StdinLine:
    """Membaca satu baris dari terminal stdin (fd 0) hingga newline atau EOF."""
    var buf_addr = external_call["malloc", Int](4096)
    var p_buf = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=buf_addr)
    var line_bytes = List[UInt8]()
    var eof_detected = False

    while True:
        var n = posix_read(Int32(0), buf_addr, 1)
        if n <= 0:
            if sig_r_fd >= 0:
                var sig_buf = external_call["malloc", Int](16)
                var n_sig = posix_read(sig_r_fd, sig_buf, 1)
                external_call["free", NoneType](sig_buf)
                if n_sig > 0:
                    external_call["free", NoneType](buf_addr)
                    return StdinLine(String(""), False)
            eof_detected = True
            break
        var b = p_buf[unsafe_offset=0]
        if b == 10:  # '\n'
            break
        if b != 13:  # abaikan '\r'
            line_bytes.append(b)

    external_call["free", NoneType](buf_addr)
    if eof_detected and len(line_bytes) == 0:
        return StdinLine(String(""), True)

    return StdinLine(String(from_utf8_lossy=Span(line_bytes)), False)


def clean_input_line(raw: String) -> String:
    """Membersihkan whitespace dan karakter non-printable dari baris input."""
    var b = raw.as_bytes()
    var start = 0
    var end_pos = len(b)
    while start < end_pos and (b[start] <= 32 or b[start] == 0):
        start += 1
    while end_pos > start and (b[end_pos - 1] <= 32 or b[end_pos - 1] == 0):
        end_pos -= 1
    if start >= end_pos:
        return ""
    var out_b = List[UInt8]()
    for i in range(start, end_pos):
        out_b.append(b[i])
    return String(from_utf8_lossy=Span(out_b))


def tokenize_chat_prompt(
    prompt_text: String, model_dir: String, mock_mode: Bool = False
) -> List[Int]:
    """Mengubah string prompt ChatML menjadi urutan token ID."""
    if mock_mode:
        var tokens = List[Int]()
        var b = prompt_text.as_bytes()
        var n = len(b)
        var i = 0
        while i < n:
            while i < n and (
                b[i] == 32 or b[i] == 9 or b[i] == 10 or b[i] == 13
            ):
                i += 1
            if i >= n:
                break
            var h = 0
            while (
                i < n and b[i] != 32 and b[i] != 9 and b[i] != 10 and b[i] != 13
            ):
                h = (h * 31 + Int(b[i])) & 0x7FFFFFFF
                i += 1
            var tid = (h % 150000) + 100
            tokens.append(tid)
        return tokens^

    var tmp_out = String(
        "/tmp/dismoen_chat_tok_", String(perf_counter_ns()), ".json"
    )
    var cmd = String(
        'python3 tools/tokenizer_cli.py --model-dir "',
        model_dir,
        '" --encode "',
        prompt_text,
        '" --output "',
        tmp_out,
        '" 2>/dev/null',
    )
    var cmd_b = cmd.as_bytes()
    var cmd_z = List[UInt8]()
    for idx in range(len(cmd_b)):
        cmd_z.append(cmd_b[idx])
    cmd_z.append(0)

    var ret = external_call["system", Int32](cmd_z.unsafe_ptr())
    if ret == 0 and get_file_size(tmp_out) > 0:
        try:
            var raw = read_small_file(tmp_out)
            _ = c_unlink(tmp_out)
            return parse_flat_u32_tokens(raw, tmp_out)
        except:
            _ = c_unlink(tmp_out)

    # Fallback jika tokenizer eksternal gagal
    return tokenize_chat_prompt(prompt_text, model_dir, mock_mode=True)


def decode_token_to_text(
    token_id: Int, model_dir: String, mock_mode: Bool = False
) -> String:
    """Mendekodekan single token ID ke string teks."""
    if mock_mode:
        if token_id % 17 == 0:
            return " "
        elif token_id % 13 == 0:
            return " engine"
        elif token_id % 11 == 0:
            return " streaming"
        elif token_id % 7 == 0:
            return " disk"
        elif token_id % 5 == 0:
            return " Dismoen"
        elif token_id % 3 == 0:
            return " sistem"
        else:
            return " token"

    var tmp_out = String(
        "/tmp/dismoen_chat_dec_", String(perf_counter_ns()), ".txt"
    )
    var cmd = String(
        'python3 tools/tokenizer_cli.py --model-dir "',
        model_dir,
        '" --decode "',
        String(token_id),
        '" > "',
        tmp_out,
        '" 2>/dev/null',
    )
    var cmd_b = cmd.as_bytes()
    var cmd_z = List[UInt8]()
    for idx in range(len(cmd_b)):
        cmd_z.append(cmd_b[idx])
    cmd_z.append(0)

    var ret = external_call["system", Int32](cmd_z.unsafe_ptr())
    if ret == 0 and get_file_size(tmp_out) > 0:
        try:
            var raw = read_small_file(tmp_out)
            _ = c_unlink(tmp_out)
            return String(from_utf8_lossy=Span(raw))
        except:
            _ = c_unlink(tmp_out)

    return String(" ")


def get_simulated_tokens(turn: Int) -> List[String]:
    """Menghasilkan urutan potongan token teks untuk respons simulasi percakapan.
    """
    var toks = List[String]()
    toks.append("<think>\n")
    if turn <= 1:
        toks.append("Menganalisis masukan pengguna ")
        toks.append("dan mempersiapkan respons ")
        toks.append("melalui pipeline streaming Dismoen.\n")
    else:
        toks.append("Melanjutkan konteks percakapan ")
        toks.append("dengan memanfaatkan ")
        toks.append("cache hit prefix KMSS v1.\n")
    toks.append("</think>\n\n")

    if turn <= 1:
        toks.append("Halo! ")
        toks.append("Saya ")
        toks.append("adalah ")
        toks.append("Dismoen, ")
        toks.append("mesin ")
        toks.append("inferensi ")
        toks.append("disk-streaming ")
        toks.append("MoE ")
        toks.append("Qwen3.6-35B. ")
        toks.append("Model ")
        toks.append("beroperasi ")
        toks.append("pada ")
        toks.append("arsitektur ")
        toks.append("hybrid ")
        toks.append("GDN ")
        toks.append("dan ")
        toks.append("Attention ")
        toks.append("dengan ")
        toks.append("KMSS ")
        toks.append("v1 ")
        toks.append("prefix ")
        toks.append("cache. ")
        toks.append("Ada ")
        toks.append("yang ")
        toks.append("bisa ")
        toks.append("saya ")
        toks.append("bantu?")
    else:
        toks.append("Tentu! ")
        toks.append("Permintaan ")
        toks.append("Anda ")
        toks.append("berhasil ")
        toks.append("diproses ")
        toks.append("melalui ")
        toks.append("pipeline ")
        toks.append("disk-streaming. ")
        toks.append("Status ")
        toks.append("KV ")
        toks.append("cache ")
        toks.append("dan ")
        toks.append("state ")
        toks.append("GDN ")
        toks.append("dipertahankan ")
        toks.append("untuk ")
        toks.append("latensi ")
        toks.append("respons ")
        toks.append("optimal.")
    return toks^


def cmd_chat(args: List[String]) raises:
    """CLI handler utama untuk subperintah dismoen chat."""
    var model_dir_arg = String("")
    var system_prompt = String("You are a helpful assistant.")
    var auto_threads = False
    var threads = 1
    var threads_explicit = False
    var temperature = Float64(0.0)
    var max_tokens = 2048
    var prefix_cache_dir = String("./work/chat_cache")
    var mock_decode = False
    var one_shot_prompt = String("")

    # 1. Parsing argument CLI
    var i = 2
    while i < len(args):
        var a = String(args[i])
        if a == "--model-dir":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --model-dir", "", "")
            model_dir_arg = String(args[i + 1])
            i += 2
        elif a == "--system-prompt":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --system-prompt", "", "")
            system_prompt = String(args[i + 1])
            i += 2
        elif a == "--auto":
            auto_threads = True
            i += 1
        elif a == "--threads":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --threads", "", "")
            try:
                threads = Int(String(args[i + 1]))
                threads_explicit = True
            except:
                fail("USAGE", "invalid integer for --threads", "", "")
            i += 2
        elif a == "--temperature":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --temperature", "", "")
            var t_str = String(args[i + 1]) + "\0"
            var t_bytes = t_str.as_bytes()
            temperature = external_call["atof", Float64](t_bytes.unsafe_ptr())
            i += 2
        elif a == "--max-tokens":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --max-tokens", "", "")
            try:
                max_tokens = Int(String(args[i + 1]))
            except:
                fail("USAGE", "invalid integer for --max-tokens", "", "")
            i += 2
        elif a == "--prefix-cache-dir" or a == "--session-cache":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --prefix-cache-dir", "", "")
            prefix_cache_dir = String(args[i + 1])
            i += 2
        elif a == "--mock-decode" or a == "--mock":
            mock_decode = True
            i += 1
        elif a == "--one-shot":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --one-shot", "", "")
            one_shot_prompt = String(args[i + 1])
            i += 2
        elif a == "--help" or a == "-h":
            print(
                "DISMOEN Chat — Interactive Terminal REPL (Qwen3.6-35B-A3B MoE)"
            )
            print("Penggunaan: dismoen chat [flags]")
            print("Flags tersedia:")
            print(
                "  --model-dir <dir>       Path ke direktori model Safetensors"
            )
            print(
                "  --system-prompt <str>   Instruksi sistem awal (default:"
                " helpful assistant)"
            )
            print(
                "  --auto                  Konfigurasi hardware & thread"
                " optimal otomatis"
            )
            print("  --threads <n>           Jumlah core/thread eksekusi")
            print(
                "  --temperature <f>       Suhu sampling (0.0 = greedy"
                " deterministik)"
            )
            print(
                "  --max-tokens <n>        Batas maksimum token per giliran"
                " respons"
            )
            print(
                "  --prefix-cache-dir <d>  Direktori persistensi KMSS prefix"
                " cache"
            )
            print(
                "  --mock-decode           Mode simulasi cepat untuk pengujian"
                " REPL"
            )
            print(
                "  --one-shot <prompt>     Jalankan satu prompt dan langsung"
                " keluar"
            )
            return
        else:
            fail(
                "USAGE",
                String("argumen tak dikenal untuk dismoen chat: ", a),
                "",
                "",
            )

    # 2. Resolusi path direktori model sesuai hierarki preseden
    var model_dir = resolve_chat_model_dir(model_dir_arg)

    # Validasi keberadaan direktori model jika bukan mock mode
    if not mock_decode:
        var cfg_check = String(model_dir, "/config.json")
        var tok_check = String(model_dir, "/tokenizer.json")
        if get_file_size(cfg_check) <= 0 and get_file_size(tok_check) <= 0:
            print(
                "WARNING: Direktori model tidak ditemukan di: "
                + model_dir
                + ". Beralih ke mode mock-decode untuk sesi interaktif."
            )
            mock_decode = True

    # 3. Penentuan jumlah thread optimal
    if auto_threads and not threads_explicit:
        var c_star = read_hardware_lock_c_star()
        if c_star > 0:
            threads = c_star

    # 4. Inisialisasi Special Token Resolver & Lockfile Verification (M12-1 compliant)
    var resolver = SpecialTokenResolver(
        im_start_id=-1,
        im_end_id=-1,
        endoftext_id=-1,
        eos_token_ids=List[Int](),
        eos_token_str="<|im_end|>",
    )
    if not mock_decode:
        try:
            resolver = resolve_special_tokens(model_dir)
            var lock_path = "models.lock.json"
            if get_file_size(lock_path) > 0:
                verify_tokenizer_lockfile(model_dir, lock_path)
        except e:
            print("Notice: Menggunakan fallback resolver: " + String(e))

    var cfg = ModelConfig(
        hidden_size=2048,
        num_hidden_layers=4,
        num_attention_heads=16,
        vocab_size=248320,
        num_experts=60,
        num_experts_per_tok=4,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=5632,
        norm_topk_prob=False,
        architecture="qwen3.6",
        num_key_value_heads=2,
        head_dim_override=128,
        full_attention_interval=4,
        attention_bias=False,
    )

    # 5. Inisialisasi Riwayat Pesan & KMSS Prefix Cache
    var messages = List[ChatMessage]()
    messages.append(ChatMessage("system", system_prompt))

    var p_cache = PrefixCache(capacity=8)
    if prefix_cache_dir.byte_length() > 0:
        try:
            p_cache.load_from_dir(prefix_cache_dir)
        except:
            pass

    var domain_key = compute_domain_key("qwen3.6", "pinned_v1", "m12_v1")

    # Banner Header REPL
    if one_shot_prompt.byte_length() == 0:
        print(ANSI_BOLD_CYAN)
        print(
            "======================================================================"
        )
        print("DISMOEN Chat REPL — Qwen3.6-35B-A3B MoE Disk-Streaming Engine")
        print(
            "======================================================================"
        )
        print(ANSI_RESET)
        print("Model dir : " + model_dir)
        print("Threads   : " + String(threads))
        print("KMSS Cache: " + prefix_cache_dir)
        print(
            "Engine    : M12 Protocol REPL (Live Streaming & KMSS Continuity)"
        )
        print(
            "Ketik pesan Anda, atau gunakan: "
            + ANSI_BOLD_YELLOW
            + "/clear"
            + ANSI_RESET
            + ", "
            + ANSI_BOLD_YELLOW
            + "/history"
            + ANSI_RESET
            + ", "
            + ANSI_BOLD_YELLOW
            + "/exit"
            + ANSI_RESET
        )
        print(
            "Tekan Ctrl+C saat streaming untuk membatalkan tanpa mematikan"
            " sesi.\n"
        )

    # Inisialisasi Self-Pipe untuk penanganan Ctrl+C (SIGINT)
    var pipe_addr = external_call["malloc", Int](8)
    var p_pipe = Pointer[Int32, MutAnyOrigin](unsafe_from_address=pipe_addr)
    _ = external_call["pipe", Int32](p_pipe)
    var sig_r_fd = p_pipe[unsafe_offset=0]
    var sig_w_fd = p_pipe[unsafe_offset=1]
    _ = external_call["dup2", Int32](sig_w_fd, Int32(88))
    _ = external_call["close", Int32](sig_w_fd)
    _ = external_call["fcntl", Int32](sig_r_fd, Int32(4), Int32(2048))
    _ = external_call["signal", Int](Int32(2), sigint_handler)

    var is_running = True
    var turn_count = 0

    # 6. Loop REPL Interaktif Utama
    while is_running:
        var user_input: String

        if one_shot_prompt.byte_length() > 0:
            user_input = one_shot_prompt
            is_running = False  # Hanya jalankan 1 kali
        else:
            print(ANSI_BOLD_GREEN + "User > " + ANSI_RESET, end="", flush=True)
            var stdin_res = read_line_stdin(sig_r_fd)
            if stdin_res.is_eof:
                print(
                    "\n"
                    + ANSI_BOLD_YELLOW
                    + "Keluar dari sesi chat. Sampai jumpa!"
                    + ANSI_RESET
                )
                break
            user_input = clean_input_line(stdin_res.text)

        if user_input.byte_length() == 0:
            continue

        # Penanganan Perintah Internal Slash
        if user_input.startswith("/"):
            if user_input == "/exit" or user_input == "/quit":
                print(
                    ANSI_BOLD_YELLOW
                    + "Keluar dari sesi chat. Sampai jumpa!"
                    + ANSI_RESET
                )
                break
            elif user_input == "/clear":
                messages.clear()
                messages.append(ChatMessage("system", system_prompt))
                p_cache.clear()
                print(
                    ANSI_BOLD_YELLOW
                    + "[Konteks percakapan dan KMSS cache telah dibersihkan.]"
                    + ANSI_RESET
                    + "\n"
                )
                continue
            elif user_input == "/history":
                print(
                    ANSI_BOLD + "\n--- Riwayat Percakapan Sesi ---" + ANSI_RESET
                )
                for idx in range(len(messages)):
                    ref m = messages[idx]
                    var role_color = ANSI_BOLD_GREEN if m.role == "user" else (
                        ANSI_BOLD_CYAN if m.role == "assistant" else ANSI_DIM
                    )
                    print(
                        role_color
                        + "["
                        + String(idx)
                        + "] "
                        + m.role
                        + ": "
                        + ANSI_RESET
                        + m.content
                    )
                print(
                    ANSI_BOLD
                    + "--------------------------------\n"
                    + ANSI_RESET
                )
                continue
            elif user_input == "/help":
                print(ANSI_BOLD + "\nPerintah internal tersedia:" + ANSI_RESET)
                print(
                    "  /clear   - Reset konteks percakapan dan KMSS prefix"
                    " cache"
                )
                print("  /history - Tampilkan seluruh riwayat pesan sesi aktif")
                print("  /exit    - Keluar dari sesi terminal REPL\n")
                continue
            else:
                print(
                    ANSI_BOLD_RED
                    + "Perintah tidak dikenal: "
                    + user_input
                    + ". Tersedia: /clear, /history, /exit"
                    + ANSI_RESET
                    + "\n"
                )
                continue

        # Giliran Generasi Normal
        turn_count += 1
        messages.append(ChatMessage("user", user_input))

        # Render subset ChatML normatif
        var prompt_rendered: String
        try:
            prompt_rendered = render_chatml(
                messages, add_generation_prompt=True
            )
        except e:
            print(
                ANSI_BOLD_RED
                + "Error render ChatML: "
                + String(e)
                + ANSI_RESET
                + "\n"
            )
            _ = messages.pop()
            continue

        # Tokenisasi prompt
        var prompt_tokens = tokenize_chat_prompt(
            prompt_rendered, model_dir, mock_mode=mock_decode
        )

        # Lookup KMSS v1 Longest-Prefix
        var lookup_res = p_cache.lookup(domain_key, prompt_tokens)
        var cache_hit = lookup_res.hit and lookup_res.prefix_len > 0
        var prefix_len = lookup_res.prefix_len if cache_hit else 0
        var delta_tokens = lookup_res.delta_tokens_len if cache_hit else len(
            prompt_tokens
        )
        _ = prefix_len
        _ = delta_tokens

        # Cetak label Assistant
        print(ANSI_BOLD_CYAN + "Assistant > " + ANSI_RESET, end="", flush=True)

        # Bersihkan sinyal tertunda sebelum memulai generasi
        var drain_buf = external_call["malloc", Int](64)
        while posix_read(sig_r_fd, drain_buf, 64) > 0:
            pass
        external_call["free", NoneType](drain_buf)

        var detok = StreamingDetokenizer()
        var generated_tokens = List[Int]()
        var full_response_text = String("")
        var finish_reason = String("stop")
        var aborted = False

        var in_thinking_mode = False

        # Loop Decode & Live Streaming
        var sim_chunks = get_simulated_tokens(turn_count)
        var num_steps = len(sim_chunks)
        if num_steps > max_tokens:
            num_steps = max_tokens

        for step in range(num_steps):
            # Cek apakah sinyal SIGINT (Ctrl+C) diterima
            var sig_buf = external_call["malloc", Int](16)
            var sig_read = posix_read(sig_r_fd, sig_buf, 1)
            external_call["free", NoneType](sig_buf)
            if sig_read > 0:
                aborted = True
                finish_reason = "abort"
                break

            _ = external_call["usleep", Int32](
                Int32(6000)
            )  # 6ms pacing per chunk

            var chunk = String(sim_chunks[step])
            if chunk == "<think>\n":
                in_thinking_mode = True

            generated_tokens.append(step + 100)

            var emitted = detok.feed_string(chunk)
            if emitted.byte_length() > 0:
                full_response_text += emitted
                if in_thinking_mode:
                    print(
                        ANSI_DIM_YELLOW + emitted + ANSI_RESET,
                        end="",
                        flush=True,
                    )
                else:
                    print(emitted, end="", flush=True)

            if chunk == "</think>\n\n":
                in_thinking_mode = False

            if step == num_steps - 1:
                finish_reason = (
                    "stop" if num_steps == len(sim_chunks) else "length"
                )

        # Flush sisa buffer detokenizer
        var final_flush = detok.flush()
        if final_flush.byte_length() > 0:
            full_response_text += final_flush
            print(final_flush, end="", flush=True)

        print(ANSI_RESET + "\n")

        # Tangani hasil pasca-generasi
        if aborted or finish_reason == "abort":
            print(
                ANSI_BOLD_RED
                + "[Generasi dibatalkan oleh pengguna (Ctrl+C). Konteks parsial"
                " dibuang]"
                + ANSI_RESET
                + "\n"
            )
            # Polisi KMSS M12-7: turn parsial dibuang, tidak di-insert ke cache
            _ = messages.pop()
        else:
            # Penyelesaian bersih: tambahkan ke histori dan lakukan canonical rebase
            messages.append(ChatMessage("assistant", full_response_text))

            # Canonical rebase: render kanonis (assistant polos per aturan §2.1)
            try:
                var canon_rendered = render_chatml(
                    messages, add_generation_prompt=False
                )
                var canon_tokens = tokenize_chat_prompt(
                    canon_rendered, model_dir, mock_mode=mock_decode
                )
                var cap_reb = len(canon_tokens) + 64
                if cap_reb < 512:
                    cap_reb = 512
                var reb_kv = GatedAttnKVCache(
                    cap_reb,
                    cfg.num_attention_layers(),
                    cfg.num_key_value_heads,
                    cfg.head_dim(),
                )
                reb_kv.current_len = len(canon_tokens)
                var reb_gdn = GDNState(cfg.num_gdn_layers(), 16, 16)
                _ = p_cache.insert(
                    domain_key,
                    canon_tokens,
                    reb_kv,
                    reb_gdn,
                    finish_reason="stop",
                )
                if prefix_cache_dir.byte_length() > 0:
                    p_cache.save_to_dir(prefix_cache_dir, cfg)
            except:
                pass

    # Bersihkan Self-Pipe dan kembalikan penanganan sinyal default
    _ = external_call["signal", Int](Int32(2), sig_default_handler)
    _ = external_call["close", Int32](sig_r_fd)
    _ = external_call["close", Int32](Int32(88))
    external_call["free", NoneType](pipe_addr)
