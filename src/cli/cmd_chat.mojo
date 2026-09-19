# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah chat CLI dismoen (Interactive Terminal REPL M12-W2b, fix #3).

Kontrak non-negotiable (fix #3):
- Tokenisasi SELALU BPE real via helper HF atas tokenizer.json model
  (encode_via_hf / decode_via_hf). TIDAK ADA hash, TIDAK ADA fallback,
  TIDAK ADA respons canned-text.
- Inferensi SELALU komputasi GGUF-backed per turn (prefill delta +
  decode autoregresif + argmax), sama seperti jalur forward/decode.
  Tanpa --quant-model -> fail-closed NO_QUANTIZER_MODEL.
- Tanpa tokenizer.json / config yang tak-konsisten -> fail-closed di
  startup (bukan mode pura-pura).

Fitur yang dipertahankan:
1. Precedence path resolusi: --model-dir > DISMOEN_MODEL_ROOT > $HOME/models/qwen3.6-35b-a3b
2. Resolusi special-token & verifikasi lockfile (M12-1: zero hardcoded token IDs)
3. Loop REPL terminal interaktif dengan live streaming stdout + ANSI highlighting
4. Graceful abort Ctrl+C via self-pipe tanpa mematikan sesi (M12-7):
   abort mengembalikan snapshot KV/GDN pra-turn (state tidak tercemar)
5. Perintah internal slash: /clear, /history, /exit, /help
"""

from cli.config_parser import parse_model_config
from cli.cmd_decode import argmax_sample
from cli.errors import eprint_json, fail
from cli.sys_utils import get_file_size
from core.config import ModelConfig
from core.topology import read_hardware_lock_c_star
from core.worker_pool import WorkerPool
from format.gguf import parse_gguf_index
from layers.gated_attention import GatedAttnKVCache
from layers.gdn import GDNState
from layers.gguf_port_loader import (
    forward_port_macro_scheduler_gguf,
    gguf_embed_tokens,
    gguf_logits_from_hidden,
    resolve_quant_model_path,
    validate_gguf_port_coverage,
)
from layers.port_scheduler import SchedulerTimings
from std.collections import List
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv
from tokenizer.chatml import ChatMessage, render_chatml
from tokenizer.detokenizer import StreamingDetokenizer
from tokenizer.hf_client import decode_via_hf, encode_via_hf
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


def fail_chat(message: String) raises:
    """Gagal fail-closed dengan skema error M12 (tanpa fallback)."""
    raise Error(String('{"error":"M12_ERR_CHAT","message":"', message, '"}'))


def run_chat_turn(
    prompt_rendered: String,
    model_dir: String,
    resolver: SpecialTokenResolver,
    mut all_tokens: List[Int],
    mut kv_cache: GatedAttnKVCache,
    mut gdn_states: GDNState,
    gguf_path: String,
    cfg: ModelConfig,
    eps: Float32,
    max_tokens: Int,
    sig_r_fd: Int32,
    mut pool: WorkerPool,
) raises -> Tuple[List[Int], String, String]:
    """Satu turn inferensi REAL: BPE encode -> prefill delta GGUF ->
    decode autoregresif argmax -> BPE decode per token (live stream).

    Mengembalikan (generated_ids, response_text, finish_reason).
    finish_reason: "stop" (EOS) | "length" (max_tokens) | "abort" (Ctrl+C).
    State KV/GDN dimutasi maju; caller wajib snapshot sebelum turn untuk
    rollback bila abort (M12-7).
    """
    var gguf_index = parse_gguf_index(gguf_path)

    # 1. Encode REAL via BPE model (gagal -> raise, tanpa fallback).
    var prompt_tokens = encode_via_hf(prompt_rendered, model_dir)
    if len(prompt_tokens) == 0:
        fail_chat("prompt ter-encode menjadi 0 token (input kosong?)")

    var pos_base = len(all_tokens)
    for t in range(len(prompt_tokens)):
        all_tokens.append(prompt_tokens[t])

    # 2. Prefill delta GGUF-backed (stream per layer, discard).
    var dv = 32
    var dk = 32
    var x_pre = gguf_embed_tokens(gguf_index, prompt_tokens, cfg)
    var timings = SchedulerTimings()
    _ = forward_port_macro_scheduler_gguf(
        x_pre,
        gguf_index,
        gdn_states,
        kv_cache,
        pos_base,
        len(prompt_tokens),
        cfg,
        timings,
        pool,
        dk,
        dv,
        eps,
    )

    # 3. Decode autoregresif greedy + live streaming teks real.
    var detok = StreamingDetokenizer()
    var generated = List[Int]()
    var full_response = String("")
    var finish_reason = String("length")
    var last_tok = prompt_tokens[len(prompt_tokens) - 1]

    print(ANSI_BOLD_CYAN + "Assistant > " + ANSI_RESET, end="", flush=True)

    # Bersihkan sinyal tertunda sebelum memulai generasi
    var drain_buf = external_call["malloc", Int](64)
    while posix_read(sig_r_fd, drain_buf, 64) > 0:
        pass
    external_call["free", NoneType](drain_buf)

    for step in range(max_tokens):
        # Cek SIGINT (Ctrl+C) tiap langkah.
        var sig_buf = external_call["malloc", Int](16)
        var sig_read = posix_read(sig_r_fd, sig_buf, 1)
        external_call["free", NoneType](sig_buf)
        if sig_read > 0:
            finish_reason = "abort"
            break

        var one_id = List[Int]()
        one_id.append(last_tok)
        var one_x = gguf_embed_tokens(gguf_index, one_id, cfg)
        var one_hidden = forward_port_macro_scheduler_gguf(
            one_x,
            gguf_index,
            gdn_states,
            kv_cache,
            pos_base + len(prompt_tokens) + step,
            1,
            cfg,
            timings,
            pool,
            dk,
            dv,
            eps,
        )
        var one_logits = gguf_logits_from_hidden(
            gguf_index, one_hidden, 1, cfg, eps
        )
        _ = one_hidden^
        var gen_tok = argmax_sample(one_logits, cfg.vocab_size)
        _ = one_logits^

        if resolver.is_stop_token(gen_tok):
            finish_reason = "stop"
            break

        generated.append(gen_tok)
        all_tokens.append(gen_tok)
        last_tok = gen_tok

        var gen_id = List[Int]()
        gen_id.append(gen_tok)
        var piece = decode_via_hf(gen_id, model_dir)
        var emitted = detok.feed_string(piece)
        full_response += emitted
        if emitted.byte_length() > 0:
            var think_open = full_response.find("<think>") >= 0
            var think_shut = full_response.find("</think>") >= 0
            if think_open and not think_shut:
                print(
                    ANSI_DIM_YELLOW + emitted + ANSI_RESET,
                    end="",
                    flush=True,
                )
            else:
                print(emitted, end="", flush=True)

        if step == max_tokens - 1:
            finish_reason = "length"

    var final_flush = detok.flush()
    if final_flush.byte_length() > 0:
        full_response += final_flush
        print(final_flush, end="", flush=True)
    print(ANSI_RESET + "\n")

    if finish_reason == "abort":
        return (List[Int](), String(""), String("abort"))
    return (generated^, full_response, finish_reason)


def cmd_chat(args: List[String]) raises:
    """CLI handler utama untuk subperintah dismoen chat (REAL, fix #3)."""
    var model_dir_arg = String("")
    var system_prompt = String("You are a helpful assistant.")
    var auto_threads = False
    var threads = 1
    var threads_explicit = False
    var temperature = Float64(0.0)
    var max_tokens = 64
    var quant_model_arg = String("")
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
        elif a == "--quant-model":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --quant-model", "", "")
            quant_model_arg = String(args[i + 1])
            i += 2
        elif a == "--one-shot":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --one-shot", "", "")
            one_shot_prompt = String(args[i + 1])
            i += 2
        elif a == "--help" or a == "-h":
            print(
                "DISMOEN Chat — Interactive Terminal REPL (Qwen3.6 MoE, REAL"
                " inference)"
            )
            print("Penggunaan: dismoen chat [flags]")
            print("Flags tersedia:")
            print(
                "  --model-dir <dir>       Direktori model (wajib ada"
                " tokenizer.json + config.json; tanpa fallback)"
            )
            print(
                "  --quant-model <file>    Berkas kuantisasi GGUF (wajib;"
                " tanpa fallback)"
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
                "  --temperature <f>       Harus 0.0 (greedy; sampling lain"
                " ditolak fail-closed)"
            )
            print(
                "  --max-tokens <n>        Batas maksimum token per giliran"
                " respons"
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

    # Sampling selain greedy DITOLAK jujur (engine hanya argmax).
    if temperature != Float64(0.0):
        fail_chat("sampling temperature != 0.0 tidak didukung (hanya greedy)")
    if max_tokens <= 0:
        fail("USAGE", "max-tokens harus positif", "", "")

    # 2. Resolusi path direktori model sesuai hierarki preseden
    var model_dir = resolve_chat_model_dir(model_dir_arg)

    # 3. Fail-closed startup: tokenizer + config wajib ada (tanpa pura-pura).
    if get_file_size(String(model_dir, "/tokenizer.json")) <= 0:
        fail_chat(
            String(
                (
                    "no tokenizer.json in model dir (simulasi dilarang per fix"
                    " #3): "
                ),
                model_dir,
            )
        )
    var config_cand = String(model_dir, "/config.json")
    if get_file_size(config_cand) <= 0:
        fail_chat(String("config.json tidak ditemukan di: ", model_dir))

    # 4. Config REAL dari model (bukan hardcoded) + validasi GGUF.
    var cfg = ModelConfig(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=1024,
        num_key_value_heads=1,
        head_dim_override=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        full_attention_interval=4,
        norm_topk_prob=False,
        attention_bias=False,
        architecture="qwen3.6",
    )
    var eps = Float32(1e-6)
    try:
        var parsed = parse_model_config(config_cand)
        cfg = parsed[0].copy()
        eps = parsed[1]
    except e:
        fail_chat(String("config model tidak valid: ", String(e)))

    var quant_path = resolve_quant_model_path(quant_model_arg, model_dir)
    if quant_path.byte_length() == 0 or get_file_size(quant_path) <= 0:
        fail_chat(
            String(
                (
                    "no quantizer model found: berikan --quant-model"
                    " <file.gguf> (mode simulasi dilarang per fix #3; got"
                    " --quant-model='"
                ),
                quant_model_arg,
                "')",
            )
        )
    try:
        var q_index = parse_gguf_index(quant_path)
        validate_gguf_port_coverage(q_index, cfg)
    except e:
        fail_chat(String("model kuantisasi tidak valid: ", String(e)))

    # 5. Special tokens REAL (gagal -> fail, bukan resolver dummy).
    var resolver = resolve_special_tokens(model_dir)
    var lock_path = "models.lock.json"
    if get_file_size(lock_path) > 0:
        try:
            verify_tokenizer_lockfile(model_dir, lock_path)
        except e:
            print("Notice: verifikasi lockfile tokenizer: " + String(e))

    # 6. Thread pool + KV/GDN states sesi.
    if auto_threads and not threads_explicit:
        var c_star = read_hardware_lock_c_star()
        if c_star > 0:
            threads = c_star
    var pool = WorkerPool(threads)
    var kv_cap = 512
    var kv_cache = GatedAttnKVCache(
        kv_cap,
        cfg.num_attention_layers(),
        cfg.num_key_value_heads,
        cfg.head_dim(),
    )
    var gdn_states = GDNState(cfg.num_gdn_layers(), 32, 32)
    var all_tokens = List[Int]()

    # 7. Riwayat pesan ChatML.
    var messages = List[ChatMessage]()
    messages.append(ChatMessage("system", system_prompt))

    # Banner Header REPL
    if one_shot_prompt.byte_length() == 0:
        print(ANSI_BOLD_CYAN)
        print(
            "======================================================================"
        )
        print(
            "DISMOEN Chat REPL — Qwen MoE Disk-Streaming Engine (REAL"
            " inference)"
        )
        print(
            "======================================================================"
        )
        print(ANSI_RESET)
        print("Model dir : " + model_dir)
        print("Quant     : " + quant_path)
        print("Threads   : " + String(threads))
        print(
            "Engine    : BPE real + GGUF streaming + argmax greedy "
            "(tanpa simulasi)"
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

    # Self-pipe SIGINT (M12-7)
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

    # 8. Loop REPL Interaktif Utama (REAL per turn)
    while is_running:
        var user_input: String

        if one_shot_prompt.byte_length() > 0:
            user_input = one_shot_prompt
            is_running = False
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

        # Perintah internal slash
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
                kv_cache = GatedAttnKVCache(
                    kv_cap,
                    cfg.num_attention_layers(),
                    cfg.num_key_value_heads,
                    cfg.head_dim(),
                )
                gdn_states = GDNState(cfg.num_gdn_layers(), 32, 32)
                all_tokens.clear()
                print(
                    ANSI_BOLD_YELLOW
                    + "[Konteks percakapan dan state sesi telah dibersihkan.]"
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
                print("  /clear   - Reset konteks percakapan dan state sesi")
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

        # Giliran Generasi REAL
        messages.append(ChatMessage("user", user_input))
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

        # Snapshot state untuk rollback bila abort (M12-7).
        var snap_kv = kv_cache.copy()
        var snap_gdn = gdn_states.copy()
        var snap_hist = len(all_tokens)
        var snap_msgs = len(messages)

        var turn_ok = False
        try:
            var turn_res = run_chat_turn(
                prompt_rendered,
                model_dir,
                resolver,
                all_tokens,
                kv_cache,
                gdn_states,
                quant_path,
                cfg,
                eps,
                max_tokens,
                sig_r_fd,
                pool,
            )
            var finish = turn_res[2]
            if finish == "abort":
                kv_cache = snap_kv.copy()
                gdn_states = snap_gdn.copy()
                while len(all_tokens) > snap_hist:
                    _ = all_tokens.pop()
                while len(messages) > snap_msgs:
                    _ = messages.pop()
                print(
                    ANSI_BOLD_RED
                    + "[Generasi dibatalkan oleh pengguna (Ctrl+C). Konteks"
                    " parsial dibuang; state sesi dikembalikan.]"
                    + ANSI_RESET
                    + "\n"
                )
            else:
                messages.append(ChatMessage("assistant", turn_res[1]))
                turn_ok = True
                turn_count += 1
                # Telemetri turn jujur ke stderr (stdout murni teks chat):
                # bukti turn inferensi REAL berjalan (bukan simulasi).
                var n_prompt = len(all_tokens) - snap_hist - len(turn_res[0])
                eprint_json(
                    String(
                        '{"turn":',
                        turn_count,
                        ',"prompt_tokens":',
                        n_prompt,
                        ',"generated_tokens":',
                        len(turn_res[0]),
                        ',"finish_reason":"',
                        turn_res[2],
                        '"}',
                    )
                )
        except e:
            var err_s = String(e)
            kv_cache = snap_kv.copy()
            gdn_states = snap_gdn.copy()
            while len(all_tokens) > snap_hist:
                _ = all_tokens.pop()
            while len(messages) > snap_msgs:
                _ = messages.pop()
            if one_shot_prompt.byte_length() > 0:
                pool.shutdown()
                _ = external_call["signal", Int](Int32(2), sig_default_handler)
                _ = external_call["close", Int32](sig_r_fd)
                _ = external_call["close", Int32](Int32(88))
                external_call["free", NoneType](pipe_addr)
                fail_chat(String("turn gagal: ", err_s))
            print(ANSI_BOLD_RED + "Error turn: " + err_s + ANSI_RESET + "\n")
        _ = turn_ok

    pool.shutdown()
    _ = external_call["signal", Int](Int32(2), sig_default_handler)
    _ = external_call["close", Int32](sig_r_fd)
    _ = external_call["close", Int32](Int32(88))
    external_call["free", NoneType](pipe_addr)
