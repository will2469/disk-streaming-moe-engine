# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""ChatML Template Formatter & Rejection Protocol (M12-W1b).

Mengimplementasikan subset text-only dari template Qwen3.6 sesuai §2.1 & Gate G-M12-1:
- 5 cabang rendering normatif:
  1. System message pertama pada messages[0] (<|im_start|>system\\n{trimmed}<|im_end|>\\n)
  2. User message literal (<|im_start|>user\\n{trimmed}<|im_end|>\\n), tag <think> literal tanpa semantik
  3. Ekstraksi think assistant upstream:
     - Assistant sebelum/sama dengan query user terakhir: render polos (blok think dibuang)
     - Assistant setelah query user terakhir (trailing): render think-wrapped (<think>...<think>)
  4. Pemindai query user terakhir (menolak input tanpa query user asli)
  5. Sufiks prompt generasi (<|im_start|>assistant\\n<think>\\n)
- Penolakan terstruktur input di luar subset (400/422 + unsupported: ...).
- STRICT PROHIBITION hardcoded token IDs (M12-1).
"""

from std.collections import List


struct ChatMessage(Copyable, Movable):
    """Pesan dalam percakapan ChatML."""

    var role: String
    var content: String
    var reasoning_content: String
    var has_reasoning_content: Bool
    var has_tool_calls: Bool

    def __init__(out self, role: String, content: String):
        """Inisialisasi pesan teks biasa dalam subset M12."""
        self.role = role
        self.content = content
        self.reasoning_content = ""
        self.has_reasoning_content = False
        self.has_tool_calls = False

    def __init__(
        out self,
        role: String,
        content: String,
        reasoning_content: String,
        has_reasoning_content: Bool = True,
        has_tool_calls: Bool = False,
    ):
        """Inisialisasi pesan dengan field tambahan untuk pengujian penolakan non-subset.
        """
        self.role = role
        self.content = content
        self.reasoning_content = reasoning_content
        self.has_reasoning_content = has_reasoning_content
        self.has_tool_calls = has_tool_calls


@fieldwise_init
struct AssistantThink(Copyable, Movable):
    """Hasil ekstraksi blok reasoning dan konten assistant."""

    var reasoning: String
    var content: String


def find_last_substring(s: String, sub: String) -> Int:
    """Mencari posisi byte kemunculan terakhir substring sub di dalam s."""
    var sub_len = sub.byte_length()
    var s_len = s.byte_length()
    if sub_len == 0 or s_len < sub_len:
        return -1
    var last_pos = -1
    var idx = 0
    while idx + sub_len <= s_len:
        var slice_str = s[byte=idx:s_len]
        var rel_pos = slice_str.find(sub)
        if rel_pos < 0:
            break
        last_pos = idx + rel_pos
        idx = last_pos + sub_len
    return last_pos


def extract_assistant_think(raw: String) -> AssistantThink:
    """Mengekstrak blok reasoning dan konten bersih assistant persis logika template upstream.

    Logika upstream:
        if '</think>' in content:
            reasoning = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')
            content = content.split('</think>')[-1].lstrip('\\n')
    """
    var think_close_pos = raw.find("</think>")
    if think_close_pos < 0:
        return AssistantThink(String(""), String(raw.strip()))

    var raw_bytes = raw.as_bytes()

    # 1. Bagian sebelum </think> pertama: raw[:think_close_pos].rstrip('\n')
    var end_bc = think_close_pos
    while end_bc > 0 and raw_bytes[end_bc - 1] == 10:  # '\n'
        end_bc -= 1
    var before_close = String(raw[byte=0:end_bc])

    # Cari <think> terakhir di before_close: before_close.split('<think>')[-1].lstrip('\n')
    var bc_bytes = before_close.as_bytes()
    var think_open_pos = find_last_substring(before_close, "<think>")
    var start_r: Int
    if think_open_pos >= 0:
        start_r = think_open_pos + 7
    else:
        start_r = 0

    while start_r < len(bc_bytes) and bc_bytes[start_r] == 10:  # '\n'
        start_r += 1

    var reasoning = String(
        before_close[byte = start_r : before_close.byte_length()]
    )

    # 2. Bagian setelah </think> terakhir: raw.split('</think>')[-1].lstrip('\n')
    var last_think_close_pos = find_last_substring(raw, "</think>")
    var start_ac: Int
    if last_think_close_pos >= 0:
        start_ac = last_think_close_pos + 8
    else:
        start_ac = 0

    while start_ac < len(raw_bytes) and raw_bytes[start_ac] == 10:  # '\n'
        start_ac += 1

    var after_close = String(raw[byte = start_ac : raw.byte_length()])

    return AssistantThink(
        String(reasoning.strip()), String(after_close.strip())
    )


def extract_last_query_index(messages: List[ChatMessage]) -> Int:
    """Mencari indeks pesan query user terakhir (yang bukan sekadar tool_response).
    """
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == "user":
            var trimmed = messages[idx].content.strip()
            if not (
                trimmed.startswith("<tool_response>")
                and trimmed.endswith("</tool_response>")
            ):
                return idx
    return -1


def render_chatml(
    messages: List[ChatMessage],
    add_generation_prompt: Bool = True,
) raises -> String:
    """Me-render pesan percakapan ke string prompt ChatML byte-exact (subset text-only).

    Menolak setiap input di luar subset dengan format error terstruktur (400/422 + unsupported: ...).
    """
    # 1. Validasi keberadaan pesan
    if len(messages) == 0:
        raise Error("400 Bad Request: unsupported: empty_messages")

    # 2. Validasi field dan role yang didukung
    for i in range(len(messages)):
        ref m = messages[i]
        if m.has_reasoning_content:
            raise Error(
                "422 Unprocessable Entity: unsupported: field=reasoning_content"
            )
        if m.has_tool_calls:
            raise Error(
                "422 Unprocessable Entity: unsupported: field=tool_calls"
            )
        if m.role != "system" and m.role != "user" and m.role != "assistant":
            raise Error("400 Bad Request: unsupported: role=" + m.role)

    # 3. Validasi posisi dan keunikan system message
    var system_count = 0
    for i in range(len(messages)):
        if messages[i].role == "system":
            system_count += 1
            if i != 0:
                if system_count > 1:
                    raise Error(
                        "400 Bad Request: unsupported: duplicate_system"
                    )
                raise Error("400 Bad Request: unsupported: system_not_first")

    # 4. Pemindai query user terakhir
    var last_query_idx = extract_last_query_index(messages)
    if last_query_idx == -1:
        raise Error("400 Bad Request: unsupported: no_user_query")

    # 5. Rendering subset normatif
    var out = String("")

    # System message pertama (jika ada di index 0)
    if len(messages) > 0 and messages[0].role == "system":
        var sys_trimmed = messages[0].content.strip()
        out += "<|im_start|>system\n" + sys_trimmed + "<|im_end|>\n"

    # Perulangan pesan
    for i in range(len(messages)):
        ref m = messages[i]
        if m.role == "system":
            # Sudah di-render di atas
            pass
        elif m.role == "user":
            var u_trimmed = m.content.strip()
            out += "<|im_start|>user\n" + u_trimmed + "<|im_end|>\n"
        elif m.role == "assistant":
            var think_pair = extract_assistant_think(m.content)
            var reasoning = think_pair.reasoning
            var content = think_pair.content

            if i > last_query_idx:
                # Trailing assistant (setelah last query user) -> wrap think
                out += (
                    "<|im_start|>assistant\n<think>\n"
                    + reasoning
                    + "\n</think>\n\n"
                    + content
                    + "<|im_end|>\n"
                )
            else:
                # Assistant sebelum/setara query user terakhir -> buang think
                out += "<|im_start|>assistant\n" + content + "<|im_end|>\n"

    # 6. Sufiks generasi
    if add_generation_prompt:
        out += "<|im_start|>assistant\n<think>\n"

    return out
