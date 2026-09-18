# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Streaming UTF-8 Detokenizer (M12-W1b).

Menangani streaming decoding token/byte secara inkremental tanpa korupsi karakter:
- Menyangga potongan byte multi-byte UTF-8 (2-byte, 3-byte CJK, 4-byte emoji)
  yang terbelah di antara batas token / chunk.
- Hanya memancarkan urutan karakter UTF-8 yang utuh dan valid.
- Zero encoding corruption pada pengiriman streaming interaktif / SSE.
"""

from std.collections import List


@fieldwise_init
struct StreamingDetokenizer(Movable):
    """Detokenizer streaming berpenyangga untuk penanganan batas multi-byte UTF-8.
    """

    var buffer: List[UInt8]

    def __init__(out self):
        """Inisialisasi penyangga detokenizer kosong."""
        self.buffer = List[UInt8]()

    def feed_bytes(mut self, data: List[UInt8]) -> String:
        """Menambahkan byte baru ke penyangga dan mengembalikan string ber-karakter lengkap.
        """
        for i in range(len(data)):
            self.buffer.append(data[i])
        return self._consume_valid_utf8()

    def feed_string(mut self, text: String) -> String:
        """Menambahkan string mentah ke penyangga dan mengembalikan teks yang aman dipancarkan.
        """
        var b = text.as_bytes()
        var l = List[UInt8]()
        l.reserve(len(b))
        for i in range(len(b)):
            l.append(b[i])
        return self.feed_bytes(l)

    def _consume_valid_utf8(mut self) -> String:
        var buf_len = len(self.buffer)
        if buf_len == 0:
            return ""

        var i = 0
        var valid_end = 0

        while i < buf_len:
            var b = self.buffer[i]
            if b < 0x80:
                # 1-byte ASCII
                i += 1
                valid_end = i
            elif (b & 0xE0) == 0xC0:
                # 2-byte UTF-8
                if i + 1 >= buf_len:
                    # Belum lengkap di akhir penyangga
                    break
                if (self.buffer[i + 1] & 0xC0) != 0x80:
                    # Bukan continuation byte sah
                    i += 1
                    valid_end = i
                else:
                    i += 2
                    valid_end = i
            elif (b & 0xF0) == 0xE0:
                # 3-byte UTF-8
                if i + 2 >= buf_len:
                    # Belum lengkap di akhir penyangga
                    break
                if (self.buffer[i + 1] & 0xC0) != 0x80 or (
                    self.buffer[i + 2] & 0xC0
                ) != 0x80:
                    i += 1
                    valid_end = i
                else:
                    i += 3
                    valid_end = i
            elif (b & 0xF8) == 0xF0:
                # 4-byte UTF-8
                if i + 3 >= buf_len:
                    # Belum lengkap di akhir penyangga
                    break
                if (
                    (self.buffer[i + 1] & 0xC0) != 0x80
                    or (self.buffer[i + 2] & 0xC0) != 0x80
                    or (self.buffer[i + 3] & 0xC0) != 0x80
                ):
                    i += 1
                    valid_end = i
                else:
                    i += 4
                    valid_end = i
            else:
                # Byte lead tidak valid (0x80..0xBF atau >= 0xF8)
                i += 1
                valid_end = i

        if valid_end == 0:
            return ""

        # Ekstrak potongan byte yang lengkap
        var out_bytes = List[UInt8]()
        out_bytes.reserve(valid_end)
        for j in range(valid_end):
            out_bytes.append(self.buffer[j])

        # Simpan sisa byte yang belum lengkap di penyangga
        var remaining = List[UInt8]()
        remaining.reserve(buf_len - valid_end)
        for k in range(valid_end, buf_len):
            remaining.append(self.buffer[k])
        self.buffer = remaining^

        return String(from_utf8_lossy=Span(out_bytes))

    def flush(mut self) -> String:
        """Mengeluarkan sisa byte yang tersisa saat aliran token selesai."""
        if len(self.buffer) == 0:
            return ""
        var res = String(from_utf8_lossy=Span(self.buffer))
        self.buffer.clear()
        return res

    def reset(mut self):
        """Mengosongkan isi penyangga detokenizer."""
        self.buffer.clear()
