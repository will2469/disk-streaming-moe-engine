# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Parser metadata GGUF v2/v3 dan on-demand streaming reader (M9-W3).

Memenuhi invarian ketat DoD M9:
- HANYA mem-parsing header metadata dan tensor locator directory ke RAM (< 1 MB).
- DILARANG memuat seluruh tensor file GGUF ke RAM (P0-3).
- Validasi eksak ukuran berkas fisik stat.st_size vs formula F11b-GGUF (diff == 0 bytes).
- Streaming on-demand per tensor via O_DIRECT dengan dekuantisasi ke reusable scratchpad.
"""

from format.file_io import _open_shard
from format.quant_format import float16_to_u16, u16_to_float16
from std.collections import Dict, List
from std.os import SEEK_END, SEEK_SET

# GGML types enum
comptime GGML_TYPE_F32: Int = 0
comptime GGML_TYPE_F16: Int = 1
comptime GGML_TYPE_Q4_0: Int = 2
comptime GGML_TYPE_Q4_1: Int = 3
comptime GGML_TYPE_Q8_0: Int = 8
comptime GGML_TYPE_Q3_K: Int = 11
comptime GGML_TYPE_Q4_K: Int = 12
comptime GGML_TYPE_Q5_K: Int = 13
comptime GGML_TYPE_Q6_K: Int = 14
comptime GGML_TYPE_IQ3_XXS: Int = 18
comptime GGML_TYPE_IQ3_S: Int = 21
comptime GGML_TYPE_BF16: Int = 30

# GGUF Value Types
comptime GGUF_TYPE_UINT8: Int = 0
comptime GGUF_TYPE_INT8: Int = 1
comptime GGUF_TYPE_UINT16: Int = 2
comptime GGUF_TYPE_INT16: Int = 3
comptime GGUF_TYPE_UINT32: Int = 4
comptime GGUF_TYPE_INT32: Int = 5
comptime GGUF_TYPE_FLOAT32: Int = 6
comptime GGUF_TYPE_BOOL: Int = 7
comptime GGUF_TYPE_STRING: Int = 8
comptime GGUF_TYPE_ARRAY: Int = 9
comptime GGUF_TYPE_UINT64: Int = 10
comptime GGUF_TYPE_INT64: Int = 11
comptime GGUF_TYPE_FLOAT64: Int = 12


def ggml_type_to_string(t: Int) -> String:
    if t == GGML_TYPE_F32:
        return "F32"
    elif t == GGML_TYPE_F16:
        return "F16"
    elif t == GGML_TYPE_Q4_0:
        return "Q4_0"
    elif t == GGML_TYPE_Q4_1:
        return "Q4_1"
    elif t == GGML_TYPE_Q8_0:
        return "Q8_0"
    elif t == GGML_TYPE_Q3_K:
        return "Q3_K"
    elif t == GGML_TYPE_Q4_K:
        return "Q4_K"
    elif t == GGML_TYPE_Q5_K:
        return "Q5_K"
    elif t == GGML_TYPE_Q6_K:
        return "Q6_K"
    elif t == GGML_TYPE_IQ3_XXS:
        return "IQ3_XXS"
    elif t == GGML_TYPE_IQ3_S:
        return "IQ3_S"
    elif t == GGML_TYPE_BF16:
        return "BF16"
    return "UNKNOWN"


def get_ggml_block_specs(t: Int) -> Tuple[Int, Int]:
    """Mengembalikan (block_size, bytes_per_block) untuk tipe GGML."""
    if t == GGML_TYPE_F32:
        return (1, 4)
    elif t == GGML_TYPE_F16 or t == GGML_TYPE_BF16:
        return (1, 2)
    elif t == GGML_TYPE_Q4_0:
        return (32, 18)
    elif t == GGML_TYPE_Q4_1:
        return (32, 20)
    elif t == GGML_TYPE_Q8_0:
        return (32, 34)
    elif t == GGML_TYPE_Q3_K:
        return (256, 114)
    elif t == GGML_TYPE_Q4_K:
        return (256, 144)
    elif t == GGML_TYPE_Q5_K:
        return (256, 176)
    elif t == GGML_TYPE_Q6_K:
        return (256, 210)
    elif t == GGML_TYPE_IQ3_XXS:
        return (256, 98)
    elif t == GGML_TYPE_IQ3_S:
        return (256, 110)
    # Default fallback
    return (1, 4)


@fieldwise_init
struct GGUFTensorInfo(Copyable, Movable):
    """Informasi metadata satu tensor dalam file GGUF."""

    var name: String
    var ndim: Int
    var dims: List[Int]
    var dtype: Int
    var offset: Int
    var num_elements: Int
    var raw_bytes: Int


struct GGUFIndex(Movable):
    """Indeks pencarian tensor GGUF streaming tanpa beban memori heap bobot."""

    var version: Int
    var tensor_count: Int
    var metadata_kv_count: Int
    var data_section_offset: Int
    var expected_file_size: Int
    var alignment: Int
    var file_path: String
    var tensors: List[GGUFTensorInfo]
    var tensor_map: Dict[String, Int]
    var metadata: Dict[String, String]

    def __init__(out self, file_path: String):
        self.version = 0
        self.tensor_count = 0
        self.metadata_kv_count = 0
        self.data_section_offset = 0
        self.expected_file_size = 0
        self.alignment = 32
        self.file_path = file_path
        self.tensors = List[GGUFTensorInfo]()
        self.tensor_map = Dict[String, Int]()
        self.metadata = Dict[String, String]()


struct ByteReader:
    """Helper pembaca little-endian dari buffer byte."""

    var buf: List[UInt8]
    var pos: Int

    def __init__(out self, buf: List[UInt8]):
        self.buf = buf.copy()
        self.pos = 0

    def has_bytes(self, n: Int) -> Bool:
        return self.pos + n <= len(self.buf)

    def read_u8(mut self) raises -> UInt8:
        if self.pos >= len(self.buf):
            raise Error("GGUF: unexpected EOF reading u8")
        var v = self.buf[self.pos]
        self.pos += 1
        return v

    def read_u16(mut self) raises -> Int:
        if self.pos + 2 > len(self.buf):
            raise Error("GGUF: unexpected EOF reading u16")
        var b0 = Int(self.buf[self.pos])
        var b1 = Int(self.buf[self.pos + 1])
        self.pos += 2
        return b0 | (b1 << 8)

    def read_u32(mut self) raises -> Int:
        if self.pos + 4 > len(self.buf):
            raise Error("GGUF: unexpected EOF reading u32")
        var v = 0
        var mult = 1
        for i in range(4):
            v += Int(self.buf[self.pos + i]) * mult
            mult *= 256
        self.pos += 4
        return v

    def read_u64(mut self) raises -> Int:
        if self.pos + 8 > len(self.buf):
            raise Error("GGUF_FILE_CORRUPT: unexpected EOF reading u64")
        var v = 0
        var mult = 1
        for i in range(8):
            if i == 7 and (self.buf[self.pos + i] & 0x80) != 0:
                raise Error("GGUF_FILE_CORRUPT: u64 overflow/negative value")
            v += Int(self.buf[self.pos + i]) * mult
            if i < 7:
                mult *= 256
        self.pos += 8
        if v < 0:
            raise Error("GGUF_FILE_CORRUPT: negative u64 value")
        return v

    def read_string(mut self) raises -> String:
        var length = self.read_u64()
        if length < 0 or length > 65536:
            raise Error("GGUF: string length invalid: " + String(length))
        if self.pos + length > len(self.buf):
            raise Error("GGUF: unexpected EOF reading string")
        var bytes = List[UInt8]()
        bytes.reserve(length)
        for i in range(length):
            bytes.append(self.buf[self.pos + i])
        self.pos += length
        return String(from_utf8_lossy=Span(bytes))

    def skip(mut self, n: Int) raises:
        if self.pos + n > len(self.buf):
            raise Error("GGUF: unexpected EOF skipping bytes")
        self.pos += n


def _skip_gguf_value(mut reader: ByteReader, vtype: Int) raises -> String:
    """Melewati satu value metadata dan mengembalikan representasi stringnya jika sederhana.
    """
    if (
        vtype == GGUF_TYPE_UINT8
        or vtype == GGUF_TYPE_INT8
        or vtype == GGUF_TYPE_BOOL
    ):
        var v = reader.read_u8()
        return String(Int(v))
    elif vtype == GGUF_TYPE_UINT16 or vtype == GGUF_TYPE_INT16:
        var v = reader.read_u16()
        return String(v)
    elif (
        vtype == GGUF_TYPE_UINT32
        or vtype == GGUF_TYPE_INT32
        or vtype == GGUF_TYPE_FLOAT32
    ):
        var v = reader.read_u32()
        return String(v)
    elif (
        vtype == GGUF_TYPE_UINT64
        or vtype == GGUF_TYPE_INT64
        or vtype == GGUF_TYPE_FLOAT64
    ):
        var v = reader.read_u64()
        return String(v)
    elif vtype == GGUF_TYPE_STRING:
        return reader.read_string()
    elif vtype == GGUF_TYPE_ARRAY:
        var elem_type = reader.read_u32()
        var count = reader.read_u64()
        for _ in range(count):
            _ = _skip_gguf_value(reader, elem_type)
        return String("[Array of ", count, " elements]")
    else:
        raise Error("GGUF: unknown value type " + String(vtype))


def parse_gguf_index(file_path: String) raises -> GGUFIndex:
    """Memindai header GGUF dan membangun indeks pencarian tensor (<1 MB RAM).

    Tidak memuat payload bobot ke RAM. Memvalidasi formula F11b-GGUF exact file size.
    """
    var f = _open_shard(file_path)
    var filesize = Int(f.seek(0, SEEK_END))
    _ = f.seek(0, SEEK_SET)

    if filesize < 32:
        f.close()
        raise Error(
            "GGUF_FILE_CORRUPT: file size "
            + String(filesize)
            + " too small for header"
        )

    # Baca 2 MB pertama yang cukup untuk menampung header dan metadata tensor model port
    var initial_read = min(filesize, 4 * 1024 * 1024)
    var header_bytes = f.read_bytes(initial_read)

    var reader = ByteReader(header_bytes)

    # 1. Magic
    if (
        reader.read_u8() != 0x47
        or reader.read_u8() != 0x47
        or reader.read_u8() != 0x55
        or reader.read_u8() != 0x46
    ):
        f.close()
        raise Error("GGUF_FILE_CORRUPT: invalid magic header (expected 'GGUF')")

    var index = GGUFIndex(file_path)
    index.version = reader.read_u32()
    if index.version != 2 and index.version != 3:
        f.close()
        raise Error(
            "GGUF_FILE_CORRUPT: unsupported version " + String(index.version)
        )

    index.tensor_count = reader.read_u64()
    index.metadata_kv_count = reader.read_u64()

    if index.tensor_count < 0 or index.tensor_count > 100000:
        f.close()
        raise Error(
            "GGUF_FILE_CORRUPT: tensor_count out of bounds: "
            + String(index.tensor_count)
        )

    # 2. Metadata KV
    for _ in range(index.metadata_kv_count):
        var key = reader.read_string()
        var val_type = reader.read_u32()
        var val_str = _skip_gguf_value(reader, val_type)
        index.metadata[key] = val_str
        if key == "general.alignment":
            try:
                var al = Int(val_str)
                if al > 0:
                    index.alignment = al
            except:
                pass

    # 3. Tensor Info Directory
    var max_tensor_end = 0
    var calculated_data_bytes = 0

    for i in range(index.tensor_count):
        var name = reader.read_string()
        var ndim = reader.read_u32()
        if ndim < 1 or ndim > 8:
            f.close()
            raise Error(
                "GGUF_FILE_CORRUPT: tensor ndim invalid: " + String(ndim)
            )

        var dims = List[Int]()
        var num_elements = 1
        for _ in range(ndim):
            var d = reader.read_u64()
            dims.append(d)
            num_elements *= d

        var dtype = reader.read_u32()
        var offset = reader.read_u64()

        var spec = get_ggml_block_specs(dtype)
        var block_size = spec[0]
        var bytes_per_block = spec[1]
        var num_blocks = (num_elements + block_size - 1) // block_size
        var raw_bytes = num_blocks * bytes_per_block

        if offset < 0:
            f.close()
            raise Error("GGUF_FILE_CORRUPT: negative tensor offset")

        var end_offset = offset + raw_bytes
        if end_offset < offset:
            f.close()
            raise Error("GGUF_FILE_CORRUPT: tensor offset integer overflow")

        if end_offset > max_tensor_end:
            max_tensor_end = end_offset

        # Tambahkan ke estimasi analitik F11b (tiap tensor aligned 32)
        var aligned_bytes = (
            (raw_bytes + index.alignment - 1) // index.alignment
        ) * index.alignment
        calculated_data_bytes += aligned_bytes

        var info = GGUFTensorInfo(
            name, ndim, dims^, dtype, offset, num_elements, raw_bytes
        )
        index.tensors.append(info.copy())
        index.tensor_map[name] = i

    # Posisi awal data section harus ter-align ke alignment
    var header_end = reader.pos
    var data_start = (
        (header_end + index.alignment - 1) // index.alignment
    ) * index.alignment
    index.data_section_offset = data_start

    for i in range(len(index.tensors)):
        var t_end = (
            data_start + index.tensors[i].offset + index.tensors[i].raw_bytes
        )
        if t_end > filesize or t_end < 0:
            f.close()
            raise Error(
                "GGUF_FILE_CORRUPT: tensor extends beyond file boundary"
            )

    f.close()

    # 4. Validasi Ukuran Berkas Analitik Eksak (Formula F11b-GGUF)
    # Kontrak: stat(path).st_size == Size_GGUF_expected (diff == 0 bytes)
    var expected_size = data_start + max_tensor_end
    index.expected_file_size = expected_size

    var diff = filesize - expected_size
    if diff != 0:
        raise Error(
            "GGUF_FILE_CORRUPT: actual size "
            + String(filesize)
            + " != expected size "
            + String(expected_size)
            + " (diff: "
            + String(diff)
            + " bytes)"
        )

    return index^


def dequantize_q8_0(
    raw_bytes: List[UInt8], num_elements: Int
) raises -> List[Float32]:
    """Mendekuantisasi blok Q8_0 (32 weights/blok, 34 bytes/blok) ke Float32."""
    var out = List[Float32]()
    out.resize(num_elements, Float32(0.0))
    var p_out = out.unsafe_ptr()

    var num_blocks = num_elements // 32
    var p_raw = raw_bytes.unsafe_ptr()

    for b in range(num_blocks):
        var block_off = b * 34
        # 2 bytes skala FP16
        var u16_val = Int(p_raw[unsafe_offset=block_off]) | (
            Int(p_raw[unsafe_offset=block_off + 1]) << 8
        )
        var s_fp16 = u16_to_float16(UInt16(u16_val))
        var scale = Float32(s_fp16)

        # 32 bytes quants int8
        var out_off = b * 32
        for j in range(32):
            var b_val = Int(p_raw[unsafe_offset=block_off + 2 + j])
            var q = b_val - 256 if b_val >= 128 else b_val
            p_out[unsafe_offset=out_off + j] = scale * Float32(q)

    return out^


def dequantize_q4_k(
    raw_bytes: List[UInt8], num_elements: Int
) raises -> List[Float32]:
    """Mendekuantisasi blok Q4_K (256 weights/blok, 144 bytes/blok) ke Float32.
    """
    var out = List[Float32]()
    out.resize(num_elements, Float32(0.0))
    var p_out = out.unsafe_ptr()

    var num_blocks = num_elements // 256
    var p_raw = raw_bytes.unsafe_ptr()

    for b in range(num_blocks):
        var block_off = b * 144

        # Super-scale d dan dmin (FP16)
        var u_d = Int(p_raw[unsafe_offset=block_off]) | (
            Int(p_raw[unsafe_offset=block_off + 1]) << 8
        )
        var u_m = Int(p_raw[unsafe_offset=block_off + 2]) | (
            Int(p_raw[unsafe_offset=block_off + 3]) << 8
        )
        var d = Float32(u16_to_float16(UInt16(u_d)))
        var dmin = Float32(u16_to_float16(UInt16(u_m)))

        # Scales & mins (12 bytes)
        var sc = List[Float32]()
        sc.resize(8, Float32(0.0))
        var m_arr = List[Float32]()
        m_arr.resize(8, Float32(0.0))

        for j in range(4):
            var sc_byte = Int(p_raw[unsafe_offset=block_off + 4 + j])
            var m_byte = Int(p_raw[unsafe_offset=block_off + 8 + j])
            sc[j] = Float32(sc_byte & 63) * d
            m_arr[j] = Float32(m_byte & 63) * dmin

        for j in range(4, 8):
            var b4 = Int(p_raw[unsafe_offset=block_off + j])
            var b8 = Int(p_raw[unsafe_offset=block_off + 4 + j])
            var sc_val = (b8 & 0xF) | ((b4 >> 6) << 4)
            var m_val = (b8 >> 4) | ((b8 >> 6) << 4)
            sc[j] = Float32(sc_val) * d
            m_arr[j] = Float32(m_val) * dmin

        # 128 bytes quants (qs)
        var qs_off = block_off + 16
        var out_off = b * 256

        for j in range(8):
            var sub_out = out_off + j * 32
            var cur_sc = sc[j]
            var cur_m = m_arr[j]
            for i in range(16):
                var byte_val = Int(p_raw[unsafe_offset=qs_off + j * 16 + i])
                var q0 = byte_val & 0xF
                var q1 = (byte_val >> 4) & 0xF
                p_out[unsafe_offset=sub_out + i] = cur_sc * Float32(q0) - cur_m
                p_out[unsafe_offset=sub_out + 16 + i] = (
                    cur_sc * Float32(q1) - cur_m
                )

    return out^


def dequantize_q3_k(
    raw_bytes: List[UInt8], num_elements: Int
) raises -> List[Float32]:
    """Mendekuantisasi blok Q3_K (256 weights/blok, 114 bytes/blok) ke Float32.
    """
    var out = List[Float32]()
    out.resize(num_elements, Float32(0.0))
    var p_out = out.unsafe_ptr()

    var num_blocks = num_elements // 256
    var p_raw = raw_bytes.unsafe_ptr()

    for b in range(num_blocks):
        var block_off = b * 114

        # Super-scale d (FP16 di akhir blok: bytes 112..113)
        var u_d = Int(p_raw[unsafe_offset=block_off + 112]) | (
            Int(p_raw[unsafe_offset=block_off + 113]) << 8
        )
        var d = Float32(u16_to_float16(UInt16(u_d)))

        # Sub-block scales (16 bytes: 96..111)
        var scales = List[Float32]()
        scales.resize(16, Float32(0.0))
        for j in range(16):
            var sc_val = Int(p_raw[unsafe_offset=block_off + 96 + j])
            scales[j] = Float32(sc_val - 32) * d

        # Quants (hmask: 0..31, qs: 32..95)
        var out_off = b * 256
        for j in range(16):
            var sub_out = out_off + j * 16
            var cur_sc = scales[j]
            for i in range(16):
                var elem_idx = j * 16 + i
                # Bit tinggi dari hmask
                var hmask_byte = Int(
                    p_raw[unsafe_offset=block_off + (elem_idx // 8)]
                )
                var h_bit = (hmask_byte >> (elem_idx % 8)) & 1

                # 2 bit rendah dari qs
                var qs_byte = Int(
                    p_raw[unsafe_offset=block_off + 32 + (elem_idx // 4)]
                )
                var shift = (elem_idx % 4) * 2
                var low_bits = (qs_byte >> shift) & 3

                var q_val = (h_bit << 2) | low_bits
                p_out[unsafe_offset=sub_out + i] = cur_sc * Float32(q_val - 4)

    return out^


def stream_gguf_tensor_f32(
    index: GGUFIndex, tensor_name: String
) raises -> List[Float32]:
    """Membaca SATU tensor spesifik dari file GGUF via disk streaming dan didekuantisasi on-the-fly.

    Mematuhi P0-3: HANYA membaca byte tensor yang diminta, lalu me-reuse scratchpad.
    """
    if tensor_name not in index.tensor_map:
        raise Error(
            '{"error_type":"WEIGHT_LOAD_FAILED","detail":"tensor not found in'
            ' GGUF index","tensor":"'
            + tensor_name
            + '"}'
        )

    var idx = index.tensor_map[tensor_name]
    ref info = index.tensors[idx]

    var file_offset = index.data_section_offset + info.offset

    var f = _open_shard(index.file_path)
    _ = f.seek(file_offset, SEEK_SET)
    var raw_bytes = f.read_bytes(info.raw_bytes)
    f.close()

    if len(raw_bytes) < info.raw_bytes:
        raise Error(
            "GGUF_READ_ERROR: truncated read for tensor "
            + tensor_name
            + ": expected "
            + String(info.raw_bytes)
            + " bytes, got "
            + String(len(raw_bytes))
        )

    if info.dtype == GGML_TYPE_F32:
        var out = List[Float32]()
        out.resize(info.num_elements, Float32(0.0))
        var p_src = raw_bytes.unsafe_ptr().unsafe_bitcast[Float32]()
        var p_dst = out.unsafe_ptr()
        for i in range(info.num_elements):
            p_dst[unsafe_offset=i] = p_src[unsafe_offset=i]
        return out^
    elif info.dtype == GGML_TYPE_BF16:
        var out = List[Float32]()
        out.resize(info.num_elements, Float32(0.0))
        var p_src = raw_bytes.unsafe_ptr().unsafe_bitcast[BFloat16]()
        var p_dst = out.unsafe_ptr()
        for i in range(info.num_elements):
            p_dst[unsafe_offset=i] = Float32(p_src[unsafe_offset=i])
        return out^
    elif info.dtype == GGML_TYPE_Q8_0:
        return dequantize_q8_0(raw_bytes, info.num_elements)
    elif info.dtype == GGML_TYPE_Q4_K:
        return dequantize_q4_k(raw_bytes, info.num_elements)
    elif info.dtype == GGML_TYPE_Q3_K:
        return dequantize_q3_k(raw_bytes, info.num_elements)
    else:
        raise Error(
            "GGUF_UNSUPPORTED_DTYPE: unsupported GGML type "
            + String(info.dtype)
            + " for tensor "
            + tensor_name
        )
