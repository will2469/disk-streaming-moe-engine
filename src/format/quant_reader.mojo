# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Reader berkas kuantisasi 4-bit dengan pengerasan parser SEC-4."""

from format.half_float import (
    QUANT_DEFAULT_GROUP_SIZE,
    float16_to_u16,
    is_allowed_group_size,
    safe_multiply_int,
    u16_to_float16,
)
from format.scanner import Scanner
from format.types import json_escape
from quant.dequant_kernel import dequant_kernel_simd, dequant_kernel_simd_f32
from std.collections import Dict, List
from std.os import SEEK_END, SEEK_SET

comptime QUANT_HEADER_SIZE: Int = 256
comptime QUANT_FORMAT_NAME: String = "4-bit per-group"
comptime QUANT_SCALE_DTYPE: String = "FP16"
comptime QUANTIZED_DTYPE_NAME: String = "4-bit"
comptime CONFIGURED_MAX_TENSORS: Int = 100000
comptime CONFIGURED_MAX_NAME: Int = 512
comptime CONFIGURED_MAX_NDIM: Int = 8


@fieldwise_init
struct BlockHeader(Copyable, Movable):
    """Header berkas kuantisasi (tepat 256 byte)."""

    var version: Int
    var model: String
    var format: String
    var group_size: Int
    var scale_dtype: String
    var num_tensors: Int
    var total_bytes: Int

    def __init__(
        out self,
        model: String,
        num_tensors: Int,
        total_bytes: Int,
        group_size: Int = QUANT_DEFAULT_GROUP_SIZE,
        version: Int = 1,
    ):
        self.version = version
        self.model = model
        self.format = QUANT_FORMAT_NAME
        self.group_size = group_size
        self.scale_dtype = QUANT_SCALE_DTYPE
        self.num_tensors = num_tensors
        self.total_bytes = total_bytes

    def to_json(self) -> String:
        """Menghasilkan string JSON kompak untuk header."""
        return String(
            '{"version":',
            String(self.version),
            ',"model":"',
            json_escape(self.model),
            '","quantization":{"format":"',
            json_escape(self.format),
            '","group_size":',
            String(self.group_size),
            ',"scale_dtype":"',
            json_escape(self.scale_dtype),
            '"},"num_tensors":',
            String(self.num_tensors),
            ',"total_bytes":',
            String(self.total_bytes),
            "}",
        )

    def to_header_bytes(self) raises -> List[UInt8]:
        """Mengemas header ke dalam list 256 byte dengan padding spasi."""
        var js = self.to_json()
        var b = js.as_bytes()
        var sz = len(b)
        if sz > QUANT_HEADER_SIZE:
            raise Error(
                "Header JSON exceeds 256 bytes: "
                + String(sz)
                + " > "
                + String(QUANT_HEADER_SIZE)
            )
        var out = List[UInt8]()
        for i in range(sz):
            out.append(b[i])
        while len(out) < QUANT_HEADER_SIZE:
            out.append(32)  # Spasi (0x20)
        return out^

    @staticmethod
    def from_bytes(raw: List[UInt8]) raises -> BlockHeader:
        """Mem-parsing header berkas kuantisasi dari 256 byte data."""
        if len(raw) != QUANT_HEADER_SIZE:
            raise Error(
                "Invalid quant header size: expected 256, got "
                + String(len(raw))
            )
        var copy_raw = List[UInt8]()
        for i in range(len(raw)):
            copy_raw.append(raw[i])
        var sc = Scanner(copy_raw^, "quant_header")

        sc.expect(123)  # '{'
        var version = 1
        var model = String("")
        var format_str = String("")
        var group_size = QUANT_DEFAULT_GROUP_SIZE
        var scale_dtype = QUANT_SCALE_DTYPE
        var num_tensors = 0
        var total_bytes = 0

        while True:
            sc.skip_ws()
            if sc.peek() == 125:  # '}'
                sc.pos += 1
                break
            var key = sc.parse_string()
            sc.expect(58)  # ':'
            if key == "version":
                version = sc.parse_uint()
            elif key == "model":
                model = sc.parse_string()
            elif key == "quantization":
                sc.expect(123)  # '{'
                while True:
                    sc.skip_ws()
                    if sc.peek() == 125:
                        sc.pos += 1
                        break
                    var qk = sc.parse_string()
                    sc.expect(58)
                    if qk == "format":
                        format_str = sc.parse_string()
                    elif qk == "group_size":
                        group_size = sc.parse_uint()
                    elif qk == "scale_dtype":
                        scale_dtype = sc.parse_string()
                    else:
                        sc.skip_value()
                    sc.skip_ws()
                    if sc.peek() == 44:  # ','
                        sc.pos += 1
            elif key == "num_tensors":
                num_tensors = sc.parse_uint()
            elif key == "total_bytes":
                total_bytes = sc.parse_uint()
            else:
                sc.skip_value()

            sc.skip_ws()
            if sc.peek() == 44:  # ','
                sc.pos += 1

        var res = BlockHeader(
            model=model,
            num_tensors=num_tensors,
            total_bytes=total_bytes,
            group_size=group_size,
            version=version,
        )
        res.format = format_str
        res.scale_dtype = scale_dtype
        return res^


@fieldwise_init
struct BlockTensorMeta(Copyable, Movable):
    """Metadata untuk satu tensor terkuantisasi."""

    var name: String
    var shape: List[Int]
    var dtype: String
    var quantized_dtype: String
    var group_size: Int
    var num_groups: Int
    var scale_offset: Int
    var data_offset: Int

    def __init__(
        out self,
        name: String,
        shape: List[Int],
        dtype: String = "BF16",
        group_size: Int = QUANT_DEFAULT_GROUP_SIZE,
    ) raises:
        if not is_allowed_group_size(group_size):
            raise Error("BlockTensorMeta: group_size not in {32, 64, 128, 256}")
        self.name = name
        self.shape = shape.copy()
        self.dtype = dtype
        self.quantized_dtype = QUANTIZED_DTYPE_NAME
        self.group_size = group_size

        var n_elem = 1
        for i in range(len(shape)):
            n_elem *= shape[i]

        if n_elem % group_size != 0:
            raise Error(
                "BlockTensorMeta: N % G != 0 (tail group not supported)"
            )
        self.num_groups = n_elem // group_size
        self.scale_offset = 0
        self.data_offset = self.num_groups * 2

    def num_elements(self) -> Int:
        """Menghitung total elemen dalam tensor."""
        var n = 1
        for i in range(len(self.shape)):
            n *= self.shape[i]
        return n

    def scales_bytes(self) -> Int:
        """Panjang buffer skala FP16 dalam byte."""
        return self.num_groups * 2

    def weights_bytes(self) -> Int:
        """Panjang buffer bobot ter-pack 4-bit dalam byte."""
        return (self.num_elements() + 1) // 2

    def payload_bytes(self) -> Int:
        """Total panjang data kuantisasi (skala + bobot ter-pack)."""
        return self.scales_bytes() + self.weights_bytes()

    def to_json(self) -> String:
        """Menghasilkan representasi JSON metadata tensor."""
        var shape_str = String("[")
        for i in range(len(self.shape)):
            if i > 0:
                shape_str += ","
            shape_str += String(self.shape[i])
        shape_str += "]"

        return String(
            '{"name":"',
            json_escape(self.name),
            '","shape":',
            shape_str,
            ',"dtype":"',
            json_escape(self.dtype),
            '","quantized_dtype":"',
            json_escape(self.quantized_dtype),
            '","group_size":',
            String(self.group_size),
            ',"num_groups":',
            String(self.num_groups),
            ',"scale_offset":',
            String(self.scale_offset),
            ',"data_offset":',
            String(self.data_offset),
            "}",
        )

    def to_record_bytes(self) -> List[UInt8]:
        """Mengemas metadata menjadi [u32 LE length][JSON UTF-8 bytes]."""
        var js = self.to_json()
        var b = js.as_bytes()
        var sz = len(b)
        var out = List[UInt8]()

        # 4-byte LE length
        out.append(UInt8(sz & 0xFF))
        out.append(UInt8((sz >> 8) & 0xFF))
        out.append(UInt8((sz >> 16) & 0xFF))
        out.append(UInt8((sz >> 24) & 0xFF))

        # JSON bytes
        for i in range(sz):
            out.append(b[i])
        return out^

    @staticmethod
    def from_json_bytes(raw: List[UInt8]) raises -> BlockTensorMeta:
        """Mem-parsing BlockTensorMeta dari buffer UTF-8 JSON."""
        var copy_raw = List[UInt8]()
        for i in range(len(raw)):
            copy_raw.append(raw[i])
        var sc = Scanner(copy_raw^, "tensor_metadata")

        sc.expect(123)  # '{'
        var name = String("")
        var shape = List[Int]()
        var dtype = String("BF16")
        var q_dtype = QUANTIZED_DTYPE_NAME
        var group_size = QUANT_DEFAULT_GROUP_SIZE
        var num_groups = 0
        var scale_off = 0
        var data_off = 0

        while True:
            sc.skip_ws()
            if sc.peek() == 125:  # '}'
                sc.pos += 1
                break
            var key = sc.parse_string()
            sc.expect(58)  # ':'
            if key == "name":
                name = sc.parse_string()
            elif key == "shape":
                sc.expect(91)  # '['
                while True:
                    sc.skip_ws()
                    if sc.peek() == 93:  # ']'
                        sc.pos += 1
                        break
                    shape.append(sc.parse_uint())
                    sc.skip_ws()
                    if sc.peek() == 44:  # ','
                        sc.pos += 1
            elif key == "dtype":
                dtype = sc.parse_string()
            elif key == "quantized_dtype":
                q_dtype = sc.parse_string()
            elif key == "group_size":
                group_size = sc.parse_uint()
            elif key == "num_groups":
                num_groups = sc.parse_uint()
            elif key == "scale_offset":
                scale_off = sc.parse_uint()
            elif key == "data_offset":
                data_off = sc.parse_uint()
            else:
                sc.skip_value()

            sc.skip_ws()
            if sc.peek() == 44:  # ','
                sc.pos += 1

        var meta = BlockTensorMeta(
            name=name, shape=shape, dtype=dtype, group_size=group_size
        )
        meta.quantized_dtype = q_dtype
        meta.num_groups = num_groups
        meta.scale_offset = scale_off
        meta.data_offset = data_off
        return meta^


def validate_block_header(header: BlockHeader, file_size: Int = -1) raises:
    """Memvalidasi integritas header kuantisasi."""
    if header.version != 1:
        raise Error(
            "Invalid quant header version: expected 1, got "
            + String(header.version)
        )
    if header.group_size <= 0:
        raise Error(
            "Group size must be positive, got " + String(header.group_size)
        )
    if not is_allowed_group_size(header.group_size):
        raise Error(
            "Group size must be one of {32, 64, 128, 256}, got "
            + String(header.group_size)
        )
    if header.format != QUANT_FORMAT_NAME:
        raise Error("Unsupported quantization format: " + header.format)
    if header.scale_dtype != QUANT_SCALE_DTYPE:
        raise Error("Unsupported scale dtype: " + header.scale_dtype)
    if header.num_tensors <= 0:
        raise Error(
            "Header num_tensors must be positive, got "
            + String(header.num_tensors)
        )
    if header.num_tensors > CONFIGURED_MAX_TENSORS:
        raise Error(
            "Header num_tensors exceeds configured max: "
            + String(header.num_tensors)
            + " > "
            + String(CONFIGURED_MAX_TENSORS)
        )
    if file_size > 0 and header.total_bytes != file_size:
        raise Error(
            "File size mismatch: header total_bytes="
            + String(header.total_bytes)
            + ", physical size="
            + String(file_size)
        )


def validate_block_tensor_meta(meta: BlockTensorMeta) raises:
    """Memvalidasi integritas metadata tensor."""
    if meta.name.byte_length() == 0:
        raise Error("Tensor name cannot be empty")
    if meta.name.byte_length() > CONFIGURED_MAX_NAME:
        raise Error(
            "Tensor name length exceeds configured max: "
            + String(meta.name.byte_length())
            + " > "
            + String(CONFIGURED_MAX_NAME)
        )
    if not is_allowed_group_size(meta.group_size):
        raise Error(
            "Group size must be one of {32, 64, 128, 256}, got "
            + String(meta.group_size)
        )
    if len(meta.shape) == 0:
        raise Error("Tensor shape cannot be empty")
    if len(meta.shape) > CONFIGURED_MAX_NDIM:
        raise Error(
            "Tensor ndim exceeds configured max: "
            + String(len(meta.shape))
            + " > "
            + String(CONFIGURED_MAX_NDIM)
        )

    var num_elements = 1
    for i in range(len(meta.shape)):
        var dim = meta.shape[i]
        if dim <= 0:
            raise Error("Dimension must be positive, got " + String(dim))
        num_elements = safe_multiply_int(num_elements, dim)

    if num_elements % meta.group_size != 0:
        raise Error(
            "Tensor num_elements must be divisible by group_size: "
            + String(num_elements)
            + " % "
            + String(meta.group_size)
            + " != 0"
        )
    var expected_groups = num_elements // meta.group_size
    if meta.num_groups != expected_groups:
        raise Error(
            "num_groups mismatch: meta="
            + String(meta.num_groups)
            + ", expected="
            + String(expected_groups)
        )


def calculate_tensor_quant_size(
    shape: List[Int], group_size: Int = QUANT_DEFAULT_GROUP_SIZE
) raises -> Tuple[Int, Int, Int, Int]:
    """Menghitung metrik ukuran kuantisasi untuk sebuah shape tensor.

    Kontrak (SSOT M6): group_size wajib himpunan izin dan N % G == 0
    (tanpa grup parsial); num_groups eksak.
    Mengembalikan Tuple (num_groups, scales_bytes, weights_bytes, total_bytes).
    Contoh: [2048, 2048] -> (32768, 65536, 2097152, 2162688).
    """
    if not is_allowed_group_size(group_size):
        raise Error(
            "calculate_tensor_quant_size: group_size not in {32, 64, 128, 256}"
        )
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    if n % group_size != 0:
        raise Error(
            "calculate_tensor_quant_size: N % G != 0 (tail group not supported)"
        )
    var num_groups = n // group_size
    var scales_bytes = num_groups * 2
    var weights_bytes = (n + 1) // 2
    var total_bytes = scales_bytes + weights_bytes
    return (num_groups, scales_bytes, weights_bytes, total_bytes)


def validate_quant_payload(
    scales: List[Float16], packed_weights: List[UInt8], num_elements: Int
) -> Bool:
    """Memvalidasi bahwa scales finite dan weights dalam range [-7, 7].

    Nibble reserved 0x8 (-8) ditolak; panjang buffer wajib eksak.
    """
    for i in range(len(scales)):
        var s = scales[i]
        var u = float16_to_u16(s)
        var exp_bits = (u >> 10) & 0x1F
        # NaN or Inf in FP16: exp == 0x1F
        if exp_bits == 0x1F:
            return False
        # Scale harus positif non-zero
        if s <= Float16(0.0):
            return False

    var exp_packed = (num_elements + 1) // 2
    if len(packed_weights) != exp_packed:
        return False

    for i in range(len(packed_weights)):
        var b = packed_weights[i]
        var lo = Int(b & 0x0F)
        var hi = Int((b >> 4) & 0x0F)
        if lo == 8 or hi == 8:
            return False

    return True


@fieldwise_init
struct QuantTensorEntry(Copyable, Movable):
    """Informasi lokasi tensor terkuantisasi di dalam berkas model binary."""

    var meta: BlockTensorMeta
    var record_offset: Int
    var payload_offset: Int
    var scales_offset: Int
    var data_offset: Int
    var scales_bytes: Int
    var weights_bytes: Int
    var total_record_bytes: Int


@fieldwise_init
struct QuantModelIndex(Movable):
    """Indeks pencarian tensor dalam model quant biner dengan validasi SEC-4."""

    var header: BlockHeader
    var entries: List[QuantTensorEntry]
    var tensor_map: Dict[String, Int]
    var file_path: String


def scan_quant_file(file_path: String) raises -> QuantModelIndex:
    """Memindai berkas biner kuantisasi dan memvalidasi seluruh invarian SEC-4.

    Header dipindai terlebih dahulu, diikuti validasi seluruh rentang record
    tanpa tumpang tindih (non-overlapping) dan penegakan batas ukuran eksak.
    """
    var f = open(file_path, "r")
    var file_size = Int(f.seek(0, SEEK_END))
    if file_size < QUANT_HEADER_SIZE:
        f.close()
        raise Error(
            "scan_quant_file: file truncated, size "
            + String(file_size)
            + " < header size "
            + String(QUANT_HEADER_SIZE)
        )

    _ = f.seek(0, SEEK_SET)
    var hdr_bytes = f.read_bytes(QUANT_HEADER_SIZE)
    var header: BlockHeader
    try:
        header = BlockHeader.from_bytes(hdr_bytes)
    except e:
        f.close()
        raise Error(
            "scan_quant_file: failed parsing quant header: " + String(e)
        )

    try:
        validate_block_header(header, file_size)
    except e:
        f.close()
        raise Error(
            "scan_quant_file: header validation failed (SEC-4): " + String(e)
        )

    var entries = List[QuantTensorEntry]()
    var tensor_map = Dict[String, Int]()
    var cur_off = QUANT_HEADER_SIZE

    for t_idx in range(header.num_tensors):
        if cur_off + 4 > file_size:
            f.close()
            raise Error(
                "scan_quant_file: framing truncated before meta_len at tensor "
                + String(t_idx)
            )

        _ = f.seek(cur_off, SEEK_SET)
        var len_bytes = f.read_bytes(4)
        var meta_len = (
            Int(len_bytes[0])
            | (Int(len_bytes[1]) << 8)
            | (Int(len_bytes[2]) << 16)
            | (Int(len_bytes[3]) << 24)
        )

        if meta_len <= 0 or cur_off + 4 + meta_len > file_size:
            f.close()
            raise Error(
                "scan_quant_file: invalid meta_len ("
                + String(meta_len)
                + ") at offset "
                + String(cur_off)
            )

        var json_bytes = f.read_bytes(meta_len)
        var meta: BlockTensorMeta
        try:
            meta = BlockTensorMeta.from_json_bytes(json_bytes)
        except e:
            f.close()
            raise Error(
                "scan_quant_file: failed parsing tensor metadata JSON at"
                " tensor "
                + String(t_idx)
                + ": "
                + String(e)
            )

        try:
            validate_block_tensor_meta(meta)
        except e:
            f.close()
            raise Error(
                "scan_quant_file: tensor metadata failed SEC-4 validation: "
                + String(e)
            )

        var s_bytes = meta.scales_bytes()
        var w_bytes = meta.weights_bytes()
        var payload_bytes = s_bytes + w_bytes
        var rec_total_bytes = 4 + meta_len + payload_bytes

        # Region non-overlap & boundary check
        if cur_off + rec_total_bytes > file_size:
            f.close()
            raise Error(
                "scan_quant_file: tensor payload truncated: "
                + meta.name
                + " (offset "
                + String(cur_off + rec_total_bytes)
                + " > "
                + String(file_size)
                + ")"
            )

        var payload_offset = cur_off + 4 + meta_len
        var scales_offset = payload_offset + meta.scale_offset
        var data_offset = payload_offset + meta.data_offset

        var entry = QuantTensorEntry(
            meta=meta.copy(),
            record_offset=cur_off,
            payload_offset=payload_offset,
            scales_offset=scales_offset,
            data_offset=data_offset,
            scales_bytes=s_bytes,
            weights_bytes=w_bytes,
            total_record_bytes=rec_total_bytes,
        )

        tensor_map[meta.name] = len(entries)
        entries.append(entry^)
        cur_off += rec_total_bytes

    if cur_off != file_size:
        f.close()
        raise Error(
            "scan_quant_file: trailing data or size mismatch (read "
            + String(cur_off)
            + " != file_size "
            + String(file_size)
            + ")"
        )

    f.close()
    return QuantModelIndex(
        header=header^,
        entries=entries^,
        tensor_map=tensor_map^,
        file_path=file_path,
    )


def pread_tensor_quant(
    file_path: String, entry: QuantTensorEntry
) raises -> Tuple[List[Float16], List[UInt8]]:
    """Membaca data kuantisasi mentah (skala FP16 dan bobot 4-bit ter-pack) dari disk.
    """
    var f = open(file_path, "r")
    _ = f.seek(entry.scales_offset, SEEK_SET)
    var s_raw = f.read_bytes(entry.scales_bytes)

    _ = f.seek(entry.data_offset, SEEK_SET)
    var w_raw = f.read_bytes(entry.weights_bytes)
    f.close()

    var num_groups = entry.meta.num_groups
    var scales = List[Float16]()
    scales.resize(num_groups, Float16(0.0))
    for g in range(num_groups):
        var u = UInt16(s_raw[g * 2]) | (UInt16(s_raw[g * 2 + 1]) << 8)
        scales[g] = u16_to_float16(u)

    var packed = List[UInt8]()
    packed.resize(len(w_raw), 0)
    for i in range(len(w_raw)):
        packed[i] = w_raw[i]

    return (scales^, packed^)


def pread_and_dequant_tensor(
    file_path: String, entry: QuantTensorEntry
) raises -> List[BFloat16]:
    """Membaca tensor dari disk dan mendekuantisasi ke List[BFloat16] via kernel SIMD.
    """
    var raw_tuple = pread_tensor_quant(file_path, entry)
    var scales = raw_tuple[0].copy()
    var packed = raw_tuple[1].copy()

    return dequant_kernel_simd(
        scales,
        packed,
        entry.meta.num_elements(),
        entry.meta.group_size,
    )


def pread_and_dequant_tensor_f32(
    file_path: String, entry: QuantTensorEntry
) raises -> List[Float32]:
    """Membaca tensor dari disk dan mendekuantisasi ke List[Float32] untuk layer forward.
    """
    var raw_tuple = pread_tensor_quant(file_path, entry)
    var scales = raw_tuple[0].copy()
    var packed = raw_tuple[1].copy()

    return dequant_kernel_simd_f32(
        scales,
        packed,
        entry.meta.num_elements(),
        entry.meta.group_size,
    )
