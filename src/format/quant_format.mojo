# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Spesifikasi, packing, parsing, dan validasi format kuantisasi 4-bit (M6-W1)."""

from format.scanner import Scanner
from format.types import json_escape
from std.collections import List


comptime QUANT_HEADER_SIZE: Int = 256
comptime QUANT_DEFAULT_GROUP_SIZE: Int = 128
comptime QUANT_FORMAT_NAME: String = "4-bit per-group"
comptime QUANT_SCALE_DTYPE: String = "FP16"
comptime QUANTIZED_DTYPE_NAME: String = "4-bit"


def is_allowed_group_size(group_size: Int) -> Bool:
    """Himpunan izin G ∈ {32, 64, 128, 256} (SSOT M6; satu-satunya definisi)."""
    return (
        group_size == 32
        or group_size == 64
        or group_size == 128
        or group_size == 256
    )


def pack_4bit_pair(w0: Int8, w1: Int8) raises -> UInt8:
    """Mengemas dua bobot bertanda [-7, 7] ke dalam 1 byte Little-Endian.

    w0 berada di low nibble (bits 0-3), w1 berada di high nibble (bits 4-7).
    Nibble 0x8 (-8) reserved/invalid: encoder tak pernah memancarkannya.
    """
    if Int(w0) < -7 or Int(w0) > 7 or Int(w1) < -7 or Int(w1) > 7:
        raise Error("pack_4bit_pair: values must be in [-7, 7] (0x8 reserved)")
    var u0 = UInt8(Int(w0) & 0x0F)
    var u1 = UInt8(Int(w1) & 0x0F)
    return (u1 << 4) | u0


def unpack_4bit_pair(b: UInt8) raises -> Tuple[Int8, Int8]:
    """Membongkar 1 byte Little-Endian menjadi dua bobot bertanda [-7, 7].

    Menolak nibble reserved 0x8 (-8) dengan Error.
    """
    var u0 = Int(b & 0x0F)
    var u1 = Int((b >> 4) & 0x0F)
    if u0 == 8 or u1 == 8:
        raise Error("unpack_4bit_pair: reserved nibble 0x8 (-8)")
    var w0 = Int8(u0 - 16 if u0 >= 8 else u0)
    var w1 = Int8(u1 - 16 if u1 >= 8 else u1)
    return (w0, w1)


def float16_to_u16(val: Float16) -> UInt16:
    """Mengonversi bit representasi Float16 ke UInt16."""
    var buf = List[UInt8]()
    buf.resize(2, 0)
    buf.unsafe_ptr().unsafe_bitcast[Float16]()[unsafe_offset=0] = val
    return buf.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=0]


def u16_to_float16(u: UInt16) -> Float16:
    """Mengonversi bit UInt16 ke Float16."""
    var buf = List[UInt8]()
    buf.resize(2, 0)
    buf.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=0] = u
    return buf.unsafe_ptr().unsafe_bitcast[Float16]()[unsafe_offset=0]


def compute_fp16_scale_ceil(max_abs: Float32) raises -> Float16:
    """Menghitung skala s_g = ceil_FP16(max_abs / 7.0).

    Bila max_abs == 0, mengembalikan 1.0 (grup nol, q = 0).
    Pembulatan ke atas menjamin Float32(s_g) * 7.0 >= max_abs (tanpa saturasi).
    """
    if max_abs <= 0.0:
        return Float16(1.0)

    var target = max_abs / 7.0
    var s = Float16(target)
    if Float32(s) * 7.0 < max_abs:
        var u = float16_to_u16(s)
        if u >= 0x7C00:  # FP16 Infinity
            raise Error("FP16 scale overflow (exceeds finite range)")
        u += 1
        s = u16_to_float16(u)
    return s


@fieldwise_init
struct QuantHeader(Copyable, Movable):
    """Header berkas kuantisasi 4-bit (tepat 256 byte)."""

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
    def from_bytes(raw: List[UInt8]) raises -> QuantHeader:
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

        var res = QuantHeader(
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
struct QuantTensorMetadata(Copyable, Movable):
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
            raise Error(
                "QuantTensorMetadata: group_size not in {32, 64, 128, 256}"
            )
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
                "QuantTensorMetadata: N % G != 0 (tail group not supported)"
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
    def from_json_bytes(raw: List[UInt8]) raises -> QuantTensorMetadata:
        """Mem-parsing QuantTensorMetadata dari buffer UTF-8 JSON."""
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
            if sc.peek() == 44:
                sc.pos += 1

        var meta = QuantTensorMetadata(
            name=name, shape=shape, dtype=dtype, group_size=group_size
        )
        meta.quantized_dtype = q_dtype
        meta.group_size = group_size
        meta.num_groups = num_groups
        meta.scale_offset = scale_off
        meta.data_offset = data_off
        return meta^


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


def validate_quant_header(header: QuantHeader, file_size: Int = -1) raises:
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
    # Group size wajib himpunan izin SSOT (bukan sembarang pangkat 2)
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
    if file_size > 0 and header.total_bytes != file_size:
        raise Error(
            "File size mismatch: header total_bytes="
            + String(header.total_bytes)
            + ", physical size="
            + String(file_size)
        )


def validate_tensor_meta(meta: QuantTensorMetadata) raises:
    """Memvalidasi integritas metadata tensor."""
    if meta.name.byte_length() == 0:
        raise Error("Tensor name cannot be empty")
    if not is_allowed_group_size(meta.group_size):
        raise Error(
            "Group size must be one of {32, 64, 128, 256}, got "
            + String(meta.group_size)
        )
    if len(meta.shape) == 0:
        raise Error("Tensor shape cannot be empty")
    var n = 1
    for i in range(len(meta.shape)):
        if meta.shape[i] <= 0:
            raise Error("Dimension must be positive: " + String(meta.shape[i]))
        n *= meta.shape[i]
    if n % meta.group_size != 0:
        raise Error("N % G != 0 (tail group not supported)")
    var exp_groups = n // meta.group_size
    if meta.num_groups != exp_groups:
        raise Error(
            "Mismatch in num_groups: expected "
            + String(exp_groups)
            + ", got "
            + String(meta.num_groups)
        )
    if meta.scale_offset != 0:
        raise Error("scale_offset must be 0, got " + String(meta.scale_offset))
    if meta.data_offset != meta.num_groups * 2:
        raise Error(
            "data_offset must be num_groups * 2 ("
            + String(meta.num_groups * 2)
            + "), got "
            + String(meta.data_offset)
        )


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
