# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle Dequantisasi & Conformance Harness G-M6-K (M6-W4).

Spesifikasi (docs/milestones/M6-quantizer.md § Conformance vs Oracle (G-M6-K)):
1. Fixture seed-42:
   - >= 1024 grup x 128 = 131.072 bobot
   - Skala FP16 acak finite positif (termasuk subnormal min 0x0001 dan max finite 65504)
   - Nibble acak mencakup 15 nilai valid [-7, 7]
   - Perilaku penolakan identik pada nibble reserved 0b1000 (-8)
2. Single-rounding bit-identical:
   y = float32(s_g) * q_j
   bf16_val = torch.tensor(y, dtype=torch.float32).to(torch.bfloat16)
3. Verifikasi file-level:
   Dequant oracle vs SIMD pada berkas quant biner (.kimo.bin).
"""

import argparse
import json
import os
import struct
import sys
from typing import Any

import numpy as np
import torch

from tools.quant.quant_format import QUANT_HEADER_SIZE, unpack_4bit_pair


def dequant_kernel_oracle(
    scales_raw: bytes, packed_raw: bytes, group_size: int = 128
) -> bytes:
    """Oracle dekuantisasi 4-bit ke BF16 single-rounding.

    Menolak reserved nibble 0b1000 (-8) dan skala non-finite / non-positif.
    """
    if len(scales_raw) % 2 != 0:
        raise ValueError("scales_raw length must be even (FP16)")
    num_groups = len(scales_raw) // 2
    num_elements = num_groups * group_size
    exp_packed = (num_elements + 1) // 2
    if len(packed_raw) != exp_packed:
        raise ValueError(
            f"packed_raw length mismatch: got {len(packed_raw)}, expected {exp_packed}"
        )

    # Parse scales FP16
    scales = []
    for g in range(num_groups):
        u16_val = struct.unpack_from("<H", scales_raw, g * 2)[0]
        # Cek NaN / Inf: bits 10-14 all 1
        exp_bits = (u16_val >> 10) & 0x1F
        if exp_bits == 0x1F:
            raise ValueError(f"scale at group {g} is NaN or Inf")
        # Konversi bits ke FP16 via numpy
        f16_val = np.frombuffer(struct.pack("<H", u16_val), dtype=np.float16)[0]
        if f16_val <= 0.0:
            raise ValueError(f"scale at group {g} is non-positive: {f16_val}")
        scales.append(float(np.float32(f16_val)))

    # Dequantize elements
    bf16_bytes = bytearray(num_elements * 2)
    elem_idx = 0

    for g in range(num_groups):
        s = scales[g]
        bytes_in_group = group_size // 2
        g_byte_start = g * bytes_in_group

        for b in range(bytes_in_group):
            byte_val = packed_raw[g_byte_start + b]
            w0, w1 = unpack_4bit_pair(byte_val)

            # Elemen 0 (low nibble)
            y0 = s * float(w0)
            t0 = torch.tensor(y0, dtype=torch.float32).to(torch.bfloat16)
            b0_u16 = t0.view(torch.int16).item() & 0xFFFF
            struct.pack_into("<H", bf16_bytes, elem_idx * 2, b0_u16)
            elem_idx += 1

            # Elemen 1 (high nibble)
            y1 = s * float(w1)
            t1 = torch.tensor(y1, dtype=torch.float32).to(torch.bfloat16)
            b1_u16 = t1.view(torch.int16).item() & 0xFFFF
            struct.pack_into("<H", bf16_bytes, elem_idx * 2, b1_u16)
            elem_idx += 1

    return bytes(bf16_bytes)


def generate_conformance_fixtures(
    output_dir: str, num_groups: int = 1024, group_size: int = 128
) -> dict[str, str]:
    """Membangkitkan fixture seed-42 untuk pengujian G-M6-K."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. Bangkitkan skala FP16
    scales_u16 = []
    # Titik pojok khusus FP16
    scales_u16.append(0x0001)  # subnormal terkecil (~5.96e-8)
    scales_u16.append(0x03FF)  # subnormal terbesar
    scales_u16.append(0x0400)  # normal terkecil (6.10e-5)
    scales_u16.append(0x3C00)  # 1.0 tepat
    scales_u16.append(0x7BFF)  # nilai finite maksimum (65504.0)

    # Sisa skala: nilai normal acak dalam rentang representabel positif
    for _ in range(num_groups - len(scales_u16)):
        # Rentang exponent 1..30 (finite, non-zero)
        exp = rng.integers(1, 31)
        mant = rng.integers(0, 1024)
        u16 = (exp << 10) | mant
        scales_u16.append(int(u16))

    scales_raw = bytearray()
    for u in scales_u16:
        scales_raw.extend(struct.pack("<H", u))

    # 2. Bangkitkan bobot 4-bit acak valid (tanpa nibble 0x8)
    valid_nibbles = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    total_weights = num_groups * group_size
    packed_raw = bytearray(total_weights // 2)

    for i in range(total_weights // 2):
        n0 = int(rng.choice(valid_nibbles))
        n1 = int(rng.choice(valid_nibbles))
        packed_raw[i] = (n1 << 4) | n0

    # 3. Hitung golden output BF16 via oracle
    golden_bf16 = dequant_kernel_oracle(
        bytes(scales_raw), bytes(packed_raw), group_size
    )

    # 4. Bangkitkan fixture corrupt dengan reserved nibble 0b1000 (0x8)
    corrupt_packed = bytearray(packed_raw)
    corrupt_packed[42] = (corrupt_packed[42] & 0x0F) | 0x80  # high nibble = 8

    # Simpan berkas
    path_scales = os.path.join(output_dir, "fixture_seed42_scales.bin")
    path_packed = os.path.join(output_dir, "fixture_seed42_packed.bin")
    path_golden = os.path.join(output_dir, "fixture_seed42_golden_bf16.bin")
    path_corrupt = os.path.join(output_dir, "fixture_reserved_packed.bin")

    with open(path_scales, "wb") as f:
        f.write(scales_raw)
    with open(path_packed, "wb") as f:
        f.write(packed_raw)
    with open(path_golden, "wb") as f:
        f.write(golden_bf16)
    with open(path_corrupt, "wb") as f:
        f.write(corrupt_packed)

    # Verifikasi oracle menolak corrupt fixture
    try:
        dequant_kernel_oracle(bytes(scales_raw), bytes(corrupt_packed), group_size)
        raise RuntimeError("Oracle gagal menolak reserved nibble 0x8!")
    except ValueError:
        pass  # Ditolak dengan benar!

    return {
        "scales": path_scales,
        "packed": path_packed,
        "golden": path_golden,
        "corrupt": path_corrupt,
        "num_groups": str(num_groups),
        "group_size": str(group_size),
        "total_elements": str(total_weights),
    }


def verify_file_level_conformance(
    quant_file: str,
) -> dict[str, Any]:
    """Membaca file .kimo.bin dan mendekuantisasi setiap tensor via oracle."""
    with open(quant_file, "rb") as f:
        hdr_bytes = f.read(QUANT_HEADER_SIZE)
        hdr_json_str = hdr_bytes.split(b"\x00")[0].decode("utf-8")
        header = json.loads(hdr_json_str)

        num_tensors = header["num_tensors"]
        group_size = header["quantization"]["group_size"]

        results = []
        for t in range(num_tensors):
            len_b = f.read(4)
            meta_len = struct.unpack("<I", len_b)[0]
            meta_str = f.read(meta_len).decode("utf-8")
            meta = json.loads(meta_str)

            num_groups = meta["num_groups"]
            s_bytes = num_groups * 2
            n_elem = 1
            for d in meta["shape"]:
                n_elem *= d
            w_bytes = (n_elem + 1) // 2

            scales_raw = f.read(s_bytes)
            packed_raw = f.read(w_bytes)

            bf16_out = dequant_kernel_oracle(
                scales_raw, packed_raw, group_size=group_size
            )
            results.append(
                {
                    "name": meta["name"],
                    "num_elements": n_elem,
                    "bf16_bytes_len": len(bf16_out),
                }
            )

    return {"status": "success", "tensors": results}


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Dequantisasi & Conformance G-M6-K"
    )
    parser.add_argument(
        "--generate-fixtures",
        type=str,
        help="Direktori target untuk membangkitkan fixture seed-42",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=1024,
        help="Jumlah grup untuk fixture seed-42 (default: 1024)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size (default: 128)",
    )
    parser.add_argument(
        "--verify-file",
        type=str,
        help="File quant .kimo.bin untuk verifikasi file-level",
    )
    args = parser.parse_args()

    if args.generate_fixtures:
        info = generate_conformance_fixtures(
            args.generate_fixtures,
            num_groups=args.num_groups,
            group_size=args.group_size,
        )
        print(json.dumps(info, indent=2))
    elif args.verify_file:
        res = verify_file_level_conformance(args.verify_file)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
