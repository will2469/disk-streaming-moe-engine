#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Generator untuk fixture sintetis GGUF v3 mini (<5 MB) untuk M9-W3.

Menghasilkan berkas fixtures/m9_port_mini.gguf yang valid, mematuhi spesifikasi
GGUF v3 dan formula analitik F11b-GGUF exact size match (diff == 0 bytes).
"""

import os
import struct
from typing import List, Tuple

# GGML Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_BF16 = 30

# GGUF Value Types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8


def align_to(offset: int, alignment: int = 32) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def encode_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def encode_metadata_kv(key: str, val_type: int, val_bytes: bytes) -> bytes:
    return encode_string(key) + struct.pack("<I", val_type) + val_bytes


def create_f32_block(num_elements: int, fill: float) -> bytes:
    """Payload F32 deterministik (norm-like / bias-like tensors)."""
    return struct.pack(f"<{num_elements}f", *[fill] * num_elements)


def create_q8_0_block(num_elements: int) -> bytes:
    """Sintetis Q8_0 payload (32 weights/block, 34 bytes/block)."""
    assert num_elements % 32 == 0
    num_blocks = num_elements // 32
    out = bytearray()
    for b in range(num_blocks):
        # 2 bytes skala FP16 (1.0 = 0x3C00)
        out.extend(struct.pack("<H", 0x3C00))
        # 32 bytes quants int8
        for j in range(32):
            val = ((b + j) % 15) - 7  # [-7, 7]
            out.append(val if val >= 0 else (256 + val))
    return bytes(out)


def create_q4_k_block(num_elements: int) -> bytes:
    """Sintetis Q4_K payload (256 weights/block, 144 bytes/block)."""
    assert num_elements % 256 == 0
    num_blocks = num_elements // 256
    out = bytearray()
    for b in range(num_blocks):
        # d (FP16 = 0.5 = 0x3800), dmin (FP16 = 0.1 = 0x2E66)
        out.extend(struct.pack("<HH", 0x3800, 0x2E66))
        # scales[12]
        out.extend(bytes([10] * 12))
        # qs[128]
        out.extend(bytes([0x23] * 128))
    return bytes(out)


def create_q3_k_block(num_elements: int) -> bytes:
    """Sintetis Q3_K payload (256 weights/block, 114 bytes/block)."""
    assert num_elements % 256 == 0
    num_blocks = num_elements // 256
    out = bytearray()
    for b in range(num_blocks):
        # hmask[32]
        out.extend(bytes([0x55] * 32))
        # qs[64]
        out.extend(bytes([0xAA] * 64))
        # scales[16]
        out.extend(bytes([34] * 16))
        # d (FP16 = 0.25 = 0x3400)
        out.extend(struct.pack("<H", 0x3400))
    return bytes(out)


def main():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    out_path = os.path.join(repo_root, "fixtures", "m9_port_mini.gguf")
    alignment = 32

    # Metadata KV
    metadata_kvs = [
        ("general.architecture", GGUF_TYPE_STRING, encode_string("qwen3.6")),
        ("general.alignment", GGUF_TYPE_UINT32, struct.pack("<I", alignment)),
        ("qwen3.6.vocab_size", GGUF_TYPE_UINT32, struct.pack("<I", 1024)),
        ("qwen3.6.embedding_length", GGUF_TYPE_UINT32, struct.pack("<I", 128)),
        ("qwen3.6.block_count", GGUF_TYPE_UINT32, struct.pack("<I", 4)),
    ]

    # Definisi Tensors untuk topologi mini 4 layer
    # [name, shape, dtype, data_bytes]
    tensors: List[Tuple[str, List[int], int, bytes]] = []

    # 1. token_embd (Q8_0, 1024 x 128 = 131072 elems -> 4096 blocks -> 139264 bytes)
    emb_elems = 1024 * 128
    tensors.append(
        ("token_embd.weight", [128, 1024], GGML_TYPE_Q8_0, create_q8_0_block(emb_elems))
    )

    # 2. output.weight / lm_head (Q8_0)
    tensors.append(
        ("output.weight", [128, 1024], GGML_TYPE_Q8_0, create_q8_0_block(emb_elems))
    )

    # 3. Model norms (F32, 128 elems)
    f32_norm = struct.pack(f"<{128}f", *[1.0] * 128)
    tensors.append(("output_norm.weight", [128], GGML_TYPE_F32, f32_norm))

    # 4. Layer 0..3 tensors (full coverage untuk compute GGUF-backed:
    #    norm + attention/GDN + router + 8 routed experts + shared expert.
    #    Tensor legacy (attn_norm, ffn_norm, linear_attn.k/v, attn_q/k,
    #    ffn_gate_exps, ffn_down_exps.0) dipertahankan apa adanya untuk
    #    kompatibilitas backward (M9-W3 loader tests menunjuk ke nama itu).
    for layer_idx in range(4):
        pfx = f"blk.{layer_idx}."
        tensors.append((pfx + "attn_norm.weight", [128], GGML_TYPE_F32, f32_norm))
        tensors.append((pfx + "ffn_norm.weight", [128], GGML_TYPE_F32, f32_norm))

        if layer_idx % 4 != 3:  # GDN
            # linear_attn weights Q4_K (256 elems)
            tensors.append(
                (
                    pfx + "linear_attn.k.weight",
                    [128, 32],
                    GGML_TYPE_Q4_K,
                    create_q4_k_block(128 * 32),
                )
            )
            tensors.append(
                (
                    pfx + "linear_attn.v.weight",
                    [128, 32],
                    GGML_TYPE_Q4_K,
                    create_q4_k_block(128 * 32),
                )
            )
            # Pelengkap GDN: beta proj + output proj (diperlukan loader
            # GGUF-backed; tidak ada di fixture 27-tensor awal).
            # NOTA LAYOUT: dims GGUF [dv, hidden] agar torch-oracle
            # (reshape reversed) melihat [hidden, dv] untuk matmul(wout, ot).
            tensors.append(
                (
                    pfx + "linear_attn.beta.weight",
                    [128],
                    GGML_TYPE_F32,
                    create_f32_block(128, 0.02),
                )
            )
            tensors.append(
                (
                    pfx + "linear_attn.out.weight",
                    [32, 128],
                    GGML_TYPE_Q4_K,
                    create_q4_k_block(32 * 128),
                )
            )
        else:  # GatedAttn
            tensors.append(
                (
                    pfx + "attn_q.weight",
                    [128, 128],
                    GGML_TYPE_Q4_K,
                    create_q4_k_block(128 * 128),
                )
            )
            tensors.append(
                (
                    pfx + "attn_k.weight",
                    [128, 32],
                    GGML_TYPE_Q4_K,
                    create_q4_k_block(128 * 32),
                )
            )
            # Pelengkap GatedAttn: V + gate + output proj.
            tensors.append(
                (
                    pfx + "attn_v.weight",
                    [128, 32],
                    GGML_TYPE_Q4_K,
                    create_q4_k_block(128 * 32),
                )
            )
            tensors.append(
                (
                    pfx + "attn_gate.weight",
                    [128, 128],
                    GGML_TYPE_Q4_K,
                    create_q4_k_block(128 * 128),
                )
            )
            tensors.append(
                (
                    pfx + "attn_o.weight",
                    [128, 128],
                    GGML_TYPE_Q4_K,
                    create_q4_k_block(128 * 128),
                )
            )

        # MoE router (Q8_0, 8 experts x 128 hidden) — dipertahankan.
        tensors.append(
            (
                pfx + "ffn_gate_exps.weight",
                [128, 8],
                GGML_TYPE_Q8_0,
                create_q8_0_block(128 * 8),
            )
        )
        # MoE legacy probe tensor — dipertahankan untuk kompatibilitas.
        tensors.append(
            (
                pfx + "ffn_down_exps.0.weight",
                [64, 128],
                GGML_TYPE_Q3_K,
                create_q3_k_block(64 * 128),
            )
        )

        # Routed experts penuh 8 x (gate, up, down), inter=64, hidden=128.
        # Loader GGUF-backed memakai tensor ini (bukan tensor legacy di atas).
        for e in range(8):
            tensors.append(
                (
                    pfx + f"ffn_gate.{e}.weight",
                    [64, 128],
                    GGML_TYPE_Q3_K,
                    create_q3_k_block(64 * 128),
                )
            )
            tensors.append(
                (
                    pfx + f"ffn_up.{e}.weight",
                    [64, 128],
                    GGML_TYPE_Q3_K,
                    create_q3_k_block(64 * 128),
                )
            )
            tensors.append(
                (
                    pfx + f"ffn_down.{e}.weight",
                    [128, 64],
                    GGML_TYPE_Q3_K,
                    create_q3_k_block(128 * 64),
                )
            )

        # Shared expert penuh + sigmoid gate.
        # NOTA LAYOUT: dims GGUF [in, out] agar torch-oracle (reshape
        # reversed, F.linear) melihat [out, in] yang benar.
        tensors.append(
            (
                pfx + "ffn_shared_gate.weight",
                [128, 64],
                GGML_TYPE_Q3_K,
                create_q3_k_block(128 * 64),
            )
        )
        tensors.append(
            (
                pfx + "ffn_shared_up.weight",
                [128, 64],
                GGML_TYPE_Q3_K,
                create_q3_k_block(128 * 64),
            )
        )
        tensors.append(
            (
                pfx + "ffn_shared_down.weight",
                [64, 128],
                GGML_TYPE_Q3_K,
                create_q3_k_block(64 * 128),
            )
        )
        tensors.append(
            (
                pfx + "shared_gate.weight",
                [128],
                GGML_TYPE_F32,
                create_f32_block(128, 0.02),
            )
        )

    # Encode Header
    header = bytearray()
    header.extend(b"GGUF")  # magic
    header.extend(struct.pack("<I", 3))  # version 3
    header.extend(struct.pack("<Q", len(tensors)))  # tensor_count
    header.extend(struct.pack("<Q", len(metadata_kvs)))  # metadata_kv_count

    # Encode Metadata
    for key, val_type, val_bytes in metadata_kvs:
        header.extend(encode_metadata_kv(key, val_type, val_bytes))

    # Hitung tensor info directory dan data payload
    tensor_info_dir = bytearray()
    data_payload = bytearray()

    # Hitung data_start setelah tensor info dir
    # Kita buat 2 pass untuk mendapatkan offset yang tepat
    cur_data_offset = 0
    tensor_offsets = []

    for name, shape, dtype, raw_bytes in tensors:
        cur_data_offset = align_to(cur_data_offset, alignment)
        tensor_offsets.append(cur_data_offset)
        cur_data_offset += len(raw_bytes)

    # Encode tensor info
    for idx, (name, shape, dtype, raw_bytes) in enumerate(tensors):
        tensor_info_dir.extend(encode_string(name))
        tensor_info_dir.extend(struct.pack("<I", len(shape)))
        for dim in shape:
            tensor_info_dir.extend(struct.pack("<Q", dim))
        tensor_info_dir.extend(struct.pack("<I", dtype))
        tensor_info_dir.extend(struct.pack("<Q", tensor_offsets[idx]))

    # Hitung data_start
    total_header_bytes = len(header) + len(tensor_info_dir)
    data_start = align_to(total_header_bytes, alignment)
    pad_header = data_start - total_header_bytes

    # Assemble data payload dengan padding antar tensor
    for idx, (name, shape, dtype, raw_bytes) in enumerate(tensors):
        pad_len = tensor_offsets[idx] - len(data_payload)
        if pad_len > 0:
            data_payload.extend(b"\x00" * pad_len)
        data_payload.extend(raw_bytes)

    # Tulis file biner
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(tensor_info_dir)
        if pad_header > 0:
            f.write(b"\x00" * pad_header)
        f.write(data_payload)

    file_size = os.path.getsize(out_path)
    expected_size = data_start + len(data_payload)
    assert file_size == expected_size, f"Size mismatch: {file_size} != {expected_size}"
    print(
        f"Generated synthetic GGUF: {out_path} "
        f"({file_size} bytes, {len(tensors)} tensors)"
    )


if __name__ == "__main__":
    main()
