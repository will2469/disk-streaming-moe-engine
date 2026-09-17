# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Uji integrasi Layer Forward streaming on-the-fly dari berkas quant (M6-W4)."""

from core.config import ModelConfig
from format.quant_reader import scan_quant_file
from layers.quant_loader import (
    load_one_quant_tensor_f32,
    load_quant_routed_expert_weights,
)
from layers.swiglu import swiglu_forward
from std.collections import List
from std.math import isinf, isnan
from std.sys import argv


def main() raises:
    var args = argv()
    var quant_file = String("")
    for i in range(1, len(args)):
        if args[i] == "--quant-file" and i + 1 < len(args):
            quant_file = args[i + 1]

    if quant_file == "":
        raise Error("Usage: test_m6_w4_forward --quant-file <path>")

    # 1. Pindai berkas kuantisasi dan validasi invarian SEC-4
    var index = scan_quant_file(quant_file)

    # 2. Parameter konfigurasi Qwen MoE
    var cfg = ModelConfig(2048, 24, 16, 151936)

    # 3. Muat bobot routed expert on-the-fly via pread + dequant SIMD
    var swiglu = load_quant_routed_expert_weights(
        index, layer_idx=0, expert_id=0, cfg=cfg
    )

    # 4. Siapkan aktivasi input token x [1, 2048]
    var hidden = cfg.hidden_size
    var x = List[Float32]()
    x.resize(hidden, 0.0)
    for j in range(hidden):
        x[j] = Float32((j % 100) - 50) * Float32(0.01)

    # 5. Jalankan eksekusi SwiGLU forward
    var y = swiglu_forward(x, swiglu, seq_len=1)
    if len(y) != hidden:
        raise Error("Output length mismatch: expected " + String(hidden))

    # 6. Validasi tidak ada NaN atau Inf
    for j in range(hidden):
        if isnan(y[j]) or isinf(y[j]):
            raise Error("Non-finite output in forward on quantized weights")

    print(
        '{"status":"success","forward_ok":true,"output_len":'
        + String(len(y))
        + "}"
    )
