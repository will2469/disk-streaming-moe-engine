# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Operasi residual connection tensor: out = y + x."""

from std.collections import List
from std.math import isinf, isnan


def add_residual(
    y: List[Float32],
    x: List[Float32],
    layer_idx: Int = 0,
) raises -> List[Float32]:
    """Residual connection: out = y + x.

    @spec scratch/wave/m2/m2-w3-attention.md
    """
    if len(y) != len(x):
        raise Error(
            '{"error_type":"ATTENTION_ERROR","detail":"residual length'
            ' mismatch","stage":"residual","layer":'
            + String(layer_idx)
            + "}"
        )
    var n = len(y)
    var out = List[Float32]()
    out.resize(n, Float32(0.0))
    var p_y = y.unsafe_ptr()
    var p_x = x.unsafe_ptr()
    var p_out = out.unsafe_ptr()
    for i in range(n):
        var vy = p_y[unsafe_offset=i]
        var vx = p_x[unsafe_offset=i]
        if isnan(vy) or isinf(vy) or isnan(vx) or isinf(vx):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"non-finite value in'
                ' residual","stage":"residual","layer":'
                + String(layer_idx)
                + "}"
            )
        var val = vy + vx
        if isnan(val) or isinf(val):
            raise Error(
                '{"error_type":"ATTENTION_ERROR","detail":"overflow in'
                ' residual addition","stage":"residual","layer":'
                + String(layer_idx)
                + "}"
            )
        p_out[unsafe_offset=i] = val
    return out^
