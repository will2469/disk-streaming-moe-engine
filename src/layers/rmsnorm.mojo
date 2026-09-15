# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""RMSNorm Layer (F6) — Root Mean Square Normalization fp32."""

from std.collections import List
from std.math import isinf, isnan, sqrt


def rmsnorm(
    x: List[Float32],
    gamma: List[Float32],
    eps: Float32,
) raises -> List[Float32]:
    """RMSNorm F6: y = x / sqrt(mean(x^2) + eps) ⊙ gamma, fp32.

    @spec scratch/wave/m1/m1-w1-rmsnorm.md (F6)
    Kontrak eps: parameter wajib tanpa default diam-diam. Sumber kanonis =
    field `rms_norm_eps` di `fixtures/m1/model_config.json` (artefak W4);
    kernel ini menerima nilainya sebagai argumen. eps NaN/Inf/<= 0 →
    CONFIG_ERROR (NaN <= 0 adalah false, jadi isnan/isinf wajib eksplisit);
    input kosong / panjang mismatch → NORM_ERROR.
    """
    var n = len(x)
    if n == 0:
        raise Error(
            '{"error_type":"NORM_ERROR","detail":"empty'
            ' input","stage":"rmsnorm"}'
        )
    if len(gamma) != n:
        raise Error(
            '{"error_type":"NORM_ERROR","detail":"length'
            ' mismatch","stage":"rmsnorm"}'
        )
    if isnan(eps) or isinf(eps) or eps <= Float32(0.0):
        raise Error(
            '{"error_type":"CONFIG_ERROR","detail":"rms_norm_eps must be'
            ' finite and > 0",'
            '"stage":"config"}'
        )
    var acc = Float32(0.0)
    for i in range(n):
        acc += x[i] * x[i]
    var denom = sqrt(acc / Float32(n) + eps)
    var out = List[Float32]()
    for i in range(n):
        out.append(x[i] / denom * gamma[i])
    return out^
