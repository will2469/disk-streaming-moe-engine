# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Model analitis F3b (traffic per token) dan F5 (forecast latensi serial) untuk decode incremental (M5)."""

from layers.kv_cache import NUM_LAYERS

# Konstanta Traffic Disk Terkunci (F3b DoD M5)
# 2.0668 B parameter BF16 streamed per token (attn + 4 routed + shared + router/norm/gate)
comptime B_TOK_DISK_BYTES: Int = 4133600000  # ≈ 4.134 GB

# Ukuran transfer KV per token per layer: 8 KiB (K + V BF16)
comptime KV_SLOT_BYTES_PER_LAYER: Int = 8192

# Traffic per layer transfer konstan (L * 8 KiB = 192 KiB)
comptime KV_LAYER_TOTAL_SLOT_BYTES: Int = NUM_LAYERS * KV_SLOT_BYTES_PER_LAYER  # 196,608 B = 192 KiB


@fieldwise_init
struct F3bTraffic(Copyable, Movable):
    """Traffic per token decode steady-state terdekomposisi.

    Catatan normatif: label B_tok telanjang DILARANG tanpa subscript.
    """

    var b_tok_disk: Int

    def __init__(out self):
        self.b_tok_disk = B_TOK_DISK_BYTES

    @staticmethod
    def compute_b_tok_kv_read(s: Int) -> Int:
        """Traffic baca histori KV cache: L * S * 8 KiB = S * 192 KiB."""
        return s * KV_LAYER_TOTAL_SLOT_BYTES

    @staticmethod
    def compute_b_tok_kv_write() -> Int:
        """Traffic tulis slot KV baru: L * 8 KiB = 192 KiB konstan."""
        return KV_LAYER_TOTAL_SLOT_BYTES

    @staticmethod
    def compute_b_tok_kv_total(s: Int) -> Int:
        """Total traffic KV cache (read + write): (S + 1) * 192 KiB."""
        return (s + 1) * KV_LAYER_TOTAL_SLOT_BYTES

    @staticmethod
    def compute_b_tok_ram(s: Int, rho_b: Float64) -> Float64:
        """Traffic RAM per token: rho_B * B_tok_disk + B_tok_kv."""
        var b_disk = Float64(B_TOK_DISK_BYTES)
        var b_kv = Float64(F3bTraffic.compute_b_tok_kv_total(s))
        return (rho_b * b_disk) + b_kv

    @staticmethod
    def compute_b_tok_total(s: Int) -> Int:
        """Total traffic (union disk + RAM KV tanpa double-counting)."""
        return B_TOK_DISK_BYTES + F3bTraffic.compute_b_tok_kv_total(s)


@fieldwise_init
struct F5Forecast(Copyable, Movable):
    """Prediksi latensi per-token F5 (v1): serial T_tok = T_data + T_kv + T_comp + T_ovh.
    """

    var t_data: Float64
    var bw_eff: Float64
    var t_kv: Float64
    var t_comp: Float64
    var t_ovh: Float64
    var t_tok: Float64

    def __init__(
        out self,
        s: Int,
        rho_b: Float64,
        bw_ram_bytes_sec: Float64,
        bw_ssd_bytes_sec: Float64,
        t_comp_sec: Float64 = 0.05,
        t_ovh_sec: Float64 = 0.005,
    ):
        """Menghitung forecast serial waktu eksekusi 1 token decode."""
        var b_disk = Float64(B_TOK_DISK_BYTES)
        # T_data = B_tok_disk * (rho_B / BW_RAM + (1 - rho_B) / BW_SSD)
        self.t_data = b_disk * (
            (rho_b / bw_ram_bytes_sec) + ((1.0 - rho_b) / bw_ssd_bytes_sec)
        )
        if self.t_data > 0.0:
            self.bw_eff = b_disk / self.t_data
        else:
            self.bw_eff = 0.0

        # T_kv = (B_tok_kv_read + B_tok_kv_write) / BW_RAM
        var b_kv = Float64(F3bTraffic.compute_b_tok_kv_total(s))
        self.t_kv = b_kv / bw_ram_bytes_sec

        self.t_comp = t_comp_sec
        self.t_ovh = t_ovh_sec
        self.t_tok = self.t_data + self.t_kv + self.t_comp + self.t_ovh

    def compute_relative_time_error(self, t_measured_sec: Float64) -> Float64:
        """Compute relative time error: e_T = |T_pred - T_meas| / T_meas."""
        if t_measured_sec <= 0.0:
            return 1.0
        var diff = self.t_tok - t_measured_sec
        if diff < 0.0:
            diff = -diff
        return diff / t_measured_sec

    def is_within_time_gate(
        self, t_measured_sec: Float64, threshold: Float64 = 0.30
    ) -> Bool:
        """Gate G-M5-4: e_T <= 30% terhadap prediksi v1 frozen."""
        return self.compute_relative_time_error(t_measured_sec) <= threshold
