# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Static Worker Pool dan Kontrak Determinisme Multithreading CPU (§3.2, M11-W1).

Mengimplementasikan:
1. Worker pool berbasis POSIX thread (pthread) dengan alokasi statis di inisialisasi.
2. Sinkronisasi dual-barrier (pthread_barrier) bebas alokasi di hot loop inferensi.
3. Fungsi partisi deterministik partition_range(total, thread_id, num_threads).
4. Kontrak 4 invarian determinisme (§3.2):
   - Partisi fixed f(thread_id, c)
   - Urutan akumulasi fixed (indeks worker naik)
   - Reduction tree deterministik
   - Bebas race condition dan tanpa atomic float tak-berurutan.
"""

from layers.swiglu import silu_f32
from std.builtin.dtype import DType
from std.collections import List
from std.ffi import external_call
from std.memory import Pointer

# Tipe tugas yang didukung oleh worker pool
comptime TASK_NONE = 0
comptime TASK_DEQUANT_SIMD = 1
comptime TASK_MOE_EXPERTS = 2


def partition_range(
    total_items: Int, thread_id: Int, num_threads: Int
) -> Tuple[Int, Int]:
    """Membagi total_items ke dalam interval [start, end) deterministik untuk thread_id.

    Invarian:
    - start(0) == 0
    - end(num_threads - 1) == total_items
    - start(i+1) == end(i) untuk seluruh i
    - Selisih beban antar worker maksimal 1 elemen (load balance optimal).
    """
    if num_threads <= 1:
        return (0, total_items)
    if total_items <= 0:
        return (0, 0)
    var chunk = total_items // num_threads
    var rem = total_items % num_threads
    var start = thread_id * chunk + (thread_id if thread_id < rem else rem)
    var count = chunk + (1 if thread_id < rem else 0)
    var end = start + count
    return (start, end)


def _worker_dequant_task(
    worker_id: Int, num_threads: Int, p_shared: Pointer[Int, MutAnyOrigin]
) raises:
    """Eksekusi partisi dequantisasi SIMD 4-bit ke BF16 untuk worker_id."""
    var num_elements = p_shared[unsafe_offset=20]
    var group_size = p_shared[unsafe_offset=21]
    var p_scales = Pointer[Float16, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=22]
    )
    var p_packed = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=23]
    )
    var p_out = Pointer[BFloat16, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=24]
    )

    var num_groups = num_elements // group_size
    var part = partition_range(num_groups, worker_id, num_threads)
    var start_g = part[0]
    var end_g = part[1]

    for g in range(start_g, end_g):
        var s_g = p_scales[unsafe_offset=g]
        var s_val = Float32(s_g)
        var s_simd = SIMD[DType.float32, 8](s_val)

        var g_elem_base = g * group_size
        var bytes_in_group = group_size // 2
        var g_byte_base = g * bytes_in_group

        var b = 0
        while b + 8 <= bytes_in_group:
            var raw = p_packed.unsafe_load[width=8](g_byte_base + b)
            var lo_s = (raw << 4).cast[DType.int8]() >> 4
            var hi_s = raw.cast[DType.int8]() >> 4

            if lo_s.reduce_min() < -7 or hi_s.reduce_min() < -7:
                raise Error(
                    "dequant_kernel: reserved nibble 0b1000 (-8) encountered"
                )

            var lo_out = (s_simd * lo_s.cast[DType.float32]()).cast[
                DType.bfloat16
            ]()
            var hi_out = (s_simd * hi_s.cast[DType.float32]()).cast[
                DType.bfloat16
            ]()

            var out_base = g_elem_base + b * 2
            for j in range(8):
                p_out[unsafe_offset=out_base + 2 * j] = lo_out[j]
                p_out[unsafe_offset=out_base + 2 * j + 1] = hi_out[j]

            b += 8


def _worker_moe_task(
    worker_id: Int, num_threads: Int, p_shared: Pointer[Int, MutAnyOrigin]
) raises:
    """Eksekusi partisi MoE expert SwiGLU GEMM untuk worker_id."""
    var top_k = p_shared[unsafe_offset=20]
    var hidden_dim = p_shared[unsafe_offset=21]
    var inter_dim = p_shared[unsafe_offset=22]
    var p_x_tok = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=23]
    )
    var p_out_routed = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=24]
    )
    var p_selected = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=25]
    )
    var p_w_gate_ptrs = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=26]
    )
    var p_w_up_ptrs = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=27]
    )
    var p_w_down_ptrs = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=28]
    )
    var p_h_scratch = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=p_shared[unsafe_offset=29]
    )

    var part = partition_range(top_k, worker_id, num_threads)
    var start_k = part[0]
    var end_k = part[1]
    var p_h = p_h_scratch.unsafe_offset(worker_id * inter_dim)

    for k in range(start_k, end_k):
        var exp_id = p_selected[unsafe_offset=k]
        var p_wg = Pointer[Float32, MutAnyOrigin](
            unsafe_from_address=p_w_gate_ptrs[unsafe_offset=exp_id]
        )
        var p_wu = Pointer[Float32, MutAnyOrigin](
            unsafe_from_address=p_w_up_ptrs[unsafe_offset=exp_id]
        )
        var p_wd = Pointer[Float32, MutAnyOrigin](
            unsafe_from_address=p_w_down_ptrs[unsafe_offset=exp_id]
        )

        var p_out_k = p_out_routed.unsafe_offset(k * hidden_dim)

        for j in range(inter_dim):
            var w_row = j * hidden_dim
            var g_simd = SIMD[DType.float32, 16](0.0)
            var u_simd = SIMD[DType.float32, 16](0.0)
            var c = 0
            while c + 16 <= hidden_dim:
                var xv = p_x_tok.unsafe_load[width=16](c)
                g_simd += xv * p_wg.unsafe_load[width=16](w_row + c)
                u_simd += xv * p_wu.unsafe_load[width=16](w_row + c)
                c += 16
            var g = g_simd.reduce_add()
            var u = u_simd.reduce_add()
            while c < hidden_dim:
                var xv = p_x_tok[unsafe_offset=c]
                g += xv * p_wg[unsafe_offset=w_row + c]
                u += xv * p_wu[unsafe_offset=w_row + c]
                c += 1
            var hj = silu_f32(g) * u
            p_h[unsafe_offset=j] = hj

        for d in range(hidden_dim):
            var wd_row = d * inter_dim
            var acc_simd = SIMD[DType.float32, 16](0.0)
            var c = 0
            while c + 16 <= inter_dim:
                acc_simd += p_wd.unsafe_load[width=16](
                    wd_row + c
                ) * p_h.unsafe_load[width=16](c)
                c += 16
            var y_d = acc_simd.reduce_add()
            while c < inter_dim:
                y_d += p_wd[unsafe_offset=wd_row + c] * p_h[unsafe_offset=c]
                c += 1
            p_out_k[unsafe_offset=d] = y_d


def _worker_thread_loop(p_arg: Pointer[Int, MutAnyOrigin]) -> Int:
    """Loop eksekusi utama thread pekerja latar belakang."""
    var worker_id = p_arg[unsafe_offset=0]
    var shared_addr = p_arg[unsafe_offset=1]
    var p_shared = Pointer[Int, MutAnyOrigin](unsafe_from_address=shared_addr)
    var p_bar_start = p_shared.unsafe_offset(0)
    var p_bar_done = p_shared.unsafe_offset(8)

    while True:
        # Menunggu sinyal mulai kerja dari koordinator
        _ = external_call["pthread_barrier_wait", Int32](p_bar_start)

        # Cek bendera terminasi
        if p_shared[unsafe_offset=16] == 1:
            break

        var task_type = p_shared[unsafe_offset=18]
        var num_threads = p_shared[unsafe_offset=17]

        if task_type == TASK_DEQUANT_SIMD:
            try:
                _worker_dequant_task(worker_id, num_threads, p_shared)
            except:
                p_shared[unsafe_offset=19] = 1
        elif task_type == TASK_MOE_EXPERTS:
            try:
                _worker_moe_task(worker_id, num_threads, p_shared)
            except:
                p_shared[unsafe_offset=19] = 1

        # Menunggu seluruh thread selesai sebelum siklus berikutnya
        _ = external_call["pthread_barrier_wait", Int32](p_bar_done)

    return 0


struct WorkerPool:
    """Worker pool statis zero-allocation di hot loop inferensi (M11-W1)."""

    var num_threads: Int
    var is_active: Bool
    var shared_mem: List[Int]
    var worker_args: List[Int]
    var thread_ids: List[Int]
    var moe_scratch: List[Float32]

    def __init__(out self, num_threads: Int = 1):
        self.num_threads = num_threads if num_threads > 0 else 1
        self.is_active = False
        # shared_mem layout:
        # [0..7]:   pthread_barrier_t barrier_start (32-64 bytes)
        # [8..15]:  pthread_barrier_t barrier_done (32-64 bytes)
        # [16]:     exit_flag (0 / 1)
        # [17]:     num_threads
        # [18]:     task_type
        # [19]:     error_flag (0 = ok, 1 = error)
        # [20..31]: task arguments
        self.shared_mem = List[Int]()
        for _ in range(64):
            self.shared_mem.append(0)

        # worker_args: 4 Int per thread
        self.worker_args = List[Int]()
        for _ in range(self.num_threads * 4):
            self.worker_args.append(0)

        self.thread_ids = List[Int]()
        for _ in range(self.num_threads):
            self.thread_ids.append(0)

        self.moe_scratch = List[Float32]()
        self.moe_scratch.resize(self.num_threads * 8192, Float32(0.0))

        if self.num_threads > 1:
            var p_shared = self.shared_mem.unsafe_ptr()
            var p_bar_start = p_shared.unsafe_offset(0)
            var p_bar_done = p_shared.unsafe_offset(8)

            _ = external_call["pthread_barrier_init", Int32](
                p_bar_start, Int(0), Int32(self.num_threads)
            )
            _ = external_call["pthread_barrier_init", Int32](
                p_bar_done, Int(0), Int32(self.num_threads)
            )

            p_shared[unsafe_offset=16] = 0
            p_shared[unsafe_offset=17] = self.num_threads
            p_shared[unsafe_offset=18] = TASK_NONE
            p_shared[unsafe_offset=19] = 0

            var shared_addr = Int(p_shared)

            for w in range(1, self.num_threads):
                self.worker_args[w * 4 + 0] = w
                self.worker_args[w * 4 + 1] = shared_addr
                var p_arg = self.worker_args.unsafe_ptr().unsafe_offset(w * 4)
                var p_tid = self.thread_ids.unsafe_ptr().unsafe_offset(w)

                _ = external_call["pthread_create", Int32](
                    p_tid, Int(0), _worker_thread_loop, p_arg
                )

            self.is_active = True

    def parallel_dequant_bf16(
        mut self,
        scales_addr: Int,
        packed_addr: Int,
        out_addr: Int,
        num_elements: Int,
        group_size: Int,
    ) raises:
        """Menjalankan dekuantisasi 4-bit ke BF16 secara paralel pada worker pool.
        """
        if self.num_threads <= 1:
            var p_scales = Pointer[Float16, MutAnyOrigin](
                unsafe_from_address=scales_addr
            )
            var p_packed = Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=packed_addr
            )
            var p_out = Pointer[BFloat16, MutAnyOrigin](
                unsafe_from_address=out_addr
            )

            var num_groups = num_elements // group_size
            for g in range(num_groups):
                var s_g = p_scales[unsafe_offset=g]
                var s_val = Float32(s_g)
                var s_simd = SIMD[DType.float32, 8](s_val)

                var g_elem_base = g * group_size
                var bytes_in_group = group_size // 2
                var g_byte_base = g * bytes_in_group

                var b = 0
                while b + 8 <= bytes_in_group:
                    var raw = p_packed.unsafe_load[width=8](g_byte_base + b)
                    var lo_s = (raw << 4).cast[DType.int8]() >> 4
                    var hi_s = raw.cast[DType.int8]() >> 4

                    if lo_s.reduce_min() < -7 or hi_s.reduce_min() < -7:
                        raise Error(
                            "dequant_kernel: reserved nibble 0b1000 (-8)"
                            " encountered"
                        )

                    var lo_out = (s_simd * lo_s.cast[DType.float32]()).cast[
                        DType.bfloat16
                    ]()
                    var hi_out = (s_simd * hi_s.cast[DType.float32]()).cast[
                        DType.bfloat16
                    ]()

                    var out_base = g_elem_base + b * 2
                    for j in range(8):
                        p_out[unsafe_offset=out_base + 2 * j] = lo_out[j]
                        p_out[unsafe_offset=out_base + 2 * j + 1] = hi_out[j]

                    b += 8
            return

        var p_shared = self.shared_mem.unsafe_ptr()
        p_shared[unsafe_offset=18] = TASK_DEQUANT_SIMD
        p_shared[unsafe_offset=19] = 0
        p_shared[unsafe_offset=20] = num_elements
        p_shared[unsafe_offset=21] = group_size
        p_shared[unsafe_offset=22] = scales_addr
        p_shared[unsafe_offset=23] = packed_addr
        p_shared[unsafe_offset=24] = out_addr

        var p_bar_start = p_shared.unsafe_offset(0)
        var p_bar_done = p_shared.unsafe_offset(8)

        # Lepas seluruh worker
        _ = external_call["pthread_barrier_wait", Int32](p_bar_start)

        # Koordinator mengeksekusi partisi worker_id = 0
        var p_shared_mut = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=Int(p_shared)
        )
        _worker_dequant_task(0, self.num_threads, p_shared_mut)

        # Tunggu seluruh worker selesai
        _ = external_call["pthread_barrier_wait", Int32](p_bar_done)

        if p_shared[unsafe_offset=19] != 0:
            raise Error(
                "WorkerPool: dequantization error encountered in worker thread"
            )

    def parallel_moe_experts(
        mut self,
        top_k: Int,
        hidden_dim: Int,
        inter_dim: Int,
        x_tok_addr: Int,
        out_routed_addr: Int,
        selected_experts_addr: Int,
        w_gate_ptrs_addr: Int,
        w_up_ptrs_addr: Int,
        w_down_ptrs_addr: Int,
    ) raises:
        """Menjalankan evaluasi SwiGLU 8 MoE experts secara paralel pada worker pool.
        """
        var p_shared = self.shared_mem.unsafe_ptr()
        p_shared[unsafe_offset=18] = TASK_MOE_EXPERTS
        p_shared[unsafe_offset=19] = 0
        p_shared[unsafe_offset=20] = top_k
        p_shared[unsafe_offset=21] = hidden_dim
        p_shared[unsafe_offset=22] = inter_dim
        p_shared[unsafe_offset=23] = x_tok_addr
        p_shared[unsafe_offset=24] = out_routed_addr
        p_shared[unsafe_offset=25] = selected_experts_addr
        p_shared[unsafe_offset=26] = w_gate_ptrs_addr
        p_shared[unsafe_offset=27] = w_up_ptrs_addr
        p_shared[unsafe_offset=28] = w_down_ptrs_addr
        if len(self.moe_scratch) < self.num_threads * inter_dim:
            self.moe_scratch.resize(self.num_threads * inter_dim, Float32(0.0))
        p_shared[unsafe_offset=29] = Int(self.moe_scratch.unsafe_ptr())

        if self.num_threads <= 1:
            var p_shared_mut = Pointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(p_shared)
            )
            _worker_moe_task(0, 1, p_shared_mut)
            return

        var p_bar_start = p_shared.unsafe_offset(0)
        var p_bar_done = p_shared.unsafe_offset(8)

        # Lepas seluruh worker
        _ = external_call["pthread_barrier_wait", Int32](p_bar_start)

        # Koordinator mengeksekusi partisi worker_id = 0
        var p_shared_mut = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=Int(p_shared)
        )
        _worker_moe_task(0, self.num_threads, p_shared_mut)

        # Tunggu seluruh worker selesai
        _ = external_call["pthread_barrier_wait", Int32](p_bar_done)

        if p_shared[unsafe_offset=19] != 0:
            raise Error(
                "WorkerPool: MoE expert GEMM error encountered in worker thread"
            )

    def shutdown(mut self):
        """Menghentikan seluruh thread pekerja dan melepaskan resource barrier.
        """
        if self.is_active and self.num_threads > 1:
            var p_shared = self.shared_mem.unsafe_ptr()
            p_shared[unsafe_offset=16] = 1  # exit_flag = 1
            var p_bar_start = p_shared.unsafe_offset(0)
            var p_bar_done = p_shared.unsafe_offset(8)

            _ = external_call["pthread_barrier_wait", Int32](p_bar_start)

            for w in range(1, self.num_threads):
                _ = external_call["pthread_join", Int32](
                    self.thread_ids[w], Int(0)
                )

            _ = external_call["pthread_barrier_destroy", Int32](p_bar_start)
            _ = external_call["pthread_barrier_destroy", Int32](p_bar_done)

            self.is_active = False
