# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Implementasi subperintah tune CLI dismoen (§2.2, §2.4, §3.4, M11-W3b).

Menjalankan:
1. Dynamic Hardware Prober via Linux sysfs (CPU cores, SMT siblings, RAM available).
2. Partisi Mask Core & Plafon Komputasi K_alloc, K_io, K_compute (P1-2, tanpa max(1, ·)).
3. Eksekusi kalibrasi core scaling & sintesis profil hardware tri-pilar.
4. Emisi artefak dismoen.hardware.lock 10-field.
"""

from core.topology import (
    CoreAllocation,
    CpuTopology,
    OS_RAM_RESERVE_BYTES,
    build_core_allocation,
    probe_cpu_topology,
    probe_ram_available,
    read_hardware_lock_c_star,
)
from cli.errors import fail
from std.collections import List
from std.ffi import external_call
from std.sys.terminate import exit


def _c_system(cmd: String) -> Int:
    """Mengeksekusi perintah shell via POSIX system()."""
    var b = cmd.as_bytes()
    var z = List[UInt8]()
    for i in range(len(b)):
        z.append(b[i])
    z.append(0)
    var ret = external_call["system", Int32](z.unsafe_ptr())
    return Int(ret)


def cmd_tune(args: List[String]) raises:
    """Handler subperintah dismoen tune."""
    var output_lock = "dismoen.hardware.lock"
    var dry_run = False
    var emit_json = False

    var i = 2
    while i < len(args):
        var a = args[i]
        if a == "--help" or a == "-h":
            print("Penggunaan: dismoen tune [options]")
            print(
                "Mem-probe hardware host dan mengkalibrasi profil core scaling"
                " optimal."
            )
            print("Options:")
            print(
                "  --output <path>    Path output dismoen.hardware.lock"
                " (default: dismoen.hardware.lock)"
            )
            print(
                "  --dry-run          Tampilkan probing topologi tanpa"
                " mengeksekusi benchmark"
            )
            print("  --json             Tampilkan hasil dalam format JSON")
            print("  --help, -h         Tampilkan bantuan ini")
            return
        elif a == "--output":
            if i + 1 >= len(args):
                fail("USAGE", "missing argument for --output", "", "")
            output_lock = args[i + 1]
            i += 2
        elif a == "--dry-run":
            dry_run = True
            i += 1
        elif a == "--json":
            emit_json = True
            i += 1
        else:
            fail("USAGE", "unknown option for tune: " + a, "", "")

    # 1. Probing Topologi CPU & Memori
    var topo = probe_cpu_topology()
    var alloc = build_core_allocation(topo, c_io=1)
    var ram = probe_ram_available()

    var total_mem_gib = Float64(ram[0]) / (1024.0 * 1024.0 * 1024.0)
    var avail_mem_gib = Float64(ram[1]) / (1024.0 * 1024.0 * 1024.0)
    var safe_budget_gib = Float64(ram[2]) / (1024.0 * 1024.0 * 1024.0)

    if dry_run:
        if emit_json:
            print("{")
            print('  "mode": "dry-run",')
            print(
                '  "total_online_cpus": ' + String(topo.total_online_cpus) + ","
            )
            print(
                '  "physical_cores": ' + String(topo.physical_core_count) + ","
            )
            print('  "c_io": ' + String(alloc.c_io) + ",")
            print('  "c_compute_max": ' + String(alloc.c_compute_max) + ",")
            print(
                '  "is_async_supported": '
                + ("true" if alloc.is_async_supported else "false")
                + ","
            )
            print('  "total_mem_gib": ' + String(total_mem_gib) + ",")
            print('  "avail_mem_gib": ' + String(avail_mem_gib) + ",")
            print('  "safe_budget_gib": ' + String(safe_budget_gib))
            print("}")
        else:
            print(
                "======================================================================"
            )
            print("DISMOEN Hardware Prober (Dry-Run)")
            print(
                "======================================================================"
            )
            print(
                "CPU Online:       "
                + String(topo.total_online_cpus)
                + " logical CPUs"
            )
            print(
                "Core Fisik:       "
                + String(topo.physical_core_count)
                + " physical cores"
            )
            print(
                "SMT Siblings:     "
                + String(len(topo.secondary_smt_cpus))
                + " secondary SMT"
            )
            print(
                "Alokasi I/O:      C_io = "
                + String(alloc.c_io)
                + " (K_io terisolasi)"
            )
            print(
                "Plafon Komputasi: C_compute_max = "
                + String(alloc.c_compute_max)
            )
            print(
                "Mode Pipeline:    "
                + (
                    "async_double_buffer" if alloc.is_async_mode() else "sync_fallback"
                )
            )
            print("RAM Total:        " + String(total_mem_gib) + " GiB")
            print("RAM Available:    " + String(avail_mem_gib) + " GiB")
            print(
                "Safe RAM Budget:  "
                + String(safe_budget_gib)
                + " GiB (M_OS_reserve = 512 MiB)"
            )
            print(
                "======================================================================"
            )
        return

    # 2. Eksekusi Driver Kalibrasi Python
    var cmd = (
        "python3 tools/bench/bench_core_scaling.py --output-lock '"
        + output_lock
        + "'"
    )
    var ret = _c_system(cmd)
    var exit_code = (ret >> 8) & 255
    if exit_code != 0:
        fail(
            "TUNE_ERROR",
            "gagal menjalankan kalibrasi bench_core_scaling.py",
            "",
            "",
        )
        exit(exit_code)
