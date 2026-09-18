# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Unit tests untuk M11-W3a: Prober Topologi CPU, Mask Alokasi K_alloc, dan Cabang Fallback (§2.2, §3.4).
"""

from core.topology import (
    MODE_ASYNC_DOUBLE_BUFFER,
    MODE_SYNC_FALLBACK,
    OS_RAM_RESERVE_BYTES,
    CoreAllocation,
    CpuInfo,
    CpuTopology,
    build_core_allocation,
    parse_cpu_list,
    probe_cpu_topology,
    probe_ram_available,
    validate_runtime_feasibility,
)
from std.collections import List
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)


def test_parse_cpu_list() raises:
    """Verifikasi fungsi parse_cpu_list untuk format range dan list CPU Linux.
    """
    # 1. Single core
    var l1 = parse_cpu_list("4")
    assert_equal(len(l1), 1)
    assert_equal(l1[0], 4)

    # 2. Simple range
    var l2 = parse_cpu_list("0-3")
    assert_equal(len(l2), 4)
    assert_equal(l2[0], 0)
    assert_equal(l2[1], 1)
    assert_equal(l2[2], 2)
    assert_equal(l2[3], 3)

    # 3. Multiple ranges dan individual IDs
    var l3 = parse_cpu_list("0-1,4-5,7")
    assert_equal(len(l3), 5)
    assert_equal(l3[0], 0)
    assert_equal(l3[1], 1)
    assert_equal(l3[2], 4)
    assert_equal(l3[3], 5)
    assert_equal(l3[4], 7)

    # 4. Trailing newlines / whitespace
    var l4 = parse_cpu_list("0-7\n\r")
    assert_equal(len(l4), 8)
    assert_equal(l4[0], 0)
    assert_equal(l4[7], 7)

    # 5. Empty string
    var l5 = parse_cpu_list("")
    assert_equal(len(l5), 0)


def test_sysfs_topology_live_probe() raises:
    """Verifikasi pembacaan topologi riil Linux host via sysfs."""
    var topo = probe_cpu_topology()
    assert_true(topo.total_online_cpus >= 1)
    assert_true(topo.physical_core_count >= 1)
    assert_true(topo.physical_core_count <= topo.total_online_cpus)
    assert_equal(len(topo.cpus), topo.total_online_cpus)
    assert_equal(len(topo.primary_physical_cpus), topo.physical_core_count)
    assert_equal(
        len(topo.primary_physical_cpus) + len(topo.secondary_smt_cpus),
        topo.total_online_cpus,
    )
    assert_true(topo.source == "sysfs" or topo.source == "sysconf_fallback")


def test_sysconf_fallback_probe() raises:
    """Verifikasi transisi fallback ke sysconf saat sysfs path tidak valid."""
    var topo = probe_cpu_topology("/nonexistent_sysfs_path_mock_test")
    assert_equal(topo.source, "sysconf_fallback")
    assert_true(topo.total_online_cpus >= 1)
    assert_equal(topo.physical_core_count, topo.total_online_cpus)
    assert_equal(len(topo.secondary_smt_cpus), 0)
    assert_equal(len(topo.primary_physical_cpus), topo.total_online_cpus)


def test_core_allocation_async_mode() raises:
    """Verifikasi alokasi K_alloc, isolasi K_io, dan ordering physical-first pada mode async.
    """
    var topo = probe_cpu_topology()
    if topo.total_online_cpus < 2:
        return  # Host 1 core tidak menjalankan tes async

    var c_io = 1
    var alloc = build_core_allocation(topo, c_io=c_io)
    assert_true(alloc.is_async_mode())
    assert_true(alloc.is_async_supported)
    assert_equal(alloc.c_io, c_io)
    assert_equal(len(alloc.k_io), c_io)
    assert_equal(alloc.c_compute_max, len(alloc.k_compute))
    assert_equal(alloc.c_compute_max, len(alloc.k_alloc) - c_io)

    # Invarian K_io terisolasi (K_io cap K_compute == empty)
    for i in range(len(alloc.k_io)):
        var io_cpu = alloc.k_io[i]
        for c in range(len(alloc.k_compute)):
            assert_true(alloc.k_compute[c] != io_cpu)

    # Invarian penjadwalan: core fisik primer dijadwalkan lebih dahulu daripada SMT secondary
    var last_primary_idx = -1
    var first_secondary_idx = -1
    for idx in range(len(alloc.k_compute)):
        var cpu_id = alloc.k_compute[idx]
        var is_sec = False
        for s in range(len(topo.secondary_smt_cpus)):
            if topo.secondary_smt_cpus[s] == cpu_id:
                is_sec = True
                break
        if not is_sec:
            last_primary_idx = idx
        else:
            if first_secondary_idx < 0:
                first_secondary_idx = idx

    if first_secondary_idx >= 0 and last_primary_idx >= 0:
        assert_true(last_primary_idx < first_secondary_idx)


def test_core_allocation_sync_fallback_branch() raises:
    """Verifikasi penegakan prasyarat |K_alloc| >= C_io + 1 dan cabang fallback.
    """
    # 1. Kasus sintesis 1-core topology
    var siblings = List[Int]()
    siblings.append(0)
    var cpus = List[CpuInfo]()
    cpus.append(
        CpuInfo(
            cpu_id=0,
            core_id=0,
            package_id=0,
            is_online=True,
            is_smt_secondary=False,
            smt_siblings=siblings.copy(),
            l2_domain_id=0,
            l3_domain_id=0,
            numa_node_id=0,
        )
    )
    var primary = List[Int]()
    primary.append(0)

    var topo_single = CpuTopology(
        total_online_cpus=1,
        physical_core_count=1,
        numa_nodes_count=1,
        cpus=cpus.copy(),
        primary_physical_cpus=primary.copy(),
        secondary_smt_cpus=List[Int](),
        source="synthetic_single_core",
    )

    # Evaluasi alokasi dengan C_io = 1 (1 core < 1 + 1 -> fallback)
    var alloc_single = build_core_allocation(topo_single, c_io=1)
    assert_false(alloc_single.is_async_supported)
    assert_true(alloc_single.is_fallback_mode())
    assert_equal(alloc_single.c_compute_max, 1)
    assert_equal(alloc_single.c_io, 0)
    assert_equal(len(alloc_single.k_io), 0)
    assert_true(alloc_single.fallback_reason.byte_length() > 0)

    # 2. Kasus requested_mask hanya menyisakan 1 core pada real topology
    var topo = probe_cpu_topology()
    var mask_1 = List[Int]()
    mask_1.append(topo.cpus[0].cpu_id)

    var alloc_mask1 = build_core_allocation(topo, c_io=1, requested_mask=mask_1)
    assert_false(alloc_mask1.is_async_supported)
    assert_true(alloc_mask1.is_fallback_mode())
    assert_equal(alloc_mask1.c_compute_max, 1)


def test_runtime_shrink_fail_fast() raises:
    """Verifikasi bahwa penyusutan core aktif (CPU hot-unplug) memicu FAIL-FAST.
    """
    var topo = probe_cpu_topology()
    if topo.total_online_cpus < 3:
        return

    var alloc = build_core_allocation(topo, c_io=1)

    # 1. Happy path: seluruh CPU online masih lengkap
    var full_online = List[Int]()
    for i in range(len(topo.cpus)):
        full_online.append(topo.cpus[i].cpu_id)

    validate_runtime_feasibility(alloc, full_online, c_system=2)

    # 2. Failure path: CPU hot-unplug menyusutkan core aktif di bawah C_total (mis. c_system=2 + c_io=1 = 3)
    var shrunk_online = List[Int]()
    shrunk_online.append(alloc.k_alloc[0])
    shrunk_online.append(
        alloc.k_alloc[1]
    )  # hanya 2 CPU (kurang dari C_total = 3)

    var caught_error = False
    try:
        validate_runtime_feasibility(alloc, shrunk_online, c_system=2)
    except:
        caught_error = True

    assert_true(caught_error)


def test_ram_available_prober() raises:
    """Verifikasi kalkulasi anggaran RAM aktif M_budget dan M_OS_reserve >= 0.5 GiB.
    """
    var ram = probe_ram_available()
    var tot_bytes = ram[0]
    var avail_bytes = ram[1]
    var safe_budget = ram[2]

    assert_true(tot_bytes > 0)
    assert_true(avail_bytes > 0)
    assert_true(avail_bytes <= tot_bytes)
    assert_true(safe_budget <= avail_bytes)

    if avail_bytes > OS_RAM_RESERVE_BYTES:
        assert_equal(safe_budget, avail_bytes - OS_RAM_RESERVE_BYTES)
    else:
        assert_equal(safe_budget, 0)


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
