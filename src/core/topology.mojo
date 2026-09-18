# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""On-Device Hardware Prober & Mask-Based Core Budget Accounting (§2.2, §3.4, M11-W3a).

Mengimplementasikan:
1. Dynamic CPU Topology Prober via Linux sysfs (/sys/devices/system/cpu/):
   - core_id, thread_siblings_list, physical_package_id.
   - cache/index*/shared_cpu_list (domain L2 & L3 cache).
   - /sys/devices/system/node/ (klasifikasi NUMA nodes).
   - Fallback bersih ke sysconf(_SC_NPROCESSORS_ONLN) bila sysfs tidak tersedia.
2. RAM Available Prober via sysconf(_SC_AVPHYS_PAGES) & /proc/meminfo:
   - Menghitung active RAM budget M_budget dengan M_OS_reserve >= 0.5 GiB.
3. Mask-Based Core Allocation (K_alloc, K_io, K_compute):
   - K_io (|K_io| = C_io >= 1) terisolasi dari K_compute.
   - C_compute_max = |K_compute| = |K_alloc| - C_io (TANPA penambalan max(1, ·)).
   - Penjadwalan worker komputasi physical-first sebelum SMT secondary siblings.
4. Penegakan Prasyarat |K_alloc| >= C_io + 1 (§2.2):
   - Jika terpenuhi -> MODE_ASYNC_DOUBLE_BUFFER.
   - Jika tidak terpenuhi -> MODE_SYNC_FALLBACK (c=1, pread blocking, E_overlap = N/A).
5. Deteksi Penyusutan Runtime (CPU Hot-Unplug):
   - Invarian C_total = c*_system + C_io <= |K_alloc| <= C_online.
   - Pelanggaran memicu fail-fast error seketika tanpa kompromi.
"""

from std.collections import List
from std.ffi import external_call
from std.memory import Pointer

# Konstanta Mode Eksekusi Pipeline Engine (§2.2)
comptime MODE_ASYNC_DOUBLE_BUFFER = 0
comptime MODE_SYNC_FALLBACK = 1

# Default Reserve Kernel OS: 0.5 GiB = 536,870,912 B (§3.4)
comptime OS_RAM_RESERVE_BYTES = 536870912


def parse_cpu_list(s: String) -> List[Int]:
    """Mengurai string representasi range CPU Linux menjadi daftar integer terurut.

    Mendukung format:
    - Single ID: "4" -> [4]
    - Range: "0-3" -> [0, 1, 2, 3]
    - Kombinasi koma: "0-1,4-5,7" -> [0, 1, 4, 5, 7]
    - Spasi dan newline diabaikan secara otomatis.
    """
    var result = List[Int]()
    var clean_str = String()
    var s_bytes = s.as_bytes()
    for i in range(len(s_bytes)):
        var c = s_bytes[i]
        if (
            (c >= 48 and c <= 57)  # '0'..'9'
            or c == 44  # ','
            or c == 45  # '-'
        ):
            clean_str += chr(Int(c))

    if clean_str.byte_length() == 0:
        return result.copy()

    var tokens = clean_str.split(",")
    for i in range(len(tokens)):
        var tok = tokens[i]
        if tok.byte_length() == 0:
            continue
        var tok_bytes = tok.as_bytes()
        var dash_pos = -1
        for j in range(len(tok_bytes)):
            if tok_bytes[j] == 45:  # '-'
                dash_pos = j
                break

        if dash_pos >= 0:
            var start_str = String()
            var end_str = String()
            for k in range(dash_pos):
                start_str += chr(Int(tok_bytes[k]))
            for k in range(dash_pos + 1, len(tok_bytes)):
                end_str += chr(Int(tok_bytes[k]))
            try:
                var start_val = Int(start_str)
                var end_val = Int(end_str)
                for cpu_idx in range(start_val, end_val + 1):
                    result.append(cpu_idx)
            except:
                pass
        else:
            try:
                result.append(Int(tok))
            except:
                pass

    return result.copy()


def read_sysfs_string(path: String) raises -> String:
    """Membaca baris teks pertama dari antarmuka Linux sysfs."""
    var path_z = path + "\0"
    var fd = external_call["openat", Int32](
        -100, path_z.unsafe_ptr(), 0, 0  # O_RDONLY
    )
    if fd < 0:
        raise Error("read_sysfs_string: failed to open " + path)

    var buf = external_call["malloc", Int](1024)
    var n = external_call["pread", Int](fd, buf, 1024, 0)
    _ = external_call["close", Int32](fd)

    if n <= 0:
        external_call["free", NoneType](buf)
        return ""

    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=buf)
    var res = String()
    for i in range(n):
        var c = p[unsafe_offset=i]
        if c == 10 or c == 13:  # '\n' atau '\r'
            break
        res += chr(Int(c))

    external_call["free", NoneType](buf)
    return res


@fieldwise_init
struct CpuInfo(Copyable, Movable):
    """Informasi atribut topologi satu CPU logical (§3.4)."""

    var cpu_id: Int
    var core_id: Int
    var package_id: Int
    var is_online: Bool
    var is_smt_secondary: Bool
    var smt_siblings: List[Int]
    var l2_domain_id: Int
    var l3_domain_id: Int
    var numa_node_id: Int


struct CpuTopology(Copyable, Movable):
    """Deskripsi menyeluruh topologi prosesor host hasil probe (§3.4)."""

    var total_online_cpus: Int
    var physical_core_count: Int
    var numa_nodes_count: Int
    var cpus: List[CpuInfo]
    var primary_physical_cpus: List[Int]
    var secondary_smt_cpus: List[Int]
    var source: String

    def __init__(
        out self,
        total_online_cpus: Int,
        physical_core_count: Int,
        numa_nodes_count: Int,
        cpus: List[CpuInfo],
        primary_physical_cpus: List[Int],
        secondary_smt_cpus: List[Int],
        source: String,
    ):
        self.total_online_cpus = total_online_cpus
        self.physical_core_count = physical_core_count
        self.numa_nodes_count = numa_nodes_count
        self.cpus = cpus.copy()
        self.primary_physical_cpus = primary_physical_cpus.copy()
        self.secondary_smt_cpus = secondary_smt_cpus.copy()
        self.source = source


def _probe_sysconf_fallback() -> CpuTopology:
    """Fallback topologi minimal berbasis POSIX sysconf(_SC_NPROCESSORS_ONLN).
    """
    var num_online = external_call["sysconf", Int](
        Int32(84)
    )  # 84 = _SC_NPROCESSORS_ONLN
    if num_online <= 0:
        num_online = 1

    var cpus = List[CpuInfo]()
    var primary = List[Int]()
    for i in range(num_online):
        var siblings = List[Int]()
        siblings.append(i)
        cpus.append(
            CpuInfo(
                cpu_id=i,
                core_id=i,
                package_id=0,
                is_online=True,
                is_smt_secondary=False,
                smt_siblings=siblings.copy(),
                l2_domain_id=0,
                l3_domain_id=0,
                numa_node_id=0,
            )
        )
        primary.append(i)

    return CpuTopology(
        total_online_cpus=num_online,
        physical_core_count=num_online,
        numa_nodes_count=1,
        cpus=cpus,
        primary_physical_cpus=primary,
        secondary_smt_cpus=List[Int](),
        source="sysconf_fallback",
    )


def probe_cpu_topology(
    sysfs_root: String = "/sys/devices/system/cpu",
) -> CpuTopology:
    """Mem-probe topologi CPU Linux via sysfs dengan fallback otomatis (§3.4).

    Membaca:
    - sysfs_root/online
    - sysfs_root/cpu*/topology/{core_id, thread_siblings_list, physical_package_id}
    - sysfs_root/cpu*/cache/index*/shared_cpu_list
    - /sys/devices/system/node/
    """
    var online_str: String
    try:
        online_str = read_sysfs_string(sysfs_root + "/online")
    except:
        return _probe_sysconf_fallback()

    var online_cpus = parse_cpu_list(online_str)
    if len(online_cpus) == 0:
        return _probe_sysconf_fallback()

    var cpus = List[CpuInfo]()
    var seen_cores = List[Int]()
    var primary_physical = List[Int]()
    var secondary_smt = List[Int]()

    for i in range(len(online_cpus)):
        var cpu_id = online_cpus[i]
        var cpu_prefix = sysfs_root + "/cpu" + String(cpu_id)

        var core_id: Int
        var pkg_id: Int
        var siblings = List[Int]()
        var l2_domain = 0
        var l3_domain = 0
        var numa_node = 0

        try:
            core_id = Int(read_sysfs_string(cpu_prefix + "/topology/core_id"))
        except:
            core_id = cpu_id

        try:
            pkg_id = Int(
                read_sysfs_string(cpu_prefix + "/topology/physical_package_id")
            )
        except:
            pkg_id = 0

        try:
            var sib_str = read_sysfs_string(
                cpu_prefix + "/topology/thread_siblings_list"
            )
            siblings = parse_cpu_list(sib_str)
        except:
            siblings.append(cpu_id)

        # L2 Cache domain (biasanya index2)
        try:
            var l2_str = read_sysfs_string(
                cpu_prefix + "/cache/index2/shared_cpu_list"
            )
            var l2_list = parse_cpu_list(l2_str)
            if len(l2_list) > 0:
                l2_domain = l2_list[0]
        except:
            l2_domain = 0

        # L3 Cache domain (biasanya index3)
        try:
            var l3_str = read_sysfs_string(
                cpu_prefix + "/cache/index3/shared_cpu_list"
            )
            var l3_list = parse_cpu_list(l3_str)
            if len(l3_list) > 0:
                l3_domain = l3_list[0]
        except:
            l3_domain = 0

        # Klasifikasi Core Fisik vs SMT Secondary
        var is_already_seen = False
        for c in range(len(seen_cores)):
            if seen_cores[c] == core_id:
                is_already_seen = True
                break

        var is_smt_sec = False
        if not is_already_seen:
            seen_cores.append(core_id)
            primary_physical.append(cpu_id)
        else:
            is_smt_sec = True
            secondary_smt.append(cpu_id)

        cpus.append(
            CpuInfo(
                cpu_id=cpu_id,
                core_id=core_id,
                package_id=pkg_id,
                is_online=True,
                is_smt_secondary=is_smt_sec,
                smt_siblings=siblings.copy(),
                l2_domain_id=l2_domain,
                l3_domain_id=l3_domain,
                numa_node_id=numa_node,
            )
        )

    # Deteksi jumlah node NUMA aktif
    var numa_count = 1
    try:
        var numa_str = read_sysfs_string("/sys/devices/system/node/online")
        var numa_nodes = parse_cpu_list(numa_str)
        if len(numa_nodes) > 0:
            numa_count = len(numa_nodes)
    except:
        numa_count = 1

    return CpuTopology(
        total_online_cpus=len(online_cpus),
        physical_core_count=len(seen_cores),
        numa_nodes_count=numa_count,
        cpus=cpus,
        primary_physical_cpus=primary_physical,
        secondary_smt_cpus=secondary_smt,
        source="sysfs",
    )


@fieldwise_init
struct CoreAllocation(Copyable, Movable):
    """Hasil partisi mask alokasi sistem dismoen (§2.2)."""

    var k_alloc: List[Int]
    var k_io: List[Int]
    var k_compute: List[Int]
    var c_compute_max: Int
    var c_io: Int
    var mode: Int
    var is_async_supported: Bool
    var fallback_reason: String

    def is_async_mode(self) -> Bool:
        """Memeriksa apakah alokasi beroperasi dalam mode async double-buffering.
        """
        return self.mode == MODE_ASYNC_DOUBLE_BUFFER

    def is_fallback_mode(self) -> Bool:
        """Memeriksa apakah alokasi fallback ke mode synchronous single-thread.
        """
        return self.mode == MODE_SYNC_FALLBACK


def build_core_allocation(
    topo: CpuTopology,
    c_io: Int = 1,
    requested_mask: List[Int] = List[Int](),
) -> CoreAllocation:
    """Membangun partisi mask alokasi (K_alloc, K_io, K_compute) (§2.2, §3.4).

    Invarian:
    1. K_alloc subset dari CPU online host.
    2. |K_alloc| >= C_io + 1 adalah syarat wajib mode async.
       Bila gagal: mode async UNSUPPORTED -> fallback ke synchronous I/O.
    3. C_compute_max = |K_compute| = |K_alloc| - C_io (TANPA max(1, ·)).
    4. Worker komputasi dipetakan physical-first sebelum SMT secondary.
    """
    var target_io = c_io if c_io >= 1 else 1

    var k_alloc = List[Int]()
    if len(requested_mask) > 0:
        for i in range(len(requested_mask)):
            var req_id = requested_mask[i]
            # Validasi apakah req_id memang online pada topologi
            var is_online = False
            for j in range(len(topo.cpus)):
                if topo.cpus[j].cpu_id == req_id:
                    is_online = True
                    break
            if is_online:
                k_alloc.append(req_id)
    else:
        for i in range(len(topo.cpus)):
            k_alloc.append(topo.cpus[i].cpu_id)

    # -------------------------------------------------------------
    # Evaluasi Prasyarat Mode Async (|K_alloc| >= C_io + 1) (§2.2)
    # -------------------------------------------------------------
    if len(k_alloc) < target_io + 1:
        # Prasyarat tak terpenuhi: fallback ke single-thread synchronous mode
        var fb_reason = (
            "Insufficient logical cores: |K_alloc|="
            + String(len(k_alloc))
            + " < C_io + 1 ("
            + String(target_io + 1)
            + "). Host fallback to synchronous single-thread I/O."
        )
        var k_comp_fb = List[Int]()
        if len(k_alloc) > 0:
            k_comp_fb.append(k_alloc[0])
        else:
            k_comp_fb.append(0)

        return CoreAllocation(
            k_alloc=k_alloc.copy(),
            k_io=List[Int](),
            k_compute=k_comp_fb.copy(),
            c_compute_max=1,
            c_io=0,
            mode=MODE_SYNC_FALLBACK,
            is_async_supported=False,
            fallback_reason=fb_reason,
        )

    # -------------------------------------------------------------
    # Pemilihan K_io Terisolasi & K_compute Physical-First (§3.4)
    # -------------------------------------------------------------
    var k_io = List[Int]()
    var k_compute = List[Int]()

    # Memilih core I/O: utamakan SMT secondary terlebih dahulu jika ada
    var io_candidates = List[Int]()
    for s in range(len(topo.secondary_smt_cpus)):
        var cpu_s = topo.secondary_smt_cpus[s]
        for a in range(len(k_alloc)):
            if k_alloc[a] == cpu_s:
                io_candidates.append(cpu_s)
                break

    # Jika kandidat SMT tidak mencukupi target_io, ambil dari CPU terakhir di k_alloc
    for a in range(len(k_alloc) - 1, -1, -1):
        var cand = k_alloc[a]
        var already_io = False
        for u in range(len(io_candidates)):
            if io_candidates[u] == cand:
                already_io = True
                break
        if not already_io:
            io_candidates.append(cand)

    for idx in range(target_io):
        k_io.append(io_candidates[idx])

    # Bangun K_compute dengan prinsip physical-first
    # Tahap 1: seluruh physical primary cores yang ada di k_alloc dan bukan K_io
    for p in range(len(topo.primary_physical_cpus)):
        var cpu_p = topo.primary_physical_cpus[p]
        var in_alloc = False
        for a in range(len(k_alloc)):
            if k_alloc[a] == cpu_p:
                in_alloc = True
                break
        var in_io = False
        for u in range(len(k_io)):
            if k_io[u] == cpu_p:
                in_io = True
                break
        if in_alloc and not in_io:
            k_compute.append(cpu_p)

    # Tahap 2: SMT secondary siblings yang ada di k_alloc dan bukan K_io
    for s in range(len(topo.secondary_smt_cpus)):
        var cpu_s = topo.secondary_smt_cpus[s]
        var in_alloc = False
        for a in range(len(k_alloc)):
            if k_alloc[a] == cpu_s:
                in_alloc = True
                break
        var in_io = False
        for u in range(len(k_io)):
            if k_io[u] == cpu_s:
                in_io = True
                break
        if in_alloc and not in_io:
            k_compute.append(cpu_s)

    # Invarian Mengikat (P1-2): C_compute_max = |K_compute| = |K_alloc| - C_io (TANPA max(1, ·))
    var c_comp_max = len(k_compute)

    return CoreAllocation(
        k_alloc=k_alloc.copy(),
        k_io=k_io.copy(),
        k_compute=k_compute.copy(),
        c_compute_max=c_comp_max,
        c_io=len(k_io),
        mode=MODE_ASYNC_DOUBLE_BUFFER,
        is_async_supported=True,
        fallback_reason="",
    )


def validate_runtime_feasibility(
    alloc: CoreAllocation,
    current_online_cpus: List[Int],
    c_system: Int,
) raises:
    """Menegakkan Invarian Kelayakan Alokasi saat runtime (§2.2).

    C_total = c*_system + C_io <= |K_alloc| <= C_online.
    Bila terjadi runtime shrink (mis. CPU hot-unplug), sistem wajib FAIL-FAST
    tanpa penambalan max(1, ·)!
    """
    if alloc.is_fallback_mode():
        # Mode fallback tidak terikat syarat alokasi multi-core async
        return

    var c_total = c_system + alloc.c_io

    # 1. Periksa apakah seluruh CPU di k_alloc masih online saat ini
    var current_k_alloc_online = 0
    for a in range(len(alloc.k_alloc)):
        var allocated_cpu = alloc.k_alloc[a]
        var still_online = False
        for o in range(len(current_online_cpus)):
            if current_online_cpus[o] == allocated_cpu:
                still_online = True
                break
        if still_online:
            current_k_alloc_online += 1

    # 2. Invarian: C_total <= current_k_alloc_online
    if current_k_alloc_online < c_total:
        raise Error(
            "CoreAllocation Invariant Violation (Runtime Shrink / Hot-Unplug):"
            " active allocated CPUs ("
            + String(current_k_alloc_online)
            + ") dropped below required C_total = c*_system ("
            + String(c_system)
            + ") + C_io ("
            + String(alloc.c_io)
            + ") = "
            + String(c_total)
            + "! Refusing execution (FAIL-FAST)."
        )


def probe_ram_available() -> Tuple[Int, Int, Int]:
    """Mengukur kapasitas RAM dan anggaran aman M_budget (§3.4).

    Mengembalikan tuple (total_ram_bytes, available_ram_bytes, safe_budget_bytes).
    safe_budget_bytes = available_ram_bytes - M_OS_reserve (dengan M_OS_reserve >= 0.5 GiB).
    """
    var page_size = external_call["sysconf", Int](Int32(30))  # _SC_PAGESIZE
    var av_pages = external_call["sysconf", Int](Int32(86))  # _SC_AVPHYS_PAGES
    var tot_pages = external_call["sysconf", Int](Int32(85))  # _SC_PHYS_PAGES

    if page_size <= 0:
        page_size = 4096

    var total_bytes = tot_pages * page_size
    var avail_bytes = av_pages * page_size

    var safe_budget = avail_bytes - OS_RAM_RESERVE_BYTES
    if safe_budget < 0:
        safe_budget = 0

    return (total_bytes, avail_bytes, safe_budget)


def read_hardware_lock_c_star(
    lock_path: String = "dismoen.hardware.lock",
) -> Int:
    """Membaca c*_system dari profile aktif dismoen.hardware.lock dengan dynamic probe fallback (§2.4, §3.4).

    Bila file lock tidak ditemukan atau gagal diurai, prober topologi lokal
    akan dipanggil secara dinamis untuk mengembalikan alokasi aman c_compute_max.
    """
    var path_z = lock_path + "\0"
    var fd = external_call["openat", Int32](
        -100, path_z.unsafe_ptr(), 0, 0  # O_RDONLY
    )
    if fd >= 0:
        var max_read = 16384
        var buf = external_call["malloc", Int](max_read)
        var n = external_call["pread", Int](fd, buf, max_read, 0)
        _ = external_call["close", Int32](fd)

        if n > 0:
            var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=buf)
            # Cari substring "c_star_system":
            # "c_star_system": -> len = 16
            var found_idx = -1
            for i in range(n - 16):
                if (
                    p[unsafe_offset=i] == 99  # c
                    and p[unsafe_offset=i + 1] == 95  # _
                    and p[unsafe_offset=i + 2] == 115  # s
                    and p[unsafe_offset=i + 3] == 116  # t
                    and p[unsafe_offset=i + 4] == 97  # a
                    and p[unsafe_offset=i + 5] == 114  # r
                    and p[unsafe_offset=i + 6] == 95  # _
                    and p[unsafe_offset=i + 7] == 115  # s
                    and p[unsafe_offset=i + 8] == 121  # y
                    and p[unsafe_offset=i + 9] == 115  # s
                    and p[unsafe_offset=i + 10] == 116  # t
                    and p[unsafe_offset=i + 11] == 101  # e
                    and p[unsafe_offset=i + 12] == 109  # m
                    and p[unsafe_offset=i + 13] == 34  # "
                    and p[unsafe_offset=i + 14] == 58  # :
                ):
                    found_idx = i + 15
                    break

            if found_idx >= 0:
                var val = 0
                var parsing_digit = False
                for i in range(found_idx, n):
                    var c = p[unsafe_offset=i]
                    if c >= 48 and c <= 57:
                        parsing_digit = True
                        val = val * 10 + Int(c - 48)
                    elif parsing_digit:
                        break

                external_call["free", NoneType](buf)
                if val >= 1:
                    return val
            else:
                external_call["free", NoneType](buf)
        else:
            external_call["free", NoneType](buf)

    # Dynamic Fallback: probe topologi lokal langsung
    var topo = probe_cpu_topology()
    var alloc = build_core_allocation(topo, c_io=1)
    return alloc.c_compute_max
