#!/bin/bash
# Kontrak I/O M0 (hard gate): seluruh read pada shard HARUS dalam [0, data_base).
# Bukti via strace (lseek+read/pread), bukan klaim kode. read_bytes total observasional.
set -u
DISMOEN="${DISMOEN:-./dismoen}"
FX="fixtures/m0"
LOG=/tmp/dismoen-io-contract.log
strace -f -e trace=openat,lseek,read,pread64 -o "$LOG" \
  "$DISMOEN" check-index "$FX"/fixture-0000*.safetensors >/dev/null 2>&1
code=$?
[ "$code" -eq 0 ] || { echo "check-index exit $code"; exit 1; }
python3 - "$LOG" "$FX" <<'EOF'
import os, re, struct, sys

log, fx = sys.argv[1], sys.argv[2]
# data_base per file absolut
base = {}
for s in sorted(os.listdir(fx)):
    if not s.endswith(".safetensors"):
        continue
    p = os.path.abspath(os.path.join(fx, s))
    n = struct.unpack("<Q", open(p, "rb").read(8))[0]
    base[p] = 8 + n
# petakan fd -> path per pid, lacak offset, verifikasi rentang read
fdmap, pos, bad, total = {}, {}, [], 0
for line in open(log):
    m = re.search(r"(\d+)\s+openat\(AT_FDCWD,\s*\"([^\"]+)\".*?\)\s+=\s+(\d+)\s*$", line)
    if m:
        fdmap[(m.group(1), m.group(3))] = m.group(2)
        continue
    m = re.search(r"(\d+)\s+lseek\((\d+),\s*(-?\d+),\s*(SEEK_\w+)\)\s+=\s+(-?\d+)", line)
    if m:
        pid, fd, off, wh = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        key = (pid, fd)
        if wh == "SEEK_SET":
            pos[key] = off
        elif wh == "SEEK_CUR":
            pos[key] = pos.get(key, 0) + off
        elif wh == "SEEK_END":
            p = fdmap.get(key, "")
            try:
                pos[key] = os.path.getsize(p) + off
            except OSError:
                pass
        continue
    m = re.search(r"(\d+)\s+(pread64|read)\((\d+),.*,\s*(\d+)\)\s+=\s+(-?\d+)", line)
    if m:
        pid, op, fd, _cnt, ret = m.groups()
        ret = int(ret)
        if ret <= 0:
            continue
        p = fdmap.get((pid, fd), "")
        if not p.endswith(".safetensors"):
            continue
        ap = os.path.abspath(p)
        if ap not in base:
            bad.append(f"file shard tak dikenal: {p}")
            continue
        if op == "pread64":
            mm = re.search(r", (\d+), (\d+)\)\s+=", line)
            off = int(mm.group(2)) if mm else None
        else:
            off = pos.get((pid, fd))
        if off is None:
            bad.append(f"offset tak terlacak: {line.strip()}")
            continue
        total += ret
        if not (0 <= off and off + ret <= base[ap]):
            bad.append(f"keluar [{ap}]: [{off},{off + ret}) vs data_base {base[ap]}")
if bad:
    print("KONTRAK I/O GAGAL:")
    print("\n".join(f"  - {b}" for b in bad))
    sys.exit(1)
print(f"kontrak I/O hijau: semua read shard dalam header; total {total} byte (observasional)")
EOF
