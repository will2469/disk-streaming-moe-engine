#!/bin/bash
# E2E check-index: happy (0), subset, hilang (2), duplikat (1), salah-tempat (1),
# impostor cross-directory (1), quote-filename JSON validity.
set -u
DISMOEN="${DISMOEN:-./dismoen}"
FX="fixtures/m0"
fail=0
echo "== happy =="
"$DISMOEN" check-index "$FX"/fixture-0000*.safetensors >/dev/null 2>&1 || { echo "happy FAIL"; fail=1; }
echo "== subset =="
"$DISMOEN" check-index "$FX/fixture-00001-of-00003.safetensors" 2>/dev/null | grep -q '"scope":"subset"' || { echo "subset FAIL"; fail=1; }
echo "== hilang (terisolasi: index ada, shard tak ada) =="
ISO=/tmp/dismoen-iso-missing
rm -rf "$ISO" && mkdir -p "$ISO"
cp "$FX/fixture-00001-of-00003.safetensors" "$FX/model.safetensors.index.json" "$ISO/"
"$DISMOEN" check-index "$ISO/nope.st" 2>/tmp/dismoen-iso-err.txt && { echo "hilang FAIL"; fail=1; } || [ $? -eq 2 ] || { echo "hilang exit FAIL"; fail=1; }
grep -q FILE_NOT_FOUND /tmp/dismoen-iso-err.txt || { echo "hilang tipe FAIL"; fail=1; }
rm -rf "$ISO" /tmp/dismoen-iso-err.txt
echo "== duplikat (nama sama di 2 file -> exit 1, DUPLICATE_TENSOR_NAME) =="
DUP=/tmp/dismoen-iso-dup
rm -rf "$DUP" && mkdir -p "$DUP"
cp "$FX/model.safetensors.index.json" "$FX/fixture-00001-of-00003.safetensors" "$DUP/"
cp "$FX/fixture-00001-of-00003.safetensors" "$DUP/fixture-00002-of-00003.safetensors"
"$DISMOEN" check-index "$DUP"/fixture-0000*.safetensors >/tmp/dismoen-dup-out.txt 2>&1 && { echo "duplikat exit FAIL"; fail=1; } || [ $? -eq 1 ] || { echo "duplikat exit FAIL"; fail=1; }
grep -q DUPLICATE_TENSOR_NAME /tmp/dismoen-dup-out.txt || { echo "duplikat tipe FAIL"; fail=1; }
rm -rf "$DUP" /tmp/dismoen-dup-out.txt
echo "== salah-tempat (isi shard 2 bernama shard 1 -> exit 1, WRONG_SHARD) =="
WRG=/tmp/dismoen-iso-wrong
rm -rf "$WRG" && mkdir -p "$WRG"
cp "$FX/model.safetensors.index.json" "$WRG/"
cp "$FX/fixture-00002-of-00003.safetensors" "$WRG/fixture-00001-of-00003.safetensors"
"$DISMOEN" check-index "$WRG"/fixture-0000*.safetensors >/tmp/dismoen-wrong-out.txt 2>&1 && { echo "salah-tempat exit FAIL"; fail=1; } || [ $? -eq 1 ] || { echo "salah-tempat exit FAIL"; fail=1; }
grep -q WRONG_SHARD /tmp/dismoen-wrong-out.txt || { echo "salah-tempat tipe FAIL"; fail=1; }
rm -rf "$WRG" /tmp/dismoen-wrong-out.txt
echo "== impostor cross-directory (basename sama, payload beda -> exit 1, WRONG_SHARD) =="
IMP=/tmp/dismoen-iso-impostor
rm -rf "$IMP" && mkdir -p "$IMP/model" "$IMP/other"
cp "$FX"/fixture-0000*.safetensors "$FX/model.safetensors.index.json" "$IMP/model/"
python3 -c "
import struct
p = '$IMP/model/fixture-00002-of-00003.safetensors'
data = open(p, 'rb').read()
n = struct.unpack('<Q', data[:8])[0]
hdr, payload = data[8:8+n], bytearray(data[8+n:])
for i in range(0, len(payload), 64):
    payload[i] ^= 0xFF
open('$IMP/other/fixture-00002-of-00003.safetensors', 'wb').write(data[:8] + hdr + bytes(payload))
"
"$DISMOEN" check-index "$IMP/model/fixture-00001-of-00003.safetensors" "$IMP/other/fixture-00002-of-00003.safetensors" "$IMP/model/fixture-00003-of-00003.safetensors" >/tmp/dismoen-impostor-out.txt 2>&1 && { echo "impostor exit FAIL"; fail=1; } || [ $? -eq 1 ] || { echo "impostor exit FAIL"; fail=1; }
grep -q WRONG_SHARD /tmp/dismoen-impostor-out.txt || { echo "impostor tipe FAIL"; fail=1; }
grep -q '"scope":"subset"' /tmp/dismoen-impostor-out.txt || { echo "impostor scope FAIL"; fail=1; }
rm -rf "$IMP" /tmp/dismoen-impostor-out.txt
echo "== quote-filename (output tetap JSON valid) =="
QTE=/tmp/dismoen-iso-quote
rm -rf "$QTE" && mkdir -p "$QTE"
cp "$FX/model.safetensors.index.json" "$QTE/"
cp "$FX/fixture-00001-of-00003.safetensors" "$QTE/evil\".safetensors"
"$DISMOEN" check-index "$QTE/evil\".safetensors" >/tmp/dismoen-quote-out.txt 2>&1
python3 -c "
import json
d = json.load(open('/tmp/dismoen-quote-out.txt'))
assert d['supplied_shards'] == ['evil\".safetensors'], d['supplied_shards']
" || { echo "quote JSON FAIL"; fail=1; }
rm -rf "$QTE" /tmp/dismoen-quote-out.txt
[ "$fail" -eq 0 ] && echo "e2e: hijau"
exit $fail
