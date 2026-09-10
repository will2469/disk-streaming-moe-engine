#!/bin/bash
# E2E check-index: happy fixture (0), salah-tempat (1), hilang (2), subset.
set -u
KIMO="${KIMO:-./kimo}"
FX="fixtures/m0"
fail=0
echo "== happy =="
"$KIMO" check-index "$FX"/fixture-0000*.safetensors >/dev/null 2>&1 || { echo "happy FAIL"; fail=1; }
echo "== subset =="
"$KIMO" check-index "$FX/fixture-00001-of-00003.safetensors" 2>/dev/null | grep -q '"scope":"subset"' || { echo "subset FAIL"; fail=1; }
echo "== hilang (terisolasi: index ada, shard tak ada) =="
ISO=/tmp/kimo-iso-missing
rm -rf "$ISO" && mkdir -p "$ISO"
cp "$FX/fixture-00001-of-00003.safetensors" "$FX/model.safetensors.index.json" "$ISO/"
"$KIMO" check-index "$ISO/nope.st" 2>/tmp/kimo-iso-err.txt && { echo "hilang FAIL"; fail=1; } || [ $? -eq 2 ] || { echo "hilang exit FAIL"; fail=1; }
grep -q FILE_NOT_FOUND /tmp/kimo-iso-err.txt || { echo "hilang tipe FAIL"; fail=1; }
rm -rf "$ISO" /tmp/kimo-iso-err.txt
[ "$fail" -eq 0 ] && echo "e2e: hijau"
exit $fail
