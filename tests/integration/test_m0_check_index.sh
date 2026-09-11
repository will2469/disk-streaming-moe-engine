#!/bin/bash
# E2E check-index: happy (0), subset, hilang (2), duplikat (1), salah-tempat (1).
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
echo "== duplikat (nama sama di 2 file -> exit 1, DUPLICATE_TENSOR_NAME) =="
DUP=/tmp/kimo-iso-dup
rm -rf "$DUP" && mkdir -p "$DUP"
cp "$FX/model.safetensors.index.json" "$FX/fixture-00001-of-00003.safetensors" "$DUP/"
cp "$FX/fixture-00001-of-00003.safetensors" "$DUP/fixture-00002-of-00003.safetensors"
"$KIMO" check-index "$DUP"/fixture-0000*.safetensors >/tmp/kimo-dup-out.txt 2>&1 && { echo "duplikat exit FAIL"; fail=1; } || [ $? -eq 1 ] || { echo "duplikat exit FAIL"; fail=1; }
grep -q DUPLICATE_TENSOR_NAME /tmp/kimo-dup-out.txt || { echo "duplikat tipe FAIL"; fail=1; }
rm -rf "$DUP" /tmp/kimo-dup-out.txt
echo "== salah-tempat (isi shard 2 bernama shard 1 -> exit 1, WRONG_SHARD) =="
WRG=/tmp/kimo-iso-wrong
rm -rf "$WRG" && mkdir -p "$WRG"
cp "$FX/model.safetensors.index.json" "$WRG/"
cp "$FX/fixture-00002-of-00003.safetensors" "$WRG/fixture-00001-of-00003.safetensors"
"$KIMO" check-index "$WRG"/fixture-0000*.safetensors >/tmp/kimo-wrong-out.txt 2>&1 && { echo "salah-tempat exit FAIL"; fail=1; } || [ $? -eq 1 ] || { echo "salah-tempat exit FAIL"; fail=1; }
grep -q WRONG_SHARD /tmp/kimo-wrong-out.txt || { echo "salah-tempat tipe FAIL"; fail=1; }
rm -rf "$WRG" /tmp/kimo-wrong-out.txt
[ "$fail" -eq 0 ] && echo "e2e: hijau"
exit $fail
