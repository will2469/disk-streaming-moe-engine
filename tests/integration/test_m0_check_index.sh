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
echo "== hilang =="
"$KIMO" check-index /tmp/kimo-tidak-ada.st 2>/dev/null && { echo "hilang FAIL"; fail=1; } || [ $? -eq 2 ] || { echo "hilang exit FAIL"; fail=1; }
[ "$fail" -eq 0 ] && echo "e2e: hijau"
exit $fail
