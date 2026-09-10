#!/bin/bash
# G-M0-3: 20+ korpus fuzz -> clean error (exit 2) atau PASS kontrol (exit 0).
# Timeout 10 dtk/kasus (anti-hang). OOM tak terpicu pada ukuran korpus ini (catat).
set -u
KIMO="${KIMO:-./kimo}"
DIR="fixtures/m0-fuzz"
pass=0; fail=0
for f in "$DIR"/*.st; do
  base=$(basename "$f")
  want_exit=$(python3 -c "import json;[print(c['want_exit']) for c in json.load(open('$DIR/manifest.json')) if c['file']=='$base']")
  want_err=$(python3 -c "import json;[print(c['want_err']) for c in json.load(open('$DIR/manifest.json')) if c['file']=='$base']")
  out=$(timeout 10 "$KIMO" check-index "$f" 2>&1)
  code=$?
  if [ "$code" -eq 124 ]; then echo "HANG: $base"; fail=$((fail+1)); continue; fi
  if [ "$code" -ne "$want_exit" ]; then echo "EXIT: $base dapat $code harap $want_exit :: $out"; fail=$((fail+1)); continue; fi
  if [ -n "$want_err" ] && ! echo "$out" | grep -q "$want_err"; then echo "ERRTYPE: $base harap $want_err :: $out"; fail=$((fail+1)); continue; fi
  pass=$((pass+1))
done
echo "fuzz: $pass lolos, $fail gagal"
[ "$fail" -eq 0 ]
