#!/bin/bash
# scripts/commit.sh — commit bersemantik lint-staged:
#   hook format -> file terformat di-stage otomatis -> valid -> commit (sekali jalan).
# Gagal murni (bukan format, mis. clippy/test) -> fail + tampilkan log hook.
# Hanya file yang SUDAH staged yang disentuh; worktree lain tidak terseret.
# Pakai: scripts/commit.sh "pesan commit"
set -u
msg="${1:?pakai: scripts/commit.sh \"pesan commit\"}"
staged=$(git diff --cached --name-only)
[ -z "$staged" ] && { echo "tidak ada staged changes (git add dulu)"; exit 1; }
# shellcheck disable=SC2086
for round in 1 2 3; do
    if pre-commit run --files $staged >/tmp/kimo-commit-hook.log 2>&1; then
        git commit -m "$msg"
        exit $?
    fi
    # shellcheck disable=SC2086
    touched=$(git status --porcelain | awk '$1 ~ /M/ {print $2}')
    restaged=""
    # shellcheck disable=SC2086
    for f in $touched; do
        if echo "$staged" | grep -qx "$f"; then
            git add -- "$f"
            restaged="$restaged $f"
        fi
    done
    if [ -z "$restaged" ]; then
        cat /tmp/kimo-commit-hook.log
        exit 1
    fi
    echo "[ronde $round] terformat ulang:$restaged"
done
cat /tmp/kimo-commit-hook.log
echo "gagal setelah 3 ronde"
exit 1
