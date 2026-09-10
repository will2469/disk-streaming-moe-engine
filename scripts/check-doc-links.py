#!/usr/bin/env python3
"""Cek tautan relatif antar-dokumen di docs/ (stdlib only, tanpa network).

Aturan (kontrak dokumen hidup §5.3):
- Setiap `[teks](target)` relatif harus menunjuk file yang ada.
- Anchor `#...` diabaikan (tidak divalidasi).
- Link http(s) dan `mailto:` dilewati (bukan ranah checker ini).

Keluar 0 bila bersih, 1 + daftar rusak bila tidak.
"""

import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> int:
    broken: list[str] = []
    files = sorted(DOCS.rglob("*.md"))
    if not files:
        print("tidak ada .md di docs/")
        return 1
    for md in files:
        for lineno, line in enumerate(md.read_text().splitlines(), 1):
            for m in LINK_RE.finditer(line):
                target = m.group(1).split("#", 1)[0].strip()
                if not target or target.startswith(SKIP_PREFIXES):
                    continue
                if (md.parent / target).exists():
                    continue
                broken.append(f"{md.relative_to(DOCS)}:{lineno}: {m.group(1)}")
    if broken:
        print(f"{len(broken)} tautan rusak:")
        print("\n".join(f"  - {b}" for b in broken))
        return 1
    print(f"bersih: {len(files)} file .md, semua tautan relatif valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
