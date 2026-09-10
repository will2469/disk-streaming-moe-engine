#!/usr/bin/env python3
"""Generate deterministik 20+ korpus fuzz M0 (G-M0-3).

Tiap kasus: (nama_file, bytes, exit_harapan, error_terkandung).
Kontrol PASS ikut serta (valid, empty-valid, unknown-field) agar runner
mendeteksi over-reject, bukan cuma under-reject.
Jalankan: uv run tools/fixtures/generate_m0_fuzz.py
Runner: tests/integration/test_m0_fuzz.sh (timeout 10s/kasus, anti-hang).
"""

import json
import os
import struct

OUT = os.path.join(os.path.dirname(__file__), "../../fixtures/m0-fuzz")
CASES: list[tuple[str, bytes, int, str]] = []


def blob(hdr: bytes, payload_len: int) -> bytes:
    return struct.pack("<Q", len(hdr)) + hdr + bytes(payload_len)


def good_hdr() -> bytes:
    return json.dumps(
        {
            "a00": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
            "b00": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        },
        separators=(",", ":"),
    ).encode()


def add(name: str, data: bytes, want_exit: int, want_err: str) -> None:
    CASES.append((name, data, want_exit, want_err))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    g = good_hdr()
    # --- kontrol PASS ---
    add("00_valid.st", blob(g, 16), 0, "")
    add(
        "01_empty_valid.st",
        blob(
            json.dumps(
                {
                    "a01": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
                    "e01": {"dtype": "F32", "shape": [0], "data_offsets": [8, 8]},
                },
                separators=(",", ":"),
            ).encode(),
            8,
        ),
        0,
        "",
    )
    add(
        "02_unknown_field_ok.st",
        blob(
            json.dumps(
                {
                    "__metadata__": {"note": "bebas"},
                    "a02": {
                        "dtype": "BF16",
                        "shape": [4],
                        "data_offsets": [0, 8],
                        "future_field": [1, 2],
                    },
                },
                separators=(",", ":"),
            ).encode(),
            8,
        ),
        0,
        "",
    )
    # --- header liar ---
    add("03_empty_file.st", b"", 2, "INVALID_HEADER")
    add("04_short_prefix.st", b"\x05\x00", 2, "INVALID_HEADER")
    add("05_huge_len.st", struct.pack("<Q", 200_000_000) + b"{}", 2, "INVALID_HEADER")
    add("06_not_json.st", blob(b"not json!!", 0), 2, "JSON_PARSE_ERROR")
    add("07_truncated.st", struct.pack("<Q", 100) + b'{"a":', 2, "INVALID_HEADER")
    # --- offset ---
    add(
        "08_begin_gt_end.st",
        blob(
            json.dumps(
                {"t": {"dtype": "BF16", "shape": [4], "data_offsets": [8, 0]}},
                separators=(",", ":"),
            ).encode(),
            16,
        ),
        2,
        "OFFSET_OVERFLOW",
    )
    add(
        "09_negative.st",
        struct.pack("<Q", 60)
        + b'{"t":{"dtype":"BF16","shape":[4],"data_offsets":[-1,8]}}'
        + bytes(16),
        2,
        "JSON_PARSE_ERROR",
    )
    add(
        "10_overflow_filesize.st",
        blob(
            json.dumps(
                {"t": {"dtype": "BF16", "shape": [4000], "data_offsets": [0, 8000]}},
                separators=(",", ":"),
            ).encode(),
            8,
        ),
        2,
        "OFFSET_OVERFLOW",
    )
    add(
        "11_hole.st",
        blob(
            json.dumps(
                {
                    "a": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
                    "b": {"dtype": "BF16", "shape": [4], "data_offsets": [12, 20]},
                },
                separators=(",", ":"),
            ).encode(),
            20,
        ),
        2,
        "OFFSET_OVERFLOW",
    )
    add(
        "12_overlap.st",
        blob(
            json.dumps(
                {
                    "a": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
                    "b": {"dtype": "BF16", "shape": [4], "data_offsets": [4, 12]},
                },
                separators=(",", ":"),
            ).encode(),
            12,
        ),
        2,
        "OFFSET_OVERFLOW",
    )
    # --- dtype/layout ---
    add(
        "13_dtype_asing.st",
        blob(
            json.dumps(
                {"t": {"dtype": "Q8", "shape": [4], "data_offsets": [0, 8]}},
                separators=(",", ":"),
            ).encode(),
            8,
        ),
        2,
        "UNKNOWN_DTYPE",
    )
    add(
        "14_dtype_lower.st",
        blob(
            json.dumps(
                {"t": {"dtype": "bf16", "shape": [4], "data_offsets": [0, 8]}},
                separators=(",", ":"),
            ).encode(),
            8,
        ),
        2,
        "UNKNOWN_DTYPE",
    )
    add(
        "15_layout_mismatch.st",
        blob(
            json.dumps(
                {"t": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 4]}},
                separators=(",", ":"),
            ).encode(),
            8,
        ),
        2,
        "LAYOUT_MISMATCH",
    )
    # --- duplikat/nama ---
    add(
        "16_dup_key.st",
        blob(
            b'{"t":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},'
            b'"t":{"dtype":"BF16","shape":[4],"data_offsets":[8,16]}}',
            16,
        ),
        2,
        "DUPLICATE_JSON_KEY",
    )
    # --- bentuk ---
    add(
        "17_shape_neg.st",
        struct.pack("<Q", 60)
        + b'{"t":{"dtype":"BF16","shape":[-4],"data_offsets":[0,8]}}'
        + bytes(8),
        2,
        "JSON_PARSE_ERROR",
    )
    add(
        "18_missing_field.st",
        blob(
            json.dumps(
                {"t": {"dtype": "BF16", "shape": [4]}}, separators=(",", ":")
            ).encode(),
            8,
        ),
        2,
        "INVALID_HEADER",
    )
    add(
        "19_bad_arity.st",
        blob(
            json.dumps(
                {"t": {"dtype": "BF16", "shape": [4], "data_offsets": [0]}},
                separators=(",", ":"),
            ).encode(),
            8,
        ),
        2,
        "JSON_PARSE_ERROR",
    )
    add("20_garbage.st", bytes(range(256)) * 4, 2, "INVALID_HEADER")
    manifest = []
    for name, data, want_exit, want_err in CASES:
        open(os.path.join(OUT, name), "wb").write(data)
        manifest.append(
            {
                "file": name,
                "want_exit": want_exit,
                "want_err": want_err,
                "size": len(data),
            }
        )
    open(os.path.join(OUT, "manifest.json"), "w").write(json.dumps(manifest, indent=2))
    wm = {
        "a00": "00_valid.st",
        "b00": "00_valid.st",
        "a01": "01_empty_valid.st",
        "e01": "01_empty_valid.st",
        "a02": "02_unknown_field_ok.st",
    }
    open(os.path.join(OUT, "model.safetensors.index.json"), "w").write(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": wm})
    )
    print(f"fuzz ok: {len(CASES)} kasus di {OUT}")


if __name__ == "__main__":
    main()
