# Cakupan cabang parser M0 (audit manual — toolchain Mojo 1.0 belum ada coverage)

37 titik `raise Error` di `src/safetensors.mojo` dipetakan ke test.
Target §5.3: utility parser ≥ 85% (32/37). Status: 37/37 punya test.

| Kode (jumlah situs) | Test |
|---|---|
| INVALID_HEADER (9): prefix pendek, len raksasa, data_base > filesize, header terpotong, objek putus (3), integer overflow, dimensi negatif, numel overflow, tensor > 100rb | test_header_too_big, test_truncated_header, test_missing_field + fuzz 03/04/05/07 |
| JSON_PARSE_ERROR (19): tunas string/array/objek, escape, literal, koma/kurung, nilai liar, pecahan/eksponen, surrogate (3), kontrol mentah, tutup silang | test_not_json, test_bad_arity, test_control_rejected, test_lone_surrogate, test_cross_nesting + fuzz 06/09/17/19 + prop-invalid |
| OFFSET_OVERFLOW (4): BEGIN>END, END > filesize-base, lubang/overlap, buffer tak penuh | test_begin_gt_end, test_hole, test_overflow_beyond_filesize + prop-invalid |
| UNKNOWN_DTYPE (1) | test_unknown_dtype + fuzz 13/14 |
| LAYOUT_MISMATCH (1) | test_layout_mismatch + fuzz 15 |
| DUPLICATE_JSON_KEY (1) | test_duplicate_json_key + fuzz 16 |
| FILE_NOT_FOUND (1) | test_file_not_found |
| Jalur terima: valid multi-tensor, tensor kosong, field tak dikenal, surrogate pair, 40 header acak | test_valid_two_tensors, test_empty_accepted, fuzz 02, test_surrogate_pair, test_prop_random_valid |

DUPLICATE_TENSOR_NAME (merge) milik W2 — teruji di e2e (`sX` + fixture), bukan unit parser.
