# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Handler subcommand check-index (M0)."""

from cli.errors import basename, dirname, eprint_json, fail
from cli.sys_utils import c_realpath
from format.index import parse_index
from format.reader import read_header
from format.types import STHeader, json_escape
from std.collections import Dict, List
from std.sys.terminate import exit
from std.time import perf_counter_ns


def cmd_check_index(shards: List[String]) raises:
    var t0 = perf_counter_ns()
    if len(shards) < 1:
        fail("USAGE", "butuh N≥1 path shard", "", "")
    var d = dirname(shards[0])
    var index_path = String(d, "/model.safetensors.index.json")
    if d == "":
        index_path = String("model.safetensors.index.json")
    var packed = List[String]()
    try:
        packed = parse_index(index_path)
    except e:
        eprint_json(String(e))
        exit(2)
    # Identitas file = canonical path, bukan basename: satu-satunya
    # canonicalize model root; semua pembanding di bawah memakai ini.
    var model_root = c_realpath(d if d != "" else ".")
    if model_root == "":
        fail(
            "FILE_NOT_FOUND",
            "cannot resolve model directory: " + (d if d != "" else "."),
            "",
            "",
        )
    var nn = 0
    # packed[0] = count as String -> parse manual
    var cs = packed[0].as_bytes()
    for i in range(len(cs)):
        nn = nn * 10 + (Int(cs[i]) - 48)
    var wnames = List[String]()
    var wfiles = List[String]()
    var wpos = Dict[String, Int]()
    for i in range(nn):
        wnames.append(packed[1 + 2 * i])
        wfiles.append(packed[1 + 2 * i + 1])
        wpos[wnames[i]] = i
    # baca header semua shard + verifikasi identitas pasca-baca (TOCTOU):
    # entry direktori bisa berganti antara resolve dan read; bila identitas
    # berubah di tengah baca, baca ulang terbatas lalu atribut ke identitas
    # terkini. Gagal konvergen → error (tidak pernah mengaudit file salah
    # secara diam-diam). Residual: pergantian ISI byte (inode sama) tak
    # terdeteksi via path — butuh fd+digest (di luar threat model CLI).
    var headers = List[STHeader]()
    var supplied_canon = List[String]()
    for i in range(len(shards)):
        var attempt = 0
        while True:
            if attempt >= 4:
                fail(
                    "FILE_NOT_FOUND",
                    "shard unstable during audit: " + shards[i],
                    shards[i],
                    "",
                )
            var before = c_realpath(shards[i])
            try:
                var st = read_header(shards[i])
                headers.append(st^)
            except e:
                eprint_json(String(e))
                exit(2)
            var after = c_realpath(shards[i])
            attempt += 1
            if before != "" and before == after:
                supplied_canon.append(after)
                break
            _ = headers.pop()
    # scope deterministik: full iff semua file unik weight_map ada di argumen.
    # "Ada" = canonical path pasokan == canonical path harapan (model_root +
    # nama index), bukan basename: impostor beda direktori dengan nama sama
    # tidak dihitung (fail-closed ke subset).
    var supplied = List[String]()
    for i in range(len(shards)):
        supplied.append(basename(shards[i]))
    var exp_canon = List[String]()
    for i in range(nn):
        exp_canon.append(c_realpath(String(model_root, "/", wfiles[i])))
    var full = True
    for i in range(nn):
        var found = False
        for j in range(len(supplied_canon)):
            if supplied_canon[j] != "" and supplied_canon[j] == exp_canon[i]:
                found = True
        if not found:
            full = False
    var scope = String("subset")
    if full:
        scope = String("full")
    # peta global nama -> file aktual + deteksi DUPLICATE_TENSOR_NAME (O(1) via Dict).
    # gfiles/gnames untuk laporan (basename, kontrak output stabil);
    # gcanon untuk pembanding identitas.
    var gnames = List[String]()
    var gfiles = List[String]()
    var gcanon = List[String]()
    var gpos = Dict[String, Int]()
    var mismatch = List[String]()
    for i in range(len(headers)):
        var base = basename(shards[i])
        ref st = headers[i]
        for k in range(len(st.entries)):
            ref e = st.entries[k]
            if e.name in gpos:
                var seen = gpos[e.name]
                mismatch.append(
                    String(
                        '{"tensor_name":"',
                        json_escape(e.name),
                        '","kind":"DUPLICATE_TENSOR_NAME","files":["',
                        json_escape(gfiles[seen]),
                        '","',
                        json_escape(base),
                        '"]}',
                    )
                )
            else:
                gpos[e.name] = len(gnames)
                gnames.append(e.name)
                gfiles.append(base)
                gcanon.append(supplied_canon[i])
    # compare vs weight_map (mode full + subset; subset tak boleh sembunyikan salah tempat).
    # total_tensors = len(weight_map) SELALU; assessed = yang dinilai; matched ⊆ assessed.
    var assessed = 0
    var matched = 0
    for i in range(nn):
        var gi = -1
        if wnames[i] in gpos:
            gi = gpos[wnames[i]]
        var exp_in = False
        for j in range(len(supplied_canon)):
            if supplied_canon[j] != "" and supplied_canon[j] == exp_canon[i]:
                exp_in = True
        if exp_in:
            assessed += 1
            if gi < 0:
                mismatch.append(
                    String(
                        '{"tensor_name":"',
                        json_escape(wnames[i]),
                        '","kind":"MISSING_IN_SHARD","expected_shard":"',
                        json_escape(wfiles[i]),
                        '"}',
                    )
                )
            elif gcanon[gi] != exp_canon[i]:
                mismatch.append(
                    String(
                        '{"tensor_name":"',
                        json_escape(wnames[i]),
                        '","kind":"WRONG_SHARD","expected_shard":"',
                        json_escape(wfiles[i]),
                        '","found_shard":"',
                        json_escape(gfiles[gi]),
                        '"}',
                    )
                )
            else:
                matched += 1
        elif gi >= 0:
            # subset: harapan di luar subset, tapi fisik ditemukan di pasokan -> tetap mismatch
            assessed += 1
            mismatch.append(
                String(
                    '{"tensor_name":"',
                    json_escape(wnames[i]),
                    '","kind":"WRONG_SHARD","expected_shard":"',
                    json_escape(wfiles[i]),
                    '","found_shard":"',
                    json_escape(gfiles[gi]),
                    '"}',
                )
            )
        # else: di luar subset dan tak ditemukan -> tidak dinilai
    # MISSING_IN_INDEX: nama di header pasokan tapi ∉ weight_map
    for g in range(len(gnames)):
        if gnames[g] not in wpos:
            mismatch.append(
                String(
                    '{"tensor_name":"',
                    json_escape(gnames[g]),
                    '","kind":"MISSING_IN_INDEX","found_shard":"',
                    json_escape(gfiles[g]),
                    '"}',
                )
            )
    var ms = Float64(perf_counter_ns() - t0) / 1000000.0
    var sup_json = String("")
    for j in range(len(supplied)):
        if j > 0:
            sup_json += ","
        sup_json += String('"', json_escape(supplied[j]), '"')
    var mm_json = String("")
    for j in range(len(mismatch)):
        if j > 0:
            mm_json += ","
        mm_json += mismatch[j]
    var status = String("mismatch")
    if len(mismatch) == 0:
        status = String("match")
    print(
        String(
            '{"status":"',
            status,
            '","scope":"',
            scope,
            '","supplied_shards":[',
            sup_json,
            '],"total_tensors":',
            nn,
            ',"assessed_tensors":',
            assessed,
            ',"matched_tensors":',
            matched,
            ',"mismatches":[',
            mm_json,
            '],"parse_time_ms":',
            ms,
            "}",
        )
    )
    if len(mismatch) == 0:
        exit(0)
    exit(1)
