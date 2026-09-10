# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Kimo check-index — validasi silang dua-sumber index vs header (M0-W2)."""

from std.sys.arg import argv
from std.sys.terminate import exit
from std.time import perf_counter_ns
from safetensors import Scanner, STHeader, json_escape, read_header


def eprint_json(msg: String) raises:
    # /dev/stderr dibuka append (tanpa truncate: O_TRUNC di pipe -> ENXIO).
    # Bila device tak ada, fallback stdout agar error tetap terlihat.
    try:
        var e = open("/dev/stderr", "a")
        e.write_all(msg.as_bytes())
        e.close()
    except:
        print(msg)


def err_json(
    code: String, detail: String, shard: String, tensor: String
) -> String:
    # SEMUA field lolos json_escape (detail/shard/tensor bisa dari path CLI).
    return String(
        '{"error_type":"',
        json_escape(code),
        '","detail":"',
        json_escape(detail),
        '","shard":"',
        json_escape(shard),
        '","tensor_name":"',
        json_escape(tensor),
        '"}',
    )


def basename(path: String) -> String:
    var cut = -1
    var bl = path.as_bytes()
    for i in range(len(bl)):
        if Int(bl[i]) == 47:
            cut = i
    if cut < 0:
        return path
    var out = List[UInt8]()
    for i in range(cut + 1, len(bl)):
        out.append(bl[i])
    return String(from_utf8_lossy=Span(out))


def dirname(path: String) -> String:
    var cut = -1
    var bl = path.as_bytes()
    for i in range(len(bl)):
        if Int(bl[i]) == 47:
            cut = i
    if cut < 0:
        return ""
    var out = List[UInt8]()
    for i in range(cut):
        out.append(bl[i])
    return String(from_utf8_lossy=Span(out))


def read_small_file(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var n = Int(f.seek(0, 2))
    _ = f.seek(0, 0)
    if n > 100000000:
        f.close()
        raise Error(
            String(
                (
                    '{"error_type":"INVALID_HEADER","detail":"index too'
                    ' large","shard":"'
                ),
                path,
                '","tensor_name":""}',
            )
        )
    var out = f.read_bytes(n)
    f.close()
    if len(out) < n:
        raise Error(
            String(
                (
                    '{"error_type":"INVALID_HEADER","detail":"index'
                    ' truncated","shard":"'
                ),
                path,
                '","tensor_name":""}',
            )
        )
    return out^


def parse_index(path: String) raises -> List[String]:
    # return [names..., files...] sejajar: names[i] <-> files[i]
    var raw = read_small_file(path)
    var sc = Scanner(raw^, path)
    var names = List[String]()
    var files = List[String]()
    var seen = Dict[String, Int]()
    sc.skip_ws()
    sc.expect(123)
    while True:
        sc.skip_ws()
        if sc.eof():
            raise Error(
                String(
                    (
                        '{"error_type":"JSON_PARSE_ERROR","detail":"index'
                        ' cut","shard":"'
                    ),
                    path,
                    '","tensor_name":""}',
                )
            )
        if sc.peek() == 125:
            sc.pos += 1
            break
        var key = sc.parse_string()
        sc.expect(58)
        if key == "metadata":
            sc.skip_value()
        elif key == "weight_map":
            sc.skip_ws()
            sc.expect(123)
            while True:
                sc.skip_ws()
                if sc.eof():
                    raise Error(
                        String(
                            (
                                '{"error_type":"JSON_PARSE_ERROR","detail":"weight_map'
                                ' cut","shard":"'
                            ),
                            path,
                            '","tensor_name":""}',
                        )
                    )
                if sc.peek() == 125:
                    sc.pos += 1
                    break
                var nm = sc.parse_string()
                sc.expect(58)
                var fname = sc.parse_string()
                if nm in seen:
                    raise Error(
                        String(
                            (
                                '{"error_type":"DUPLICATE_JSON_KEY","detail":"dup'
                                ' weight_map key","shard":"'
                            ),
                            path,
                            '","tensor_name":"',
                            json_escape(nm),
                            '"}',
                        )
                    )
                seen[nm] = len(names)
                names.append(nm)
                files.append(fname)
                if len(names) > 100000:
                    raise Error(
                        String(
                            (
                                '{"error_type":"INVALID_HEADER","detail":"weight_map'
                                ' > 100000 entri","shard":"'
                            ),
                            path,
                            '","tensor_name":""}',
                        )
                    )
                sc.skip_ws()
                if sc.eof():
                    raise Error(
                        String(
                            (
                                '{"error_type":"JSON_PARSE_ERROR","detail":"weight_map'
                                ' cut","shard":"'
                            ),
                            path,
                            '","tensor_name":""}',
                        )
                    )
                var c = sc.peek()
                sc.pos += 1
                if c == 125:
                    break
                if c != 44:
                    raise Error(
                        String(
                            (
                                '{"error_type":"JSON_PARSE_ERROR","detail":"expect'
                                ' , or }","shard":"'
                            ),
                            path,
                            '","tensor_name":""}',
                        )
                    )
        else:
            sc.skip_value()
        sc.skip_ws()
        if sc.eof():
            raise Error(
                String(
                    (
                        '{"error_type":"JSON_PARSE_ERROR","detail":"index'
                        ' cut","shard":"'
                    ),
                    path,
                    '","tensor_name":""}',
                )
            )
        var sep = sc.peek()
        sc.pos += 1
        if sep == 125:
            break
        if sep != 44:
            raise Error(
                String(
                    (
                        '{"error_type":"JSON_PARSE_ERROR","detail":"expect , or'
                        ' }","shard":"'
                    ),
                    path,
                    '","tensor_name":""}',
                )
            )
    var packed = List[String]()
    packed.append(String(len(names)))
    for i in range(len(names)):
        packed.append(names[i])
        packed.append(files[i])
    return packed^


def fail(code: String, detail: String, shard: String, tensor: String) raises:
    eprint_json(err_json(code, detail, shard, tensor))
    exit(2)


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
    # baca header semua shard
    var headers = List[STHeader]()
    for i in range(len(shards)):
        try:
            var st = read_header(shards[i])
            headers.append(st^)
        except e:
            eprint_json(String(e))
            exit(2)
    # scope deterministik: full iff semua file unik weight_map ada di argumen
    var supplied = List[String]()
    for i in range(len(shards)):
        supplied.append(basename(shards[i]))
    var full = True
    for i in range(nn):
        var found = False
        for j in range(len(supplied)):
            if supplied[j] == wfiles[i]:
                found = True
        if not found:
            full = False
    var scope = String("subset")
    if full:
        scope = String("full")
    # peta global nama -> file aktual + deteksi DUPLICATE_TENSOR_NAME (O(1) via Dict)
    var gnames = List[String]()
    var gfiles = List[String]()
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
                        gfiles[seen],
                        '","',
                        base,
                        '"]}',
                    )
                )
            else:
                gpos[e.name] = len(gnames)
                gnames.append(e.name)
                gfiles.append(base)
    # compare vs weight_map (mode full + subset; subset tak boleh sembunyikan salah tempat).
    # total_tensors = len(weight_map) SELALU; assessed = yang dinilai; matched ⊆ assessed.
    var assessed = 0
    var matched = 0
    for i in range(nn):
        var gi = -1
        if wnames[i] in gpos:
            gi = gpos[wnames[i]]
        var exp_in = False
        for j in range(len(supplied)):
            if supplied[j] == wfiles[i]:
                exp_in = True
        if exp_in:
            assessed += 1
            if gi < 0:
                mismatch.append(
                    String(
                        '{"tensor_name":"',
                        json_escape(wnames[i]),
                        '","kind":"MISSING_IN_SHARD","expected_shard":"',
                        wfiles[i],
                        '"}',
                    )
                )
            elif gfiles[gi] != wfiles[i]:
                mismatch.append(
                    String(
                        '{"tensor_name":"',
                        json_escape(wnames[i]),
                        '","kind":"WRONG_SHARD","expected_shard":"',
                        wfiles[i],
                        '","found_shard":"',
                        gfiles[gi],
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
                    wfiles[i],
                    '","found_shard":"',
                    gfiles[gi],
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
                    gfiles[g],
                    '"}',
                )
            )
    var ms = Float64(perf_counter_ns() - t0) / 1000000.0
    var sup_json = String("")
    for j in range(len(supplied)):
        if j > 0:
            sup_json += ","
        sup_json += String('"', supplied[j], '"')
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


def main() raises:
    var args = argv()
    if len(args) < 2:
        fail("USAGE", "pakai: kimo check-index <shard>...", "", "")
    var cmd = String(args[1])
    if cmd != "check-index":
        fail("USAGE", String("subcommand tak dikenal: ", cmd), "", "")
    var shards = List[String]()
    for i in range(2, len(args)):
        shards.append(String(args[i]))
    cmd_check_index(shards^)
