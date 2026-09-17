# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Tool CLI Benchmark I/O O_DIRECT untuk Mengukur BW_seq, BW_exp(q), dan D_sus (M7)."""

from format.types import json_escape
from io.odirect import ODirectReader, ReadToken
from std.collections import List
from std.sys.arg import argv
from std.time import perf_counter_ns


def parse_offsets_from_fixture(
    fixture_path: String, pattern_id: String
) raises -> List[Int]:
    """Mengekstrak list offset integer dari file fixture JSON m7_io_patterns.json.
    """
    var f = open(fixture_path, "r")
    var content = f.read()
    f.close()

    var id_marker = String('"id": "') + pattern_id + String('"')
    var id_pos = content.find(id_marker)
    if id_pos < 0:
        raise Error("Pattern ID not found in fixture: " + pattern_id)

    var off_marker = String('"offsets": [')
    var off_pos = content.find(off_marker, id_pos)
    if off_pos < 0:
        raise Error("offsets array not found for pattern: " + pattern_id)

    var start_bracket = off_pos + off_marker.byte_length()
    var end_bracket = content.find("]", start_bracket)
    if end_bracket < 0:
        raise Error("Closing bracket not found for offsets array")

    var raw_bytes = content.as_bytes()
    var res = List[Int]()

    var current_num: Int = 0
    var has_num = False
    for idx in range(start_bracket, end_bracket):
        var b = raw_bytes[idx]
        if b >= 48 and b <= 57:  # '0'..'9'
            current_num = current_num * 10 + Int(b - 48)
            has_num = True
        elif (
            b == 44 or b == 10 or b == 13 or b == 32 or b == 9
        ):  # comma or whitespace
            if has_num:
                res.append(current_num)
                current_num = 0
                has_num = False

    if has_num:
        res.append(current_num)

    return res^


def parse_sha_from_fixture(
    fixture_path: String, pattern_id: String
) raises -> String:
    """Mengekstrak hash offsets_sha256 dari fixture JSON untuk validasi integritas.
    """
    var f = open(fixture_path, "r")
    var content = f.read()
    f.close()

    var id_marker = String('"id": "') + pattern_id + String('"')
    var id_pos = content.find(id_marker)
    if id_pos < 0:
        raise Error("Pattern ID not found in fixture: " + pattern_id)

    var sha_marker = String('"offsets_sha256": "')
    var sha_pos = content.find(sha_marker, id_pos)
    if sha_pos < 0:
        raise Error("offsets_sha256 not found for pattern: " + pattern_id)

    var start_sha = sha_pos + sha_marker.byte_length()
    var end_sha = content.find('"', start_sha)
    if end_sha < 0:
        raise Error("Closing quote not found for offsets_sha256")

    return String(content[byte=start_sha:end_sha])


def main() raises:
    var args = argv()

    var pattern = String("sequential")
    var block_size = 4194304
    var block_count = 100
    var queue_depth = 1
    var file_path = String("")
    var fixture_path = String("tools/fixtures/m7_io_patterns.json")
    var output_path = String("")
    var warmup_runs = 0

    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--pattern" and i + 1 < len(args):
            pattern = String(args[i + 1])
            i += 2
        elif a == "--block-size" and i + 1 < len(args):
            block_size = Int(String(args[i + 1]))
            i += 2
        elif a == "--block-count" and i + 1 < len(args):
            block_count = Int(String(args[i + 1]))
            i += 2
        elif a == "--queue-depth" and i + 1 < len(args):
            queue_depth = Int(String(args[i + 1]))
            i += 2
        elif a == "--file" and i + 1 < len(args):
            file_path = String(args[i + 1])
            i += 2
        elif a == "--offsets-fixture" and i + 1 < len(args):
            fixture_path = String(args[i + 1])
            i += 2
        elif a == "--output" and i + 1 < len(args):
            output_path = String(args[i + 1])
            i += 2
        elif a == "--warmup-runs" and i + 1 < len(args):
            warmup_runs = Int(String(args[i + 1]))
            i += 2
        else:
            i += 1

    if file_path.byte_length() == 0:
        raise Error("Parameter --file <path> wajib diisi.")

    # 1. Tentukan pattern_id dari parameter
    var pattern_id = String("trunk_sequential")
    if pattern == "random_jump" or pattern == "expert_miss":
        pattern_id = String("expert_miss")

    # 2. Muat offsets dari fixture
    var offsets = parse_offsets_from_fixture(fixture_path, pattern_id)
    var sha256_lock = parse_sha_from_fixture(fixture_path, pattern_id)

    if len(offsets) < block_count:
        block_count = len(offsets)

    # 3. Inisialisasi ODirectReader
    var reader = ODirectReader.discover(
        file_path, requested_block_size=4096, queue_depth=queue_depth
    )

    # 4. Eksekusi Warmup jika diminta
    if warmup_runs > 0:
        for w in range(min(5, block_count)):
            _ = reader.read_logical_payload(offsets[w], min(65536, block_size))

    # 5. Eksekusi Pengukuran Waktu Nyata
    var t_start = perf_counter_ns()

    var burst_ns: Int = 0
    var sustained_start_ns: Int = 0
    var burst_blocks = block_count // 10
    if burst_blocks < 1:
        burst_blocks = 1
    var sustained_blocks = (block_count * 25) // 100
    if sustained_blocks < 1:
        sustained_blocks = 1
    var sustained_start_idx = block_count - sustained_blocks

    var batch_idx = 0
    while batch_idx < block_count:
        var current_batch = min(queue_depth, block_count - batch_idx)
        var tokens = List[ReadToken]()

        # Submit batch hingga queue_depth
        for j in range(current_batch):
            var off = offsets[batch_idx + j]
            var tok = reader.submit_read(off, block_size)
            tokens.append(tok^)

        # Catat waktu burst (10% blok awal selesai)
        if (
            batch_idx < burst_blocks
            and (batch_idx + current_batch) >= burst_blocks
        ):
            burst_ns = perf_counter_ns() - t_start

        # Catat waktu awal sustained (25% blok akhir dimulai)
        if (
            batch_idx < sustained_start_idx
            and (batch_idx + current_batch) >= sustained_start_idx
        ):
            sustained_start_ns = perf_counter_ns()

        # Complete seluruh token dalam batch
        for j in range(len(tokens)):
            var tok = tokens[j].copy()
            _ = reader.complete_read(tok^)

        batch_idx += current_batch

    var t_end = perf_counter_ns()
    var total_elapsed_ns = t_end - t_start
    if total_elapsed_ns <= 0:
        total_elapsed_ns = 1

    var total_bytes = block_count * block_size
    var elapsed_sec = Float64(total_elapsed_ns) / 1000000000.0
    var bandwidth_gb_s = Float64(total_bytes) / elapsed_sec / 1000000000.0

    # Kalkulasi burst dan sustained
    if burst_ns <= 0:
        burst_ns = total_elapsed_ns
    var burst_bytes = burst_blocks * block_size
    var burst_sec = Float64(burst_ns) / 1000000000.0
    var bw_burst = Float64(burst_bytes) / burst_sec / 1000000000.0

    var sustained_ns = t_end - sustained_start_ns
    if sustained_ns <= 0:
        sustained_ns = total_elapsed_ns
    var sustained_bytes = sustained_blocks * block_size
    var sustained_sec = Float64(sustained_ns) / 1000000000.0
    var bw_sustained = Float64(sustained_bytes) / sustained_sec / 1000000000.0

    var d_sus = 0.0
    if bw_burst > 0.0 and bw_burst > bw_sustained:
        d_sus = (bw_burst - bw_sustained) / bw_burst

    var max_obs = reader.max_outstanding_observed

    # 6. Susun Output JSON strict RFC 8259
    var json_res = String(
        '{"status":"success","pattern":"',
        json_escape(pattern),
        '","pattern_id":"',
        json_escape(pattern_id),
        '","file":"',
        json_escape(file_path),
        '","block_size":',
        String(block_size),
        ',"block_count":',
        String(block_count),
        ',"queue_depth":',
        String(queue_depth),
        ',"max_outstanding_observed":',
        String(max_obs),
        ',"elapsed_sec":',
        String(elapsed_sec),
        ',"total_bytes":',
        String(total_bytes),
        ',"bandwidth_gb_s":',
        String(bandwidth_gb_s),
        ',"bw_burst_gb_s":',
        String(bw_burst),
        ',"bw_sustained_gb_s":',
        String(bw_sustained),
        ',"d_sus":',
        String(d_sus),
        ',"dio_alignment":',
        String(reader.dio_alignment),
        ',"offsets_sha256":"',
        json_escape(sha256_lock),
        '"}',
    )

    if output_path.byte_length() > 0:
        var f_out = open(output_path, "w")
        f_out.write(json_res)
        f_out.close()

    print(json_res)
    reader.close()
