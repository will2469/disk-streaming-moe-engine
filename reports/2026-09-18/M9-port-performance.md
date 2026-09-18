# Performance Baseline Report: Qwen3.6-35B-A3B Port

- **Date**: 2026-09-18
- **Run ID Range**: `M9-20260918-001` .. `M9-20260918-035`
- **CPU Governor**: `powersave`
- **Standard Protocol**: `docs/03-testing.md` §4.4 (Prefill N=5, Decode N=30)
- **Threads**: 1 (Single-threaded deterministic baseline)

---

## 1. Summary Statistics: Prefill Phase (N=5, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0140 s | 0.0148 s | 0.0130 s | 0.0150 s |
| `tokens_per_sec` | 9123.5540 tok/s | 9342.5034 tok/s | 8116.0650 tok/s | 9350.9640 tok/s |
| `gdn_time_ms` | 1.4770 ms | 1.9280 ms | 1.3450 ms | 2.0250 ms |
| `gated_attn_time_ms` | 7.8790 ms | 8.6488 ms | 7.7320 ms | 8.8230 ms |
| `moe_time_ms` | 4.0670 ms | 5.1764 ms | 3.9630 ms | 5.4070 ms |
| `moe_percent` | 30.3540 % | 34.6254 % | 28.3080 % | 35.4470 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 3073.1050 tok/s | 3842.6694 tok/s | 1395.2390 tok/s | 3977.1230 tok/s |
| `gdn_time_ms` | 0.0390 ms | 0.0515 ms | 0.0250 ms | 0.0560 ms |
| `gated_attn_time_ms` | 0.1700 ms | 0.2891 ms | 0.1370 ms | 0.4280 ms |
| `moe_time_ms` | 0.0830 ms | 0.1004 ms | 0.0590 ms | 0.1560 ms |
| `moe_percent` | 25.9040 % | 32.7891 % | 20.0680 % | 34.9480 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.039 ms (13.4%)
- **Gated Attention Sublayer**: p50 = 0.170 ms (58.2%)
- **MoE Channel Mixer Sublayer**: p50 = 0.083 ms (25.9%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (25.9%).
