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
| `walltime_sec` | 0.0160 s | 0.0196 s | 0.0150 s | 0.0200 s |
| `tokens_per_sec` | 7629.6470 tok/s | 8029.6596 tok/s | 6161.2940 tok/s | 8123.1100 tok/s |
| `gdn_time_ms` | 2.0530 ms | 2.8478 ms | 1.9260 ms | 2.9240 ms |
| `gated_attn_time_ms` | 9.1960 ms | 9.3374 ms | 8.1550 ms | 9.3620 ms |
| `moe_time_ms` | 5.9810 ms | 7.4282 ms | 4.5100 ms | 7.7810 ms |
| `moe_percent` | 33.7540 % | 38.4672 % | 29.6780 % | 38.7750 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 2317.9300 tok/s | 3493.0630 tok/s | 1362.8730 tok/s | 3646.6530 tok/s |
| `gdn_time_ms` | 0.0460 ms | 0.0617 ms | 0.0290 ms | 0.0970 ms |
| `gated_attn_time_ms` | 0.2475 ms | 0.3266 ms | 0.1310 ms | 0.3540 ms |
| `moe_time_ms` | 0.0900 ms | 0.1502 ms | 0.0690 ms | 0.3170 ms |
| `moe_percent` | 26.0050 % | 41.9467 % | 20.4560 % | 45.3740 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.046 ms (12.0%)
- **Gated Attention Sublayer**: p50 = 0.247 ms (64.5%)
- **MoE Channel Mixer Sublayer**: p50 = 0.090 ms (26.0%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (26.0%).
