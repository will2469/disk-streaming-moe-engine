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
| `walltime_sec` | 0.0370 s | 0.0380 s | 0.0290 s | 0.0380 s |
| `tokens_per_sec` | 3430.8360 tok/s | 4172.4792 tok/s | 3333.1870 tok/s | 4350.6760 tok/s |
| `gdn_time_ms` | 3.9290 ms | 4.8308 ms | 3.4890 ms | 5.0170 ms |
| `gated_attn_time_ms` | 20.9400 ms | 21.5388 ms | 15.6430 ms | 21.5390 ms |
| `moe_time_ms` | 11.3040 ms | 12.1582 ms | 9.2120 ms | 12.3400 ms |
| `moe_percent` | 31.9810 % | 33.0314 % | 30.6090 % | 33.1640 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0015 s | 0.0000 s | 0.0020 s |
| `tokens_per_sec` | 1111.6830 tok/s | 1550.7604 tok/s | 456.5010 tok/s | 1914.0880 tok/s |
| `gdn_time_ms` | 0.0965 ms | 0.2029 ms | 0.0530 ms | 0.2360 ms |
| `gated_attn_time_ms` | 0.5305 ms | 1.2424 ms | 0.3180 ms | 1.3630 ms |
| `moe_time_ms` | 0.1965 ms | 0.4097 ms | 0.1050 ms | 0.4370 ms |
| `moe_percent` | 24.6880 % | 27.4864 % | 20.8520 % | 29.2080 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.097 ms (11.7%)
- **Gated Attention Sublayer**: p50 = 0.530 ms (64.4%)
- **MoE Channel Mixer Sublayer**: p50 = 0.197 ms (24.7%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (24.7%).
