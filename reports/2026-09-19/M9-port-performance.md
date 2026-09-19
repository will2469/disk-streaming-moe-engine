# Performance Baseline Report: Qwen3.6-35B-A3B Port

- **Date**: 2026-09-19
- **Run ID Range**: `M9-20260919-001` .. `M9-20260919-035`
- **CPU Governor**: `powersave`
- **Standard Protocol**: `docs/03-testing.md` §4.4 (Prefill N=5, Decode N=30)
- **Threads**: 1 (Single-threaded deterministic baseline)

---

## 1. Summary Statistics: Prefill Phase (N=5, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0380 s | 0.0424 s | 0.0360 s | 0.0430 s |
| `tokens_per_sec` | 3308.3560 tok/s | 3459.6658 tok/s | 2950.8550 tok/s | 3493.5780 tok/s |
| `gdn_time_ms` | 3.3620 ms | 3.9032 ms | 3.1960 ms | 3.9990 ms |
| `gated_attn_time_ms` | 17.6450 ms | 18.9606 ms | 15.1810 ms | 19.1540 ms |
| `moe_time_ms` | 9.0230 ms | 9.5890 ms | 8.2140 ms | 9.6870 ms |
| `moe_percent` | 29.6180 % | 31.7382 % | 28.8910 % | 32.0430 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0110 s | 0.0181 s | 0.0090 s | 0.0230 s |
| `tokens_per_sec` | 86.9005 tok/s | 99.9624 tok/s | 42.7280 tok/s | 104.6470 tok/s |
| `gdn_time_ms` | 0.0700 ms | 0.1625 ms | 0.0510 ms | 0.2790 ms |
| `gated_attn_time_ms` | 0.4440 ms | 0.6996 ms | 0.3290 ms | 0.7970 ms |
| `moe_time_ms` | 0.1565 ms | 0.3081 ms | 0.1150 ms | 0.4290 ms |
| `moe_percent` | 23.8920 % | 28.9318 % | 18.1950 % | 32.9500 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.070 ms (10.4%)
- **Gated Attention Sublayer**: p50 = 0.444 ms (66.2%)
- **MoE Channel Mixer Sublayer**: p50 = 0.157 ms (23.9%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (23.9%).
