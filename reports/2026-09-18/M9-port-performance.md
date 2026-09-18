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
| `walltime_sec` | 0.0220 s | 0.0240 s | 0.0150 s | 0.0240 s |
| `tokens_per_sec` | 5750.2980 tok/s | 7764.2234 tok/s | 5162.5330 tok/s | 8090.1830 tok/s |
| `gdn_time_ms` | 3.5270 ms | 3.5410 ms | 1.6080 ms | 3.5440 ms |
| `gated_attn_time_ms` | 9.7430 ms | 11.4964 ms | 9.2250 ms | 11.6450 ms |
| `moe_time_ms` | 8.2340 ms | 9.2618 ms | 4.5090 ms | 9.3980 ms |
| `moe_percent` | 36.8540 % | 39.1866 % | 29.3600 % | 39.4120 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 2945.8995 tok/s | 3247.1564 tok/s | 1707.8950 tok/s | 3381.4170 tok/s |
| `gdn_time_ms` | 0.0365 ms | 0.0607 ms | 0.0280 ms | 0.0630 ms |
| `gated_attn_time_ms` | 0.1855 ms | 0.3311 ms | 0.1650 ms | 0.3440 ms |
| `moe_time_ms` | 0.0820 ms | 0.1279 ms | 0.0680 ms | 0.1330 ms |
| `moe_percent` | 26.2370 % | 31.8017 % | 21.1870 % | 32.5950 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.036 ms (12.0%)
- **Gated Attention Sublayer**: p50 = 0.185 ms (61.0%)
- **MoE Channel Mixer Sublayer**: p50 = 0.082 ms (26.2%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (26.2%).
