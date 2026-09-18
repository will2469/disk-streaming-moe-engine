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
| `walltime_sec` | 0.0220 s | 0.0308 s | 0.0170 s | 0.0310 s |
| `tokens_per_sec` | 5750.1020 tok/s | 7282.3360 tok/s | 4116.1540 tok/s | 7424.5730 tok/s |
| `gdn_time_ms` | 2.8970 ms | 3.3384 ms | 2.4520 ms | 3.3720 ms |
| `gated_attn_time_ms` | 11.4020 ms | 17.3466 ms | 8.7310 ms | 17.4050 ms |
| `moe_time_ms` | 7.2960 ms | 9.7894 ms | 5.4210 ms | 9.8580 ms |
| `moe_percent` | 32.6470 % | 34.7984 % | 31.5850 % | 35.0520 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 2314.1985 tok/s | 3730.3155 tok/s | 1416.4180 tok/s | 3904.2670 tok/s |
| `gdn_time_ms` | 0.0440 ms | 0.0679 ms | 0.0280 ms | 0.2650 ms |
| `gated_attn_time_ms` | 0.2635 ms | 0.2987 ms | 0.1360 ms | 0.3200 ms |
| `moe_time_ms` | 0.0845 ms | 0.1343 ms | 0.0640 ms | 0.1480 ms |
| `moe_percent` | 24.4610 % | 32.0398 % | 18.8660 % | 40.5360 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.044 ms (11.2%)
- **Gated Attention Sublayer**: p50 = 0.264 ms (67.2%)
- **MoE Channel Mixer Sublayer**: p50 = 0.085 ms (24.5%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (24.5%).
