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
| `walltime_sec` | 0.0190 s | 0.0226 s | 0.0170 s | 0.0230 s |
| `tokens_per_sec` | 6550.4200 tok/s | 7358.0656 tok/s | 5432.1770 tok/s | 7369.2770 tok/s |
| `gdn_time_ms` | 1.8410 ms | 2.7172 ms | 1.3540 ms | 2.8110 ms |
| `gated_attn_time_ms` | 7.9450 ms | 8.0242 ms | 7.7000 ms | 8.0280 ms |
| `moe_time_ms` | 4.8510 ms | 6.9146 ms | 3.7010 ms | 7.2830 ms |
| `moe_percent` | 33.1410 % | 39.6346 % | 28.8930 % | 40.9300 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0040 s | 0.0050 s | 0.0030 s | 0.0050 s |
| `tokens_per_sec` | 238.3720 tok/s | 266.0582 tok/s | 171.4580 tok/s | 269.6410 tok/s |
| `gdn_time_ms` | 0.0235 ms | 0.0381 ms | 0.0190 ms | 0.1540 ms |
| `gated_attn_time_ms` | 0.1400 ms | 0.2751 ms | 0.1200 ms | 0.2820 ms |
| `moe_time_ms` | 0.0465 ms | 0.0820 ms | 0.0400 ms | 0.0820 ms |
| `moe_percent` | 21.7780 % | 27.3362 % | 14.0990 % | 27.8770 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.024 ms (11.2%)
- **Gated Attention Sublayer**: p50 = 0.140 ms (66.7%)
- **MoE Channel Mixer Sublayer**: p50 = 0.046 ms (21.8%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (21.8%).
