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
| `walltime_sec` | 0.0180 s | 0.0196 s | 0.0170 s | 0.0200 s |
| `tokens_per_sec` | 6996.8520 tok/s | 7388.7336 tok/s | 6301.0510 tok/s | 7392.4640 tok/s |
| `gdn_time_ms` | 1.4980 ms | 2.2074 ms | 1.3170 ms | 2.3570 ms |
| `gated_attn_time_ms` | 8.1920 ms | 8.6376 ms | 7.7820 ms | 8.7420 ms |
| `moe_time_ms` | 3.9520 ms | 4.8878 ms | 3.6990 ms | 5.0720 ms |
| `moe_percent` | 29.5670 % | 32.1836 % | 27.8770 % | 32.7470 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0040 s | 0.0050 s | 0.0030 s | 0.0060 s |
| `tokens_per_sec` | 205.8885 tok/s | 260.1338 tok/s | 156.4920 tok/s | 267.0770 tok/s |
| `gdn_time_ms` | 0.0300 ms | 0.0370 ms | 0.0190 ms | 0.0390 ms |
| `gated_attn_time_ms` | 0.1395 ms | 0.3365 ms | 0.1200 ms | 0.3860 ms |
| `moe_time_ms` | 0.0625 ms | 0.0831 ms | 0.0400 ms | 0.0900 ms |
| `moe_percent` | 23.6850 % | 29.5826 % | 15.5180 % | 30.2740 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.030 ms (12.9%)
- **Gated Attention Sublayer**: p50 = 0.140 ms (60.1%)
- **MoE Channel Mixer Sublayer**: p50 = 0.062 ms (23.7%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (23.7%).
