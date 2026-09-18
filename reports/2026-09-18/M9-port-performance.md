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
| `walltime_sec` | 0.0280 s | 0.0342 s | 0.0260 s | 0.0350 s |
| `tokens_per_sec` | 4570.6180 tok/s | 4836.3486 tok/s | 3643.7340 tok/s | 4846.3200 tok/s |
| `gdn_time_ms` | 2.7660 ms | 4.4904 ms | 2.5770 ms | 4.8920 ms |
| `gated_attn_time_ms` | 15.3110 ms | 18.9256 ms | 15.1040 ms | 19.2610 ms |
| `moe_time_ms` | 8.8050 ms | 10.9498 ms | 7.6390 ms | 11.3820 ms |
| `moe_percent` | 31.7750 % | 33.8074 % | 28.4490 % | 33.8550 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0010 s | 0.0000 s | 0.0010 s |
| `tokens_per_sec` | 1286.9595 tok/s | 1567.1354 tok/s | 657.0330 tok/s | 1817.3650 tok/s |
| `gdn_time_ms` | 0.0850 ms | 0.1241 ms | 0.0650 ms | 0.1660 ms |
| `gated_attn_time_ms` | 0.4360 ms | 0.7578 ms | 0.3050 ms | 0.8840 ms |
| `moe_time_ms` | 0.1720 ms | 0.2916 ms | 0.1340 ms | 0.3580 ms |
| `moe_percent` | 24.8375 % | 26.8779 % | 20.6290 % | 27.9470 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.085 ms (12.3%)
- **Gated Attention Sublayer**: p50 = 0.436 ms (62.9%)
- **MoE Channel Mixer Sublayer**: p50 = 0.172 ms (24.8%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (24.8%).
