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
| `walltime_sec` | 0.0160 s | 0.0170 s | 0.0130 s | 0.0170 s |
| `tokens_per_sec` | 7675.1560 tok/s | 9336.6984 tok/s | 7359.8700 tok/s | 9434.7690 tok/s |
| `gdn_time_ms` | 2.3810 ms | 2.4394 ms | 1.3120 ms | 2.4510 ms |
| `gated_attn_time_ms` | 8.0210 ms | 8.1736 ms | 7.6320 ms | 8.1920 ms |
| `moe_time_ms` | 5.7180 ms | 6.5012 ms | 4.0010 ms | 6.5830 ms |
| `moe_percent` | 35.4740 % | 39.0082 % | 30.3900 % | 39.5000 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 2356.5400 tok/s | 3798.4750 tok/s | 1442.0210 tok/s | 4006.6830 tok/s |
| `gdn_time_ms` | 0.0440 ms | 0.0557 ms | 0.0250 ms | 0.1020 ms |
| `gated_attn_time_ms` | 0.2540 ms | 0.3294 ms | 0.1330 ms | 0.3530 ms |
| `moe_time_ms` | 0.0835 ms | 0.1221 ms | 0.0550 ms | 0.1800 ms |
| `moe_percent` | 23.5550 % | 28.5362 % | 20.0910 % | 29.0280 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.044 ms (11.5%)
- **Gated Attention Sublayer**: p50 = 0.254 ms (66.6%)
- **MoE Channel Mixer Sublayer**: p50 = 0.084 ms (23.6%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (23.6%).
