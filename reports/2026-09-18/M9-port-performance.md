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
| `tokens_per_sec` | 9081.5300 tok/s | 9349.2492 tok/s | 8363.3660 tok/s | 9382.2110 tok/s |
| `gdn_time_ms` | 1.4050 ms | 1.7886 ms | 1.3730 ms | 1.8590 ms |
| `gated_attn_time_ms` | 7.8350 ms | 7.8920 ms | 7.7830 ms | 7.9050 ms |
| `moe_time_ms` | 4.1950 ms | 5.3946 ms | 3.9760 ms | 5.6780 ms |
| `moe_percent` | 31.1650 % | 36.7590 % | 30.1150 % | 38.1420 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 3161.7365 tok/s | 3636.5669 tok/s | 2046.8980 tok/s | 3895.2790 tok/s |
| `gdn_time_ms` | 0.0320 ms | 0.0495 ms | 0.0260 ms | 0.0540 ms |
| `gated_attn_time_ms` | 0.1800 ms | 0.2857 ms | 0.1460 ms | 0.3050 ms |
| `moe_time_ms` | 0.0780 ms | 0.0951 ms | 0.0600 ms | 0.1120 ms |
| `moe_percent` | 24.9765 % | 31.4992 % | 19.6120 % | 34.3930 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.032 ms (11.0%)
- **Gated Attention Sublayer**: p50 = 0.180 ms (62.1%)
- **MoE Channel Mixer Sublayer**: p50 = 0.078 ms (25.0%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (25.0%).
