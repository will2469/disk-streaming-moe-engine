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
| `walltime_sec` | 0.0150 s | 0.0160 s | 0.0140 s | 0.0160 s |
| `tokens_per_sec` | 8295.5990 tok/s | 8868.6710 tok/s | 7659.7940 tok/s | 8921.5830 tok/s |
| `gdn_time_ms` | 1.9780 ms | 2.4598 ms | 1.3940 ms | 2.4770 ms |
| `gated_attn_time_ms` | 7.9160 ms | 8.6088 ms | 7.8440 ms | 8.7130 ms |
| `moe_time_ms` | 5.1030 ms | 5.6452 ms | 4.2020 ms | 5.6600 ms |
| `moe_percent` | 34.1910 % | 35.3672 % | 29.2800 % | 35.4460 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 2220.5910 tok/s | 3607.2443 tok/s | 1785.9910 tok/s | 3624.0020 tok/s |
| `gdn_time_ms` | 0.0440 ms | 0.0613 ms | 0.0270 ms | 0.0650 ms |
| `gated_attn_time_ms` | 0.2625 ms | 0.2973 ms | 0.1290 ms | 0.3160 ms |
| `moe_time_ms` | 0.0870 ms | 0.1249 ms | 0.0650 ms | 0.1400 ms |
| `moe_percent` | 23.6585 % | 32.6804 % | 17.0470 % | 34.4800 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.044 ms (11.2%)
- **Gated Attention Sublayer**: p50 = 0.263 ms (66.7%)
- **MoE Channel Mixer Sublayer**: p50 = 0.087 ms (23.7%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (23.7%).
