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
| `walltime_sec` | 0.0140 s | 0.0168 s | 0.0140 s | 0.0170 s |
| `tokens_per_sec` | 8694.7270 tok/s | 9101.7446 tok/s | 7527.9550 tok/s | 9117.8590 tok/s |
| `gdn_time_ms` | 2.0670 ms | 2.4892 ms | 1.6460 ms | 2.4910 ms |
| `gated_attn_time_ms` | 7.9440 ms | 8.0732 ms | 7.7850 ms | 8.0790 ms |
| `moe_time_ms` | 4.0760 ms | 5.9250 ms | 4.0500 ms | 6.0150 ms |
| `moe_percent` | 29.6500 % | 36.2924 % | 28.6600 % | 36.5830 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 2534.7240 tok/s | 3699.0671 tok/s | 1960.3070 tok/s | 3754.7120 tok/s |
| `gdn_time_ms` | 0.0420 ms | 0.0481 ms | 0.0290 ms | 0.0500 ms |
| `gated_attn_time_ms` | 0.2275 ms | 0.2912 ms | 0.1330 ms | 0.2970 ms |
| `moe_time_ms` | 0.0840 ms | 0.1036 ms | 0.0590 ms | 0.1190 ms |
| `moe_percent` | 24.2350 % | 32.1623 % | 19.7610 % | 33.3600 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.042 ms (11.9%)
- **Gated Attention Sublayer**: p50 = 0.227 ms (64.4%)
- **MoE Channel Mixer Sublayer**: p50 = 0.084 ms (24.2%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (24.2%).
