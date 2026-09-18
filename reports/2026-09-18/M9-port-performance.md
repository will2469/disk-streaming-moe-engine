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
| `walltime_sec` | 0.0140 s | 0.0158 s | 0.0140 s | 0.0160 s |
| `tokens_per_sec` | 8903.0320 tok/s | 9049.6078 tok/s | 7771.1030 tok/s | 9077.7240 tok/s |
| `gdn_time_ms` | 1.4540 ms | 1.8748 ms | 1.3530 ms | 1.9380 ms |
| `gated_attn_time_ms` | 8.3110 ms | 9.1020 ms | 7.7700 ms | 9.2980 ms |
| `moe_time_ms` | 4.1080 ms | 5.2934 ms | 3.9390 ms | 5.3390 ms |
| `moe_percent` | 29.8190 % | 34.2670 % | 28.4550 % | 34.4870 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 2520.0430 tok/s | 3760.2090 tok/s | 1590.9400 tok/s | 3796.6940 tok/s |
| `gdn_time_ms` | 0.0435 ms | 0.0655 ms | 0.0270 ms | 0.1170 ms |
| `gated_attn_time_ms` | 0.2385 ms | 0.3037 ms | 0.1420 ms | 0.4250 ms |
| `moe_time_ms` | 0.0825 ms | 0.1089 ms | 0.0580 ms | 0.1180 ms |
| `moe_percent` | 23.3655 % | 29.7776 % | 17.4200 % | 34.3330 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.043 ms (11.9%)
- **Gated Attention Sublayer**: p50 = 0.238 ms (65.4%)
- **MoE Channel Mixer Sublayer**: p50 = 0.083 ms (23.4%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (23.4%).
