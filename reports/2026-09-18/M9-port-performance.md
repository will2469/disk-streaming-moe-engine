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
| `walltime_sec` | 0.0160 s | 0.0188 s | 0.0130 s | 0.0190 s |
| `tokens_per_sec` | 7804.9770 tok/s | 9400.4350 tok/s | 6618.0180 tok/s | 9419.3900 tok/s |
| `gdn_time_ms` | 2.4040 ms | 2.9450 ms | 1.3560 ms | 2.9570 ms |
| `gated_attn_time_ms` | 7.9060 ms | 8.4280 ms | 7.6960 ms | 8.5310 ms |
| `moe_time_ms` | 5.7420 ms | 7.1114 ms | 3.8550 ms | 7.1720 ms |
| `moe_percent` | 36.2420 % | 39.1168 % | 29.2160 % | 39.2870 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0010 s |
| `tokens_per_sec` | 2200.3115 tok/s | 3497.2318 tok/s | 731.6750 tok/s | 3691.0170 tok/s |
| `gdn_time_ms` | 0.0440 ms | 0.0719 ms | 0.0270 ms | 0.1370 ms |
| `gated_attn_time_ms` | 0.2670 ms | 0.3617 ms | 0.1420 ms | 0.7120 ms |
| `moe_time_ms` | 0.0890 ms | 0.1468 ms | 0.0630 ms | 0.3490 ms |
| `moe_percent` | 23.5420 % | 31.5969 % | 20.0890 % | 33.8080 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.044 ms (11.0%)
- **Gated Attention Sublayer**: p50 = 0.267 ms (66.8%)
- **MoE Channel Mixer Sublayer**: p50 = 0.089 ms (23.5%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (23.5%).
