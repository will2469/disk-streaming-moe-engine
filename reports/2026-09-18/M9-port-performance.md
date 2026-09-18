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
| `tokens_per_sec` | 9013.9390 tok/s | 9339.1092 tok/s | 8428.3370 tok/s | 9402.3920 tok/s |
| `gdn_time_ms` | 1.7510 ms | 2.1852 ms | 1.6430 ms | 2.1960 ms |
| `gated_attn_time_ms` | 7.8550 ms | 8.3278 ms | 7.7090 ms | 8.4350 ms |
| `moe_time_ms` | 4.0810 ms | 4.2426 ms | 3.7990 ms | 4.2810 ms |
| `moe_percent` | 29.7240 % | 30.1284 % | 27.8780 % | 30.1760 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0000 s | 0.0000 s | 0.0000 s | 0.0000 s |
| `tokens_per_sec` | 2340.8945 tok/s | 3842.1249 tok/s | 1319.4400 tok/s | 3970.7270 tok/s |
| `gdn_time_ms` | 0.0440 ms | 0.0574 ms | 0.0300 ms | 0.1010 ms |
| `gated_attn_time_ms` | 0.2580 ms | 0.3002 ms | 0.1270 ms | 0.4130 ms |
| `moe_time_ms` | 0.0860 ms | 0.1210 ms | 0.0580 ms | 0.1900 ms |
| `moe_percent` | 24.1725 % | 29.0173 % | 20.3100 % | 31.4270 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.044 ms (11.3%)
- **Gated Attention Sublayer**: p50 = 0.258 ms (66.5%)
- **MoE Channel Mixer Sublayer**: p50 = 0.086 ms (24.2%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (24.2%).
