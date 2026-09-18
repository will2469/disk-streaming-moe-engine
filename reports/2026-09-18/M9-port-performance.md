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
| `walltime_sec` | 0.0360 s | 0.0428 s | 0.0300 s | 0.0440 s |
| `tokens_per_sec` | 3467.9790 tok/s | 4132.5820 tok/s | 2895.3310 tok/s | 4247.8470 tok/s |
| `gdn_time_ms` | 4.0570 ms | 5.1234 ms | 3.6260 ms | 5.2600 ms |
| `gated_attn_time_ms` | 19.4990 ms | 26.0938 ms | 15.7850 ms | 27.6790 ms |
| `moe_time_ms` | 11.1630 ms | 12.1546 ms | 9.6950 ms | 12.3760 ms |
| `moe_percent` | 33.3090 % | 33.4446 % | 26.0220 % | 33.4740 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0010 s | 0.0010 s | 0.0000 s | 0.0010 s |
| `tokens_per_sec` | 866.4620 tok/s | 1481.2283 tok/s | 527.0020 tok/s | 1942.1360 tok/s |
| `gdn_time_ms` | 0.1260 ms | 0.2030 ms | 0.0520 ms | 0.2150 ms |
| `gated_attn_time_ms` | 0.6605 ms | 0.9170 ms | 0.3050 ms | 1.0890 ms |
| `moe_time_ms` | 0.2515 ms | 0.4553 ms | 0.1120 ms | 0.5820 ms |
| `moe_percent` | 25.4540 % | 29.7100 % | 22.4260 % | 37.8300 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.126 ms (12.1%)
- **Gated Attention Sublayer**: p50 = 0.660 ms (63.6%)
- **MoE Channel Mixer Sublayer**: p50 = 0.252 ms (25.5%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (25.5%).
