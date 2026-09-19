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
| `walltime_sec` | 0.0250 s | 0.0250 s | 0.0190 s | 0.0250 s |
| `tokens_per_sec` | 5044.9780 tok/s | 6306.8684 tok/s | 4940.9890 tok/s | 6439.3190 tok/s |
| `gdn_time_ms` | 2.7180 ms | 3.3380 ms | 1.6790 ms | 3.4510 ms |
| `gated_attn_time_ms` | 9.0830 ms | 9.9670 ms | 8.5800 ms | 10.0660 ms |
| `moe_time_ms` | 5.8830 ms | 7.6090 ms | 4.3070 ms | 7.8610 ms |
| `moe_percent` | 31.1170 % | 39.3488 % | 28.5830 % | 40.6730 % |

---

## 2. Summary Statistics: Decode Continuation Phase (N=30, 2 Warmup)

| Metric | p50 | p95 | min | max |
| :--- | :---: | :---: | :---: | :---: |
| `walltime_sec` | 0.0060 s | 0.0075 s | 0.0040 s | 0.0080 s |
| `tokens_per_sec` | 163.7795 tok/s | 242.5493 tok/s | 120.9970 tok/s | 246.6400 tok/s |
| `gdn_time_ms` | 0.0395 ms | 0.0690 ms | 0.0210 ms | 0.0880 ms |
| `gated_attn_time_ms` | 0.1810 ms | 0.5095 ms | 0.1460 ms | 0.5440 ms |
| `moe_time_ms` | 0.0855 ms | 0.1311 ms | 0.0450 ms | 0.1400 ms |
| `moe_percent` | 23.9865 % | 30.3931 % | 12.6420 % | 31.2700 % |

---

## 3. Sublayer Breakdown & MoE Bottleneck Verification

- **GDN Sublayer**: p50 = 0.040 ms (12.9%)
- **Gated Attention Sublayer**: p50 = 0.181 ms (59.2%)
- **MoE Channel Mixer Sublayer**: p50 = 0.085 ms (24.0%)
- **Verdict**: [PASS] MoE Channel Mixer sublayer terukur (24.0%).
