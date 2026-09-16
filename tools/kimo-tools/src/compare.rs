// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! Subcommand `kimo-tools compare` — evaluasi ekivalensi numerik F10.
//!
//! Menghitung kelima metrik F10:
//!   - delta_max: selisih absolut maksimum
//!   - epsilon_rel: error relatif L2
//!   - cos_theta: cosine similarity
//!   - agreement: top-1 token agreement (%)
//!   - delta_ce: selisih cross entropy / entropy rata-rata
//!
//! Exit codes:
//!   0: MATCH (threshold gate terpenuhi)
//!   1: FAIL / MISMATCH (threshold tidak terpenuhi)
//!   2: ERROR (file hilang, layout/shape mismatch, JSON malformed)

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Serialize)]
pub struct CompareMetrics {
    pub delta_max: f64,
    pub epsilon_rel: f64,
    pub cos_theta: f64,
    pub agreement: f64,
    pub delta_ce: f64,
}

#[derive(Serialize)]
pub struct CompareReport {
    pub status: String,
    pub run_id: String,
    pub reference_path: String,
    pub candidate_path: String,
    pub metrics: CompareMetrics,
    pub verdict: String,
    pub threshold: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fail_category: Option<String>,
}

#[derive(Serialize)]
pub struct ErrorReport {
    pub error_type: String,
    pub detail: String,
    pub stage: String,
}

#[derive(Deserialize)]
struct RoutingFile {
    #[serde(default)]
    selected_experts: Vec<Vec<usize>>,
    #[serde(default)]
    routing_info: Option<RoutingInner>,
}

#[derive(Deserialize)]
struct RoutingInner {
    selected_experts: Vec<Vec<usize>>,
}

pub fn extract_selected_experts(path: &str) -> Result<Vec<Vec<usize>>, (String, String)> {
    if !Path::new(path).exists() {
        return Err((
            "FILE_NOT_FOUND".to_string(),
            format!("routing file not found: {}", path),
        ));
    }
    let content = fs::read_to_string(path).map_err(|e| {
        (
            "FILE_NOT_FOUND".to_string(),
            format!("cannot read routing file {}: {}", path, e),
        )
    })?;
    let rf: RoutingFile = serde_json::from_str(&content).map_err(|e| {
        (
            "ROUTING_PARSE_ERROR".to_string(),
            format!("cannot parse routing json {}: {}", path, e),
        )
    })?;
    if !rf.selected_experts.is_empty() {
        Ok(rf.selected_experts)
    } else if let Some(inner) = rf.routing_info {
        Ok(inner.selected_experts)
    } else {
        Err((
            "ROUTING_PARSE_ERROR".to_string(),
            format!("missing selected_experts in {}", path),
        ))
    }
}

pub fn check_routing_match(oracle_experts: &[Vec<usize>], cand_experts: &[Vec<usize>]) -> bool {
    if oracle_experts.len() != cand_experts.len() {
        return false;
    }
    for (o_row, c_row) in oracle_experts.iter().zip(cand_experts.iter()) {
        let o_set: HashSet<usize> = o_row.iter().copied().collect();
        let c_set: HashSet<usize> = c_row.iter().copied().collect();
        if o_set != c_set {
            return false;
        }
    }
    true
}

fn emit_error(err_type: &str, detail: &str) -> i32 {
    let err = ErrorReport {
        error_type: err_type.to_string(),
        detail: detail.to_string(),
        stage: "compare".to_string(),
    };
    eprintln!("{}", serde_json::to_string(&err).unwrap());
    2
}

pub fn read_floats(path: &str) -> Result<Vec<f32>, (String, String)> {
    if !Path::new(path).exists() {
        return Err((
            "FILE_NOT_FOUND".to_string(),
            format!("file not found: {}", path),
        ));
    }
    let bytes = fs::read(path).map_err(|e| {
        (
            "FILE_NOT_FOUND".to_string(),
            format!("cannot read file {}: {}", path, e),
        )
    })?;
    if bytes.len() % 4 != 0 {
        return Err((
            "LAYOUT_MISMATCH".to_string(),
            format!("file size {} is not a multiple of 4 bytes", bytes.len()),
        ));
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    let floats: Vec<f32> = chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    Ok(floats)
}

pub fn compute_f10_metrics(
    ref_data: &[f32],
    cand_data: &[f32],
    vocab_size: usize,
) -> Result<CompareMetrics, String> {
    if ref_data.len() != cand_data.len() {
        return Err(format!(
            "length mismatch: ref has {} elements, cand has {}",
            ref_data.len(),
            cand_data.len()
        ));
    }
    if ref_data.is_empty() {
        return Err("empty data buffer".to_string());
    }
    if !ref_data.len().is_multiple_of(vocab_size) {
        return Err(format!(
            "total elements {} not divisible by vocab_size {}",
            ref_data.len(),
            vocab_size
        ));
    }

    let n = ref_data.len();
    let num_tokens = n / vocab_size;

    let mut delta_max: f64 = 0.0;
    let mut sum_diff_sq: f64 = 0.0;
    let mut sum_ref_sq: f64 = 0.0;
    let mut sum_cand_sq: f64 = 0.0;
    let mut dot_prod: f64 = 0.0;

    for i in 0..n {
        let r = ref_data[i] as f64;
        let c = cand_data[i] as f64;

        if !r.is_finite() || !c.is_finite() {
            return Err(format!("non-finite value detected at index {}", i));
        }

        let diff = (c - r).abs();
        if diff > delta_max {
            delta_max = diff;
        }
        sum_diff_sq += diff * diff;
        sum_ref_sq += r * r;
        sum_cand_sq += c * c;
        dot_prod += r * c;
    }

    let epsilon_rel = if sum_ref_sq > 0.0 {
        (sum_diff_sq / sum_ref_sq).sqrt()
    } else {
        0.0
    };

    let norm_prod = (sum_ref_sq.sqrt()) * (sum_cand_sq.sqrt());
    let cos_theta = if norm_prod > 0.0 {
        dot_prod / norm_prod
    } else {
        1.0
    };

    // Agreement (A) & Cross Entropy (delta_ce)
    let mut matched_tokens = 0;
    let mut total_ce_ref: f64 = 0.0;
    let mut total_ce_cand: f64 = 0.0;

    for t in 0..num_tokens {
        let start = t * vocab_size;
        let end = start + vocab_size;

        let r_slice = &ref_data[start..end];
        let c_slice = &cand_data[start..end];

        let mut r_argmax = 0;
        let mut r_max_val = f32::NEG_INFINITY;
        let mut c_argmax = 0;
        let mut c_max_val = f32::NEG_INFINITY;

        for v in 0..vocab_size {
            if r_slice[v] > r_max_val {
                r_max_val = r_slice[v];
                r_argmax = v;
            }
            if c_slice[v] > c_max_val {
                c_max_val = c_slice[v];
                c_argmax = v;
            }
        }

        if r_argmax == c_argmax {
            matched_tokens += 1;
        }

        // Stable softmax entropy
        let mut r_exp_sum: f64 = 0.0;
        let mut c_exp_sum: f64 = 0.0;
        for v in 0..vocab_size {
            r_exp_sum += ((r_slice[v] - r_max_val) as f64).exp();
            c_exp_sum += ((c_slice[v] - c_max_val) as f64).exp();
        }

        let mut ce_ref_tok: f64 = 0.0;
        let mut ce_cand_tok: f64 = 0.0;
        for v in 0..vocab_size {
            let p_r = ((r_slice[v] - r_max_val) as f64).exp() / r_exp_sum;
            let p_c = ((c_slice[v] - c_max_val) as f64).exp() / c_exp_sum;

            let safe_p_r = if p_r > 1e-12 { p_r } else { 1e-12 };
            let safe_p_c = if p_c > 1e-12 { p_c } else { 1e-12 };

            ce_ref_tok -= p_r * safe_p_r.ln();
            ce_cand_tok -= p_c * safe_p_c.ln();
        }

        total_ce_ref += ce_ref_tok;
        total_ce_cand += ce_cand_tok;
    }

    let agreement = (matched_tokens as f64 / num_tokens as f64) * 100.0;
    let avg_ce_ref = total_ce_ref / (num_tokens as f64);
    let avg_ce_cand = total_ce_cand / (num_tokens as f64);
    let delta_ce = (avg_ce_cand - avg_ce_ref).abs();

    Ok(CompareMetrics {
        delta_max,
        epsilon_rel,
        cos_theta,
        agreement,
        delta_ce,
    })
}

pub fn run(args: &[String]) -> i32 {
    let mut ref_path = String::new();
    let mut cand_path = String::new();
    let mut gate = "G-M1-1".to_string();
    let mut vocab_size: usize = 512;
    let mut explicit_dim = false;
    let mut run_id = String::new();
    let mut oracle_routing = String::new();
    let mut cand_routing = String::new();

    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ref" => {
                i += 1;
                if i < args.len() {
                    ref_path = args[i].clone();
                }
            }
            "--cand" => {
                i += 1;
                if i < args.len() {
                    cand_path = args[i].clone();
                }
            }
            "--gate" => {
                i += 1;
                if i < args.len() {
                    gate = args[i].clone();
                }
            }
            "--dim" | "--dim-size" | "--vocab" | "--vocab-size" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        vocab_size = v;
                        explicit_dim = true;
                    }
                }
            }
            "--run-id" => {
                i += 1;
                if i < args.len() {
                    run_id = args[i].clone();
                }
            }
            "--oracle-routing" => {
                i += 1;
                if i < args.len() {
                    oracle_routing = args[i].clone();
                }
            }
            "--cand-routing" => {
                i += 1;
                if i < args.len() {
                    cand_routing = args[i].clone();
                }
            }
            s if s.starts_with('-') => {
                return emit_error("USAGE", &format!("unknown option: {}", s));
            }
            _ => {
                positional.push(args[i].clone());
            }
        }
        i += 1;
    }

    if ref_path.is_empty() && !positional.is_empty() {
        ref_path = positional.remove(0);
    }
    if cand_path.is_empty() && !positional.is_empty() {
        cand_path = positional.remove(0);
    }

    if ref_path.is_empty() || cand_path.is_empty() {
        return emit_error(
            "USAGE",
            "pakai: kimo-tools compare <ref.bin> <cand.bin> [--gate G-M1-1|G-M2-1|G-M3-1|G-M4-1|G-M5-1] [--dim <N>] [--oracle-routing <json>] [--cand-routing <json>]",
        );
    }

    let has_routing = !oracle_routing.is_empty() || !cand_routing.is_empty();
    let mut routing_ok = true;
    if has_routing {
        if oracle_routing.is_empty() || cand_routing.is_empty() {
            return emit_error(
                "USAGE",
                "both --oracle-routing and --cand-routing must be provided together",
            );
        }
        let o_exp = match extract_selected_experts(&oracle_routing) {
            Ok(v) => v,
            Err((code, detail)) => return emit_error(&code, &detail),
        };
        let c_exp = match extract_selected_experts(&cand_routing) {
            Ok(v) => v,
            Err((code, detail)) => return emit_error(&code, &detail),
        };
        routing_ok = check_routing_match(&o_exp, &c_exp);
    }

    if run_id.is_empty() {
        run_id = match gate.as_str() {
            "G-M4-1" => "M4-F10-001".to_string(),
            "G-M5-1" => "M5-F10-001".to_string(),
            "G-M3-1" => "M3-F10-001".to_string(),
            "G-M2-1" => "M2-F10-001".to_string(),
            _ => "M1-F10-001".to_string(),
        };
    }

    let ref_floats = match read_floats(&ref_path) {
        Ok(v) => v,
        Err((code, detail)) => return emit_error(&code, &detail),
    };

    let cand_floats = match read_floats(&cand_path) {
        Ok(v) => v,
        Err((code, detail)) => return emit_error(&code, &detail),
    };

    if !explicit_dim && (gate == "G-M2-1" || gate == "G-M3-1") {
        if ref_floats.len() % 2048 == 0 {
            vocab_size = 2048;
        } else if ref_floats.len() % 64 == 0 {
            vocab_size = 64;
        } else {
            vocab_size = ref_floats.len();
        }
    }

    let metrics = match compute_f10_metrics(&ref_floats, &cand_floats, vocab_size) {
        Ok(m) => m,
        Err(e) => return emit_error("LAYOUT_MISMATCH", &e),
    };

    // Evaluate gate
    let (mut is_pass, threshold_str, mut fail_cat) = evaluate_gate(&gate, &metrics);

    if has_routing && !routing_ok {
        is_pass = false;
        fail_cat = Some("router-selection".to_string());
    }

    let report = CompareReport {
        status: if is_pass {
            "MATCH".to_string()
        } else {
            "MISMATCH".to_string()
        },
        run_id,
        reference_path: ref_path,
        candidate_path: cand_path,
        metrics,
        verdict: if is_pass {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        threshold: threshold_str.to_string(),
        fail_category: fail_cat,
    };

    println!("{}", serde_json::to_string_pretty(&report).unwrap());

    if is_pass {
        0
    } else {
        1
    }
}

pub fn evaluate_gate(gate: &str, metrics: &CompareMetrics) -> (bool, &'static str, Option<String>) {
    match gate {
        "G-M1-1" => {
            let pass = metrics.delta_max <= 1e-3
                && metrics.epsilon_rel <= 1e-4
                && (metrics.agreement - 100.0).abs() < 1e-5;
            let thresh = "delta_max <= 1e-3 && epsilon_rel <= 1e-4 && agreement == 100.0";
            let cat = if pass {
                None
            } else if metrics.agreement < 99.0 || metrics.delta_max > 0.05 {
                Some("dtype-layout".to_string())
            } else {
                Some("numeric-order".to_string())
            };
            (pass, thresh, cat)
        }
        "G-M2-1" | "G-M3-1" => {
            let pass = metrics.delta_max <= 1e-3 && metrics.epsilon_rel <= 1e-4;
            let thresh = "delta_max <= 1e-3 && epsilon_rel <= 1e-4";
            let cat = if pass {
                None
            } else if metrics.delta_max > 1.0 || metrics.cos_theta < 0.90 {
                Some("dtype-layout".to_string())
            } else if metrics.delta_max > 0.25 && metrics.delta_max <= 1.0 {
                Some("bias-placement".to_string())
            } else if metrics.delta_max >= 0.05 && metrics.delta_max <= 0.25 {
                Some("rope-style".to_string())
            } else {
                Some("numeric-order".to_string())
            };
            (pass, thresh, cat)
        }
        "G-M4-1" | "G-M5-1" => {
            // Loose full-forward gate: FP32 vs FP32 (03-testing.md §4.3).
            // Kandidat BF16 DILARANG — noise kuantisasi serialisasi
            // (round-trip murni ε_rel ≈ 1,7e-3) menenggelamkan threshold 1e-4.
            let pass = metrics.delta_max <= 1e-2
                && metrics.epsilon_rel <= 1e-4
                && metrics.agreement >= 99.9
                && metrics.delta_ce <= 0.02;
            let thresh =
                "delta_max <= 1e-2 && epsilon_rel <= 1e-4 && agreement >= 99.9 && delta_ce <= 0.02";
            // F10-A lebih dulu: agreement, lalu pita Δ (prosedur verdict M4 A→N→S).
            // Pita-pita ini screening (Tier-2); bukti definitif router = SET equality
            // via --oracle-routing/--cand-routing (ditangani di run(): override
            // menjadi "router-selection"). "tensor-mapping" adalah alias `dtype-layout`.
            let cat = if pass {
                None
            } else if metrics.agreement < 99.9 {
                Some("argmax-mismatch".to_string())
            } else if metrics.delta_max > 1.0 || metrics.cos_theta < 0.90 {
                Some("dtype-layout".to_string())
            } else if metrics.delta_max > 0.25 {
                Some("bias-placement".to_string())
            } else if metrics.delta_max >= 0.05 {
                Some("rope-style".to_string())
            } else {
                Some("numeric-order".to_string())
            };
            (pass, thresh, cat)
        }
        _ => {
            // Default loose check
            let pass = metrics.delta_max <= 1e-2 && metrics.epsilon_rel <= 1e-4;
            let thresh = "delta_max <= 1e-2 && epsilon_rel <= 1e-4";
            let cat = if pass {
                None
            } else {
                Some("numeric-order".to_string())
            };
            (pass, thresh, cat)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_metrics() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let m = compute_f10_metrics(&a, &a, 2).unwrap();
        assert_eq!(m.delta_max, 0.0);
        assert_eq!(m.epsilon_rel, 0.0);
        assert!((m.cos_theta - 1.0).abs() < 1e-6);
        assert_eq!(m.agreement, 100.0);
        assert_eq!(m.delta_ce, 0.0);
    }

    #[test]
    fn test_small_noise_pass_gate() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.00001f32, 1.99999, 3.00002, 3.99998];
        let m = compute_f10_metrics(&a, &b, 2).unwrap();
        assert!(m.delta_max <= 1e-3);
        assert!(m.epsilon_rel <= 1e-4);
        assert_eq!(m.agreement, 100.0);
    }

    #[test]
    fn test_mismatch_argmax() {
        let a = vec![1.0f32, 5.0];
        let b = vec![5.0f32, 1.0];
        let m = compute_f10_metrics(&a, &b, 2).unwrap();
        assert_eq!(m.agreement, 0.0);
        assert!(m.delta_max > 1.0);
    }

    #[test]
    fn test_gate_g_m2_1_pass() {
        let m = CompareMetrics {
            delta_max: 5e-4,
            epsilon_rel: 5e-5,
            cos_theta: 0.9999,
            agreement: 100.0,
            delta_ce: 0.0,
        };
        let (pass, _, cat) = evaluate_gate("G-M2-1", &m);
        assert!(pass);
        assert_eq!(cat, None);
    }

    #[test]
    fn test_gate_g_m3_1_pass() {
        let m = CompareMetrics {
            delta_max: 1e-7,
            epsilon_rel: 1e-7,
            cos_theta: 0.9999999,
            agreement: 100.0,
            delta_ce: 0.0,
        };
        let (pass, thresh, cat) = evaluate_gate("G-M3-1", &m);
        assert!(pass);
        assert_eq!(thresh, "delta_max <= 1e-3 && epsilon_rel <= 1e-4");
        assert_eq!(cat, None);
    }

    #[test]
    fn test_gate_g_m3_1_fail_categories() {
        // dtype-layout: delta_max > 1.0 or cos_theta < 0.90
        let m_layout = CompareMetrics {
            delta_max: 1.5,
            epsilon_rel: 0.5,
            cos_theta: 0.85,
            agreement: 50.0,
            delta_ce: 1.0,
        };
        let (pass, _, cat) = evaluate_gate("G-M3-1", &m_layout);
        assert!(!pass);
        assert_eq!(cat.as_deref(), Some("dtype-layout"));

        // bias-placement: 0.25 < delta_max <= 1.0
        let m_bias = CompareMetrics {
            delta_max: 0.45,
            epsilon_rel: 1e-2,
            cos_theta: 0.95,
            agreement: 80.0,
            delta_ce: 0.2,
        };
        let (pass, _, cat) = evaluate_gate("G-M3-1", &m_bias);
        assert!(!pass);
        assert_eq!(cat.as_deref(), Some("bias-placement"));

        // numeric-order: delta_max < 0.05 but > 1e-3
        let m_numeric = CompareMetrics {
            delta_max: 0.005,
            epsilon_rel: 0.001,
            cos_theta: 0.999,
            agreement: 99.0,
            delta_ce: 0.01,
        };
        let (pass, _, cat) = evaluate_gate("G-M3-1", &m_numeric);
        assert!(!pass);
        assert_eq!(cat.as_deref(), Some("numeric-order"));
    }

    #[test]
    fn test_gate_g_m4_1_loose_four_metrics() {
        // Semua 4 metrik gate M4 terpenuhi → PASS (FP32 vs FP32).
        let m = CompareMetrics {
            delta_max: 5e-3,
            epsilon_rel: 5e-5,
            cos_theta: 0.99999,
            agreement: 100.0,
            delta_ce: 0.01,
        };
        for gate in ["G-M4-1", "G-M5-1"] {
            let (pass, thresh, cat) = evaluate_gate(gate, &m);
            assert!(pass, "gate {gate} harus PASS");
            assert_eq!(
                thresh,
                "delta_max <= 1e-2 && epsilon_rel <= 1e-4 && agreement >= 99.9 && delta_ce <= 0.02"
            );
            assert_eq!(cat, None);
        }
    }

    #[test]
    fn test_gate_g_m4_1_rejects_bf16_noise_floor() {
        // Noise lantai kuantisasi BF16 murni (ε_rel ≈ 1,7e-3) wajib FAIL
        // di gate correctness — inilah alasan kandidat BF16 dilarang.
        let m = CompareMetrics {
            delta_max: 5e-3,
            epsilon_rel: 1.7e-3,
            cos_theta: 0.99999,
            agreement: 100.0,
            delta_ce: 0.005,
        };
        let (pass, _, _) = evaluate_gate("G-M4-1", &m);
        assert!(!pass, "noise lantai BF16 harus FAIL G-M4-1");
    }

    #[test]
    fn test_gate_g_m4_1_agreement_and_ce_gated() {
        // Agreement di bawah 99,9 → FAIL kategori argmax-mismatch.
        let m_low_a = CompareMetrics {
            delta_max: 5e-3,
            epsilon_rel: 5e-5,
            cos_theta: 0.99999,
            agreement: 99.0,
            delta_ce: 0.01,
        };
        let (pass, _, cat) = evaluate_gate("G-M4-1", &m_low_a);
        assert!(!pass);
        assert_eq!(cat.as_deref(), Some("argmax-mismatch"));

        // Δ_CE di atas 0,02 → FAIL meski 3 metrik lain lolos.
        let m_high_ce = CompareMetrics {
            delta_max: 5e-3,
            epsilon_rel: 5e-5,
            cos_theta: 0.99999,
            agreement: 100.0,
            delta_ce: 0.05,
        };
        let (pass, _, _) = evaluate_gate("G-M4-1", &m_high_ce);
        assert!(!pass);
    }

    #[test]
    fn test_gate_g_m4_1_delta_bands() {
        // Pita Δ M4 (Tier-2 screening F10-A): layout > bias > rope > numeric.
        let mk = |delta_max: f64, cos_theta: f64, epsilon_rel: f64| CompareMetrics {
            delta_max,
            epsilon_rel,
            cos_theta,
            agreement: 100.0,
            delta_ce: 0.01,
        };
        for (dm, ct, er, expect) in [
            (1.5, 0.85, 5e-5, "dtype-layout"), // tensor-mapping alias
            (1.5, 0.99, 5e-5, "dtype-layout"), // Δ saja cukup
            (0.5, 0.95, 5e-5, "bias-placement"),
            (0.10, 0.999, 5e-5, "rope-style"),
            (0.005, 0.9999, 1e-3, "numeric-order"), // FAIL lewat ε_rel
        ] {
            let (pass, _, cat) = evaluate_gate("G-M4-1", &mk(dm, ct, er));
            assert!(!pass, "Δ={dm} harus FAIL");
            assert_eq!(cat.as_deref(), Some(expect), "Δ={dm}");
        }
        // Pita numeric-order: Δ kecil tapi FAIL lewat metrik lain (di sini ε_rel).
        let (pass, _, cat) = evaluate_gate(
            "G-M4-1",
            &CompareMetrics {
                delta_max: 0.005,
                epsilon_rel: 1e-3,
                cos_theta: 0.9999,
                agreement: 100.0,
                delta_ce: 0.01,
            },
        );
        assert!(!pass);
        assert_eq!(cat.as_deref(), Some("numeric-order"));
        // Agreement mengalahkan pita: A gagal → argmax-mismatch walau Δ besar.
        let (pass, _, cat) = evaluate_gate(
            "G-M4-1",
            &CompareMetrics {
                delta_max: 2.0,
                epsilon_rel: 0.5,
                cos_theta: 0.8,
                agreement: 98.0,
                delta_ce: 0.5,
            },
        );
        assert!(!pass);
        assert_eq!(cat.as_deref(), Some("argmax-mismatch"));
    }

    #[test]
    fn test_routing_check_match_and_mismatch() {
        let oracle = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        // Same elements in different order
        let cand_same = vec![vec![4, 3, 2, 1], vec![8, 7, 6, 5]];
        assert!(check_routing_match(&oracle, &cand_same));

        // Different element
        let cand_diff = vec![vec![1, 2, 3, 99], vec![5, 6, 7, 8]];
        assert!(!check_routing_match(&oracle, &cand_diff));

        // Different length
        let cand_len = vec![vec![1, 2, 3, 4]];
        assert!(!check_routing_match(&oracle, &cand_len));
    }
}
