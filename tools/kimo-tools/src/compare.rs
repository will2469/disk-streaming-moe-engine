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

use serde::Serialize;
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
    let mut run_id = "M1-F10-001".to_string();

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
            "--vocab" | "--vocab-size" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        vocab_size = v;
                    }
                }
            }
            "--run-id" => {
                i += 1;
                if i < args.len() {
                    run_id = args[i].clone();
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
            "pakai: kimo-tools compare <ref.bin> <cand.bin> [--gate G-M1-1] [--vocab <N>]",
        );
    }

    let ref_floats = match read_floats(&ref_path) {
        Ok(v) => v,
        Err((code, detail)) => return emit_error(&code, &detail),
    };

    let cand_floats = match read_floats(&cand_path) {
        Ok(v) => v,
        Err((code, detail)) => return emit_error(&code, &detail),
    };

    let metrics = match compute_f10_metrics(&ref_floats, &cand_floats, vocab_size) {
        Ok(m) => m,
        Err(e) => return emit_error("LAYOUT_MISMATCH", &e),
    };

    // Evaluate gate
    let (is_pass, threshold_str, fail_cat) = match gate.as_str() {
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
    };

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
}
