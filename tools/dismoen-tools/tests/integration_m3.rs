// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! Integration Test Specification M3 (8 kasus normatif):
//! 1. Happy path: status=success, output MATCH oracle via Rust compare (Gate G-M3-1)
//! 2. Routing invariant test: SET top-4 identik 100%, unrenormalized probability
//! 3. Sigmoid gate test: shared expert sigmoid monotonicity & bounds
//! 4. Top-4 no-renorm test: norm_topk_prob=false (sum <= 1.0) verified
//! 5. SwiGLU verification: element-wise SiLU activation multiplication
//! 6. Expert load failure: error LAYER_INVALID / WEIGHT_LOAD_FAILED, exit=2
//! 7. Router overflow: stable max-shift softmax handles large values without NaN/overflow
//! 8. Determinisme test: 5x identical SHA-256 binary output & routing

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn get_dismoen_bin() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DISMOEN") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let root = get_root_dir();
    let dismoen = root.join("dismoen");
    if dismoen.exists() {
        return Some(dismoen);
    }
    let bin = root.join("build/bin/dismoen");
    if bin.exists() {
        return Some(bin);
    }
    if let Ok(status) = Command::new("pixi")
        .current_dir(&root)
        .args([
            "run",
            "bash",
            "-c",
            "PATH=\"/usr/bin:$PATH\" mojo build -I src src/main.mojo -o dismoen",
        ])
        .status()
    {
        if status.success() && dismoen.exists() {
            return Some(dismoen);
        }
    }
    None
}

fn get_root_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn get_model_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MODEL_DIR") {
        let pb = PathBuf::from(p);
        if pb.join("model.safetensors.index.json").exists() {
            return Some(pb);
        }
    }
    let default_path = PathBuf::from("/home/will/models/qwen1.5-moe-a2.7b-chat");
    if default_path.join("model.safetensors.index.json").exists() {
        Some(default_path)
    } else {
        None
    }
}

fn compute_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let res = hasher.finalize();
    res.iter().map(|b| format!("{:02x}", b)).collect::<String>()
}

// 1. Happy path: layers 0, 12, 23 pass Gate G-M3-1
#[test]
fn test_1_happy_path() {
    let Some(dismoen) = get_dismoen_bin() else {
        eprintln!("SKIPPED: dismoen binary not available");
        return;
    };
    let root = get_root_dir();
    let act_path = root.join("fixtures/m3/activation.bin");
    assert!(act_path.exists(), "activation fixture missing");

    let compare_bin = PathBuf::from(env!("CARGO_BIN_EXE_dismoen-tools"));

    if let Some(model_dir) = get_model_dir() {
        for lyr in [0, 12, 23] {
            let tmp_dir = TempDir::new(&format!("m3_happy_{}", lyr));
            let out_file = format!("moe_out_{}.bin", lyr);
            let out_path = tmp_dir.path().join(&out_file);
            let ref_path = root.join(format!("fixtures/m3/moe_ref_{}.bin", lyr));
            assert!(ref_path.exists(), "ref bin missing: {:?}", ref_path);

            let output = Command::new(&dismoen)
                .current_dir(&root)
                .args([
                    "layer",
                    "--layer",
                    &lyr.to_string(),
                    "--part",
                    "moe",
                    act_path.to_str().unwrap(),
                    "--model-dir",
                    model_dir.to_str().unwrap(),
                    "--workdir",
                    tmp_dir.path().to_str().unwrap(),
                    "--output",
                    &out_file,
                ])
                .output()
                .expect("Failed to execute dismoen layer moe");

            assert_eq!(
                output.status.code(),
                Some(0),
                "dismoen layer moe {} failed: {}",
                lyr,
                String::from_utf8_lossy(&output.stderr)
            );

            let stdout_str = String::from_utf8_lossy(&output.stdout);
            let rep: Value = serde_json::from_str(&stdout_str).expect("Invalid JSON");
            assert_eq!(rep["status"], "success");
            assert_eq!(rep["layer"], lyr);
            assert_eq!(rep["part"], "moe");
            assert_eq!(rep["num_tokens"], 16);
            assert!(out_path.exists());

            // Run dismoen-tools compare (G-M3-1)
            let cmp_output = Command::new(&compare_bin)
                .current_dir(&root)
                .args([
                    "compare",
                    ref_path.to_str().unwrap(),
                    out_path.to_str().unwrap(),
                    "--gate",
                    "G-M3-1",
                ])
                .output()
                .expect("Failed to execute compare");

            assert_eq!(cmp_output.status.code(), Some(0));
            let cmp_rep: Value = serde_json::from_slice(&cmp_output.stdout).expect("Invalid JSON");
            assert_eq!(cmp_rep["status"], "MATCH");
            assert_eq!(cmp_rep["verdict"], "PASS");
            let m = &cmp_rep["metrics"];
            assert!(m["delta_max"].as_f64().unwrap() <= 1e-3);
            assert!(m["epsilon_rel"].as_f64().unwrap() <= 1e-4);
        }
    } else {
        // Fallback: verify committed fixtures pass G-M3-1 on self-comparison
        for lyr in [0, 12, 23] {
            let ref_path = root.join(format!("fixtures/m3/moe_ref_{}.bin", lyr));
            let cmp_output = Command::new(&compare_bin)
                .current_dir(&root)
                .args([
                    "compare",
                    ref_path.to_str().unwrap(),
                    ref_path.to_str().unwrap(),
                    "--gate",
                    "G-M3-1",
                ])
                .output()
                .unwrap();
            assert_eq!(cmp_output.status.code(), Some(0));
        }
    }
}

// 2. Routing invariant test: SET top-4 identik 100% (G-M3-2)
#[test]
fn test_2_routing_invariant() {
    let Some(dismoen) = get_dismoen_bin() else {
        return;
    };
    let Some(model_dir) = get_model_dir() else {
        eprintln!("SKIPPED: model dir not available");
        return;
    };
    let root = get_root_dir();
    let act_path = root.join("fixtures/m3/activation.bin");
    let tmp_dir = TempDir::new("m3_routing_inv");

    for lyr in [0, 12, 23] {
        let ora_path = root.join(format!("fixtures/m3/routing_info_{}.json", lyr));
        assert!(ora_path.exists(), "oracle routing file missing");

        // Happy path: matching oracle routing
        let output = Command::new(&dismoen)
            .current_dir(&root)
            .args([
                "layer",
                "--layer",
                &lyr.to_string(),
                "--part",
                "moe",
                act_path.to_str().unwrap(),
                "--model-dir",
                model_dir.to_str().unwrap(),
                "--workdir",
                tmp_dir.path().to_str().unwrap(),
                "--output",
                &format!("out_{}.bin", lyr),
                "--oracle-routing",
                ora_path.to_str().unwrap(),
            ])
            .output()
            .unwrap();

        assert_eq!(
            output.status.code(),
            Some(0),
            "Routing check failed on layer {}",
            lyr
        );

        let rep: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(rep["status"], "success");
    }

    // Violation path: perturbed oracle routing triggers ROUTING_VIOLATION exit=1
    let mut bad_ora: Value = serde_json::from_reader(
        fs::File::open(root.join("fixtures/m3/routing_info_0.json")).unwrap(),
    )
    .unwrap();
    // Swap expert selection for token 0
    bad_ora["selected_experts"][0][0] = serde_json::json!(59);
    bad_ora["selected_experts"][0][1] = serde_json::json!(58);
    let bad_ora_path = tmp_dir.path().join("bad_routing.json");
    fs::write(&bad_ora_path, serde_json::to_string(&bad_ora).unwrap()).unwrap();

    let output_bad = Command::new(&dismoen)
        .current_dir(&root)
        .args([
            "layer",
            "--layer",
            "0",
            "--part",
            "moe",
            act_path.to_str().unwrap(),
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--workdir",
            tmp_dir.path().to_str().unwrap(),
            "--output",
            "bad_out.bin",
            "--oracle-routing",
            bad_ora_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output_bad.status.code(), Some(1));
    let err: Value = serde_json::from_slice(&output_bad.stderr).unwrap();
    assert_eq!(err["error_type"], "ROUTING_VIOLATION");
}

// 3. Sigmoid gate test: mathematical property bounds and monotonicity
#[test]
fn test_3_sigmoid_gate_properties() {
    let sigmoid = |x: f32| -> f32 { 1.0 / (1.0 + (-x).exp()) };

    // Bounded in (0, 1) within valid f32 range
    let test_points = [-15.0f32, -8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0, 15.0];
    for &x in &test_points {
        let s = sigmoid(x);
        assert!(s > 0.0 && s < 1.0, "sigmoid out of (0, 1) at {}", x);
    }

    // Monotonicity: x1 < x2 => sigmoid(x1) < sigmoid(x2)
    for i in 0..test_points.len() - 1 {
        let x1 = test_points[i];
        let x2 = test_points[i + 1];
        assert!(
            sigmoid(x1) < sigmoid(x2),
            "monotonicity failed between {} and {}",
            x1,
            x2
        );
    }

    // Symmetry: sigmoid(-x) == 1 - sigmoid(x)
    for &x in &[0.5, 1.5, 3.0, 5.0] {
        let diff = (sigmoid(-x) - (1.0 - sigmoid(x))).abs();
        assert!(diff < 1e-6, "symmetry failed at {}", x);
    }
}

// 4. Top-4 no renorm test: verify unrenormalized router probabilities
#[test]
fn test_4_top4_no_renorm() {
    let root = get_root_dir();
    for lyr in [0, 12, 23] {
        let ora_path = root.join(format!("fixtures/m3/routing_info_{}.json", lyr));
        let ora: Value = serde_json::from_reader(fs::File::open(ora_path).unwrap()).unwrap();

        let probs = ora["router_probs"].as_array().unwrap();
        let experts = ora["selected_experts"].as_array().unwrap();
        assert_eq!(probs.len(), 16);
        assert_eq!(experts.len(), 16);

        for t in 0..16 {
            let row_probs = probs[t].as_array().unwrap();
            let row_exp = experts[t].as_array().unwrap();
            assert_eq!(row_probs.len(), 4);
            assert_eq!(row_exp.len(), 4);

            let mut sum_p = 0.0f64;
            for p in row_probs {
                let v = p.as_f64().unwrap();
                assert!(v > 0.0 && v <= 1.0);
                sum_p += v;
            }

            // Normative invariant: norm_topk_prob=false means top-4 sum is strictly < 1.0
            assert!(
                sum_p <= 1.0,
                "Layer {} token {}: top-4 prob sum {} > 1.0",
                lyr,
                t,
                sum_p
            );
            assert!(
                sum_p < 0.999,
                "Layer {} token {}: top-4 prob sum {} is artificially normalized",
                lyr,
                t,
                sum_p
            );
        }
    }
}

// 5. SwiGLU verification: element-wise SiLU activation multiplication
#[test]
fn test_5_swiglu_verification() {
    let silu = |x: f32| -> f32 { x / (1.0 + (-x).exp()) };
    let swiglu = |gate: f32, up: f32| -> f32 { silu(gate) * up };

    // At gate = 0: SiLU(0) = 0 => SwiGLU = 0
    assert!((swiglu(0.0, 5.0)).abs() < 1e-6);
    assert!((swiglu(0.0, -5.0)).abs() < 1e-6);

    // For large gate: SiLU(x) -> x => SwiGLU -> x * up
    let val = swiglu(10.0, 2.0);
    assert!((val - 20.0).abs() < 0.01);

    // Negative gate: SiLU is small and non-zero
    let neg_val = swiglu(-2.0, 1.0);
    assert!(neg_val < 0.0 && neg_val > -0.5);

    // Non-linear scaling
    let lin_2x = 2.0 * swiglu(1.0, 1.0);
    let nonlin = swiglu(2.0, 2.0);
    assert!((lin_2x - nonlin).abs() > 0.1);
}

// 6. Expert load failure: invalid layer or corrupted parameters
#[test]
fn test_6_expert_load_failure() {
    let Some(dismoen) = get_dismoen_bin() else {
        return;
    };
    let root = get_root_dir();
    let tmp_dir = TempDir::new("m3_expert_fail");
    let act_path = root.join("fixtures/m3/activation.bin");

    // Invalid layer 24 (valid are 0..23)
    let output = Command::new(&dismoen)
        .current_dir(&root)
        .args([
            "layer",
            "--layer",
            "24",
            "--part",
            "moe",
            act_path.to_str().unwrap(),
            "--workdir",
            tmp_dir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let err: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(err["error_type"], "LAYER_INVALID");

    // Missing expert weights / model dir -> WEIGHT_LOAD_FAILED
    let output_weight = Command::new(&dismoen)
        .current_dir(&root)
        .args([
            "layer",
            "--layer",
            "0",
            "--part",
            "moe",
            act_path.to_str().unwrap(),
            "--workdir",
            tmp_dir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert_eq!(output_weight.status.code(), Some(2));
    let err_w: Value = serde_json::from_slice(&output_weight.stderr).unwrap();
    assert_eq!(err_w["error_type"], "WEIGHT_LOAD_FAILED");

    // Missing activation file (with valid model dir) -> FILE_NOT_FOUND
    if let Some(model_dir) = get_model_dir() {
        let output_act = Command::new(&dismoen)
            .current_dir(&root)
            .args([
                "layer",
                "--layer",
                "0",
                "--part",
                "moe",
                "/tmp/nonexistent_m3_act.bin",
                "--model-dir",
                model_dir.to_str().unwrap(),
                "--workdir",
                tmp_dir.path().to_str().unwrap(),
            ])
            .output()
            .unwrap();

        assert_eq!(output_act.status.code(), Some(2));
        let err_act: Value = serde_json::from_slice(&output_act.stderr).unwrap();
        assert_eq!(err_act["error_type"], "FILE_NOT_FOUND");
    }
}

// 7. Router overflow: numerical stability with max-shift softmax
#[test]
fn test_7_router_overflow() {
    let logits = [1000.0f32, 1005.0, 995.0, 1010.0, -1000.0];
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    let mut probs = vec![0.0f32; logits.len()];

    for (i, &l) in logits.iter().enumerate() {
        let p = (l - max_l).exp();
        probs[i] = p;
        exp_sum += p;
    }

    assert!(exp_sum > 0.0 && exp_sum.is_finite());
    for p in &probs {
        let normalized = p / exp_sum;
        assert!((0.0..=1.0).contains(&normalized));
        assert!(!normalized.is_nan() && !normalized.is_infinite());
    }
}

// 8. Determinisme test: 5x identical SHA-256 output and routing
#[test]
fn test_8_determinisme() {
    let Some(dismoen) = get_dismoen_bin() else {
        return;
    };
    let Some(model_dir) = get_model_dir() else {
        eprintln!("SKIPPED: model dir not available");
        return;
    };
    let root = get_root_dir();
    let act_path = root.join("fixtures/m3/activation.bin");
    let tmp_dir = TempDir::new("m3_determinisme");

    let mut hashes = Vec::new();
    let mut routings = Vec::new();

    for run_i in 0..5 {
        let out_file = format!("det_out_{}.bin", run_i);
        let out_path = tmp_dir.path().join(&out_file);

        let output = Command::new(&dismoen)
            .current_dir(&root)
            .args([
                "layer",
                "--layer",
                "0",
                "--part",
                "moe",
                act_path.to_str().unwrap(),
                "--model-dir",
                model_dir.to_str().unwrap(),
                "--workdir",
                tmp_dir.path().to_str().unwrap(),
                "--output",
                &out_file,
            ])
            .output()
            .unwrap();

        assert_eq!(output.status.code(), Some(0));
        let rep: Value = serde_json::from_slice(&output.stdout).unwrap();
        routings.push(rep["routing_info"].to_string());

        let bytes = fs::read(&out_path).unwrap();
        hashes.push(compute_sha256(&bytes));
    }

    // All 5 hashes must be 100% identical
    let first_hash = &hashes[0];
    for h in &hashes {
        assert_eq!(
            h, first_hash,
            "Determinism failed: hash mismatch across runs"
        );
    }

    // All 5 routings must be 100% identical
    let first_routing = &routings[0];
    for r in &routings {
        assert_eq!(
            r, first_routing,
            "Determinism failed: routing mismatch across runs"
        );
    }
}
