// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! Integration Test Specification M1 (7 kasus normatif):
//! 1. Happy path: status=success, logits MATCH oracle via Rust compare (G-M1-1)
//! 2. Token validation: error TOKEN_INVALID, exit=2
//! 3. Weight load failure: (a) reader error M0 propagasi, (b) WEIGHT_LOAD_FAILED semantik
//! 4. Oracle mismatch: status=MISMATCH, verdict=FAIL, exit=1
//! 5. Atomic write failure: error OUTPUT_WRITE_FAILED, exit=2, no partial file
//! 6. Memory boundary: peak <= 3.5 GiB, telemetri caps <= 64 MiB
//! 7. Config contract: error CONFIG_ERROR pada config tanpa rms_norm_eps, exit=2

use serde_json::Value;
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

fn get_kimo_bin() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let kimo = root.join("kimo");
    if !kimo.exists() {
        let status = Command::new("pixi")
            .current_dir(&root)
            .args([
                "run",
                "bash",
                "-c",
                "PATH=\"/usr/bin:$PATH\" mojo build -I src src/main.mojo -o kimo",
            ])
            .status()
            .expect("Failed to build kimo with pixi");
        assert!(status.success(), "Building kimo failed");
    }
    kimo
}

fn get_root_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_1_happy_path_g_m1_1() {
    let root = get_root_dir();
    let kimo = get_kimo_bin();
    let tmp_dir = TempDir::new("m1_test");
    let out_bin = tmp_dir.path().join("logits_mojo.bin");
    let tokens_path = root.join("fixtures/m1/tokens.json");
    let model_dir = root.join("fixtures/m1");
    let ref_bin = root.join("fixtures/m1/logits_ref.bin");

    // 1. Run kimo head
    let output = Command::new(&kimo)
        .current_dir(&root)
        .arg("head")
        .arg(&tokens_path)
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--workdir")
        .arg(tmp_dir.path())
        .arg("--output")
        .arg(&out_bin)
        .output()
        .expect("Failed to execute kimo head");

    assert_eq!(
        output.status.code(),
        Some(0),
        "kimo head failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let report: Value = serde_json::from_str(&stdout_str).expect("Invalid JSON from kimo head");
    assert_eq!(report["status"], "success");
    assert_eq!(report["num_prompts"], 3);
    assert_eq!(report["tokens_per_prompt"], 16);
    assert_eq!(report["num_tokens_total"], 48);
    assert_eq!(report["vocab_size"], 512);

    // 2. Run kimo-tools compare (G-M1-1)
    let compare_bin = PathBuf::from(env!("CARGO_BIN_EXE_kimo-tools"));
    let cmp_output = Command::new(&compare_bin)
        .current_dir(&root)
        .arg("compare")
        .arg(&ref_bin)
        .arg(&out_bin)
        .arg("--gate")
        .arg("G-M1-1")
        .output()
        .expect("Failed to execute kimo-tools compare");

    assert_eq!(
        cmp_output.status.code(),
        Some(0),
        "kimo-tools compare failed: {}",
        String::from_utf8_lossy(&cmp_output.stderr)
    );

    let cmp_report: Value =
        serde_json::from_slice(&cmp_output.stdout).expect("Invalid compare report JSON");
    assert_eq!(cmp_report["status"], "MATCH");
    assert_eq!(cmp_report["verdict"], "PASS");

    let metrics = &cmp_report["metrics"];
    assert!(metrics["delta_max"].as_f64().unwrap() <= 1e-3);
    assert!(metrics["epsilon_rel"].as_f64().unwrap() <= 1e-4);
    assert_eq!(metrics["agreement"].as_f64().unwrap(), 100.0);
}

#[test]
fn test_2_token_validation() {
    let root = get_root_dir();
    let kimo = get_kimo_bin();
    let tmp_dir = TempDir::new("m1_test");
    let bad_tokens = tmp_dir.path().join("bad_tokens.json");
    let model_dir = root.join("fixtures/m1");

    // Token out of vocab bounds (> 511)
    let mut tokens: Vec<Vec<i32>> = vec![(0..16).collect(), (16..32).collect(), (32..48).collect()];
    tokens[1][5] = 999;
    fs::write(&bad_tokens, serde_json::to_string(&tokens).unwrap()).unwrap();

    let output = Command::new(&kimo)
        .current_dir(&root)
        .arg("head")
        .arg(&bad_tokens)
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--workdir")
        .arg(tmp_dir.path())
        .output()
        .expect("Failed to execute kimo head");

    assert_eq!(output.status.code(), Some(2));
    let err_str = String::from_utf8_lossy(&output.stderr);
    let err: Value = serde_json::from_str(&err_str).expect("Invalid error JSON");
    assert_eq!(err["error_type"], "TOKEN_INVALID");
    assert_eq!(err["prompt_idx"], 1);
    assert_eq!(err["token_pos"], 5);
}

#[test]
fn test_3_weight_load_failure() {
    let root = get_root_dir();
    let kimo = get_kimo_bin();
    let tmp_dir = TempDir::new("m1_test");
    let tokens_path = root.join("fixtures/m1/tokens.json");

    // (a) Missing shard on disk -> FILE_NOT_FOUND (M0 propagated)
    let output_a = Command::new(&kimo)
        .current_dir(&root)
        .arg("head")
        .arg(&tokens_path)
        .arg(root.join("fixtures/m1/fixture-00001-of-00003.safetensors"))
        .arg(root.join("fixtures/m1/nonexistent.safetensors"))
        .arg("--workdir")
        .arg(tmp_dir.path())
        .output()
        .expect("Failed to execute kimo head");

    assert_eq!(output_a.status.code(), Some(2));
    let err_a: Value = serde_json::from_slice(&output_a.stderr).expect("Invalid error JSON");
    assert_eq!(err_a["error_type"], "FILE_NOT_FOUND");

    // (b) Shards valid but missing required norm weight -> WEIGHT_LOAD_FAILED
    let output_b = Command::new(&kimo)
        .current_dir(&root)
        .arg("head")
        .arg(&tokens_path)
        .arg(root.join("fixtures/m1/fixture-00001-of-00003.safetensors"))
        .arg("--workdir")
        .arg(tmp_dir.path())
        .output()
        .expect("Failed to execute kimo head");

    assert_eq!(output_b.status.code(), Some(2));
    let err_b: Value = serde_json::from_slice(&output_b.stderr).expect("Invalid error JSON");
    assert_eq!(err_b["error_type"], "WEIGHT_LOAD_FAILED");
    let missing = err_b["missing_tensors"]
        .as_array()
        .expect("missing_tensors not array");
    assert!(missing.iter().any(|v| v == "model.norm.weight"));
}

#[test]
fn test_4_oracle_mismatch() {
    let root = get_root_dir();
    let tmp_dir = TempDir::new("m1_test");
    let ref_bin = root.join("fixtures/m1/logits_ref.bin");
    let mut cand_data = fs::read(&ref_bin).expect("Failed to read ref_bin");

    // Mutate candidate bytes to trigger mismatch (mutate exponent in byte 3)
    cand_data[3] ^= 0x7F;
    let cand_bin = tmp_dir.path().join("cand_mutated.bin");
    fs::write(&cand_bin, &cand_data).unwrap();

    let compare_bin = PathBuf::from(env!("CARGO_BIN_EXE_kimo-tools"));
    let output = Command::new(&compare_bin)
        .current_dir(&root)
        .arg("compare")
        .arg(&ref_bin)
        .arg(&cand_bin)
        .arg("--gate")
        .arg("G-M1-1")
        .output()
        .expect("Failed to execute kimo-tools compare");

    assert_eq!(output.status.code(), Some(1));
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("Invalid compare report JSON");
    assert_eq!(report["status"], "MISMATCH");
    assert_eq!(report["verdict"], "FAIL");
    assert!(report.get("fail_category").is_some());
}

#[test]
fn test_5_atomic_write_failure() {
    let root = get_root_dir();
    let kimo = get_kimo_bin();
    let tmp_dir = TempDir::new("m1_test");
    let ro_dir = tmp_dir.path().join("readonly_dir");
    fs::create_dir(&ro_dir).unwrap();

    // Set read-only permissions
    let mut perms = fs::metadata(&ro_dir).unwrap().permissions();
    perms.set_readonly(true);
    fs::set_permissions(&ro_dir, perms).unwrap();

    let out_bin = ro_dir.join("logits.bin");
    let tokens_path = root.join("fixtures/m1/tokens.json");
    let model_dir = root.join("fixtures/m1");

    let output = Command::new(&kimo)
        .current_dir(&root)
        .arg("head")
        .arg(&tokens_path)
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--workdir")
        .arg(tmp_dir.path())
        .arg("--output")
        .arg(&out_bin)
        .output()
        .expect("Failed to execute kimo head");

    assert_eq!(output.status.code(), Some(2));
    let err: Value = serde_json::from_slice(&output.stderr).expect("Invalid error JSON");
    assert_eq!(err["error_type"], "OUTPUT_WRITE_FAILED");

    // Restore permissions for cleanup & verify no partial file
    let mut restore_perms = fs::metadata(&ro_dir).unwrap().permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        restore_perms.set_mode(0o777);
    }
    let _ = fs::set_permissions(&ro_dir, restore_perms);

    assert!(!out_bin.exists(), "Leftover partial file found!");
    assert!(
        !ro_dir.join("logits.bin.tmp.bin").exists(),
        "Leftover tmp file found!"
    );
}

#[test]
fn test_6_memory_boundary_and_caps() {
    let root = get_root_dir();
    let kimo = get_kimo_bin();
    let tmp_dir = TempDir::new("m1_test_mem");
    let out_bin = tmp_dir.path().join("logits.bin");
    let tokens_path = root.join("fixtures/m1/tokens.json");
    let model_dir = root.join("fixtures/m1");

    let output = Command::new(&kimo)
        .current_dir(&root)
        .arg("head")
        .arg(&tokens_path)
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--workdir")
        .arg(tmp_dir.path())
        .arg("--output")
        .arg(&out_bin)
        .output()
        .expect("Failed to execute kimo head");

    assert_eq!(output.status.code(), Some(0));
    let report: Value = serde_json::from_slice(&output.stdout).expect("Invalid JSON report");

    let mem = &report["memory"];
    let conv_buf = mem["conversion_buffer_bytes"].as_u64().unwrap();
    let src_buf = mem["source_buffer_bytes"].as_u64().unwrap();
    let res_target = mem["resident_target_bytes"].as_u64().unwrap();
    let vmhwm = mem["vmhwm_bytes"].as_u64().unwrap();

    // Caps telemetri: conversion <= 64 MiB, source <= 64 MiB
    assert!(
        conv_buf <= 64 * 1024 * 1024,
        "conversion buffer {} exceeds 64 MiB",
        conv_buf
    );
    assert!(
        src_buf <= 64 * 1024 * 1024,
        "source buffer {} exceeds 64 MiB",
        src_buf
    );
    assert_eq!(res_target, 262400); // 512*64*4 * 2 + 64*4
    assert!(vmhwm <= 3758096384, "VmHWM {} exceeds 3.5 GiB", vmhwm); // G-M1-2
}

#[test]
fn test_7_config_contract() {
    let root = get_root_dir();
    let kimo = get_kimo_bin();
    let tmp_dir = TempDir::new("m1_test_cfg");
    let tokens_path = root.join("fixtures/m1/tokens.json");
    let m0_dir = root.join("fixtures/m0"); // lacks rms_norm_eps

    let output = Command::new(&kimo)
        .current_dir(&root)
        .arg("head")
        .arg(&tokens_path)
        .arg("--model-dir")
        .arg(&m0_dir)
        .arg("--workdir")
        .arg(tmp_dir.path())
        .output()
        .expect("Failed to execute kimo head");

    assert_eq!(output.status.code(), Some(2));
    let err: Value = serde_json::from_slice(&output.stderr).expect("Invalid error JSON");
    assert_eq!(err["error_type"], "CONFIG_ERROR");
    assert_eq!(err["stage"], "config");
}
