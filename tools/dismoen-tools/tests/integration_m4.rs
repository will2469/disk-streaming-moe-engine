// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! Integration Test Specification M4-W4 (Verdict A -> N -> S & Gate G-M4-1):
//! 1. test_ans_verdict_short_circuit: Hard FAIL (router-selection) overrides numeric PASS
//! 2. test_agreement_discrete_gating_n80: 80/80 (100%) PASS, 79/80 (98.75%) FAIL argmax-mismatch
//! 3. test_routing_tier1_set_equality: Order-insensitive SET equality (PASS), set mismatch (FAIL)
//! 4. test_delta_bands_hierarchy: dtype-layout > bias-placement > rope-style > numeric-order
//! 5. test_numeric_order_pass_loose: Small numeric differences PASS loose gate G-M4-1
//! 6. test_bf16_noise_floor_rejected_in_g_m4_1: BF16 quantization noise floor rejected in G-M4-1
//! 7. test_golden_prompt1_oracle_identity: Identity compare on real golden prompt1 oracle artifact

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

fn get_root_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn get_compare_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_dismoen-tools"))
}

fn write_f32_le_file(path: &Path, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &f in data {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

// 1. Short-circuit: jika routing dump mismatch, verdict langsung FAIL ("router-selection")
// meskipun metrik numerik sempurna identik.
#[test]
fn test_ans_verdict_short_circuit() {
    let tmp = TempDir::new("m4_ans_sc");
    let compare_bin = get_compare_bin();

    let vocab_size = 512;
    let n_tokens = 16;
    let total = n_tokens * vocab_size;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();

    let ref_path = tmp.path().join("ref.bin");
    let cand_path = tmp.path().join("cand.bin");
    write_f32_le_file(&ref_path, &data);
    write_f32_le_file(&cand_path, &data);

    // Routing oracle: token 0 memilih expert [0, 1, 2, 3]
    let mut o_routing = Vec::new();
    for _ in 0..n_tokens {
        o_routing.push(vec![0, 1, 2, 3]);
    }
    let o_json = serde_json::json!({ "selected_experts": o_routing });
    let o_path = tmp.path().join("oracle_routing.json");
    fs::write(&o_path, serde_json::to_string(&o_json).unwrap()).unwrap();

    // Routing candidate mismatch: token 0 memilih expert [0, 1, 2, 4]
    let mut c_routing = o_routing.clone();
    c_routing[0] = vec![0, 1, 2, 4];
    let c_json = serde_json::json!({ "selected_experts": c_routing });
    let c_path = tmp.path().join("cand_routing.json");
    fs::write(&c_path, serde_json::to_string(&c_json).unwrap()).unwrap();

    let out = Command::new(&compare_bin)
        .args([
            "compare",
            "--ref",
            ref_path.to_str().unwrap(),
            "--cand",
            cand_path.to_str().unwrap(),
            "--gate",
            "G-M4-1",
            "--dim",
            &vocab_size.to_string(),
            "--oracle-routing",
            o_path.to_str().unwrap(),
            "--cand-routing",
            c_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run dismoen-tools compare");

    assert_eq!(
        out.status.code(),
        Some(1),
        "routing mismatch must exit 1 (MISMATCH)"
    );
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["status"], "MISMATCH");
    assert_eq!(v["verdict"], "FAIL");
    assert_eq!(v["fail_category"], "router-selection");
}

// 2. Definisi A (Argmax Agreement) & sifat diskrit n=80:
// 80/80 (100.0%) PASS; 1 token flip (79/80 = 98.75%) FAIL argmax-mismatch.
#[test]
fn test_agreement_discrete_gating_n80() {
    let tmp = TempDir::new("m4_agreement_n80");
    let compare_bin = get_compare_bin();

    let vocab_size = 512;
    let n_tokens = 80;
    let total = n_tokens * vocab_size;

    // Inisialisasi: setiap baris memiliki nilai argmax di indeks 10
    let mut ref_data = vec![0.0f32; total];
    for t in 0..n_tokens {
        ref_data[t * vocab_size + 10] = 5.0;
    }
    let ref_path = tmp.path().join("ref.bin");
    write_f32_le_file(&ref_path, &ref_data);

    // Kasus 2a: Cand identik 80/80 (100.0%) -> PASS
    let cand_path_pass = tmp.path().join("cand_pass.bin");
    write_f32_le_file(&cand_path_pass, &ref_data);

    let out_pass = Command::new(&compare_bin)
        .args([
            "compare",
            "--ref",
            ref_path.to_str().unwrap(),
            "--cand",
            cand_path_pass.to_str().unwrap(),
            "--gate",
            "G-M4-1",
            "--dim",
            &vocab_size.to_string(),
        ])
        .output()
        .expect("run compare");

    assert_eq!(out_pass.status.code(), Some(0));
    let v_pass: Value = serde_json::from_slice(&out_pass.stdout).unwrap();
    assert_eq!(v_pass["status"], "MATCH");
    assert_eq!(v_pass["verdict"], "PASS");
    assert_eq!(v_pass["metrics"]["agreement"], 100.0);

    // Kasus 2b: Ubah 1 token agar argmax bergeser ke indeks 20
    // Agreement = 79 / 80 = 98.75% < 99.9% -> FAIL ("argmax-mismatch")
    let mut cand_data_fail = ref_data.clone();
    cand_data_fail[20] = 10.0;
    let cand_path_fail = tmp.path().join("cand_fail.bin");
    write_f32_le_file(&cand_path_fail, &cand_data_fail);

    let out_fail = Command::new(&compare_bin)
        .args([
            "compare",
            "--ref",
            ref_path.to_str().unwrap(),
            "--cand",
            cand_path_fail.to_str().unwrap(),
            "--gate",
            "G-M4-1",
            "--dim",
            &vocab_size.to_string(),
        ])
        .output()
        .expect("run compare");

    assert_eq!(out_fail.status.code(), Some(1));
    let v_fail: Value = serde_json::from_slice(&out_fail.stdout).unwrap();
    assert_eq!(v_fail["status"], "MISMATCH");
    assert_eq!(v_fail["verdict"], "FAIL");
    assert_eq!(v_fail["fail_category"], "argmax-mismatch");
    let a_val = v_fail["metrics"]["agreement"].as_f64().unwrap();
    assert!((a_val - 98.75).abs() < 1e-4, "agreement must be 98.75%");
}

// 3. Kesetaraan SET Tier-1: order-insensitive ([1,2,3,4] vs [4,3,2,1]) = PASS
#[test]
fn test_routing_tier1_set_equality() {
    let tmp = TempDir::new("m4_routing_set");
    let compare_bin = get_compare_bin();

    let vocab_size = 256;
    let n_tokens = 4;
    let data = vec![1.0f32; n_tokens * vocab_size];

    let ref_path = tmp.path().join("ref.bin");
    let cand_path = tmp.path().join("cand.bin");
    write_f32_le_file(&ref_path, &data);
    write_f32_le_file(&cand_path, &data);

    let o_json = serde_json::json!({
        "selected_experts": [
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            [5, 6, 7, 8],
            [55, 56, 57, 58]
        ]
    });
    // Urutan acak tapi elemen SET identik
    let c_json = serde_json::json!({
        "selected_experts": [
            [4, 1, 3, 2],
            [40, 30, 20, 10],
            [8, 7, 6, 5],
            [57, 58, 55, 56]
        ]
    });

    let o_path = tmp.path().join("oracle.json");
    let c_path = tmp.path().join("cand.json");
    fs::write(&o_path, serde_json::to_string(&o_json).unwrap()).unwrap();
    fs::write(&c_path, serde_json::to_string(&c_json).unwrap()).unwrap();

    let out = Command::new(&compare_bin)
        .args([
            "compare",
            "--ref",
            ref_path.to_str().unwrap(),
            "--cand",
            cand_path.to_str().unwrap(),
            "--gate",
            "G-M4-1",
            "--dim",
            &vocab_size.to_string(),
            "--oracle-routing",
            o_path.to_str().unwrap(),
            "--cand-routing",
            c_path.to_str().unwrap(),
        ])
        .output()
        .expect("run compare");

    assert_eq!(
        out.status.code(),
        Some(0),
        "set-equal routing must pass compare"
    );
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["status"], "MATCH");
    assert_eq!(v["verdict"], "PASS");
}

// 4. Hierarki Pita Delta: layout > bias > rope > numeric
#[test]
fn test_delta_bands_hierarchy() {
    let tmp = TempDir::new("m4_bands");
    let compare_bin = get_compare_bin();
    let vocab_size = 256;
    let n_tokens = 4;
    let ref_data = vec![1.0f32; n_tokens * vocab_size];
    let ref_path = tmp.path().join("ref.bin");
    write_f32_le_file(&ref_path, &ref_data);

    let cases = [
        (1.5f32, "dtype-layout"),
        (0.5f32, "bias-placement"),
        (0.1f32, "rope-style"),
    ];

    for (noise, expected_cat) in cases {
        let mut cand_data = ref_data.clone();
        cand_data[0] += noise;
        let cand_path = tmp.path().join(format!("cand_{}.bin", expected_cat));
        write_f32_le_file(&cand_path, &cand_data);

        let out = Command::new(&compare_bin)
            .args([
                "compare",
                "--ref",
                ref_path.to_str().unwrap(),
                "--cand",
                cand_path.to_str().unwrap(),
                "--gate",
                "G-M4-1",
                "--dim",
                &vocab_size.to_string(),
            ])
            .output()
            .expect("run compare");

        assert_eq!(out.status.code(), Some(1));
        let v: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(v["status"], "MISMATCH");
        assert_eq!(
            v["fail_category"], expected_cat,
            "failed category for noise {}",
            noise
        );
    }
}

// 5. Perbedaan kecil di bawah pita (< 0.05) yang lolos semua 4 metrik -> PASS
#[test]
fn test_numeric_order_pass_loose() {
    let tmp = TempDir::new("m4_loose_pass");
    let compare_bin = get_compare_bin();
    let vocab_size = 512;
    let n_tokens = 16;
    let total = n_tokens * vocab_size;
    let ref_data = vec![1.0f32; total];
    let ref_path = tmp.path().join("ref.bin");
    write_f32_le_file(&ref_path, &ref_data);

    // Delta 0.0001 (1e-4 <= 1e-2), epsilon_rel ~1e-4 <= 1e-4, A=100.0%
    let cand_data: Vec<f32> = ref_data.iter().map(|&x| x + 0.00005).collect();
    let cand_path = tmp.path().join("cand.bin");
    write_f32_le_file(&cand_path, &cand_data);

    let out = Command::new(&compare_bin)
        .args([
            "compare",
            "--ref",
            ref_path.to_str().unwrap(),
            "--cand",
            cand_path.to_str().unwrap(),
            "--gate",
            "G-M4-1",
            "--dim",
            &vocab_size.to_string(),
        ])
        .output()
        .expect("run compare");

    assert_eq!(out.status.code(), Some(0));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["status"], "MATCH");
    assert_eq!(v["verdict"], "PASS");
}

// 6. Tolak kandidat kuantisasi BF16 di Gate G-M4-1:
// Noise kuantisasi BF16 (eps_rel ~1.7e-3) wajib ditolak oleh threshold 1e-4.
#[test]
fn test_bf16_noise_floor_rejected_in_g_m4_1() {
    let tmp = TempDir::new("m4_bf16_rejection");
    let compare_bin = get_compare_bin();
    let vocab_size = 512;
    let n_tokens = 16;
    let total = n_tokens * vocab_size;

    let ref_data: Vec<f32> = (0..total)
        .map(|i| ((i % 100) as f32 - 50.0) * 0.1)
        .collect();
    let ref_path = tmp.path().join("ref.bin");
    write_f32_le_file(&ref_path, &ref_data);

    // Simulasikan truncasi BF16: nol-kan 16 bit terbawah dari representasi IEEE 754 f32
    let bf16_cand: Vec<f32> = ref_data
        .iter()
        .map(|&f| {
            let u = f.to_bits();
            let truncated = u & 0xFFFF0000;
            f32::from_bits(truncated)
        })
        .collect();
    let cand_path = tmp.path().join("cand_bf16.bin");
    write_f32_le_file(&cand_path, &bf16_cand);

    let out = Command::new(&compare_bin)
        .args([
            "compare",
            "--ref",
            ref_path.to_str().unwrap(),
            "--cand",
            cand_path.to_str().unwrap(),
            "--gate",
            "G-M4-1",
            "--dim",
            &vocab_size.to_string(),
        ])
        .output()
        .expect("run compare");

    assert_eq!(
        out.status.code(),
        Some(1),
        "BF16 noise floor must fail G-M4-1"
    );
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["status"], "MISMATCH");
    assert_eq!(v["verdict"], "FAIL");
}

// 7. Identity Compare pada Prompt 1 Golden Oracle:
// Memastikan artefak golden asli tools/fixtures/m4_prompt1_oracle.bin
// dievaluasi sempurna terhadap dirinya sendiri (identity proof).
#[test]
fn test_golden_prompt1_oracle_identity() {
    let root = get_root_dir();
    let oracle_bin = root.join("tools/fixtures/m4_prompt1_oracle.bin");
    let routing_l0 = root.join("tools/fixtures/m4_prompt1_routing/routing_L0.json");

    if !oracle_bin.exists() || !routing_l0.exists() {
        eprintln!("SKIPPED: m4_prompt1_oracle.bin or routing fixture not present");
        return;
    }

    let compare_bin = get_compare_bin();
    let out = Command::new(&compare_bin)
        .args([
            "compare",
            "--ref",
            oracle_bin.to_str().unwrap(),
            "--cand",
            oracle_bin.to_str().unwrap(),
            "--gate",
            "G-M4-1",
            "--dim",
            "151936",
            "--oracle-routing",
            routing_l0.to_str().unwrap(),
            "--cand-routing",
            routing_l0.to_str().unwrap(),
        ])
        .output()
        .expect("run compare");

    assert_eq!(out.status.code(), Some(0));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["status"], "MATCH");
    assert_eq!(v["verdict"], "PASS");
    assert_eq!(v["metrics"]["delta_max"], 0.0);
    assert_eq!(v["metrics"]["epsilon_rel"], 0.0);
    let cos_val = v["metrics"]["cos_theta"].as_f64().unwrap();
    assert!((cos_val - 1.0).abs() < 1e-6, "cos_theta must be ~1.0");
    assert_eq!(v["metrics"]["agreement"], 100.0);
    assert_eq!(v["metrics"]["delta_ce"], 0.0);
}
