// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! Subcommand `dismoen-tools compare` — evaluasi ekivalensi numerik F10.
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
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt::Write as _;
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

/// Compute mean Jaccard set similarity across routing rows.
/// J(O, C) = |O ∩ C| / |O ∪ C|.
pub fn compute_routing_jaccard(oracle_experts: &[Vec<usize>], cand_experts: &[Vec<usize>]) -> f64 {
    if oracle_experts.is_empty()
        || cand_experts.is_empty()
        || oracle_experts.len() != cand_experts.len()
    {
        return 0.0;
    }
    let mut total_jaccard = 0.0;
    for (o_row, c_row) in oracle_experts.iter().zip(cand_experts.iter()) {
        let o_set: HashSet<usize> = o_row.iter().copied().collect();
        let c_set: HashSet<usize> = c_row.iter().copied().collect();
        let intersection = o_set.intersection(&c_set).count();
        let union = o_set.union(&c_set).count();
        if union > 0 {
            total_jaccard += (intersection as f64) / (union as f64);
        } else {
            total_jaccard += 1.0;
        }
    }
    total_jaccard / (oracle_experts.len() as f64)
}

/// Deterministic Top-K selection with tie-breaking policy:
/// rank(i) > rank(j) <=> (score[i] > score[j]) || (score[i] == score[j] && i < j)
/// Lower expert_id wins on score ties.
#[cfg(test)]
pub fn select_top_k_with_tie_break(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    indexed.iter().take(k).map(|(idx, _)| *idx).collect()
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

    // GDNS v1 framed binary format (normatif M8)
    if bytes.starts_with(b"GDNS") {
        if bytes.len() < 160 {
            return Err((
                "LAYOUT_MISMATCH".to_string(),
                format!(
                    "GDNS state file shorter than minimum 160 bytes (header 128B + checksum 32B): {} bytes",
                    bytes.len()
                ),
            ));
        }

        // Magic byte-by-byte check: [0x47, 0x44, 0x4E, 0x53] ('G', 'D', 'N', 'S')
        if bytes[0] != b'G' || bytes[1] != b'D' || bytes[2] != b'N' || bytes[3] != b'S' {
            return Err((
                "LAYOUT_MISMATCH".to_string(),
                "invalid GDNS magic bytes".to_string(),
            ));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 1 {
            return Err((
                "LAYOUT_MISMATCH".to_string(),
                format!("unsupported GDNS version: {}, expected 1", version),
            ));
        }

        let architecture_id = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if architecture_id != 1 {
            return Err((
                "MODEL_CONFIG_MISMATCH".to_string(),
                format!(
                    "unsupported GDNS architecture_id: {}, expected 1 (ARCH_QWEN_GDN)",
                    architecture_id
                ),
            ));
        }

        let dtype = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if dtype != 1 {
            return Err((
                "LAYOUT_MISMATCH".to_string(),
                format!("unsupported GDNS dtype: {}, expected 1 (FP32)", dtype),
            ));
        }

        let layers = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let dv = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let dk = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let state_bytes = u64::from_le_bytes(bytes[32..40].try_into().unwrap());

        let expected_payload = (layers as u64)
            .checked_mul(dv as u64)
            .and_then(|x| x.checked_mul(dk as u64))
            .and_then(|x| x.checked_mul(4));

        if expected_payload != Some(state_bytes) {
            return Err((
                "LAYOUT_MISMATCH".to_string(),
                format!(
                    "GDNS state_bytes mismatch dimensions: header claims {} bytes, but layers={} dv={} dk={} implies {:?} bytes",
                    state_bytes, layers, dv, dk, expected_payload
                ),
            ));
        }

        let total_expected = 128u64 + state_bytes + 32u64;
        if bytes.len() as u64 != total_expected {
            return Err((
                "LAYOUT_MISMATCH".to_string(),
                format!(
                    "GDNS file size mismatch: actual {} bytes, expected {} bytes (128B header + {}B payload + 32B checksum)",
                    bytes.len(),
                    total_expected,
                    state_bytes
                ),
            ));
        }

        let payload_end = 128 + state_bytes as usize;

        // Verify incremental SHA-256 over [header_bytes (128B) || state_bytes]
        let mut hasher = Sha256::new();
        hasher.update(&bytes[..128]); // header_bytes (128B)
        hasher.update(&bytes[128..payload_end]); // state_bytes
        let computed_hash: [u8; 32] = hasher.finalize().into();
        let expected_hash = &bytes[payload_end..payload_end + 32];

        if computed_hash != expected_hash {
            let mut comp_hex = String::with_capacity(64);
            let mut exp_hex = String::with_capacity(64);
            for b in computed_hash {
                let _ = write!(comp_hex, "{b:02x}");
            }
            for b in expected_hash {
                let _ = write!(exp_hex, "{b:02x}");
            }
            return Err((
                "CORRUPT_STATE_CHECKSUM".to_string(),
                format!(
                    "GDNS state checksum mismatch: computed {}, expected {}",
                    comp_hex, exp_hex
                ),
            ));
        }

        let payload = &bytes[128..payload_end];
        let (chunks, _) = payload.as_chunks::<4>();
        let floats: Vec<f32> = chunks
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        return Ok(floats);
    }

    // Raw float32 format (legacy)
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

/// Helper untuk membuat file framed binary GDNS v1 (digunakan oleh test dan pipeline serialisasi)
#[cfg(test)]
pub fn write_gdns_v1(
    path: &str,
    layers: u32,
    dv: u32,
    dk: u32,
    manifest_hash: Option<&[u8; 32]>,
    data: &[f32],
) -> std::io::Result<()> {
    let state_bytes = (layers as u64) * (dv as u64) * (dk as u64) * 4;
    assert_eq!(
        data.len(),
        (layers * dv * dk) as usize,
        "data length mismatch dimensions"
    );

    let mut buf = Vec::with_capacity(128 + state_bytes as usize + 32);
    // 128-byte header
    buf.extend_from_slice(&[0x47, 0x44, 0x4E, 0x53]); // magic: 'G', 'D', 'N', 'S'
    buf.extend_from_slice(&1u32.to_le_bytes()); // version = 1
    buf.extend_from_slice(&1u32.to_le_bytes()); // architecture_id = 1 (ARCH_QWEN_GDN)
    buf.extend_from_slice(&1u32.to_le_bytes()); // dtype = 1 (FP32)
    buf.extend_from_slice(&layers.to_le_bytes()); // layers
    buf.extend_from_slice(&dv.to_le_bytes()); // dv
    buf.extend_from_slice(&dk.to_le_bytes()); // dk
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved1 (alignment)
    buf.extend_from_slice(&state_bytes.to_le_bytes()); // state_bytes (8 bytes)
    if let Some(h) = manifest_hash {
        buf.extend_from_slice(h);
    } else {
        buf.extend_from_slice(&[0u8; 32]);
    }
    buf.extend_from_slice(&[0u8; 56]); // reserved2 (padding to 128 bytes)
    assert_eq!(buf.len(), 128, "header must be exactly 128 bytes");

    for val in data {
        buf.extend_from_slice(&val.to_le_bytes());
    }

    // Incremental digest = SHA256(header_bytes || state_bytes)
    let mut hasher = Sha256::new();
    hasher.update(&buf[..128]); // header_bytes (128B)
    hasher.update(&buf[128..]); // state_bytes (payload)
    let digest: [u8; 32] = hasher.finalize().into();
    buf.extend_from_slice(&digest);

    fs::write(path, buf)
}

#[cfg(test)]
#[derive(Debug, PartialEq, Clone)]
pub struct KmssSessionMetadata {
    pub version: u32,
    pub architecture_id: u32,
    pub manifest_hash: [u8; 32],
    pub seq_len: u32,
    pub vocab_size: u32,
    pub kv_layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub kv_dtype: u32,
    pub gdn_layers: u32,
    pub gdn_dv: u32,
    pub gdn_dk: u32,
    pub gdn_dtype: u32,
    pub kv_bytes: u64,
    pub gdn_bytes: u64,
    pub token_bytes: u64,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Clone)]
pub struct KmssSessionPayload {
    pub kv_payload: Vec<u8>,
    pub gdn_payload: Vec<u8>,
    pub token_ids: Vec<u32>,
}

#[cfg(test)]
pub fn parse_kmss_v1(
    bytes: &[u8],
) -> Result<(KmssSessionMetadata, KmssSessionPayload), (String, String)> {
    if bytes.len() < 160 {
        return Err((
            "LAYOUT_MISMATCH".to_string(),
            format!(
                "KMSS file size {} too small (minimum 160 bytes for 128B header + 32B hash)",
                bytes.len()
            ),
        ));
    }

    if &bytes[0..4] != b"KMSS" {
        return Err((
            "FORMAT_MISMATCH".to_string(),
            "invalid KMSS magic, expected 'KMSS'".to_string(),
        ));
    }

    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != 1 {
        return Err((
            "VERSION_MISMATCH".to_string(),
            format!("unsupported KMSS version: {}, expected 1", version),
        ));
    }

    let architecture_id = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if architecture_id != 2 {
        return Err((
            "CONFIG_MISMATCH".to_string(),
            format!(
                "unsupported KMSS architecture_id: {}, expected 2 (ARCH_QWEN36_HYBRID_35B)",
                architecture_id
            ),
        ));
    }

    let mut manifest_hash = [0u8; 32];
    manifest_hash.copy_from_slice(&bytes[12..44]);

    let seq_len = u32::from_le_bytes(bytes[44..48].try_into().unwrap());
    let vocab_size = u32::from_le_bytes(bytes[48..52].try_into().unwrap());
    let kv_layers = u32::from_le_bytes(bytes[52..56].try_into().unwrap());
    let kv_heads = u32::from_le_bytes(bytes[56..60].try_into().unwrap());
    let head_dim = u32::from_le_bytes(bytes[60..64].try_into().unwrap());
    let kv_dtype = u32::from_le_bytes(bytes[64..68].try_into().unwrap());
    let gdn_layers = u32::from_le_bytes(bytes[68..72].try_into().unwrap());
    let gdn_dv = u32::from_le_bytes(bytes[72..76].try_into().unwrap());
    let gdn_dk = u32::from_le_bytes(bytes[76..80].try_into().unwrap());
    let gdn_dtype = u32::from_le_bytes(bytes[80..84].try_into().unwrap());

    let kv_bytes = u64::from_le_bytes(bytes[84..92].try_into().unwrap());
    let gdn_bytes = u64::from_le_bytes(bytes[92..100].try_into().unwrap());
    let token_bytes = u64::from_le_bytes(bytes[100..108].try_into().unwrap());

    let b_kv = match kv_dtype {
        1 => 4u64, // FP32
        2 => 2u64, // BF16
        _ => {
            return Err((
                "LAYOUT_MISMATCH".to_string(),
                format!(
                    "unsupported KMSS kv_dtype: {}, expected 1 (FP32) or 2 (BF16)",
                    kv_dtype
                ),
            ));
        }
    };

    let expected_kv_bytes = (2u64)
        .checked_mul(seq_len as u64)
        .and_then(|x| x.checked_mul(kv_layers as u64))
        .and_then(|x| x.checked_mul(kv_heads as u64))
        .and_then(|x| x.checked_mul(head_dim as u64))
        .and_then(|x| x.checked_mul(b_kv));

    if expected_kv_bytes != Some(kv_bytes) {
        return Err((
            "LAYOUT_MISMATCH".to_string(),
            format!(
                "KMSS kv_bytes mismatch: header claims {}, expected {:?}",
                kv_bytes, expected_kv_bytes
            ),
        ));
    }

    let expected_gdn_bytes = (gdn_layers as u64)
        .checked_mul(gdn_dv as u64)
        .and_then(|x| x.checked_mul(gdn_dk as u64))
        .and_then(|x| x.checked_mul(4));

    if expected_gdn_bytes != Some(gdn_bytes) {
        return Err((
            "LAYOUT_MISMATCH".to_string(),
            format!(
                "KMSS gdn_bytes mismatch: header claims {}, expected {:?}",
                gdn_bytes, expected_gdn_bytes
            ),
        ));
    }

    let expected_token_bytes = (seq_len as u64).checked_mul(4);
    if expected_token_bytes != Some(token_bytes) {
        return Err((
            "LAYOUT_MISMATCH".to_string(),
            format!(
                "KMSS token_bytes mismatch: header claims {}, expected {:?}",
                token_bytes, expected_token_bytes
            ),
        ));
    }

    let total_payload = match kv_bytes
        .checked_add(gdn_bytes)
        .and_then(|x| x.checked_add(token_bytes))
    {
        Some(sum) => sum,
        None => {
            return Err((
                "LAYOUT_MISMATCH".to_string(),
                "KMSS payload size overflow".to_string(),
            ));
        }
    };

    let total_expected = 128u64 + total_payload + 32u64;
    if bytes.len() as u64 != total_expected {
        return Err((
            "LAYOUT_MISMATCH".to_string(),
            format!(
                "KMSS file size mismatch: actual {} bytes, expected {} bytes",
                bytes.len(),
                total_expected
            ),
        ));
    }

    let payload_end = 128 + total_payload as usize;
    let mut hasher = Sha256::new();
    hasher.update(&bytes[..128]);
    hasher.update(&bytes[128..payload_end]);
    let computed_hash: [u8; 32] = hasher.finalize().into();
    let expected_hash = &bytes[payload_end..payload_end + 32];

    if computed_hash != expected_hash {
        let mut comp_hex = String::with_capacity(64);
        let mut exp_hex = String::with_capacity(64);
        for b in computed_hash {
            let _ = write!(comp_hex, "{b:02x}");
        }
        for b in expected_hash {
            let _ = write!(exp_hex, "{b:02x}");
        }
        return Err((
            "CORRUPT_SESSION_CHECKSUM".to_string(),
            format!(
                "KMSS session checksum mismatch: computed {}, expected {}",
                comp_hex, exp_hex
            ),
        ));
    }

    let kv_end = 128 + kv_bytes as usize;
    let gdn_end = kv_end + gdn_bytes as usize;
    let kv_payload = bytes[128..kv_end].to_vec();
    let gdn_payload = bytes[kv_end..gdn_end].to_vec();
    let token_raw = &bytes[gdn_end..payload_end];
    let (chunks, _) = token_raw.as_chunks::<4>();
    let token_ids: Vec<u32> = chunks.iter().map(|c| u32::from_le_bytes(*c)).collect();

    let meta = KmssSessionMetadata {
        version,
        architecture_id,
        manifest_hash,
        seq_len,
        vocab_size,
        kv_layers,
        kv_heads,
        head_dim,
        kv_dtype,
        gdn_layers,
        gdn_dv,
        gdn_dk,
        gdn_dtype,
        kv_bytes,
        gdn_bytes,
        token_bytes,
    };

    let payload = KmssSessionPayload {
        kv_payload,
        gdn_payload,
        token_ids,
    };

    Ok((meta, payload))
}

#[cfg(test)]
pub fn write_kmss_v1(
    path: &str,
    meta: &KmssSessionMetadata,
    kv_payload: &[u8],
    gdn_payload: &[u8],
    token_ids: &[u32],
) -> std::io::Result<()> {
    assert_eq!(
        kv_payload.len() as u64,
        meta.kv_bytes,
        "kv_payload size mismatch meta"
    );
    assert_eq!(
        gdn_payload.len() as u64,
        meta.gdn_bytes,
        "gdn_payload size mismatch meta"
    );
    assert_eq!(
        (token_ids.len() * 4) as u64,
        meta.token_bytes,
        "token_ids size mismatch meta"
    );

    let total_size = 128 + kv_payload.len() + gdn_payload.len() + (token_ids.len() * 4) + 32;
    let mut buf = Vec::with_capacity(total_size);

    buf.extend_from_slice(b"KMSS");
    buf.extend_from_slice(&meta.version.to_le_bytes());
    buf.extend_from_slice(&meta.architecture_id.to_le_bytes());
    buf.extend_from_slice(&meta.manifest_hash);
    buf.extend_from_slice(&meta.seq_len.to_le_bytes());
    buf.extend_from_slice(&meta.vocab_size.to_le_bytes());
    buf.extend_from_slice(&meta.kv_layers.to_le_bytes());
    buf.extend_from_slice(&meta.kv_heads.to_le_bytes());
    buf.extend_from_slice(&meta.head_dim.to_le_bytes());
    buf.extend_from_slice(&meta.kv_dtype.to_le_bytes());
    buf.extend_from_slice(&meta.gdn_layers.to_le_bytes());
    buf.extend_from_slice(&meta.gdn_dv.to_le_bytes());
    buf.extend_from_slice(&meta.gdn_dk.to_le_bytes());
    buf.extend_from_slice(&meta.gdn_dtype.to_le_bytes());
    buf.extend_from_slice(&meta.kv_bytes.to_le_bytes());
    buf.extend_from_slice(&meta.gdn_bytes.to_le_bytes());
    buf.extend_from_slice(&meta.token_bytes.to_le_bytes());
    buf.extend_from_slice(&[0u8; 20]); // reserved padding to 128B
    assert_eq!(buf.len(), 128, "header must be exactly 128 bytes");

    buf.extend_from_slice(kv_payload);
    buf.extend_from_slice(gdn_payload);
    for tid in token_ids {
        buf.extend_from_slice(&tid.to_le_bytes());
    }

    let mut hasher = Sha256::new();
    hasher.update(&buf[..128]);
    hasher.update(&buf[128..]);
    let digest: [u8; 32] = hasher.finalize().into();
    buf.extend_from_slice(&digest);

    fs::write(path, buf)
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
    let mut custom_tolerance: Option<f64> = None;
    let mut output_path = String::new();
    let mut run_id = String::new();
    let mut oracle_routing = String::new();
    let mut cand_routing = String::new();
    let mut routing_min_jaccard: Option<f64> = None;

    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ref" | "--reference" => {
                i += 1;
                if i < args.len() {
                    ref_path = args[i].clone();
                }
            }
            "--cand" | "--candidate" => {
                i += 1;
                if i < args.len() {
                    cand_path = args[i].clone();
                }
            }
            "--tolerance" => {
                i += 1;
                if i < args.len() {
                    if let Ok(tol) = args[i].parse::<f64>() {
                        custom_tolerance = Some(tol);
                    }
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = args[i].clone();
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
            "--routing-min-jaccard" | "--min-jaccard" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<f64>() {
                        routing_min_jaccard = Some(v);
                    }
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
            "pakai: dismoen-tools compare <ref.bin> <cand.bin> [--gate G-M1-1|G-M2-1|G-M3-1|G-M4-1|G-M5-1|G-M8-1] [--dim <N>] [--oracle-routing <json>] [--cand-routing <json>] [--min-jaccard <val>]",
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
        if let Some(min_jaccard) = routing_min_jaccard {
            let jaccard = compute_routing_jaccard(&o_exp, &c_exp);
            routing_ok = jaccard >= min_jaccard;
        } else {
            routing_ok = check_routing_match(&o_exp, &c_exp);
        }
    }

    if custom_tolerance.is_some() && gate == "G-M1-1" {
        gate = "G-M8-1".to_string();
    }

    if run_id.is_empty() {
        run_id = match gate.as_str() {
            "G-M8-1" => "M8-F10-001".to_string(),
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

    if !explicit_dim {
        if gate == "G-M8-1" {
            if !ref_floats.len().is_multiple_of(vocab_size) {
                vocab_size = ref_floats.len();
            }
        } else if gate == "G-M2-1" || gate == "G-M3-1" {
            if ref_floats.len() % 2048 == 0 {
                vocab_size = 2048;
            } else if ref_floats.len() % 64 == 0 {
                vocab_size = 64;
            } else {
                vocab_size = ref_floats.len();
            }
        }
    }

    let metrics = match compute_f10_metrics(&ref_floats, &cand_floats, vocab_size) {
        Ok(m) => m,
        Err(e) => return emit_error("LAYOUT_MISMATCH", &e),
    };

    // Evaluate gate
    let (mut is_pass, threshold_str, mut fail_cat) = if let Some(tol) = custom_tolerance {
        let pass = metrics.delta_max <= tol && metrics.epsilon_rel <= 1e-4;
        let thresh = if (tol - 1e-3).abs() < 1e-9 {
            "delta_max <= 1e-3 && epsilon_rel <= 1e-4".to_string()
        } else {
            format!("delta_max <= {:e} && epsilon_rel <= 1e-4", tol)
        };
        let cat = if pass {
            None
        } else if metrics.delta_max > 0.05 || metrics.cos_theta < 0.99 {
            Some("dtype-layout".to_string())
        } else {
            Some("numeric-order".to_string())
        };
        (pass, thresh, cat)
    } else {
        let (p, t, c) = evaluate_gate(&gate, &metrics);
        (p, t.to_string(), c)
    };

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
        threshold: threshold_str,
        fail_category: fail_cat,
    };

    let report_json = serde_json::to_string_pretty(&report).unwrap();
    println!("{}", report_json);

    if !output_path.is_empty() {
        if let Err(e) = fs::write(&output_path, &report_json) {
            return emit_error(
                "OUTPUT_ERROR",
                &format!("failed writing compare report: {}", e),
            );
        }
    }

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
        "G-M2-1" | "G-M3-1" | "G-M9-1" => {
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
        "G-M4-1" | "G-M5-1" | "G-M9-2" | "G-M9-3" => {
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
        "G-M8-1" => {
            let pass = metrics.delta_max <= 1e-3 && metrics.epsilon_rel <= 1e-4;
            let thresh = "delta_max <= 1e-3 && epsilon_rel <= 1e-4";
            let cat = if pass {
                None
            } else if metrics.delta_max > 0.05 || metrics.cos_theta < 0.99 {
                Some("dtype-layout".to_string())
            } else {
                Some("numeric-order".to_string())
            };
            (pass, thresh, cat)
        }
        "G-M10-3" => {
            let pass = metrics.delta_max <= 1e-7;
            let thresh = "delta_max <= 1e-7";
            let cat = if pass {
                None
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

    #[test]
    fn test_read_floats_gdns_v1_roundtrip_and_tamper() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_gdns_v1.bin");
        let path_str = path.to_str().unwrap();

        let layers = 2u32;
        let dv = 4u32;
        let dk = 4u32;
        let n_floats = (layers * dv * dk) as usize; // 32 floats
        let original_data: Vec<f32> = (0..n_floats).map(|i| (i as f32) * 0.5).collect();
        let manifest_hash: [u8; 32] = [0xAB; 32];

        // 1. Write valid GDNS v1 file with 128B header and manifest hash
        write_gdns_v1(
            path_str,
            layers,
            dv,
            dk,
            Some(&manifest_hash),
            &original_data,
        )
        .expect("write failed");

        // 2. Read back and verify exact match
        let read_back = read_floats(path_str).expect("read failed");
        assert_eq!(read_back, original_data);

        // 3. Tamper a single byte in payload -> must fail with CORRUPT_STATE_CHECKSUM
        let mut raw_bytes = fs::read(path_str).expect("read failed");
        let payload_byte_idx = 128 + 10;
        raw_bytes[payload_byte_idx] ^= 0xFF;
        fs::write(path_str, &raw_bytes).expect("write tampered failed");

        let err = read_floats(path_str).expect_err("tampered payload should fail");
        assert_eq!(err.0, "CORRUPT_STATE_CHECKSUM");

        // 4. Tamper a byte in trailing checksum -> must fail with CORRUPT_STATE_CHECKSUM
        raw_bytes[payload_byte_idx] ^= 0xFF; // restore payload
        let checksum_idx = raw_bytes.len() - 5;
        raw_bytes[checksum_idx] ^= 0x01;
        fs::write(path_str, &raw_bytes).expect("write tampered failed");

        let err = read_floats(path_str).expect_err("tampered checksum should fail");
        assert_eq!(err.0, "CORRUPT_STATE_CHECKSUM");

        // 5. Tamper architecture_id in header (offset 8..12) -> MODEL_CONFIG_MISMATCH
        raw_bytes[checksum_idx] ^= 0x01; // restore checksum
        raw_bytes[8] = 99; // invalid architecture_id
        let payload_end = 128 + (layers * dv * dk * 4) as usize;
        let mut hasher = Sha256::new();
        hasher.update(&raw_bytes[..128]);
        hasher.update(&raw_bytes[128..payload_end]);
        let valid_hash: [u8; 32] = hasher.finalize().into();
        raw_bytes[payload_end..payload_end + 32].copy_from_slice(&valid_hash);
        fs::write(path_str, &raw_bytes).expect("write tampered failed");

        let err = read_floats(path_str).expect_err("architecture mismatch should fail");
        assert_eq!(err.0, "MODEL_CONFIG_MISMATCH");

        // 6. Truncated GDNS file -> LAYOUT_MISMATCH
        let truncated = &raw_bytes[..50];
        fs::write(path_str, truncated).expect("write truncated failed");
        let err = read_floats(path_str).expect_err("truncated should fail");
        assert_eq!(err.0, "LAYOUT_MISMATCH");

        // Clean up
        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_gate_g_m8_1_evaluation() {
        let pass_metrics = CompareMetrics {
            delta_max: 0.0005,
            epsilon_rel: 0.00005,
            cos_theta: 0.99999,
            agreement: 100.0,
            delta_ce: 0.001,
        };
        let (pass, thresh, cat) = evaluate_gate("G-M8-1", &pass_metrics);
        assert!(pass);
        assert_eq!(thresh, "delta_max <= 1e-3 && epsilon_rel <= 1e-4");
        assert_eq!(cat, None);

        let fail_metrics = CompareMetrics {
            delta_max: 0.01,
            epsilon_rel: 0.001,
            cos_theta: 0.98,
            agreement: 99.0,
            delta_ce: 0.1,
        };
        let (pass, _, cat) = evaluate_gate("G-M8-1", &fail_metrics);
        assert!(!pass);
        assert_eq!(cat.as_deref(), Some("dtype-layout"));
    }

    #[test]
    fn test_kmss_v1_session_roundtrip_and_tamper() {
        let path = std::env::temp_dir().join("test_session_v1.bin");
        let path_str = path.to_str().unwrap();

        let seq_len = 4u32;
        let kv_layers = 10u32;
        let kv_heads = 2u32;
        let head_dim = 128u32;
        let b_kv = 2u64; // BF16 (2B)
        let kv_bytes = 2u64
            * (seq_len as u64)
            * (kv_layers as u64)
            * (kv_heads as u64)
            * (head_dim as u64)
            * b_kv;
        let gdn_bytes = 30u64 * 128 * 128 * 4;
        let token_bytes = (seq_len as u64) * 4;

        let meta = KmssSessionMetadata {
            version: 1,
            architecture_id: 2,
            manifest_hash: [0x5au8; 32],
            seq_len,
            vocab_size: 248320,
            kv_layers,
            kv_heads,
            head_dim,
            kv_dtype: 2, // BF16
            gdn_layers: 30,
            gdn_dv: 128,
            gdn_dk: 128,
            gdn_dtype: 1, // FP32
            kv_bytes,
            gdn_bytes,
            token_bytes,
        };

        let kv_payload = vec![0x12u8; kv_bytes as usize];
        let gdn_payload = vec![0x34u8; gdn_bytes as usize];
        let token_ids = vec![101u32, 202u32, 303u32, 404u32];

        // 1. Write session
        write_kmss_v1(path_str, &meta, &kv_payload, &gdn_payload, &token_ids)
            .expect("write_kmss_v1 failed");

        // 2. Read and verify round-trip
        let raw_bytes = fs::read(path_str).expect("read session failed");
        let (parsed_meta, parsed_payload) =
            parse_kmss_v1(&raw_bytes).expect("parse_kmss_v1 failed");

        assert_eq!(parsed_meta, meta);
        assert_eq!(parsed_payload.kv_payload, kv_payload);
        assert_eq!(parsed_payload.gdn_payload, gdn_payload);
        assert_eq!(parsed_payload.token_ids, token_ids);

        // 3. Tamper checksum -> CORRUPT_SESSION_CHECKSUM
        let mut tampered = raw_bytes.clone();
        let last_idx = tampered.len() - 1;
        tampered[last_idx] ^= 0xff;
        let err = parse_kmss_v1(&tampered).expect_err("tampered checksum should fail");
        assert_eq!(err.0, "CORRUPT_SESSION_CHECKSUM");

        // 4. Architecture mismatch -> CONFIG_MISMATCH
        let mut bad_arch = raw_bytes.clone();
        bad_arch[8] = 99; // invalid arch
        let payload_end = 128 + (kv_bytes + gdn_bytes + token_bytes) as usize;
        let mut hasher = Sha256::new();
        hasher.update(&bad_arch[..128]);
        hasher.update(&bad_arch[128..payload_end]);
        let hash: [u8; 32] = hasher.finalize().into();
        bad_arch[payload_end..payload_end + 32].copy_from_slice(&hash);
        let err = parse_kmss_v1(&bad_arch).expect_err("architecture mismatch should fail");
        assert_eq!(err.0, "CONFIG_MISMATCH");

        // Clean up
        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_select_top_k_with_tie_break_and_jaccard() {
        // Tie-breaking test: scores with equal values
        // Index 0: 0.5, Index 1: 0.9, Index 2: 0.5, Index 3: 0.9, Index 4: 0.1
        let scores = vec![0.5f32, 0.9, 0.5, 0.9, 0.1];
        // Top 3 should be:
        // Rank 1 & 2 (score 0.9): tie between 1 and 3 -> 1, then 3
        // Rank 3 & 4 (score 0.5): tie between 0 and 2 -> 0 wins over 2
        let top3 = select_top_k_with_tie_break(&scores, 3);
        assert_eq!(top3, vec![1, 3, 0]);

        // Jaccard similarity test
        let o_exp = vec![vec![1, 2, 3, 4], vec![10, 20, 30, 40]];
        let c_exp_exact = vec![vec![4, 3, 2, 1], vec![40, 30, 20, 10]];
        assert!((compute_routing_jaccard(&o_exp, &c_exp_exact) - 1.0).abs() < 1e-6);

        // Perturbed row 1: 3 out of 4 match (intersection 3, union 5 -> 3/5 = 0.6)
        // Row 2: exact match (4/4 = 1.0)
        // Mean Jaccard = (0.6 + 1.0) / 2 = 0.8
        let c_exp_pert = vec![vec![1, 2, 3, 99], vec![10, 20, 30, 40]];
        let jaccard = compute_routing_jaccard(&o_exp, &c_exp_pert);
        assert!((jaccard - 0.8).abs() < 1e-6);
    }
}
