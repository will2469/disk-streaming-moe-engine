// Copyright 2026 will2469
// Licensed under the Apache License, Version 2.0 (the "License");
// See LICENSE for details.

//! SEC-1: verifikasi SHA-256 shard vs models.lock.json.
//! Tamper 1 byte -> tolak (exit 2). Entri "TBD-*" dilewati jujur (status skipped).

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;

#[derive(Debug, Deserialize)]
struct Shard {
    filename: String,
    sha256: String,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct Lock {
    model: String,
    revision: String,
    shards: Vec<Shard>,
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn hash_file(path: &std::path::Path) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Verifikasi dir terhadap lock. Kembalikan exit code (0 ok/skipped, 2 tolak).
/// Tanpa unwrap/exit di tengah: semua kegagalan mengalir sebagai nilai.
pub fn run(args: &[String]) -> i32 {
    let mut lock = String::new();
    let mut dir = String::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--lock" => {
                if let Some(v) = it.next() {
                    lock = v.clone();
                } else {
                    eprintln!(r#"{{"error_type":"USAGE","detail":"--lock butuh nilai"}}"#);
                    return 2;
                }
            }
            "--dir" => {
                if let Some(v) = it.next() {
                    dir = v.clone();
                } else {
                    eprintln!(r#"{{"error_type":"USAGE","detail":"--dir butuh nilai"}}"#);
                    return 2;
                }
            }
            x => {
                eprintln!(r#"{{"error_type":"USAGE","detail":"flag tak dikenal: {x}"}}"#);
                return 2;
            }
        }
    }
    if lock.is_empty() || dir.is_empty() {
        eprintln!(
            r#"{{"error_type":"USAGE","detail":"pakai: dismoen-tools verify --lock <models.lock.json> --dir <model-dir>"}}"#
        );
        return 2;
    }
    let text = match std::fs::read_to_string(&lock) {
        Ok(t) => t,
        Err(_) => {
            eprintln!(r#"{{"error_type":"FILE_NOT_FOUND","detail":"lock tak terbaca"}}"#);
            return 2;
        }
    };
    let lock: Lock = match serde_json::from_str(&text) {
        Ok(l) => l,
        Err(_) => {
            eprintln!(r#"{{"error_type":"INVALID_HEADER","detail":"lock JSON invalid"}}"#);
            return 2;
        }
    };
    let root = std::path::Path::new(&dir);
    let mut checked = 0u32;
    let mut skipped = 0u32;
    let mut bad = Vec::new();
    for s in &lock.shards {
        if s.sha256.starts_with("TBD-") {
            skipped += 1;
            continue;
        }
        let path = root.join(&s.filename);
        let actual_size = std::fs::metadata(&path).map(|m| m.len()).ok();
        if let (Some(exp), Some(got)) = (s.size, actual_size) {
            if exp != got {
                bad.push((
                    s.filename.clone(),
                    format!("size:{exp}"),
                    format!("size:{got}"),
                ));
                continue;
            }
        }
        match hash_file(&path) {
            Ok(got) if got == s.sha256 => checked += 1,
            Ok(got) => bad.push((s.filename.clone(), s.sha256.clone(), got)),
            Err(_) => bad.push((s.filename.clone(), s.sha256.clone(), "MISSING".into())),
        }
    }
    let model = &lock.model;
    let revision = &lock.revision;
    if bad.is_empty() {
        let status = if checked == 0 { "skipped" } else { "match" };
        println!(
            r#"{{"status":"{status}","model":"{model}","revision":"{revision}","checked":{checked},"skipped":{skipped},"mismatches":[]}}"#
        );
        0
    } else {
        let mut items = String::new();
        for (i, (f, exp, got)) in bad.iter().enumerate() {
            if i > 0 {
                items.push(',');
            }
            let _ = write!(
                items,
                r#"{{"file":"{f}","expected":"{exp}","found":"{got}"}}"#
            );
        }
        println!(
            r#"{{"status":"mismatch","checked":{checked},"skipped":{skipped},"mismatches":[{items}]}}"#
        );
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(dir: &std::path::Path, name: &str, bytes: &[u8]) -> String {
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        let h = hash_file(&p).unwrap();
        let lock = format!(
            r#"{{"model":"t","revision":"r","shards":[{{"filename":"{name}","sha256":"{h}","size":null}}]}}"#
        );
        std::fs::write(dir.join("lock.json"), &lock).unwrap();
        h
    }

    #[test]
    fn tamper_satu_byte_ditolak() {
        let d = std::env::temp_dir().join("dismoen-w3-tamper");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        tiny(&d, "a.bin", b"0123456789abcdef");
        // uji murni fungsi hash: ubah 1 byte -> hash beda (properti tamper-evident)
        let p = d.join("a.bin");
        let mut b = std::fs::read(&p).unwrap();
        b[3] ^= 1;
        std::fs::write(&p, &b).unwrap();
        let now = hash_file(&p).unwrap();
        let lock: Lock =
            serde_json::from_str(&std::fs::read_to_string(d.join("lock.json")).unwrap()).unwrap();
        assert_ne!(now, lock.shards[0].sha256, "tamper 1 byte wajib terdeteksi");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hash_deterministik() {
        let d = std::env::temp_dir().join("dismoen-w3-det");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let h1 = tiny(&d, "b.bin", b"abc");
        let h2 = hash_file(&d.join("b.bin")).unwrap();
        assert_eq!(h1, h2);
        // vektor known-answer SHA-256("abc")
        assert_eq!(
            h1,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
