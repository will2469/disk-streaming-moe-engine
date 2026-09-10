// Shape-fidelity level R (M0-W3): struktur index asli tanpa download 28 GB.
// Index ter-commit di fixtures/m0_qwen_index.json (416 KB metadata, bukan bobot).

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

fn load_index() -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/m0_qwen_index.json"
    );
    let t = std::fs::read_to_string(p).expect("index fixture hilang");
    serde_json::from_str(&t).expect("index fixture invalid")
}

#[test]
fn tensor_count_4659_pada_pin() {
    let idx = load_index();
    let wm = idx["weight_map"].as_object().unwrap();
    assert_eq!(
        wm.len(),
        4659,
        "len(weight_map) wajib 4659 pada revision pin"
    );
}

#[test]
fn delapan_shard_24_layer_72_bias() {
    let idx = load_index();
    let wm = idx["weight_map"].as_object().unwrap();
    let mut files = BTreeSet::new();
    let mut layers = BTreeSet::new();
    let mut bias_qkv = 0u32;
    let mut per_file: BTreeMap<String, u32> = BTreeMap::new();
    for (name, f) in wm {
        let f = f.as_str().unwrap().to_string();
        files.insert(f.clone());
        *per_file.entry(f).or_insert(0) += 1;
        if let Some(rest) = name.strip_prefix("model.layers.") {
            let n: u32 = rest.split('.').next().unwrap().parse().unwrap();
            layers.insert(n);
        }
        if name.ends_with(".bias") {
            bias_qkv += 1;
        }
    }
    assert_eq!(files.len(), 8, "wajib 8 shard");
    assert_eq!(layers.len(), 24, "layer 0..23 wajib ada");
    assert_eq!(*layers.iter().min().unwrap(), 0);
    assert_eq!(*layers.iter().max().unwrap(), 23);
    assert_eq!(bias_qkv, 72, "bias QKV wajib 72");
    assert_eq!(idx["metadata"]["total_size"], 28631568384u64);
    assert!(per_file.values().all(|&c| c > 0));
}
