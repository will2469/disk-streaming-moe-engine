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

#[test]
fn property_p2_config_vs_index_bias() {
    // Property P-2 (§2.3 / M2-W1):
    // Jebakan permanen: config.json TIDAK menuliskan attention_bias,
    // namun checkpoint memiliki tepat 72 tensor bias QKV (24 layer x 3 bias).
    let m0_cfg_p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/m0/model_config.json"
    );
    let m0_cfg: Value = serde_json::from_str(&std::fs::read_to_string(m0_cfg_p).unwrap()).unwrap();
    assert!(
        m0_cfg.get("attention_bias").is_none(),
        "config.json tidak boleh mendefinisikan attention_bias (jebakan §2.3)"
    );

    let m1_cfg_p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/m1/model_config.json"
    );
    let m1_cfg: Value = serde_json::from_str(&std::fs::read_to_string(m1_cfg_p).unwrap()).unwrap();
    assert!(
        m1_cfg.get("attention_bias").is_none(),
        "config.json m1 tidak boleh mendefinisikan attention_bias"
    );

    let idx = load_index();
    let wm = idx["weight_map"].as_object().unwrap();

    let mut qkv_bias_tensors = BTreeSet::new();
    let mut o_proj_bias_count = 0u32;

    for name in wm.keys() {
        if name.ends_with(".bias") && name.contains("self_attn") {
            if name.contains("q_proj.bias")
                || name.contains("k_proj.bias")
                || name.contains("v_proj.bias")
            {
                qkv_bias_tensors.insert(name.clone());
            }
            if name.contains("o_proj.bias") {
                o_proj_bias_count += 1;
            }
        }
    }

    assert_eq!(
        qkv_bias_tensors.len(),
        72,
        "Jumlah tensor bias QKV di index wajib tepat 72"
    );
    assert_eq!(
        o_proj_bias_count, 0,
        "o_proj TIDAK boleh memiliki bias pada Qwen1.5-MoE-A2.7B"
    );

    for layer in 0..24 {
        let q = format!("model.layers.{layer}.self_attn.q_proj.bias");
        let k = format!("model.layers.{layer}.self_attn.k_proj.bias");
        let v = format!("model.layers.{layer}.self_attn.v_proj.bias");
        assert!(
            qkv_bias_tensors.contains(&q),
            "Layer {layer} wajib punya q_proj.bias"
        );
        assert!(
            qkv_bias_tensors.contains(&k),
            "Layer {layer} wajib punya k_proj.bias"
        );
        assert!(
            qkv_bias_tensors.contains(&v),
            "Layer {layer} wajib punya v_proj.bias"
        );
    }
}

#[test]
fn property_router_weights_in_index() {
    // Property M3-W1 (§ Scope F8 / m3-w1-router.md):
    // Invariant: 24 layer wajib memiliki tepat 24 tensor router gate (model.layers.{layer}.mlp.gate.weight).
    // Router Qwen1.5-MoE TIDAK memiliki bias (mlp.gate.bias tidak boleh ada).
    // Config model wajib menyatakan norm_topk_prob=false.
    let m0_cfg_p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/m0/model_config.json"
    );
    let m0_cfg: Value = serde_json::from_str(&std::fs::read_to_string(m0_cfg_p).unwrap()).unwrap();
    assert_eq!(
        m0_cfg.get("norm_topk_prob").and_then(|v| v.as_bool()),
        Some(false),
        "config.json m0 wajib norm_topk_prob=false"
    );

    let m1_cfg_p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/m1/model_config.json"
    );
    let m1_cfg: Value = serde_json::from_str(&std::fs::read_to_string(m1_cfg_p).unwrap()).unwrap();
    assert_eq!(
        m1_cfg.get("norm_topk_prob").and_then(|v| v.as_bool()),
        Some(false),
        "config.json m1 wajib norm_topk_prob=false"
    );

    let idx = load_index();
    let wm = idx["weight_map"].as_object().unwrap();

    let mut router_gate_tensors = BTreeSet::new();
    let mut router_bias_count = 0u32;

    for name in wm.keys() {
        if name.contains("mlp.gate.weight") {
            router_gate_tensors.insert(name.clone());
        }
        if name.contains("mlp.gate.bias") {
            router_bias_count += 1;
        }
    }

    assert_eq!(
        router_gate_tensors.len(),
        24,
        "Jumlah tensor router mlp.gate.weight di index wajib tepat 24 (1 per layer)"
    );
    assert_eq!(
        router_bias_count, 0,
        "Router mlp.gate TIDAK boleh memiliki bias pada Qwen1.5-MoE-A2.7B"
    );

    for layer in 0..24 {
        let gate = format!("model.layers.{layer}.mlp.gate.weight");
        assert!(
            router_gate_tensors.contains(&gate),
            "Layer {layer} wajib punya mlp.gate.weight di index"
        );
    }
}

#[test]
fn property_moe_expert_weights_in_index() {
    // Property M3-W2 (§ SwiGLU Specification / m3-w2-swiglu.md):
    // Invariant:
    // - Tiap layer (0..23) wajib memiliki:
    //   - 60 routed experts x 3 (gate_proj, up_proj, down_proj) = 180 tensor
    //   - 1 shared expert x 3 (gate_proj, up_proj, down_proj) = 3 tensor
    //   - 1 shared expert gate (shared_expert_gate.weight) = 1 tensor
    // - Seluruh komponen MLP MoE TIDAK memiliki bias (0 bias tensor).
    let idx = load_index();
    let wm = idx["weight_map"].as_object().unwrap();

    let mut routed_tensors = 0u32;
    let mut shared_tensors = 0u32;
    let mut shared_gate_tensors = 0u32;
    let mut mlp_bias_count = 0u32;

    for name in wm.keys() {
        if name.contains(".mlp.") {
            if name.ends_with(".bias") {
                mlp_bias_count += 1;
            }
            if name.contains(".mlp.experts.") {
                routed_tensors += 1;
            } else if name.contains(".mlp.shared_expert.") {
                shared_tensors += 1;
            } else if name.contains(".mlp.shared_expert_gate.") {
                shared_gate_tensors += 1;
            }
        }
    }

    assert_eq!(
        routed_tensors,
        24 * 60 * 3,
        "Jumlah tensor routed experts wajib 24 layer x 60 expert x 3 = 4320"
    );
    assert_eq!(
        shared_tensors,
        24 * 3,
        "Jumlah tensor shared expert wajib 24 layer x 3 = 72"
    );
    assert_eq!(
        shared_gate_tensors, 24,
        "Jumlah tensor shared expert gate wajib tepat 24"
    );
    assert_eq!(
        mlp_bias_count, 0,
        "Komponen MLP MoE TIDAK boleh memiliki bias sama sekali"
    );

    for layer in 0..24 {
        let sh_gate = format!("model.layers.{layer}.mlp.shared_expert_gate.weight");
        assert!(
            wm.contains_key(&sh_gate),
            "Layer {layer} wajib punya shared_expert_gate.weight"
        );
        let sh_down = format!("model.layers.{layer}.mlp.shared_expert.down_proj.weight");
        assert!(
            wm.contains_key(&sh_down),
            "Layer {layer} wajib punya shared_expert down_proj"
        );
    }
}
