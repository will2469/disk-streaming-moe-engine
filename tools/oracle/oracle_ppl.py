#!/usr/bin/env python3
# Copyright 2026 will2469
# Licensed under the Apache License, Version 2.0 (the "License");
# See LICENSE for details.
"""Oracle PPL Measurement & Language Quality Conformance (M6-W5).

Menjalankan evaluasi PPL dan argmax agreement antara BF16 dan 4-bit Quant model:
1. Verifikasi Reproducibility Lock (§ Reproducibility Lock):
   - SHA256 corpus, model shards, dan tokenizer.
   - Re-tokenisasi byte-identical (drift = gate INVALID, bukan FAIL).
2. Akumulasi global lintas korpus:
   - Posisi prediksi: 1..255 per dokumen (N_pred = 255 per doc).
   - PPL_bf16 = exp(-sum(ln p_bf16) / N_pred_total)
   - PPL_quant = exp(-sum(ln p_quant) / N_pred_total)
   - Delta_PPL = PPL_quant - PPL_bf16 <= +0.5
   - Argmax Agreement A = matching_argmax / N_pred_total >= 95%
3. Diagnostik per dokumen dicatat dalam laporan JSON RFC 8259.
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

from tools.quant.quant_algo import (
    dequantize_tensor_q32,
    quantize_tensor_f11a,
)
from tools.quant.quant_format import QUANT_HEADER_SIZE, unpack_4bit_pair

torch.manual_seed(42)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cpu")


def sha256_of_file(path: str) -> str:
    """Menghitung digest SHA256 berkas."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def fail_invalid(
    output_report: str,
    reason: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Menulis laporan dengan status INVALID saat terjadi pin mismatch/drift."""
    payload = {
        "status": "INVALID",
        "reason": reason,
        "message": message,
        "details": details or {},
    }
    if output_report:
        out_dir = os.path.dirname(os.path.abspath(output_report))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.exit(1)


def parse_model_config(model_dir: str) -> dict[str, Any]:
    """Membaca konfigurasi model dari model_config.json atau config.json."""
    cfg_path = os.path.join(model_dir, "model_config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found in {model_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    cfg: dict[str, Any] = {}
    cfg["hidden_size"] = int(raw_cfg.get("hidden_size", 2048))
    cfg["num_hidden_layers"] = int(raw_cfg.get("num_hidden_layers", 24))
    cfg["num_attention_heads"] = int(raw_cfg.get("num_attention_heads", 16))
    cfg["num_experts"] = int(raw_cfg.get("num_experts", 60))
    cfg["num_experts_per_tok"] = int(raw_cfg.get("num_experts_per_tok", 4))
    cfg["moe_intermediate_size"] = int(raw_cfg.get("moe_intermediate_size", 1408))
    cfg["shared_expert_intermediate_size"] = int(
        raw_cfg.get("shared_expert_intermediate_size", 5632)
    )
    cfg["vocab_size"] = int(raw_cfg.get("vocab_size", 151936))
    cfg["rms_norm_eps"] = float(raw_cfg.get("rms_norm_eps", 1e-6))
    cfg["rope_theta"] = float(raw_cfg.get("rope_theta", 1000000.0))
    cfg["norm_topk_prob"] = bool(raw_cfg.get("norm_topk_prob", False))
    return cfg


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Formula F7 rotate_half: [-x_{half..}, x_{..half}]."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def forward_layer_attention(
    x: torch.Tensor,
    layer_idx: int,
    cfg: dict[str, Any],
    get_tensor: Any,
) -> torch.Tensor:
    pfx = f"model.layers.{layer_idx}."
    seq_len = x.shape[0]
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    head_dim = hidden_size // num_heads
    eps = cfg["rms_norm_eps"]
    base_theta = cfg["rope_theta"]

    w_in_norm = get_tensor(f"{pfx}input_layernorm.weight")
    var_in = torch.mean(x**2, dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var_in + eps) * w_in_norm

    wq = get_tensor(f"{pfx}self_attn.q_proj.weight")
    wk = get_tensor(f"{pfx}self_attn.k_proj.weight")
    wv = get_tensor(f"{pfx}self_attn.v_proj.weight", required=False)
    if wv is None:
        wv = wk

    q = torch.matmul(x_norm, wq.t())
    k = torch.matmul(x_norm, wk.t())
    v = torch.matmul(x_norm, wv.t())

    bq = get_tensor(f"{pfx}self_attn.q_proj.bias", required=False)
    bk = get_tensor(f"{pfx}self_attn.k_proj.bias", required=False)
    bv = get_tensor(f"{pfx}self_attn.v_proj.bias", required=False)
    if bq is not None:
        q = q + bq
    if bk is not None:
        k = k + bk
    if bv is not None:
        v = v + bv

    qh = q.view(seq_len, num_heads, head_dim)
    kh = k.view(seq_len, num_heads, head_dim)

    inv_freq = 1.0 / (
        base_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t_pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t_pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
    cos_emb = emb.cos()
    sin_emb = emb.sin()

    q_rot = (qh * cos_emb) + (rotate_half(qh) * sin_emb)
    k_rot = (kh * cos_emb) + (rotate_half(kh) * sin_emb)

    qh = q_rot.permute(1, 0, 2)
    kh = k_rot.permute(1, 0, 2)
    vh = v.view(seq_len, num_heads, head_dim).permute(1, 0, 2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    scores = scores + mask
    attn_probs = F.softmax(scores, dim=-1)
    context = torch.matmul(attn_probs, vh)
    context = context.permute(1, 0, 2).contiguous().view(seq_len, hidden_size)

    wo = get_tensor(f"{pfx}self_attn.o_proj.weight")
    out = torch.matmul(context, wo.t())
    return x + out


def forward_layer_moe(
    x: torch.Tensor,
    layer_idx: int,
    cfg: dict[str, Any],
    get_tensor: Any,
) -> torch.Tensor:
    pfx = f"model.layers.{layer_idx}."
    seq_len = x.shape[0]
    hidden_size = cfg["hidden_size"]
    top_k = cfg["num_experts_per_tok"]
    eps = cfg["rms_norm_eps"]

    w_post_norm = get_tensor(f"{pfx}post_attention_layernorm.weight", required=False)
    if w_post_norm is None:
        w_post_norm = get_tensor(f"{pfx}input_layernorm.weight")

    var_post = torch.mean(x**2, dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var_post + eps) * w_post_norm

    w_gate = get_tensor(f"{pfx}mlp.gate.weight")
    router_logits = torch.matmul(x_norm, w_gate.t())

    router_probs = F.softmax(router_logits, dim=-1)
    topk_probs, topk_indices = torch.topk(router_probs, top_k, dim=-1)
    if cfg["norm_topk_prob"]:
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    y_routed = torch.zeros_like(x)
    for t_idx in range(seq_len):
        xt = x_norm[t_idx : t_idx + 1]
        for k in range(top_k):
            e_idx = int(topk_indices[t_idx, k].item())
            p_val = float(topk_probs[t_idx, k].item())

            wg = get_tensor(
                f"{pfx}mlp.experts.{e_idx}.gate_proj.weight", required=False
            )
            if wg is not None:
                wu = get_tensor(f"{pfx}mlp.experts.{e_idx}.up_proj.weight")
                wd = get_tensor(f"{pfx}mlp.experts.{e_idx}.down_proj.weight")
                g_act = F.silu(torch.matmul(xt, wg.t()))
                u_act = torch.matmul(xt, wu.t())
                e_out = torch.matmul(g_act * u_act, wd.t())[0]
            else:
                w1 = get_tensor(f"{pfx}mlp.experts.{e_idx}.w1.weight", required=False)
                if w1 is not None:
                    w2 = get_tensor(
                        f"{pfx}mlp.experts.{e_idx}.w2.weight", required=False
                    )
                    if w2 is None:
                        e_out = torch.matmul(xt, w1.t())[0]
                    else:
                        e_in = F.silu(torch.matmul(xt, w1.t()))
                        e_out = torch.matmul(e_in, w2.t())[0]
                else:
                    e_out = torch.zeros(hidden_size, device=DEVICE)

            y_routed[t_idx] += p_val * e_out

    w_sh_gate_proj = get_tensor(
        f"{pfx}mlp.shared_expert.gate_proj.weight", required=False
    )
    if w_sh_gate_proj is not None:
        w_sh_up_proj = get_tensor(f"{pfx}mlp.shared_expert.up_proj.weight")
        w_sh_down_proj = get_tensor(f"{pfx}mlp.shared_expert.down_proj.weight")
        w_sh_gate = get_tensor(f"{pfx}mlp.shared_expert_gate.weight", required=False)

        sh_g = F.silu(torch.matmul(x_norm, w_sh_gate_proj.t()))
        sh_u = torch.matmul(x_norm, w_sh_up_proj.t())
        e_sh = torch.matmul(sh_g * sh_u, w_sh_down_proj.t())
        if w_sh_gate is not None:
            g_sh = torch.sigmoid(torch.matmul(x_norm, w_sh_gate.t()))
            y_shared = g_sh * e_sh
        else:
            y_shared = e_sh
        moe_out = y_routed + y_shared
    else:
        w_sh_w1 = get_tensor(f"{pfx}mlp.shared_expert.w1.weight", required=False)
        if w_sh_w1 is not None:
            y_shared = torch.matmul(x_norm, w_sh_w1.t())
            moe_out = y_routed + y_shared
        else:
            moe_out = y_routed

    return x + moe_out


def forward_model(
    token_ids: list[int],
    cfg: dict[str, Any],
    get_tensor: Any,
) -> torch.Tensor:
    """Full forward pass menghasilkan logits FP32 berukuran [seq_len, vocab_size]."""
    tokens_t = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
    w_embed = get_tensor("model.embed_tokens.weight")
    x = torch.embedding(w_embed, tokens_t).clone()

    num_layers = cfg["num_hidden_layers"]
    for l_idx in range(num_layers):
        x = forward_layer_attention(x, l_idx, cfg, get_tensor)
        x = forward_layer_moe(x, l_idx, cfg, get_tensor)

    w_norm = get_tensor("model.norm.weight")
    eps = cfg["rms_norm_eps"]
    var_f = torch.mean(x**2, dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var_f + eps) * w_norm

    w_lm = get_tensor("lm_head.weight")
    logits = torch.matmul(x_norm, w_lm.t())
    return logits


class ModelLoaderBF16:
    """Pemuat tensor BF16 dari safetensors ke FP32."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.shards: dict[str, Any] = {}
        self.weight_map: dict[str, str] = {}

        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            self.weight_map = idx.get("weight_map", {})
        else:
            st_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
            st_files.sort()
            for sf in st_files:
                p = os.path.join(model_dir, sf)
                with safe_open(p, framework="pt", device="cpu") as handle:
                    for k in handle.keys():
                        self.weight_map[k] = sf

    def get_tensor(self, name: str, required: bool = True) -> torch.Tensor | None:
        if name not in self.weight_map:
            if required:
                raise KeyError(f"Tensor {name} missing from model")
            return None
        sf = self.weight_map[name]
        if sf not in self.shards:
            p = os.path.join(self.model_dir, sf)
            self.shards[sf] = safe_open(p, framework="pt", device="cpu")
        return self.shards[sf].get_tensor(name).float()


class ModelLoaderQuant:
    """Pemuat tensor 4-bit dari berkas biner .bin atau roundtrip on-the-fly."""

    def __init__(self, quant_dir_or_bf16: str, bf16_loader: ModelLoaderBF16):
        self.tensors: dict[str, torch.Tensor] = {}
        bin_path = os.path.join(quant_dir_or_bf16, "quant_model.bin")
        if not os.path.exists(bin_path) and quant_dir_or_bf16.endswith(".bin"):
            bin_path = quant_dir_or_bf16

        if os.path.exists(bin_path):
            self._load_from_bin(bin_path)
        else:
            self._roundtrip_from_bf16(bf16_loader)

    def _load_from_bin(self, bin_path: str) -> None:
        with open(bin_path, "rb") as f:
            hdr_raw = f.read(QUANT_HEADER_SIZE)
            hdr_json_str = hdr_raw.split(b"\x00")[0].decode("utf-8")
            hdr_json = json.loads(hdr_json_str)
            num_tensors = hdr_json["num_tensors"]
            group_size = hdr_json["quantization"]["group_size"]

            for _ in range(num_tensors):
                len_b = f.read(4)
                meta_len = struct.unpack("<I", len_b)[0]
                meta_str = f.read(meta_len).decode("utf-8")
                meta = json.loads(meta_str)

                num_groups = meta["num_groups"]
                s_bytes = num_groups * 2
                n_elem = 1
                for d in meta["shape"]:
                    n_elem *= d
                w_bytes = (n_elem + 1) // 2

                scales_raw = f.read(s_bytes)
                packed_raw = f.read(w_bytes)

                scales_f16: list[float] = []
                for g in range(num_groups):
                    u = struct.unpack_from("<H", scales_raw, g * 2)[0]
                    f16_val = np.frombuffer(struct.pack("<H", u), dtype=np.float16)[0]
                    scales_f16.append(float(f16_val))

                q_weights: list[int] = []
                for b_i in range(w_bytes):
                    b = packed_raw[b_i]
                    w0, w1 = unpack_4bit_pair(b)
                    q_weights.append(w0)
                    if len(q_weights) < n_elem:
                        q_weights.append(w1)

                f32_list = dequantize_tensor_q32(
                    scales_f16, q_weights[:n_elem], group_size
                )
                t = torch.tensor(f32_list, dtype=torch.float32).view(meta["shape"])
                self.tensors[meta["name"]] = t

    def _roundtrip_from_bf16(self, bf16_loader: ModelLoaderBF16) -> None:
        for name in bf16_loader.weight_map.keys():
            raw_t = bf16_loader.get_tensor(name, required=True)
            shape = list(raw_t.shape)
            w_flat = raw_t.view(-1).tolist()
            g_sz = 64 if len(w_flat) % 128 != 0 and len(w_flat) % 64 == 0 else 128
            if len(w_flat) % g_sz == 0:
                scales, q_weights, _ = quantize_tensor_f11a(w_flat, g_sz)
                dq32 = dequantize_tensor_q32(scales, q_weights, g_sz)
                self.tensors[name] = torch.tensor(dq32, dtype=torch.float32).view(shape)
            else:
                self.tensors[name] = raw_t

    def get_tensor(self, name: str, required: bool = True) -> torch.Tensor | None:
        if name in self.tensors:
            return self.tensors[name]
        if required:
            raise KeyError(f"Quantized tensor {name} not found")
        return None


def verify_golden_pins(
    golden_pins_path: str,
    corpus_path: str,
    corpus_data: dict[str, Any],
    model_dir: str,
    output_report: str,
) -> None:
    """Verifikasi golden pins. Ketidakcocokan memicu exit status INVALID."""
    if not os.path.exists(golden_pins_path):
        fail_invalid(
            output_report,
            "pins_missing",
            f"Golden pins file not found: {golden_pins_path}",
        )

    try:
        with open(golden_pins_path, "r", encoding="utf-8") as f:
            pins = json.load(f)
    except Exception as e:
        fail_invalid(
            output_report,
            "pins_parse_error",
            f"Failed to parse golden pins: {e}",
        )

    actual_corpus_sha = sha256_of_file(corpus_path)
    expected_corpus_sha = pins.get("corpus", {}).get("sha256")
    if not expected_corpus_sha or actual_corpus_sha != expected_corpus_sha:
        exp_disp = str(expected_corpus_sha)[:16] if expected_corpus_sha else "None"
        fail_invalid(
            output_report,
            "corpus_sha256_mismatch",
            f"Corpus SHA256 mismatch: exp {exp_disp}..., "
            f"got {actual_corpus_sha[:16]}...",
            details={"expected": expected_corpus_sha, "actual": actual_corpus_sha},
        )

    tok_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        actual_tok_sha = sha256_of_file(tok_path)
        expected_tok_sha = pins.get("tokenizer", {}).get("sha256")
        if expected_tok_sha and actual_tok_sha != expected_tok_sha:
            fail_invalid(
                output_report,
                "tokenizer_sha256_mismatch",
                f"Tokenizer mismatch: exp {expected_tok_sha}, got {actual_tok_sha}",
                details={"expected": expected_tok_sha, "actual": actual_tok_sha},
            )

        tok = Tokenizer.from_file(tok_path)
        for doc in corpus_data.get("corpus", []):
            enc = tok.encode(doc["text"], add_special_tokens=False)
            if enc.ids != doc["token_ids"]:
                fail_invalid(
                    output_report,
                    "token_drift",
                    f"Tokenization drift detected on doc {doc.get('id')}",
                    details={"doc_id": doc.get("id")},
                )


def evaluate_ppl_and_agreement(
    model_bf16_dir: str,
    model_quant_dir: str,
    corpus_path: str,
    golden_pins_path: str,
    output_report: str,
    max_docs: int | None = None,
    skip_pins: bool = False,
) -> None:
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    if not skip_pins:
        verify_golden_pins(
            golden_pins_path,
            corpus_path,
            corpus_data,
            model_bf16_dir,
            output_report,
        )

    docs = corpus_data.get("corpus", [])
    if max_docs is not None and max_docs > 0:
        docs = docs[:max_docs]

    cfg_bf16 = parse_model_config(model_bf16_dir)
    loader_bf16 = ModelLoaderBF16(model_bf16_dir)
    loader_quant = ModelLoaderQuant(model_quant_dir, loader_bf16)

    total_n_pred = 0
    total_matching_argmax = 0
    total_sum_ln_p_bf16 = 0.0
    total_sum_ln_p_quant = 0.0

    diagnostics = []

    vocab_size = cfg_bf16["vocab_size"]
    for doc in docs:
        doc_id = doc["id"]
        token_ids = [t % vocab_size for t in doc["token_ids"]]
        seq_len = len(token_ids)
        if seq_len < 2:
            continue

        logits_bf16 = forward_model(token_ids, cfg_bf16, loader_bf16.get_tensor)
        logits_quant = forward_model(token_ids, cfg_bf16, loader_quant.get_tensor)

        n_pred_doc = seq_len - 1
        doc_matching = 0
        doc_ln_p_bf16 = 0.0
        doc_ln_p_quant = 0.0

        for i in range(1, seq_len):
            target_tok = token_ids[i]
            l_b = logits_bf16[i - 1]
            l_q = logits_quant[i - 1]

            log_probs_b = F.log_softmax(l_b, dim=-1)
            log_probs_q = F.log_softmax(l_q, dim=-1)

            lp_b = float(log_probs_b[target_tok].item())
            lp_q = float(log_probs_q[target_tok].item())

            doc_ln_p_bf16 += lp_b
            doc_ln_p_quant += lp_q

            pred_b = int(torch.argmax(l_b).item())
            pred_q = int(torch.argmax(l_q).item())
            if pred_b == pred_q:
                doc_matching += 1

        total_n_pred += n_pred_doc
        total_matching_argmax += doc_matching
        total_sum_ln_p_bf16 += doc_ln_p_bf16
        total_sum_ln_p_quant += doc_ln_p_quant

        doc_ppl_b = math.exp(-doc_ln_p_bf16 / n_pred_doc)
        doc_ppl_q = math.exp(-doc_ln_p_quant / n_pred_doc)
        doc_delta = doc_ppl_q - doc_ppl_b
        doc_agr = doc_matching / n_pred_doc

        diagnostics.append(
            {
                "id": doc_id,
                "n_pred": n_pred_doc,
                "ppl_bf16": round(doc_ppl_b, 4),
                "ppl_quant": round(doc_ppl_q, 4),
                "delta_ppl": round(doc_delta, 4),
                "agreement": round(doc_agr, 4),
            }
        )

    ppl_bf16_global = math.exp(-total_sum_ln_p_bf16 / total_n_pred)
    ppl_quant_global = math.exp(-total_sum_ln_p_quant / total_n_pred)
    delta_ppl_global = ppl_quant_global - ppl_bf16_global
    global_agreement = total_matching_argmax / total_n_pred

    gate_passed = (delta_ppl_global <= 0.5) and (global_agreement >= 0.95)
    status_verdict = "PASS" if gate_passed else "FAIL"

    report_payload = {
        "status": status_verdict,
        "run_id": "M6-PPL-EVAL",
        "model": "qwen1.5-moe-a2.7b-chat",
        "corpus_sha256": sha256_of_file(corpus_path),
        "n_pred_total": total_n_pred,
        "ppl_bf16": round(ppl_bf16_global, 4),
        "ppl_quant": round(ppl_quant_global, 4),
        "delta_ppl": round(delta_ppl_global, 4),
        "argmax_agreement": round(global_agreement, 4),
        "gate_delta_ppl_max": 0.5,
        "gate_argmax_agreement_min": 0.95,
        "gate_passed": gate_passed,
        "diagnostics": diagnostics,
    }

    out_dir = os.path.dirname(os.path.abspath(output_report))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
        f.write("\n")

    print(
        json.dumps(
            {
                "status": status_verdict,
                "delta_ppl": round(delta_ppl_global, 4),
                "argmax_agreement": round(global_agreement, 4),
                "n_pred_total": total_n_pred,
                "gate_passed": gate_passed,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle PPL Measurement & Conformance Runner"
    )
    parser.add_argument(
        "--model-bf16",
        required=True,
        help="Path to BF16 model directory",
    )
    parser.add_argument(
        "--model-quant",
        required=True,
        help="Path to Quantized model directory or file",
    )
    parser.add_argument(
        "--corpus",
        default="tools/fixtures/m6_ppl_corpus.json",
        help="Path to m6_ppl_corpus.json",
    )
    parser.add_argument(
        "--golden-pins",
        default="tools/fixtures/ppl_golden_pins.json",
        help="Path to ppl_golden_pins.json",
    )
    parser.add_argument(
        "--output-report",
        required=True,
        help="Path to output ppl_report.json",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to evaluate (for testing)",
    )
    parser.add_argument(
        "--skip-pins",
        action="store_true",
        help="Skip pins verification (for ad-hoc/fixture testing)",
    )
    args = parser.parse_args()

    evaluate_ppl_and_agreement(
        model_bf16_dir=args.model_bf16,
        model_quant_dir=args.model_quant,
        corpus_path=args.corpus,
        golden_pins_path=args.golden_pins,
        output_report=args.output_report,
        max_docs=args.max_docs,
        skip_pins=args.skip_pins,
    )


if __name__ == "__main__":
    main()
