#!/usr/bin/env python3
"""Quantize a merged BF16 Qwen3.6-27B checkpoint into the SAME layout as the
official Qwen/Qwen3.6-27B-FP8 checkpoint (block-wise 128x128 float8_e4m3fn
weights + bf16 `weight_scale_inv`), using the official FP8 checkpoint as the
spec: a tensor is quantized iff it is F8_E4M3 there; all other tensors are
copied from the BF16 source. config.json (incl. quantization_config) and the
tokenizer/template files are taken from the FP8 reference so vLLM serves the
student exactly like the teacher (1 GPU/replica, same kernels).

    quantize_fp8_blockwise.py --src <merged bf16 dir> --ref <official FP8 snapshot> --out <dir>
    quantize_fp8_blockwise.py --verify --src <bf16 parent> --ref <official FP8 snapshot>
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
BLOCK = 128


def quantize_block(w: torch.Tensor, block=BLOCK):
    """w (out, in) bf16 -> (q float8_e4m3fn, scale_inv bf16 [out/b, in/b]) with w ~= q * scale."""
    out_f, in_f = w.shape
    assert out_f % block == 0 and in_f % block == 0, w.shape
    wf = w.float().reshape(out_f // block, block, in_f // block, block)
    amax = wf.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(out_f, in_f), scale.reshape(out_f // block, in_f // block).to(torch.bfloat16)


def dequantize_block(q: torch.Tensor, scale_inv: torch.Tensor, block=BLOCK):
    out_f, in_f = q.shape
    qf = q.float().reshape(out_f // block, block, in_f // block, block)
    s = scale_inv.float().reshape(out_f // block, 1, in_f // block, 1)
    return (qf * s).reshape(out_f, in_f)


def load_index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    return json.load(open(p))["weight_map"]


def verify(args):
    ref_map = load_index(args.ref)
    src_map = load_index(args.src)
    keys = [k for k in ref_map if k.endswith(".weight") and (k + "_scale_inv") in ref_map]
    print(f"{len(keys)} quantized weights in reference")
    for k in keys[:6] + [k for k in keys if "layers.3.self_attn.q_proj" in k]:
        with safe_open(os.path.join(args.ref, ref_map[k]), framework="pt") as f:
            q = f.get_tensor(k); s = f.get_tensor(k + "_scale_inv")
        with safe_open(os.path.join(args.src, src_map[k]), framework="pt") as f:
            w = f.get_tensor(k)
        deq = dequantize_block(q, s)
        err = (deq - w.float()).abs()
        rel = err.norm() / w.float().norm()
        q2, s2 = quantize_block(w)
        same_q = (q2.view(torch.uint8) == q.view(torch.uint8)).float().mean().item()
        same_s = (s2 == s).float().mean().item()
        srel = ((s2.float() - s.float()).abs() / s.float().abs().clamp(min=1e-12)).max().item()
        print(f"{k}: dequant rel err {rel:.4e} | requant: identical q {same_q:.4%}, identical scale {same_s:.4%}, max scale rel diff {srel:.2e}")


def convert(args):
    ref_map = load_index(args.ref)
    src_map = load_index(args.src)
    os.makedirs(args.out, exist_ok=True)
    # group reference tensors by shard to reproduce the shard layout
    by_shard = {}
    for k, sh in ref_map.items():
        by_shard.setdefault(sh, []).append(k)
    src_cache = {}

    def get_src(k):
        sh = src_map[k]
        if sh not in src_cache:
            src_cache.clear()
            with safe_open(os.path.join(args.src, sh), framework="pt") as f:
                src_cache[sh] = {kk: f.get_tensor(kk) for kk in f.keys()}
        return src_cache[sh][k]

    n_q = n_c = 0
    for shard in sorted(by_shard):
        out = {}
        for k in sorted(by_shard[shard]):
            if k.endswith("_scale_inv"):
                continue  # produced together with its weight
            if (k + "_scale_inv") in ref_map:
                w = get_src(k)
                q, s = quantize_block(w)
                out[k] = q.contiguous(); out[k + "_scale_inv"] = s.contiguous(); n_q += 1
            else:
                out[k] = get_src(k).contiguous(); n_c += 1
        save_file(out, os.path.join(args.out, shard), metadata={"format": "pt"})
        print(f"wrote {shard}: {len(out)} tensors")
    for fn in os.listdir(args.ref):
        if fn.endswith(".safetensors") or fn.startswith("."):
            continue
        p = os.path.join(args.ref, fn)
        if os.path.isfile(p):
            shutil.copy(p, os.path.join(args.out, fn))
    for fn in ("hiw_merge_info.json",):
        p = os.path.join(args.src, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out, fn))
    json.dump({"src": args.src, "ref": args.ref, "quantized": n_q, "copied": n_c},
              open(os.path.join(args.out, "hiw_quant_info.json"), "w"), indent=1)
    print(f"done: quantized {n_q}, copied {n_c} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)
    else:
        assert args.out
        convert(args)


if __name__ == "__main__":
    main()
