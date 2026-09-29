#!/usr/bin/env python3
"""Merge a LoRA state dict (from train_sft_lora.py) into the BF16 base checkpoint.

Shard-by-shard on CPU: every base tensor is copied unchanged (incl. vision
tower and mtp.* tensors) except the LoRA'd linear weights, which receive
W += (alpha/r) * B @ A computed in fp32 and cast back to bf16. Output is a
complete HF model dir (same shard layout, index, configs, tokenizer, chat
template) that vLLM loads exactly like the parent.

    merge_lora.py --base <bf16 snapshot> --lora <step_dir/lora_state_dict.safetensors> \
                  --alpha 128 --r 64 --out ~/hiw_models/student_v1
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LORA_RE = re.compile(r"^(?:base_model\.model\.)?(.*)\.lora_(A|B)\.(?:default\.)?weight$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = load_file(args.lora)
    pairs = {}
    for k, v in sd.items():
        m = LORA_RE.match(k.replace("._fsdp_wrapped_module", ""))
        if not m:
            raise ValueError(f"unrecognised lora key {k}")
        pairs.setdefault(m.group(1), {})[m.group(2)] = v
    scale = args.alpha / args.r
    print(f"{len(pairs)} LoRA modules, scale={scale}")

    os.makedirs(args.out, exist_ok=True)
    index = json.load(open(os.path.join(args.base, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    merged, seen = 0, set()
    for shard in shards:
        tensors = {}
        with safe_open(os.path.join(args.base, shard), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                mod = k[:-len(".weight")] if k.endswith(".weight") else None
                if mod in pairs:
                    A = pairs[mod]["A"].float()
                    B = pairs[mod]["B"].float()
                    delta = (B @ A) * scale
                    assert delta.shape == t.shape, (k, delta.shape, t.shape)
                    t = (t.float() + delta).to(t.dtype)
                    merged += 1
                    seen.add(mod)
                tensors[k] = t.contiguous()
        save_file(tensors, os.path.join(args.out, shard), metadata={"format": "pt"})
        print(f"wrote {shard} ({len(tensors)} tensors)")
    missing = set(pairs) - seen
    if missing:
        raise RuntimeError(f"{len(missing)} LoRA modules not found in base: {sorted(missing)[:5]}")
    for fn in os.listdir(args.base):
        if fn.endswith(".safetensors") or fn.startswith("."):
            continue
        src = os.path.join(args.base, fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, fn))
    json.dump({"base": args.base, "lora": args.lora, "r": args.r, "alpha": args.alpha, "merged_modules": merged},
              open(os.path.join(args.out, "hiw_merge_info.json"), "w"), indent=1)
    print(f"merged {merged} weight tensors -> {args.out}")


if __name__ == "__main__":
    main()
