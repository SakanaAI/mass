#!/usr/bin/env python3
"""LoRA SFT with task-uniform, role-balanced sampling and validation loss."""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

if "pp_ddp" in sys.argv:  # each rank gets its own slice of GPUs for a naive device_map pipeline
    _g = int(sys.argv[sys.argv.index("--gpus-per-rank") + 1]) if "--gpus-per-rank" in sys.argv else 2
    _lr = int(os.environ.get("LOCAL_RANK", "0"))
    _vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    _all = _vis.split(",") if _vis else [str(i) for i in range(8)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_all[_lr * _g:(_lr + 1) * _g])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE = -100


def log(msg, rank=0):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="BF16 model dir (HF snapshot)")
    ap.add_argument("--data", required=True, help="windows JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", choices=["fsdp2", "pp", "pp_ddp"], default="fsdp2")
    ap.add_argument("--gpus-per-rank", type=int, default=2, help="pp_ddp: GPUs per replica (device_map pipeline)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps (default: ceil(epochs*windows/batch))")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=4, help="micro-steps per optimizer step PER RANK")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)$")
    ap.add_argument("--sampler", choices=["condition", "task_role"], default="task_role")
    ap.add_argument("--p-orch", type=float, default=1.0 / 3.0, help="task_role sampler: probability of drawing an orchestrator window when the task has both roles")
    ap.add_argument("--cond-weight", action="append", default=[], help="condition sampler: condition=weight (default 1)")
    ap.add_argument("--exclude-condition", action="append", default=[])
    ap.add_argument("--val-episode-regex", default=None, help="windows whose episode matches are held out")
    ap.add_argument("--val-data", default=None, help="separate windows JSONL used for validation (per-episode/role/task losses are logged)")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-at-start", action="store_true", help="run a full validation sweep before the first optimizer step")
    ap.add_argument("--max-len", type=int, default=None, help="truncate windows (debug)")
    ap.add_argument("--ce-chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from (LoRA + step + optimizer state if present)")
    ap.add_argument("--init-lora", default=None, help="lora_state_dict.safetensors to initialise from (fresh schedule, step 0)")
    ap.add_argument("--no-gradient-checkpointing", dest="grad_ckpt", action="store_false")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--tiny-config", default=None, help="smoke test: build a random tiny model from this config json instead of loading --model")
    ap.add_argument("--dry-run-steps", type=int, default=None, help="stop after N optimizer steps")
    ap.add_argument("--check-loss", action="store_true", help="debug: compare chunked CE with full-logit CE on the first window")
    ap.add_argument("--no-fla", action="store_true", help="force the torch fallback for the gated-delta-rule kernels (auto on CPU)")
    ap.add_argument("--autocast-bf16", action="store_true", help="run forward/backward under bf16 autocast (LoRA branch computes in bf16; fp32 master adapters stay in the optimizer)")
    return ap.parse_args()


# ----------------------------------------------------------------------------- data
def role_group(r):
    role = r.get("role")
    if role is None:
        role = "orchestrator" if r.get("is_main") else "worker"
    return "orch" if role == "orchestrator" else "nonmain"


def task_of(r):
    if r.get("task") is not None:
        return int(r["task"])
    import re
    m = re.match(r"t(\d+)_", r.get("condition", ""))
    return int(m.group(1)) if m else -1


def load_windows(path, args):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] in args.exclude_condition:
                continue
            if args.max_len:  # debug crop: keep the TAIL so label tokens survive
                r["input_ids"] = r["input_ids"][-args.max_len:]
                r["labels"] = r["labels"][-args.max_len:]
            if not any(l != IGNORE for l in r["labels"][1:]):
                continue
            r["_task"] = task_of(r); r["_rg"] = role_group(r)
            rows.append(r)
    return rows


def split_val(rows, regex):
    if not regex:
        return rows, []
    import re
    pat = re.compile(regex)
    tr = [r for r in rows if not pat.search(r["episode"])]
    va = [r for r in rows if pat.search(r["episode"])]
    return tr, va


class WeightedSampler:
    """draw condition ~ weights, then uniform window within the condition."""

    def __init__(self, rows, cond_weights, seed):
        self.by_cond = defaultdict(list)
        for i, r in enumerate(rows):
            self.by_cond[r["condition"]].append(i)
        self.conds = sorted(self.by_cond)
        w = [cond_weights.get(c, 1.0) for c in self.conds]
        s = sum(w)
        self.probs = [x / s for x in w]
        self.seed = seed

    def describe(self):
        return f"condition-uniform over {len(self.conds)} conditions"

    def draw(self, rank, step):
        rng = random.Random(f"{self.seed}-{rank}-{step}")
        c = rng.choices(self.conds, weights=self.probs, k=1)[0]
        return rng.choice(self.by_cond[c])


class TaskRoleSampler:
    """draw a task uniformly, then a role (orchestrator with p_orch when both exist), then a window uniformly within (task, role)."""

    def __init__(self, rows, p_orch, seed):
        self.by = defaultdict(lambda: {"orch": [], "nonmain": []})
        for i, r in enumerate(rows):
            self.by[r["_task"]][r["_rg"]].append(i)
        self.tasks = sorted(self.by)
        self.p_orch = p_orch
        self.seed = seed

    def describe(self):
        return "task-uniform over %s; role P(orch)=%.4f; per task orch/nonmain windows: %s" % (
            self.tasks, self.p_orch, {t: (len(self.by[t]["orch"]), len(self.by[t]["nonmain"])) for t in self.tasks})

    def draw(self, rank, step):
        rng = random.Random(f"{self.seed}-{rank}-{step}")
        t = rng.choice(self.tasks)
        have = [g for g in ("orch", "nonmain") if self.by[t][g]]
        if len(have) == 2:
            g = "orch" if rng.random() < self.p_orch else "nonmain"
        else:
            g = have[0]
        return rng.choice(self.by[t][g])


# ----------------------------------------------------------------------------- model
def build_model(args, rank):
    if args.no_fla or not torch.cuda.is_available():
        sys.modules["fla"] = None
        sys.modules["causal_conv1d"] = None
        log("fla/causal_conv1d hidden -> torch fallback kernels", rank)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    if args.tiny_config:
        cfg = AutoConfig.from_pretrained(args.tiny_config)
        torch.manual_seed(0)
        model = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16)
        log(f"tiny model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", rank)
    else:
        kw = dict(dtype=torch.bfloat16, attn_implementation=args.attn)
        if args.parallel in ("pp", "pp_ddp"):
            kw["device_map"] = "auto" if torch.cuda.is_available() else None
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, **kw)
    for p in model.parameters():
        p.requires_grad_(False)
    lm_head_w = model.lm_head.weight.detach().clone()
    model.lm_head = nn.Identity()
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    if rank == 0:
        model.print_trainable_parameters()
    return model, lm_head_w


def shard_fsdp2(model, device):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    world = torch.distributed.get_world_size()
    mesh = init_device_mesh("cuda", (world,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5VisionBlock
    n = 0
    for m in model.modules():
        if isinstance(m, (Qwen3_5DecoderLayer, Qwen3_5VisionBlock)):
            fully_shard(m, mesh=mesh, mp_policy=mp)
            n += 1
    fully_shard(model, mesh=mesh, mp_policy=mp)
    return model, n


def chunked_ce(hidden, lm_head_w, labels, chunk):
    T = hidden.shape[0]
    total = hidden.new_zeros((), dtype=torch.float32)
    w = lm_head_w.to(hidden.device)

    def f(h, y):
        logits = F.linear(h, w.to(h.dtype)).float()
        return F.cross_entropy(logits, y, ignore_index=IGNORE, reduction="sum")

    for i in range(0, T, chunk):
        y = labels[i:i + chunk]
        if (y != IGNORE).any():
            total = total + checkpoint(f, hidden[i:i + chunk], y, use_reentrant=False)
    return total


def forward_loss(model, lm_head_w, batch, args, device):
    ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)[None]
    labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)[None]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.autocast_bf16 and torch.cuda.is_available())):
        out = model(input_ids=ids, use_cache=False)
    hidden = out.logits[0]
    if hidden.dtype != torch.bfloat16:
        hidden = hidden.to(torch.bfloat16)
    h = hidden[:-1]
    y = labels[0, 1:].to(h.device)
    n_tok = int((y != IGNORE).sum().item())
    loss_sum = chunked_ce(h, lm_head_w, y, args.ce_chunk)
    return loss_sum, n_tok


# ----------------------------------------------------------------------------- ckpt
def lora_state_dict(model):
    sd = {}
    for k, v in model.state_dict().items():
        if "lora_" in k:
            t = v.full_tensor() if hasattr(v, "full_tensor") else v
            sd[k] = t.detach().to("cpu", dtype=torch.float32)
    return sd


def save_checkpoint(model, opt, args, step, out_dir, rank, extra):
    sd = lora_state_dict(model)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        from safetensors.torch import save_file
        save_file(sd, os.path.join(out_dir, "lora_state_dict.safetensors"))
        try:
            from peft import get_peft_model_state_dict  # noqa: F401
            adapter_dir = os.path.join(out_dir, "adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            model.peft_config["default"].save_pretrained(adapter_dir)
            peft_sd = {k.replace("._fsdp_wrapped_module", "").replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B."): v
                       for k, v in sd.items()}
            save_file(peft_sd, os.path.join(adapter_dir, "adapter_model.safetensors"))
        except Exception as e:  # noqa: BLE001
            print(f"adapter export failed: {e!r}", file=sys.stderr)
        if opt is not None:
            tmp = os.path.join(out_dir, "optimizer.pt.tmp")
            torch.save({k: v for k, v in opt.state_dict().items()}, tmp)
            os.replace(tmp, os.path.join(out_dir, "optimizer.pt"))
        json.dump({"step": step, **extra}, open(os.path.join(out_dir, "train_state.json"), "w"), indent=1)
        log(f"saved checkpoint step {step} -> {out_dir} ({len(sd)} lora tensors, optimizer state {'yes' if opt is not None else 'no'})", rank)


def load_lora_into(model, path):
    from safetensors.torch import load_file
    sd = load_file(path)
    own = model.state_dict()
    n = 0
    with torch.no_grad():
        for k, v in sd.items():
            if k not in own:
                continue
            p = dict(model.named_parameters()).get(k)
            if p is None:
                continue
            if hasattr(p, "full_tensor"):
                from torch.distributed.tensor import distribute_tensor
                dt = distribute_tensor(v.to(p.dtype), p.device_mesh, p.placements)
                p.copy_(dt)
            else:
                p.copy_(v.to(p.device, p.dtype))
            n += 1
    return n


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    dist = args.parallel in ("fsdp2", "pp_ddp")
    if args.parallel == "fsdp2":
        torch.distributed.init_process_group("nccl")
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    elif args.parallel == "pp_ddp":
        torch.cuda.set_device(0)
        torch.distributed.init_process_group("nccl")
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        device = torch.device("cuda", 0)
        log(f"pp_ddp: rank {rank}/{world} on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", rank)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    if rank == 0:
        json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)

    rows = load_windows(args.data, args)
    train_rows, val_rows = split_val(rows, args.val_episode_regex)
    if args.val_data:
        val_rows = load_windows(args.val_data, args)
    counts = Counter(r["condition"] for r in train_rows)
    log(f"windows: train={len(train_rows)} val={len(val_rows)} by condition={dict(counts)} by role={dict(Counter(r['_rg'] for r in train_rows))} by task={dict(Counter(r['_task'] for r in train_rows))}", rank)
    if args.sampler == "task_role":
        sampler = TaskRoleSampler(train_rows, args.p_orch, args.seed)
    else:
        cond_w = {}
        for spec in args.cond_weight:
            c, w = spec.split("=")
            cond_w[c] = float(w)
        sampler = WeightedSampler(train_rows, cond_w, args.seed)
    log(f"sampler: {sampler.describe()}", rank)

    model, lm_head_w = build_model(args, rank)
    if args.parallel == "fsdp2":
        model, n_shards = shard_fsdp2(model, device)
        log(f"FSDP2: sharded {n_shards} blocks over {world} ranks", rank)
        lm_head_w = lm_head_w.to(device)
    elif args.tiny_config or not torch.cuda.is_available():
        model = model.to(device)
        lm_head_w = lm_head_w.to(device)
    else:
        lm_head_w = lm_head_w.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, eps=1e-8)
    per_step = args.grad_accum * world
    total_steps = args.steps or max(1, math.ceil(args.epochs * len(train_rows) / per_step))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    start_step = 0
    resumed_opt = False
    if args.init_lora and not args.resume:
        n = load_lora_into(model, args.init_lora)
        log(f"initialised {n} lora tensors from {args.init_lora}", rank)
    if args.resume:
        st = json.load(open(os.path.join(args.resume, "train_state.json")))
        n = load_lora_into(model, os.path.join(args.resume, "lora_state_dict.safetensors"))
        start_step = st["step"]
        opt_p = os.path.join(args.resume, "optimizer.pt")
        if os.path.exists(opt_p):
            opt.load_state_dict(torch.load(opt_p, map_location="cpu"))
            resumed_opt = True
        log(f"resumed {n} lora tensors from {args.resume} at step {start_step} (optimizer state {'restored' if resumed_opt else 'RESET: no optimizer.pt'})", rank)
        if rank == 0:
            with open(os.path.join(args.out, "resume_log.jsonl"), "a") as f:
                f.write(json.dumps({"resumed_from": args.resume, "step": start_step, "optimizer_restored": resumed_opt, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")

    if args.check_loss and rank == 0 and not dist:
        with torch.no_grad():
            b = train_rows[0]
            ls, nt = forward_loss(model, lm_head_w, b, args, device)
            ids = torch.tensor(b["input_ids"], device=device)[None]
            hid = model(input_ids=ids, use_cache=False).logits[0][:-1]
            logits = F.linear(hid, lm_head_w.to(hid.device, hid.dtype)).float()
            y = torch.tensor(b["labels"][1:], device=logits.device)
            ref = F.cross_entropy(logits, y, ignore_index=IGNORE, reduction="sum")
            log(f"check-loss: chunked={ls.item():.6f} full={ref.item():.6f} tokens={nt} rel_diff={(abs(ls.item()-ref.item())/max(1e-9,ref.item())):.2e}", rank)
    log(f"training: total_steps={total_steps} windows/step={per_step} lr={args.lr} r={args.lora_r} alpha={args.lora_alpha}", rank)
    hist = open(os.path.join(args.out, "train_log.jsonl"), "a") if rank == 0 else None
    expo = open(os.path.join(args.out, "exposure_log.jsonl"), "a") if rank == 0 else None

    # validation keys: (episode, role) so per-episode / per-role / per-task losses can be reported
    vkeys = sorted({(r["episode"], r["_rg"]) for r in val_rows}); vidx = {k: i for i, k in enumerate(vkeys)}
    vtask = {r["episode"]: r["_task"] for r in val_rows}

    def run_validation(step_label):
        model.eval()
        vs, vt = 0.0, 0
        n_per = math.ceil(len(val_rows) / world)
        per = torch.zeros((max(1, len(vkeys)), 2), dtype=torch.float64, device=device)
        t0v = time.time()
        with torch.no_grad():
            for j in range(n_per):
                i = j * world + rank
                real = i < len(val_rows)
                r = val_rows[i] if real else val_rows[0]
                ls, nt = forward_loss(model, lm_head_w, r, args, device)
                if real:
                    vs += ls.item(); vt += nt
                    k = vidx[(r["episode"], r["_rg"])]
                    per[k, 0] += ls.item(); per[k, 1] += nt
        vstats = torch.tensor([vs, vt], dtype=torch.float64, device=device)
        if dist:
            torch.distributed.all_reduce(vstats); torch.distributed.all_reduce(per)
        by_ep, by_role, by_task = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
        for (ep, rg), k in vidx.items():
            s, n = per[k, 0].item(), per[k, 1].item()
            by_ep[ep][0] += s; by_ep[ep][1] += n; by_role[rg][0] += s; by_role[rg][1] += n; by_task[vtask[ep]][0] += s; by_task[vtask[ep]][1] += n
        fmt = lambda d: {str(k): round(v[0] / max(1.0, v[1]), 4) for k, v in d.items()}
        vrec = {"step": step_label, "val_loss": (vstats[0] / max(1.0, vstats[1])).item(), "val_tokens": int(vstats[1].item()),
                "val_by_role": fmt(by_role), "val_by_task": fmt(by_task), "val_per_episode": fmt(by_ep), "val_seconds": round(time.time() - t0v, 1)}
        log(json.dumps(vrec), rank)
        if hist:
            hist.write(json.dumps(vrec) + "\n"); hist.flush()
        model.train()

    model.train()
    if val_rows and args.val_at_start and start_step == 0:
        run_validation(0)
    t0 = time.time()
    exposure = Counter(); uniq = set()
    for step in range(start_step, total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_sum_acc = 0.0
        tok_acc = 0
        micro_losses = []
        drawn = []
        for micro in range(args.grad_accum):
            idx = sampler.draw(rank, step * args.grad_accum + micro)
            batch = train_rows[idx]
            loss_sum, n_tok = forward_loss(model, lm_head_w, batch, args, device)
            (loss_sum / max(1, n_tok) / args.grad_accum).backward()
            loss_sum_acc += loss_sum.item()
            tok_acc += n_tok
            micro_losses.append(loss_sum.item() / max(1, n_tok))
            drawn.append((idx, batch["_task"], batch["_rg"], n_tok))
        if args.parallel == "pp_ddp" and world > 1:
            grads = [p.grad for p in params if p.grad is not None]
            flat = torch.cat([g.detach().to(device, torch.float32).reshape(-1) for g in grads])
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.AVG)
            off = 0
            for g in grads:
                n = g.numel()
                g.copy_(flat[off:off + n].view_as(g).to(g.device, g.dtype)); off += n
            del flat
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        stats = torch.tensor([loss_sum_acc, tok_acc], dtype=torch.float64, device=device)
        if dist:
            torch.distributed.all_reduce(stats)
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, drawn)
            all_drawn = [d for g in gathered for d in g]
        else:
            all_drawn = drawn
        mean_loss = (stats[0] / max(1.0, stats[1])).item()
        if rank == 0:
            for idx, t, rg, nt in all_drawn:
                exposure[(t, rg, "draws")] += 1; exposure[(t, rg, "tokens")] += nt; uniq.add(idx)
            if expo:
                expo.write(json.dumps({"step": step + 1, "rows": [d[0] for d in all_drawn], "task_role": [f"{d[1]}:{d[2]}" for d in all_drawn], "tokens": [d[3] for d in all_drawn]}) + "\n"); expo.flush()
        if (step + 1) % args.log_every == 0:
            rec = {"step": step + 1, "loss": mean_loss, "lr": lr_at(step), "grad_norm": float(gn),
                   "tokens": int(stats[1].item()), "elapsed_s": round(time.time() - t0, 1)}
            if torch.cuda.is_available():
                rec["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
                rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
            log(json.dumps(rec), rank)
            if hist:
                hist.write(json.dumps(rec) + "\n"); hist.flush()
        if val_rows and ((step + 1) % args.val_every == 0 or step + 1 == total_steps):
            run_validation(step + 1)
        if (step + 1) % args.save_every == 0 or step + 1 == total_steps:
            summ = {"draws_since_start_or_resume": {f"{t}:{rg}": exposure[(t, rg, "draws")] for (t, rg, k) in list(exposure) if k == "draws"},
                    "tokens_since_start_or_resume": {f"{t}:{rg}": exposure[(t, rg, "tokens")] for (t, rg, k) in list(exposure) if k == "tokens"},
                    "unique_windows_since_start_or_resume": len(uniq)}
            save_checkpoint(model, opt, args, step + 1, os.path.join(args.out, f"step_{step + 1:05d}"), rank,
                            {"lr": lr_at(step), "loss": mean_loss, "total_steps": total_steps, "exposure": summ})
        if args.dry_run_steps and step + 1 - start_step >= args.dry_run_steps:
            log("dry run complete", rank)
            break
    if dist:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
