"""Portable orchestration of RHI -> post-training -> the next MASS cycle."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
from urllib.parse import urlparse

from .io import ROOT, append_record, package_path, read_json, records, write_json
from .ranking import bradley_terry


def inference_options(c):
    return dict(backend="local-vllm", context_window_tokens=262144,
                episode_max_output_tokens=32768, max_output_tokens=49152,
                context_safety_tokens=8192, reasoning_effort="medium") | c.get("inference", {})


def validate(c):
    tasks = c["tasks"]
    train, test, excluded = map(set, (tasks["train"], tasks["test"], tasks["excluded"]))
    if train & test or train & excluded or test & excluded:
        raise ValueError("Train, test, and excluded tasks must be disjoint")
    if train | excluded != set(tasks["train_candidates"]):
        raise ValueError("Training candidates must equal train + excluded tasks")
    if len(train) != len(tasks["train"]) or len(test) != len(tasks["test"]):
        raise ValueError("Duplicate task IDs")
    inference = inference_options(c)
    if inference["backend"] not in ("local-vllm", "openrouter"):
        raise ValueError("Inference backend must be local-vllm or openrouter")
    for name in ("context_window_tokens", "episode_max_output_tokens", "max_output_tokens", "context_safety_tokens"):
        if type(inference[name]) is not int or inference[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max(inference["episode_max_output_tokens"], inference["max_output_tokens"]) + inference["context_safety_tokens"] >= inference["context_window_tokens"]:
        raise ValueError("Output and safety token budgets must fit within context")
    if inference["backend"] == "openrouter":
        if c["endpoint"].rstrip("/") != "https://openrouter.ai/api/v1":
            raise ValueError("OpenRouter requires https://openrouter.ai/api/v1")
        c["endpoint"] = c["endpoint"].rstrip("/")
        if inference["reasoning_effort"] not in ("low", "medium", "high"):
            raise ValueError("OpenRouter reasoning_effort must be low, medium, or high")
    else:
        p = urlparse(c["endpoint"])
        if p.scheme != "http" or p.hostname not in ("127.0.0.1", "localhost") or p.path != "/v1" or not p.port:
            raise ValueError("Episode runner requires http://127.0.0.1:PORT/v1")
        if p.username or p.password or p.query or p.fragment or p.port == c["proxy_port"]:
            raise ValueError("Endpoint must have no credentials/query and must differ from the proxy port")
    if c["search"]["iterations"] < 1:
        raise ValueError("At least one RHI update is required")
    if c["collection"]["validation_rank"] != c["collection"]["train_ranks"] + 1:
        raise ValueError("Validation must immediately follow the selected training ranks")
    if c["collection"]["candidates"] < c["collection"]["validation_rank"]:
        raise ValueError("Too few collection candidates")
    if c["training"]["replicas"] * c["training"]["grad_accum"] != 3:
        raise ValueError("The paper recipe requires three windows per optimizer step")
    for q in train | test | excluded:
        for p in [ROOT / f"tasks/bare/query{q}.txt", ROOT / f"tasks/initial/query{q}_ourTeam.txt"]:
            if not p.is_file():
                raise FileNotFoundError(p)
    return c


def run(command, env=None):
    command = [str(x) for x in command]
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_root(c):
    return package_path(c["run_dir"])


def freeze_config(c):
    path = run_root(c) / "config.json"
    if path.exists() and read_json(path) != c:
        raise ValueError("Configuration differs from this run's saved config. Use a new run_dir.")
    if not path.exists():
        write_json(path, c)


def completed(logs):
    p = logs / "run_status.json"
    if not p.exists():
        return False
    st = read_json(p)
    return st.get("exit_code") == 0 and st.get("trace_has_success_result") is True


def episode(c, prompt, workspace, logs, seed, condition):
    if completed(logs):
        if (logs / "prompt.txt").read_text() != prompt.read_text():
            raise ValueError(f"Completed episode has a different prompt: {logs}")
        return
    # The shell runner refuses to overwrite partial/failed episodes. Keep the
    # evidence and use a new run directory after correcting an infrastructure failure.
    inference = inference_options(c)
    env = dict(os.environ, MODEL_ID=c["model_id"], PROXY_PORT=str(c["proxy_port"]),
               PROXY_PY=sys.executable, HIW_CONDITION=condition,
               MASS_BACKEND=inference["backend"], MODEL_BASE_URL=c["endpoint"],
               MODEL_CONTEXT_WINDOW=str(inference["context_window_tokens"]),
               MODEL_MAX_OUTPUT_TOKENS=str(inference["episode_max_output_tokens"]),
               MODEL_REASONING_EFFORT=inference["reasoning_effort"])
    run(["bash", ROOT / "runtime/run_episode.sh", prompt, workspace, logs,
         str(urlparse(c["endpoint"]).port or 443), str(seed)], env)


def query_name(q):
    return f"query{q}_ourTeam"


def workflow_prompt(c, q, version):
    if version == 0:
        return ROOT / f"tasks/initial/{query_name(q)}.txt"
    return run_root(c) / f"search/{query_name(q)}/updated_queries/{query_name(q)}-v{version}.txt"


def search(c, tasks):
    base = run_root(c)
    inference = inference_options(c)
    last = c["search"]["iterations"]
    for q in tasks:
        episode(c, ROOT / f"tasks/bare/query{q}.txt", base / f"reference/query{q}",
                base / f"logs/reference/query{q}", c["search"]["seed_base"] + q, f"t{q}_bare")
        output = base / f"search/{query_name(q)}"
        for k in range(last + 1):
            workspace = base / f"workspaces/v{k}/{query_name(q)}"
            logs = base / f"logs/search/v{k}/{query_name(q)}"
            episode(c, workflow_prompt(c, q, k), workspace, logs,
                    c["search"]["seed_base"] + q + k * 1000, f"t{q}_v{k}")
            if k < last and workflow_prompt(c, q, k + 1).exists():
                continue
            if k == last and (output / "retained_workflow.json").exists():
                continue
            cmd = [sys.executable, "-m", "harness_improvement.iterate_multi_agent_prompt",
                   "--query-file", workflow_prompt(c, q, k), "--current-version", k,
                   "--current-repo", workspace, "--runs-root", workspace.parent,
                   "--current-run-status-file", logs / "run_status.json", "--out-dir", output,
                   "--include-best-design", "--history-design-diffs", "--include-v0-design",
                   "--champion-repo-root-pattern", str(base / "workspaces/v{version}"),
                   "--champion-repo-v0-root", base / "workspaces/v0",
                   "--llm-backend", inference["backend"],
                   "--model", c["model_id"], "--base-url", c["endpoint"],
                   "--max-output-tokens", inference["max_output_tokens"],
                   "--seed", c["search"]["seed_base"] + q,
                   "--context-window-tokens", inference["context_window_tokens"],
                   "--context-safety-tokens", inference["context_safety_tokens"],
                   "--results-json-max-chars", "24000", "--results-json-total-max-chars", "120000"]
            if inference["backend"] == "openrouter":
                cmd += ["--api-key-env", "OPENROUTER_API_KEY", "--reasoning-effort", inference["reasoning_effort"],
                        "--omit-temperature", "--report-max-tokens", "12000", "--json-max-tokens", "6000",
                        "--code-max-tokens", "4000"]
            else:
                cmd += ["--require-local-vllm", "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20", "--local-thinking"]
            if k > 0:
                cmd += ["--previous-repo", base / f"workspaces/v{k-1}/{query_name(q)}",
                        "--judge-vs-baseline", "--baseline-repo-root", base / "reference"]
            if k == last:
                cmd += ["--evaluate-only"]
            run(cmd)


def collect(c, tasks):
    base = run_root(c)
    manifest_path = base / "pool.json"
    rows = {r["episode"]: r for r in read_json(manifest_path)} if manifest_path.exists() else {}
    for q in tasks:
        if q not in c["tasks"]["train"]:
            raise ValueError(f"Task {q} is excluded from SFT collection")
        selected = read_json(base / f"search/{query_name(q)}/retained_workflow.json")
        if not selected["beats_reference"]:
            print(f"Task {q}: no retained workflow beats the reference; excluded from collection.")
            continue
        teacher = base / f"teacher_prompts/query{q}.txt"
        teacher.parent.mkdir(parents=True, exist_ok=True)
        text = workflow_prompt(c, q, selected["version"]).read_text().rstrip() + "\n\n" + (ROOT / "tasks/teacher_directive.txt").read_text()
        if teacher.exists() and teacher.read_text() != text:
            raise ValueError(f"Teacher prompt changed: {teacher}")
        teacher.write_text(text)
        for i in range(c["collection"]["candidates"]):
            seed = c["collection"]["seed_base"] + 100 * q + i
            condition = f"t{q}_L{c['generation']}"
            name = f"{condition}_s{seed}"
            workspace, logs = base / f"pool/workspaces/{name}", base / f"pool/runlogs/{name}"
            episode(c, teacher, workspace, logs, seed, condition)
            rows[name] = {"episode": name, "task": q, "condition": condition, "seed": seed,
                          "workspace": str(workspace), "logs_dir": str(logs), "teacher_prompt": str(teacher)}
            write_json(manifest_path, sorted(rows.values(), key=lambda r: r["episode"]))


def tournament_pairs(names, opponents, seed):
    """Connected balanced ring schedule; unique unordered pairs, seeded order."""
    if opponents < 2 or opponents % 2:
        raise ValueError("Use an even opponents_per_episode >= 2")
    names = sorted(names)
    random.Random(seed).shuffle(names)
    pairs = {tuple(sorted((name, names[(i + d) % len(names)])))
             for i, name in enumerate(names) for d in range(1, min(opponents // 2, len(names) - 1) + 1)}
    pairs = sorted(pairs)
    rng = random.Random(seed + 1)
    return [(b, a) if rng.random() < .5 else (a, b) for a, b in pairs]


def rank(c, tasks):
    from .judging import self_judge
    base = run_root(c)
    pool = read_json(base / "pool.json")
    for q in tasks:
        retained = read_json(base / f"search/{query_name(q)}/retained_workflow.json")
        if not retained["beats_reference"]:
            print(f"Task {q}: no eligible retained workflow; no teacher ranking.")
            continue
        rows = {r["episode"]: r for r in pool if r["task"] == q and completed(Path(r["logs_dir"]))}
        if len(rows) < c["collection"]["validation_rank"]:
            raise ValueError(f"Task {q}: need at least {c['collection']['validation_rank']} successful candidates, found {len(rows)}")
        pairs = tournament_pairs(rows, c["ranking"]["opponents_per_episode"], c["ranking"]["seed"] + q)
        path = base / f"selection/task{q}_matches.jsonl"
        done = {(r["a"], r["b"]): r for r in records(path)}
        for i, (a, b) in enumerate(pairs):
            if (a, b) not in done:
                verdict = self_judge(c, q, Path(rows[a]["workspace"]), Path(rows[b]["workspace"]), c["ranking"]["seed"] + q * 1000 + i)
                done[(a, b)] = {"a": a, "b": b, **verdict}
                append_record(path, done[(a, b)])
        matches = {(a, b): {"self": a if done[(a, b)]["winner"] == "A" else b if done[(a, b)]["winner"] == "B" else "tie"} for a, b in pairs}
        scores = bradley_terry(sorted(rows), matches)
        ordered = sorted(rows, key=lambda name: (-scores[name], name))
        write_json(base / f"selection/task{q}_ranked.json", [{**rows[n], "score": scores[n], "rank": i + 1} for i, n in enumerate(ordered)])


def select(c):
    base = run_root(c)
    train, val = [], []
    for q in c["tasks"]["train"]:
        retained = read_json(base / f"search/{query_name(q)}/retained_workflow.json")
        if not retained["beats_reference"]:
            continue
        rows = read_json(base / f"selection/task{q}_ranked.json")
        if len(rows) < c["collection"]["validation_rank"]:
            raise ValueError(f"Missing validation rank for task {q}")
        train.extend(rows[:c["collection"]["train_ranks"]])
        val.append(rows[c["collection"]["validation_rank"] - 1])
    if not train or not val:
        raise ValueError("No eligible training/validation trajectories")
    if {r["episode"] for r in train} & {r["episode"] for r in val}:
        raise ValueError("Training and validation episodes overlap")
    write_json(base / "selection/train.json", train)
    write_json(base / "selection/val.json", val)
    print(f"Selected {len(train)} training and {len(val)} validation trajectories.")


def render(c):
    base = run_root(c)
    (base / "data").mkdir(exist_ok=True)
    for split in ("train", "val"):
        manifest = base / f"selection/{split}.json"
        rows = read_json(manifest)
        cmd = [sys.executable, ROOT / "training/prepare_sft_data.py", "--tokenizer", package_path(c["parent_bf16"]),
               "--episode-list", manifest, "--window", c["training"]["window"],
               "--context-overlap", c["training"]["context_overlap"], "--require-success", "--strict",
               "--out", base / f"data/{split}.jsonl", "--stats", base / f"data/{split}_stats.json"]
        for condition, row in {r["condition"]: r for r in rows}.items():
            cmd += ["--bare-prompt-map", f"{condition}={ROOT}/tasks/bare/query{row['task']}.txt",
                    "--teacher-prompt", f"{condition}={row['teacher_prompt']}"]
        run(cmd)
        # Prevent the renderer's logged skips from silently shrinking a split.
        found = set()
        with (base / f"data/{split}.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                found.add(row["episode"])
        if found != {r["episode"] for r in rows}:
            raise ValueError(f"{split}: selected episodes did not all yield windows; inspect rendering stats")


def training_command(c):
    t, base = c["training"], run_root(c)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={t['replicas']}",
           ROOT / "training/train_sft_lora.py", "--parallel", "pp_ddp", "--gpus-per-rank", t["gpus_per_rank"],
           "--model", package_path(c["parent_bf16"]), "--data", base / "data/train.jsonl",
           "--val-data", base / "data/val.jsonl", "--out", base / "training", "--sampler", "task_role",
           "--p-orch", str(1 / 3), "--val-at-start", "--autocast-bf16"]
    for key in ["steps", "lr", "warmup_steps", "min_lr_ratio", "lora_r", "lora_alpha", "lora_dropout",
                "save_every", "val_every", "grad_accum", "ce_chunk", "seed"]:
        cmd += ["--" + key.replace("_", "-"), t[key]]
    return cmd


def best_checkpoint(directory):
    candidates = []
    for row in records(directory / "train_log.jsonl"):
        if "val_loss" not in row:
            continue
        if not math.isfinite(row["val_loss"]) or row.get("val_tokens", 0) <= 0:
            raise ValueError("Invalid validation loss or zero validation tokens")
        step = row["step"]
        if isinstance(step, int) and step > 0:
            checkpoint = directory / f"step_{step:05d}"
            if (checkpoint / "lora_state_dict.safetensors").is_file():
                candidates.append((row["val_loss"], step, checkpoint))
    if not candidates:
        raise ValueError("No saved checkpoint has a finite validation loss")
    loss, step, path = min(candidates)
    return {"step": step, "val_loss": loss, "checkpoint": str(path), "criterion": "minimum validation loss"}


def export(c):
    base = run_root(c)
    selected = best_checkpoint(base / "training")
    write_json(base / "selected_checkpoint.json", selected)
    for key in ("next_bf16", "next_fp8"):
        p = package_path(c[key])
        if p.exists() and any(p.iterdir()):
            raise FileExistsError(f"Refusing to overwrite model output: {p}")
    run([sys.executable, ROOT / "training/merge_lora.py", "--base", package_path(c["parent_bf16"]),
         "--lora", Path(selected["checkpoint"]) / "lora_state_dict.safetensors", "--r", c["training"]["lora_r"],
         "--alpha", c["training"]["lora_alpha"], "--out", package_path(c["next_bf16"])])
    run([sys.executable, ROOT / "training/quantize_fp8_blockwise.py", "--src", package_path(c["next_bf16"]),
         "--ref", package_path(c["fp8_reference"]), "--out", package_path(c["next_fp8"])])


def rollout(c, tasks, count):
    base = run_root(c)
    rows = []
    for q in tasks:
        for i in range(count):
            workspace = base / f"evaluation/workspaces/query{q}_r{i}"
            episode(c, ROOT / f"tasks/bare/query{q}.txt", workspace,
                    base / f"evaluation/logs/query{q}_r{i}", 910000 + q * 100 + i, f"t{q}_bare")
            rows.append({"task": q, "replicate": i, "workspace": str(workspace)})
    write_json(base / "evaluation/rollouts.json", rows)


def report(c, pairs_file):
    from .judging import external_judgments
    path = run_root(c) / "evaluation/external.jsonl"
    done = {(r["pair_id"], r["provider"], r["seed"]) for r in records(path)}
    for pair in read_json(pairs_file):
        # Hash input paths to keep distinct workspace pairs distinct on resume.
        pair_id = hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()
        if all((pair_id, p, s) in done for p in ["openai", "anthropic"] for s in [42, 43, 44]):
            continue
        skip = {(provider, seed) for pid, provider, seed in done if pid == pair_id}
        for row in external_judgments(pair["task"], Path(pair["a"]).resolve(), Path(pair["b"]).resolve(), skip=skip):
            key = (pair_id, row["provider"], row["seed"])
            if key not in done:
                append_record(path, {"pair_id": pair_id, "pair": pair, **row})
                done.add(key)
    rows = records(path)
    wins = sum(r["winner"] == "A" for r in rows)
    losses = sum(r["winner"] == "B" for r in rows)
    write_json(path.with_name("summary.json"), {"wins_a": wins, "wins_b": losses, "ties": len(rows) - wins - losses,
               "win_rate_a": wins / (wins + losses) if wins + losses else None, "judgments": len(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "search", "collect", "rank", "select", "render", "train", "export", "rollout", "report"])
    parser.add_argument("--config", default=str(ROOT / "configs/paper.json"))
    parser.add_argument("--tasks", type=int, nargs="+", help="Subset for search, collect, rank, or rollout")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--count", type=int, default=10, help="Bare-task evaluation rollouts per task")
    parser.add_argument("--pairs", type=Path, help="External reporting JSON list of {task, a, b}; a and b are workspace paths")
    args = parser.parse_args()
    c = validate(read_json(args.config))
    if args.stage == "plan":
        print(json.dumps(c, indent=2))
        if inference_options(c)["backend"] == "openrouter":
            print("OpenRouter -> RHI -> teacher collection -> ranking -> selection")
            print("OpenRouter inference only: render, train, and export are not supported by this configuration.")
        else:
            print(f"L^({c['generation']}) -> RHI -> teacher collection -> SFT -> L^({c['generation'] + 1})")
            print("Training command (not executed): " + shlex.join(list(map(str, training_command(c)))))
        return
    if inference_options(c)["backend"] == "openrouter":
        if args.stage in ("render", "train", "export"):
            raise SystemExit("OpenRouter inference configuration does not support render, train, or export")
        if args.stage in ("rollout", "search", "collect", "rank") and not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise SystemExit("OPENROUTER_API_KEY is required for OpenRouter inference")
    freeze_config(c)
    tasks = args.tasks or (c["tasks"]["train"] if args.stage in ("collect", "rank") else
                          c["tasks"][args.split] if args.split != "all" else
                          c["tasks"]["train_candidates"] + c["tasks"]["test"])
    if not set(tasks) <= set(c["tasks"]["train_candidates"] + c["tasks"]["test"]):
        parser.error("Unknown task ID")
    if args.stage in ("search", "collect", "rank"):
        globals()[args.stage](c, tasks)
    elif args.stage in ("select", "render", "export"):
        globals()[args.stage](c)
    elif args.stage == "train":
        if (run_root(c) / "training/train_log.jsonl").exists():
            raise FileExistsError("Training log exists; use train_sft_lora.py --resume explicitly")
        run(training_command(c))
    elif args.stage == "rollout":
        if args.count < 1:
            parser.error("--count must be positive")
        rollout(c, tasks, args.count)
    elif args.stage == "report":
        if not args.pairs:
            parser.error("report requires --pairs")
        report(c, args.pairs)
