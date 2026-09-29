#!/usr/bin/env python3
"""Build the 102 verified ScienceAgentBench task environments.
Program-presence rewards are bookkeeping; use the official evaluator for scores."""
from pathlib import Path
import os
import sys
BENCH_ROOT = Path(os.environ.get("MASS_BENCH_ROOT", str(Path(__file__).resolve().parents[1] / "work"))).resolve()

import argparse, json, os, shutil, subprocess
from pathlib import Path
from datasets import load_dataset

BENCH = BENCH_ROOT / "sab" / "benchmark"
TASKS = BENCH_ROOT / "sab" / "tasks"
BASE_IMAGE = "sab.base.x86_64:latest"

INSTRUCTION = """You are an expert Python programming assistant that helps scientist users to write high-quality code to solve their tasks.
Given a user request, you are expected to write a complete program that accomplishes the requested task and save any outputs to `/workspace/pred_results/` in the correct format.

Here's the user request you need to work on:
{task_inst}

You can access the dataset at `{dataset_path}`. Here is the directory structure of the dataset:
```
{dataset_folder_tree}
```
Here are some helpful previews for the dataset file(s):
{dataset_preview}

Please save your program as `/workspace/pred_programs/{pred_program_name}`.
Then, please run the program to check and fix any errors.
Please do NOT run the program in the background.
If the program uses some packages that are incompatible, please figure out alternative implementations and do NOT restart the environment.
"""

TASK_TOML = """version = "1.0"

[metadata]
benchmark = "ScienceAgentBench (verified)"
instance_id = "{iid}"
domain = "{domain}"
gold_program_name = "{gold}"
output_fname = "{out}"

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 1800.0

[environment]
docker_image = "sab_task_{iid}:{image_tag}"
build_timeout_sec = 1200.0
cpus = 2
memory = "8G"
storage = "20G"
"""
IMAGE_TAG = os.environ.get("SAB_IMAGE_TAG", "20260924")   # images pre-built by bench_sab_prebuild.py from the same environment/Dockerfile

DOCKERFILE = """FROM {base}
RUN mkdir -p /workspace/pred_programs /workspace/pred_results /workspace/benchmark/datasets
COPY datasets/ /workspace/benchmark/datasets/
WORKDIR /workspace
"""

TEST_SH = """#!/bin/bash
# bookkeeping only: keep the predicted program for the official ScienceAgentBench evaluator; reward 1 iff it exists
mkdir -p /logs/verifier/pred_programs /logs/verifier/pred_results
cp -r /workspace/pred_programs/. /logs/verifier/pred_programs/ 2>/dev/null || true
ls /workspace/pred_results > /logs/verifier/pred_results_listing.txt 2>/dev/null || true
if [ -s "/workspace/pred_programs/{pred}" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", nargs="*", type=int, default=None); a = ap.parse_args()
    ds = load_dataset("osunlp/ScienceAgentBench", split="verified")
    rows = [r for r in ds if a.ids is None or int(r["instance_id"]) in a.ids]
    TASKS.mkdir(parents=True, exist_ok=True)
    index = {}
    for r in rows:
        iid = int(r["instance_id"]); folder = r["dataset_folder_tree"].split("\n")[0][4:].rstrip("/")
        src = BENCH / "datasets" / folder
        assert src.is_dir(), (iid, src)
        t = TASKS / f"sab_{iid}"
        env = t / "environment"; (env / "datasets").mkdir(parents=True, exist_ok=True); (t / "tests").mkdir(exist_ok=True)
        dst = env / "datasets" / folder
        if not dst.exists():  # hard links: same NFS filesystem, no extra space
            shutil.copytree(src, dst, copy_function=shutil.copy2)
        pred = "pred_" + r["gold_program_name"]
        (t / "instruction.md").write_text(INSTRUCTION.format(
            task_inst=r["task_inst"], dataset_path=f"/workspace/benchmark/datasets/{folder}",
            dataset_folder_tree=r["dataset_folder_tree"], dataset_preview=r["dataset_preview"], pred_program_name=pred))
        (t / "task.toml").write_text(TASK_TOML.format(iid=iid, domain=r["domain"], gold=r["gold_program_name"], out=r["output_fname"], image_tag=IMAGE_TAG))
        (env / "Dockerfile").write_text(DOCKERFILE.format(base=BASE_IMAGE))
        (t / "tests" / "test.sh").write_text(TEST_SH.format(pred=pred)); os.chmod(t / "tests" / "test.sh", 0o755)
        index[f"sab_{iid}"] = {"instance_id": iid, "domain": r["domain"], "gold_program_name": r["gold_program_name"],
                               "pred_program_name": pred, "output_fname": r["output_fname"], "dataset_folder": folder}
    (TASKS / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {len(rows)} tasks under {TASKS}")


if __name__ == "__main__":
    main()
