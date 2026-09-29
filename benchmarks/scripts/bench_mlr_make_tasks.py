#!/usr/bin/env python3
"""Build the configured MLR-Bench task environments with two-hour agent limits."""
from pathlib import Path
import os
import sys
BENCH_ROOT = Path(os.environ.get("MASS_BENCH_ROOT", str(Path(__file__).resolve().parents[1] / "work"))).resolve()

import argparse, os, shutil
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
TASKS = BENCH_ROOT / "mlr" / "tasks"
SRC = PKG / "vendor" / "mlrbench" / "tasks"
TAG = "20260925"

INSTRUCTION = """You are an expert machine learning researcher. Below is a research task description (a workshop call for papers). Conduct a small but complete research project on this task, end to end, working in `/workspace`. No GPU is available: use CPU only and keep experiments small (small public datasets or subsets, small models, few epochs); everything must finish within the 2-hour limit.

Steps:
1. Propose one concrete, novel research idea that addresses the task; write it to `/workspace/idea.md`.
2. Write a short research proposal to `/workspace/proposal.md`: hypothesis, proposed method, experimental plan including baselines, datasets and metrics.
3. Implement and run the experiments. Save all code in `/workspace/code/` with a README.md explaining how to run it. The scripts must run automatically end to end, run all baselines, save results in a structured format (CSV or JSON), and generate figures (with legends, titles and axis labels) into `/workspace/results/`. Save the execution log to `/workspace/results/log.txt`. After all experiments have completed, write `/workspace/results/results.md`: experimental setup (hyperparameters, dataset splits), tables comparing the proposed method with the baselines, the figures with explanations, a discussion of the findings, limitations and future work.
   IMPORTANT: do not use synthetic results or fake data — the results must come from experiments you actually ran. You may download open datasets from Hugging Face or other open sources; do not use closed-source datasets or models. Remove checkpoints or datasets larger than 1 MB when done.
4. Write the paper to `/workspace/results/paper.md` with these sections: Title and Abstract; Introduction; Related Work; Methodology; Experiment Setup; Experiment Results (tables and figures from your real results); Analysis; Conclusion. Report honestly: if an experiment failed or a result is negative, say so.

## Task description

{task}
"""

TASK_TOML = """version = "1.0"

[metadata]
benchmark = "MLR-Bench (end-to-end, single agent)"
task = "{name}"

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 7200.0

[environment]
docker_image = "mlr_base:{tag}"
build_timeout_sec = 600.0
cpus = 3
memory = "16G"
storage = "30G"
"""

TEST_SH = """#!/bin/bash
# bookkeeping only: keep the paper, results and (small) code files for the official MLR-Judge rubric; reward 1 iff results/paper.md exists
mkdir -p /logs/verifier/results /logs/verifier/code
cp -r /workspace/results/. /logs/verifier/results/ 2>/dev/null || true
for f in /workspace/idea.md /workspace/proposal.md; do [ -f "$f" ] && cp "$f" /logs/verifier/; done
if [ -d /workspace/code ]; then (cd /workspace/code && find . -type f -size -1024k \\( -name '*.py' -o -name '*.md' -o -name '*.txt' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.sh' -o -name '*.csv' \\) -exec cp --parents {} /logs/verifier/code/ \\;) 2>/dev/null || true; fi
find /logs/verifier/results -type f -size +5M -delete 2>/dev/null || true
if [ -s /workspace/results/paper.md ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+"); ap.add_argument("--pilot", nargs="*", default=None); a = ap.parse_args()
    TASKS.mkdir(parents=True, exist_ok=True)
    for name in a.names:
        src = SRC / f"{name}.md"; assert src.exists(), src
        t = TASKS / f"mlr_{name}"; (t / "environment").mkdir(parents=True, exist_ok=True); (t / "tests").mkdir(exist_ok=True)
        (t / "environment" / "Dockerfile").write_text(f"FROM mlr_base:{TAG}\nWORKDIR /workspace\n")
        (t / "instruction.md").write_text(INSTRUCTION.format(task=src.read_text().strip()))
        (t / "task.toml").write_text(TASK_TOML.format(name=name, tag=TAG))
        (t / "tests" / "test.sh").write_text(TEST_SH); os.chmod(t / "tests" / "test.sh", 0o755)
        shutil.copy(src, t / "task.md")
    (TASKS / "index.json").write_text("{" + ", ".join(f'"mlr_{n}": "{n}"' for n in a.names) + "}\n")
    if a.pilot:
        (TASKS.parent / "pilot_tasks.txt").write_text("\n".join(f"mlr_{n}" for n in a.pilot) + "\n")
    print(f"wrote {len(a.names)} tasks under {TASKS}; pilot: {a.pilot}")


if __name__ == "__main__":
    main()
