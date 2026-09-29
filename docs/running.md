# Run MASS

This guide covers installation, a single-task example, both MASS cycles, and
synthetic-task evaluation. See [implementation notes](reproduction.md) for
current behavior and [experiment support](experiment_coverage.md) for the
available paper experiments.

Full runs require GPUs, substantial
disk space for checkpoints and trajectories, and network access for task data.
The first-cycle paper training run used six H100 GPUs and about 19 hours for
SFT alone. RHI and teacher collection are additional work.

Use an isolated compute environment for research-task episodes: qwen-code runs
generated shell commands with `--approval-mode yolo`. The supplied synthetic-task
runner uses the current host environment, matching the experiment. It does not
create a security sandbox. Public benchmarks run task code inside Docker.

## Install

Use Linux with Python 3.12, Bash, `jq`, `curl`, `git`, Node.js 22, and `uv`.
Keep serving and training in separate Python environments because their
recorded Transformers versions differ. Run commands from the repository root.

```bash
python3.12 -m venv .venv-train
.venv-train/bin/pip install -r requirements-train.txt
python3.12 -m venv .venv-serve
.venv-serve/bin/pip install -r requirements-serve.txt
npm install --prefix .runtime @qwen-code/qwen-code@0.20.0
export PATH="$PWD/.runtime/node_modules/.bin:$PATH"
.venv-train/bin/python tools/fetch_models.py
```

The model download is large. Its public revisions are listed in
`configs/models.json`. Training uses the BF16 base, while inference uses its
FP8 counterpart. Install compatible NVIDIA drivers/CUDA for the pinned PyTorch
and vLLM builds. The top-level pins were recovered from the research environments;
this is not a transitive dependency lock file.

## Try one task

For a first experiment, create a local configuration with one workflow update
and its own output directory:

```bash
python3 - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/paper.json").read_text())
config["run_dir"] = "runs/demo"
config["search"]["iterations"] = 1
Path("configs/local-demo.json").write_text(json.dumps(config, indent=2) + "\n")
PY
```

Start the base-model vLLM server shown in the next section, then run:

```bash
source .venv-train/bin/activate
export EPISODE_CUDA_VISIBLE_DEVICES=""
python -m mass plan --config configs/local-demo.json
python -m mass search --config configs/local-demo.json --tasks 51
```

This executes a bare-task reference, the initial workflow, and one updated
workflow. It still runs the full task prompt, so it can take substantial time.
Inspect the comparisons and retained workflow under
`runs/demo/search/query51_ourTeam/`. This example covers workflow search; the
full collection and training sequence follows below.

## Cycle 1: $\mathcal L^{(0)} \to \mathcal L^{(1)}$

Start a serving process in a separate terminal, on GPUs you allocate:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-serve/bin/vllm serve models/L0-fp8 \
  --served-model-name Qwen/Qwen3.6-27B-FP8 --host 127.0.0.1 --port 8001 \
  --max-model-len 262144 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --max-num-seqs 4 --gpu-memory-utilization 0.90
```

Activate the training environment for the following stages. It also has the
client and trace-rendering dependencies. `EPISODE_CUDA_VISIBLE_DEVICES` controls
GPUs visible to generated research programs; set it to your allocated task GPU
or to an empty string for CPU-only task execution. This choice can affect results.

```bash
source .venv-train/bin/activate
export EPISODE_CUDA_VISIBLE_DEVICES=""
python -m mass plan --config configs/paper.json
python -m mass search --config configs/paper.json
python -m mass collect --config configs/paper.json
python -m mass rank --config configs/paper.json
python -m mass select --config configs/paper.json
python -m mass render --config configs/paper.json
```

The driver runs episodes sequentially through the configured endpoint and proxy.
Use `--tasks 51` to inspect a single task during search, collection, or ranking.
Complete all configured training tasks before `select`. Successful episodes and
completed comparisons are reused. Partial episode directories are preserved and
not overwritten. After an infrastructure failure, inspect the evidence and use
a new `run_dir`, or explicitly archive the affected partial episode before
retrying it. Configuration changes require a new `run_dir`.

Stop your inference process and allocate six GPUs for SFT. Training is a separate
explicit command; the driver never starts/stops a vLLM process:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m mass train --config configs/paper.json
python -m mass export --config configs/paper.json
```

`export` selects the saved checkpoint with the lowest measured validation loss,
merges its adapter into `models/L0-bf16`, and writes `models/L1-bf16` and
`models/L1-fp8`. It refuses to overwrite populated output model directories.
For interrupted training, inspect `python training/train_sft_lora.py --help`
and use its explicit `--resume` option with the printed training command.

## Cycle 2: $\mathcal L^{(1)} \to \mathcal L^{(2)}$

Serve `models/L1-fp8` with `--served-model-name L1`. Repeat the same stage
commands using `--config configs/cycle2.json`. This config reads the merged
BF16 `L^(1)` parent and writes `models/L2-bf16` and `models/L2-fp8`.
The fresh LoRA does not reuse the first cycle's optimizer state.

## Synthetic-task evaluation

With the corresponding generation served, run bare-task rollouts:

```bash
python -m mass rollout --config configs/paper.json --split test --count 10
```

For `L^(1)` use `configs/cycle2.json`. For `L^(2)`, copy that config to a
new file and set `generation` to 2, `run_dir` to `runs/L2`, and `model_id`
to the served name `L2`. The rollout command only uses inference fields.

Pair the two generations' rollout manifests:

```bash
python tools/pair_rollouts.py runs/L1/evaluation/rollouts.json \
  runs/L0/evaluation/rollouts.json --out pairs.json
```

This creates a JSON list pairing workspaces on the same task and replicate:

```json
[{"task": 60, "a": "runs/L1/evaluation/workspaces/query60_r0",
  "b": "runs/L0/evaluation/workspaces/query60_r0"}]
```

Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in the environment; never put keys
in source files. You can copy `.env.example` to the ignored `.env` file, fill
in the values locally, and load them into your shell:

```bash
set -a
source .env
set +a
```

The scripts do not load `.env` automatically. These keys are needed for the
external reporting stage, not for local workflow search or post-training.
Then run:

```bash
python -m mass report --config configs/cycle2.json --pairs pairs.json
```

There are three independent calls to each external evaluator per pair. Seeds
42–44 control randomized presentation; they do not imply deterministic provider
sampling. The report contains W/(W+L), excludes ties, and uses `null` when there
are no decisive judgments. External reporting is separate from the MASS loop.

Use a separate run directory for each evaluation pool and report comparison
set. A new rollout call replaces `evaluation/rollouts.json`, and reporting
includes earlier judgments in that run directory. The default `--split all`
includes task 221. For the paper's 11-task task-solving pool, use
`--tasks 51 53 56 202 205 302 307 324 60 207 305` instead. These behaviors are
described in the [implementation notes](reproduction.md#synthetic-evaluation).

For public benchmarks, continue with the [benchmark guide](../benchmarks/README.md).
