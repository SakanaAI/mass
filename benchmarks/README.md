# Public benchmarks: $\mathcal L^{(k)}$ + qwen-code

These adapters run ScienceAgentBench (102 verified tasks) and an earlier
12-task MLR-Bench subset. They use bare benchmark instructions and the same
qwen-code runtime for `L0`, `L1`, and `L2`. The subset is listed in
[configs/mlr_tasks.txt](configs/mlr_tasks.txt). No benchmark trajectory enters
MASS training.

The [README results](../README.md#benchmark-results) include an expanded
107-task, three-trial MLR-Bench evaluation. Its task configuration is not
included here; the MLR-Bench instructions below describe the packaged 12-task,
five-trial setup. Evaluation adapters for AstaBench (E2E-Bench-Hard), DSBench,
Terminal-Bench 2.0, and SWE-bench Verified are also not included.

Run commands from the repository root unless a subshell changes directory.
Use Python 3.12 and Docker with Compose. The adapters require the pinned Harbor
checkout; they do not depend on a local research-repository installation.

## Official benchmark resources

| Benchmark | Paper | Code or data |
|---|---|---|
| ScienceAgentBench | [Paper](https://arxiv.org/abs/2410.05080) | [Official repository](https://github.com/OSU-NLP-Group/ScienceAgentBench) |
| MLR-Bench | [Paper](https://arxiv.org/abs/2505.19955) | [Official repository](https://github.com/chchenhui/mlrbench) |
| AstaBench (E2E-Bench-Hard) | [Paper](https://arxiv.org/abs/2510.21652) | [Official repository](https://github.com/allenai/asta-bench#discovery-tasks-discovery) |
| DSBench | [Paper](https://arxiv.org/abs/2409.07703) | [Official repository](https://github.com/LiqiangJing/DSBench) |
| Terminal-Bench 2.0 | [Paper](https://arxiv.org/abs/2601.11868) | [Version 2.0 repository](https://github.com/harbor-framework/terminal-bench-2) |
| SWE-bench Verified | [Original SWE-bench paper](https://arxiv.org/abs/2310.06770) | [Verified dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) |

Related resource: [Terminal-Bench-Science](https://github.com/harbor-framework/terminal-bench-science).
No MASS results for that benchmark are reported in the README.

## AstaBench (E2E-Bench-Hard) results

Each model is evaluated on 40 tasks over three trials (120 task runs per model).
Rubric scores show the mean and standard deviation across the three trials.
Report counts combine all three trials; percentages are rounded to the nearest
whole percent. The last column reports the mean score among delivered reports.

| Player | Trial 1 | Trial 2 | Trial 3 | Mean rubric score ± std | Reports delivered (of 120) | Score among delivered reports |
|---|---:|---:|---:|---:|---:|---:|
| $\mathcal{L}^{(0)}$ + qwen-code | 0.051 | 0.052 | 0.055 | 0.053 ± 0.002 | 60 (50%) | 0.105 |
| $\mathcal{L}^{(1)}$ + qwen-code | 0.053 | 0.052 | 0.059 | 0.055 ± 0.004 | 69 (58%) | 0.095 |
| $\mathcal{L}^{(2)}$ + qwen-code | 0.062 | 0.053 | 0.085 | 0.067 ± 0.017 | 91 (76%) | 0.088 |

## Setup

```bash
python3 benchmarks/fetch_sources.py
python3.12 -m venv .venv-bench
(cd benchmarks && ../.venv-bench/bin/pip install -r requirements.txt)
source .venv-bench/bin/activate
```

The public repository URLs and exact commits are in
[configs/sources.json](configs/sources.json).
The fetcher applies the included ScienceAgentBench patch: it prevents an
optional CodeBERTScore failure from replacing executable-program scores with
zero. It does not change the task execution or success tests. Upstream licenses
remain in each downloaded checkout.

Build a portable qwen-code 0.20.0 bundle including Node.js, then extract it for
the read-only container mount. These commands download software:

```bash
docker build -t mass-qwen-bundle -f benchmarks/images/qwen.Dockerfile .
mkdir -p benchmarks/runtime/qwen-code
docker create --name mass-qwen-export mass-qwen-bundle
docker cp mass-qwen-export:/opt/qwen-code/. benchmarks/runtime/qwen-code/
docker rm mass-qwen-export
benchmarks/runtime/qwen-code/bin/qwen --version
```

Serve one generation with vLLM as described in the [run guide](../docs/running.md).
The benchmark
runtime uses a Docker bridge proxy; use `--proxy-host` if the host bridge address
differs from `172.17.0.1`. Allocate separate proxy port blocks for concurrent jobs.
Do not expose this proxy on an untrusted network.

## ScienceAgentBench

Obtain the **verified** benchmark archive through the pinned upstream
`benchmarks/vendor/ScienceAgentBench/README.md`. Extract its `benchmark/`
directory to `benchmarks/work/sab/benchmark/`. The upstream distribution requires
this separate download; the unzipped benchmark is not redistributed here.
The resulting folder contains `datasets/`, `eval_programs/`, `gold_programs/`,
and `scoring_rubrics/`.

```bash
python benchmarks/prepare_sab_base.py
python benchmarks/scripts/bench_sab_make_tasks.py
python benchmarks/scripts/bench_sab_prebuild.py
python benchmarks/run.py sab L0 1 --model-id Qwen/Qwen3.6-27B-FP8
python benchmarks/scripts/bench_sab_collect_preds.py \
  benchmarks/work/sab/jobs/sab_L0_t1 benchmarks/work/sab/predictions/L0_t1 \
  --eval --run_id L0_t1 --max_workers 4
```

The agent limit is 1,800 seconds, with two CPUs and 8 GB of RAM per task.
The official evaluator runs the recovered program with its 900-second limit.
Figure-based tests also require the upstream evaluator's OpenAI credentials.
The Harbor reward checks whether a program file exists; **it is not the
ScienceAgentBench success score**. Use the official evaluator JSONL.

Repeat trials 1–3 for each generation, changing `--model-id` to the served
generation. Aggregate each generation's three full logs:

```bash
python benchmarks/summarize.py \
  benchmarks/work/sab/predictions/L0_t1/eval_L0_t1.jsonl \
  benchmarks/work/sab/predictions/L0_t2/eval_L0_t2.jsonl \
  benchmarks/work/sab/predictions/L0_t3/eval_L0_t3.jsonl
```

The summary is the mean and sample standard deviation of the trial means.
The script requires all 102 evaluator rows. It does not fill missing trials or
substitute the manuscript's reported values. The official scorer's image and
library dependencies can be sensitive to the Docker host; retain evaluator
errors with the run record.

## MLR-Bench

```bash
docker build -t mlr_base:20260925 -f benchmarks/images/mlr.Dockerfile .
python benchmarks/scripts/bench_mlr_make_tasks.py $(cat benchmarks/configs/mlr_tasks.txt)
python benchmarks/run.py mlr L0 1 --model-id Qwen/Qwen3.6-27B-FP8
```

Each task has a two-hour agent limit, three CPUs, and 16 GB of RAM. No task GPU
is provided. Repeat five trials per generation. The seed map is shared across
generations for each task/trial. Save the built container digest for a new run;
the scientific-package recipe is not a frozen historical image.

Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, then score completed jobs:

```bash
python benchmarks/scripts/bench_mlr_judge.py benchmarks/work/mlr/jobs/mlr_L0_t* \
  --out benchmarks/work/mlr/reviews_L0.json
python benchmarks/summarize_mlr.py benchmarks/work/mlr/reviews_L0.json
```

The adapter imports the official MLR-Bench `OVERALL_RUBRIC` from
`mlrbench/evals/overall_review.py`. Each completed task is reviewed by GPT-5.5
and Claude Opus 4.8. The task score is their mean Overall rating (1–10);
a missing or shorter-than-200-byte paper receives zero. The trial score averages
all 12 tasks, including these zeros. Report the mean and sample standard
deviation across five trials. A missing judge response is an error, not a zero.

`python benchmarks/run.py ... --dry-run` prints the Harbor command without
creating tasks, calling a model, or launching containers. The offline tests
do not exercise Docker, vLLM, or the official scorers. See
[experiment support](../docs/experiment_coverage.md) for the analyses included
with these adapters.
