# Run MASS with OpenRouter

[configs/openrouter.json](../configs/openrouter.json) selects
`openai/gpt-oss-120b:nitro` through `https://openrouter.ai/api/v1` for the task
executor, workflow optimizer, and in-loop judges. It preserves the existing
paper task splits, uses one workflow update, and saves to `runs/openrouter`.
Use `--tasks` to select an existing task. Provider selection does not depend
on the task ID.

This integration supports `rollout`, `search`, `collect`, `rank`, and `select`.
It rejects `render`, `train`, and `export`: inference access does not provide
the matching weights, tokenizer, or training trace contract. The example's
training fields remain for compatibility with the existing config schema.
External reporting judges retain their separate provider settings.

## Setup

Install Python 3.12, Node.js 22, uv, Bash, jq, curl, and git. No local GPU or
model server is needed. From the repository root:

```bash
uv venv --python 3.12 .venv-core
uv pip install --python .venv-core/bin/python -r requirements-core.txt
npm install --prefix .runtime @qwen-code/qwen-code@0.20.0
source .venv-core/bin/activate
export PATH="$PWD/.runtime/node_modules/.bin:$PATH"
export EPISODE_CUDA_VISIBLE_DEVICES=""
```

Set `OPENROUTER_API_KEY` in your shell environment, or add it to the ignored
`.env` file using [.env.example](../.env.example) as a template. To export the
values from a trusted `.env` file:

```bash
set -a
source .env
set +a
```

Do not put credentials in JSON configuration or command arguments. Generated
Qwen settings reference the environment variable by name. The loopback logging
proxy forwards authentication to OpenRouter and does not follow redirects or
record headers. Keep the local proxy port (default 9001) available.

## Run

Preview needs no credentials and makes no model requests:

```bash
python -m mass plan --config configs/openrouter.json
```

These commands make paid model requests. Start with one bare task, then run
workflow search when ready. These examples use existing task 51:

```bash
python -m mass rollout --config configs/openrouter.json --tasks 51 --count 1
python -m mass search --config configs/openrouter.json --tasks 51
```

The rollout writes to `runs/openrouter/evaluation/`. Search separately
runs a bare reference, the initial workflow, and one updated workflow, plus
optimizer and judge calls. Collection, if requested later, runs 18 candidates
when the selected workflow beats the reference. Ranking makes additional paid
judgments; selection itself is offline. A single task episode is a research
run, not a bounded compatibility smoke test.

Qwen Code runs generated commands on the host in YOLO mode. Use an isolated
machine or container when the agent must not access other host data. Model
prompts and selected workspace evidence are sent to the remote service.

## Routing, budgets, and records

[Nitro](https://openrouter.ai/docs/guides/routing/model-variants/nitro) prefers
providers by throughput and admits priority tiers. It does not select a new
checkpoint or guarantee a fixed provider, speed, or price. Discovery checks
`openai/gpt-oss-120b`; inference retains the complete `:nitro` model ID. MASS
adds no automatic retries or fallback model; OpenRouter's own routing still
applies.

The example config uses a 131,072-token context, 8,192 output tokens per
request, medium reasoning effort, and 8,192 safety tokens for optimizer and
judge context checks. [Reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
share the output budget with the visible answer. These are per-request limits,
not a total episode token or dollar cap. Confirm provider-specific limits
before a long run.

Remote evidence bundling uses report=12,000, JSON=6,000, and code=4,000 token
limits through the existing bundler. Optimizer and judge prompts then undergo
the existing complete-prompt context check before a request. The count is a
heuristic, not the GPT-OSS tokenizer; overflow raises an explicit error rather
than dropping task rules or the verdict schema. Qwen Code handles episode
context using its configured model context window.

Episode logs include `run_status.json`, the agent trace, and compressed API
request/response records. Remote SSE logs retain raw chunks and reconstruct
tool arguments, reasoning details, errors, and final usage. Search and ranking
metrics retain requested/served model, provider, response ID, and raw usage
when supplied. [Provider-reported usage cost](https://openrouter.ai/docs/cookbook/administration/usage-accounting)
is recorded as `estimated_cost_usd`; missing cost stays `null`, not zero. Local
inference retains its existing zero API charge convention. Consult OpenRouter's
usage records for billing; MASS does not estimate Nitro prices from a table.

Authentication, credit, rate-limit, context, and server errors stop work or mark
episodes unsuccessful even when the Qwen process exits zero. Inspect the saved
status, stderr, and API records before retrying. Failed/partial directories are
preserved. Change `run_dir` in a copied config for a retry or when changing
model/provider/budget settings; saved configurations cannot silently change.

## Validation status

Offline tests cover backend requests, authenticated proxy forwarding, stream
assembly, failure classification, provider propagation, and context guards.
Live OpenRouter compatibility remains unverified: no API key was available
during implementation. Before a long run, separately authorize a bounded smoke
test of optimizer text, judge JSON, and a tiny episode that writes and reads a
sentinel file. Inspect the actual Qwen tool-call follow-up and reasoning fields,
successful status, usage, and absence of credentials in saved artifacts. The
pinned runtime remains Qwen Code 0.20.0.
