# OpenRouter Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Run MASS task episodes, workflow search, and trajectory ranking through OpenRouter using `openai/gpt-oss-120b:nitro`.

**Architecture:** Extend the existing OpenAI-compatible backend and logging proxy. Keep local vLLM/LM Studio as the default, and select OpenRouter explicitly in configuration. Use one model/provider configuration across executor, optimizer, and in-loop judge; keep external reporting separate.

**Tech Stack:** Existing Python 3.12, OpenAI SDK, httpx, aiohttp, Bash, Qwen Code 0.20.0, unittest. No new dependencies.

**Spec:** The scope and acceptance contract below records the user's requested OpenRouter extension. Implementation and offline verification are complete. Live OpenRouter validation remains outstanding because no API key was supplied.

## Scope and acceptance contract

Support `rollout`, `search`, `collect`, and `rank` for the existing research-task pipeline. Existing task 51 is the initial example, but provider selection must not depend on task ID. Preserve `select` and the existing paper configurations.

OpenRouter is an inference service in this integration. It does not train or export weights. Do not imply that GPT-OSS trajectories turn the existing Qwen SFT recipe into recursive GPT-OSS training. Explicitly reject `render`, `train`, and `export` under the OpenRouter example configuration until a separate training integration supplies matching model/tokenizer/trace contracts.

The implementation is complete when mocked transport tests verify every path, existing local tests still pass, and a separately authorized live smoke test verifies a tool call plus its follow-up response, optimizer text, and judge JSON. A successful `/models` request alone is insufficient. Full research-task runs are not part of the smoke test.

## Global constraints

- Keep `configs/paper.json` and `configs/cycle2.json` behavior unchanged.
- Use `https://openrouter.ai/api/v1`, `openai/gpt-oss-120b:nitro`, and `OPENROUTER_API_KEY` for the initial remote configuration.
- Read credentials only from the environment; never serialize their values into settings, commands, configs, errors, or request logs.
- Keep existing success checks, partial-run preservation, and refusal to reuse changed run configurations.
- Keep API retries disabled; surface auth, rate-limit, credit, context, and server failures without implicit provider/model substitution by MASS.
- Preserve the narrow LM Studio response-format compatibility behavior for local calls only.
- Keep Qwen Code 0.20.0 unless a demonstrated compatibility failure requires a separately reviewed upgrade.
- Use existing requirements files and unittest; do not introduce a provider framework or a second runner.

## Review focus

1. A routed `:nitro` ID may not appear in `/models`; discovery must validate its base ID without removing the suffix from inference requests (Task 1).
2. Authenticated requests pass through a local proxy; secrets must not enter generated settings/logs or be forwarded across redirects (Task 2).
3. An episode can finish at the process level while its stream contains an API failure; it must not become a reusable successful trajectory (Task 2).
4. Two large workspaces can exceed the 131,072-token context during judging even when ordinary chat fits (Task 3).
5. Missing usage/cost and split reasoning/tool-call stream chunks must not produce fabricated zero charges or lose tool arguments (Tasks 1–2).

## File map

| File | Responsibility |
| --- | --- |
| `harness_improvement/llm_backend.py` | Explicit OpenRouter chat backend, model discovery, reasoning settings, JSON validation, usage/cost provenance. |
| `harness_improvement/iterate_multi_agent_prompt.py` | CLI backend propagation, key lookup, context limits, judge inheritance, provenance checks. |
| `mass/pipeline.py` | Configuration validation, stage restrictions, episode environment, optimizer command construction. |
| `mass/judging.py` | Same backend for ranking; bounded evidence and final input/output budget check. |
| `runtime/run_episode.sh` | Configured API URL, environment key name, model settings, runtime budgets, error classification. |
| `runtime/api_proxy.py` | Correct upstream URL prefix, safe forwarding, faithful stream/error/usage evidence. |
| `tests/test_llm_backend.py`, `tests/test_pipeline.py`, `tests/test_rhi_core.py` | Backend, command propagation, and RHI regression coverage. |
| `tests/test_runtime.py` (new) | Controlled HTTP server and fake Qwen executable exercising the real shell/proxy. |
| `configs/openrouter.json` (new), `.env.example`, `docs/openrouter.md` (new) | Working example and setup/limitations. |

## Task 1: Explicit OpenRouter backend

**Files:** `harness_improvement/llm_backend.py`, `tests/test_llm_backend.py`.

**Interfaces:** Add `BACKEND_OPENROUTER = "openrouter"` to `BACKEND_CHOICES`. Preserve the `JsonLLMBackend` constructor and `call_json`/`call_text` signatures. Extend `_chat_usage_metrics(response, *, model: str, backend: str = BACKEND_LOCAL_VLLM) -> dict[str, Any]`. Rename `_local_chat` to `_chat` and update both internal callers because it now serves two transports.

- [x] Add HTTP-fixture tests through the real OpenAI client. Verify `/api/v1/chat/completions`, unchanged `openai/gpt-oss-120b:nitro`, and bearer auth; model listing containing only `openai/gpt-oss-120b` must pass. An unknown base model must fail before inference. Strip only the terminal `:nitro` suffix for discovery, only on the OpenRouter backend.
- [x] Add request/response tests: `reasoning={"effort":"medium"}`, no `chat_template_kwargs` or `top_k`, JSON-object mode for `call_json`, no response format for `call_text`. Empty final content, malformed JSON, a JSON list, and refusal/error responses must fail. Never interpret reasoning text as the final JSON answer.
- [x] Test that `usage.cost=0.0123` survives as `estimated_cost_usd=0.0123` with a note that the value is provider-reported; absent cost stays `None`. Retain raw usage, response ID, requested model, returned model, and returned provider when available. Local calls retain the existing zero-API-cost convention.
- [x] Run `rtk proxy .venv-core/bin/python -m unittest tests.test_llm_backend -v`; confirm the new provider cases fail before implementation.
- [x] Implement the explicit branch using the installed SDK. Require a nonempty OpenRouter key and the selected base URL. Keep current local calls and Responses calls unchanged; the LM Studio compatibility fallback must never retry an OpenRouter 400.
- [x] Run the backend tests again; all cases must pass, including local JSON/text and unrelated-error tests. Commit only this tested change.

## Task 2: Remote episodes through the existing logging proxy

**Files:** `runtime/run_episode.sh`, `runtime/api_proxy.py`, `tests/test_runtime.py`.

**Interfaces:** Preserve the shell's five positional arguments. Add environment inputs `MASS_BACKEND` (default `local-vllm`), `MODEL_BASE_URL` (default derived from the existing port argument), `MODEL_CONTEXT_WINDOW` (default `262144`), `MODEL_MAX_OUTPUT_TOKENS` (default `32768`), and `MODEL_REASONING_EFFORT` (used only for OpenRouter). Select the fixed credential name `OPENROUTER_API_KEY` for OpenRouter, `VLLM_API_KEY` locally. Keep `api_proxy.py --upstream` as an origin plus optional prefix: remote value `https://openrouter.ai/api`, incoming path `/v1/chat/completions`.

- [x] Write shell tests using a controlled server and fake `qwen` executable. Assert generated JSON uses the remote model/base URL, configured budgets, and `envKey: "OPENROUTER_API_KEY"`; the secret must be absent from settings, stdout, stderr, and persisted logs. No settings `env` block may copy a real key. Construct JSON using Python or jq structured arguments rather than interpolating arbitrary strings into a heredoc.
- [x] Test model discovery with the base ID only, missing key before workspace creation, and backward-compatible local invocation. Use the interpreter supplied by `PROXY_PY` consistently for runtime Python checks.
- [x] Write a real proxy HTTP test proving `/v1/chat/completions` becomes `/api/v1/chat/completions` exactly once and Authorization reaches only the intended upstream. Disable redirects for authenticated forwarding and the model-discovery check. Assert no request/response header collection writes credentials.
- [x] Add SSE fixtures with split tool arguments, `reasoning`/`reasoning_details`, usage-only final chunks, and an error after HTTP 200. Preserve raw chunks and supported fields in reconstructed evidence; never fabricate absent usage. Test 401, 402, 429, 5xx, and connection failures produce unsuccessful episode status even if the fake Qwen process exits zero with an API-error message.
- [x] Run `rtk proxy .venv-core/bin/python -m unittest tests.test_runtime -v`; observe the missing behavior fail. Implement only the tested URL/settings/error/stream changes. Do not add a separate proxy or SDK.
- [x] Repeat runtime tests, then run `rtk proxy bash -n runtime/run_episode.sh`. Both must pass. Commit this change.

## Task 3: Connect configuration, optimizer, and ranking

**Files:** `mass/pipeline.py`, `mass/judging.py`, `harness_improvement/iterate_multi_agent_prompt.py`, pipeline and RHI tests.

**Interfaces:** Add optional `inference` config fields `backend`, `context_window_tokens`, `episode_max_output_tokens`, `max_output_tokens`, `context_safety_tokens`, `reasoning_effort`. Missing fields preserve local defaults: `local-vllm`, `262144`, `32768`, `49152`, `8192`; no remote reasoning override. Keep existing endpoint/model fields. Reuse `JsonLLMBackend` from Task 1 and the runner environment from Task 2.

- [x] Test `validate(c)` accepts OpenRouter only with `inference.backend="openrouter"` and endpoint exactly `https://openrouter.ai/api/v1` (optional trailing slash normalized consistently). Continue rejecting credentials, query strings, fragments, non-HTTPS remote URLs, wrong hosts, and proxy-port collisions for local endpoints. Reject bool/noninteger/nonpositive token limits and output + safety >= context. Configuration preview must not need an API key.
- [x] Test command construction: remote search passes `--llm-backend openrouter`, endpoint, `--api-key-env OPENROUTER_API_KEY`, reasoning effort, and the configured context/output budgets. Omit `--require-local-vllm`, `--local-thinking`, and Qwen-only sampling flags remotely. Local commands retain their existing values. Ensure both default and explicitly configured RHI judges inherit the correct endpoint/key/backend; update history provenance checks that currently run only for `local-vllm`.
- [x] Test episode environment propagation and a mocked search → collection → ranking → selection flow. All model calls must use the selected remote model, never silently revert to the local model or an external reporting judge. Use a fresh run directory when switching providers.
- [x] Test that `render`, `train`, and `export` fail clearly for the OpenRouter configuration before making artifacts or invoking commands. Explain that inference support alone does not establish a compatible training pipeline.
- [x] Add context tests covering a pair of oversized workspaces. For OpenRouter, cap each evidence bundle at report=12000, JSON=6000, code=4000 tokens using the existing bundle arguments, and pass corresponding caps to RHI. Apply the existing final context guard before each remote optimizer/judge request, including ranking. It is a heuristic estimate, not the GPT-OSS tokenizer; retain 8192 safety tokens and preserve explicit overflow errors. Do not silently drop the task, verdict schema, or workflow rules.
- [x] Run the new pipeline/RHI cases to confirm failure, implement the branches and option propagation, then run `rtk proxy .venv-core/bin/python -m unittest discover -s tests -v`. All tests must pass without paid calls. Commit this integration.

## Task 4: Example, documentation, and acceptance check

**Files:** `configs/openrouter.json`, `.env.example`, `docs/openrouter.md`.

**Interfaces:** Example retains the existing paper task splits and uses one search update with `runs/openrouter`, and sets:

```json
{
  "model_id": "openai/gpt-oss-120b:nitro",
  "endpoint": "https://openrouter.ai/api/v1",
  "inference": {
    "backend": "openrouter",
    "context_window_tokens": 131072,
    "episode_max_output_tokens": 8192,
    "max_output_tokens": 8192,
    "context_safety_tokens": 8192,
    "reasoning_effort": "medium"
  }
}
```

- [x] Create the full runnable config using the existing paper config shape. Add the blank key name to `.env.example`. Document uv/core dependencies, Qwen runtime, `plan`, one bare `rollout`, and then `search`; identify which requests incur charges. Never recommend `train`/`export` with this example.
- [x] Document Nitro as routing, not a distinct checkpoint, fixed provider, guaranteed speed, or fixed price. Explain reasoning/output budgets, usage records, unavailable cost values, partial-run inspection, and why fresh directories are needed for changed settings. The default host-visible execution behavior remains relevant.
- [x] Run all offline tests, shell syntax checks, `git diff --check`, and previews for paper, cycle2, and OpenRouter configurations. No credentials or network calls should be needed for previews.
- [ ] When execution of a paid smoke test is authorized and a key is supplied via environment, make bounded calls for optimizer text and judge JSON, then one tiny episode that writes and reads a sentinel file. Check tool-call follow-up, successful status, usage, and secret absence. Do not start the full research-task search automatically. If unavailable, record live validation as outstanding rather than claiming compatibility.
- [x] Commit the example/docs and report exactly which acceptance checks ran.

## Sources and implementation notes

Checked against current documentation on 2026-10-10:

- [OpenRouter quickstart](https://openrouter.ai/docs/quickstart): OpenAI-compatible base URL and bearer authentication.
- [GPT-OSS 120B](https://openrouter.ai/openai/gpt-oss-120b): model identifier and advertised 131K context. Verify provider-specific limits during the live test.
- [Nitro routing](https://openrouter.ai/docs/guides/routing/model-variants/nitro): throughput preference and priority-tier eligibility; charges follow the serving tier.
- [Reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens): reasoning configuration and preserving provider reasoning details across tool calls. Inspect the pinned Qwen runtime's follow-up request, not only the first streamed answer.
- [Usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting): usage/cost fields; do not use static model prices for routed requests.

The main compatibility uncertainty is Qwen Code's handling of GPT-OSS reasoning and tools through OpenRouter. Resolve it with Task 4's small live test before spending on long research episodes. Offline tests establish wiring and failure handling, not model quality or trading performance.
