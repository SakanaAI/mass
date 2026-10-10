"""Self-evaluation uses the current generation; reporting uses external judges."""
import os
import random


def bundles(task, workspace_a, workspace_b, *, remote=False):
    from evaluation.bundle import build_judge_bundle
    from .io import ROOT
    task_text = (ROOT / f"tasks/bare/query{task}.txt").read_text()
    task_text = task_text.split("Use uv for all Python workflows")[0].replace("\\n", "\n").strip()
    row = {"query_id": f"query_{task}", "query": task_text}
    return [build_judge_bundle(memory_root=p, task_row=row, log_path=None,
                              report_max_tokens=12000 if remote else None, json_max_tokens=6000 if remote else 50000,
                              code_max_tokens=4000 if remote else 10000, token_count_model="gpt-5.5")
            for p in (workspace_a, workspace_b)]


def self_judge(config, task, workspace_a, workspace_b, seed):
    from harness_improvement.llm_backend import JsonLLMBackend
    from evaluation_claudecodex.pairwise_judge import build_pairwise_user_message
    from evaluation_claudecodex.pairwise_prompts import pairwise_judge_system_for_query_num
    from evaluation_claudecodex.pairwise_schema import PairwiseJudgeResult
    from .pipeline import inference_options
    inference = inference_options(config)
    remote = inference["backend"] == "openrouter"
    a, b = bundles(task, workspace_a, workspace_b, remote=remote)
    system = pairwise_judge_system_for_query_num(task)
    user = build_pairwise_user_message(a, b, label_a="candidate-a", label_b="candidate-b")
    if remote:
        from harness_improvement.iterate_multi_agent_prompt import _enforce_context_budget
        _enforce_context_budget(stage="ranking", system_prompt=system, user_prompt=user,
                                max_output_tokens=inference["max_output_tokens"],
                                context_window_tokens=inference["context_window_tokens"],
                                context_safety_tokens=inference["context_safety_tokens"])
    backend = JsonLLMBackend(backend=inference["backend"], model=config["model_id"],
                            base_url=config["endpoint"],
                            api_key=os.environ.get("OPENROUTER_API_KEY", "") if remote else os.environ.get("VLLM_API_KEY", "EMPTY"),
                            reasoning_effort=inference["reasoning_effort"] if remote else None,
                            temperature=None if remote else 0.6, top_p=None if remote else 0.95, top_k=None if remote else 20,
                            max_output_tokens=inference["max_output_tokens"], enable_thinking=True, seed=seed)
    backend.verify_model()
    result = backend.call_json(system_prompt=system, user_prompt=user)
    verdict = PairwiseJudgeResult.model_validate(result.payload)
    return {"winner": verdict.winner, "rationale": verdict.rationale,
            "model": config["model_id"], "seed": seed, "metrics": result.metrics}


def external_judgments(task, workspace_a, workspace_b, skip=()):
    from openai import OpenAI
    from anthropic import Anthropic
    from evaluation_claudecodex.pairwise_judge import run_pairwise_judge
    a, b = bundles(task, workspace_a, workspace_b)
    for provider, model, effort, client in [
        ("openai", "gpt-5.5", "xhigh", OpenAI(max_retries=0)),
        ("anthropic", "claude-opus-4-8", "max", Anthropic(max_retries=0)),
    ]:
        for seed in [42, 43, 44]:
            if (provider, seed) in skip:
                continue
            # Provider APIs do not expose identical deterministic seed controls.
            # The seed controls presentation; each call is an independent judgment.
            swapped = random.Random(seed).choice([False, True])
            left, right = (b, a) if swapped else (a, b)
            verdict, metrics = run_pairwise_judge(
                client=client, provider=provider, model=model,
                bundle_a=left, bundle_b=right, label_a="candidate-a", label_b="candidate-b",
                reasoning_effort=effort, judge_max_tokens=32768)
            winner = {"A": "B", "B": "A"}.get(verdict.winner, verdict.winner) if swapped else verdict.winner
            yield {"provider": provider, "model": model, "seed": seed, "swapped": swapped,
                   "winner": winner, "rationale": verdict.rationale, "metrics": metrics.to_json_dict()}
