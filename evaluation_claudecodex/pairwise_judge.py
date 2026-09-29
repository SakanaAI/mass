from __future__ import annotations

import re
from typing import Any
from typing import Literal

from evaluation.bundle import JudgeBundle, bundle_workspace_evidence
from evaluation.openai_responses import responses_create_adaptive
from evaluation.queries_loader import numeric_query_id
from evaluation.responses_metrics import ResponsesCallMetrics, estimate_cost_usd
from evaluation_claudecodex.pairwise_prompts import pairwise_judge_system_for_query_num, pairwise_user_suffix
from evaluation_claudecodex.pairwise_schema import PairwiseJudgeResult

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - optional dependency
    Anthropic = None  # type: ignore[assignment]
try:
    from google.genai import types as google_types
except Exception:  # pragma: no cover - optional dependency
    google_types = None  # type: ignore[assignment]


JudgeProvider = Literal["openai", "anthropic", "google"]


def build_pairwise_user_message(
    bundle_a: JudgeBundle,
    bundle_b: JudgeBundle,
    *,
    label_a: str,
    label_b: str,
) -> str:
    if bundle_a.task_text != bundle_b.task_text:
        raise ValueError("Pairwise bundles must share the same task text")
    wa = bundle_workspace_evidence(
        bundle_a,
        workspace_section_title="# Workspace for submission **A** (evidence only)",
    )
    wb = bundle_workspace_evidence(
        bundle_b,
        workspace_section_title="# Workspace for submission **B** (evidence only)",
    )
    body = "\n\n".join(
        [
            "# Task",
            "",
            bundle_a.task_text,
            "",
            "# Labels (for orientation only)",
            f"- Submission **A** corresponds to run type: `{label_a}`",
            f"- Submission **B** corresponds to run type: `{label_b}`",
            "",
            wa,
            "",
            wb,
            "",
            pairwise_user_suffix().strip(),
        ]
    )
    return body.strip()


def _text_from_response(response: Any) -> str:
    raw = getattr(response, "output_text", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "output_text":
                t = getattr(block, "text", "")
                if isinstance(t, str) and t.strip():
                    return t.strip()
    return ""


def _text_from_anthropic_message(message: Any) -> str:
    chunks: list[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            t = getattr(block, "text", "")
            if isinstance(t, str) and t.strip():
                chunks.append(t.strip())
    return "\n".join(chunks).strip()


def _extract_json_candidate(raw: str) -> str:
    s = raw.strip()
    if not s:
        return s
    # Accept markdown code-fenced JSON often returned by non-strict output routes.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1].strip()
    return s


def _usage_dict_from_anthropic_message(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    cache_create = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    out_dict: dict[str, Any] = {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
    }
    if cache_create:
        out_dict["input_tokens_details"] = {"cache_creation_input_tokens": cache_create}
    if cache_read:
        details = out_dict.setdefault("input_tokens_details", {})
        details["cache_read_input_tokens"] = cache_read
    return out_dict


def _anthropic_budget_tokens(reasoning_effort: str | None) -> int | None:
    if reasoning_effort is None:
        return None
    effort = reasoning_effort.strip().lower()
    if not effort:
        return None
    return {
        "low": 1024,
        "medium": 4096,
        "high": 8192,
    }.get(effort, 4096)


_VALID_ANTHROPIC_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _anthropic_output_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    effort = reasoning_effort.strip().lower()
    if not effort:
        return None
    if effort in _VALID_ANTHROPIC_EFFORTS:
        return effort
    return effort


def _google_thinking_level(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    effort = reasoning_effort.strip().lower()
    if not effort:
        return None
    return {
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }.get(effort, "MEDIUM")


def _usage_dict_from_google_response(response: Any) -> dict[str, Any]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    inp = int(getattr(meta, "prompt_token_count", 0) or 0)
    out = int(getattr(meta, "candidates_token_count", 0) or 0)
    tot = int(getattr(meta, "total_token_count", 0) or (inp + out))
    out_dict: dict[str, Any] = {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": tot,
    }
    thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0)
    cached = int(getattr(meta, "cached_content_token_count", 0) or 0)
    if thoughts:
        out_dict["output_tokens_details"] = {"thoughts_token_count": thoughts}
    if cached:
        details = out_dict.setdefault("input_tokens_details", {})
        details["cached_content_token_count"] = cached
    return out_dict


def run_pairwise_judge(
    *,
    client: Any,
    provider: JudgeProvider = "openai",
    model: str,
    bundle_a: JudgeBundle,
    bundle_b: JudgeBundle,
    label_a: str,
    label_b: str,
    temperature: float | None = None,
    reasoning_effort: str | None = "medium",
    judge_max_tokens: int | None = None,
) -> tuple[PairwiseJudgeResult, ResponsesCallMetrics]:
    qn = numeric_query_id(bundle_a.task_query_id)
    if bundle_b.task_query_id != bundle_a.task_query_id:
        raise ValueError("Pairwise bundles must share the same task_query_id")
    instructions = pairwise_judge_system_for_query_num(qn)
    user = build_pairwise_user_message(bundle_a, bundle_b, label_a=label_a, label_b=label_b)
    provider_norm = provider.strip().lower()
    if provider_norm == "openai":
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": user,
            "text": {"format": {"type": "json_object"}},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None and reasoning_effort != "":
            kwargs["reasoning"] = {"effort": reasoning_effort}
        response = responses_create_adaptive(client, **kwargs)
        raw = _text_from_response(response)
        if not raw:
            raise RuntimeError("Empty pairwise judge response from Responses API")
        try:
            result = PairwiseJudgeResult.model_validate_json(raw)
        except Exception as e:
            raise RuntimeError(f"Pairwise judge returned invalid JSON: {e}\n---\n{raw[:4000]}") from e
        metrics = ResponsesCallMetrics.from_response(response, model=model)
        return result, metrics

    if provider_norm == "anthropic":
        if Anthropic is None:
            raise RuntimeError("anthropic package is not installed. Please install dependencies first.")
        effort = _anthropic_output_effort(reasoning_effort)
        kwargs: dict[str, Any] = {
            "model": model,
            "system": instructions,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": judge_max_tokens if judge_max_tokens is not None else 4096,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if effort is not None:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort}
        # The Anthropic SDK requires streaming for requests whose configured
        # output ceiling could take more than ten minutes. We still consume the
        # stream into one final Message, so parsing and persisted usage remain
        # identical to the non-streaming route.
        with client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        raw = _text_from_anthropic_message(response)
        if not raw:
            raise RuntimeError(
                "Empty pairwise judge response from Anthropic Messages API "
                f"(stop_reason={getattr(response, 'stop_reason', None)!r}, "
                f"usage={_usage_dict_from_anthropic_message(response)!r})"
            )
        raw_json = _extract_json_candidate(raw)
        try:
            result = PairwiseJudgeResult.model_validate_json(raw_json)
        except Exception as e:
            raise RuntimeError(
                f"Pairwise judge returned invalid JSON: {e} "
                f"(stop_reason={getattr(response, 'stop_reason', None)!r}, "
                f"usage={_usage_dict_from_anthropic_message(response)!r})"
                f"\n---\n{raw[:4000]}"
            ) from e
        usage = _usage_dict_from_anthropic_message(response)
        est, note = estimate_cost_usd(model, usage)
        metrics = ResponsesCallMetrics(
            usage=usage,
            estimated_cost_usd=est,
            cost_estimate_note=note,
            response_id=str(getattr(response, "id", "")) or None,
        )
        return result, metrics

    if provider_norm == "google":
        if google_types is None:
            raise RuntimeError("google-genai types are not available. Please install/update google-genai.")
        thinking_level = _google_thinking_level(reasoning_effort)
        max_output_tokens = judge_max_tokens if judge_max_tokens is not None else 4096
        cfg = google_types.GenerateContentConfig(
            system_instruction=instructions,
            response_mime_type="application/json",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=google_types.ThinkingConfig(
                thinking_level=thinking_level,
                # Keep internal thoughts hidden to reduce leakage/noise in JSON judge output.
                include_thoughts=False,
            )
            if thinking_level is not None
            else None,
        )
        try:
            response = client.models.generate_content(model=model, contents=user, config=cfg)
        except Exception:
            # Fallback for SDK/model versions where thinking config is rejected.
            cfg_no_thinking = google_types.GenerateContentConfig(
                system_instruction=instructions,
                response_mime_type="application/json",
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            response = client.models.generate_content(model=model, contents=user, config=cfg_no_thinking)
        raw = (getattr(response, "text", None) or "").strip()
        if not raw:
            raise RuntimeError("Empty pairwise judge response from Google Gemini API")
        raw_json = _extract_json_candidate(raw)
        try:
            result = PairwiseJudgeResult.model_validate_json(raw_json)
        except Exception as e:
            raise RuntimeError(
                f"Pairwise judge returned invalid JSON: {e}\n---\n{raw[:4000]}"
            ) from e
        usage = _usage_dict_from_google_response(response)
        est, note = estimate_cost_usd(model, usage)
        metrics = ResponsesCallMetrics(
            usage=usage,
            estimated_cost_usd=est,
            cost_estimate_note=note,
            response_id=str(getattr(response, "response_id", "")) or None,
        )
        return result, metrics

    raise ValueError(f"Unsupported provider {provider!r}; expected 'openai', 'anthropic', or 'google'.")
