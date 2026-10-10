"""JSON-only LLM backends for recursive harness improvement.

The historical RHI path used OpenAI's Responses API.  The local path uses the
OpenAI-compatible Chat Completions endpoint exposed by vLLM.  Keeping both
behind one small interface makes the evaluator and harness optimizer use the
same explicitly selected backend.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import BadRequestError, DefaultHttpxClient, OpenAI

from evaluation.openai_responses import responses_create_adaptive
from evaluation.responses_metrics import ResponsesCallMetrics


BACKEND_OPENAI_RESPONSES = "openai-responses"
BACKEND_LOCAL_VLLM = "local-vllm"
BACKEND_OPENROUTER = "openrouter"
BACKEND_CHOICES = (BACKEND_OPENAI_RESPONSES, BACKEND_LOCAL_VLLM, BACKEND_OPENROUTER)


def _await_background_response(
    client: OpenAI,
    response: Any,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 30.0,
) -> Any:
    """Poll a background Responses job until it reaches a terminal status.

    Long non-streaming Responses calls can exceed NAT/conntrack idle windows
    (observed on GCP: the connection stays ESTABLISHED locally but the reply
    can never arrive, so the client waits until its own timeout). Background
    mode sidesteps that: the create returns immediately and each poll is a
    short, fresh HTTPS request.
    """
    deadline = time.monotonic() + timeout_seconds
    consecutive_poll_errors = 0
    while getattr(response, "status", None) in ("queued", "in_progress"):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Background response {response.id} still "
                f"{response.status!r} after {timeout_seconds:.0f}s"
            )
        time.sleep(poll_interval_seconds)
        try:
            response = client.responses.retrieve(response.id)
            consecutive_poll_errors = 0
        except Exception:
            consecutive_poll_errors += 1
            if consecutive_poll_errors > 5:
                raise
    if getattr(response, "status", None) != "completed":
        raise RuntimeError(
            f"Background response {response.id} ended with status "
            f"{getattr(response, 'status', None)!r}: "
            f"error={getattr(response, 'error', None)!r} "
            f"incomplete_details={getattr(response, 'incomplete_details', None)!r}"
        )
    return response


@dataclass
class JsonCallResult:
    payload: dict[str, Any]
    raw_text: str
    metrics: dict[str, Any]


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise RuntimeError("LLM returned an empty response.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Fallback candidates, tried in order; accept the first that parses.
        # The outermost-brace slice comes first: a fenced ``` block can also
        # appear ESCAPED inside a JSON string value (e.g. a design text that
        # itself contains a code fence), in which case the fenced regex
        # captures an unparseable fragment.
        candidates = []
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1))
        if not candidates:
            raise RuntimeError(f"LLM returned non-JSON output:\n{text[:4000]}")
        value = None
        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                last_exc = exc
        if value is None:
            raise RuntimeError(
                f"LLM returned invalid JSON: {last_exc}\n---\n{text[:4000]}"
            ) from last_exc
    if not isinstance(value, dict):
        raise RuntimeError(f"LLM returned JSON {type(value).__name__}; expected an object.")
    return value


def _responses_text(response: Any) -> str:
    raw = getattr(response, "output_text", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "output_text":
                text = getattr(block, "text", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return ""


def _chat_usage_metrics(response: Any, *, model: str, backend: str = BACKEND_LOCAL_VLLM) -> dict[str, Any]:
    usage_obj = getattr(response, "usage", None)
    usage_raw: dict[str, Any] = {}
    if usage_obj is not None and hasattr(usage_obj, "model_dump"):
        dumped = usage_obj.model_dump()
        if isinstance(dumped, dict):
            usage_raw = dumped
    prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
    completion_tokens = int(usage_raw.get("completion_tokens") or 0)
    total_tokens = int(
        usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens)
    )
    usage: dict[str, Any] = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    prompt_details = usage_raw.get("prompt_tokens_details")
    completion_details = usage_raw.get("completion_tokens_details")
    if isinstance(prompt_details, dict):
        usage["input_tokens_details"] = prompt_details
    if isinstance(completion_details, dict):
        usage["output_tokens_details"] = completion_details
    cost = usage_raw.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        cost = None
    remote = backend == BACKEND_OPENROUTER
    return {
        "usage": usage,
        "raw_usage": usage_raw,
        "estimated_cost_usd": cost if remote else 0.0,
        "cost_estimate_note": ("OpenRouter-reported cost; unavailable when null." if remote else
                               "Locally hosted inference; API cost recorded as $0."),
        "response_id": str(getattr(response, "id", "") or "") or None,
        "requested_model": model,
        "served_model": getattr(response, "model", None) or model,
        "provider": getattr(response, "provider", None),
    }


class JsonLLMBackend:
    """Call either OpenAI Responses or a local vLLM chat endpoint for JSON."""

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        enable_thinking: bool = True,
        seed: int | None = None,
        timeout_seconds: float = 3600.0,
        background: bool = False,
    ) -> None:
        if backend not in BACKEND_CHOICES:
            raise ValueError(f"Unsupported backend: {backend}")
        if backend == BACKEND_LOCAL_VLLM and not base_url:
            raise ValueError("local-vllm requires base_url")
        if backend == BACKEND_OPENROUTER:
            if not api_key.strip():
                raise ValueError("OPENROUTER_API_KEY is required")
            if (base_url or "").rstrip("/") != "https://openrouter.ai/api/v1":
                raise ValueError("OpenRouter requires https://openrouter.ai/api/v1")
        self.backend = backend
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.background = background and backend == BACKEND_OPENAI_RESPONSES
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
            **({"http_client": DefaultHttpxClient(follow_redirects=False)}
               if backend == BACKEND_OPENROUTER else {}),
        )

    def verify_model(self) -> None:
        """Fail early when the selected model is not served by local vLLM."""
        if self.backend == BACKEND_OPENAI_RESPONSES:
            return
        models = self.client.models.list()
        ids = {str(item.id) for item in models.data}
        model_id = self.model.removesuffix(":nitro") if self.backend == BACKEND_OPENROUTER else self.model
        if model_id not in ids:
            raise RuntimeError(
                f"Endpoint is reachable but does not serve {self.model!r}; "
                f"available models: {sorted(ids)}"
            )

    def call_json(self, *, system_prompt: str, user_prompt: str) -> JsonCallResult:
        if self.backend == BACKEND_OPENAI_RESPONSES:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": system_prompt,
                "input": user_prompt,
                "text": {"format": {"type": "json_object"}},
            }
            if self.reasoning_effort:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.max_output_tokens is not None:
                kwargs["max_output_tokens"] = self.max_output_tokens
            if self.background:
                kwargs["background"] = True
            response = responses_create_adaptive(self.client, **kwargs)
            if self.background:
                response = _await_background_response(
                    self.client, response, timeout_seconds=self.timeout_seconds
                )
            raw = _responses_text(response)
            metrics = ResponsesCallMetrics.from_response(
                response, model=self.model
            ).to_json_dict()
            return JsonCallResult(
                payload=_extract_json_object(raw),
                raw_text=raw,
                metrics=metrics,
            )

        response = self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
        )
        if not response.choices:
            raise RuntimeError("Local vLLM returned no completion choices.")
        message = response.choices[0].message
        if getattr(message, "refusal", None) or getattr(response, "error", None):
            raise RuntimeError("Model refused or failed the JSON request")
        raw_content = message.content
        raw = raw_content.strip() if isinstance(raw_content, str) else ""
        if not raw:
            reasoning = getattr(message, "reasoning_content", None)
            reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
            finish_reason = response.choices[0].finish_reason
            raise RuntimeError(
                "Local vLLM returned no final JSON content "
                f"(finish_reason={finish_reason!r}, reasoning_chars={reasoning_chars}). "
                "This usually means the output ceiling was consumed by reasoning."
            )
        return JsonCallResult(
            payload=_extract_json_object(raw),
            raw_text=raw,
            metrics=_chat_usage_metrics(response, model=self.model, backend=self.backend),
        )

    def call_text(self, *, system_prompt: str, user_prompt: str) -> JsonCallResult:
        """Return unwrapped text while retaining the same metrics container."""
        if self.backend == BACKEND_OPENAI_RESPONSES:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": system_prompt,
                "input": user_prompt,
            }
            if self.reasoning_effort:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.max_output_tokens is not None:
                kwargs["max_output_tokens"] = self.max_output_tokens
            if self.background:
                kwargs["background"] = True
            response = responses_create_adaptive(self.client, **kwargs)
            if self.background:
                response = _await_background_response(
                    self.client, response, timeout_seconds=self.timeout_seconds
                )
            raw = _responses_text(response)
            if not raw:
                raise RuntimeError("OpenAI Responses returned no final text.")
            metrics = ResponsesCallMetrics.from_response(
                response, model=self.model
            ).to_json_dict()
            return JsonCallResult(payload={}, raw_text=raw, metrics=metrics)

        response = self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=False,
        )
        if not response.choices:
            raise RuntimeError("Local vLLM returned no completion choices.")
        message = response.choices[0].message
        if getattr(message, "refusal", None) or getattr(response, "error", None):
            raise RuntimeError("Model refused or failed the text request")
        raw_content = message.content
        raw = raw_content.strip() if isinstance(raw_content, str) else ""
        if not raw:
            reasoning = getattr(message, "reasoning_content", None)
            reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
            finish_reason = response.choices[0].finish_reason
            raise RuntimeError(
                "Local vLLM returned no final design text "
                f"(finish_reason={finish_reason!r}, reasoning_chars={reasoning_chars})."
            )
        return JsonCallResult(
            payload={},
            raw_text=raw,
            metrics=_chat_usage_metrics(response, model=self.model, backend=self.backend),
        )

    def _chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                **({"top_k": self.top_k} if self.top_k is not None else {}),
            },
        }
        if self.backend == BACKEND_OPENROUTER:
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort or "medium"}}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.seed is not None:
            kwargs["seed"] = self.seed

        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if self.backend != BACKEND_LOCAL_VLLM or not json_mode or "'response_format.type' must be 'json_schema' or 'text'" not in str(exc):
                raise
            # LM Studio rejects json_object; retain prompt-based JSON and caller validation.
            kwargs.pop("response_format")
            return self.client.chat.completions.create(**kwargs)
