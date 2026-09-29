"""
Token usage and rough USD cost from judge-model API usage.

Costs are **estimates** from a small built-in table (standard tier, typical context).
Always persist raw ``usage`` from the API; refresh rates from
provider pricing docs when you care about exact dollars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# (input $/1M tokens, output $/1M tokens) — standard processing, typical context.
# Last aligned to public API pricing page; long-context / flex / batch differ.
_USD_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.5-pro": (15.00, 90.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Anthropic (approx standard rates; refresh against provider pricing page)
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    # Google Gemini (developer API base tier <=200k prompt tokens)
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}


def _rates_for_model(model: str) -> tuple[str, tuple[float, float]] | None:
    m = model.strip()
    if not m:
        return None
    key = m
    if key in _USD_PER_1M_TOKENS:
        return key, _USD_PER_1M_TOKENS[key]
    low = m.lower()
    if low in _USD_PER_1M_TOKENS:
        return low, _USD_PER_1M_TOKENS[low]
    # Snapshot / dated variants: gpt-5.4-2025-...
    for prefix, rates in (
        ("gpt-5.4-mini", _USD_PER_1M_TOKENS["gpt-5.4-mini"]),
        ("gpt-5.4-nano", _USD_PER_1M_TOKENS["gpt-5.4-nano"]),
        ("gpt-5.4-pro", _USD_PER_1M_TOKENS["gpt-5.4-pro"]),
        ("gpt-5.4", _USD_PER_1M_TOKENS["gpt-5.4"]),
        ("gpt-5.5-pro", _USD_PER_1M_TOKENS["gpt-5.5-pro"]),
        ("gpt-5.5", _USD_PER_1M_TOKENS["gpt-5.5"]),
        ("gpt-4o-mini", _USD_PER_1M_TOKENS["gpt-4o-mini"]),
        ("gpt-4o", _USD_PER_1M_TOKENS["gpt-4o"]),
        ("claude-opus-4-20250514", _USD_PER_1M_TOKENS["claude-opus-4-20250514"]),
        ("claude-opus-4-7", _USD_PER_1M_TOKENS["claude-opus-4-7"]),
        ("claude-opus-4-1", _USD_PER_1M_TOKENS["claude-opus-4-1"]),
        ("gemini-3.1-pro-preview", _USD_PER_1M_TOKENS["gemini-3.1-pro-preview"]),
        ("gemini-3.1-pro", _USD_PER_1M_TOKENS["gemini-3.1-pro"]),
    ):
        if low.startswith(prefix):
            return prefix, rates
    return None


def extract_usage_dict(response: Any) -> dict[str, Any]:
    """Serialize ``response.usage`` (or empty dict if missing)."""
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    if hasattr(u, "model_dump"):
        try:
            dumped = u.model_dump()
            return dict(dumped) if isinstance(dumped, dict) else {"raw": dumped}
        except Exception:
            pass
    out: dict[str, Any] = {}
    for k in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_details",
        "output_tokens_details",
    ):
        v = getattr(u, k, None)
        if v is None:
            continue
        if hasattr(v, "model_dump"):
            try:
                out[k] = v.model_dump()
            except Exception:
                out[k] = str(v)
        elif isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = int(v) if isinstance(v, float) else v
        elif isinstance(v, (str, type(None))):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def estimate_cost_usd(
    model: str,
    usage: dict[str, Any],
) -> tuple[float | None, str | None]:
    """
    Return ``(estimated_usd, note)``.

    Uses total ``input_tokens`` / ``output_tokens`` only; does not apply cached-input
    discounts or long-context multipliers.
    """
    found = _rates_for_model(model)
    if found is None:
        return None, (
            f"No built-in $/1M rates for model {model!r}; see evaluation/responses_metrics.py "
            "and verify pricing on the provider docs."
        )
    rate_key, (inp_per_m, out_per_m) = found
    inp = int(usage.get("input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    usd = inp / 1_000_000.0 * inp_per_m + out / 1_000_000.0 * out_per_m
    # Anthropic prompt-caching add-ons (only present when usage includes these fields).
    details = usage.get("input_tokens_details")
    if isinstance(details, dict):
        cache_create = int(details.get("cache_creation_input_tokens") or 0)
        cache_read = int(details.get("cache_read_input_tokens") or 0)
        if cache_create:
            usd += cache_create / 1_000_000.0 * inp_per_m * 1.25
        if cache_read:
            usd += cache_read / 1_000_000.0 * inp_per_m * 0.10
    note = (
        f"Linear estimate using table row {rate_key!r} "
        f"(input ${inp_per_m}/1M, output ${out_per_m}/1M); "
        "approximate only; may differ for tier/region/special modes."
    )
    return usd, note


@dataclass
class ResponsesCallMetrics:
    """Per ``responses.create`` call."""

    usage: dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
    cost_estimate_note: str | None = None
    response_id: str | None = None

    @classmethod
    def from_response(cls, response: Any, *, model: str) -> ResponsesCallMetrics:
        usage = extract_usage_dict(response)
        est, note = estimate_cost_usd(model, usage)
        rid = getattr(response, "id", None)
        rid_s = str(rid) if rid is not None else None
        return cls(
            usage=usage,
            estimated_cost_usd=est,
            cost_estimate_note=note,
            response_id=rid_s,
        )

    def to_json_dict(self) -> dict[str, Any]:
        est = self.estimated_cost_usd
        if est is None:
            est_out = None
        elif isinstance(est, (int, float)):
            est_out = round(float(est), 8)
        else:
            est_out = None
        return {
            "usage": self.usage,
            "estimated_cost_usd": est_out,
            "cost_estimate_note": self.cost_estimate_note,
            "response_id": self.response_id,
        }


def aggregate_usage_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum tokens and estimated USD from match/judge payload dicts with ``usage`` / ``estimated_cost_usd``."""
    inp = out = tot = 0
    cost_sum = 0.0
    cost_n = 0
    for row in rows:
        u = row.get("usage")
        if not isinstance(u, dict):
            continue
        inp += int(u.get("input_tokens") or 0)
        out += int(u.get("output_tokens") or 0)
        tot += int(u.get("total_tokens") or 0)
        c = row.get("estimated_cost_usd")
        if isinstance(c, (int, float)):
            cost_sum += float(c)
            cost_n += 1
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": tot,
        "estimated_cost_usd_sum": round(cost_sum, 6) if cost_n else None,
        "rows_with_cost_estimate": cost_n,
    }
