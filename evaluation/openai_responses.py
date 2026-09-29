"""Helpers for ``client.responses.create`` across OpenAI SDK / model quirks."""

from __future__ import annotations

from typing import Any

from openai import BadRequestError, OpenAI


def responses_create_adaptive(client: OpenAI, **kwargs: Any) -> Any:
    """
    Call ``responses.create``, retrying after stripping unsupported kwargs.

    Some models return **400** for ``temperature`` or ``reasoning`` (e.g. "Unsupported parameter:
    'temperature' is not supported with this model."). Older SDKs may raise **TypeError** for
    unknown keyword ``reasoning``.
    """
    kw = dict(kwargs)
    last_err: BaseException | None = None
    for _ in range(8):
        try:
            return client.responses.create(**kw)
        except TypeError as e:
            last_err = e
            if "reasoning" in kw:
                kw.pop("reasoning", None)
                continue
            raise
        except BadRequestError as e:
            last_err = e
            msg = str(e).lower()
            if "temperature" in msg and "temperature" in kw:
                kw.pop("temperature", None)
                continue
            if "reasoning" in msg and "reasoning" in kw:
                kw.pop("reasoning", None)
                continue
            raise
    raise RuntimeError(f"responses.create failed after retries: {last_err!r}") from last_err
