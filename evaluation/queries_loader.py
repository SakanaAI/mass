from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def parse_numeric_cli_query_id(raw: str) -> str:
    """
    Parse ``--query-id`` when the CLI requires a bare integer (e.g. ``52`` → ``query_52``).

    Raises ``ValueError`` if the string is not only digits (no ``query_`` prefix).
    """
    s = raw.strip()
    if not re.fullmatch(r"\d+", s):
        raise ValueError(
            "--query-id must be a decimal integer, e.g. 52 (do not use query_52)"
        )
    return f"query_{int(s)}"


def normalize_query_id(raw: str) -> str:
    """
    Canonical form is ``query_<n>`` as stored in the task catalog JSON (e.g. ``query_3``).

    Accepts ``3``, ``03``, ``query_3``, ``Query_3``, or ``query3`` → ``query_3``.
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty query id")
    if s.isdigit():
        return f"query_{int(s)}"
    m = re.match(r"query_(\d+)$", s, re.IGNORECASE)
    if m:
        return f"query_{int(m.group(1))}"
    m = re.match(r"query(\d+)$", s, re.IGNORECASE)
    if m:
        return f"query_{int(m.group(1))}"
    return f"query_{s}"


def numeric_query_id(canonical_query_id: str) -> int:
    """``query_3`` → ``3``. Expects output of :func:`normalize_query_id`."""
    m = re.match(r"query_(\d+)$", canonical_query_id.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"not a canonical query id: {canonical_query_id!r}")
    return int(m.group(1))


def load_task_query(queries_json: Path, query_id: str) -> dict[str, Any]:
    """Return the task_queries entry for ``query_id`` (e.g. ``query_12``)."""
    qid = normalize_query_id(query_id)
    with open(queries_json, encoding="utf-8") as f:
        data = json.load(f)
    for row in data.get("task_queries", []):
        if row.get("query_id") == qid:
            return row
    raise KeyError(f"No task query {qid!r} in {queries_json}")


def infer_query_id_from_memory_dir_name(folder_name: str) -> str | None:
    """
    Best-effort ``query_<n>`` from the **folder basename** (submission root).

    Supported patterns:

    - Kajiki MAW: ``memory_workflow_<n>_YYYYMMDD_HHMMSS``
    - Codex / ablation-style: ``query_<n>_...`` or ``query<n>_...`` (e.g. ``query51_gpt-5.3-codex_medium``)

    If nothing matches, return ``None`` — use ``--query-id`` explicitly.
    """
    m = re.match(r"memory_workflow_(\d+)_\d{8}_\d{6}$", folder_name)
    if m:
        return normalize_query_id(m.group(1))
    m = re.match(r"^query_(\d+)(?:_|$)", folder_name, re.IGNORECASE)
    if m:
        return normalize_query_id(m.group(1))
    m = re.match(r"^query(\d+)(?:_|$)", folder_name, re.IGNORECASE)
    if m:
        return normalize_query_id(m.group(1))
    return None
