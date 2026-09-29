#!/usr/bin/env python3
"""RHI: execute a workflow, compare artifacts, and propose the next workflow.

Use ``python -m mass search`` for the paper configuration.
"""

from __future__ import annotations

import argparse
import difflib
import ipaddress
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from evaluation.artifacts import scan_deliverables
from evaluation.bundle import build_judge_bundle, bundle_workspace_evidence
from evaluation_claudecodex.pairwise_judge import (
    build_pairwise_user_message,
)
from evaluation_claudecodex.pairwise_prompts import pairwise_judge_system_for_query_num
from evaluation_claudecodex.pairwise_schema import PairwiseJudgeResult
from harness_improvement.llm_backend import (
    BACKEND_CHOICES,
    BACKEND_LOCAL_VLLM,
    BACKEND_OPENAI_RESPONSES,
    JsonLLMBackend,
)

DEFAULT_START_MARKER = "Create an agent team with following agent candidates to solve this problem:"
DEFAULT_END_MARKER = "Use uv for all Python workflows"


@dataclass
class QueryTemplate:
    query_num: int
    query_name: str
    raw_text: str
    before_design: str
    design_block: str
    after_design: str
    task_text_for_eval: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_query_template(
    query_file: Path,
    *,
    start_marker: str,
    end_marker: str,
) -> QueryTemplate:
    raw = query_file.read_text(encoding="utf-8")
    m_query = re.search(r"query(\d+)_ourTeam", query_file.stem)
    if not m_query:
        raise ValueError(f"Cannot parse query number from filename: {query_file.name}")
    query_num = int(m_query.group(1))
    query_name = f"query{query_num}_ourTeam"

    start_idx = raw.find(start_marker)
    if start_idx < 0:
        raise ValueError(f"Start marker not found in {query_file}: {start_marker!r}")
    # Use the last end marker in the file because the phrase can appear
    # inside the design block as a normal instruction line.
    end_idx = raw.rfind(end_marker)
    if end_idx < 0:
        raise ValueError(f"End marker not found in {query_file}: {end_marker!r}")
    if end_idx <= start_idx:
        raise ValueError(
            f"End marker occurs before start marker in {query_file}: {end_marker!r}"
        )

    before = raw[:start_idx]
    design = raw[start_idx:end_idx].strip()
    after = raw[end_idx:]

    # Some query files may store task text with escaped newlines.
    task_text = before.rstrip()
    if "\\n" in task_text:
        task_text = task_text.replace("\\n", "\n")
    task_text = task_text.strip()

    return QueryTemplate(
        query_num=query_num,
        query_name=query_name,
        raw_text=raw,
        before_design=before,
        design_block=design,
        after_design=after,
        task_text_for_eval=task_text,
    )


def _resolve_iteration_repo(runs_root: Path, query_name: str, version: int) -> Path:
    if version < 0:
        raise ValueError("version must be >= 0")
    if version == 0:
        return runs_root / query_name
    versioned = runs_root / f"{query_name}-v{version}"
    if versioned.is_dir():
        return versioned
    # Backward-compatible fallback: some runs roots store improved versions
    # in plain query folders without the -v{n} suffix.
    plain = runs_root / query_name
    if plain.is_dir():
        return plain
    return versioned


def _infer_versioned_runs_root(base_runs_root: Path, version: int) -> Path | None:
    """Infer sibling runs root from a trailing _v{n} naming scheme.

    Supports both conventions:
    - v0 as explicit suffix:   name_v0 -> name_v1 -> name_v2
    - v0 as base name:         name -> name_v1 -> name_v2
      (for version == 0, prefer stripping the _v{n} suffix).
    """
    m = re.search(r"_v\d+$", base_runs_root.name)
    if not m:
        return None

    # If caller asks for v0, first try the unsuffixed base name.
    if version == 0:
        unsuffixed_name = re.sub(r"_v\d+$", "", base_runs_root.name)
        unsuffixed = base_runs_root.with_name(unsuffixed_name)
        if unsuffixed.is_dir():
            return unsuffixed

    inferred_name = re.sub(r"_v\d+$", f"_v{version}", base_runs_root.name)
    inferred = base_runs_root.with_name(inferred_name)
    if inferred.is_dir():
        return inferred
    return None


def _write_full_prompt_dump(path: Path, *, system_prompt: str, user_prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "=== SYSTEM PROMPT ===\n\n"
        f"{system_prompt.strip()}\n\n"
        "=== USER PROMPT ===\n\n"
        f"{user_prompt.strip()}\n"
    )
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == body:
            return
        raise RuntimeError(
            "Refusing to overwrite an existing prompt with different content: "
            f"{path}"
        )
    path.write_text(body, encoding="utf-8")


def _enforce_context_budget(
    *,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int | None,
    context_window_tokens: int | None,
    context_safety_tokens: int,
) -> tuple[int, int]:
    """Apply the same conservative context guard to every local model call."""
    estimated_input_tokens = max(
        1, (len(system_prompt) + len(user_prompt) + 3) // 4
    )
    estimated_total_with_output = estimated_input_tokens + int(
        max_output_tokens or 0
    )
    if context_window_tokens is not None:
        allowed = context_window_tokens - context_safety_tokens
        if estimated_total_with_output > allowed:
            raise SystemExit(
                f"Conservative context guard failed for {stage}: "
                f"estimated_input_tokens={estimated_input_tokens}, "
                f"max_output_tokens={max_output_tokens or 0}, "
                f"total={estimated_total_with_output}, allowed={allowed}. "
                "Reduce evidence caps or output tokens."
            )
    return estimated_input_tokens, estimated_total_with_output


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    """Write an immutable JSON artifact, accepting an identical existing value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Refusing to overwrite invalid existing JSON: {path}: {exc}"
            ) from exc
        if existing == payload:
            return
        raise RuntimeError(
            f"Refusing to overwrite an existing JSON artifact: {path}"
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_history(history_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not history_path.is_file():
        return rows
    with history_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_history_row(history_path: Path, row: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _history_key(version_a: int, version_b: int) -> str:
    return f"v{version_a}_vs_v{version_b}"


def _is_loopback_base_url(value: str) -> bool:
    """Return whether an API base URL is explicitly local to this host."""
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        if hostname.lower() == "localhost":
            return True
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _adjacent_history_provenance_errors(
    row: dict[str, Any],
    *,
    comparison_key: str,
    prev_version: int,
    current_version: int,
    prev_repo: Path,
    current_repo: Path,
    llm_backend: str,
    model: str,
    base_url: str | None,
) -> list[str]:
    """Validate that reused pairwise history belongs to this exact RHI step."""
    errors: list[str] = []
    expected_scalars = {
        "key": comparison_key,
        "version_a": prev_version,
        "version_b": current_version,
        "llm_backend": llm_backend,
        "model": model,
    }
    for field, expected in expected_scalars.items():
        if row.get(field) != expected:
            errors.append(
                f"{field}={row.get(field)!r}, expected {expected!r}"
            )

    expected_repos = {
        "repo_a": prev_repo,
        "repo_b": current_repo,
    }
    for field, expected in expected_repos.items():
        raw = row.get(field)
        if not isinstance(raw, str) or Path(raw).expanduser().resolve() != expected:
            errors.append(f"{field}={raw!r}, expected {str(expected)!r}")

    if llm_backend == BACKEND_LOCAL_VLLM:
        actual_url = str(row.get("base_url") or "").rstrip("/")
        expected_url = str(base_url or "").rstrip("/")
        if actual_url != expected_url:
            errors.append(
                f"base_url={row.get('base_url')!r}, expected {base_url!r}"
            )
        metrics = row.get("call_metrics")
        cost = metrics.get("estimated_cost_usd") if isinstance(metrics, dict) else None
        if cost != 0.0:
            errors.append(
                f"call_metrics.estimated_cost_usd={cost!r}, expected 0.0"
            )
    return errors


def _history_action_items(rationale: str, *, max_items: int = 6) -> list[str]:
    cues = (
        "fail",
        "fails",
        "failed",
        "missing",
        "inconsistent",
        "mismatch",
        "not ",
        "lack",
        "weak",
        "nan",
        "null",
        "error",
        "defect",
        "reproducibility",
    )
    cleaned = " ".join(rationale.split())
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    picks: list[str] = []
    seen = set()
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        low = s.lower()
        if not any(cue in low for cue in cues):
            continue
        if s in seen:
            continue
        seen.add(s)
        picks.append(s)
        if len(picks) >= max_items:
            break
    return picks


def _shorten(text: str, *, max_chars: int) -> str:
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3].rstrip() + "..."


def _design_diffs_for_history(
    history_rows: list[dict[str, Any]],
    versions_dir: Path,
    *,
    recent_full: int = 3,
    recent_cap: int = 1500,
    old_cap: int = 400,
    baseline_mode: bool = False,
) -> dict[str, str]:
    """Compact unified diffs of the design blocks behind each history entry.

    Deterministic and API-free: reads versions/multi_agent_design_v{k}.txt.
    The most recent `recent_full` pairs get `recent_cap` chars each; older
    pairs get `old_cap` — recency carries the credit-assignment signal while
    keeping total prompt mass bounded for the local optimizer. In
    baseline_mode the entries are v{k}_vs_no_harness rows and each carries
    the diff v{k-1} -> v{k} (the v0 row has no diff).
    """
    adjacent: list[tuple[int, str]] = []
    for row in history_rows:
        key = str(row.get("key") or "")
        if baseline_mode:
            m = re.fullmatch(r"v(\d+)_vs_no_harness", key)
            if m and int(m.group(1)) >= 1:
                adjacent.append((int(m.group(1)), key))
            continue
        m = re.fullmatch(r"v(\d+)_vs_v(\d+)", key)
        if m and int(m.group(2)) == int(m.group(1)) + 1:
            adjacent.append((int(m.group(2)), key))
    adjacent = sorted(set(adjacent))
    diffs: dict[str, str] = {}
    for rank, (newer, key) in enumerate(reversed(adjacent)):
        older = newer - 1
        pa = versions_dir / f"multi_agent_design_v{older}.txt"
        pb = versions_dir / f"multi_agent_design_v{newer}.txt"
        if not (pa.is_file() and pb.is_file()):
            continue
        diff_lines = difflib.unified_diff(
            pa.read_text(encoding="utf-8").splitlines(),
            pb.read_text(encoding="utf-8").splitlines(),
            fromfile=f"multi_agent_design_v{older}",
            tofile=f"multi_agent_design_v{newer}",
            lineterm="",
            n=1,
        )
        text = "\n".join(diff_lines).strip()
        if not text:
            text = "(designs are identical)"
        cap = recent_cap if rank < recent_full else old_cap
        diffs[key] = _shorten(text, max_chars=cap)
    return diffs


def _history_summary_text(
    history_rows: list[dict[str, Any]],
    design_diffs: dict[str, str] | None = None,
) -> str:
    if not history_rows:
        return "- No prior pairwise history yet."
    lines: list[str] = []
    for row in history_rows:
        key = str(row.get("key", ""))
        comp = str(row.get("comparison", ""))
        winner = row.get("winner", "unknown")
        rationale = str(row.get("rationale", "")).strip()
        judged_at = str(row.get("judged_at_utc", ""))
        model = str(row.get("model", ""))
        repo_a = str(row.get("repo_a", ""))
        repo_b = str(row.get("repo_b", ""))
        action_items = _history_action_items(rationale)

        lines.append(f"- Comparison: {comp}")
        if key:
            lines.append(f"  - Key: {key}")
        lines.append(f"  - Winner: {winner}")
        if model:
            lines.append(f"  - Judge model: {model}")
        if judged_at:
            lines.append(f"  - Judged at (UTC): {judged_at}")
        if repo_a:
            lines.append(f"  - Repo A: {repo_a}")
        if repo_b:
            lines.append(f"  - Repo B: {repo_b}")
        if rationale:
            lines.append(f"  - Rationale (verbatim excerpt): {_shorten(rationale, max_chars=2400)}")
        if action_items:
            lines.append("  - Actionable takeaways for next version:")
            for item in action_items:
                lines.append(f"    - {item}")
        if design_diffs and key in design_diffs:
            lines.append(
                "  - Design change behind this verdict (unified diff of the "
                "design block; '-' lines were removed, '+' lines added):"
            )
            for diff_line in design_diffs[key].splitlines():
                lines.append(f"    {diff_line}")
        lines.append("  - Use this history as constraints: preserve winning traits, fix listed defects.")
    return "\n".join(lines)


def _history_delta_checklist_text(history_rows: list[dict[str, Any]]) -> str:
    if not history_rows:
        return "- No recurring issues yet."

    # Heuristic issue buckets for recurring weak signals in pairwise rationales.
    patterns: list[tuple[str, str]] = [
        ("report-metrics inconsistency", r"(inconsistent|mismatch|contradict|does not match|conflict)"),
        ("missing deliverables or sections", r"(missing|absent|not found|not present)"),
        ("stability/robustness gap", r"(stability|robustness|subperiod|pre_2020|post_2020)"),
        ("null/nan quality defect", r"(nan|null|none|non-standard json)"),
        ("reproducibility gap", r"(reproducibility|not reproducible|cannot reproduce|requirements|pin)"),
        ("validation/test weakness", r"(validation|test|insufficient evidence|not proven|unclear)"),
    ]

    counts = {name: 0 for name, _ in patterns}
    examples: dict[str, str] = {}
    for row in history_rows:
        rationale = str(row.get("rationale", "")).strip()
        if not rationale:
            continue
        low = rationale.lower()
        for name, pat in patterns:
            if re.search(pat, low):
                counts[name] += 1
                if name not in examples:
                    items = _history_action_items(rationale, max_items=1)
                    if items:
                        examples[name] = items[0]

    ranked = sorted(((name, c) for name, c in counts.items() if c > 0), key=lambda x: x[1], reverse=True)
    if not ranked:
        return "- No repeated issue patterns detected yet."

    lines = [
        "- Prioritize fixes for recurring issues below (higher count = more repeated in history):"
    ]
    for name, c in ranked:
        ex = examples.get(name)
        if ex:
            lines.append(f"- [{c}x] {name}: e.g., {ex}")
        else:
            lines.append(f"- [{c}x] {name}")
    return "\n".join(lines)


def _derive_champion(
    adjacent_rows: list[dict[str, Any]],
    champion_rows: list[dict[str, Any]],
    current_version: int,
) -> tuple[int, bool]:
    """Return (champion_version, arbitration_needed_for_current_version).

    Walk the adjacent verdicts v0..v{current}: the champion advances to the
    newer version only when that version beat the reigning champion DIRECTLY —
    either in its adjacent pair (champion == predecessor) or in a recorded
    champion-arbitration row. If the current version won its adjacent pair
    while the champion is older and no arbitration row exists yet, the second
    element is True: one champion-vs-current judge call is required. Older
    unresolved rounds leave the champion unchanged (conservative).
    """
    adj: dict[int, str | None] = {}
    for row in adjacent_rows:
        m = re.fullmatch(r"v(\d+)_vs_v(\d+)", str(row.get("key") or ""))
        if m and int(m.group(2)) == int(m.group(1)) + 1:
            adj[int(m.group(2))] = row.get("winner")
    arb: dict[tuple[int, int], str | None] = {}
    for row in champion_rows:
        m = re.fullmatch(r"champion_v(\d+)_vs_v(\d+)", str(row.get("key") or ""))
        if m:
            arb[(int(m.group(1)), int(m.group(2)))] = row.get("winner")
    champ = 0
    needs = False
    for k in range(1, current_version + 1):
        if adj.get(k) != "B":
            continue
        if champ == k - 1:
            champ = k
        elif (champ, k) in arb:
            if arb[(champ, k)] == "B":
                champ = k
        elif k == current_version:
            needs = True
    return champ, needs


def _derive_champion_baseline(
    history_rows: list[dict[str, Any]],
    champion_rows: list[dict[str, Any]],
    current_version: int,
) -> tuple[int, bool, bool]:
    """Baseline-anchored champion. Returns (champion, champion_has_baseline_win,
    arbitration_needed_for_current_version).

    Verdict convention: in v{k}_vs_no_harness rows, A = the baseline, B = the
    version, so winner "B" means the version beat the baseline. The champion
    advances only on DIRECT wins: a version's baseline win takes the crown
    outright while no champion has one (valid transitively: champ > baseline >
    others is not needed there); once a champion holds a baseline win, a new
    baseline-winner must beat it in a recorded arbitration. Losses and ties
    leave the champion untouched. With no baseline-winner at all, v0 (the
    seed) holds the crown by default WITHOUT a proven win.
    """
    wins: dict[int, Any] = {}
    for row in history_rows:
        m = re.fullmatch(r"v(\d+)_vs_no_harness", str(row.get("key") or ""))
        if m:
            wins[int(m.group(1))] = row.get("winner")
    arb: dict[tuple[int, int], Any] = {}
    for row in champion_rows:
        m = re.fullmatch(r"champion_v(\d+)_vs_v(\d+)", str(row.get("key") or ""))
        if m:
            arb[(int(m.group(1)), int(m.group(2)))] = row.get("winner")
    champ = 0
    champ_win = wins.get(0) == "B"
    needs = False
    for k in range(1, current_version + 1):
        if wins.get(k) != "B":
            continue
        if not champ_win:
            champ = k
            champ_win = True
        elif (champ, k) in arb:
            if arb[(champ, k)] == "B":
                champ = k
        elif k == current_version:
            needs = True
    return champ, champ_win, needs


def _baseline_scoreboard(
    history_rows: list[dict[str, Any]],
    current_version: int,
    champion: int | None,
    champion_has_win: bool,
) -> str:
    """One-line W/L/T record vs the baseline, with the champion starred."""
    verdicts: dict[int, Any] = {}
    for row in history_rows:
        m = re.fullmatch(r"v(\d+)_vs_no_harness", str(row.get("key") or ""))
        if m:
            verdicts[int(m.group(1))] = row.get("winner")
    marks = {"B": "W", "A": "L", "tie": "T"}
    parts = []
    for k in range(0, current_version + 1):
        mark = marks.get(verdicts.get(k), "?")
        star = "★" if (champion == k and champion_has_win) else ""
        parts.append(f"v{k} {mark}{star}")
    return "Record vs no_harness: " + " · ".join(parts)


def _build_single_submission_improvement_prompt(
    *,
    template: QueryTemplate,
    current_version: int,
    current_design: str,
    workspace_evidence: str,
    history_rows: list[dict[str, Any]],
    submission_label: str,
    execution_metadata: str | None,
    output_mode: str,
    max_design_chars: int | None,
    target_design_chars: int | None,
    forbid_fixed_agent_limits: bool = False,
    compact_era_prompt: bool = False,
    best_design: str | None = None,
    best_version: int | None = None,
    best_rationale: str | None = None,
    design_diffs: dict[str, str] | None = None,
    v0_design: str | None = None,
    baseline_mode: bool = False,
    baseline_scoreboard: str | None = None,
    best_baseline_case: str | None = None,
) -> tuple[str, str]:
    if output_mode == "json":
        response_rule = "Return valid JSON only."
        output_contract = (
            "Output schema:\n"
            "{{\n"
            '  "improved_multi_agent_design": "<full replacement text for the design block only>",\n'
            '  "changes_from_previous": ["<specific change 1>", "..."],\n'
            '  "why_it_should_improve": ["<mechanistic rationale 1>", "..."],\n'
            '  "evidence_used": ["<file/metric/history citation 1>", "..."],\n'
            '  "expected_impact": ["<expected gain in quality/reliability/reproducibility 1>", "..."],\n'
            '  "verification_checks": ["<how to validate this change worked in next iteration>", "..."]\n'
            "}}\n"
        )
    elif output_mode == "design-text":
        response_rule = (
            "Return only the full replacement multi-agent design block as plain text. "
            "Do not wrap it in JSON, Markdown fences, or explanatory prose."
        )
        output_contract = (
            "Output contract: return only the full replacement design block as plain "
            "text. Its first characters must be the required design marker. Do not "
            "include JSON metadata, Markdown fences, the task text, or the post-design "
            "prompt suffix.\n"
        )
    else:
        raise ValueError(f"Unsupported improvement output mode: {output_mode}")

    if target_design_chars is not None:
        if max_design_chars is None:
            raise ValueError(
                "target_design_chars requires a finite max_design_chars value"
            )
        if target_design_chars > max_design_chars:
            raise ValueError(
                "target_design_chars cannot exceed max_design_chars"
            )
        compact_system_rule = (
            "target improved_multi_agent_design at no more than "
            f"{target_design_chars:,} characters and never exceed "
            f"{max_design_chars:,} characters, and keep each explanatory "
        )
        compact_user_rule = (
            f"Target at most {target_design_chars:,} characters and never exceed "
            f"{max_design_chars:,} characters. Prefer shared "
        )
    elif max_design_chars is not None:
        compact_system_rule = (
            "keep improved_multi_agent_design under "
            f"{max_design_chars:,} characters, and keep each explanatory "
        )
        compact_user_rule = (
            f"Keep the replacement design under {max_design_chars:,} characters. "
            "Prefer shared "
        )
    else:
        # No explicit size flags: no compactness directive at all (claude-era
        # behavior; the qwen35-era generic "keep it compact" wording is only
        # emitted when a size flag is set).
        compact_system_rule = None
        compact_user_rule = None

    # --compact-era-prompt: byte-exact restoration of the compact-lineage
    # optimizer prompt (the generic compact wording that predated the
    # size-flag variants). Used to CONTINUE the archived compact lineage;
    # never set for the free lineage.
    if compact_era_prompt and compact_system_rule is None:
        compact_system_rule = (
            "keep improved_multi_agent_design compact, and keep each explanatory "
        )
        compact_user_rule = "Keep the replacement design compact. Prefer shared "

    if compact_system_rule is not None:
        size_rule_system = (
            "Keep the replacement focused and compact: preserve useful existing instructions, "
            "make evidence-targeted edits, avoid repeating the same protocol in every agent, "
            f"{compact_system_rule}"
            "JSON list to at most 8 concise items. This artifact-size constraint is not an "
            "execution or agent-count constraint.\n"
        )
        size_rule_user = (
            f"{compact_user_rule}"
            "protocols plus compact agent-specific deltas over duplicated schemas.\n"
        )
    else:
        size_rule_system = ""
        size_rule_user = ""

    # The no-numeric-caps directive is tied to the flag that also validates
    # the output for such caps; without the flag the prompt matches the
    # claude-era wording. --compact-era-prompt emits the SENTENCES without
    # the validator (the compact lineage had the sentences unconditionally
    # but never ran the validator).
    if forbid_fixed_agent_limits or compact_era_prompt:
        caps_rule_system = (
            "Do not introduce arbitrary numerical caps on agent or subagent invocations, "
            "recall/re-delegation loops, orchestration hops, tool calls, execution turns, "
            "or wall-clock duration. Terminate through acceptance gates or a genuinely "
            "unrecoverable documented blocker.\n"
        )
        caps_rule_user = (
            "Do not impose a fixed maximum number of recalls, agent/subagent calls, "
            "hops, turns, tools, or duration. Use acceptance-based convergence.\n"
        )
    else:
        caps_rule_system = ""
        caps_rule_user = ""

    system = (
        "You are a principal prompt engineer for autonomous coding agents.\n"
        f"The following query was executed by {submission_label}.\n"
        "Your task is to improve only the 'multi agent design' block of the query prompt to improve the quality of the query's deliverables (see 'Current multi agent design' below).\n"
        "Prioritize harnesses that multi-agent systems can do well (or have) and single-agent systems cannot (or does not have), "
        "such as parallel specialist decomposition, cross-agent critique, agent to agent communication, and iterative reconciliation.\n"
        "Prioritize two improvements: (1) stronger agent-to-agent communication contracts "
        "for each subagent 'Output to orchestrator' schema, and "
        "(2) more orchestrator-subagent hops with explicit iterative recall/re-delegation "
        "instead of one-pass unidirectional execution.\n"
        f"{caps_rule_system}"
        f"{size_rule_system}"
        "You must ground changes in concrete evidence from submission artifacts and comparison history.\n"
        f"{response_rule}"
    )
    save_rules= """
    Use uv for all Python workflows—run code with uv run, install dependencies with uv add, use uvx for tools. write your history log: write your plan, execution (or tool-execution), reflection during the reasoning logs in a logs.txt file. In logs.txt, Document all created and spawned agents, describe the workflow between agents (e.g., as a structured outline or diagram), track which agents were executed and their roles in the process. 
    """
    # BEST-H anchoring (default off): when the in-loop judge chain says an
    # OLDER design is still the best, show it next to the current design so
    # the optimizer can ratchet on the champion instead of a regressed base.
    best_sections: list[str] = []
    if best_design is not None and best_version is not None:
        rationale_text = ""
        if best_rationale:
            rationale_text = _shorten(best_rationale.strip(), max_chars=2000)
            if rationale_text and rationale_text[-1] not in ".!?\"'":
                rationale_text += "."
        if baseline_mode and best_baseline_case == "champion_only_win":
            rationale_sentence = (
                (
                    " The judge's stated reasons for the baseline's win over "
                    f"v{current_version}: {rationale_text}"
                )
                if rationale_text
                else ""
            )
            note_text = (
                f"NOTE: this earlier design v{best_version} is shown because it "
                f"BEAT the no-harness baseline while the current design "
                f"v{current_version} did not (see the record in section 3)."
                f"{rationale_sentence} Refer to BOTH v{best_version} and "
                f"v{current_version} when writing v{current_version + 1}: keep "
                f"what made v{best_version} strong, and merge in only the "
                f"genuinely useful changes attempted since."
            )
        elif baseline_mode and best_baseline_case == "both_win_arbitration":
            rationale_sentence = (
                f" The judge's stated reasons: {rationale_text}"
                if rationale_text
                else ""
            )
            note_text = (
                f"NOTE: both this earlier design v{best_version} and the current "
                f"design v{current_version} beat the no-harness baseline; in a "
                f"direct comparison the judge found v{best_version}'s "
                f"deliverables better.{rationale_sentence} Refer to BOTH "
                f"v{best_version} and v{current_version} when writing "
                f"v{current_version + 1}: keep what made v{best_version} strong, "
                f"and merge in only the genuinely useful changes attempted since."
            )
        else:
            rationale_sentence = (
                f" The judge's stated reasons for that verdict: {rationale_text}"
                if rationale_text
                else ""
            )
            note_text = (
                f"NOTE: this earlier design v{best_version} is shown because the in-loop "
                f"pairwise judge determined it produced BETTER deliverables than the "
                f"current design v{current_version} (v{best_version} > v{current_version} "
                f"in the comparison history).{rationale_sentence} Refer to BOTH "
                f"v{best_version} and "
                f"v{current_version} when writing v{current_version + 1}: keep what made "
                f"v{best_version} strong, and merge in only the genuinely useful changes "
                f"attempted since."
            )
        best_sections = [
            f"# Previous best multi agent design (v{best_version})",
            note_text,
            best_design,
        ]
        if baseline_mode:
            improve_lead = (
                "Improve the multi-agent design to v{n} by combining the "
                f"strengths of the previous best design v{best_version} with "
                "the current design v{v}, with the explicit objective of "
                "producing deliverables that decisively OUTPERFORM the "
                "no-harness baseline (see section 3), not merely improving "
                "on v{v}.\n"
            )
        else:
            improve_lead = (
                "Improve the multi-agent design to v{n} by combining the strengths of the "
                f"previous best design v{best_version} with the current design v{{v}}, "
                "to enhance the quality of the query's deliverables by creating genuine "
                "multi-agent advantages over single-agent execution.\n"
            )
    elif baseline_mode:
        improve_lead = (
            "Improve the multi-agent design from v{v} to v{n} to enhance the "
            "quality of the query's deliverables by creating genuine "
            "multi-agent advantages over single-agent execution, with the "
            "explicit objective of producing deliverables that decisively "
            "OUTPERFORM the no-harness baseline (see section 3), not merely "
            "improving on v{v}.\n"
        )
    else:
        improve_lead = (
            "Improve the multi-agent design from v{v} to v{n} to enhance the quality of the query's deliverables by creating genuine multi-agent advantages over single-agent execution.\n"
        )

    v0_sections: list[str] = []
    if v0_design is not None:
        v0_sections = [
            "# Original multi agent design (v0 - the initial starting point)",
            (
                "NOTE: this is the original starting design, shown as the ROOT "
                "of the version track whose per-step changes appear in the "
                "pairwise history below. Use it to follow how the design "
                "evolved and to notice drift from the original intent. The "
                "current and best designs above supersede it - do not revert "
                "to it wholesale."
            ),
            v0_design,
        ]

    history_header = "3. Pairwise history summary"
    history_body = _history_summary_text(history_rows, design_diffs=design_diffs)
    if baseline_mode:
        history_header = "3. Pairwise history vs the no-harness baseline"
        framing_lines: list[str] = []
        if baseline_scoreboard:
            framing_lines.append(baseline_scoreboard)
            if "★" in baseline_scoreboard:
                framing_lines.append(
                    "★ = best design so far: among the versions that beat the "
                    "baseline, direct pairwise arbitration judged this one's "
                    "deliverables best."
                )
        framing_lines.append(
            "Each entry below compares that version's outputs against the "
            "FIXED no-harness baseline - the same task executed by a bare "
            "single agent with no multi-agent design. The diff attached to an "
            "entry shows what changed in the design before that verdict. Each "
            "verdict is a single judgment; treat trends across rounds as more "
            "informative than any single win/loss flip."
        )
        history_body = "\n".join(framing_lines) + "\n\n" + history_body

    user = "\n\n".join(
        [
            f"1. Query executed by {submission_label}",
            "--------------------------------",
            "# Task",
            template.task_text_for_eval,
            f"# Current multi agent design (v{current_version})",
            current_design,
            *best_sections,
            *v0_sections,
            "# Save rules",
            save_rules,
            "--------------------------------",
            f"2. Current submission code repo from {submission_label} (evidence only)",
            workspace_evidence,
            "2b. Execution metadata",
            execution_metadata or "- No execution metadata was supplied.",
            history_header,
            history_body,
            "3b. Pairwise history delta checklist (recurring issues to fix)",
            _history_delta_checklist_text(history_rows),
            "4. Instructions",
            (
                improve_lead +
                "Preserve the original intent and required deliverables, but address weaknesses revealed by evidence.\n"
                "Explicitly strengthen 'Output to orchestrator' contracts so downstream agents can consume structured outputs.\n"
                "Explicitly increase orchestrator-subagent feedback loops: allow orchestrator recall of previously called subagents "
                "with narrower follow-up scopes and updated acceptance criteria.\n"
                f"{caps_rule_user}"
                f"{size_rule_user}"
                "Most importantly, prioritize improvements that create genuine multi-agent advantages over single-agent execution "
                "(e.g., specialist parallelism, cross-agent validation, agent to agent communication, and conflict resolution loops, etc).\n"
                "Feedback quality requirements:\n"
                "- Be specific and evidence-grounded. Reference concrete files/metrics/history signals from the provided evidence.\n"
                "- Avoid generic claims (e.g., 'better coordination'); explain mechanism and expected effect.\n"
                "Do not change non-design parts of the query.\n"
                f"{output_contract}"
                "The `improved_multi_agent_design` should start with "
                f"'{DEFAULT_START_MARKER}' and contain the full block."
            ).format(v=current_version, n=current_version + 1),
        ]
    )
    return system, user


def _render_query_with_design(template: QueryTemplate, design_text: str) -> str:
    return f"{template.before_design}{design_text.strip()}\n\n{template.after_design.lstrip()}"


def _normalize_generated_design_block(
    design_text: str,
    *,
    start_marker: str,
    end_marker: str,
    after_design: str,
) -> str:
    """Normalize model output so only the design block body is kept.

    The model occasionally returns text that continues past the intended
    design block and includes the end marker or trailing prompt sections.
    Keep only the design block content (start marker .. before end marker).
    """
    text = design_text.strip()
    if not text.startswith(start_marker):
        raise RuntimeError(
            "LLM output missing required design block prefix. "
            f"Expected to start with: {start_marker!r}"
        )

    # Prefer trimming only if the model copied the exact post-design tail.
    after_prefix = after_design.lstrip()
    if after_prefix:
        probe = after_prefix[:200]
        tail_idx = text.find(probe)
        if tail_idx >= 0:
            text = text[:tail_idx].rstrip()
            return text

    # Fallback: trim on end_marker only if it appears near the end.
    # This avoids false truncation when the phrase appears naturally in
    # the middle of a generated instruction sentence.
    end_idx = text.find(end_marker)
    if end_idx >= 0 and end_idx >= int(0.7 * len(text)):
        text = text[:end_idx].rstrip()
    return text


def _fixed_agent_limit_matches(design_text: str) -> list[str]:
    """Find numerical caps that contradict the uncapped experiment policy."""
    patterns = [
        re.compile(
            r"\b(?:maximum|max(?:imum)?\s+of|at\s+most|up\s+to)\s+\d+\s+"
            r"(?:agent|subagent|recall|re-delegation|redelegation|hop|turn|tool|"
            r"iteration|round|minute|hour)s?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:agent|subagent|recall|re-delegation|redelegation|hop|turn|tool)"
            r"(?:\s+\w+){0,3}\s+(?:is|are)?\s*(?:limited|capped)\s+to\s+\d+\b",
            flags=re.IGNORECASE,
        ),
    ]
    matches: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(design_text):
            excerpt = " ".join(match.group(0).split())
            if excerpt not in matches:
                matches.append(excerpt)
    return matches


def _build_all_results_json_evidence(
    repo_root: Path,
    query_name: str,
    *,
    per_file_max_chars: int | None = None,
    total_max_chars: int | None = None,
) -> str:
    results_dir = repo_root / "results"
    if not results_dir.is_dir():
        return "- `results/` directory is missing."

    excluded_for_query202 = {
        # Large prediction blobs that frequently push prompts over context limits.
        "curve_data.json",
        "distances_angles_predictions.json",
        "distances_only_predictions.json",
        "run_B0_family_s0.json",
        "run_B0_family_s1.json",
        "run_B0_family_s2.json",
        "run_B0_random_s0.json",
        "run_B1_family_s0.json",
        "run_B1_family_s1.json",
        "run_B1_family_s2.json",
        "run_B1_random_s0.json",
        "run_M1_family_s0.json",
        "run_M1_family_s1.json",
        "run_M1_family_s2.json",
        "run_M1_random_s0.json",
        "run_M2_family_s0.json",
        "run_M2_family_s1.json",
        "run_M2_family_s2.json",
        "run_M2_random_s0.json",
        "run_M3_family_s0.json",
        "run_M3_family_s1.json",
        "run_M3_family_s2.json",
        "run_M3_random_s0.json",
    }
    excluded_for_query207 = {
        # Extremely large file; including full raw JSON can exceed model context window.
        "esmfold_plddts.json",
    }
    excluded_for_query220 = {
        # Large per-sample test result dumps that can dominate prompt token budget.
        "invariant_baseline_test_results.json",
        "se3_gvp_small_test_results.json",
        "se3_gvp_test_results.json",
    }
    excluded_for_query57 = {
        # Omit this JSON from prompt evidence for query57.
        "data_quality_issues.json",
    }
    excluded_globally = {
        # Large per-run raw record dumps that dominate prompt token budget.
        "raw_records.json",
        "pilot_raw_records.json",
    }
    excluded_by_query: dict[str, set[str]] = {
        "query57_ourTeam": excluded_for_query57,
        "query202_ourTeam": excluded_for_query202,
        "query207_ourTeam": excluded_for_query207,
        "query220_ourTeam": excluded_for_query220,
    }

    json_paths = sorted(p for p in results_dir.rglob("*.json") if p.is_file())
    excluded_names = excluded_globally | excluded_by_query.get(query_name, set())
    if excluded_names:
        json_paths = [p for p in json_paths if p.name not in excluded_names]
    if not json_paths:
        return "- No JSON files found under `results/`."

    blocks: list[str] = []
    total_chars = 0
    omitted: list[str] = []
    for path in json_paths:
        rel = path.relative_to(repo_root)
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            body = f"<read error: {exc}>"
        truncated = False
        if per_file_max_chars is not None and len(body) > per_file_max_chars:
            half = per_file_max_chars // 2
            body = (
                body[:half]
                + "\n... [truncated: middle omitted] ...\n"
                + body[len(body) - (per_file_max_chars - half) :]
            )
            truncated = True
        if total_max_chars is not None and total_chars + len(body) > total_max_chars:
            omitted.append(str(rel))
            continue
        total_chars += len(body)
        flag = " [truncated]" if truncated else ""
        blocks.extend(
            [
                f"### File: `{rel}`{flag}",
                "```",
                body,
                "```",
            ]
        )
    if omitted:
        blocks.append(
            "- Omitted due to appendix character budget: "
            + ", ".join(f"`{name}`" for name in omitted)
        )
    return "\n\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Iterative multi-agent design prompt improver")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Complete the final comparison and champion arbitration without proposing another workflow.")
    parser.add_argument("--query-file", type=Path, required=True, help="Path to query*_ourTeam.txt")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Root directory containing query*_ourTeam and query*_ourTeam-v* output repos",
    )
    parser.add_argument(
        "--current-repo",
        type=Path,
        default=None,
        help=(
            "Explicit normalized submission root for the current version. "
            "When provided, it takes precedence over --runs-root resolution."
        ),
    )
    parser.add_argument(
        "--previous-repo",
        type=Path,
        default=None,
        help=(
            "Explicit normalized submission root for the previous version's pairwise "
            "comparison. Used only when --current-version >= 1."
        ),
    )
    parser.add_argument(
        "--current-run-status-file",
        type=Path,
        default=None,
        help="Optional run_status.json included as provenance in the improvement prompt.",
    )
    parser.add_argument(
        "--prev-runs-root",
        type=Path,
        default=None,
        help=(
            "Optional previous-version runs root for comparison step. "
            "If omitted and --runs-root ends with _vN, the script tries to infer _v(N-1)."
        ),
    )
    parser.add_argument(
        "--current-version",
        type=int,
        required=True,
        help="Current available version index. 0 means base query*_ourTeam repo exists.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root for this harness. Default: ./harness_improvement/runs/<query_name>",
    )
    parser.add_argument(
        "--llm-backend",
        choices=BACKEND_CHOICES,
        default=BACKEND_OPENAI_RESPONSES,
        help=(
            "Backend used for both pairwise evaluation and harness optimization. "
            "External GPT is never selected implicitly when local-vllm is requested."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID. Defaults to gpt-5.5 externally or Qwen/Qwen3.5-9B locally.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL for --llm-backend local-vllm.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help=(
            "Environment variable containing the API key. Defaults to OPENAI_API_KEY "
            "externally; local vLLM uses VLLM_API_KEY when set, otherwise EMPTY."
        ),
    )
    parser.add_argument("--reasoning-effort", type=str, default="xhigh", help="Reasoning effort for OpenAI Responses API")
    parser.add_argument(
        "--openai-background",
        action="store_true",
        help=(
            "Submit OpenAI Responses calls with background=True and poll for "
            "completion (short requests each poll). Use when long non-streaming "
            "calls die on NAT/conntrack idle timeouts (observed on GCP hosts). "
            "openai-responses backend only; default off (identical call path)."
        ),
    )
    parser.add_argument(
        "--judge-llm-backend",
        choices=list(BACKEND_CHOICES),
        default=None,
        help=(
            "Optional separate backend for the in-loop pairwise judge ONLY. "
            "Default (unset): the judge shares the main --llm-backend backend "
            "object unchanged (byte-identical call path). When set, a second "
            "backend is built for judge calls (e.g. an external gpt-5.5 judge "
            "steering a local-model optimizer). The judge backend does NOT "
            "inherit the main sampling knobs: it uses --judge-model / "
            "--judge-reasoning-effort with temperature omitted (claude-line "
            "judge parity) and honors --openai-background."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "Model for --judge-llm-backend. Defaults to gpt-5.5 for "
            "openai-responses; required in spirit for local-vllm."
        ),
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        type=str,
        default=None,
        help="Reasoning effort for the judge backend (e.g. xhigh). Omitted when unset.",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="Base URL for --judge-llm-backend local-vllm (required in that case).",
    )
    parser.add_argument(
        "--include-best-design",
        action="store_true",
        help=(
            "BEST-H anchoring: track the best-so-far design (champion) from the "
            "in-loop pairwise verdicts and, when the champion is OLDER than the "
            "current design, show it to the optimizer alongside the current "
            "design with adapted instructions. When the current version wins its "
            "adjacent comparison while the champion is older, ONE extra judge "
            "call (champion vs current) arbitrates; those rows live in a "
            "separate records/champion_history.jsonl ledger so the standard "
            "history pipeline is untouched. Default off (byte-identical prompts)."
        ),
    )
    parser.add_argument(
        "--champion-repo-root-pattern",
        type=str,
        default=None,
        help=(
            "Runs-root pattern for champion arbitration repos, with a "
            "'{version}' placeholder (e.g. 'runs/L0/workspaces/v{version}'). "
            "Required with --include-best-design at --current-version >= 1."
        ),
    )
    parser.add_argument(
        "--champion-repo-v0-root",
        type=Path,
        default=None,
        help=(
            "Runs root holding the v0 repos for champion arbitration (the fork "
            "point may live outside the pattern's namespace). Required with "
            "--include-best-design at --current-version >= 1."
        ),
    )
    parser.add_argument(
        "--history-design-diffs",
        action="store_true",
        help=(
            "DIFF-HISTORY: render each adjacent pairwise-history entry with a "
            "compact unified diff of the design blocks behind it, so the "
            "optimizer sees WHAT change produced WHAT judgment (credit "
            "assignment). Deterministic and API-free (reads versions/*.txt); "
            "recent pairs get fuller diffs than old ones. Default off "
            "(byte-identical history sections)."
        ),
    )
    parser.add_argument(
        "--judge-vs-baseline",
        action="store_true",
        help=(
            "BASELINE-ANCHORED JUDGING: replace the adjacent v{k-1}-vs-v{k} "
            "in-loop comparison with v{k} vs the FIXED no-harness baseline "
            "repo (plus a one-time v0-vs-baseline seed at the first judged "
            "step). History entries, scoreboard, champion rule, and the "
            "improvement instructions all become baseline-anchored: the "
            "stated objective is to decisively outperform the no-harness "
            "baseline, not merely the previous round. Champion advances only "
            "on direct wins (a baseline win when no champion has one, else "
            "one arbitration call). Incompatible with "
            "--strict-adjacent-history-provenance. Default off."
        ),
    )
    parser.add_argument(
        "--baseline-repo-root",
        type=Path,
        default=None,
        help=(
            "Root holding the no-harness baseline workspaces, resolved as "
            "<root>/query{N} (no _ourTeam suffix). Required with "
            "--judge-vs-baseline at --current-version >= 1."
        ),
    )
    parser.add_argument(
        "--include-v0-design",
        action="store_true",
        help=(
            "Show the ORIGINAL design v0 (the human-written starting point) in "
            "the improvement prompt as the root of the version track, so the "
            "history diffs are groundable and drift from the original intent "
            "is visible. Skipped at step 0 (v0 IS the current design) and when "
            "v0 is already shown as the champion. Default off."
        ),
    )
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature; omitted by default")
    parser.add_argument("--omit-temperature", action="store_true", help="Force omitting temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Optional local-vLLM top-p.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional local-vLLM top-k.")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens. The Qwen local recipe uses 32768.",
    )
    parser.add_argument(
        "--local-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Qwen thinking through chat_template_kwargs (local-vllm only).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional local-vLLM sampling seed.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=3600.0,
        help="Per-call API timeout.",
    )
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=None,
        help="Optional context ceiling used for a conservative pre-call guard.",
    )
    parser.add_argument(
        "--context-safety-tokens",
        type=int,
        default=8192,
        help="Reserved context headroom for the conservative local context guard.",
    )
    parser.add_argument(
        "--submission-label",
        type=str,
        default="the coding agent",
        help="Neutral provenance label used inside evaluator/optimizer prompts.",
    )
    parser.add_argument(
        "--compact-era-prompt",
        action="store_true",
        help=(
            "Byte-exact restoration of the compact-lineage optimizer prompt: "
            "emits the two generic keep-it-compact sentences and the two "
            "no-numeric-caps sentences WITHOUT enabling the fixed-agent-limit "
            "output validator. For continuing the archived compact lineage "
            "only; never set for the free lineage."
        ),
    )
    parser.add_argument(
        "--results-json-max-chars",
        type=int,
        default=None,
        help=(
            "Per-file character cap for the full results/*.json appendix "
            "(head+tail kept, middle omitted). Default: no cap, matching the "
            "original claude-era behavior."
        ),
    )
    parser.add_argument(
        "--results-json-total-max-chars",
        type=int,
        default=None,
        help=(
            "Total character budget for the results/*.json appendix; files "
            "beyond it are listed as omitted. Default: no cap."
        ),
    )
    parser.add_argument(
        "--omit-full-results-json",
        action="store_true",
        help=(
            "Do not append every results/*.json file after the already capped evidence "
            "bundle. Recommended for local models to avoid duplicate evidence."
        ),
    )
    parser.add_argument(
        "--forbid-fixed-agent-limits",
        action="store_true",
        help=(
            "Reject generated designs that numerically cap agents, subagents, recalls, "
            "hops, turns, tools, or duration."
        ),
    )
    parser.add_argument(
        "--max-design-chars",
        type=int,
        default=None,
        help="Reject a generated design longer than this many characters.",
    )
    parser.add_argument(
        "--target-design-chars",
        type=int,
        default=None,
        help=(
            "Optional soft size target included in the optimizer prompt. Requires "
            "--max-design-chars and must not exceed it."
        ),
    )
    parser.add_argument(
        "--improvement-output-mode",
        choices=("json", "design-text"),
        default="json",
        help=(
            "Optimizer response format. design-text is a recorded fallback for local "
            "models that truncate while JSON-escaping a long harness."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and save prompt artifacts, enforce the context guard, but make no API call.",
    )
    parser.add_argument(
        "--require-local-vllm",
        action="store_true",
        help=(
            "Refuse every non-local backend or non-loopback base URL. Use this for "
            "self-RHI trajectories that must not contact an external API."
        ),
    )
    parser.add_argument(
        "--strict-adjacent-history-provenance",
        action="store_true",
        help=(
            "For version 1 and later, require exactly one adjacent pairwise-history "
            "row matching this backend, model, base URL, and normalized repo pair."
        ),
    )
    parser.add_argument("--start-marker", type=str, default=DEFAULT_START_MARKER)
    parser.add_argument("--end-marker", type=str, default=DEFAULT_END_MARKER)
    parser.add_argument("--report-max-tokens", type=int, default=None)
    parser.add_argument("--json-max-tokens", type=int, default=50000)
    parser.add_argument("--code-max-tokens", type=int, default=10000)
    args = parser.parse_args()

    query_file = args.query_file.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve() if args.runs_root else None
    if not query_file.is_file():
        raise SystemExit(f"query file not found: {query_file}")
    if runs_root is not None and not runs_root.is_dir():
        raise SystemExit(f"runs root not found: {runs_root}")
    if runs_root is None and args.current_repo is None:
        raise SystemExit("Provide either --runs-root or --current-repo.")
    if args.current_version < 0:
        raise SystemExit("--current-version must be >= 0")
    if args.context_safety_tokens < 0:
        raise SystemExit("--context-safety-tokens must be >= 0")
    if args.require_local_vllm:
        if args.llm_backend != BACKEND_LOCAL_VLLM:
            raise SystemExit(
                "--require-local-vllm refuses every backend except local-vllm"
            )
        if not _is_loopback_base_url(args.base_url):
            raise SystemExit(
                "--require-local-vllm requires a loopback base URL; "
                f"received {args.base_url!r}"
            )
        if args.judge_llm_backend and args.judge_llm_backend != BACKEND_LOCAL_VLLM:
            raise SystemExit(
                "--require-local-vllm refuses a non-local --judge-llm-backend"
            )
        if args.judge_llm_backend == BACKEND_LOCAL_VLLM and not _is_loopback_base_url(
            args.judge_base_url
        ):
            raise SystemExit(
                "--require-local-vllm requires a loopback --judge-base-url; "
                f"received {args.judge_base_url!r}"
            )
    if args.strict_adjacent_history_provenance and args.current_version < 1:
        raise SystemExit(
            "--strict-adjacent-history-provenance requires --current-version >= 1"
        )
    if args.judge_vs_baseline and args.strict_adjacent_history_provenance:
        raise SystemExit(
            "--judge-vs-baseline is incompatible with "
            "--strict-adjacent-history-provenance (no adjacent rows exist)"
        )
    if args.judge_vs_baseline and args.current_version >= 1:
        if args.baseline_repo_root is None:
            raise SystemExit(
                "--judge-vs-baseline at --current-version >= 1 requires "
                "--baseline-repo-root"
            )
        if args.champion_repo_v0_root is None:
            raise SystemExit(
                "--judge-vs-baseline at --current-version >= 1 requires "
                "--champion-repo-v0-root (for the one-time v0 seed judging)"
            )
    if args.include_best_design and args.current_version >= 1:
        if not args.champion_repo_root_pattern or "{version}" not in args.champion_repo_root_pattern:
            raise SystemExit(
                "--include-best-design at --current-version >= 1 requires "
                "--champion-repo-root-pattern containing '{version}'"
            )
        if args.champion_repo_v0_root is None:
            raise SystemExit(
                "--include-best-design at --current-version >= 1 requires "
                "--champion-repo-v0-root"
            )

    model = args.model or (
        "Qwen/Qwen3.5-9B"
        if args.llm_backend == BACKEND_LOCAL_VLLM
        else "gpt-5.5"
    )

    template = _read_query_template(
        query_file,
        start_marker=args.start_marker,
        end_marker=args.end_marker,
    )

    if args.out_dir is None:
        assert runs_root is not None
        out_dir = (
            query_file.parent.parent.parent
            / "harness_improvement"
            / "runs"
            / runs_root.name
            / template.query_name
        ).resolve()
    else:
        out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    versions_dir = out_dir / "versions"
    rendered_queries_dir = out_dir / "updated_queries"
    prompts_dir = out_dir / "evaluation_prompts"
    comparisons_dir = out_dir / "comparisons"
    records_dir = out_dir / "records"
    for d in (versions_dir, rendered_queries_dir, prompts_dir, comparisons_dir, records_dir):
        d.mkdir(parents=True, exist_ok=True)

    v0_path = versions_dir / "multi_agent_design_v0.txt"
    if not v0_path.is_file():
        v0_path.write_text(template.design_block.strip() + "\n", encoding="utf-8")

    current_design_path = versions_dir / f"multi_agent_design_v{args.current_version}.txt"
    if not current_design_path.is_file() and args.current_version > 0:
        # Recovery path: if user has rendered/edited query for this version, bootstrap design from it.
        candidate_query = rendered_queries_dir / f"{template.query_name}-v{args.current_version}.txt"
        if candidate_query.is_file():
            recovered = _read_query_template(
                candidate_query,
                start_marker=args.start_marker,
                end_marker=args.end_marker,
            ).design_block
            current_design_path.write_text(recovered.strip() + "\n", encoding="utf-8")
    if not current_design_path.is_file():
        raise SystemExit(
            f"Missing current design file: {current_design_path}. "
            f"Expected existing history up to v{args.current_version}."
        )
    current_design = current_design_path.read_text(encoding="utf-8").strip()

    current_repo = (
        args.current_repo.expanduser().resolve()
        if args.current_repo is not None
        else _resolve_iteration_repo(runs_root, template.query_name, args.current_version)
    )
    if not current_repo.is_dir():
        raise SystemExit(f"Current iteration repo not found: {current_repo}")

    history_path = records_dir / "comparison_history.jsonl"
    history_rows = _load_history(history_path)

    temperature_kw: float | None = None if args.omit_temperature else args.temperature
    run_cost_estimates: list[float] = []
    run_api_calls_total = 0

    api_key_env = args.api_key_env or (
        "VLLM_API_KEY"
        if args.llm_backend == BACKEND_LOCAL_VLLM
        else "OPENAI_API_KEY"
    )
    api_key = os.getenv(api_key_env)
    if args.llm_backend == BACKEND_LOCAL_VLLM:
        api_key = api_key or "EMPTY"
    elif not api_key and not args.dry_run:
        raise SystemExit(f"{api_key_env} is not set.")

    backend: JsonLLMBackend | None = None
    if not args.dry_run:
        assert api_key is not None
        backend = JsonLLMBackend(
            backend=args.llm_backend,
            model=model,
            api_key=api_key,
            base_url=args.base_url if args.llm_backend == BACKEND_LOCAL_VLLM else None,
            reasoning_effort=args.reasoning_effort,
            temperature=temperature_kw,
            top_p=args.top_p,
            top_k=args.top_k,
            max_output_tokens=args.max_output_tokens,
            enable_thinking=args.local_thinking,
            seed=args.seed,
            timeout_seconds=args.timeout_seconds,
            background=args.openai_background,
        )
        backend.verify_model()

    # In-loop judge backend. Default (no --judge-llm-backend): the judge IS
    # the main backend object — identical behavior to every earlier run. With
    # the override, judge calls go to a second backend that deliberately does
    # NOT inherit the main sampling knobs (claude-line judge parity: model +
    # reasoning effort only, temperature omitted).
    judge_backend = backend
    judge_backend_name = args.llm_backend
    judge_model = model
    judge_reasoning_effort = args.reasoning_effort
    judge_temperature = temperature_kw
    judge_base_url = args.base_url if args.llm_backend == BACKEND_LOCAL_VLLM else None
    if args.judge_llm_backend:
        judge_backend_name = args.judge_llm_backend
        judge_model = args.judge_model or (
            "Qwen/Qwen3.5-9B"
            if args.judge_llm_backend == BACKEND_LOCAL_VLLM
            else "gpt-5.5"
        )
        judge_reasoning_effort = args.judge_reasoning_effort
        judge_temperature = None
        judge_base_url = (
            args.judge_base_url
            if args.judge_llm_backend == BACKEND_LOCAL_VLLM
            else None
        )
        if args.judge_llm_backend == BACKEND_LOCAL_VLLM and not args.judge_base_url:
            raise SystemExit("--judge-llm-backend local-vllm requires --judge-base-url")
        judge_api_key_env = (
            "VLLM_API_KEY"
            if args.judge_llm_backend == BACKEND_LOCAL_VLLM
            else "OPENAI_API_KEY"
        )
        judge_api_key = os.getenv(judge_api_key_env)
        if args.judge_llm_backend == BACKEND_LOCAL_VLLM:
            judge_api_key = judge_api_key or "EMPTY"
        elif not judge_api_key and not args.dry_run:
            raise SystemExit(
                f"{judge_api_key_env} is not set (required for --judge-llm-backend)."
            )
        if not args.dry_run:
            assert judge_api_key is not None
            judge_backend = JsonLLMBackend(
                backend=args.judge_llm_backend,
                model=judge_model,
                api_key=judge_api_key,
                base_url=judge_base_url,
                reasoning_effort=judge_reasoning_effort,
                temperature=None,
                timeout_seconds=args.timeout_seconds,
                background=args.openai_background,
            )
            judge_backend.verify_model()

    # Resolve comparison roots:
    # - current repo always comes from --runs-root
    # - previous repo can come from --prev-runs-root, inferred sibling _v(prev), or fallback --runs-root
    prev_runs_root: Path | None = None
    if args.current_version >= 1:
        prev_version = args.current_version - 1
        if args.previous_repo is not None:
            prev_runs_root = None
        elif args.prev_runs_root is not None:
            prev_runs_root = args.prev_runs_root.expanduser().resolve()
        else:
            assert runs_root is not None
            prev_runs_root = _infer_versioned_runs_root(runs_root, prev_version)
            if prev_runs_root is None:
                prev_runs_root = runs_root

        if prev_runs_root is not None and not prev_runs_root.is_dir():
            raise SystemExit(f"prev runs root not found: {prev_runs_root}")

    # Step A: pairwise comparison for v(i-1) vs v(i) if current_version >= 1 and not already recorded.
    # BASELINE-ANCHORED judge phase (--judge-vs-baseline): replaces the
    # adjacent comparison below. Judges the current version (and, once per
    # query ever, v0 as a seed) against the FIXED no-harness baseline repo.
    if args.current_version >= 1 and args.judge_vs_baseline:
        baseline_repo = (
            args.baseline_repo_root.expanduser().resolve()
            / f"query{template.query_num}"
        )
        if not baseline_repo.is_dir():
            raise SystemExit(f"Baseline repo not found: {baseline_repo}")

        def _judge_version_vs_baseline(version: int, version_repo: Path) -> None:
            key = f"v{version}_vs_no_harness"
            matching = [r for r in history_rows if r.get("key") == key]
            comp_dir = comparisons_dir / key
            result_path = comp_dir / "result.json"
            if matching:
                _write_json_once(result_path, matching[0])
                return
            if result_path.exists():
                raise SystemExit(
                    "Found an orphan baseline comparison result without its "
                    f"history row; refusing another judge call: {result_path}"
                )
            if not version_repo.is_dir():
                raise SystemExit(
                    f"Repo for baseline judging not found: {version_repo}"
                )
            task_row = {
                "query_id": f"query_{template.query_num}",
                "query": template.task_text_for_eval,
            }
            bundle_a = build_judge_bundle(
                memory_root=baseline_repo,
                task_row=task_row,
                log_path=None,
                scan=scan_deliverables(baseline_repo, template.task_text_for_eval),
                report_max_tokens=args.report_max_tokens,
                json_max_tokens=args.json_max_tokens,
                code_max_tokens=args.code_max_tokens,
                token_count_model=model,
            )
            bundle_b = build_judge_bundle(
                memory_root=version_repo,
                task_row=task_row,
                log_path=None,
                scan=scan_deliverables(version_repo, template.task_text_for_eval),
                report_max_tokens=args.report_max_tokens,
                json_max_tokens=args.json_max_tokens,
                code_max_tokens=args.code_max_tokens,
                token_count_model=model,
            )
            system_prompt = pairwise_judge_system_for_query_num(template.query_num)
            user_prompt = build_pairwise_user_message(
                bundle_a=bundle_a,
                bundle_b=bundle_b,
                label_a="no-harness-single-agent-baseline",
                label_b=f"multi-agent-design-v{version}",
            )
            _write_full_prompt_dump(
                comp_dir / "full_prompt.txt",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            _enforce_context_budget(
                stage="baseline judge",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=args.max_output_tokens,
                context_window_tokens=args.context_window_tokens,
                context_safety_tokens=args.context_safety_tokens,
            )
            if args.dry_run:
                return
            assert judge_backend is not None
            judge_call = judge_backend.call_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            verdict = PairwiseJudgeResult.model_validate(judge_call.payload)
            nonlocal_counters["api_calls"] += 1
            record = {
                "key": key,
                "comparison": (
                    "no-harness-single-agent-baseline vs "
                    f"multi-agent-design-v{version}"
                ),
                "version_b": version,
                "baseline": "no_harness",
                "repo_a": str(baseline_repo),
                "repo_b": str(version_repo),
                "winner": verdict.winner,
                "rationale": verdict.rationale,
                "judged_at_utc": _utc_now_iso(),
                "llm_backend": judge_backend_name,
                "model": judge_model,
                "base_url": judge_base_url,
                "reasoning_effort": judge_reasoning_effort,
                "temperature": judge_temperature,
                "call_metrics": judge_call.metrics,
            }
            cost = judge_call.metrics.get("estimated_cost_usd")
            if isinstance(cost, (int, float)):
                run_cost_estimates.append(float(cost))
            _append_history_row(history_path, record)
            _write_json_once(result_path, record)
            history_rows.append(record)

        nonlocal_counters = {"api_calls": 0}
        seed_repo = (
            args.champion_repo_v0_root.expanduser().resolve()
            / template.query_name
        )
        _judge_version_vs_baseline(0, seed_repo)
        _judge_version_vs_baseline(args.current_version, current_repo)
        run_api_calls_total += nonlocal_counters["api_calls"]

    if args.current_version >= 1 and not args.judge_vs_baseline:
        prev_version = args.current_version - 1
        comparison_key = _history_key(prev_version, args.current_version)
        prev_repo = (
            args.previous_repo.expanduser().resolve()
            if args.previous_repo is not None
            else _resolve_iteration_repo(
                prev_runs_root, template.query_name, prev_version
            )
        )
        if not prev_repo.is_dir():
            raise SystemExit(
                "Previous iteration repo required for adjacent self-comparison "
                f"was not found: {prev_repo}"
            )

        matching_history = [
            row for row in history_rows if row.get("key") == comparison_key
        ]
        if len(matching_history) > 1 and args.strict_adjacent_history_provenance:
            raise SystemExit(
                f"Expected exactly one {comparison_key} history row, found "
                f"{len(matching_history)} in {history_path}"
            )
        if matching_history and args.strict_adjacent_history_provenance:
            provenance_errors = _adjacent_history_provenance_errors(
                matching_history[0],
                comparison_key=comparison_key,
                prev_version=prev_version,
                current_version=args.current_version,
                prev_repo=prev_repo,
                current_repo=current_repo,
                llm_backend=judge_backend_name,
                model=judge_model,
                base_url=judge_base_url,
            )
            if provenance_errors:
                details = "\n- ".join(provenance_errors)
                raise SystemExit(
                    "Existing adjacent history provenance check failed:\n"
                    f"- {details}"
                )

        comp_dir = comparisons_dir / comparison_key
        comparison_result_path = comp_dir / "result.json"
        if matching_history:
            # Recover the immutable per-comparison result if a process stopped
            # after appending history but before mirroring that row to result.json.
            _write_json_once(comparison_result_path, matching_history[0])
        elif comparison_result_path.exists():
            raise SystemExit(
                "Found an orphan comparison result without its history row; "
                "refusing another judge call until provenance is repaired: "
                f"{comparison_result_path}"
            )

        if not matching_history:
            task_row = {
                "query_id": f"query_{template.query_num}",
                "query": template.task_text_for_eval,
            }
            scan_a = scan_deliverables(prev_repo, template.task_text_for_eval)
            scan_b = scan_deliverables(current_repo, template.task_text_for_eval)
            bundle_a = build_judge_bundle(
                memory_root=prev_repo,
                task_row=task_row,
                log_path=None,
                scan=scan_a,
                report_max_tokens=args.report_max_tokens,
                json_max_tokens=args.json_max_tokens,
                code_max_tokens=args.code_max_tokens,
                token_count_model=model,
            )
            bundle_b = build_judge_bundle(
                memory_root=current_repo,
                task_row=task_row,
                log_path=None,
                scan=scan_b,
                report_max_tokens=args.report_max_tokens,
                json_max_tokens=args.json_max_tokens,
                code_max_tokens=args.code_max_tokens,
                token_count_model=model,
            )
            label_a = f"multi-agent-design-v{prev_version}"
            label_b = f"multi-agent-design-v{args.current_version}"
            system_prompt = pairwise_judge_system_for_query_num(template.query_num)
            user_prompt = build_pairwise_user_message(
                bundle_a=bundle_a,
                bundle_b=bundle_b,
                label_a=label_a,
                label_b=label_b,
            )
            _write_full_prompt_dump(
                comp_dir / "full_prompt.txt",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            _enforce_context_budget(
                stage="pairwise judge",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=args.max_output_tokens,
                context_window_tokens=args.context_window_tokens,
                context_safety_tokens=args.context_safety_tokens,
            )
            if not args.dry_run:
                assert judge_backend is not None
                judge_call = judge_backend.call_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                verdict = PairwiseJudgeResult.model_validate(judge_call.payload)
                run_api_calls_total += 1
                record = {
                    "key": comparison_key,
                    "comparison": f"multi-agent-design-v{prev_version} vs multi-agent-design-v{args.current_version}",
                    "version_a": prev_version,
                    "version_b": args.current_version,
                    "repo_a": str(prev_repo),
                    "repo_b": str(current_repo),
                    "winner": verdict.winner,
                    "rationale": verdict.rationale,
                    "judged_at_utc": _utc_now_iso(),
                    "llm_backend": judge_backend_name,
                    "model": judge_model,
                    "base_url": judge_base_url,
                    "reasoning_effort": judge_reasoning_effort,
                    "temperature": judge_temperature,
                    "call_metrics": judge_call.metrics,
                }
                estimated_cost = judge_call.metrics.get("estimated_cost_usd")
                if isinstance(estimated_cost, (int, float)):
                    run_cost_estimates.append(float(estimated_cost))
                _append_history_row(history_path, record)
                _write_json_once(comparison_result_path, record)
                history_rows.append(record)
                matching_history = [record]

        if args.strict_adjacent_history_provenance and not args.dry_run:
            if len(matching_history) != 1:
                raise SystemExit(
                    f"Expected exactly one {comparison_key} history row after "
                    f"self-comparison, found {len(matching_history)}"
                )
            provenance_errors = _adjacent_history_provenance_errors(
                matching_history[0],
                comparison_key=comparison_key,
                prev_version=prev_version,
                current_version=args.current_version,
                prev_repo=prev_repo,
                current_repo=current_repo,
                llm_backend=judge_backend_name,
                model=judge_model,
                base_url=judge_base_url,
            )
            if provenance_errors:
                details = "\n- ".join(provenance_errors)
                raise SystemExit(
                    "Adjacent history provenance check failed:\n"
                    f"- {details}"
                )

    # Step B: single-submission evaluation prompt to improve v(i-1) -> v(i)
    task_row_current = {
        "query_id": f"query_{template.query_num}",
        "query": template.task_text_for_eval,
    }
    current_scan = scan_deliverables(current_repo, template.task_text_for_eval)
    current_bundle = build_judge_bundle(
        memory_root=current_repo,
        task_row=task_row_current,
        log_path=None,
        scan=current_scan,
        report_max_tokens=args.report_max_tokens,
        json_max_tokens=args.json_max_tokens,
        code_max_tokens=args.code_max_tokens,
        token_count_model=model,
    )
    workspace_evidence = bundle_workspace_evidence(
        current_bundle,
        workspace_section_title=f"# Workspace for current iteration (multi-agent-design-v{args.current_version})",
    )
    if not args.omit_full_results_json:
        results_json_evidence = _build_all_results_json_evidence(
            current_repo,
            template.query_name,
            per_file_max_chars=args.results_json_max_chars,
            total_max_chars=args.results_json_total_max_chars,
        )
        workspace_evidence = (
            f"{workspace_evidence}\n\n"
            "# Full JSON artifacts under results/\n"
            f"{results_json_evidence}"
        )

    execution_metadata: str | None = None
    current_run_status_file: Path | None = None
    if args.current_run_status_file is not None:
        current_run_status_file = args.current_run_status_file.expanduser().resolve()
        if not current_run_status_file.is_file():
            raise SystemExit(
                f"current run status file not found: {current_run_status_file}"
            )
        try:
            metadata_obj = json.loads(
                current_run_status_file.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid JSON in current run status file: {current_run_status_file}: {exc}"
            ) from exc
        execution_metadata = json.dumps(metadata_obj, indent=2, ensure_ascii=False)
        run_failed = isinstance(metadata_obj, dict) and (
            metadata_obj.get("exit_code") not in (0, None)
            or metadata_obj.get("trace_has_success_result") is False
        )
        if run_failed:
            execution_metadata += (
                "\n\n"
                "FAILURE NOTICE: this submission's execution ended in a force quit "
                "before a successful final result (see `exit_code` and "
                "`trace_has_success_result` above; when present, `failure.message` "
                "records the halt reason and "
                "`failure.last_tool_calls_most_recent_first` shows what the agent "
                "was doing when it was halted). Treat this failure as first-order "
                "evidence and pursue BOTH goals together when revising the design:\n"
                "(1) Make the next run complete without a force quit, targeting "
                "the specific stall pattern shown in `failure`: tighten each "
                "subagent's 'Output to orchestrator' contract with explicit "
                "completion and acceptance criteria so agents return a definitive "
                "result instead of repeating equivalent tool calls; add "
                "orchestrator progress checks with recovery/re-delegation hops "
                "that redirect a stalled subagent to a narrower follow-up scope; "
                "state in shared protocols how agents must detect and break "
                "repeated tool-call patterns (e.g., after re-reading the same file "
                "or re-running the same command without new information, change "
                "strategy or report the blocker to the orchestrator).\n"
                "(2) Still improve the agentic workflow (hops, contracts) and "
                "agent design (roles, instructions) for deliverable quality as "
                "instructed below.\n"
                "Do not achieve (1) with numerical caps on agents, subagents, "
                "hops, tool calls, turns, or duration, and do not collapse the "
                "design toward single-agent execution; resolve stalls through "
                "acceptance gates, explicit fallback instructions, and "
                "re-delegation."
            )
            if metadata_obj.get("prior_failures"):
                execution_metadata += (
                    "\nRECURRENT FAILURE: `prior_failures` above shows that "
                    "earlier harness version(s) of this task also ended in a "
                    "force quit. Countermeasures written into the current "
                    "design did not change execution behavior. Compare the "
                    "failure fingerprints across versions to diagnose why, and "
                    "revise the approach materially — restructure the workflow "
                    "so the stall pattern cannot arise (e.g., size "
                    "long-running operations to complete within a single tool "
                    "call, or require an explicit strategy change after any "
                    "repeated result) — rather than restating similar "
                    "anti-repetition rules."
                )
        elif isinstance(metadata_obj, dict) and metadata_obj.get("prior_failures"):
            execution_metadata += (
                "\n\n"
                "RESOLVED FAILURE HISTORY: `prior_failures` above shows that "
                "earlier harness version(s) of this task ended in a force "
                "quit, while the current version completed successfully. "
                "Compare the current design with the prior failure "
                "fingerprints to identify which provisions plausibly fixed "
                "the failure, and PRESERVE them in the revised design — do "
                "not remove or weaken anti-stall protections that are now "
                "working. Otherwise continue improving the agentic workflow "
                "(hops, contracts) and agent design (roles, instructions) "
                "for deliverable quality as instructed below."
            )

    # BEST-H champion resolution (default off). Champion rows live in a
    # separate ledger so the standard adjacent-history pipeline is untouched.
    best_design_text: str | None = None
    best_design_version: int | None = None
    best_design_rationale: str | None = None
    best_baseline_case: str | None = None
    champ_has_baseline_win = True
    scoreboard_champion: int | None = None
    if args.include_best_design and args.current_version >= 1:
        champion_path = records_dir / "champion_history.jsonl"
        champion_rows = _load_history(champion_path)
        if args.judge_vs_baseline:
            champ, champ_has_baseline_win, needs_arbitration = (
                _derive_champion_baseline(
                    history_rows, champion_rows, args.current_version
                )
            )
        else:
            champ, needs_arbitration = _derive_champion(
                history_rows, champion_rows, args.current_version
            )
        if needs_arbitration and not args.dry_run:
            assert judge_backend is not None
            if champ == 0:
                champ_repo = (
                    args.champion_repo_v0_root.expanduser().resolve()
                    / template.query_name
                )
            else:
                champ_repo = Path(
                    args.champion_repo_root_pattern.format(version=champ)
                ).expanduser().resolve() / template.query_name
            if not champ_repo.is_dir():
                raise SystemExit(
                    f"Champion arbitration repo not found: {champ_repo}"
                )
            arb_key = f"champion_v{champ}_vs_v{args.current_version}"
            arb_task_row = {
                "query_id": f"query_{template.query_num}",
                "query": template.task_text_for_eval,
            }
            arb_bundle_a = build_judge_bundle(
                memory_root=champ_repo,
                task_row=arb_task_row,
                log_path=None,
                scan=scan_deliverables(champ_repo, template.task_text_for_eval),
                report_max_tokens=args.report_max_tokens,
                json_max_tokens=args.json_max_tokens,
                code_max_tokens=args.code_max_tokens,
                token_count_model=model,
            )
            arb_bundle_b = build_judge_bundle(
                memory_root=current_repo,
                task_row=arb_task_row,
                log_path=None,
                scan=scan_deliverables(current_repo, template.task_text_for_eval),
                report_max_tokens=args.report_max_tokens,
                json_max_tokens=args.json_max_tokens,
                code_max_tokens=args.code_max_tokens,
                token_count_model=model,
            )
            arb_system = pairwise_judge_system_for_query_num(template.query_num)
            arb_user = build_pairwise_user_message(
                bundle_a=arb_bundle_a,
                bundle_b=arb_bundle_b,
                label_a=f"multi-agent-design-v{champ}",
                label_b=f"multi-agent-design-v{args.current_version}",
            )
            arb_dir = comparisons_dir / arb_key
            _write_full_prompt_dump(
                arb_dir / "full_prompt.txt",
                system_prompt=arb_system,
                user_prompt=arb_user,
            )
            arb_call = judge_backend.call_json(
                system_prompt=arb_system, user_prompt=arb_user
            )
            arb_verdict = PairwiseJudgeResult.model_validate(arb_call.payload)
            run_api_calls_total += 1
            arb_record = {
                "key": arb_key,
                "comparison": (
                    f"multi-agent-design-v{champ} vs "
                    f"multi-agent-design-v{args.current_version}"
                ),
                "kind": "champion_arbitration",
                "version_a": champ,
                "version_b": args.current_version,
                "repo_a": str(champ_repo),
                "repo_b": str(current_repo),
                "winner": arb_verdict.winner,
                "rationale": arb_verdict.rationale,
                "judged_at_utc": _utc_now_iso(),
                "llm_backend": judge_backend_name,
                "model": judge_model,
                "base_url": judge_base_url,
                "reasoning_effort": judge_reasoning_effort,
                "temperature": judge_temperature,
                "call_metrics": arb_call.metrics,
            }
            arb_cost = arb_call.metrics.get("estimated_cost_usd")
            if isinstance(arb_cost, (int, float)):
                run_cost_estimates.append(float(arb_cost))
            _append_history_row(champion_path, arb_record)
            _write_json_once(arb_dir / "result.json", arb_record)
            champion_rows.append(arb_record)
            if arb_verdict.winner == "B":
                champ = args.current_version
        scoreboard_champion = (
            champ
            if (not args.judge_vs_baseline or champ_has_baseline_win)
            else None
        )
        if args.evaluate_only:
            result = {"version": scoreboard_champion,
                      "beats_reference": scoreboard_champion is not None,
                      "model": model, "last_evaluated_version": args.current_version}
            (out_dir / "retained_workflow.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result))
            return 0
        # Best-section renders only for a PROVEN champion: always true in
        # adjacent mode (the chain crowned it); in baseline mode it requires
        # an actual baseline win (a by-default v0 champion is suppressed —
        # no verdict supports a "previous best" claim there).
        if champ != args.current_version and (
            not args.judge_vs_baseline or champ_has_baseline_win
        ):
            best_design_version = champ
            champ_design_path = versions_dir / f"multi_agent_design_v{champ}.txt"
            if not champ_design_path.is_file():
                raise SystemExit(
                    f"Champion design file missing: {champ_design_path}"
                )
            best_design_text = champ_design_path.read_text(encoding="utf-8").strip()
            # Rationale from the verdict where the CURRENT design was directly
            # beaten: the champion-arbitration row if one exists for this pair,
            # else (adjacent mode) the adjacent row it just lost, or (baseline
            # mode) the current version's baseline row (why the baseline won).
            rationale_row: dict[str, Any] | None = None
            arb_lookup_key = f"champion_v{champ}_vs_v{args.current_version}"
            for row in reversed(champion_rows):
                if row.get("key") == arb_lookup_key:
                    rationale_row = row
                    break
            if args.judge_vs_baseline:
                best_baseline_case = (
                    "both_win_arbitration"
                    if rationale_row is not None
                    else "champion_only_win"
                )
            if rationale_row is None:
                if args.judge_vs_baseline:
                    nh_lookup_key = f"v{args.current_version}_vs_no_harness"
                    for row in reversed(history_rows):
                        if row.get("key") == nh_lookup_key:
                            rationale_row = row
                            break
                else:
                    adj_lookup_key = _history_key(
                        args.current_version - 1, args.current_version
                    )
                    for row in reversed(history_rows):
                        if row.get("key") == adj_lookup_key:
                            rationale_row = row
                            break
            if rationale_row is not None and rationale_row.get("rationale"):
                best_design_rationale = str(rationale_row["rationale"])

    history_design_diffs: dict[str, str] | None = None
    if args.history_design_diffs:
        history_design_diffs = _design_diffs_for_history(
            history_rows,
            versions_dir,
            baseline_mode=args.judge_vs_baseline,
        )

    baseline_scoreboard_text: str | None = None
    if args.judge_vs_baseline and args.current_version >= 1:
        baseline_scoreboard_text = _baseline_scoreboard(
            history_rows,
            args.current_version,
            scoreboard_champion,
            champ_has_baseline_win,
        )

    # v0 root anchor: skipped at step 0 (v0 IS the current design) and when
    # v0 is already in the prompt as the champion (dedup).
    v0_design_text: str | None = None
    if (
        args.include_v0_design
        and args.current_version >= 1
        and best_design_version != 0
    ):
        v0_root_path = versions_dir / "multi_agent_design_v0.txt"
        if not v0_root_path.is_file():
            raise SystemExit(f"v0 design file missing: {v0_root_path}")
        v0_design_text = v0_root_path.read_text(encoding="utf-8").strip()

    improve_system, improve_user = _build_single_submission_improvement_prompt(
        template=template,
        current_version=args.current_version,
        current_design=current_design,
        workspace_evidence=workspace_evidence,
        history_rows=history_rows,
        submission_label=args.submission_label,
        execution_metadata=execution_metadata,
        output_mode=args.improvement_output_mode,
        max_design_chars=args.max_design_chars,
        target_design_chars=args.target_design_chars,
        forbid_fixed_agent_limits=args.forbid_fixed_agent_limits,
        compact_era_prompt=args.compact_era_prompt,
        best_design=best_design_text,
        best_version=best_design_version,
        best_rationale=best_design_rationale,
        design_diffs=history_design_diffs,
        v0_design=v0_design_text,
        baseline_mode=args.judge_vs_baseline,
        baseline_scoreboard=baseline_scoreboard_text,
        best_baseline_case=best_baseline_case,
    )
    improve_key = f"v{args.current_version}_to_v{args.current_version + 1}"
    improve_dir = prompts_dir / improve_key
    next_version = args.current_version + 1
    next_design_path = versions_dir / f"multi_agent_design_v{next_version}.txt"
    rendered_query_path = (
        rendered_queries_dir / f"{template.query_name}-v{next_version}.txt"
    )
    improve_result_path = improve_dir / "result.json"
    if not args.dry_run:
        existing_outputs = [
            path
            for path in (next_design_path, rendered_query_path, improve_result_path)
            if path.exists()
        ]
        if existing_outputs:
            joined = "\n- ".join(str(path) for path in existing_outputs)
            raise SystemExit(
                "Refusing to overwrite an existing RHI result. Existing paths:\n"
                f"- {joined}"
            )
    _write_full_prompt_dump(
        improve_dir / "full_prompt.txt",
        system_prompt=improve_system,
        user_prompt=improve_user,
    )
    estimated_input_tokens, estimated_total_with_output = (
        _enforce_context_budget(
            stage="harness optimizer",
            system_prompt=improve_system,
            user_prompt=improve_user,
            max_output_tokens=args.max_output_tokens,
            context_window_tokens=args.context_window_tokens,
            context_safety_tokens=args.context_safety_tokens,
        )
    )

    if args.dry_run:
        dry_summary = {
            "query_name": template.query_name,
            "query_num": template.query_num,
            "dry_run": True,
            "llm_backend": args.llm_backend,
            "model": model,
            "current_version": args.current_version,
            "current_repo": str(current_repo),
            "current_run_status_file": (
                str(current_run_status_file)
                if current_run_status_file is not None
                else None
            ),
            "full_prompt_path": str(improve_dir / "full_prompt.txt"),
            "prompt_characters": len(improve_system) + len(improve_user),
            "estimated_input_tokens_chars_div_4": estimated_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "context_window_tokens": args.context_window_tokens,
            "context_safety_tokens": args.context_safety_tokens,
            "estimated_total_with_output": estimated_total_with_output,
        }
        dry_summary_path = (
            out_dir / "last_dry_run_summary.json"
            if args.current_version == 0
            else out_dir
            / f"last_dry_run_summary_v{args.current_version}_to_v{next_version}.json"
        )
        dry_summary_path.write_text(
            json.dumps(dry_summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(dry_summary, indent=2))
        return 0

    assert backend is not None
    if args.improvement_output_mode == "json":
        improve_call = backend.call_json(
            system_prompt=improve_system,
            user_prompt=improve_user,
        )
        improve_payload = improve_call.payload
        improved_design_raw = str(
            improve_payload.get("improved_multi_agent_design", "")
        ).strip()
    else:
        improve_call = backend.call_text(
            system_prompt=improve_system,
            user_prompt=improve_user,
        )
        improved_design_raw = improve_call.raw_text.strip()
        improve_payload = {
            "improved_multi_agent_design": improved_design_raw,
            "changes_from_previous": [],
            "why_it_should_improve": [],
            "evidence_used": [],
            "expected_impact": [],
            "verification_checks": [],
            "metadata_note": (
                "design-text fallback: explanatory arrays were intentionally omitted "
                "to avoid JSON-escaping/truncation failures."
            ),
        }
    run_api_calls_total += 1
    estimated_cost = improve_call.metrics.get("estimated_cost_usd")
    if isinstance(estimated_cost, (int, float)):
        run_cost_estimates.append(float(estimated_cost))
    improved_design = _normalize_generated_design_block(
        improved_design_raw,
        start_marker=args.start_marker,
        end_marker=args.end_marker,
        after_design=template.after_design,
    )
    fixed_limit_matches = (
        _fixed_agent_limit_matches(improved_design)
        if args.forbid_fixed_agent_limits
        else []
    )
    if fixed_limit_matches:
        rejected_result = {
            "rejected_at_utc": _utc_now_iso(),
            "reason": "Generated harness violates the no-fixed-agent-limits policy.",
            "matches": fixed_limit_matches,
            "llm_backend": args.llm_backend,
            "model": model,
            "seed": args.seed,
            "call_metrics": improve_call.metrics,
            "response_json": improve_payload,
        }
        rejected_result_path = (
            improve_dir / f"rejected_result_seed_{args.seed or 'none'}.json"
        )
        _write_json_once(rejected_result_path, rejected_result)
        raise SystemExit(
            "Rejected generated harness because it introduced fixed agent limits: "
            + ", ".join(fixed_limit_matches)
        )
    if args.max_design_chars is not None and len(improved_design) > args.max_design_chars:
        rejected_result = {
            "rejected_at_utc": _utc_now_iso(),
            "reason": "Generated harness exceeds the configured artifact-size limit.",
            "design_characters": len(improved_design),
            "max_design_characters": args.max_design_chars,
            "llm_backend": args.llm_backend,
            "model": model,
            "seed": args.seed,
            "call_metrics": improve_call.metrics,
            "response_json": improve_payload,
        }
        rejected_result_path = (
            improve_dir / f"rejected_result_seed_{args.seed or 'none'}.json"
        )
        _write_json_once(rejected_result_path, rejected_result)
        raise SystemExit(
            "Rejected generated harness because it is too large: "
            f"{len(improved_design)} characters > {args.max_design_chars}"
        )

    next_design_path.write_text(improved_design + "\n", encoding="utf-8")

    rendered_query = _render_query_with_design(template, improved_design)
    rendered_query_path.write_text(rendered_query, encoding="utf-8")

    improve_result = {
        "improvement": f"multi-agent-design-v{args.current_version} -> multi-agent-design-v{next_version}",
        "current_repo": str(current_repo),
        "query_file_source": str(query_file),
        "rendered_query_file": str(rendered_query_path),
        "design_file": str(next_design_path),
        "created_at_utc": _utc_now_iso(),
        "llm_backend": args.llm_backend,
        "model": model,
        "base_url": (
            args.base_url if args.llm_backend == BACKEND_LOCAL_VLLM else None
        ),
        "reasoning_effort": args.reasoning_effort,
        "temperature": temperature_kw,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_output_tokens": args.max_output_tokens,
        "local_thinking": (
            args.local_thinking
            if args.llm_backend == BACKEND_LOCAL_VLLM
            else None
        ),
        "seed": args.seed,
        "submission_label": args.submission_label,
        "improvement_output_mode": args.improvement_output_mode,
        "current_run_status_file": (
            str(current_run_status_file)
            if current_run_status_file is not None
            else None
        ),
        "estimated_input_tokens_chars_div_4": estimated_input_tokens,
        "call_metrics": improve_call.metrics,
        "response_json": improve_payload,
    }
    improve_result_path.write_text(
        json.dumps(improve_result, indent=2), encoding="utf-8"
    )

    run_summary = {
        "query_name": template.query_name,
        "query_num": template.query_num,
        "runs_root": str(runs_root) if runs_root is not None else None,
        "prev_runs_root": str(prev_runs_root) if prev_runs_root is not None else None,
        "current_version": args.current_version,
        "next_version": next_version,
        "current_repo": str(current_repo),
        "history_path": str(history_path),
        "next_design_path": str(next_design_path),
        "rendered_query_path": str(rendered_query_path),
        "out_dir": str(out_dir),
        "llm_backend": args.llm_backend,
        "model": model,
        "base_url": (
            args.base_url if args.llm_backend == BACKEND_LOCAL_VLLM else None
        ),
        "estimated_cost_usd_this_run": (round(sum(run_cost_estimates), 8) if run_cost_estimates else None),
        "api_calls_count_this_run": run_api_calls_total,
        "latest_improvement_call": improve_call.metrics,
    }
    run_summary_path = (
        out_dir / "last_run_summary.json"
        if args.current_version == 0
        else out_dir
        / f"last_run_summary_v{args.current_version}_to_v{next_version}.json"
    )
    run_summary_path.write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(run_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
