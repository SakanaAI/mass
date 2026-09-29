from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.artifacts import (
    DeliverableScan,
    build_file_tree,
    bundle_supporting_reads,
    format_scan_for_prompt,
    read_log_excerpt,
    scan_deliverables,
)


@dataclass
class JudgeBundle:
    """All text context passed to the judge model."""

    task_query_id: str
    task_text: str
    memory_root: str
    log_path: str | None
    programmatic_preflight: str
    file_tree: str
    file_excerpts: list[tuple[str, str]]  # (header, body)
    log_excerpt: str | None


def build_judge_bundle(
    *,
    memory_root: Path,
    task_row: dict[str, Any],
    log_path: Path | None,
    scan: DeliverableScan | None = None,
    tree_max_files: int = 400,
    tree_max_depth: int = 6,
    tree_max_children_per_dir: int = 100,
    report_max_chars: int | None = 28_000,
    json_max_chars: int | None = 24_000,
    code_max_chars: int | None = 12_000,
    report_max_tokens: int | None = None,
    json_max_tokens: int | None = None,
    code_max_tokens: int | None = None,
    token_count_model: str | None = None,
    max_code_files: int = 4,
    log_head: int = 12_000,
    log_tail: int = 16_000,
    log_full: bool = False,
) -> JudgeBundle:
    qid = str(task_row.get("query_id", ""))
    task_text = str(task_row.get("query", ""))

    if scan is None:
        scan = scan_deliverables(memory_root, task_text)

    tree = build_file_tree(
        memory_root,
        max_files=tree_max_files,
        max_depth=tree_max_depth,
        max_children_per_dir=tree_max_children_per_dir,
    )
    reads = bundle_supporting_reads(
        memory_root,
        report_max_chars=report_max_chars,
        json_max_chars=json_max_chars,
        code_max_chars=code_max_chars,
        report_max_tokens=report_max_tokens,
        json_max_tokens=json_max_tokens,
        code_max_tokens=code_max_tokens,
        token_count_model=token_count_model,
        max_code_files=max_code_files,
    )
    excerpts: list[tuple[str, str]] = []
    for fr in reads:
        flag = " [truncated]" if fr.truncated else ""
        header = f"### File: `{fr.path}`{flag}"
        excerpts.append((header, fr.content))

    log_ex = (
        read_log_excerpt(
            log_path,
            head_chars=log_head,
            tail_chars=log_tail,
            full=log_full,
        )
        if log_path
        else None
    )

    return JudgeBundle(
        task_query_id=qid,
        task_text=task_text,
        memory_root=str(memory_root.resolve()),
        log_path=str(log_path.resolve()) if log_path else None,
        programmatic_preflight=format_scan_for_prompt(scan),
        file_tree=tree,
        file_excerpts=excerpts,
        log_excerpt=log_ex,
    )


def bundle_workspace_evidence(
    b: JudgeBundle,
    *,
    workspace_section_title: str = "# Submission workspace (evidence only; path may be multi-agent memory or any other run root)",
) -> str:
    """Tree, preflight, optional log, and excerpts — same layout as :func:`bundle_to_user_message` without the task block."""
    parts: list[str] = [
        workspace_section_title,
        f"Path: `{b.memory_root}`",
        "",
        "## Directory tree (representative)",
        "```",
        b.file_tree,
        "```",
        "",
        b.programmatic_preflight,
        "",
    ]
    if b.log_path:
        parts.append("# Workflow log")
        parts.append(f"Path: `{b.log_path}`")
        if b.log_excerpt:
            parts.append("```")
            parts.append(b.log_excerpt)
            parts.append("```")
        parts.append("")
    parts.append("# File excerpts (size-capped; binaries omitted)")
    parts.append("")
    for header, body in b.file_excerpts:
        parts.append(header)
        parts.append("```")
        parts.append(body)
        parts.append("```")
        parts.append("")
    return "\n".join(parts).strip()


def bundle_to_user_message(b: JudgeBundle) -> str:
    return "\n".join(
        [
            "# Task",
            "",
            b.task_text,
            "",
            bundle_workspace_evidence(b),
        ]
    ).strip()
