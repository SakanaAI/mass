from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore[assignment]


SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
    "venv",
}

# Large or binary artifacts we only mention in the tree, not inline.
SKIP_SUFFIXES = (
    ".parquet",
    ".pq",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".onnx",
    ".bin",
    ".h5",
    ".hdf5",
    ".feather",
    ".arrow",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
)


@dataclass
class FileRead:
    """One file excerpt for the judge bundle."""

    path: str  # relative to memory root
    truncated: bool
    content: str


@dataclass
class DeliverableScan:
    """Heuristic checklist derived from the task text + filesystem."""

    png_mentions_in_query: list[str] = field(default_factory=list)
    png_found: list[str] = field(default_factory=list)
    png_missing: list[str] = field(default_factory=list)
    core_paths: dict[str, bool] = field(default_factory=dict)


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _token_counter_for_model(model_name: str | None) -> tuple[Callable[[str], int], str]:
    """
    Return a token counting function and a short mode label.

    - For OpenAI GPT models, use ``tiktoken`` (exact tokenizer for this use-case).
    - Fallback to a conservative chars->tokens estimate when tokenizer is unavailable.
    """
    model = (model_name or "").strip().lower()
    if model.startswith("gpt-") and tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("o200k_base")
            return (lambda s: len(enc.encode(s))), "tiktoken:o200k_base"
        except Exception:
            pass
    # Conservative fallback for non-OpenAI models or missing tokenizer.
    return (lambda s: max(1, math.ceil(len(s) / 4))), "estimate:chars_div_4"


def _truncate_text_to_token_budget(
    text: str,
    *,
    max_tokens: int,
    token_count: Callable[[str], int],
) -> tuple[str, bool]:
    if max_tokens <= 0:
        return text, False
    if token_count(text) <= max_tokens:
        return text, False

    marker = "\n\n[... middle omitted (token-capped) ...]\n\n"
    marker_tokens = token_count(marker)
    if marker_tokens >= max_tokens:
        return marker, True

    budget = max_tokens - marker_tokens
    head_chars = max(1, len(text) // 4)
    tail_chars = max(1, len(text) // 4)
    # Shrink head/tail windows until the excerpt fits token budget.
    for _ in range(24):
        candidate = text[:head_chars] + marker + text[-tail_chars:]
        if token_count(candidate) <= max_tokens:
            return candidate, True
        head_chars = max(1, int(head_chars * 0.7))
        tail_chars = max(1, int(tail_chars * 0.7))

    # Last-resort tiny excerpt.
    tiny = text[:256] + marker + text[-256:]
    return tiny, True


def extract_png_paths_from_query(query_text: str) -> list[str]:
    """Pull likely plot paths from the Deliverables section (best-effort)."""
    # deliverables/plots/foo.png or results/... not png usually; focus on .png
    paths = set(
        m.group(1)
        for m in re.finditer(
            r"([\w./\\-]+deliverables/plots/[\w.-]+\.png)", query_text, re.IGNORECASE
        )
    )
    # Also bare filenames under plots/
    for m in re.finditer(
        r"(?:deliverables/plots/|\b)([a-z0-9_-]+\.png)\b", query_text, re.IGNORECASE
    ):
        name = m.group(1)
        paths.add(f"deliverables/plots/{name}")
    return sorted(paths)


def scan_deliverables(memory_root: Path, query_text: str) -> DeliverableScan:
    mentions = extract_png_paths_from_query(query_text)
    found: list[str] = []
    missing: list[str] = []
    for rel in mentions:
        p = memory_root / rel
        if p.is_file():
            found.append(rel)
        else:
            missing.append(rel)
    core = {
        "research_report.md": (memory_root / "research_report.md").is_file(),
        "deliverables/index.json": (memory_root / "deliverables" / "index.json").is_file(),
        "results/metrics.json": (memory_root / "results" / "metrics.json").is_file(),
        "requirements.txt": (memory_root / "requirements.txt").is_file(),
    }
    return DeliverableScan(
        png_mentions_in_query=mentions,
        png_found=found,
        png_missing=missing,
        core_paths=core,
    )


def build_file_tree(
    root: Path,
    *,
    max_files: int = 400,
    max_depth: int = 6,
    max_children_per_dir: int = 100,
) -> str:
    """Return a bounded, representative workspace tree.

    ``max_files`` is retained as the public argument name for compatibility,
    but it now limits all displayed filesystem entries, including
    directories.  Previously, directory-only dataset trees could grow
    without bound because only files consumed the budget.

    Directories with more than ``max_children_per_dir`` immediate children
    are represented by one explicit omission marker instead of enumerating
    raw dataset/cache contents.  The directory itself remains visible as
    provenance evidence.
    """
    lines: list[str] = []
    remaining_entries = max(0, max_files)
    tree_truncated = False

    def walk(cur: Path, depth: int) -> None:
        nonlocal remaining_entries, tree_truncated
        if remaining_entries <= 0:
            tree_truncated = True
            return
        if depth > max_depth:
            return
        if not cur.is_dir():
            return
        name = cur.name
        if name in SKIP_DIR_NAMES:
            return
        try:
            subs = sorted(
                (
                    path
                    for path in cur.iterdir()
                    if not (path.is_dir() and path.name in SKIP_DIR_NAMES)
                ),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError:
            return
        if max_children_per_dir > 0 and len(subs) > max_children_per_dir:
            lines.append(
                f"{'  ' * depth}... [contents omitted: {len(subs)} entries "
                f"exceed per-directory limit {max_children_per_dir}]"
            )
            return
        for p in subs:
            if remaining_entries <= 0:
                tree_truncated = True
                return
            if p.is_dir():
                lines.append(f"{'  ' * depth}{p.name}/")
                remaining_entries -= 1
                walk(p, depth + 1)
            else:
                suf = p.suffix.lower()
                if suf in SKIP_SUFFIXES:
                    lines.append(f"{'  ' * depth}{p.name}  [{suf} binary/large]")
                else:
                    lines.append(f"{'  ' * depth}{p.name}")
                remaining_entries -= 1

    lines.append(f"{root.name}/")
    walk(root, 1)
    if tree_truncated:
        lines.append("... (tree truncated)")
    return "\n".join(lines)


def read_text_file(
    memory_root: Path,
    rel: str,
    *,
    max_chars: int | None,
    max_tokens: int | None = None,
    token_count: Callable[[str], int] | None = None,
) -> FileRead | None:
    path = memory_root / rel
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    truncated = False
    if max_tokens is not None:
        token_fn = token_count if token_count is not None else (lambda s: max(1, math.ceil(len(s) / 4)))
        text, tok_truncated = _truncate_text_to_token_budget(
            text,
            max_tokens=max_tokens,
            token_count=token_fn,
        )
        truncated = truncated or tok_truncated
    truncated_by_chars = max_chars is not None and len(text) > max_chars
    truncated = truncated or truncated_by_chars
    if truncated:
        if max_chars is not None and len(text) > max_chars:
            half = max_chars // 2
            text = (
                text[:half]
                + "\n\n[... middle omitted ...]\n\n"
                + text[-half:]
            )
    return FileRead(path=rel.replace("\\", "/"), truncated=truncated, content=text)


def read_log_excerpt(
    log_path: Path,
    *,
    head_chars: int,
    tail_chars: int,
    full: bool = False,
) -> str | None:
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if full:
        return text
    if len(text) <= head_chars + tail_chars + 200:
        return text
    return (
        text[:head_chars]
        + "\n\n[... log middle omitted ...]\n\n"
        + text[-tail_chars:]
    )


def discover_entry_scripts(memory_root: Path) -> list[str]:
    """Likely orchestration scripts at repo root of the memory folder."""
    names: list[str] = []
    for pattern in ("run_*.py", "main.py", "pipeline.py"):
        for p in memory_root.glob(pattern):
            if p.is_file():
                names.append(_rel(p, memory_root))
    return sorted(set(names))


def bundle_supporting_reads(
    memory_root: Path,
    *,
    report_max_chars: int | None = 28_000,
    json_max_chars: int | None = 24_000,
    code_max_chars: int | None = 12_000,
    report_max_tokens: int | None = None,
    json_max_tokens: int | None = None,
    code_max_tokens: int | None = None,
    token_count_model: str | None = None,
    max_code_files: int = 4,
) -> list[FileRead]:
    """Load high-signal text artifacts (size-capped)."""
    out: list[FileRead] = []
    token_count, _token_mode = _token_counter_for_model(token_count_model)
    priority_rel = [
        "research_report.md",
        "deliverables/index.json",
        "results/metrics.json",
        "results/model_comparison.json",
        "requirements.txt",
    ]
    for rel in priority_rel:
        is_json = rel.endswith(".json")
        fr = read_text_file(
            memory_root,
            rel,
            max_chars=json_max_chars if is_json else report_max_chars,
            max_tokens=json_max_tokens if is_json else report_max_tokens,
            token_count=token_count,
        )
        if fr:
            out.append(fr)

    scripts = discover_entry_scripts(memory_root)
    for rel in scripts[:max_code_files]:
        fr = read_text_file(
            memory_root,
            rel,
            max_chars=code_max_chars,
            max_tokens=code_max_tokens,
            token_count=token_count,
        )
        if fr:
            out.append(fr)

    # One pytest / test file if present
    tests_dir = memory_root / "tests"
    if tests_dir.is_dir():
        for p in sorted(tests_dir.glob("test_*.py")):
            test_cap = None if code_max_chars is None else min(8000, code_max_chars)
            test_token_cap = None if code_max_tokens is None else min(8000, code_max_tokens)
            fr = read_text_file(
                memory_root,
                _rel(p, memory_root),
                max_chars=test_cap,
                max_tokens=test_token_cap,
                token_count=token_count,
            )
            if fr:
                out.append(fr)
            break

    return out


def format_scan_for_prompt(scan: DeliverableScan) -> str:
    lines = [
        "## Programmatic preflight (heuristic evidence only)",
        "Derived from the **task text** (expected paths) vs this workspace; not a substitute for reading the Deliverables block.",
        "",
        "### Core files",
    ]
    for k, ok in sorted(scan.core_paths.items()):
        lines.append(f"- `{k}`: {'FOUND' if ok else 'MISSING'}")
    lines.append("")
    lines.append("### Plot paths extracted from task text (substring match)")
    if not scan.png_mentions_in_query:
        lines.append("- (none detected — task may use different wording)")
    else:
        lines.append(f"- mentioned: {len(scan.png_mentions_in_query)}")
        for p in scan.png_mentions_in_query[:40]:
            status = "ok" if p in scan.png_found else "missing"
            lines.append(f"  - [{status}] `{p}`")
        if len(scan.png_mentions_in_query) > 40:
            lines.append("  - ...")
    return "\n".join(lines)
