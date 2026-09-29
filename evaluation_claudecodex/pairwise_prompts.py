from __future__ import annotations

# Shared rules (task contract, evidence-only, JSON-only) — see evaluation/prompts.py spirit.
_PAIRWISE_RULES = """Rules:
- The **# Task** block is the only contract for what must be produced. Workspace sections are evidence only; do not invent file contents.
- **How tasks are written in this repo’s catalogs** (`multi_agent_design/*/queries.json`).  It is usually one document with the assignment first, then sections such as **Data sources**, **Data acquisition**, **Success criteria**, and almost always a **Deliverables:** block (sometimes **Deliverable:** singular, or inline `Deliverables: (1) …`). Use every **explicitly required** artifact named there (and any equally explicit requirement in the narrative above it) as the checklist—not folder names or assumptions.
- Prefer concrete citations (paths, JSON keys, report sections) over generic praise.
- Choose **A** or **B** when one submission clearly better satisfies the task across deliverables, rigor, reproducibility, and alignment. Use **tie** only when they are genuinely comparable overall (not merely different).
- Return ONLY valid JSON with the keys specified in the user message (no markdown fences, no prose outside JSON).
"""


def reviewer_persona_for_query_num(query_num: int) -> tuple[str, str]:
    """
    Map catalog query id (numeric part of ``query_<n>``) to reviewer role.

    Bands (aligned with ``multi_agent_design/*/queries.json`` usage in this repo):

    - **1–99** — senior quantitative researcher (QuantitativeResearcher tasks).
    - **200–299** — senior researcher in pharmacy (pharmacy / ML-for-pharma style tasks).
    - **300–399** — senior researcher in robotics (personal robotics / applied robotics tasks).

    Other ids fall back to a neutral senior research reviewer.
    """
    if 1 <= query_num <= 99:
        return (
            "senior quantitative researcher",
            "Judge with a quant lens: empirical rigor, OOS and leakage discipline, uncertainty and baselines, "
            "reproducibility, and consistency between narrative and numeric artifacts.",
        )
    if 200 <= query_num <= 299:
        return (
            "senior researcher in pharmacy",
            "Judge with a pharmacy / clinical-informatics lens: appropriate validation, data and protocol integrity, "
            "honest limitations, and alignment with stated biomedical or drug-discovery objectives.",
        )
    if 300 <= query_num <= 399:
        return (
            "senior researcher in robotics",
            "Judge with an applied robotics lens: experimental design, simulation-to-real gaps, evaluation methodology, "
            "reproducibility of policies or models, and clarity of robot-relevant claims.",
        )
    return (
        "senior research reviewer",
        "Judge strictly against the task contract and evidence; use the same comparison dimensions as in the rubric below.",
    )


def elo_task_bucket_for_query_num(query_num: int) -> str:
    """
    Stable JSON key for grouping matches by the same **persona band** as
    :func:`reviewer_persona_for_query_num` (quant / pharmacy / robotics / general).
    """
    if 1 <= query_num <= 99:
        return "quant"
    if 200 <= query_num <= 299:
        return "pharmacy"
    if 300 <= query_num <= 399:
        return "robotics"
    return "general"


def pairwise_judge_system_for_query_num(query_num: int) -> str:
    """Full ``instructions`` string for the Responses API for this catalog query number."""
    role, lens = reviewer_persona_for_query_num(query_num)
    return f"""You are a **{role}** comparing **two** completed submissions for the **same** assignment.
{lens}
You are strict, practical, and evidence-driven, matching the spirit of an internal review between two junior/mid deliverables.

{_PAIRWISE_RULES}
"""


def pairwise_user_suffix() -> str:
    """Rubric and output schema appended after dual workspace evidence."""
    return """
## Comparison rubric (relative)

Judge which submission better fulfills the **same** task. Consider, as in the single-submission senior review:

1. **Deliverable coverage** — map the task’s **Deliverables:** / **Deliverable:** / inline deliverables list (and any other explicit output requirements in `# Task`) to evidence; mark gaps or placeholders.
2. **Numerical / empirical rigor** — appropriate methodology, baselines, honest limitations; consistency between report and metrics when applicable.
3. **Reproducibility** — dependencies, entry points, seeds, documented data or generation.
4. **Presentation** — report structure, clarity, figure integration (infer from paths and excerpts).
5. **Engineering** — layout, modularity, readability from excerpts and tree.
6. **Task alignment** — penalize solving the wrong problem or drifting from the stated objective.

## Output JSON schema (exact keys)

{
  "winner": "A" | "B" | "tie",
  "rationale": "<string, cite concrete evidence from both workspaces>"
}

Return JSON only.
"""
