# Experiment support

This page describes which parts of the paper can be run with the repository
and which require additional experiment or analysis code. Availability of a
runner does not establish reproduction of the reported results. See the
[implementation notes](reproduction.md) for protocol differences and validation
limits.

## Available components

| Component | Included functionality | Notes |
|---|---|---|
| Synthetic tasks | All 12 prompts and initial workflows, with the nine/three split | Task 221 is included but excluded from SFT in the supplied configurations. |
| Recursive pipeline | Configurations and stage commands for both MASS cycles | The current fixed-reference RHI rule differs from Algorithm 1. |
| Post-training | Conversation rendering, assistant loss masks, LoRA, validation-loss checkpoint selection, merge, and FP8 export | Full training and exact second-cycle data/budget provenance remain unverified. |
| Synthetic evaluation | Bare-task rollouts and six external judgments per supplied workspace pair | Evaluation-pool selection, saved manifests, and cumulative summaries require care; see the implementation notes. Reference eligibility, complete failure/retry handling, and per-task/per-split reporting are not automated. |
| ScienceAgentBench | Task construction, execution, official scoring, and trial mean/sample SD | The historical instance ledger and aggregation are unavailable; fresh environment execution is unverified. |
| MLR-Bench | The 12-task subset, execution, two-reviewer scoring, and trial mean/sample SD | Uses zero for missing/short papers. Additional delivery, behavior, and statistical analyses are not included. |

## Additional paper experiments

The following experiment drivers and analyses are not included:

| Experiment or analysis | Additional code required |
|---|---|
| Workflow-optimization performance | Cumulative unanimous 6/6 coverage and per-iteration win-rate aggregation/plots |
| Evaluator agreement across generations | Matched self/reference judgment sets and agreement aggregation |
| Fixed-executor workflow transfer | Execution across 12 tasks and 11 iterations with the base executor, followed by the four reported comparisons |
| Task information and component redundancy | Component extraction, both embedding encoders, PCA, Gaussian information estimates, task centering, permutation adjustment, and figures |
| Teacher-composition controls S, M, X, S+, S++ | Episode filters/manifests, two-window sampler configurations, external checkpoint selection, evaluation, and token-efficiency plots |
| Teacher-output quality and trace length | Eligible pools, length-stratified pairs, four-judgment reporting, and matched-length/same-workflow comparisons |
| Optimizer/evaluator/history ablations | Six experiment presets, the 20-iteration sweep, and coverage/win-rate aggregation; some low-level RHI flags are available |
| Workflow edit size and execution variability | Token-alignment edit measures, Spearman/flip aggregation, and unchanged-workflow comparisons |
| Behavioral internalization and SFT exposure | Delegation, handoff, revision, main-edit, and action-to-SFT-window/sampled-exposure analyses |
| Generic-directive control | The separate design-free control, matching, and reporting; the teacher directive itself is included |
| Additional MLR-Bench statistics | Delivered-paper counts, conditional scores, delegation statistics, and Welch/Fisher/paired sign tests |
| Terminal-Bench 2.0 and DSBench | Execution/scoring adapters |

For implementation locations and model notation, see the
[paper-to-code mapping](paper_to_code.md). For commands, start with the
[run guide](running.md) or [public benchmarks](../benchmarks/README.md).
