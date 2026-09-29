# Paper-to-code mapping

| Paper concept | Implementation |
|---|---|
| Current model $\mathcal L^{(k)}$ | `model_id` in the cycle config; shared by executor, evaluator, and optimizer |
| Task $x_t$ | `tasks/bare/query<ID>.txt` |
| Initial workflow | `tasks/initial/query<ID>_ourTeam.txt` |
| RHI / workflow search | `python -m mass search`; `harness_improvement/iterate_multi_agent_prompt.py` |
| Reference execution | Bare-task workspace under `runs/Lk/reference/` |
| Retained workflow $w_t^*$ | `search/query<ID>_ourTeam/retained_workflow.json` and its versioned prompt |
| Teacher collection | `python -m mass collect`; 18 seeds per eligible task |
| Self-evaluation and ranking | `python -m mass rank`; within-task comparisons and Bradley–Terry fitting |
| Training and validation split | `python -m mass select`; ranks 1–15 / rank 16 |
| Workflow removal from the training prompt | `training/prepare_sft_data.py`; orchestrator only |
| Post-training | `training/train_sft_lora.py`; fresh LoRA on the BF16 parent |
| Checkpoint selection | `mass.pipeline.best_checkpoint`; minimum assistant-token validation loss |
| Next inference model $\mathcal L^{(k+1)}$ | Merge the selected adapter, then block-FP8 conversion |
| External reporting | `python -m mass report`; six judgments per workspace pair |

The nine designated training tasks are **51, 53, 56, 202, 205, 221, 302, 307, 324**.
All nine prompts are included. Task 221 was excluded from teacher collection
in the reported experiment, leaving **51, 53, 56, 202, 205, 302, 307, 324**
as the eight SFT training tasks. The three test tasks are **60, 207, 305**.
See the [task index](../tasks/README.md) for the full inventory.
Workflow search may evaluate all 12 tasks; test tasks never
enter SFT, its validation set, or checkpoint selection. The configured split
is held fixed in the second cycle.

The renderer keeps assistant reasoning, text, and tool calls as targets.
System/user messages, tool outputs, and overlapping context have label `-100`.
Only the orchestrator's initial task–workflow–directive prompt is replaced by
the bare task; worker instructions remain. A strict rendering error stops the
pipeline rather than silently dropping selected examples.

The paper configuration uses 49,152-token windows, up to 8,192 masked overlap
tokens, task-uniform sampling, and orchestrator probability 1/3. The remaining
2/3 covers worker and auxiliary conversations. Sampling is with replacement.

LoRA uses rank 64, alpha 128, and dropout 0.05. AdamW uses betas (0.9, 0.95),
peak learning rate 3e-5, five warmup steps, cosine decay to 10%, and 1,624 steps.
Three replicas each draw one window per step. Each replica places the model on
two GPUs. Checkpoints are saved every 50 steps; validation runs initially,
every 150 steps, and at the final step.

The supplied RHI implementation executes each proposed workflow. Its champion
record retains a workflow that beats the reference; it does not make the
executed sequence monotonic. The final `--evaluate-only` addition records the
last comparison and champion without generating an unused next proposal.
This is the fixed-reference branch with champion arbitration; it is not
equivalent to Algorithm 1's direct comparison with the retained best output.
See [experiment support](experiment_coverage.md) and the
[implementation notes](reproduction.md#workflow-search).
